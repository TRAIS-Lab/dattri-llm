"""On-disk storage and retrieval for GradientRecord objects."""

from __future__ import annotations

import json
import operator
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from dattri_llm.utils.distributed import dist_rank
from dattri_llm.utils.hashing import hash_sample

if TYPE_CHECKING:
    from dattri_llm.gradient.gradient import Gradient, GradientRecord


def _merge_index(dst: dict[str, list[dict]], src: dict[str, list[dict]]) -> None:
    for h, entries in src.items():
        existing = dst.setdefault(h, [])
        for e in entries:
            if e not in existing:
                existing.append(e)


class GradientFileManager:
    """Manages on-disk storage and retrieval of :class:`GradientRecord` objects.

    Responsible for file naming, maintaining ``index.json`` hash-to-location
    mappings, and loading records back.  It knows nothing about *when* to
    save -- that is :class:`OffloadCallback`'s job.

    Under ``DistributedDataParallel`` (or any context where
    ``torch.distributed`` is initialised), each rank writes to its own
    subdirectory so that file names and ``index.json`` never collide:

    Under single-GPU / non-distributed::

        save_dir/
            index.json
            batch_000000.pt
            batch_000001.pt
            ...

    Under DDP with 4 ranks::

        save_dir/
            rank_0/
                index.json
                batch_000000.pt
                ...
            rank_1/
                index.json
                batch_000000.pt
                ...
            rank_2/ ...
            rank_3/ ...

    The three bugs that this layout prevents:

    * **File-name collision** -- each rank's ``_next_batch_id`` counter is
      local to its own subdirectory, so ``batch_000000.pt`` on rank 0 and
      ``batch_000000.pt`` on rank 1 live in different directories.
    * **``index.json`` race** -- every rank writes only to its own
      ``rank_N/index.json``; there are no concurrent writes to a shared file.
    * **Silent data loss** -- in DDP each rank processes a different micro-batch,
      so all ranks must save; restricting saves to rank 0 would discard 3/4 of
      the gradient data.

    Index format (inside each ``index.json``)::

        {
            "<sha256_hex>": [
                {"file": "rank_0/batch_000000.pt", "idx": 2, "step": 0, "sample_idx":
                5},
                {"file": "rank_0/batch_000002.pt", "idx": 0, "step": 4, "sample_idx": 1}
            ]
        }

    The mapping is ``input_hash -> (step, sample_idx) -> file``: the content hash is
    sample-position independent (a sample hashes the same wherever shuffling
    put it), and each entry records **where** that sample landed -- the training
    ``step`` and its ``sample_idx`` position within the stored record's batch -- so
    a per-sample gradient is retrieved by direct slicing, never by scanning a
    batch record.  :meth:`lookup` returns a hash's ``(step, sample_idx)`` pairs and
    :meth:`load_sample` retrieves one pair's gradient.

    Paths in ``"file"`` are always relative to *save_dir* (the root), so
    load methods work regardless of which rank originally wrote the record.

    The public load API is rank-transparent: a fresh :class:`GradientFileManager`
    opened on the same *save_dir* after training automatically merges all
    per-rank indexes and loads from whichever rank's file contains the
    requested sample.

    Args:
        save_dir: Root directory.  Under DDP, subdirectories ``rank_N/`` are
            created automatically; the caller does not need to add a rank
            suffix.  Created on first save if it does not exist.
    """

    _INDEX_FILE = "index.json"

    def __init__(self, save_dir: str) -> None:
        self._root_dir = Path(save_dir)

        rank = dist_rank()
        if rank is not None:
            self._save_dir = self._root_dir / f"rank_{rank}"
            self._local_prefix = f"rank_{rank}/"
        else:
            self._save_dir = self._root_dir
            self._local_prefix = ""

        # Merged view across all ranks (used for all load operations).
        self._index: dict[str, list[dict]] = self._read_all_indexes()
        # Per-rank batch counter, scoped to _save_dir so there is no collision.
        self._next_batch_id: int = self._compute_next_batch_id()

    # ---------------------------------------------------------------------- #
    # Saving                                                                   #
    # ---------------------------------------------------------------------- #

    def save(self, record: GradientRecord) -> Path:
        """Persist one :class:`GradientRecord` to its own file.

        Args:
            record: The record to persist.

        Returns:
            Path: The written file.

        Raises:
            TypeError: If the record carries a per-batch hash list (use
                :meth:`save_bulk` for those).
        """
        if isinstance(record.input_hash, list):
            raise TypeError(
                "save() takes a single-hash record; this record carries a "
                "per-batch hash list -- use save_bulk([record]) instead.",
            )
        self._save_dir.mkdir(parents=True, exist_ok=True)
        h = record.input_hash
        s = record.step
        path = self._save_dir / f"step_{s:06d}_{h}.pt"
        torch.save(record, path)
        rel = self._local_prefix + path.name
        self._index_entry(record, filename=rel, idx=0)
        self._write_index()
        return path

    def save_bulk(self, records: list[GradientRecord]) -> Path:
        """Pack multiple :class:`GradientRecord` objects into a single file.

        "Bulk" is file-level packing (amortising file I/O -- e.g.
        ``OffloadCallback`` flushes its accumulated records through here every
        ``offload_interval`` steps); it says nothing about the records' shape.
        Each record may be per-sample (single ``input_hash``) or per-batch
        (``input_hash`` list), and the two may be mixed in one call.  Every
        sample hash of every record is indexed with its ``(step, sample_idx)``
        coordinates, so per-sample lookup works without a directory scan.

        Args:
            records: The records to persist together.  Each record's hash-list
                length must equal its gradient's batch size (checked).

        Returns:
            Path: The written batch file.
        """
        self._save_dir.mkdir(parents=True, exist_ok=True)
        batch_id = self._next_batch_id
        self._next_batch_id += 1
        path = self._save_dir / f"batch_{batch_id:06d}.pt"
        torch.save(records, path)
        rel = self._local_prefix + path.name
        for idx, record in enumerate(records):
            self._index_entry(record, filename=rel, idx=idx)
        self._write_index()
        return path

    def _index_entry(self, record: GradientRecord, filename: str, idx: int) -> None:
        hashes = (
            record.input_hash
            if isinstance(record.input_hash, list)
            else [record.input_hash]
        )
        if isinstance(record.input_hash, list):
            batch = record.gradient.batch_size
            if len(hashes) != batch:
                raise ValueError(
                    f"Record at step {record.step} carries {len(hashes)} sample "
                    f"hashes but its gradient batch size is {batch}; the indexed "
                    "sample_idx positions would not match the gradient rows.",
                )
        for pos, h in enumerate(hashes):
            # "sample_idx" is the sample's position within the record's batch, so
            # a per-sample gradient is retrieved by direct slicing (see load_sample).
            entry = {
                "file": filename,
                "idx": idx,
                "step": record.step,
                "sample_idx": pos,
            }
            entries = self._index.setdefault(h, [])
            if entry not in entries:
                entries.append(entry)

    # ---------------------------------------------------------------------- #
    # Loading                                                                  #
    # ---------------------------------------------------------------------- #

    # Every load-family method comes as a pair differing only by how the
    # sample is identified: ``F(inputs, ...)`` takes ONE sample's model-input
    # dict and hashes it; ``F_by_hash(input_hash, ...)`` takes the precomputed
    # content hash.  Under the ``input_hash -> (step, sample_idx) -> file`` index a
    # step alone no longer identifies a gradient -- only a ``(step, sample_idx)``
    # pair does -- so there is no step-only loader.

    def lookup(
        self,
        inputs: dict[str, torch.Tensor],
    ) -> list[tuple[int, int]]:
        """Every ``(step, sample_idx)`` occurrence of a sample, sorted by step.

        Args:
            inputs: **One sample's** model-input dict (e.g. ``dataset[i]``).

        Returns:
            Sorted list of ``(step, sample_idx)`` pairs -- see :meth:`lookup_by_hash`.
        """
        return self.lookup_by_hash(hash_sample(inputs))

    def lookup_by_hash(self, input_hash: str) -> list[tuple[int, int]]:
        """Every ``(step, sample_idx)`` occurrence of a sample, sorted by step.

        The content hash identifies *what* the sample is (independent of
        shuffling); the returned pairs say *where* it was recorded -- the
        training step, and its position within that step's stored batch.
        Feed a pair to :meth:`load_sample_by_hash` to retrieve the gradient.

        Args:
            input_hash: Full 64-character SHA-256 hash (see
                :func:`~dattri_llm.utils.hashing.hash_sample`).

        Returns:
            Sorted list of ``(step, sample_idx)`` pairs.

        Raises:
            KeyError: If the hash is not found in the index.
        """
        if input_hash not in self._index:
            raise KeyError(
                f"Hash {input_hash[:16]}... not in index. "
                "Has the collect() context closed, and is save_dir correct?",
            )
        return sorted((e["step"], e["sample_idx"]) for e in self._index[input_hash])

    def load_sample(
        self,
        inputs: dict[str, torch.Tensor],
        step: int,
        sample_idx: int,
    ) -> Gradient:
        """Load one sample's gradient at one ``(step, sample_idx)`` occurrence.

        Args:
            inputs: **One sample's** model-input dict (e.g. ``dataset[i]``).
            step: Training step, as returned by :meth:`lookup`.
            sample_idx: The sample's position within the stored batch, as
                returned by :meth:`lookup`.

        Returns:
            Gradient: The sample's gradient -- see :meth:`load_sample_by_hash`.
        """
        return self.load_sample_by_hash(hash_sample(inputs), step, sample_idx)

    def load_sample_by_hash(
        self,
        input_hash: str,
        step: int,
        sample_idx: int,
    ) -> Gradient:
        """Load one sample's gradient at one training step, by direct slicing.

        Uses the indexed ``sample`` position to slice the stored record's batch
        gradient -- an O(1) retrieval, with no scan over the batch.

        Args:
            input_hash: Full 64-character SHA-256 hash of the sample.
            step: Training step, as returned by :meth:`lookup_by_hash`.
            sample_idx: The sample's position within the stored batch, as
                returned by :meth:`lookup_by_hash`.

        Returns:
            Gradient: The sample's gradient (batch dimension 1).

        Raises:
            KeyError: If no record matches ``(input_hash, step, sample_idx)``.
        """
        for e in self._index.get(input_hash, []):
            if e["step"] == step and e["sample_idx"] == sample_idx:
                record = self._load_entry(e)
                return record.gradient.slice(dim="batch", index=sample_idx)
        pairs = self.lookup_by_hash(input_hash) if input_hash in self._index else []
        raise KeyError(
            f"No record for hash {input_hash[:16]}... at (step={step}, "
            f"sample_idx={sample_idx}). Available (step, sample_idx) pairs: {pairs}.",
        )

    def load_all(
        self,
        inputs: dict[str, torch.Tensor],
    ) -> list[GradientRecord]:
        """Every saved :class:`GradientRecord` containing one sample.

        Args:
            inputs: **One sample's** model-input dict (e.g. ``dataset[i]``).

        Returns:
            List of :class:`GradientRecord` sorted by step -- see
            :meth:`load_all_by_hash`.
        """
        return self.load_all_by_hash(hash_sample(inputs))

    def load_all_by_hash(self, input_hash: str) -> list[GradientRecord]:
        """Load every saved :class:`GradientRecord` for a given sample hash.

        Returns whole records (a per-batch record includes the sample's whole
        batch); use :meth:`load_sample_by_hash` for the sample's own gradient.

        Args:
            input_hash: Full 64-character SHA-256 hash.

        Returns:
            List of :class:`GradientRecord` sorted by step.

        Raises:
            KeyError: If the hash is not found in the index.
        """
        if input_hash not in self._index:
            raise KeyError(
                f"Hash {input_hash[:16]}... not in index. "
                "Has the collect() context closed, and is save_dir correct?",
            )
        entries = sorted(self._index[input_hash], key=operator.itemgetter("step"))
        return [self._load_entry(e) for e in entries]

    def available_steps(self) -> list[int]:
        """Return every step present in the (merged) index, ascending.

        Returns:
            Sorted list of distinct step indices across all saved records.
        """
        steps = {e["step"] for entries in self._index.values() for e in entries}
        return sorted(steps)

    def iter_step(self, step: int) -> list[tuple[str, list[int]]]:
        """Enumerate the on-disk record slots for one step, in disk order.

        Walks the merged index and groups, for the requested ``step``, the
        record positions by file.  This is the ordering used to assign
        attribution-matrix rows/columns: it depends only on what is on disk,
        not on any reconstructed dataset or DataLoader.

        Args:
            step: The step to enumerate.

        Returns:
            List of ``(file_relpath, sorted_record_idxs)`` tuples, ordered by
            file name.  ``file_relpath`` is relative to the root *save_dir* and
            can be passed to :meth:`load_records`.
        """
        by_file: dict[str, set[int]] = {}
        for entries in self._index.values():
            for e in entries:
                if e["step"] == step:
                    by_file.setdefault(e["file"], set()).add(e["idx"])
        return [(f, sorted(by_file[f])) for f in sorted(by_file)]

    def iter_steps(self, steps: list[int]) -> list[tuple[str, dict[int, list[int]]]]:
        """Enumerate record slots for multiple steps, grouped by file.

        This is the multi-step companion to :meth:`iter_step`.  A file that
        contains records for several requested steps appears once, with its
        matching record indices grouped by step, so callers can load the file a
        single time and consume every relevant record from it.

        Args:
            steps: Step IDs to enumerate.

        Returns:
            List of ``(file_relpath, {step: sorted_record_idxs})`` tuples,
            ordered by file name.
        """
        wanted = set(steps)
        by_file: dict[str, dict[int, set[int]]] = {}
        for entries in self._index.values():
            for e in entries:
                step = e["step"]
                if step in wanted:
                    by_file.setdefault(e["file"], {}).setdefault(
                        step,
                        set(),
                    ).add(e["idx"])
        out: list[tuple[str, dict[int, list[int]]]] = []
        for file_rel in sorted(by_file):
            by_step = {
                step: sorted(idxs) for step, idxs in sorted(by_file[file_rel].items())
            }
            out.append((file_rel, by_step))
        return out

    def load_records(self, file_relpath: str) -> list[GradientRecord]:
        """Load every :class:`GradientRecord` stored in one file.

        Args:
            file_relpath: Path relative to the root *save_dir*, as returned by
                :meth:`iter_step`.

        Returns:
            The file's records as a list (a single-record file is wrapped in a
            one-element list).
        """
        obj = torch.load(self._root_dir / file_relpath, weights_only=False)
        return obj if isinstance(obj, list) else [obj]

    def _load_entry(self, entry: dict) -> GradientRecord:
        # "file" is always relative to _root_dir, not _save_dir.
        path = self._root_dir / entry["file"]
        obj = torch.load(path, weights_only=False)
        if isinstance(obj, list):
            return obj[entry["idx"]]
        return obj

    # ---------------------------------------------------------------------- #
    # Index                                                                    #
    # ---------------------------------------------------------------------- #

    @property
    def index(self) -> dict[str, list[dict]]:
        """Merged mapping from ``input_hash`` to ``{file, idx, step}`` entries.

        Spans all ranks.  Updated after every :meth:`save` /
        :meth:`save_bulk` call on this instance.
        """
        return self._index

    def _read_all_indexes(self) -> dict[str, list[dict]]:
        """Merge index.json from the root dir and every rank_N/ subdirectory."""
        merged: dict[str, list[dict]] = {}
        # Root-level index: non-distributed saves or old-format saves.
        root_idx = self._root_dir / self._INDEX_FILE
        if root_idx.exists():
            with Path(root_idx).open(encoding="utf-8") as f:
                _merge_index(merged, json.load(f))
        # Per-rank indexes written by this class under DDP.
        for rank_dir in sorted(self._root_dir.glob("rank_*")):
            if not rank_dir.is_dir():
                continue
            rank_idx = rank_dir / self._INDEX_FILE
            if rank_idx.exists():
                with Path(rank_idx).open(encoding="utf-8") as f:
                    _merge_index(merged, json.load(f))
        return merged

    def _write_index(self) -> None:
        """Write only this rank's entries to _save_dir/index.json."""
        self._save_dir.mkdir(parents=True, exist_ok=True)
        # Filter to entries whose file path belongs to this rank's save_dir.
        # For non-distributed _local_prefix is "", so all entries match.
        local_index: dict[str, list[dict]] = {}
        for h, entries in self._index.items():
            local = [e for e in entries if e["file"].startswith(self._local_prefix)]
            if local:
                local_index[h] = local
        with Path(self._save_dir / self._INDEX_FILE).open("w", encoding="utf-8") as f:
            json.dump(local_index, f)

    def _compute_next_batch_id(self) -> int:
        if not self._save_dir.exists():
            return 0
        ids = [
            int(p.stem.split("_")[1])
            for p in self._save_dir.glob("batch_*.pt")
            if p.stem.split("_")[1].isdigit()
        ]
        return max(ids) + 1 if ids else 0

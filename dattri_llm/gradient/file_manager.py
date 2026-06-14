"""On-disk storage and retrieval for GradientRecord objects."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from dattri_llm.gradient.gradient import GradientRecord
from dattri_llm.gradient.utils import hash_sample


def _dist_rank() -> int | None:
    """Return the current distributed rank, or None if not in a distributed context."""
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return None


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
    save — that is :class:`OffloadCallback`'s job.

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

    * **File-name collision** — each rank's ``_next_batch_id`` counter is
      local to its own subdirectory, so ``batch_000000.pt`` on rank 0 and
      ``batch_000000.pt`` on rank 1 live in different directories.
    * **``index.json`` race** — every rank writes only to its own
      ``rank_N/index.json``; there are no concurrent writes to a shared file.
    * **Silent data loss** — in DDP each rank processes a different micro-batch,
      so all ranks must save; restricting saves to rank 0 would discard ¾ of
      the gradient data.

    Index format (inside each ``index.json``)::

        {
            "<sha256_hex>": [
                {"file": "rank_0/batch_000000.pt", "idx": 2, "step": 0},
                {"file": "rank_0/batch_000002.pt", "idx": 0, "step": 4}
            ]
        }

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

        rank = _dist_rank()
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
            The :class:`~pathlib.Path` of the written file.
        """
        self._save_dir.mkdir(parents=True, exist_ok=True)
        h = record.input_hash
        s = record.step
        path = self._save_dir / f"step_{s:06d}_{h}.pt"
        torch.save(record, path)
        rel = self._local_prefix + path.name
        self._index_entry(record, filename=rel, idx=0)
        self._write_index()
        return path

    def save_batch(self, records: list[GradientRecord]) -> Path:
        """Persist multiple :class:`GradientRecord` objects to a single file.

        All sample hashes from each record (including per-batch
        ``input_hash`` lists) are indexed so that per-sample lookup works
        without a directory scan.

        Args:
            records: The records to persist together.

        Returns:
            The :class:`~pathlib.Path` of the written batch file.
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
        entry = {"file": filename, "idx": idx, "step": record.step}
        hashes = record.input_hash if isinstance(record.input_hash, list) else [record.input_hash]
        for h in hashes:
            entries = self._index.setdefault(h, [])
            if entry not in entries:
                entries.append(entry)

    # ---------------------------------------------------------------------- #
    # Loading                                                                  #
    # ---------------------------------------------------------------------- #

    def load(
        self,
        step: int,
        inputs: dict[str, torch.Tensor],
        sample_idx: int = 0,
    ) -> GradientRecord:
        """Load a :class:`GradientRecord` for a sample given its inputs and step.

        Args:
            step: The batch-step index assigned by the collector.
            inputs: The model input dict for the batch.
            sample_idx: Index of the target sample within the batch dimension.

        Returns:
            The :class:`GradientRecord` for that sample at that step.
        """
        return self.load_by_hash(step, hash_sample(inputs, sample_idx))

    def load_by_hash(self, step: int, input_hash: str) -> GradientRecord:
        """Load a :class:`GradientRecord` by step and sample hash.

        Args:
            step: The batch-step index assigned by the collector.
            input_hash: Full 64-character SHA-256 hash of the sample.

        Returns:
            The :class:`GradientRecord` for that sample at that step.

        Raises:
            KeyError: If the hash or step is not in the index.
        """
        entries = self._index.get(input_hash)
        if not entries:
            raise KeyError(f"Hash {input_hash[:16]}… not in index.")
        for entry in entries:
            if entry["step"] == step:
                return self._load_entry(entry)
        known = sorted({e["step"] for e in entries})
        raise KeyError(
            f"step {step} not found for hash {input_hash[:16]}…  known steps: {known}"
        )

    def load_all(
        self,
        inputs: dict[str, torch.Tensor],
    ) -> list[GradientRecord]:
        """Load every saved :class:`GradientRecord` that contains any sample in *inputs*.

        Computes the SHA-256 hash for every sample in the batch, unions their
        index entries, deduplicates (a per-batch file appears once even if
        multiple samples in the query batch hit it), and returns all unique
        records sorted by step.

        Args:
            inputs: The model input dict for the batch — the same dict passed
                to the model during collection.

        Returns:
            List of unique :class:`GradientRecord` sorted by step.
        """
        batch_size = next(
            (v.shape[0] for v in inputs.values()
             if isinstance(v, torch.Tensor) and v.ndim > 0),
            1,
        )
        seen: set[tuple] = set()
        records: list[GradientRecord] = []
        for i in range(batch_size):
            h = hash_sample(inputs, i)
            for entry in self._index.get(h, []):
                key = (entry["file"], entry["idx"])
                if key not in seen:
                    seen.add(key)
                    records.append(self._load_entry(entry))
        records.sort(key=lambda r: r.step)
        return records

    def load_all_by_hash(self, input_hash: str) -> list[GradientRecord]:
        """Load every saved :class:`GradientRecord` for a given sample hash.

        Args:
            input_hash: Full 64-character SHA-256 hash.

        Returns:
            List of :class:`GradientRecord` sorted by step.

        Raises:
            KeyError: If the hash is not found in the index.
        """
        if input_hash not in self._index:
            raise KeyError(
                f"Hash {input_hash[:16]}… not in index. "
                "Has the collect() context closed, and is save_dir correct?"
            )
        entries = sorted(self._index[input_hash], key=lambda e: e["step"])
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
        :meth:`save_batch` call on this instance.
        """
        return self._index

    def _read_all_indexes(self) -> dict[str, list[dict]]:
        """Merge index.json from the root dir and every rank_N/ subdirectory."""
        merged: dict[str, list[dict]] = {}
        # Root-level index: non-distributed saves or old-format saves.
        root_idx = self._root_dir / self._INDEX_FILE
        if root_idx.exists():
            with open(root_idx) as f:
                _merge_index(merged, json.load(f))
        # Per-rank indexes written by this class under DDP.
        for rank_dir in sorted(self._root_dir.glob("rank_*")):
            if not rank_dir.is_dir():
                continue
            rank_idx = rank_dir / self._INDEX_FILE
            if rank_idx.exists():
                with open(rank_idx) as f:
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
        with open(self._save_dir / self._INDEX_FILE, "w") as f:
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

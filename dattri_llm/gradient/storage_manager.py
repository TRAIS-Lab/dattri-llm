"""Residency-managed storage and retrieval for GradientRecord objects.

A store's *residency* (chosen once, at construction) decides where its records
physically live -- transparent to every reader above the
:class:`GradientStorageManager`:

* ``"disk"`` (default) -- serialize each group to a file (``torch.save`` /
  ``torch.load``), the classic store-then-attribute layout.  Crash-safe and
  re-openable across processes / ranks.
* ``"memory"`` -- hold the (CPU) records in RAM, never serialized.  Zero I/O;
  a re-iterable source reads them back instantly.  Ephemeral (not crash-safe,
  single-process) -- for replay caches, not a system of record.
* ``"tiered"`` -- hold records in RAM up to a byte budget, then spill the
  oldest groups to the disk backend.  The budget defaults to ~half of
  available RAM when not given.

The residency is invisible to :class:`DiskGradientSource`, the attributors,
and the index: records are addressed by an opaque *location* handle (a
filename for disk, a synthetic key for memory) plus the hash index, so the
whole read path is location-agnostic.
"""

from __future__ import annotations

import contextlib
import json
import operator
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from dattri_llm.gradient.gradient import Factorized, Gradient, GradientRecord
from dattri_llm.utils.distributed import dist_rank
from dattri_llm.utils.hashing import hash_sample

if TYPE_CHECKING:
    from collections.abc import Iterable

    from typing_extensions import Self

RESIDENCIES = ("disk", "memory", "tiered")

# On-disk serialization for the ``disk`` backend.  ``"pickle"`` (default) writes
# one ``torch.save`` file per group.  ``"memmap"`` writes materialized groups as
# a flat ``.mmap.bin`` (concatenated raw tensor bytes, 8-byte aligned) beside a
# small ``.mmap.meta`` (the record/gradient metadata + per-layer byte offsets),
# so reads are copy-on-write ``np.memmap`` views with no pickle deserialization
# (~4x faster warm reads for compact materialized blocks).  Groups that are not
# purely materialized -- any factorized (a, g) layer -- fall back to ``.pt``
# transparently, so a memmap store may hold both handle kinds.
DISK_FORMATS = ("pickle", "memmap")

# numpy has no bfloat16, so a group holding bf16 tensors cannot be memmapped and
# falls back to pickle.
_MEMMAP_UNSUPPORTED_DTYPES = (torch.bfloat16,)


def _group_memmappable(records: list[GradientRecord]) -> bool:
    """Whether every record's every layer is a plain, numpy-representable tensor.

    Memmap stores concatenated dense bytes, so a factorized ``(a, g)`` layer (or
    a bf16 tensor numpy cannot view) disqualifies the group -> pickle fallback.
    """
    for record in records:
        for value in record.gradient.data.values():
            if not isinstance(value, torch.Tensor):
                return False
            if value.dtype in _MEMMAP_UNSUPPORTED_DTYPES:
                return False
    return True


def _gradient_nbytes(gradient: Gradient) -> int:
    """Total bytes of a gradient block's tensor payloads (factors + dense)."""
    total = 0
    for value in gradient.data.values():
        tensors = (
            (value.activation, value.pre_activation_grad)
            if isinstance(value, Factorized)
            else (value,)
        )
        for t in tensors:
            total += t.numel() * t.element_size()
    return total


def _records_nbytes(records: list[GradientRecord]) -> int:
    return sum(_gradient_nbytes(r.gradient) for r in records)


def _auto_budget_bytes() -> int:
    """Half of currently-available RAM (from /proc/meminfo), or 8 GiB fallback."""
    try:
        with Path("/proc/meminfo").open(encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    kb = int(line.split()[1])
                    return int(kb * 1024 * 0.5)
    except (OSError, ValueError, IndexError):
        pass
    return 8 * 2**30


def _to_cpu_record(record: GradientRecord) -> GradientRecord:
    """Return *record* with its gradient payloads on CPU.

    Captures may be device-resident (``HookManager(offload_to_cpu=False)``,
    the default); saving them as-is would serialise CUDA tensors, which reload
    pinned to the original ``cuda:*`` device and break loading on other
    machines.  This is the single, batched device transfer of the disk
    workflow -- a no-op (same tensor objects) when the payloads are already
    on CPU.
    """
    if record.gradient.device.type == "cpu":
        return record
    return GradientRecord(
        step=record.step,
        input_hash=record.input_hash,
        gradient=record.gradient.to("cpu"),
        sample_id_key=getattr(record, "sample_id_key", None),
    )


def _merge_index(dst: dict[str, list[dict]], src: dict[str, list[dict]]) -> None:
    for h, entries in src.items():
        existing = dst.setdefault(h, [])
        for e in entries:
            if e not in existing:
                existing.append(e)


class GradientStorageManager:  # noqa: PLR0904 - load-family pairs + residency API
    """Residency-managed storage and retrieval of :class:`GradientRecord` objects.

    Responsible for record naming, maintaining the hash-to-location index, and
    loading records back.  It knows nothing about *when* to save -- that is
    :class:`OffloadCallback`'s (or an attributor's ``cache``) job.

    **Residency** (``residency=``, chosen at construction) decides where records
    physically live; it is invisible to every reader above this class:

    * ``"disk"`` (default) -- the durable store-then-attribute layout below:
      ``torch.save`` per group, a persisted ``index.json``, crash-safe, and
      re-openable across processes / ranks.  ``close()`` is a no-op.
    * ``"memory"`` -- records held in RAM, never serialized (zero I/O).
      **Ephemeral**: nothing is written to ``save_dir`` (which is ignored,
      including any pre-existing index), and the records are gone once the
      manager is dropped -- a fresh manager cannot recover them.  For replay
      caches, not a system of record.
    * ``"tiered"`` -- RAM up to ``budget_bytes`` (default ~half of available
      RAM), then the oldest groups spill to a **temporary** subdir of
      ``save_dir``.  Also ephemeral: no ``index.json`` is written, and
      :meth:`close` (called on ``with``-exit and best-effort at GC) removes the
      spill dir, so a tiered store never leaves orphaned files behind.  Use it
      as a context manager::

          with GradientStorageManager(d, residency="tiered") as store:
              ...  # spill files auto-removed on exit

    The residency is transparent to :class:`DiskGradientSource`, the
    attributors, and the index: records are addressed by an opaque *location*
    handle (a relative file path for disk / spilled tiered groups, a synthetic
    ``mem_*`` key for in-RAM groups) plus the hash index.

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
            "sample_id_key": null,
            "index": {
                "<identifier>": [
                    {"file": "rank_0/batch_000000.pt", "idx": 2, "step": 0,
                     "sample_idx": 5},
                    {"file": "rank_0/batch_000002.pt", "idx": 0, "step": 4,
                     "sample_idx": 1}
                ]
            }
        }

    The mapping is ``identifier -> (step, sample_idx) -> file``.  The
    identifier is whatever the capturing
    :class:`~dattri_llm.gradient.hooks.HookManager` assigned: the SHA-256
    content hash by default (sample-position independent -- a sample hashes
    the same wherever shuffling put it), or, under ``sample_id_key``, the
    stringified values of that input field.  ``sample_id_key`` records which
    scheme the store was collected under (``null`` = content hashing); it is
    adopted from the first saved record and every later record must agree.
    Each entry records **where** the sample landed -- the training ``step``
    and its ``sample_idx`` position within the stored record's batch -- so a
    per-sample gradient is retrieved by direct slicing, never by scanning a
    batch record.  :meth:`lookup` returns an identifier's
    ``(step, sample_idx)`` pairs and :meth:`load_sample` retrieves one pair's
    gradient.

    Paths in ``"file"`` are always relative to *save_dir* (the root), so
    load methods work regardless of which rank originally wrote the record.

    The public load API is rank-transparent: a fresh :class:`GradientStorageManager`
    opened on the same *save_dir* after training automatically merges all
    per-rank indexes and loads from whichever rank's file contains the
    requested sample.

    .. warning::
        Records are deserialised with ``torch.load(..., weights_only=False)``
        (they contain :class:`GradientRecord` objects, not bare tensors),
        which executes arbitrary pickled code.  Only open gradient
        directories you produced yourself or otherwise trust.

    Args:
        save_dir: Root directory.  Under DDP, subdirectories ``rank_N/`` are
            created automatically; the caller does not need to add a rank
            suffix.  Created on first save if it does not exist.  The rank
            routing is resolved at the *first save*, not at construction, so
            the manager may safely be built before ``init_process_group`` has
            run (e.g. before the HF Trainer, which initializes distributed
            internally).
    """

    _INDEX_FILE = "index.json"

    def __init__(
        self,
        save_dir: str,
        *,
        residency: str = "disk",
        budget_bytes: int | None = None,
        disk_format: str = "pickle",
    ) -> None:
        if residency not in RESIDENCIES:
            raise ValueError(
                f"residency must be one of {list(RESIDENCIES)}, got {residency!r}.",
            )
        if disk_format not in DISK_FORMATS:
            raise ValueError(
                f"disk_format must be one of {list(DISK_FORMATS)}, got "
                f"{disk_format!r}.",
            )
        self._root_dir = Path(save_dir)
        self._residency = residency
        self._disk_format = disk_format
        # Memory / tiered state: location handle -> in-RAM record group, plus a
        # running byte count and (tiered only) the spill budget.  A group is
        # spilled to the disk backend when the count exceeds the budget.
        self._mem_groups: dict[str, list[GradientRecord]] = {}
        self._mem_bytes = 0
        self._group_seq = 0  # monotonic id for synthetic memory location handles
        self._budget_bytes = (
            budget_bytes
            if budget_bytes is not None
            else (_auto_budget_bytes() if residency == "tiered" else None)
        )
        # Reverse map location -> its index entries, so a spill can rewrite the
        # entries' ``file`` handle in place (they are shared dicts, so updating
        # here updates the main index too).
        self._group_entries: dict[str, list[dict]] = {}
        # Tiered residency spills to a dedicated temp subdir (created lazily on
        # the first spill) that ``close()`` removes -- so a tiered store never
        # litters ``save_dir`` with orphaned spill files.
        self._spill_dir: Path | None = None
        self._closed: bool = False

        # Where this process writes (root, or its rank_N/ subdirectory) is
        # resolved lazily at the first save, NOT here: managers are commonly
        # constructed before ``init_process_group`` has run (e.g. before the
        # HF Trainer, which initializes distributed internally), and a
        # construction-time rank probe would see no process group and route
        # every rank to the root directory -- exactly the file-name and index
        # collisions the rank layout exists to prevent.  By the first save,
        # training is running and the true rank is known.
        self._save_dir: Path | None = None
        self._local_prefix: str = ""
        self._next_batch_id: int = 0

        # Identifier scheme the store was collected under (see the class
        # docstring): None = content hashing, else the sample_id_key input
        # field.  Adopted from existing indexes / the first saved record;
        # ``_id_key_known`` distinguishes "content hashing" from "not yet
        # determined" so a mismatch can be rejected loudly.
        self._sample_id_key: str | int | None = None
        self._id_key_known: bool = False
        # Gradient-accumulation convention the store was collected under:
        # each stored step covers this many training micro-batches (1 = every
        # backward is its own step).  Declared by the writer (see
        # OffloadCallback) or adopted from existing indexes; a mismatch
        # raises, so one store cannot mix step conventions.
        self._accumulation_steps: int | None = None
        # Merged view across all ranks (used for all load operations).  Only a
        # ``disk`` store adopts an existing on-disk index; ``memory``/``tiered``
        # are ephemeral caches that always start empty and ignore whatever
        # ``save_dir`` happens to contain (they never persist an index).
        self._index: dict[str, list[dict]] = (
            self._read_all_indexes() if residency == "disk" else {}
        )

    def _ensure_save_dir(self) -> Path:
        """Resolve this process's write directory on first use (idempotent).

        Decides between the root directory (non-distributed) and this rank's
        ``rank_N/`` subdirectory, then scopes the batch counter to it.  The
        decision is frozen after the first call so a mid-run
        ``destroy_process_group`` cannot switch directories.

        Returns:
            The directory this process saves into.
        """
        if self._save_dir is None:
            rank = dist_rank()
            if rank is not None:
                self._save_dir = self._root_dir / f"rank_{rank}"
                self._local_prefix = f"rank_{rank}/"
            else:
                self._save_dir = self._root_dir
            # Per-rank batch counter, scoped to _save_dir -- no collision.
            self._next_batch_id = self._compute_next_batch_id()
        return self._save_dir

    @property
    def sample_id_key(self) -> str | int | None:
        """The identifier scheme of this store: the input field the capturing
        manager read sample ids from, or ``None`` for content hashing.
        """
        return self._sample_id_key

    def _adopt_sample_id_key(self, key: str | int | None) -> None:
        """Adopt the store's identifier scheme, rejecting a mixed store."""
        if self._id_key_known:
            if key != self._sample_id_key:
                raise ValueError(
                    f"Record was captured with sample_id_key={key!r} but this "
                    f"store uses sample_id_key={self._sample_id_key!r}; one "
                    "store cannot mix identifier schemes.",
                )
            return
        self._sample_id_key = key
        self._id_key_known = True

    @property
    def gradient_accumulation_steps(self) -> int:
        """Micro-batches per stored step for this store (1 when undeclared).

        A writer that merges accumulation windows (see
        ``OffloadCallback(gradient_accumulation_steps=...)``) declares its
        window size here, making the store's step convention -- optimizer
        steps vs. raw micro-batch backwards -- auditable at read time.
        """
        return self._accumulation_steps if self._accumulation_steps is not None else 1

    def declare_gradient_accumulation_steps(self, n: int) -> None:
        """Declare the store's micro-batches-per-step convention.

        Idempotent for a matching value; raises if the store (on disk or via
        an earlier declaration) already uses a different one.

        Args:
            n: Micro-batches merged into each stored step (>= 1).
        """
        if n < 1:
            raise ValueError(f"gradient_accumulation_steps must be >= 1, got {n}.")
        if self._accumulation_steps is not None and self._accumulation_steps != n:
            raise ValueError(
                f"This store was collected with gradient_accumulation_steps="
                f"{self._accumulation_steps}; cannot mix with {n} -- one store "
                "cannot hold two step conventions.",
            )
        self._accumulation_steps = n

    # ---------------------------------------------------------------------- #
    # Saving                                                                   #
    # ---------------------------------------------------------------------- #

    def save(self, record: GradientRecord) -> str:
        """Persist one :class:`GradientRecord` through the residency backend.

        Args:
            record: The record to persist.

        Returns:
            The group's location handle (a root-relative file path for ``disk``
            residency, a synthetic ``mem_*`` key for ``memory``).

        Raises:
            TypeError: If the record carries a per-batch hash list (use
                :meth:`save_bulk` for those).
        """
        if isinstance(record.input_hash, list):
            raise TypeError(
                "save() takes a single-hash record; this record carries a "
                "per-batch hash list -- use save_bulk([record]) instead.",
            )
        record = _to_cpu_record(record)
        base_name = f"step_{record.step:06d}_{record.input_hash}"
        location = self._write_group([record], base_name)
        self._index_entry(record, filename=location, idx=0)
        self._persist_index()
        if self._residency == "tiered":
            self._maybe_spill()
        return location

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
            The location handle of the written group (a relative file path for
            ``disk`` residency, a synthetic ``mem_*`` key for ``memory``).
        """
        records = [_to_cpu_record(r) for r in records]
        location = self._write_group(records, None)
        for idx, record in enumerate(records):
            self._index_entry(record, filename=location, idx=idx)
        self._persist_index()
        if self._residency == "tiered":
            self._maybe_spill()
        return location

    # ---------------------------------------------------------------------- #
    # Residency backend (disk / memory / tiered)                              #
    # ---------------------------------------------------------------------- #

    @property
    def residency(self) -> str:
        """This store's residency policy (``"disk"``/``"memory"``/``"tiered"``)."""
        return self._residency

    def _write_group(
        self,
        records: list[GradientRecord],
        base_name: str | None,
    ) -> str:
        """Persist one record group through the active residency backend.

        *base_name* is the extension-less file name to use for the ``disk``
        backend (the format decides the extension), or ``None`` to auto-name it
        ``batch_<counter>``.  Returns the group's location handle: a relative
        file path (``disk``, or a spilled ``tiered`` group) or a synthetic
        ``mem_*`` key (``memory``/unspilled ``tiered``).
        """
        if self._residency in ("memory", "tiered"):
            loc = f"mem_{self._group_seq:08d}"
            self._group_seq += 1
            self._mem_groups[loc] = records
            self._mem_bytes += _records_nbytes(records)
            return loc
        return self._write_group_to_disk(records, base_name)

    def _write_group_to_disk(
        self,
        records: list[GradientRecord],
        base_name: str | None,
    ) -> str:
        """Serialize a group to a file; returns the root-relative path handle.

        The batch counter is read **after** :meth:`_ensure_save_dir` (which
        settles it from any existing files on the first call), so auto-named
        files never collide or overwrite.  Under ``disk_format="memmap"`` a
        purely materialized group is written as a ``.mmap`` handle (flat
        ``.mmap.bin`` + ``.mmap.meta``); everything else (and every group under
        ``"pickle"``) is a ``.pt`` ``torch.save`` file.  The handle's extension
        is what the read path dispatches on, so the two kinds coexist freely.
        """
        save_dir = self._ensure_save_dir()
        save_dir.mkdir(parents=True, exist_ok=True)
        if base_name is None:
            base_name = f"batch_{self._next_batch_id:06d}"
            self._next_batch_id += 1
        if self._disk_format == "memmap" and _group_memmappable(records):
            handle = f"{base_name}.mmap"
            self._write_group_memmap(records, save_dir / handle)
        else:
            handle = f"{base_name}.pt"
            torch.save(records, save_dir / handle)
        return self._local_prefix + handle

    @staticmethod
    def _write_group_memmap(records: list[GradientRecord], handle_path: Path) -> None:
        """Write a materialized group as ``<handle>.bin`` + ``<handle>.meta``.

        The ``.bin`` is the concatenation of every layer's raw tensor bytes,
        each 8-byte aligned so the reader can ``view`` any dtype at its offset.
        The ``.meta`` (a small ``torch.save``) carries the record + gradient
        metadata and each layer's ``(byte offset, shape, dtype)`` into the bin.
        """
        meta_records: list[dict] = []
        byte_off = 0
        with Path(f"{handle_path}.bin").open("wb") as binf:
            for record in records:
                layers: dict[str, dict] = {}
                for name, tensor in record.gradient.data.items():
                    arr = tensor.detach().to("cpu").contiguous().numpy()
                    raw = arr.tobytes()
                    binf.write(raw)
                    layers[name] = {
                        "byte_off": byte_off,
                        "shape": list(arr.shape),
                        "dtype": str(arr.dtype),
                    }
                    byte_off += len(raw)
                    pad = (-byte_off) % 8  # keep the next tensor 8-byte aligned
                    if pad:
                        binf.write(b"\x00" * pad)
                        byte_off += pad
                grad = record.gradient
                meta_records.append(
                    {
                        "step": record.step,
                        "input_hash": record.input_hash,
                        "sample_id_key": getattr(record, "sample_id_key", None),
                        "representation": dict(grad.representation),
                        "layer_types": dict(grad.layer_types),
                        "indexing": dict(grad.indexing),
                        "layers": layers,
                    },
                )
        torch.save(meta_records, Path(f"{handle_path}.meta"))

    @staticmethod
    def _read_group_memmap(handle_path: Path) -> list[GradientRecord]:
        """Reconstruct a group written by :meth:`_write_group_memmap`.

        The bin is mapped copy-on-write (``mode="c"``): reads are lazy, zero-copy
        page-cache hits, and the reconstructed tensors are writable (private COW
        pages) so nothing downstream is surprised by a read-only buffer.
        """
        import numpy as np

        meta_records = torch.load(Path(f"{handle_path}.meta"), weights_only=False)
        mm = np.memmap(Path(f"{handle_path}.bin"), dtype=np.uint8, mode="c")
        records: list[GradientRecord] = []
        for meta in meta_records:
            data: dict[str, torch.Tensor] = {}
            for name, spec in meta["layers"].items():
                np_dtype = np.dtype(spec["dtype"])
                shape = tuple(spec["shape"])
                count = int(np.prod(shape)) if shape else 1
                off = spec["byte_off"]
                arr = mm[off : off + count * np_dtype.itemsize].view(np_dtype)
                data[name] = torch.from_numpy(arr.reshape(shape))
            grad = Gradient(
                representation=meta["representation"],
                data=data,
                layer_types=meta["layer_types"],
                indexing=meta["indexing"],
                validate_on_init=False,
            )
            records.append(
                GradientRecord(
                    step=meta["step"],
                    input_hash=meta["input_hash"],
                    gradient=grad,
                    sample_id_key=meta["sample_id_key"],
                ),
            )
        return records

    @property
    def disk_format(self) -> str:
        """This store's on-disk serialization (``"pickle"``/``"memmap"``)."""
        return self._disk_format

    def _maybe_spill(self) -> None:
        """Spill the oldest in-RAM groups to disk until under the byte budget."""
        if self._budget_bytes is None:
            return
        # Oldest-first: mem_* keys are zero-padded and monotonically assigned.
        for loc in sorted(self._mem_groups):
            if self._mem_bytes <= self._budget_bytes:
                break
            self._spill_group(loc)

    def _ensure_spill_dir(self) -> Path:
        """Create (once) the temp subdir tiered spill files live in.

        A subdir of *save_dir* so spill I/O stays on the same filesystem and a
        single :meth:`close` ``rmtree`` cleans it up; the ``mem_*`` group key
        names each spill file, so they never collide.
        """
        if self._spill_dir is None:
            self._root_dir.mkdir(parents=True, exist_ok=True)
            self._spill_dir = Path(
                tempfile.mkdtemp(prefix="tiered_spill_", dir=self._root_dir),
            )
        return self._spill_dir

    def _spill_group(self, loc: str) -> None:
        """Move one in-RAM group to the spill dir and repoint its index entries."""
        records = self._mem_groups.pop(loc)
        self._mem_bytes -= _records_nbytes(records)
        path = self._ensure_spill_dir() / f"{loc}.pt"
        torch.save(records, path)
        disk_loc = str(path.relative_to(self._root_dir))
        for entry in self._group_entries.get(loc, []):
            entry["file"] = disk_loc  # shared dict -> also updates self._index
        if loc in self._group_entries:
            self._group_entries[disk_loc] = self._group_entries.pop(loc)

    def close(self) -> None:
        """Release an ephemeral store's resources; idempotent.

        Drops the in-RAM record groups and deletes the tiered spill directory.
        A **no-op for ``disk``** residency -- its files are the durable store.
        Called automatically on context-manager exit and (best-effort) at
        garbage collection, so a ``tiered`` store never leaves orphaned spill
        files behind.
        """
        if self._closed:
            return
        self._closed = True
        self._mem_groups.clear()
        self._mem_bytes = 0
        if self._spill_dir is not None:
            shutil.rmtree(self._spill_dir, ignore_errors=True)
            self._spill_dir = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> bool:
        self.close()
        return False

    def __del__(self) -> None:
        # Safety net if the caller neither closed nor used a ``with`` block.
        # Guarded: at interpreter shutdown modules may already be torn down.
        with contextlib.suppress(Exception):
            self.close()

    def _index_entry(self, record: GradientRecord, filename: str, idx: int) -> None:
        # ``getattr``: records pickled before sample_id_key existed read as
        # content-hashed (None), which is what they were.
        self._adopt_sample_id_key(getattr(record, "sample_id_key", None))
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
            # str(): JSON object keys are strings, so a non-str identifier
            # would silently change type across a save/load round trip.
            entries = self._index.setdefault(str(h), [])
            if entry not in entries:
                entries.append(entry)
                # Reverse map (location -> entries) so a tiered spill can
                # repoint every entry of a group to its new disk handle.
                self._group_entries.setdefault(filename, []).append(entry)

    # ---------------------------------------------------------------------- #
    # Loading                                                                  #
    # ---------------------------------------------------------------------- #

    # Every load-family method comes as a pair differing only by how the
    # sample is identified: ``F(inputs, ...)`` takes ONE sample's model-input
    # dict and derives the identifier from it (content hash, or the store's
    # ``sample_id_key`` field); ``F_by_hash(identifier, ...)`` takes the
    # precomputed identifier.  Under the ``identifier -> (step, sample_idx) ->
    # file`` index a step alone no longer identifies a gradient -- only a
    # ``(step, sample_idx)`` pair does -- so there is no step-only loader.

    def _identifier_for(self, inputs: dict[str, object]) -> str:
        """One sample's identifier under this store's scheme.

        Content hash when the store was collected without a
        ``sample_id_key``; otherwise the stringified value of that field in
        *inputs* (an ``int`` key reads ``inputs[key]`` of a sequence, or the
        ``_arg{key}`` entry of a dict -- the positional-capture convention).

        Args:
            inputs: One sample's model-input dict (or sequence, for a
                positional ``sample_id_key``).

        Returns:
            The identifier string.

        Raises:
            KeyError: If the store uses a ``sample_id_key`` and *inputs* does
                not carry that field.
        """
        key = self._sample_id_key
        if key is None:
            return hash_sample(inputs)
        if isinstance(key, int) and isinstance(inputs, (list, tuple)):
            value = inputs[key]
        else:
            field = f"_arg{key}" if isinstance(key, int) else key
            if field not in inputs:
                raise KeyError(
                    f"This store identifies samples by sample_id_key={key!r}, "
                    f"but the provided inputs carry no {field!r} field "
                    f"(keys: {sorted(inputs)}).",
                )
            value = inputs[field]
        if isinstance(value, torch.Tensor):
            value = value.item()
        return str(value)

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
        return self.lookup_by_hash(self._identifier_for(inputs))

    def lookup_by_hash(self, input_hash: str) -> list[tuple[int, int]]:
        """Every ``(step, sample_idx)`` occurrence of a sample, sorted by step.

        The identifier says *what* the sample is (independent of shuffling);
        the returned pairs say *where* it was recorded -- the training step,
        and its position within that step's stored batch.  Feed a pair to
        :meth:`load_sample_by_hash` to retrieve the gradient.

        Args:
            input_hash: The sample's identifier -- the SHA-256 content hash
                (see :func:`~dattri_llm.utils.hashing.hash_sample`), or, for a
                store collected under ``sample_id_key``, the sample-id value
                (non-string values are stringified, so ``42`` and ``"42"``
                are interchangeable).

        Returns:
            Sorted list of ``(step, sample_idx)`` pairs.

        Raises:
            KeyError: If the identifier is not found in the index.
        """
        input_hash = str(input_hash)
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
        return self.load_sample_by_hash(
            self._identifier_for(inputs),
            step,
            sample_idx,
        )

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
        input_hash = str(input_hash)
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
        return self.load_all_by_hash(self._identifier_for(inputs))

    def load_all_by_hash(self, input_hash: str) -> list[GradientRecord]:
        """Load every saved :class:`GradientRecord` for a given sample hash.

        Returns whole records (a per-batch record includes the sample's whole
        batch); use :meth:`load_sample_by_hash` for the sample's own gradient.

        Args:
            input_hash: Full 64-character SHA-256 hash.

        Returns:
            List of :class:`GradientRecord` sorted by step.

        Raises:
            KeyError: If the identifier is not found in the index.
        """
        input_hash = str(input_hash)
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

    def iter_steps(
        self,
        steps: int | Iterable[int],
    ) -> list[tuple[str, dict[int, list[int]]]]:
        """Enumerate record slots for one or more steps, grouped by file.

        Walks the merged index and groups the requested steps' record
        positions by file.  A file that contains records for several
        requested steps appears once, with its matching record indices
        grouped by step, so callers can load the file a single time and
        consume every relevant record from it.  This is the ordering used to
        assign attribution-matrix rows/columns: it depends only on what is on
        disk, not on any reconstructed dataset or DataLoader.

        Args:
            steps: Step IDs to enumerate.

        Returns:
            List of ``(file_relpath, {step: sorted_record_idxs})`` tuples,
            ordered by file name.  ``file_relpath`` is relative to the root
            *save_dir* and can be passed to :meth:`load_records`.
        """
        wanted = {steps} if isinstance(steps, int) else set(steps)
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
                :meth:`iter_steps`.

        Returns:
            The group's records as a list (a single-record file is wrapped in
            a one-element list).  Served from RAM for an in-memory group, else
            ``torch.load``-ed from disk.
        """
        if file_relpath in self._mem_groups:
            return self._mem_groups[file_relpath]
        if file_relpath.endswith(".mmap"):
            return self._read_group_memmap(self._root_dir / file_relpath)
        obj = torch.load(self._root_dir / file_relpath, weights_only=False)
        return obj if isinstance(obj, list) else [obj]

    def _load_entry(self, entry: dict) -> GradientRecord:
        # "file" is a location handle: a synthetic mem_* key (in-RAM group) or a
        # path relative to _root_dir.
        loc = entry["file"]
        if loc in self._mem_groups:
            return self._mem_groups[loc][entry["idx"]]
        if loc.endswith(".mmap"):
            return self._read_group_memmap(self._root_dir / loc)[entry["idx"]]
        obj = torch.load(self._root_dir / loc, weights_only=False)
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
        """Merge index.json from the root dir and every rank_N/ subdirectory.

        Every index file also carries the store's ``sample_id_key``; it is
        adopted from the first file read, and all files must agree (see
        :meth:`_adopt_sample_id_key`).
        """
        merged: dict[str, list[dict]] = {}

        def read_one(path: Path) -> None:
            with path.open(encoding="utf-8") as f:
                payload = json.load(f)
            key = payload["sample_id_key"]
            # JSON has no int/str key distinction problem for values, but a
            # positional (int) sample_id_key round-trips as int natively.
            self._adopt_sample_id_key(key)
            self.declare_gradient_accumulation_steps(
                payload.get("gradient_accumulation_steps", 1),
            )
            _merge_index(merged, payload["index"])

        # Root-level index: non-distributed saves.
        root_idx = self._root_dir / self._INDEX_FILE
        if root_idx.exists():
            read_one(root_idx)
        # Per-rank indexes written by this class under DDP.
        for rank_dir in sorted(self._root_dir.glob("rank_*")):
            if not rank_dir.is_dir():
                continue
            rank_idx = rank_dir / self._INDEX_FILE
            if rank_idx.exists():
                read_one(rank_idx)
        return merged

    def _persist_index(self) -> None:
        """Persist the index for a ``disk`` store; a no-op otherwise.

        ``memory``/``tiered`` stores keep the index in RAM only -- they are
        ephemeral (single-process, not re-opened), and a spilled tiered group's
        files carry no standalone index (the live index in RAM already points
        at them).
        """
        if self._residency == "disk":
            self._write_index()

    def _write_index(self) -> None:
        """Write only this rank's entries to _save_dir/index.json.

        The write is atomic (temp file + ``os.replace``): a crash mid-write
        leaves the previous index intact instead of a truncated JSON that
        would lose every prior entry for this rank.
        """
        save_dir = self._ensure_save_dir()
        save_dir.mkdir(parents=True, exist_ok=True)
        # Filter to entries whose file path belongs to this rank's save_dir.
        # For non-distributed _local_prefix is "", so all entries match.
        local_index: dict[str, list[dict]] = {}
        for h, entries in self._index.items():
            local = [e for e in entries if e["file"].startswith(self._local_prefix)]
            if local:
                local_index[h] = local
        tmp_path = save_dir / (self._INDEX_FILE + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "sample_id_key": self._sample_id_key,
                    "gradient_accumulation_steps": self.gradient_accumulation_steps,
                    "index": local_index,
                },
                f,
            )
        tmp_path.replace(save_dir / self._INDEX_FILE)

    def _compute_next_batch_id(self) -> int:
        if not self._save_dir.exists():
            return 0
        ids = [
            int(p.stem.split("_")[1])
            for p in self._save_dir.glob("batch_*.pt")
            if p.stem.split("_")[1].isdigit()
        ]
        return max(ids) + 1 if ids else 0

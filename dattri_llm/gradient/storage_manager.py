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
import warnings
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

import torch

from dattri_llm.gradient.gradient import Factorized, Gradient, GradientRecord
from dattri_llm.utils.distributed import dist_rank
from dattri_llm.utils.hashing import hash_sample

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from typing_extensions import Self

RESIDENCIES = ("disk", "memory", "tiered")

# On-disk serialization for the ``disk`` backend.  ``"pickle"`` (default) writes
# one ``torch.save`` file per group.  ``"memmap"`` writes a flat ``.mmap.bin``
# (concatenated raw tensor bytes, 8-byte aligned) beside a small ``.mmap.meta``
# (record/gradient metadata + per-layer byte offsets), read back as
# copy-on-write ``torch.from_file`` views with no pickle deserialization.  A
# materialized layer contributes one byte range, a factorized layer two (its
# ``a`` and ``g`` factors); any other payload goes to ``.pt``, so a memmap
# store may hold both handle kinds.
DISK_FORMATS = ("pickle", "memmap")

# Layout version of the .mmap pair; the reader rejects any other value.
_MEMMAP_FORMAT = 2


def _group_memmappable(records: list[GradientRecord]) -> bool:
    """Whether every layer of every record is a payload the memmap writer knows.

    That is a plain tensor (one byte range) or a :class:`Factorized` pair (two);
    anything else sends the group to pickle.
    """
    return all(
        isinstance(value, (torch.Tensor, Factorized))
        for record in records
        for value in record.gradient.data.values()
    )


def _dtype_name(dtype: torch.dtype) -> str:
    """``torch.bfloat16`` -> ``"bfloat16"``, for a JSON/pickle-safe meta field."""
    return str(dtype).removeprefix("torch.")


# Every dtype torch exposes, keyed by the name the meta stores.
_DTYPE_BY_NAME = {
    _dtype_name(value): value
    for value in vars(torch).values()
    if isinstance(value, torch.dtype)
}


def _dtype_from_name(name: str) -> torch.dtype:
    """Inverse of :func:`_dtype_name`; raises on a name torch does not define."""
    try:
        return _DTYPE_BY_NAME[name]
    except KeyError:
        raise ValueError(
            f"Unknown tensor dtype {name!r} in memmap metadata.",
        ) from None


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


def _expand_log_line(payload: dict) -> dict[str, list[dict]]:
    """Rebuild index entries from one append-only log line.

    A log line stores the group's location and, for each record in it, an
    ``idx``, a ``step`` and its hashes in batch order; a sample's
    ``sample_idx`` is its position in that hash list.

    Args:
        payload: One decoded log line.

    Returns:
        Mapping from identifier to the entries that line contributes.
    """
    out: dict[str, list[dict]] = {}
    location = payload["file"]
    for record in payload["records"]:
        idx = record["idx"]
        step = record["step"]
        for sample_idx, h in enumerate(record["hashes"]):
            out.setdefault(h, []).append(
                {
                    "file": location,
                    "idx": idx,
                    "step": step,
                    "sample_idx": sample_idx,
                },
            )
    return out


class _PhaseTimings:
    """Accumulated wall-clock time for the phases of a save.

    A save splits into five phases:

    * ``to_cpu`` -- moving the captured payloads off the GPU
      (:func:`_to_cpu_record`).  Pure bandwidth; scales with gradient size.
    * ``write_group`` -- handing the group to the residency backend:
      ``torch.save``, a memmap write, or (``memory``/unspilled ``tiered``)
      just stashing it in RAM.  Scales with gradient size, and is where the
      ``disk_format`` choice shows up.
    * ``index_update`` -- updating the in-memory hash index.  Cheap, but
      scales with the number of samples in the flush.
    * ``index_write`` -- persisting the index delta (``disk`` residency only;
      a no-op for the ephemeral residencies).
    * ``spill`` -- evicting in-RAM groups to disk once a ``tiered`` store is
      over budget (:meth:`GradientStorageManager._maybe_spill`).  ~0 for every
      other residency, but a full ``torch.save`` per evicted group when it
      does fire, which is why it is timed rather than left off the report.

    Every phase is entered once per save regardless of residency, so the call
    counts stay uniform and a phase that did no work reads as 0 seconds.

    Timing is always on; there is no flag to enable.
    """

    _PHASES = ("to_cpu", "write_group", "index_update", "index_write", "spill")

    def __init__(self) -> None:
        self._seconds: dict[str, float] = dict.fromkeys(self._PHASES, 0.0)
        self._calls: dict[str, int] = dict.fromkeys(self._PHASES, 0)

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        """Time the wrapped block into phase *name* (counted even if it raises)."""
        start = perf_counter()
        try:
            yield
        finally:
            self._seconds[name] += perf_counter() - start
            self._calls[name] += 1

    def as_dict(self) -> dict[str, dict[str, float]]:
        """Per-phase ``{"seconds", "calls"}``, in the order phases run."""
        return {
            name: {"seconds": self._seconds[name], "calls": self._calls[name]}
            for name in self._PHASES
        }

    def reset(self) -> None:
        """Zero every counter (e.g. to exclude a warm-up phase)."""
        for name in self._PHASES:
            self._seconds[name] = 0.0
            self._calls[name] = 0

    def report(self) -> str:
        """A human-readable table of the accumulated timings."""
        saves = max(self._calls.values()) if self._calls else 0
        lines = [f"GradientStorageManager save timings ({saves} saves):"]
        for name in self._PHASES:
            seconds = self._seconds[name]
            calls = self._calls[name]
            per_call = (seconds / calls * 1e3) if calls else 0.0
            lines.append(
                f"  {name:<13} {seconds:8.3f} s  ({calls} calls, "
                f"{per_call:7.2f} ms/call)",
            )
        lines.append(f"  {'total':<13} {sum(self._seconds.values()):8.3f} s")
        return "\n".join(lines)


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
      ``torch.save`` per group, a persisted index, and re-openable across
      processes / ranks.  ``close()`` is a no-op.
    * ``"memory"`` -- records held in RAM, never serialized (zero I/O).
      **Ephemeral**: nothing is written to ``save_dir`` (which is ignored,
      including any pre-existing index), and the records are gone once the
      manager is dropped -- a fresh manager cannot recover them.  For replay
      caches, not a system of record.
    * ``"tiered"`` -- RAM up to ``budget_bytes`` (default ~half of available
      RAM), then the oldest groups spill to a **temporary** subdir of
      ``save_dir``.  Also ephemeral: no index is written, and
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
    subdirectory so that file names and index files never collide:

    Under single-GPU / non-distributed::

        save_dir/
            index_meta.json
            index.jsonl
            batch_000000.pt
            batch_000001.pt
            ...

    Under DDP with 4 ranks::

        save_dir/
            rank_0/
                index_meta.json
                index.jsonl
                batch_000000.pt
                ...
            rank_1/
                index_meta.json
                index.jsonl
                batch_000000.pt
                ...
            rank_2/ ...
            rank_3/ ...

    The three bugs that this layout prevents:

    * **File-name collision** -- each rank's ``_next_batch_id`` counter is
      local to its own subdirectory, so ``batch_000000.pt`` on rank 0 and
      ``batch_000000.pt`` on rank 1 live in different directories.
    * **Index race** -- every rank appends only to its own
      ``rank_N/index.jsonl``; there are no concurrent writes to a shared file.
    * **Silent data loss** -- in DDP each rank processes a different micro-batch,
      so all ranks must save; restricting saves to rank 0 would discard 3/4 of
      the gradient data.

    The index is stored as an **append-only log**: each save appends one line
    to ``index.jsonl`` describing only what that save wrote, and the
    store-wide settings live in a small constant-size ``index_meta.json``::

        index_meta.json:
            {"format": 2, "sample_id_key": null,
             "gradient_accumulation_steps": 1}

        index.jsonl (one line per save):
            {"file": "rank_0/batch_000000.pt", "records": [
                {"idx": 0, "step": 0, "hashes": ["<h0>", "<h1>"]},
                {"idx": 1, "step": 1, "hashes": ["<h2>", "<h3>"]}]}
            {"file": "rank_0/batch_000001.pt", "records": [...]}

    Each save's cost is proportional to what that save added rather than to
    everything the store holds.  Only ``disk`` residency persists an index at
    all -- see :meth:`_persist_index`.

    Reading expands the log back into the in-memory mapping
    ``identifier -> (step, sample_idx) -> file``: the line's ``file`` plus a
    record's ``idx``/``step`` plus each hash's position in ``hashes``
    reconstructs one ``{file, idx, step, sample_idx}`` entry.  The
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

    # The append-only entry log plus its constant-size settings sidecar.
    _INDEX_LOG_FILE = "index.jsonl"
    _INDEX_META_FILE = "index_meta.json"
    _INDEX_FORMAT = 2

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
        # Per-phase save timings; see _PhaseTimings.
        self._timings = _PhaseTimings()
        # Last settings payload written to the meta sidecar, so a save only
        # rewrites it when it actually changed; see _write_index_meta.
        self._meta_written: dict | None = None

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
    def timing(self) -> dict[str, dict[str, float]]:
        """Accumulated seconds and call counts for each save phase.

        Phases are ``to_cpu`` (device-to-host transfer), ``write_group``
        (the residency backend's write), ``index_update`` (in-memory index)
        and ``index_write`` (index delta to disk) -- see
        :class:`_PhaseTimings`::

            with hookmanager.collect():
                trainer.train()
            print(storage_manager.timing_report())
        """
        return self._timings.as_dict()

    def reset_timing(self) -> None:
        """Zero the save timings, e.g. to exclude warm-up steps."""
        self._timings.reset()

    def timing_report(self) -> str:
        """A printable per-phase breakdown of time spent saving.

        Returns:
            A table of accumulated seconds, call counts and per-call
            milliseconds for each save phase.
        """
        return self._timings.report()

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
        with self._timings.phase("to_cpu"):
            record = _to_cpu_record(record)
        base_name = f"step_{record.step:06d}_{record.input_hash}"
        with self._timings.phase("write_group"):
            location = self._write_group([record], base_name)
        with self._timings.phase("index_update"):
            self._index_entry(record, filename=location, idx=0)
        with self._timings.phase("index_write"):
            self._persist_index([record], location)
        with self._timings.phase("spill"):
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
        with self._timings.phase("to_cpu"):
            records = [_to_cpu_record(r) for r in records]
        with self._timings.phase("write_group"):
            location = self._write_group(records, None)
        with self._timings.phase("index_update"):
            for idx, record in enumerate(records):
                self._index_entry(record, filename=location, idx=idx)
        with self._timings.phase("index_write"):
            self._persist_index(records, location)
        with self._timings.phase("spill"):
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
        group of tensor / :class:`Factorized` payloads is written as a
        ``.mmap`` handle (flat ``.mmap.bin`` + ``.mmap.meta``); anything the
        memmap writer does not know (and every group under ``"pickle"``) is a
        ``.pt`` ``torch.save`` file.  The handle's extension is what the read
        path dispatches on, so the two kinds coexist freely.
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
        """Write a group as ``<handle>.bin`` + ``<handle>.meta``.

        The ``.bin`` concatenates every layer's raw tensor bytes, each 8-byte
        aligned so the reader can ``view`` any dtype at its offset.  The
        ``.meta`` (a small ``torch.save``) holds the record + gradient metadata
        and, per layer, one *part* per tensor: ``data`` for a materialized
        layer, ``activation`` + ``pre_activation_grad`` for a factorized one.
        A factorized layer's non-tensor state (``module_kwargs``,
        ``batch_first``) is stored in the meta beside its parts.
        """
        meta_records: list[dict] = []
        byte_off = 0

        with Path(f"{handle_path}.bin").open("wb") as binf:

            def _write_part(tensor: torch.Tensor) -> dict:
                """Append one tensor's bytes, returning its meta spec."""
                nonlocal byte_off
                dense = tensor.detach().to("cpu").contiguous()
                raw = dense.view(torch.uint8).flatten().numpy().tobytes()
                binf.write(raw)
                spec = {
                    "byte_off": byte_off,
                    "nbytes": len(raw),
                    "shape": list(dense.shape),
                    "dtype": _dtype_name(dense.dtype),
                }
                byte_off += len(raw)
                pad = (-byte_off) % 8
                if pad:
                    binf.write(b"\x00" * pad)
                    byte_off += pad
                return spec

            for record in records:
                layers: dict[str, dict] = {}
                for name, value in record.gradient.data.items():
                    if isinstance(value, Factorized):
                        layers[name] = {
                            "kind": "factorized",
                            "parts": {
                                "activation": _write_part(value.activation),
                                "pre_activation_grad": _write_part(
                                    value.pre_activation_grad,
                                ),
                            },
                            "module_kwargs": value.module_kwargs,
                            "batch_first": value.batch_first,
                        }
                    else:
                        layers[name] = {
                            "kind": "dense",
                            "parts": {"data": _write_part(value)},
                        }
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

        torch.save(
            {"format": _MEMMAP_FORMAT, "records": meta_records},
            Path(f"{handle_path}.meta"),
        )

    @staticmethod
    def _read_group_memmap(handle_path: Path) -> list[GradientRecord]:
        """Reconstruct a group written by :meth:`_write_group_memmap`.

        The bin is mapped with ``torch.from_file(shared=False)`` -- a private
        (copy-on-write) mapping, so reads are lazy page-cache hits and the
        reconstructed tensors are writable without those writes reaching the
        file.
        """
        bin_path = Path(f"{handle_path}.bin")
        payload = torch.load(Path(f"{handle_path}.meta"), weights_only=False)
        if not isinstance(payload, dict) or payload.get("format") != _MEMMAP_FORMAT:
            found = payload.get("format") if isinstance(payload, dict) else "pre-v2"
            raise ValueError(
                f"{handle_path.name} was written in memmap format {found!r}, but "
                f"this version reads format {_MEMMAP_FORMAT}. Re-collect the "
                "store; the byte offsets are not compatible.",
            )

        total = bin_path.stat().st_size
        buffer = (
            torch.from_file(str(bin_path), shared=False, size=total, dtype=torch.uint8)
            if total
            else torch.empty(0, dtype=torch.uint8)
        )

        def _read_part(spec: dict) -> torch.Tensor:
            dtype = _dtype_from_name(spec["dtype"])
            shape = tuple(spec["shape"])
            if spec["nbytes"] == 0:  # a 0-element layer has nothing to view
                return torch.empty(shape, dtype=dtype)
            off = spec["byte_off"]
            raw = buffer[off : off + spec["nbytes"]]
            return raw.view(dtype).reshape(shape)

        records: list[GradientRecord] = []
        for meta in payload["records"]:
            data: dict[str, torch.Tensor | Factorized] = {}
            for name, layer in meta["layers"].items():
                parts = layer["parts"]
                if layer["kind"] == "factorized":
                    data[name] = Factorized(
                        activation=_read_part(parts["activation"]),
                        pre_activation_grad=_read_part(parts["pre_activation_grad"]),
                        module_kwargs=layer["module_kwargs"],
                        batch_first=layer["batch_first"],
                    )
                else:
                    data[name] = _read_part(parts["data"])
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
        """Merge the index of the root dir and of every rank_N/ subdirectory.

        Each directory's log also carries the store's ``sample_id_key`` and
        accumulation convention in its settings sidecar; they are adopted
        from the first directory read, and all must agree (see
        :meth:`_adopt_sample_id_key`).
        """
        merged: dict[str, list[dict]] = {}
        directories = [self._root_dir]
        directories += [d for d in sorted(self._root_dir.glob("rank_*")) if d.is_dir()]
        for directory in directories:
            self._read_index_dir(directory, merged)
        return merged

    def _read_index_dir(self, directory: Path, merged: dict[str, list[dict]]) -> None:
        """Expand one directory's append-only log into *merged*, in write order."""
        log = directory / self._INDEX_LOG_FILE
        if not log.exists():
            return

        meta_path = directory / self._INDEX_META_FILE
        if meta_path.exists():
            with meta_path.open(encoding="utf-8") as f:
                meta = json.load(f)
            # A positional (int) sample_id_key round-trips through JSON natively.
            self._adopt_sample_id_key(meta["sample_id_key"])
            self.declare_gradient_accumulation_steps(
                meta.get("gradient_accumulation_steps", 1),
            )

        with log.open(encoding="utf-8") as f:
            for line in f:
                entry = line.strip()
                if not entry:
                    continue
                try:
                    payload = json.loads(entry)
                except json.JSONDecodeError:
                    # Appends are whole lines, so a short line means the
                    # process died mid-append.
                    warnings.warn(
                        f"Ignoring a truncated final line in {log} -- the "
                        "records it described are not indexed.",
                        stacklevel=2,
                    )
                    break
                _merge_index(merged, _expand_log_line(payload))

    def _persist_index(
        self,
        records: list[GradientRecord],
        location: str,
    ) -> None:
        """Persist one save's index delta for a ``disk`` store; else a no-op.

        ``memory``/``tiered`` stores keep the index in RAM only -- they are
        ephemeral (single-process, not re-opened), and a spilled tiered group's
        files carry no standalone index (the live index in RAM already points
        at them).

        Args:
            records: The records just written, in group order.
            location: The group's location handle.
        """
        if self._residency == "disk":
            self._append_index(records, location)

    def _write_index_meta(self, save_dir: Path) -> None:
        """Write the settings sidecar iff its content changed since last write.

        The payload (format, identifier scheme, accumulation convention) is
        adopted once and then never varies, so writing it on every save cost a
        temp file plus a rename per save for a file that never changed --
        which measured as the bulk of the index write.  The first save of a
        process still writes it before appending any log line, so a log never
        exists without the settings to read it by.
        """
        meta = {
            "format": self._INDEX_FORMAT,
            "sample_id_key": self._sample_id_key,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
        }
        if meta == self._meta_written:
            return
        meta_tmp = save_dir / (self._INDEX_META_FILE + ".tmp")
        with meta_tmp.open("w", encoding="utf-8") as f:
            json.dump(meta, f)
        meta_tmp.replace(save_dir / self._INDEX_META_FILE)
        self._meta_written = meta

    def _append_index(self, records: list[GradientRecord], location: str) -> None:
        """Append one line describing *records* to this rank's index log.

        Called after the group file has been written.  The line is emitted as
        a single ``write`` of a complete, newline-terminated JSON object and
        flushed to the OS.

        ``flush()`` reaches the OS page cache, not the platter -- nothing in
        this class is ``fsync``ed.  The write-then-index ordering therefore
        holds across a process crash but not a power loss, which may leave the
        log durable and its group file not.

        Args:
            records: The records just written, in group order -- their position
                in this list is the ``idx`` used to address them on load.
            location: The group's location handle, relative to the root
                *save_dir*.
        """
        save_dir = self._ensure_save_dir()
        save_dir.mkdir(parents=True, exist_ok=True)
        self._write_index_meta(save_dir)

        line = {
            "file": location,
            "records": [
                {
                    "idx": idx,
                    "step": record.step,
                    "hashes": [
                        str(h)
                        for h in (
                            record.input_hash
                            if isinstance(record.input_hash, list)
                            else [record.input_hash]
                        )
                    ],
                }
                for idx, record in enumerate(records)
            ],
        }
        with (save_dir / self._INDEX_LOG_FILE).open("a", encoding="utf-8") as f:
            f.write(json.dumps(line, separators=(",", ":")) + "\n")
            f.flush()

    def _compute_next_batch_id(self) -> int:
        """Settle the auto-name counter above every ``batch_*`` file present.

        Must see *every* disk format's files: a group is ``batch_<id>.pt``
        under pickle but ``batch_<id>.mmap.bin`` / ``.mmap.meta`` under memmap,
        so the id is read from the name between ``batch_`` and the first dot
        rather than from a single assumed extension.
        """
        if not self._save_dir.exists():
            return 0
        prefix = "batch_"
        ids = []
        for path in self._save_dir.glob(f"{prefix}*"):
            digits = path.name[len(prefix) :].split(".")[0]
            if digits.isdigit():
                ids.append(int(digits))
        return max(ids) + 1 if ids else 0

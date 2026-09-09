"""One cache abstraction for the whole library.

Every intermediate the library may keep around -- projection matrices,
materialized train blocks during scoring, preconditioned test representations,
in-RAM gradient record groups -- goes through the same two objects:

* :class:`CacheBudget` -- how many bytes a cache may take on a device.  A
  cache is an optimization, so it is never allowed to be the reason a run
  dies: the budget is a fraction of the *free* device memory, and a
  :class:`TensorCache` refuses an entry that would exceed it instead of
  evicting or raising.
* :class:`TensorCache` -- a keyed store of tensors (or nested containers of
  tensors) with a *residency* chosen once at construction:

  - ``"memory"`` -- entries stay in RAM / on their device, bounded by the
    budget (an entry that does not fit is simply not cached).
  - ``"disk"`` -- every entry is written to ``spill_dir`` and read back on
    access.
  - ``"tiered"`` -- entries stay in memory up to the budget, then the oldest
    entries spill to ``spill_dir``.

  A cache is a context manager: ``close()`` drops the in-memory entries and
  removes any spill directory it created, so a cache scoped by ``with`` can
  never outlive its owner.

The vocabulary (:data:`CACHE_RESIDENCIES`) is shared with
:class:`~dattri_llm.gradient.storage_manager.GradientStorageManager`, whose
``memory``/``tiered`` residencies are a :class:`TensorCache` of record groups.
This module depends on torch only.
"""

from __future__ import annotations

import contextlib
import shutil
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable

    from typing_extensions import Self

CACHE_RESIDENCIES = ("disk", "memory", "tiered")

# Fraction of *free* device memory a cache may occupy by default.  Scoring
# still needs room for the model, activations and the GEMM workspace, so a
# cache is deliberately given a minority share.
DEFAULT_CACHE_FRACTION = 0.35


def tensor_nbytes(value: object) -> int:
    """Bytes held by *value*: a tensor, an object exposing ``nbytes``, or a
    ``dict``/``list``/``tuple`` of those (recursively).  Anything else counts
    as zero.
    """
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    nbytes = getattr(value, "nbytes", None)
    if isinstance(nbytes, int):
        return nbytes
    if isinstance(value, dict):
        return sum(tensor_nbytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_nbytes(v) for v in value)
    return 0


def available_host_bytes() -> int:
    """Currently available host RAM (from ``/proc/meminfo``), or 16 GiB when
    it cannot be read.
    """
    try:
        with Path("/proc/meminfo").open(encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return 16 * 2**30


class CacheBudget:
    """Byte budget a cache may occupy on a device.

    Args:
        device: The device whose free memory bounds the budget.  ``None``
            means the host.
        fraction: Share of the currently *free* memory the budget covers.
        limit_bytes: Explicit cap instead of the free-memory fraction.

    The budget is re-read from the device on every :meth:`available` call,
    so a cache adapts to what the rest of the run is using.
    """

    def __init__(
        self,
        device: torch.device | str | None = None,
        *,
        fraction: float = DEFAULT_CACHE_FRACTION,
        limit_bytes: int | None = None,
    ) -> None:
        self._device = torch.device(device) if device is not None else None
        self._fraction = fraction
        self._limit = limit_bytes

    @property
    def device(self) -> torch.device | None:
        """The device this budget is measured on (``None`` = host)."""
        return self._device

    def total(self) -> int:
        """Bytes the cache may hold in total, right now."""
        if self._limit is not None:
            return self._limit
        if self._device is not None and self._device.type == "cuda":
            free, _total = torch.cuda.mem_get_info(self._device)
            return int(free * self._fraction)
        return int(available_host_bytes() * self._fraction)

    def fits(self, nbytes: int, held: int = 0) -> bool:
        """Whether *nbytes* more may be cached given *held* bytes already are."""
        return held + nbytes <= self.total()


class TensorCache:
    """Keyed, budgeted, residency-aware cache of tensors.

    Args:
        residency: One of :data:`CACHE_RESIDENCIES`.
        budget: Byte budget for the in-memory tier (``memory`` and
            ``tiered``).  ``None`` means unbounded.
        spill_dir: Directory the ``disk``/``tiered`` residencies write to.
            Created on demand; when ``None`` a temporary directory is created
            and removed by :meth:`close`.  A callable returning a path is
            invoked lazily on the first spill and the directory it returns is
            likewise owned (removed) by this cache.
        writer: ``(path, value) -> Path | None`` serialiser for spilled
            entries; defaults to ``torch.save``.  It may return the path it
            actually wrote (e.g. with a format-specific extension), which is
            then the entry's recorded location.
        reader: ``(path) -> value`` deserialiser; defaults to ``torch.load``.
        on_spill: Called with ``(key, path)`` after an entry is written to
            disk, for owners that track entry locations themselves.
        nbytes_fn: Byte counter used for the budget; defaults to
            :func:`tensor_nbytes`.

    Entries are addressed by any hashable key.  The cache is a context
    manager; :meth:`close` releases everything it holds.
    """

    def __init__(
        self,
        residency: str = "memory",
        *,
        budget: CacheBudget | int | None = None,
        spill_dir: str | Path | Callable[[], Path] | None = None,
        writer: Callable[[Path, object], Path | None] | None = None,
        reader: Callable[[Path], object] | None = None,
        on_spill: Callable[[Hashable, Path], None] | None = None,
        nbytes_fn: Callable[[object], int] | None = None,
    ) -> None:
        if residency not in CACHE_RESIDENCIES:
            raise ValueError(
                f"residency must be one of {list(CACHE_RESIDENCIES)}, "
                f"got {residency!r}.",
            )
        self._residency = residency
        self._budget = (
            CacheBudget(limit_bytes=budget) if isinstance(budget, int) else budget
        )
        self._spill_dir_factory = spill_dir if callable(spill_dir) else None
        self._spill_dir = (
            Path(spill_dir)
            if spill_dir is not None and not callable(spill_dir)
            else None
        )
        self._owns_spill_dir = spill_dir is None or callable(spill_dir)
        self._writer = writer if writer is not None else _torch_save
        self._reader = reader if reader is not None else _torch_load
        self._on_spill = on_spill
        self._nbytes = nbytes_fn if nbytes_fn is not None else tensor_nbytes
        # Insertion-ordered so the tiered residency spills the oldest first.
        self._memory: OrderedDict[Hashable, object] = OrderedDict()
        self._sizes: dict[Hashable, int] = {}
        self._held = 0
        self._disk: dict[Hashable, Path] = {}
        self._seq = 0
        self._closed = False

    # ------------------------------------------------------------------ #
    # Introspection                                                       #
    # ------------------------------------------------------------------ #

    @property
    def residency(self) -> str:
        """Where entries live: ``"disk"``, ``"memory"`` or ``"tiered"``."""
        return self._residency

    @property
    def nbytes(self) -> int:
        """Bytes currently held in the in-memory tier."""
        return self._held

    @property
    def spill_dir(self) -> Path | None:
        """The directory spilled entries are written to (``None`` until the
        first spill of a cache created without an explicit directory).
        """
        return self._spill_dir

    def __len__(self) -> int:
        return len(self._memory) + len(self._disk)

    def __contains__(self, key: Hashable) -> bool:
        return key in self._memory or key in self._disk

    def keys(self) -> list[Hashable]:
        """Every cached key, memory tier first."""
        return [*self._memory.keys(), *self._disk.keys()]

    def location(self, key: Hashable) -> Path | None:
        """Path of a spilled entry, or ``None`` while it is in memory."""
        return self._disk.get(key)

    # ------------------------------------------------------------------ #
    # Access                                                              #
    # ------------------------------------------------------------------ #

    def get(self, key: Hashable, default: object = None) -> object:
        """The cached value for *key*, or *default*."""
        if key in self._memory:
            return self._memory[key]
        path = self._disk.get(key)
        if path is None:
            return default
        return self._reader(path)

    def put(self, key: Hashable, value: object) -> bool:
        """Cache *value* under *key*.

        Returns ``True`` when the entry is retained.  Under ``memory``
        residency an entry that does not fit the budget is **not** cached
        (and ``False`` is returned) -- caching is an optimization and must
        never be the reason a run runs out of memory.  Under ``tiered`` the
        oldest entries spill to disk to make room; under ``disk`` the entry
        is written immediately.
        """
        self._check_open()
        self.discard(key)
        if self._residency == "disk":
            self._spill(key, value)
            return True
        nbytes = self._nbytes(value)
        if self._budget is not None and not self._budget.fits(nbytes, self._held):
            if self._residency == "memory":
                return False
            # tiered: evict oldest until this entry fits (or nothing is left).
            while self._memory and not self._budget.fits(nbytes, self._held):
                oldest = next(iter(self._memory))
                self._spill(oldest, self._memory.pop(oldest))
                self._held -= self._sizes.pop(oldest)
            if not self._budget.fits(nbytes, self._held):
                self._spill(key, value)
                return True
        self._memory[key] = value
        self._sizes[key] = nbytes
        self._held += nbytes
        return True

    def get_or_compute(self, key: Hashable, compute: Callable[[], object]) -> object:
        """Return the cached value for *key*, computing and caching it if absent.

        The computed value is returned even when it could not be retained.
        """
        if key in self:
            return self.get(key)
        value = compute()
        self.put(key, value)
        return value

    def discard(self, key: Hashable) -> None:
        """Drop *key* if present (a spilled file is deleted)."""
        if key in self._memory:
            del self._memory[key]
            self._held -= self._sizes.pop(key)
        path = self._disk.pop(key, None)
        if path is not None:
            for p in _entry_files(path):
                p.unlink(missing_ok=True)

    def clear(self) -> None:
        """Drop every entry (memory and disk)."""
        for key in self.keys():
            self.discard(key)

    def close(self) -> None:
        """Release everything; idempotent.  Removes the spill directory when
        this cache created it.
        """
        if self._closed:
            return
        self._closed = True
        self._memory.clear()
        self._sizes.clear()
        self._held = 0
        self._disk.clear()
        if self._owns_spill_dir and self._spill_dir is not None:
            shutil.rmtree(self._spill_dir, ignore_errors=True)
            self._spill_dir = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> bool:
        self.close()
        return False

    def __del__(self) -> None:
        # Guarded: at interpreter shutdown modules may already be torn down.
        with contextlib.suppress(Exception):
            self.close()

    # ------------------------------------------------------------------ #
    # Internals                                                           #
    # ------------------------------------------------------------------ #

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("TensorCache is closed.")

    def _ensure_spill_dir(self) -> Path:
        if self._spill_dir is None:
            if self._spill_dir_factory is not None:
                self._spill_dir = Path(self._spill_dir_factory())
            else:
                self._spill_dir = Path(tempfile.mkdtemp(prefix="dattri_cache_"))
        self._spill_dir.mkdir(parents=True, exist_ok=True)
        return self._spill_dir

    def _spill(self, key: Hashable, value: object) -> None:
        directory = self._ensure_spill_dir()
        path = directory / f"entry_{self._seq:08d}.pt"
        self._seq += 1
        written = self._writer(path, value)
        path = Path(written) if written is not None else path
        self._disk[key] = path
        if self._on_spill is not None:
            self._on_spill(key, path)


def _torch_save(path: Path, value: object) -> None:
    torch.save(value, path)


def _torch_load(path: Path) -> object:
    return torch.load(path, map_location="cpu", weights_only=False)


def _entry_files(path: Path) -> list[Path]:
    """The file(s) backing a spilled entry (a writer may add sidecars that
    share the entry's stem).
    """
    if not path.parent.exists():
        return []
    return [p for p in path.parent.iterdir() if p.name.startswith(path.stem)]

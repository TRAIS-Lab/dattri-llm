"""Pinned baseline-library versions, asserted at adapter start.

Every baseline adapter calls :func:`require` before doing any work and runs
only against the version recorded here.  ``log.device_details`` writes every
baseline's installed version into each result row.

``dattri_llm`` is not listed: it runs from the working tree.
"""

from __future__ import annotations

import importlib.metadata

BASELINE_VERSIONS: dict[str, str] = {
    "bergson": "0.26.1",
    "kronfluence": "1.0.1",
    "logix": "0.1.1",
}


# Import name -> distribution name where they differ (``pip install logix-ai``
# provides ``import logix``); ``importlib.metadata`` looks up the latter.
DIST_NAMES: dict[str, str] = {"logix": "logix-ai"}


def installed(lib: str) -> str | None:
    """Installed version of *lib*, or ``None`` if it is not importable."""
    try:
        return importlib.metadata.version(DIST_NAMES.get(lib, lib))
    except importlib.metadata.PackageNotFoundError:
        return None


def require(lib: str) -> str:
    """Raise RuntimeError unless *lib* is installed at the pinned version; return it."""
    expected = BASELINE_VERSIONS[lib]
    found = installed(lib)
    if found != expected:
        raise RuntimeError(
            f"{lib} version mismatch: this benchmark is pinned to {lib}=={expected} "
            f"(versions.py) but {found or 'nothing'} is installed.",
        )
    return found

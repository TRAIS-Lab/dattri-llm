"""Pinned baseline-library versions, asserted at adapter start.

The cross-library tables compare *implementations*, so the version behind a
cell is part of the result.  Every baseline adapter calls :func:`require` before
doing any work and refuses to run against anything but the version recorded
here, and ``log.device_details`` writes every baseline's installed version into
each result row so the provenance can be read back from the data.

These are the versions the paper's A40 tables were produced with (see
``paper/EXPERIMENT_LOG.md``); the H200 ladders pin the Modal image to the same
values from this dict, so the pin and the assertion cannot drift apart.

Our own library (``dattri_llm``) is deliberately not listed: it is the thing
under test and runs from the working tree.
"""

from __future__ import annotations

import importlib.metadata

BASELINE_VERSIONS: dict[str, str] = {
    "bergson": "0.26.1",
    "kronfluence": "1.0.1",
    "logix": "0.1.1",
}


# Import name -> distribution name where they differ (``pip install logix-ai``
# provides ``import logix``); ``importlib.metadata`` only knows the latter.
DIST_NAMES: dict[str, str] = {"logix": "logix-ai"}


def installed(lib: str) -> str | None:
    """Installed version of *lib*, or ``None`` if it is not importable."""
    try:
        return importlib.metadata.version(DIST_NAMES.get(lib, lib))
    except importlib.metadata.PackageNotFoundError:
        return None


def require(lib: str) -> str:
    """Refuse to run unless *lib* is at the pinned version; return that version."""
    expected = BASELINE_VERSIONS[lib]
    found = installed(lib)
    if found != expected:
        raise RuntimeError(
            f"{lib} version mismatch: this benchmark is pinned to {lib}=={expected} "
            f"(versions.py) but {found or 'nothing'} is installed. Install the "
            f"pinned version rather than editing the pin -- a different version "
            f"is a different baseline.",
        )
    return found

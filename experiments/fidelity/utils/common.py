"""Shared pieces of the attribution drivers: the run directory with its log
and phase timing, the final-checkpoint trajectory, Bergson's on-disk
datasets, version check and score loading, and the fidelity scoring of a
``(n_train, n_queries)`` matrix against a run's ground truth."""

from __future__ import annotations

import json
import pathlib
import shutil
import time

import numpy as np
import torch

from protocol import make_batches, peak_gb, run_trajectory, spearman_per_column


class Run:
    """The run directory ``<setting's out_dir>_<variant>/`` with a log and phase timing.

    A phase is the wall-clock time between ``start(phase)`` and ``end()``,
    with the device synchronized at both ends, together with the peak
    allocated device memory inside it.  ``finish`` writes ``result.json``:
    the driver's fields plus ``train_s`` (the ``train`` phase),
    ``attribute_s`` (all other phases), ``total_s``, ``peak_gb`` (the
    maximum over the phases) and ``phases``.
    """

    def __init__(self, s, variant: str) -> None:
        self.s = s
        self.out = s.out_dir.parent / f"{s.out_dir.name}_{variant}"
        self.out.mkdir(parents=True, exist_ok=True)
        self._log = self.out / "log.txt"
        self.phases: dict[str, float] = {}
        self.peaks: dict[str, float] = {}
        self._t0 = None
        self._phase = None

    def log(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with self._log.open("a") as f:
            f.write(line + "\n")

    def start(self, phase: str) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self._t0, self._phase = time.time(), phase

    def end(self, note: str = "") -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        secs = time.time() - self._t0
        self.phases[self._phase] = round(secs, 1)
        self.peaks[self._phase] = round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else 0.0
        self.log(f"  {self._phase}: {secs:.0f}s, {peak_gb()}{(' ' + note) if note else ''}")

    def timing(self) -> dict:
        train = self.phases.get("train", 0.0)
        return {"train_s": train, "attribute_s": round(sum(self.phases.values()) - train, 1),
                "total_s": round(sum(self.phases.values()), 1),
                "peak_gb": max(self.peaks.values()) if self.peaks else None, "phases": self.phases}

    def finish(self, result: dict) -> dict:
        result = {**result, **self.timing()}
        self.log(f"  result: {json.dumps(result)}")
        (self.out / "result.json").write_text(json.dumps(result, indent=2))
        return result


def batches_of(s) -> list[torch.Tensor]:
    """The index batches of the setting's trajectory, one per step."""
    return make_batches(s.n_train, s.batch_size, s.epochs, s.seed)


def final_checkpoint(s, batches, path: pathlib.Path):
    """Run the trajectory; save the final model as an HF checkpoint at *path*."""
    model, _ = run_trajectory(s, batches)
    shutil.rmtree(path, ignore_errors=True)
    model.save_pretrained(path, safe_serialization=True)
    return model


def token_dataset(x: torch.Tensor, path: pathlib.Path) -> str:
    """Bergson's on-disk dataset of fixed-length token blocks."""
    from datasets import Dataset

    ids = x.cpu().tolist()
    ds = Dataset.from_dict({"input_ids": ids, "length": [len(r) for r in ids]})
    shutil.rmtree(path, ignore_errors=True)
    ds.save_to_disk(str(path))
    return str(path)


BERGSON_VERSION = "0.26.1"  # the Bergson version the baseline drivers require


def require_bergson() -> None:
    """Raise unless the installed Bergson is ``BERGSON_VERSION``."""
    import importlib.metadata

    try:
        found = importlib.metadata.version("bergson")
    except importlib.metadata.PackageNotFoundError:
        found = None
    if found != BERGSON_VERSION:
        raise RuntimeError(
            f"bergson=={BERGSON_VERSION} is required but {found or 'nothing'} is installed"
        )


def bergson_scores(path: pathlib.Path) -> torch.Tensor:
    """``(n_train, n_queries)`` from a Bergson ``scores`` directory."""
    from bergson.data import load_scores

    return torch.as_tensor(np.array(load_scores(path)[:])).float()


def fidelity(s, batches, scores: torch.Tensor, truth_dir: str | None, run: Run, method: str,
             n_queries: int) -> dict:
    """Mean Spearman correlation of *scores* with the ground truth of *truth_dir*.

    *scores* is ``(n_train, n_queries)`` with rows in training order and
    positive = removal raises the loss.  The rows of the ground truth's
    ``(sample, step)`` pairs are compared with its ``tsloo`` matrix column by
    column; the mean and the standard deviation over the queries are
    returned and the compared rows are saved as ``matrices.pt`` in the run
    directory.  Without *truth_dir* the scores are saved as
    ``<method>_scores.pt`` and nothing is returned.
    """
    if not truth_dir:
        torch.save({"scores": scores}, run.out / f"{method}_scores.pt")
        return {}
    ref = torch.load(pathlib.Path(truth_dir) / "matrices.pt", weights_only=False)
    pairs = ref["pairs"]
    n_q = min(n_queries, scores.shape[1])
    truth = ref["tsloo"][:, :n_q]
    pred = torch.stack([scores[i, :n_q] for i, _ in pairs])
    rho = spearman_per_column(pred, truth)
    out = {method: float(rho.mean()), f"{method}_std_over_val": float(rho.std()), "n_queries": n_q}
    torch.save({"pred": pred, "pairs": pairs}, run.out / "matrices.pt")
    return out

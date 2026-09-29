"""Timing hook for Bergson's spawned worker processes (benchmark only).

Bergson runs its sharded steps in worker processes it starts with the
``spawn`` method, and every worker loads the model
(``setup_model_and_peft``).  Python imports ``sitecustomize`` at interpreter
start-up, so with this directory on ``PYTHONPATH`` each process appends, to
``$BERGSON_BENCH_TIMING_DIR/<pid>.jsonl``,

    {"ev": "proc_start", "t": ...}                       # interpreter start
    {"ev": "setup_model_and_peft", "t_start": ..., "wall_s": ...}
    {"ev": "setup_data_pipeline", "t_start": ..., "wall_s": ...}

and ``run_bergson.py`` (``_worker_overhead``) sums those into the workers'
model-load and start-up times of each phase.  Inactive (one ``os.environ``
lookup) unless the variable is set; ``run_bergson.py`` sets both variables
for sharded runs.
"""

import os

_DIR = os.environ.get("BERGSON_BENCH_TIMING_DIR")

if _DIR:
    import importlib.abc
    import json
    import sys
    import time

    _T0 = time.time()
    _NAMES = ("setup_model_and_peft", "setup_data_pipeline")
    # Bergson modules that bind the two functions by name (``from ... import``).
    _TARGETS = {
        "bergson.utils.worker_utils",
        "bergson.build",
        "bergson.score.score",
        "bergson.hessians.hessian_approximations",
        "bergson.query.query_index",
    }

    def _log(rec: dict) -> None:
        try:
            with open(os.path.join(_DIR, f"{os.getpid()}.jsonl"), "a") as fh:
                fh.write(json.dumps({**rec, "pid": os.getpid()}) + "\n")
        except OSError:
            pass

    _log({"ev": "proc_start", "t": _T0, "argv": sys.argv[:2]})
    _wrapped: dict = {}

    def _wrap(name: str, fn):
        if getattr(fn, "_bench_timed", False):
            return fn
        if id(fn) in _wrapped:
            return _wrapped[id(fn)]

        def timed(*args, **kwargs):
            t0 = time.time()
            try:
                return fn(*args, **kwargs)
            finally:
                try:
                    import torch

                    if torch.cuda.is_available() and torch.cuda.is_initialized():
                        torch.cuda.synchronize()
                except Exception:  # noqa: BLE001 - timing must never break the run
                    pass
                _log({"ev": name, "t_start": t0, "wall_s": time.time() - t0})

        timed._bench_timed = True
        _wrapped[id(fn)] = timed
        return timed

    class _PatchAfterImport(importlib.abc.MetaPathFinder):
        """Wrap the two functions in each target module right after it loads."""

        def find_spec(self, fullname, path, target=None):
            if fullname not in _TARGETS:
                return None
            for finder in sys.meta_path:
                if finder is self or not hasattr(finder, "find_spec"):
                    continue
                spec = finder.find_spec(fullname, path, target)
                if spec is not None and spec.loader is not None:
                    break
            else:
                return None
            real_exec = spec.loader.exec_module

            def exec_module(module, _real=real_exec):
                _real(module)
                for name in _NAMES:
                    if hasattr(module, name):
                        setattr(module, name, _wrap(name, getattr(module, name)))

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _PatchAfterImport())

"""bergson side: README-quickstart gradient index build + batch query.

Build (their CLI, one subprocess, single GPU):
    bergson build <index> --model EleutherAI/pythia-14m
        --dataset NeelNanda/pile-10k --truncation --token_batch_size 2048
        --projection_dim 16

Query: programmatic ``Attributor`` over the on-disk index, scoring the first
16 pile-10k documents against the whole index (plain grad-dot; the CLI's
``bergson query`` is an interactive top-k demo around the same machinery).

Peak GPU memory for the build subprocess is sampled device-wide via NVML
(coarser than torch's allocator counter; noted in results).

    python run_bergson.py --stage build
    python run_bergson.py --stage query
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # common.py
sys.path.insert(0, str(HERE))

import torch

from common import Meter, dir_bytes, env_info
from data import DATASET, MODEL_NAME, N_QUERY

INDEX = HERE / "bergson_index"


class SmiPeak:
    """Polls nvidia-smi for device memory.used while a subprocess runs."""

    def __init__(self, gpu: str) -> None:
        self.gpu = gpu
        self.peak_mb = 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._poll, daemon=True)

    def _poll(self):
        while not self._stop.is_set():
            out = subprocess.run(
                ["nvidia-smi", "-i", self.gpu,
                 "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=False,
            ).stdout.strip()
            if out.isdigit():
                self.peak_mb = max(self.peak_mb, int(out))
            time.sleep(0.5)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join(timeout=2)


def run_build(meter: Meter, gpu: str):
    cmd = [
        sys.executable, "-m", "bergson", "build", str(INDEX),
        "--model", MODEL_NAME, "--dataset", DATASET,
        "--truncation", "--token_batch_size", "2048",
        "--projection_dim", "16",
    ]
    t0 = time.perf_counter()
    with SmiPeak(gpu) as smi:
        subprocess.run(cmd, check=True)
    wall = time.perf_counter() - t0
    meter.rows.append({
        "lib": meter.lib, "setting": meter.setting, "phase": "build",
        "wall_s": round(wall, 2), "throughput": round(10_000 / wall, 2),
        "unit": "docs/s", "work_units": 10_000,
        "peak_mem_gb_smi": round(smi.peak_mb / 1024, 2), **meter.meta,
    })
    meter.note(store_bytes=dir_bytes(INDEX))


def run_query(meter: Meter):
    import torch.nn.functional as F

    from bergson.query.attributor import Attributor
    from data import Docs, load_model, load_tokenized

    ds = load_tokenized()
    queries = Docs(ds.select(range(N_QUERY)))
    model = load_model().cuda()

    with meter.phase("query", N_QUERY, unit="queries"):
        attributor = Attributor(str(INDEX), device="cuda", unit_norm=False)
        # The index's module names are rooted at the base transformer
        # (``layers.*``), so trace that submodule; hooks still fire during the
        # full-model backward.  All queries go through ONE padded backward --
        # the collector emits per-sample grads (N, d) per module, and
        # concatenates *separate* backwards along the feature dim.
        rows = [queries[i]["input_ids"] for i in range(N_QUERY)]
        n = max(len(r) for r in rows)
        ids = torch.zeros((N_QUERY, n), dtype=torch.long)
        mask = torch.zeros((N_QUERY, n), dtype=torch.long)
        for j, r in enumerate(rows):
            ids[j, :len(r)] = r
            mask[j, :len(r)] = 1
        ids, mask = ids.cuda(), mask.cuda()
        with attributor.trace(model.gpt_neox, k=None) as result:
            logits = model(input_ids=ids, attention_mask=mask).logits
            labels = ids.masked_fill(mask == 0, -100)
            loss = F.cross_entropy(
                logits[:, :-1].flatten(0, 1), labels[:, 1:].flatten(),
                reduction="sum", ignore_index=-100)
            model.zero_grad()
            loss.backward()
        # scatter top-k(=N) scores back into index order -> (n_query, N)
        scores = result.scores.float().cpu()
        indices = result.indices.long().cpu()
        n = scores.shape[-1]
        full = torch.zeros(N_QUERY, n)
        full.scatter_(1, indices, scores)
    torch.save({"score": full.T}, HERE / "score_bergson.pt")  # (N, n_query)
    meter.note(score_shape=[n, N_QUERY])


def main(a):
    meter = Meter("bergson", "pythia14m-pile10k-proj16-index",
                  HERE / "results.jsonl", meta=env_info())
    if a.stage == "build":
        run_build(meter, a.gpu)
    else:
        run_query(meter)
    meter.flush()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["build", "query"], required=True)
    p.add_argument("--gpu", default="0")
    main(p.parse_args())

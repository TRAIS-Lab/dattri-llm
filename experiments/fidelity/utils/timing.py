"""Cost unit for the timing comparison: one plain training run of the
GPT-2 setting (no hooks) plus one validation pass, with peak memory.

    python utils/timing.py --setting gpt2 --lr 1e-5 --seed 0
"""

from __future__ import annotations

import json
import time

import torch

from protocol import make_batches, peak_gb, run_trajectory
from settings import build, parser


def main() -> None:
    a = parser().parse_args()
    s = build(a)
    out_dir = s.out_dir.parent / f"{s.out_dir.name}_timing_baseline"
    batches = make_batches(s.n_train, s.batch_size, s.epochs, s.seed)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    model, _ = run_trajectory(s, batches)
    torch.cuda.synchronize()
    t_train = time.time() - t0
    t0 = time.time()
    losses = s.val_losses(model)
    torch.cuda.synchronize()
    t_val = time.time() - t0
    out = {
        "name": s.name,
        "seed": s.seed,
        "lr": s.extra["lr"],
        "method": "baseline",
        "steps": len(batches),
        "train_s": round(t_train, 1),
        "val_s": round(t_val, 1),
        "peak": peak_gb(),
        "mean_val_loss": float(losses.mean()),
    }
    print(json.dumps(out))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

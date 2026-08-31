"""logix adapter for the universal benchmark.

Runs logix's own LoGRA extract + compute_influence pipeline (rank-64 LoRA, the
kernel-free analog of our proj-64 K-FAC) on a universal (HF model, dataset) task
and logs through ``log.BenchRun``.

    method -> logix hessian:  kfac -> "kfac",  graddot -> "raw"

Single process: watch + add_lora, extract (grad/hessian), then score.  The layer
name filter ["att", "mlp"] matches attention + MLP linears across gpt2 / GPT-NeoX
(pythia) / Qwen / Llama.  Run in the ``dattri`` conda env.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
sys.path.insert(0, str(BENCH))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import models
from data import load_task_data
from log import BenchRun

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True

LIB = "logix"
_HESSIAN = {"kfac": "kfac", "graddot": "raw", "ekfac": "ekfac"}

# logix predates torch 2.6's weights_only=True default; its saved state holds
# populated defaultdicts the weights-only unpickler cannot rebuild.  Every file
# loaded here is this run's own output, so restore the pre-2.6 behaviour.
_orig_load = torch.load


def _trusting_load(*a, **k):
    k.setdefault("weights_only", False)
    return _orig_load(*a, **k)


torch.load = _trusting_load


def shift_ce_sum(logits, targets):
    sl = logits[..., :-1, :].contiguous()
    st = targets[..., 1:].contiguous()
    return F.cross_entropy(sl.view(-1, sl.size(-1)), st.view(-1),
                           reduction="sum", ignore_index=-100)


def build_model(model_id: str, params_b: float):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    # logix's LoGra/LoRA covariance machinery runs in float32; a bf16 model
    # triggers a dtype-mismatch matmul, so force float32 here.
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model.cuda().eval(), tok


def run(task: dict, out_root: Path) -> None:
    import logix
    from logix import LogIXScheduler
    from logix.utils import merge_logs
    from transformers import default_data_collator

    method = task["method"]
    if method not in _HESSIAN:
        msg = f"logix does not cover method {method!r}"
        raise ValueError(msg)
    hessian = _HESSIAN[method]
    n_train = task.get("n_train", 1024)
    n_test = task.get("n_test", 16)
    block_size = task.get("block_size", 512)
    batch = task.get("batch", 8)
    seed = task.get("seed", 0)

    tag = f"{task.get('family','?')}-{task.get('scale','?')}-{task['dataset']}-{method}"
    run_dir = out_root / "runs" / f"logix-{tag}"
    bench = BenchRun({**task, "n_train": n_train, "n_test": n_test,
                      "block_size": block_size, "batch": batch, "lora_rank": 64},
                     results_path=out_root / "results.jsonl",
                     run_dir=run_dir, lib=LIB)

    with bench.phase("build_model"):
        model, tok = build_model(task["model"], task["params_b"])
    with bench.phase("load_data"):
        train_ds, test_ds = load_task_data(task["model"], task["dataset"],
                                           block_size, n_train, n_test, seed)

    log_root = Path(tempfile.mkdtemp(prefix="logix_"))
    cfg = log_root / "config.yaml"
    cfg.write_text(f"root_dir: {log_root}/logs\nlora:\n  init: random\n  rank: 64\n")
    project = f"{tag}_{hessian}"
    lx = logix.init(project, config=str(cfg))
    lx.watch(model, name_filter=["att", "mlp"])

    proj_mode = task.get("proj_mode", "rank64")
    bench.set(proj_mode=proj_mode)
    with bench.phase("extract", n_train):
        if proj_mode != "full":
            lx.add_lora()  # rank-64 LoRA projection; full-dim skips this
        scheduler = LogIXScheduler(lx, lora="none", hessian=hessian, save="grad")
        loader = DataLoader(train_ds, batch_size=batch, shuffle=False,
                            collate_fn=default_data_collator)
        for _ in scheduler:
            for b in loader:
                data_id = tok.batch_decode(b["input_ids"])
                tgt = b.pop("labels").cuda()
                b = {k: v.cuda() for k, v in b.items()}
                with lx(data_id=data_id, mask=b["attention_mask"]):
                    model.zero_grad()
                    loss = shift_ce_sum(model(**b).logits, tgt)
                    loss.backward()
            logix.finalize()
    bench.record_disk("log_store", log_root / "logs" / project)

    with bench.phase("score", n_train + n_test):
        logix.initialize_from_log()
        log_loader = logix.build_log_dataloader(batch_size=64)
        qloader = DataLoader(test_ds, batch_size=1, shuffle=False,
                             collate_fn=default_data_collator)
        logix.setup({"grad": ["log"]})
        logix.eval()
        test_logs = []
        for b in qloader:
            data_id = tok.batch_decode(b["input_ids"], skip_special_tokens=True)
            tgt = b.pop("labels").cuda()
            b = {k: v.cuda() for k, v in b.items()}
            with lx(data_id=data_id, mask=b["attention_mask"]):
                model.zero_grad()
                loss = shift_ce_sum(model(**b).logits, tgt)
                loss.backward()
            test_logs.append(copy.deepcopy(logix.get_log()))
        result = lx.influence.compute_influence_all(merge_logs(test_logs), log_loader)

    score = result["influence"].T.cpu().float()  # -> (n_train, n_test)
    torch.save({"score": score, "train_ids": result["tgt_ids"]}, run_dir / "score.pt")
    bench.set(score_shape=list(score.shape))
    bench.finish(status="ok")
    print(f"[done] logix {tag}: score {tuple(score.shape)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--task")
    g.add_argument("--task-file", dest="task_file")
    ap.add_argument("--out", default=str(BENCH / "out"))
    a = ap.parse_args()
    if a.task_file:
        payload = json.loads(Path(a.task_file).read_text())
        task = payload.get("task", payload)
    else:
        task = json.loads(a.task)
    run(task, Path(a.out))


if __name__ == "__main__":
    main()

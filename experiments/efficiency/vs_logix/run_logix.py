"""logix side: LoGRA extract + influence, following the library's own
``examples/language_modeling/{extract_log,compute_influence}.py`` verbatim
(single process, single GPU).

    python run_logix.py --stage extract --hessian raw
    python run_logix.py --stage score   --hessian raw
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # common.py

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import Meter, dir_bytes, env_info
from data import BATCH, N_QUERY, load_datasets, load_model_and_tokenizer

import logix
from logix import LogIXScheduler
from logix.utils import merge_logs

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True

# logix predates torch 2.6's weights_only=True default; its saved state holds
# populated defaultdicts, which the weights-only unpickler cannot rebuild at
# all.  Every file loaded here is this run's own output, so fall back to the
# pre-2.6 behaviour for this process.
_orig_torch_load = torch.load


def _trusting_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)


torch.load = _trusting_load


def shift_ce_sum(logits, targets):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = targets[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="sum",
        ignore_index=-100,
    )


def main(a):
    from transformers import default_data_collator

    model, tokenizer = load_model_and_tokenizer()
    train_ds, query_ds = load_datasets(tokenizer)
    model = model.cuda().eval()

    project = f"gpt2_random_{a.hessian}"
    run = logix.init(project, config=str(HERE / "config.yaml"))
    run.watch(model, name_filter=["att", "mlp"])

    meter = Meter(f"logix({a.hessian})", "gpt2-wikitext103-logra64",
                  HERE / "results.jsonl", meta=env_info())

    if a.stage == "extract":
        run.add_lora()
        scheduler = LogIXScheduler(
            run, lora="none", hessian=a.hessian, save="grad")
        loader = DataLoader(train_ds, batch_size=BATCH,
                            shuffle=False, collate_fn=default_data_collator)
        with meter.phase("extract", len(train_ds)):
            for _ in scheduler:
                for batch in loader:
                    data_id = tokenizer.batch_decode(batch["input_ids"])
                    targets = batch.pop("labels").cuda()
                    batch = {k: v.cuda() for k, v in batch.items()}
                    with run(data_id=data_id, mask=batch["attention_mask"]):
                        model.zero_grad()
                        lm_logits = model(**batch)
                        loss = shift_ce_sum(lm_logits, targets)
                        loss.backward()
                logix.finalize()
        meter.note(n_train=len(train_ds),
                   log_store_bytes=dir_bytes(HERE / "logix_logs" / project))
    else:  # score
        logix.initialize_from_log()  # restores LoRA modules + hessian state
        log_loader = logix.build_log_dataloader(batch_size=64)
        qloader = DataLoader(query_ds, batch_size=1,
                             shuffle=False, collate_fn=default_data_collator)
        with meter.phase("score", len(train_ds) + N_QUERY):
            logix.setup({"grad": ["log"]})
            logix.eval()
            merged_test_logs = []
            for batch in qloader:
                data_id = tokenizer.batch_decode(
                    batch["input_ids"], skip_special_tokens=True)
                targets = batch.pop("labels").cuda()
                batch = {k: v.cuda() for k, v in batch.items()}
                with run(data_id=data_id, mask=batch["attention_mask"]):
                    model.zero_grad()
                    logits = model(**batch)
                    loss = shift_ce_sum(logits, targets)
                    loss.backward()
                merged_test_logs.append(copy.deepcopy(logix.get_log()))
            merged_test_log = merge_logs(merged_test_logs)
            result = run.influence.compute_influence_all(
                merged_test_log, log_loader)
        score = result["influence"].T.cpu().float()  # -> (n_train, n_query)
        torch.save({"score": score, "train_ids": result["tgt_ids"]},
                   HERE / f"score_logix_{a.hessian}.pt")
        meter.note(score_shape=list(score.shape))
    meter.flush()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["extract", "score"], required=True)
    p.add_argument("--hessian", choices=["raw", "kfac"], default="raw")
    main(p.parse_args())

"""dattri_llm side: on-the-fly EKFAC on the identical GPT-2 / wikitext-2
setup (same fine-tuned checkpoint, same 48 tracked modules, same batch sizes,
full parameter space -- no projection), estimator-parity with kronfluence's
``ekfac`` strategy.

Pipeline shape also matches: kronfluence streams the train set three times
(covariance pass, lambda pass, scoring pass) and so does our OTF ``attribute``
(K-FAC fit, Lambda fit, scoring); neither stores per-sample gradients.

    python run_dattri_llm.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))              # common.py
sys.path.insert(0, "/scratch/shixuanl/kronfluence")  # examples package
sys.path.insert(0, str(HERE))

import torch
import torch.nn.functional as F
from torch.func import functional_call

from common import Meter, env_info
from examples.wikitext.pipeline import construct_gpt2, get_wikitext_dataset
from task_def import N_QUERY, QUERY_BATCH, TRAIN_BATCH

from dattri.task import AttributionTask
from dattri_llm.attribution.algorithm.kronecker import EKFACAttributor
from dattri_llm.attribution.arguments import AttributionArguments
from dattri_llm.gradient.hooks import HookManagerConfig
from dattri_llm.utils.hashing import hash_sample


class TensorBlocks(torch.utils.data.Dataset):
    def __init__(self, hf_dataset) -> None:
        self.ds = hf_dataset

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, i: int) -> dict:
        row = self.ds[i]
        return {
            "input_ids": torch.tensor(row["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(row["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(row["labels"], dtype=torch.long),
        }


def main():
    train_hf = get_wikitext_dataset(split="eval_train")
    query_hf = get_wikitext_dataset(split="valid", indices=list(range(N_QUERY)))
    train_ds, query_ds = TensorBlocks(train_hf), TensorBlocks(query_hf)

    model = construct_gpt2()
    model.load_state_dict(
        torch.load(HERE / "checkpoints" / "model.pth", weights_only=True))
    model = model.cuda().eval()

    def gl(params, batch):
        logits = functional_call(
            model, params, args=(),
            kwargs={"input_ids": batch["input_ids"],
                    "attention_mask": batch["attention_mask"]},
        ).logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = batch["labels"][..., 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1), reduction="sum",
        )

    ckpt = {k: v.detach().clone() for k, v in model.state_dict().items()}
    task = AttributionTask(loss_func=gl, model=model, checkpoints=[ckpt],
                           target_func=gl)
    args = AttributionArguments(
        output_dir=tempfile.mkdtemp(prefix="eff_kron_"),
        # Half of kronfluence's batch: the scoring pass streams train batches
        # under capture while holding the 32 queries' rotated test reps
        # (~11 GB); batch 32 OOMs the 44 GB A40.
        per_device_train_batch_size=TRAIN_BATCH // 2,
        per_device_eval_batch_size=QUERY_BATCH,
        dataloader_num_workers=0, dataloader_pin_memory=False,
    )
    hook_config = HookManagerConfig(linear_io=[r"attn\.", r"mlp\."])

    meter = Meter("dattri_llm(ekfac,otf)", "gpt2-wikitext2-ekfac",
                  HERE / "results.jsonl", meta=env_info())
    with meter.phase("attribute", len(train_ds) + N_QUERY):
        score = EKFACAttributor(args, task=task).attribute(
            train_ds, query_ds, damping=1e-8,  # kronfluence's default
            hook_config=hook_config)

    th = [hash_sample({"input_ids": train_ds[i]["input_ids"],
                       "attention_mask": train_ds[i]["attention_mask"]})
          for i in range(len(train_ds))]
    vh = [hash_sample({"input_ids": query_ds[i]["input_ids"],
                       "attention_mask": query_ds[i]["attention_mask"]})
          for i in range(N_QUERY)]
    matrix = score.query(th, vh).cpu().float()
    torch.save({"score": matrix}, HERE / "score_ours.pt")
    meter.note(n_train=len(train_ds), score_shape=list(matrix.shape),
               store_bytes=0)
    meter.flush()


if __name__ == "__main__":
    main()

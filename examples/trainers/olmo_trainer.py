"""This example shows gradient collection around the OLMo Trainer.

The training loop is NOT modified — TDA is added by wrapping trainer.fit() in a
HookManager collection context.  A tiny OLMo model is built entirely in Python
(no config YAML) and trained with the real OLMo Trainer on CPU; OLMo's
feed-forward projections (ff_proj / ff_out inside transformer.blocks.<i>) are
selected by regex.

Note: this file must not be named ``olmo.py`` — it would shadow the ai2-olmo
package on import.

Run (with ai2-olmo installed):
    python examples/trainers/olmo_trainer.py
"""

from __future__ import annotations

import os
import pathlib
import sys
import tempfile

# Make the repo importable when running the script directly (no install needed).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

try:
    from olmo.config import DistributedStrategy, ModelConfig, TrainConfig
    from olmo.data.iterable_dataset import IterableDataset as OLMoIterableDataset
    from olmo.model import OLMo
    from olmo.optim import build_optimizer, build_scheduler
    from olmo.train import Trainer
except ImportError:
    raise SystemExit(
        "ai2-olmo is required for this example.\n"
        "Install with:  pip install ai2-olmo"
    )

from dattri_llm.gradient.callbacks import OffloadCallback
from dattri_llm.gradient.file_manager import GradientFileManager
from dattri_llm.gradient.hooks import HookManager, HookManagerConfig
from dattri_llm.utils.hashing import hash_sample

VOCAB, SEQ, BATCH = 256, 32, 2

if __name__ == "__main__":
    # OLMo's Trainer calls dist.reduce() inside train_step(), so a process
    # group must exist even for a single-process CPU run.
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29511")
    dist.init_process_group("gloo", rank=0, world_size=1)

    # a hard-coded tiny TrainConfig: 2 layers, d_model=64 — runs on CPU in
    # seconds; checkpointing is disabled so no persistent save_folder is needed
    cfg = TrainConfig(
        model=ModelConfig(
            d_model=64,
            n_heads=2,
            n_layers=2,
            vocab_size=VOCAB,
            max_sequence_length=SEQ,
        ),
        save_folder=tempfile.mkdtemp(prefix="olmo_ckpt_"),
        max_duration=4,                    # 4 steps = 2 epochs over 4 samples
        device_train_batch_size=BATCH,
        global_train_batch_size=BATCH,
        precision="fp32",
        distributed_strategy=DistributedStrategy.ddp,
        save_interval=None,
        save_interval_unsharded=None,
        save_interval_ephemeral=None,
        save_num_checkpoints_to_keep=0,
    )

    # build the model and the real OLMo Trainer (a dataclass taking pre-built
    # components); hooks go on the UNWRAPPED model so they see the real Linear
    # modules regardless of the DDP wrapping
    device = torch.device("cpu")
    model = OLMo(cfg.model).to(device)
    dist_model = DDP(model)
    optim = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg)

    torch.manual_seed(0)
    sequences = [{"input_ids": torch.randint(0, VOCAB, (SEQ,))} for _ in range(4)]
    ds = OLMoIterableDataset(sequences, global_batch_size=BATCH,
                             shuffle=False, drop_last=True)
    loader = DataLoader(ds, batch_size=BATCH)

    trainer = Trainer(
        cfg=cfg,
        model=model,
        dist_model=dist_model,
        optim=optim,
        scheduler=scheduler,
        train_loader=loader,
        device=device,
        evaluators=[],
    )

    with tempfile.TemporaryDirectory() as grad_dir:
        # select OLMo's feed-forward projections by regex — the layers live at
        # transformer.blocks.<i>.ff_proj / ff_out and are plain nn.Linear, so
        # the factorized ("ghost") linear_io hooks apply directly
        fm = GradientFileManager(grad_dir)
        collector = HookManager(
            model,
            config=HookManagerConfig(linear_io=[r"ff_proj", r"ff_out"]),
            callbacks=[OffloadCallback(offload_interval=1, file_manager=fm,
                                       recording_type="per_batch")],
        )
        print(f"{'Hooked FFN layers':<25}{collector.layer_names}")

        # the training loop itself is untouched: wrap trainer.fit() and every
        # backward pass inside is captured automatically
        print("\nRunning trainer.fit() with gradient collection ...")
        with collector.collect():
            trainer.fit()
        collector.remove()

        # Retrieval: the content hash of ONE sample's model inputs — here
        # simply sequences[0], since OLMo's train step feeds the model
        # input_ids — identifies the sample independently of where shuffling
        # put it; lookup() then reveals every (step, sample_idx) position it was
        # recorded at, and load_sample() retrieves each pair's gradient by
        # direct slicing.
        h0 = hash_sample(sequences[0])
        pairs = fm.lookup_by_hash(h0)                          # [(step, sample_idx), ...]
        g_first = fm.load_sample_by_hash(h0, *pairs[0])
        g_last = fm.load_sample_by_hash(h0, *pairs[-1])
        drift = g_first.similarity(g_last, metric="cosine", reduce="all").item()

        print(f"\n{'Steps collected':<28}{collector.steps_collected}")
        print(f"{'Sample records':<28}{len(fm.index)}")
        print("-" * 50)
        print(f"{'Sample 0 (step, sample_idx)':<28}{pairs}")
        print(f"{'cos(first, last)':<28}{drift:.6f}")
        print("-" * 50)

    dist.destroy_process_group()

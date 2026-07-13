"""Gradient accumulation in :class:`GradientStreamer` trajectories.

``args.gradient_accumulation_steps == N`` mirrors HF ``Trainer``: the
optimizer/scheduler advance once per window of N micro-batches, each
micro-batch loss is scaled by ``1/N`` before backward, ``zero_grad`` happens
only at window starts, and a ragged trailing window still updates (as at a
Trainer epoch end).  Every micro-batch keeps yielding its own gradient block.

Pinned here:

1. **Trajectory semantics**: window-by-window, the parameter update equals
   SGD applied to the sum of the window's captured per-sample gradients
   (which carry the 1/N loss scaling), including the ragged final window --
   reconstructed purely from the streamer's own yielded records.
2. **Scheduler cadence**: the LR advances per *update*, not per micro-batch;
   micro-batches of one window record the same LR, and the schedule is sized
   in updates (``ceil(batches / N)``).
3. **Trainer parity**: identical final parameters to a real
   ``transformers.Trainer`` run with ``gradient_accumulation_steps=N``.  The
   streamer implements Trainer's *classic* convention (``loss / N``), which
   Trainer applies whenever the model does not accept loss kwargs -- pinned
   by wrapping the model with an explicit-signature module.  (For models
   that do accept them, newer Trainer versions instead normalise by the
   window's token count -- ``num_items_in_batch`` -- which the streamer does
   not replicate; a custom ``loss_fn`` can bake in any normalisation.)
4. **Frozen probes ignore the setting** entirely.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from dattri_llm.attribution.arguments import AttributionArguments
from dattri_llm.gradient import ops
from dattri_llm.gradient.streaming import GradientStreamer

SEED = 0
IN_DIM, HID_DIM, OUT_DIM = 4, 8, 3
BATCH = 2
LR = 0.1
ATOL = 1e-5


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Sequential()
        self.mlp.add_module("fc1", nn.Linear(IN_DIM, HID_DIM, bias=False))
        self.mlp.add_module("fc2", nn.Linear(HID_DIM, OUT_DIM, bias=False))

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        return self.mlp(x)


class DictDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self.x, self.y = x, y

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, i: int) -> dict:
        return {"x": self.x[i], "y": self.y[i]}


def _loss_fn(model: nn.Module, batch: dict) -> torch.Tensor:
    return ((model(batch["x"]) - batch["y"]) ** 2).sum()


def _make_data(n: int):
    g = torch.Generator().manual_seed(SEED)
    return DictDataset(
        torch.randn(n, IN_DIM, generator=g),
        torch.randn(n, OUT_DIM, generator=g),
    )


def _args(out_dir, **overrides) -> AttributionArguments:
    kwargs = {
        "output_dir": str(out_dir),
        "per_device_train_batch_size": BATCH,
        "use_cpu": True,
        "dataloader_pin_memory": False,
        "optim": "sgd",
        "learning_rate": LR,
        "lr_scheduler_type": "constant",
        "max_grad_norm": None,
        "weight_decay": 0.0,
    }
    kwargs.update(overrides)
    return AttributionArguments(**kwargs)


def _batch_weight_grads(grad) -> dict[str, torch.Tensor]:
    """Sum of the block's per-sample weight gradients, in parameter shape."""
    shapes = {"fc1": (HID_DIM, IN_DIM), "fc2": (OUT_DIM, HID_DIM)}
    return {
        f"mlp.{name}.weight": ops.materialize(grad.data[f"mlp.{name}"], "nn.Linear")
        .sum(0)
        .reshape(dims)
        for name, dims in shapes.items()
    }


class TestAccumulationTrajectory:
    def test_updates_fire_per_window_including_ragged_tail(self, tmp_path):
        """n=6, batch=2, accum=2 -> 3 micro-batches: one full window (micro
        0-1) and one ragged window (micro 2).  Each update must equal SGD on
        the sum of that window's captured gradients (records carry the 1/N
        loss scaling), reconstructed from the streamer's own yields.
        """
        torch.manual_seed(SEED)
        model = MLP()
        streamer = GradientStreamer(
            model,
            _make_data(6),
            _args(tmp_path, gradient_accumulation_steps=2),
            batch_size=BATCH,
            enable_update=True,
            loss_fn=_loss_fn,
        )
        window_grad: dict[str, torch.Tensor] = {}
        params_before = {n: p.detach().clone() for n, p in model.named_parameters()}
        n_updates = 0
        with streamer:
            for i, (_step, grad, _hashes) in enumerate(streamer):
                for name, g in _batch_weight_grads(grad).items():
                    window_grad[name] = window_grad.get(name, 0) + g
                boundary = (i + 1) % 2 == 0 or i == 2  # accum window or tail
                params_now = dict(model.named_parameters())
                if boundary:
                    n_updates += 1
                    for name, before in params_before.items():
                        expected = before - LR * window_grad[name]
                        assert torch.allclose(
                            params_now[name],
                            expected,
                            atol=ATOL,
                        ), name
                    params_before = {
                        n: p.detach().clone() for n, p in params_now.items()
                    }
                    window_grad = {}
                else:
                    # Mid-window: no update happened.
                    for name, before in params_before.items():
                        assert torch.equal(params_now[name], before), name
        assert n_updates == 2  # ceil(3 / 2): one full window + the ragged tail

    def test_scheduler_advances_per_update(self, tmp_path):
        """Linear schedule sized in updates: micro-batches of one window
        share an LR; the next window sees the decayed one.
        """
        torch.manual_seed(SEED)
        model = MLP()
        streamer = GradientStreamer(
            model,
            _make_data(6),  # 3 micro-batches -> 2 updates
            _args(
                tmp_path,
                gradient_accumulation_steps=2,
                lr_scheduler_type="linear",
            ),
            batch_size=BATCH,
            enable_update=True,
            loss_fn=_loss_fn,
        )
        with streamer:
            for _ in streamer:
                pass
        lrs = streamer.learning_rates
        assert lrs[0] == lrs[1] == pytest.approx(LR)  # window 1, before decay
        assert lrs[2] == pytest.approx(LR / 2)  # linear decay over 2 updates


class TestFrozenProbeIgnoresAccumulation:
    def test_records_identical_to_no_accumulation(self, tmp_path):
        def run(accum: int):
            torch.manual_seed(SEED)
            model = MLP()
            streamer = GradientStreamer(
                model,
                _make_data(4),
                _args(tmp_path / str(accum), gradient_accumulation_steps=accum),
                batch_size=BATCH,
                enable_update=False,
                loss_fn=_loss_fn,
            )
            out = []
            with streamer:
                for _step, grad, _hashes in streamer:
                    out.append(_batch_weight_grads(grad))
            return out

        for g1, g4 in zip(run(1), run(4), strict=True):
            for name in g1:
                assert torch.equal(g1[name], g4[name]), name


def _run_trainer_parity(
    tmp_path,
    *,
    accum: int,
    batch: int,
    ckpt_kwargs: dict | None = None,
) -> float:
    """Run Trainer and streamer with identical settings; return the max
    absolute final-parameter difference.

    The Trainer-side model is wrapped with an explicit forward signature so
    Trainer takes its classic ``loss / N`` accumulation path -- the
    convention the streamer implements (the loss-kwargs path instead
    normalises by window token count, which is Trainer-version-specific and
    not replicated).  A single update window over the whole dataset keeps
    the comparison micro-batch-order-independent.
    """
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("accelerate")

    set_seed = transformers.set_seed
    cfg = transformers.GPT2Config(
        vocab_size=64,
        n_positions=16,
        n_embd=32,
        n_layer=2,
        n_head=2,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )

    class TokenDataset(Dataset):
        def __init__(self, n=4, seq_len=8):
            torch.manual_seed(0)
            self.data = torch.randint(0, 64, (n, seq_len))

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            ids = self.data[idx]
            return {"input_ids": ids, "labels": ids.clone()}

    class NoLossKwargs(nn.Module):
        """Explicit signature -> Trainer uses classic loss/N scaling."""

        def __init__(self, inner: nn.Module) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, input_ids, labels=None):
            return self.inner(input_ids=input_ids, labels=labels)

    dataset = TokenDataset()
    lr, seed = 0.05, 42

    set_seed(seed)
    model_t = transformers.GPT2LMHeadModel(cfg)
    if ckpt_kwargs is not None:
        # Enable on the inner model directly: the plain wrapper does not
        # forward gradient_checkpointing_enable, and checkpointing is a
        # model-side property once set.
        model_t.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=ckpt_kwargs,
        )
    targs = transformers.TrainingArguments(
        output_dir=str(tmp_path / "trainer"),
        num_train_epochs=1,
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=accum,
        learning_rate=lr,
        lr_scheduler_type="constant",
        # 0 disables clipping on every transformers version: v4 gates on
        # ``is not None and > 0``, v5 on a bare ``> 0`` that crashes on None.
        max_grad_norm=0,
        optim="sgd",
        use_cpu=True,
        report_to="none",
        seed=seed,
        save_strategy="no",  # plain-wrapper save chokes on tied weights
    )
    transformers.Trainer(
        model=NoLossKwargs(model_t),
        args=targs,
        train_dataset=dataset,
    ).train()

    set_seed(seed)
    model_s = transformers.GPT2LMHeadModel(cfg)
    sargs = _args(
        tmp_path / "streamer",
        learning_rate=lr,
        gradient_accumulation_steps=accum,
        seed=seed,
        gradient_checkpointing=ckpt_kwargs is not None,
        gradient_checkpointing_kwargs=ckpt_kwargs,
    )
    streamer = GradientStreamer(
        model_s,
        dataset,
        sargs,
        batch_size=batch,
        enable_update=True,
    )
    with streamer:
        for _ in streamer:
            pass

    sd_t = model_t.state_dict()
    sd_s = model_s.state_dict()
    return max((sd_t[k] - sd_s[k]).abs().max().item() for k in sd_t)


class TestTrainerParityUnderAccumulation:
    def test_final_params_match_trainer(self, tmp_path):
        """Streamer with accum=2 lands on the same parameters as a real
        Trainer run (n=4, batch=2, accum=2: one window over the dataset).
        """
        max_diff = _run_trainer_parity(tmp_path, accum=2, batch=BATCH)
        assert max_diff < 1e-6, (
            f"streamer diverged from Trainer under accumulation: "
            f"max diff {max_diff:.2e}"
        )


class TestTrainerParityUnderCheckpointing:
    @pytest.mark.parametrize("use_reentrant", [True, False])
    @pytest.mark.parametrize("accum", [1, 2])
    def test_final_params_match_trainer(self, tmp_path, use_reentrant, accum):
        """Gradient checkpointing (either variant), alone and combined with
        accumulation, must not perturb the trajectory: the streamer's capture
        hooks ride the recomputation without touching the update.
        """
        max_diff = _run_trainer_parity(
            tmp_path,
            accum=accum,
            batch=4 // accum,  # keep a single window over the dataset
            ckpt_kwargs={"use_reentrant": use_reentrant},
        )
        assert max_diff < 1e-6, (
            f"streamer diverged from Trainer under checkpointing "
            f"(use_reentrant={use_reentrant}, accum={accum}): "
            f"max diff {max_diff:.2e}"
        )

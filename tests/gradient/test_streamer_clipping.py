"""Gradient-clipping correctness for :class:`GradientStreamer` trajectories.

``max_grad_norm`` clips by the **global** gradient norm, exactly as HF
``Trainer`` does (via ``accelerate.clip_grad_norm_``).  Three properties are
pinned, matching the wrapping regimes:

1. **Captured gradients are pre-clip** and still reconstruct ``param.grad``:
   clipping happens after step completion, so the per-sample records sum to
   the unclipped gradient, and ``param.grad`` equals that sum scaled by the
   clip coefficient.
2. **Trainer parity**: a streamer trajectory with clipping active lands on
   the same parameters as a real ``transformers.Trainer`` run with the same
   arguments.
3. **Distributed**: under FSDP the clip must use the *global* norm across
   shards (regression: the rank-local ``nn.utils`` clip scaled each rank
   differently); under DDP the vanilla clip on replicated gradients must
   remain exact and identical across ranks.  Both are checked end-to-end
   against a single-process reference trajectory.
"""

from __future__ import annotations

import os
import pathlib
import socket
import tempfile

import pytest
import torch
import torch.multiprocessing as mp
from torch import nn
from torch.utils.data import Dataset, DistributedSampler

from dattri_llm.attribution.arguments import AttributionArguments
from dattri_llm.gradient import ops
from dattri_llm.gradient.streaming import GradientStreamer

SEED = 0
IN_DIM, HID_DIM, OUT_DIM = 4, 8, 3
N_SAMPLES = 8
BATCH = 2
LR = 0.1
MAX_NORM = 0.05  # far below the raw gradient norm, so clipping always engages
ATOL = 1e-5


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Sequential()
        self.mlp.add_module("fc1", nn.Linear(IN_DIM, HID_DIM, bias=False))
        self.mlp.add_module("act", nn.ReLU())
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


def _make_data(n: int = N_SAMPLES):
    g = torch.Generator().manual_seed(SEED)
    return DictDataset(
        torch.randn(n, IN_DIM, generator=g),
        torch.randn(n, OUT_DIM, generator=g),
    )


def _sgd_args(out_dir, **overrides) -> AttributionArguments:
    kwargs = {
        "output_dir": str(out_dir),
        "per_device_train_batch_size": BATCH,
        "use_cpu": True,
        "dataloader_pin_memory": False,
        "optim": "sgd",
        "learning_rate": LR,
        "lr_scheduler_type": "constant",
        "max_grad_norm": MAX_NORM,
        "weight_decay": 0.0,
    }
    kwargs.update(overrides)
    return AttributionArguments(**kwargs)


def _clip_coef(flat_norm: torch.Tensor) -> torch.Tensor:
    """nn.utils.clip_grad_norm_'s coefficient (clamped at 1)."""
    return (MAX_NORM / (flat_norm + 1e-6)).clamp(max=1.0)


# --------------------------------------------------------------------------- #
# 1. Captured gradients are pre-clip and reconstruct param.grad                #
# --------------------------------------------------------------------------- #


class TestCaptureVsClippedGrad:
    def test_records_sum_to_preclip_and_param_grad_is_scaled(self, tmp_path):
        torch.manual_seed(SEED)
        model = MLP()
        streamer = GradientStreamer(
            model,
            _make_data(),
            _sgd_args(tmp_path),
            batch_size=BATCH,
            enable_update=True,
            loss_fn=_loss_fn,
        )
        n_clipped_steps = 0
        with streamer:
            for _step, grad, _hashes in streamer:
                # Sum of captured per-sample gradients == the *unclipped*
                # batch gradient (capture happens before the clip).
                shapes = {"fc1": (HID_DIM, IN_DIM), "fc2": (OUT_DIM, HID_DIM)}
                pre_clip = {
                    f"mlp.{name}.weight": ops.materialize(
                        grad.data[f"mlp.{name}"],
                        "nn.Linear",
                    )
                    .sum(0)
                    .reshape(dims)
                    for name, dims in shapes.items()
                }
                total_norm = torch.linalg.vector_norm(
                    torch.cat([v.reshape(-1) for v in pre_clip.values()]),
                )
                coef = _clip_coef(total_norm)
                if coef < 1:
                    n_clipped_steps += 1
                # param.grad (read after the optimizer step, before the next
                # zero_grad) is exactly the clipped version of that sum.
                params = dict(model.named_parameters())
                for pname, expected_pre in pre_clip.items():
                    actual = params[pname].grad
                    assert torch.allclose(actual, coef * expected_pre, atol=ATOL), (
                        f"{pname}: max diff "
                        f"{(actual - coef * expected_pre).abs().max():.2e}"
                    )
        assert n_clipped_steps > 0, "clipping never engaged; test is vacuous"


# --------------------------------------------------------------------------- #
# 2. Trainer parity under clipping (single process)                            #
# --------------------------------------------------------------------------- #


class TestTrainerParityUnderClipping:
    def test_final_params_match_trainer(self, tmp_path):
        transformers = pytest.importorskip("transformers")
        pytest.importorskip("accelerate")
        from torch.utils.data import Dataset as TorchDataset

        set_seed = transformers.set_seed
        # Dropout off: the two runs consume RNG differently before the
        # forward, so active dropout would diverge them for unrelated reasons.
        cfg = transformers.GPT2Config(
            vocab_size=64,
            n_positions=16,
            n_embd=32,
            n_layer=1,
            n_head=2,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
        )

        class TokenDataset(TorchDataset):
            def __init__(self, n=4, seq_len=8):
                torch.manual_seed(0)
                self.data = torch.randint(0, 64, (n, seq_len))

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                ids = self.data[idx]
                return {"input_ids": ids, "labels": ids.clone()}

        dataset = TokenDataset()
        lr, max_norm, seed = 0.05, 0.01, 42

        # --- Trainer run: one full-batch step (order-independent), clipped.
        set_seed(seed)
        model_t = transformers.GPT2LMHeadModel(cfg)
        targs = transformers.TrainingArguments(
            output_dir=str(tmp_path / "trainer"),
            num_train_epochs=1,
            per_device_train_batch_size=len(dataset),
            learning_rate=lr,
            lr_scheduler_type="constant",
            max_grad_norm=max_norm,
            optim="sgd",
            use_cpu=True,
            report_to="none",
            seed=seed,
        )
        transformers.Trainer(
            model=model_t,
            args=targs,
            train_dataset=dataset,
        ).train()

        # --- Streamer run: identical arguments, identically-seeded init.
        set_seed(seed)
        model_s = transformers.GPT2LMHeadModel(cfg)
        sargs = _sgd_args(
            tmp_path / "streamer",
            per_device_train_batch_size=len(dataset),
            learning_rate=lr,
            max_grad_norm=max_norm,
            seed=seed,
        )
        streamer = GradientStreamer(
            model_s,
            dataset,
            sargs,
            batch_size=len(dataset),
            enable_update=True,
        )
        with streamer:
            for _ in streamer:
                pass

        sd_t = model_t.state_dict()
        sd_s = model_s.state_dict()
        max_diff = max((sd_t[k] - sd_s[k]).abs().max().item() for k in sd_t)
        assert max_diff < 1e-6, (
            f"streamer diverged from Trainer: max diff {max_diff:.2e}"
        )


# --------------------------------------------------------------------------- #
# 3. FSDP / DDP trajectories vs a single-process reference                     #
# --------------------------------------------------------------------------- #


def _can_bind_localhost() -> bool:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind(("127.0.0.1", 0))
    except OSError:
        return False
    finally:
        probe.close()
    return True


def _reference_final_params(world_size: int, data_seed: int) -> dict:
    """Single-process trajectory: averaged rank grads -> global clip -> SGD."""
    torch.manual_seed(SEED)
    ref = MLP()
    ds = _make_data()
    # Reconstruct each rank's deterministic shard order (epoch 0).
    rank_batches = []
    for r in range(world_size):
        idx = list(
            DistributedSampler(
                ds,
                num_replicas=world_size,
                rank=r,
                shuffle=True,
                seed=data_seed,
            ),
        )
        rank_batches.append([idx[i : i + BATCH] for i in range(0, len(idx), BATCH)])
    n_steps = len(rank_batches[0])
    for t in range(n_steps):
        grads = None
        for r in range(world_size):
            sel = rank_batches[r][t]
            batch = {"x": ds.x[sel], "y": ds.y[sel]}
            ref.zero_grad()
            _loss_fn(ref, batch).backward()
            g = {n: p.grad.detach().clone() for n, p in ref.named_parameters()}
            grads = g if grads is None else {n: grads[n] + g[n] for n in grads}
        grads = {n: v / world_size for n, v in grads.items()}  # DDP/FSDP average
        total = torch.linalg.vector_norm(
            torch.cat([v.reshape(-1) for v in grads.values()]),
        )
        coef = _clip_coef(total)
        assert coef < 1  # clipping must engage for the test to mean anything
        with torch.no_grad():
            for n, param in ref.named_parameters():
                param.sub_(LR * coef * grads[n])
    return {n: p.detach().clone() for n, p in ref.named_parameters()}


def _dist_worker(rank, world_size, mode, result_queue, rendezvous_path, out_dir):
    import torch.distributed as dist

    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous_path}",
        rank=rank,
        world_size=world_size,
    )
    try:
        torch.manual_seed(SEED)
        model = MLP()
        args = _sgd_args(out_dir, fsdp="full_shard" if mode == "fsdp" else "")
        streamer = GradientStreamer(
            model,
            _make_data(),
            args,
            batch_size=BATCH,
            enable_update=True,
            loss_fn=_loss_fn,
        )
        with streamer:
            for _ in streamer:
                pass

        # Gather the final full parameters.
        if mode == "fsdp":
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            with FSDP.summon_full_params(streamer._fwd_model):
                final = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            final = {k: v.detach().clone() for k, v in model.state_dict().items()}
            # DDP replicas must remain identical across ranks.
            identical = True
            for v in final.values():
                lo, hi = v.clone(), v.clone()
                dist.all_reduce(lo, op=dist.ReduceOp.MIN)
                dist.all_reduce(hi, op=dist.ReduceOp.MAX)
                identical = identical and torch.allclose(lo, hi, atol=1e-7)

        if rank == 0:
            ref = _reference_final_params(world_size, data_seed=args.data_seed)
            ok = True
            report = [f"mode={mode}"]
            if mode == "ddp":
                ok = identical
                report.append(f"  replicas identical: {identical}")
            for name, expected in ref.items():
                actual = final[name]
                close = torch.allclose(actual, expected, atol=1e-5, rtol=1e-4)
                ok = ok and close
                report.append(
                    f"  {name}: match={close} "
                    f"maxdiff={(actual - expected).abs().max().item():.2e}",
                )
            result_queue.put((ok, "\n".join(report)))
    except Exception:  # noqa: BLE001 - surface any worker failure to the test
        import traceback

        if rank == 0:
            result_queue.put((False, "WORKER EXC:\n" + traceback.format_exc()))
    finally:
        dist.destroy_process_group()


class TestDistributedClipping:
    @pytest.mark.parametrize("mode", ["fsdp", "ddp"])
    def test_trajectory_matches_globally_clipped_reference(self, mode, tmp_path):
        if not _can_bind_localhost():
            pytest.skip("local socket binds are not permitted in this environment")

        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()
        fd, rendezvous_path = tempfile.mkstemp()
        os.close(fd)
        try:
            mp.spawn(
                _dist_worker,
                args=(2, mode, result_queue, rendezvous_path, str(tmp_path)),
                nprocs=2,
                join=True,
            )
        finally:
            if pathlib.Path(rendezvous_path).exists():
                pathlib.Path(rendezvous_path).unlink()

        assert not result_queue.empty(), "rank-0 worker did not report a result"
        ok, report = result_queue.get()
        assert ok, (
            f"{mode.upper()} trajectory disagrees with the globally-clipped "
            f"single-process reference:\n{report}"
        )

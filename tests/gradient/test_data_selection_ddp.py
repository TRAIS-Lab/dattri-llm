"""DDP correctness tests for :class:`DataSelectionCallback`.

Under ``DistributedDataParallel`` every rank holds the same full, replicated
``param.grad = (1/world_size) * sum_r G_r`` after the allreduce.  A rank-local
subtraction of the dropped samples' contribution would therefore be wrong twice
over: it misses the ``1/world_size`` scaling, and -- since each rank drops
different samples -- it desynchronises the replicas.  Removal must be
collective (see ``DataSelectionCallback._remove_contributions_ddp``).

These tests launch two ``gloo`` CPU workers via ``torch.multiprocessing.spawn``
(no ``torchrun`` needed) and check, end-to-end, that the corrected replicated
gradients (a) equal ``(1/world_size) * sum_r G_r^kept`` -- the averaged
gradient of the surviving samples -- and (b) remain bit-identical across ranks,
for three regimes:

* ``none``  -- nothing dropped (callback must be a no-op, no deadlock).
* ``half``  -- bottom 50 % of every rank dropped (symmetric).
* ``hard0`` -- drop negatively-influential samples (counts differ per rank,
  exercising the lock-step collective when a rank drops nothing).
* ``half_renorm`` -- like ``half`` but with a mean-reduced loss and
  ``renormalize=True``: each rank's corrected gradient must equal the mean
  loss over its kept samples only (the rank-local ``B/(B-k)`` rescale
  travelling through the packed collective).
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

from dattri_llm.gradient.callbacks import DataSelectionCallback
from dattri_llm.gradient.hooks import REGISTER_ALL, HookManager, HookManagerConfig

SEED = 0
VOCAB = 32
EMBED = 8
HIDDEN = 16
OUT = 4
SEQ = 5
BATCH = 4
ATOL = 1e-4
RTOL = 1e-3


class EmbeddingMLP(nn.Module):
    """Embedding -> two-layer MLP with biases (mirrors the FSDP test model)."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(VOCAB, EMBED)
        self.mlp = nn.Sequential(
            nn.Linear(EMBED, HIDDEN, bias=True),
            nn.ReLU(),
            nn.Linear(HIDDEN, OUT, bias=True),
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.embedding(token_ids))


def _can_bind_localhost() -> bool:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind(("127.0.0.1", 0))
    except OSError:
        return False
    finally:
        probe.close()
    return True


def _hooked_params(model: EmbeddingMLP):
    """Yield ``(name, param)`` for the hooked layers' weights and biases."""
    yield "embedding.weight", model.embedding.weight
    for i, m in enumerate(model.mlp):
        if isinstance(m, nn.Linear):
            yield f"mlp.{i}.weight", m.weight
            yield f"mlp.{i}.bias", m.bias


def _callback_kwargs(mode: str) -> dict:
    if mode == "none":
        return {"threshold_mode": "bottom_fraction", "threshold": 0.0}
    if mode == "half":
        return {"threshold_mode": "bottom_fraction", "threshold": 0.5}
    if mode == "hard0":
        return {"threshold_mode": "hard", "threshold": 0.0}
    if mode == "half_renorm":
        return {
            "threshold_mode": "bottom_fraction",
            "threshold": 0.5,
            "renormalize": True,
        }
    raise ValueError(mode)


def _ddp_ds_worker(rank, world_size, mode, result_queue, rendezvous_path):
    import torch.distributed as dist

    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous_path}",
        rank=rank,
        world_size=world_size,
    )
    try:
        # Same initial weights on every rank; distinct per-rank batch so the
        # drop sets (and therefore the required corrections) differ per rank.
        torch.manual_seed(SEED)
        model = EmbeddingMLP()
        init_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        gen = torch.Generator().manual_seed(100 + rank)
        token_ids = torch.randint(0, VOCAB, (BATCH, SEQ), generator=gen)

        # HookManager on the unwrapped model (hooks survive DDP wrapping).
        collector = HookManager(
            model,
            config=HookManagerConfig(linear_io=REGISTER_ALL),
        )
        ddp_model = nn.parallel.DistributedDataParallel(model)
        ds_cb = DataSelectionCallback(
            model=ddp_model,
            score_mode="ghost",
            target="batch",
            **_callback_kwargs(mode),
        )
        collector.add_callback(ds_cb)

        # ``sum`` reduction keeps each sample's gradient contribution
        # independent of batch size, so the surviving-sample reference below
        # is exact (DDP still averages the summed gradients across ranks).
        # The renormalize regime uses ``mean`` instead: the ``B/(B-k)``
        # rescale is exactly what makes the mean-loss reference exact.
        use_mean = mode == "half_renorm"
        ddp_model.zero_grad()
        with collector.collect():
            out = ddp_model(token_ids)
            (out.mean() if use_mean else out.sum()).backward()

        dropped = sorted(ds_cb.last_dropped)

        # --- Corrected replicated gradients must be identical across ranks. ---
        actual = {
            name: param.grad.detach().reshape(-1).float().clone()
            for name, param in _hooked_params(model)
        }
        identical = True
        for t in actual.values():
            t_min, t_max = t.clone(), t.clone()
            dist.all_reduce(t_min, op=dist.ReduceOp.MIN)
            dist.all_reduce(t_max, op=dist.ReduceOp.MAX)
            identical = identical and torch.allclose(t_min, t_max, atol=1e-6)

        # --- Expected: (1/W) * sum_r gradient over rank r's surviving samples. ---
        keep = [i for i in range(BATCH) if i not in set(dropped)]
        ref = EmbeddingMLP()
        ref.load_state_dict(init_state)
        ref.zero_grad()
        if keep:
            ref_out = ref(token_ids[keep])
            (ref_out.mean() if use_mean else ref_out.sum()).backward()
        ref_full = {}
        for name, param in _hooked_params(ref):
            g = param.grad if param.grad is not None else torch.zeros_like(param)
            t = g.reshape(-1).float().clone()
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            ref_full[name] = t / world_size

        if rank == 0:
            ok = identical
            report = [
                f"mode={mode} dropped(rank0)={dropped} ranks_identical={identical}",
            ]
            for name, a in actual.items():
                e = ref_full[name]
                close = torch.allclose(a, e, atol=ATOL, rtol=RTOL)
                ok = ok and close
                report.append(
                    f"  {name}: match={close} maxdiff={(a - e).abs().max().item():.2e}",
                )
            result_queue.put((ok, "\n".join(report)))
    except Exception:  # noqa: BLE001 - surface any worker failure to the test
        import traceback

        if rank == 0:
            result_queue.put((False, "WORKER EXC:\n" + traceback.format_exc()))
    finally:
        dist.destroy_process_group()


class TestDataSelectionDDP:
    """End-to-end DDP gradient-correction checks for DataSelectionCallback."""

    @pytest.mark.parametrize("mode", ["none", "half", "hard0", "half_renorm"])
    def test_corrected_grad_matches_surviving_reference(self, mode):
        if not _can_bind_localhost():
            pytest.skip("local socket binds are not permitted in this environment")

        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()
        fd, rendezvous_path = tempfile.mkstemp()
        os.close(fd)
        try:
            mp.spawn(
                _ddp_ds_worker,
                args=(2, mode, result_queue, rendezvous_path),
                nprocs=2,
                join=True,
            )
        finally:
            if pathlib.Path(rendezvous_path).exists():
                pathlib.Path(rendezvous_path).unlink()

        assert not result_queue.empty(), "rank-0 worker did not report a result"
        ok, report = result_queue.get()
        assert ok, (
            "DDP-corrected gradients disagree with the surviving-sample "
            f"reference, or diverged across ranks:\n{report}"
        )

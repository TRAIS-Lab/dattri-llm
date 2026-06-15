"""FSDP correctness tests for :class:`DataSelectionCallback`.

Under ``FullyShardedDataParallel`` (``use_orig_params=True``) every original
``param.grad`` is a 1-D *shard* of the flattened, unsharded parameter holding
FSDP's *averaged* gradient.  A dropped sample lives on a single rank, but the
weight elements it touches may be owned by another rank's shard, so removing its
contribution requires a cross-rank reduction (see
``DataSelectionCallback._remove_contributions_fsdp``).

These tests launch two ``gloo`` CPU workers via ``torch.multiprocessing.spawn``
(no ``torchrun`` needed) and check, end-to-end, that the corrected sharded
gradients reconstruct to ``(1/world_size) · Σ_r G_r^kept`` — i.e. exactly the
averaged gradient of the surviving samples — across three regimes:

* ``none``  — nothing dropped (callback must be a no-op, no deadlock).
* ``half``  — bottom 50 % of every rank dropped (symmetric).
* ``hard0`` — drop negatively-influential samples (counts differ per rank,
  exercising the lock-step collective when a rank drops nothing).
"""

from __future__ import annotations

import math
import os
import socket
import tempfile

import pytest
import torch
import torch.multiprocessing as mp
import torch.nn as nn

from dattri_llm.gradient.callbacks import DataSelectionCallback
from dattri_llm.gradient.hooks import HookManager, HookManagerConfig

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
    """Embedding → two-layer MLP with biases.

    The embedding makes every MLP input require grad (matching real LLM
    training), and the biases exercise 1-D shard mapping alongside the 2-D
    weight shards.
    """

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


def _hooked_linear_grads(model: EmbeddingMLP):
    """Yield ``(name, param)`` for the hooked MLP linears' weight & bias."""
    for i, m in enumerate(model.mlp):
        if isinstance(m, nn.Linear):
            yield f"mlp.{i}.weight", m.weight
            yield f"mlp.{i}.bias", m.bias


def _callback_kwargs(mode: str) -> dict:
    if mode == "none":
        return dict(threshold_mode="bottom_fraction", threshold=0.0)
    if mode == "half":
        return dict(threshold_mode="bottom_fraction", threshold=0.5)
    if mode == "hard0":
        return dict(threshold_mode="hard", threshold=0.0)
    raise ValueError(mode)


def _fsdp_ds_worker(rank, world_size, mode, result_queue, rendezvous_path):
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import ShardingStrategy

    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous_path}", rank=rank, world_size=world_size
    )
    try:
        # Same initial weights on every rank; distinct per-rank batch so that a
        # sample on one rank genuinely affects shards owned by the other.
        torch.manual_seed(SEED)
        model = EmbeddingMLP()
        init_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        gen = torch.Generator().manual_seed(100 + rank)
        token_ids = torch.randint(0, VOCAB, (BATCH, SEQ), generator=gen)

        # HookManager on the unwrapped model (hooks survive FSDP wrapping).
        collector = HookManager(
            model, config=HookManagerConfig(hook_types={"mlp_io": None})
        )
        fsdp_model = FSDP(
            model,
            device_id=torch.device("cpu"),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            use_orig_params=True,
        )
        # DataSelectionCallback needs the *wrapped* module to discover the shard
        # layout; attach it after wrapping via the manager's add_callback().
        ds_cb = DataSelectionCallback(
            model=fsdp_model, score_mode="ghost", target="batch", **_callback_kwargs(mode)
        )
        collector.add_callback(ds_cb)

        # ``sum`` reduction keeps each sample's gradient contribution
        # independent of batch size, so the surviving-sample reference below is
        # exact.
        fsdp_model.zero_grad()
        with collector.collect():
            fsdp_model(token_ids).sum().backward()

        dropped = sorted(ds_cb.last_dropped)

        # --- Reconstruct the actual corrected *full* gradient from shards. ---
        shard_map = ds_cb._fsdp_shard_map
        actual_full = {}
        for name, param in _hooked_linear_grads(model):
            spec = shard_map[id(param)]
            buf = torch.zeros(math.prod(spec.orig_shape), dtype=torch.float32)
            if spec.in_shard and param.grad is not None:
                buf[spec.start : spec.end + 1] = param.grad.reshape(-1).float()
            dist.all_reduce(buf, op=dist.ReduceOp.SUM)  # disjoint shards → full
            actual_full[name] = buf

        # --- Expected: (1/W) · Σ_r gradient over rank r's surviving samples. ---
        keep = [i for i in range(BATCH) if i not in set(dropped)]
        ref = EmbeddingMLP()
        ref.load_state_dict(init_state)
        ref.zero_grad()
        if keep:
            ref(token_ids[keep]).sum().backward()
        ref_full = {}
        for name, param in _hooked_linear_grads(ref):
            g = param.grad if param.grad is not None else torch.zeros_like(param)
            t = g.reshape(-1).float().clone()
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            ref_full[name] = t / world_size

        if rank == 0:
            ok = True
            report = [f"mode={mode} dropped(rank0)={dropped}"]
            for name in actual_full:
                a, e = actual_full[name], ref_full[name]
                close = torch.allclose(a, e, atol=ATOL, rtol=RTOL)
                ok = ok and close
                report.append(
                    f"  {name}: match={close} maxdiff={(a - e).abs().max().item():.2e}"
                )
            result_queue.put((ok, "\n".join(report)))
    except Exception:  # pragma: no cover - surface worker failures to the test
        import traceback

        if rank == 0:
            result_queue.put((False, "WORKER EXC:\n" + traceback.format_exc()))
    finally:
        dist.destroy_process_group()


class TestDataSelectionFSDP:
    """End-to-end FSDP gradient-correction checks for DataSelectionCallback."""

    @pytest.mark.parametrize("mode", ["none", "half", "hard0"])
    def test_corrected_grad_matches_surviving_reference(self, mode):
        if not _can_bind_localhost():
            pytest.skip("local socket binds are not permitted in this environment")

        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()
        fd, rendezvous_path = tempfile.mkstemp()
        os.close(fd)
        try:
            mp.spawn(
                _fsdp_ds_worker,
                args=(2, mode, result_queue, rendezvous_path),
                nprocs=2,
                join=True,
            )
        finally:
            if os.path.exists(rendezvous_path):
                os.unlink(rendezvous_path)

        assert not result_queue.empty(), "rank-0 worker did not report a result"
        ok, report = result_queue.get()
        assert ok, (
            "FSDP-corrected gradients disagree with the surviving-sample "
            f"reference:\n{report}"
        )

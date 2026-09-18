"""Live scoring under DDP/FSDP: the pieces the sharded live path rests on.

Two ``gloo`` CPU workers are spawned per test (as in
``test_streamer_clipping.py``).  Pinned:

1. **Accumulator reduction** -- after ``all_reduce`` the K-FAC covariance
   and empirical-Fisher accumulators equal a single-process accumulation of
   every rank's data, so a fit over per-rank shards is a fit over the whole
   set.
2. **Shared wrapper, unsharded test set** -- a test streamer riding the train
   streamer's hooks *and* ``forward_model`` streams every query on every
   rank (``shard=False``), the train streamer's shards partition the train
   set, and both yield the same per-sample gradients as a single process.
   Sharing the hooks under distributed execution without the wrapper is
   refused.
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
from torch.utils.data import Dataset

from dattri_llm.attribution.arguments import AttributionArguments
from dattri_llm.gradient import ops
from dattri_llm.gradient.streaming import GradientStreamer
from dattri_llm.utils.hashing import hash_sample

SEED = 0
IN_DIM, HID_DIM, OUT_DIM = 4, 8, 3
N_TRAIN, N_TEST = 8, 3
BATCH = 2
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
    # Keyword inputs: the captured model inputs are then the dataset row, so
    # the streamer's row hashes equal ``hash_sample(ds[i])``.
    return ((model(**batch) - batch["y"]) ** 2).sum()


def _make_data(n: int, seed: int) -> DictDataset:
    g = torch.Generator().manual_seed(seed)
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
    }
    kwargs.update(overrides)
    return AttributionArguments(**kwargs)


def _can_bind_localhost() -> bool:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind(("127.0.0.1", 0))
    except OSError:
        return False
    finally:
        probe.close()
    return True


def _spawn(worker, *worker_args) -> tuple[bool, str]:
    """Run *worker* on two gloo ranks; return rank 0's ``(ok, report)``."""
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    fd, rendezvous_path = tempfile.mkstemp()
    os.close(fd)
    try:
        mp.spawn(
            worker,
            args=(2, result_queue, rendezvous_path, *worker_args),
            nprocs=2,
            join=True,
        )
    finally:
        if pathlib.Path(rendezvous_path).exists():
            pathlib.Path(rendezvous_path).unlink()
    assert not result_queue.empty(), "rank-0 worker did not report a result"
    return result_queue.get()


def _init_group(rank: int, world_size: int, rendezvous_path: str) -> None:
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


def _report(rank: int, result_queue, fn) -> None:
    """Run *fn* on every rank; rank 0 reports its ``(ok, report)``."""
    import torch.distributed as dist

    try:
        ok, report = fn()
        if rank == 0:
            result_queue.put((ok, report))
    except Exception:  # noqa: BLE001 - surface any worker failure to the test
        import traceback

        if rank == 0:
            result_queue.put((False, "WORKER EXC:\n" + traceback.format_exc()))
    finally:
        dist.destroy_process_group()


# --------------------------------------------------------------------------- #
# 1. Accumulator reduction                                                     #
# --------------------------------------------------------------------------- #


def _factors(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(n, 5, IN_DIM, generator=g)  # (B, T, d_in)
    grad = torch.randn(n, 5, OUT_DIM, generator=g)  # (B, T, d_out)
    return a, grad


def _reduce_worker(rank, world_size, result_queue, rendezvous_path):
    _init_group(rank, world_size, rendezvous_path)

    def body():
        # Each rank accumulates its own (different-size) shard ...
        a, g = _factors(2 + rank, seed=rank)
        kron = ops.LayerKroneckerAccumulator()
        kron.update(a, g, "linear", include_bias=False)
        fisher = ops.LayerFisherAccumulator()
        fisher.update(a, g, "linear", include_bias=False)
        kron.all_reduce()
        fisher.all_reduce()
        # ... and the reduced result is the single-process fit of both shards.
        ref_kron = ops.LayerKroneckerAccumulator()
        ref_fisher = ops.LayerFisherAccumulator()
        for r in range(world_size):
            a_r, g_r = _factors(2 + r, seed=r)
            ref_kron.update(a_r, g_r, "linear", include_bias=False)
            ref_fisher.update(a_r, g_r, "linear", include_bias=False)
        A, G = kron.result()
        A_ref, G_ref = ref_kron.result()
        ok = (
            torch.allclose(A, A_ref, atol=ATOL)
            and torch.allclose(G, G_ref, atol=ATOL)
            and torch.allclose(fisher.result(), ref_fisher.result(), atol=ATOL)
        )
        return ok, f"A {A}\nA_ref {A_ref}"

    _report(rank, result_queue, body)


class TestAccumulatorAllReduce:
    def test_reduced_fit_equals_single_process_fit(self):
        if not _can_bind_localhost():
            pytest.skip("local socket binds are not permitted in this environment")
        ok, report = _spawn(_reduce_worker)
        assert ok, report

    def test_single_process_is_a_no_op(self):
        a, g = _factors(3, seed=1)
        acc = ops.LayerKroneckerAccumulator()
        acc.update(a, g, "linear", include_bias=False)
        before = tuple(t.clone() for t in acc.result())
        acc.all_reduce()
        after = acc.result()
        assert torch.equal(before[0], after[0])
        assert torch.equal(before[1], after[1])


# --------------------------------------------------------------------------- #
# 2. Shared wrapper, unsharded test set                                        #
# --------------------------------------------------------------------------- #


def _reference_gradients(model_seed: int) -> dict[str, dict[str, torch.Tensor]]:
    """Single-process per-sample gradients: ``{hash: {layer: flat grad}}``."""
    torch.manual_seed(model_seed)
    model = MLP()
    out: dict[str, dict[str, torch.Tensor]] = {}
    for ds in (_make_data(N_TRAIN, seed=1), _make_data(N_TEST, seed=2)):
        for i in range(len(ds)):
            sample = ds[i]
            model.zero_grad()
            _loss_fn(model, {k: v.unsqueeze(0) for k, v in sample.items()}).backward()
            out[hash_sample(sample)] = {
                name: module.weight.grad.flatten().clone()
                for name, module in model.named_modules()
                if isinstance(module, nn.Linear)
            }
    return out


def _collect(streamer: GradientStreamer) -> dict[str, dict[str, torch.Tensor]]:
    """Stream, materialize every layer, and key the rows by hash."""
    rows: dict[str, dict[str, torch.Tensor]] = {}
    for _step, grad, hashes in streamer:
        dense = grad.materialize()
        for i, h in enumerate(hashes):
            rows[h] = {
                name: dense.data[name][i].reshape(-1).detach().clone()
                for name in grad.layer_names
            }
    return rows


def _same(rows: dict, ref: dict) -> bool:
    return all(
        set(layers) == set(ref[h])
        and all(torch.allclose(v, ref[h][n], atol=ATOL) for n, v in layers.items())
        for h, layers in rows.items()
    )


def _shared_worker(rank, world_size, result_queue, rendezvous_path, mode, out_dir):
    _init_group(rank, world_size, rendezvous_path)

    def body():
        import torch.distributed as dist

        torch.manual_seed(SEED)
        model = MLP()
        args = _args(out_dir, fsdp="full_shard" if mode == "fsdp" else "")
        train_ds, test_ds = _make_data(N_TRAIN, seed=1), _make_data(N_TEST, seed=2)
        train = GradientStreamer(
            model, train_ds, args, batch_size=BATCH, loss_fn=_loss_fn
        )
        test = GradientStreamer(
            model,
            test_ds,
            args,
            batch_size=BATCH,
            loss_fn=_loss_fn,
            hook_manager=train.hook_manager,
            forward_model=train.forward_model,
            shard=False,
        )
        assert test.forward_model is train.forward_model
        with train, test:
            train_rows = _collect(train)
            test_rows = _collect(test)
        # Every rank sees every query; the train shards partition the set.
        all_test = {hash_sample(test_ds[i]) for i in range(N_TEST)}
        all_train = {hash_sample(train_ds[i]) for i in range(N_TRAIN)}
        gathered: list[list[str]] = [None] * world_size  # type: ignore[list-item]
        dist.all_gather_object(gathered, sorted(train_rows))
        shards = [set(s) for s in gathered]
        ok = set(test_rows) == all_test
        ok = ok and set().union(*shards) == all_train
        ok = ok and sum(len(s) for s in shards) == N_TRAIN
        # ... and the values match the single-process gradients.
        ok = ok and _same({**train_rows, **test_rows}, _reference_gradients(SEED))
        return ok, f"test rows {len(test_rows)} shards {[len(s) for s in shards]}"

    _report(rank, result_queue, body)


def _missing_wrapper_worker(rank, world_size, result_queue, rendezvous_path, out_dir):
    _init_group(rank, world_size, rendezvous_path)

    def body():
        model = MLP()
        args = _args(out_dir, fsdp="full_shard")
        train = GradientStreamer(
            model, _make_data(N_TRAIN, seed=1), args, batch_size=BATCH, loss_fn=_loss_fn
        )
        try:
            GradientStreamer(
                model,
                _make_data(N_TEST, seed=2),
                args,
                batch_size=BATCH,
                loss_fn=_loss_fn,
                hook_manager=train.hook_manager,
            )
        except ValueError as exc:
            return "forward_model" in str(exc), str(exc)
        return False, "no ValueError for a shared hook_manager without forward_model"

    _report(rank, result_queue, body)


class TestSharedForwardModel:
    @pytest.mark.parametrize("mode", ["fsdp", "ddp"])
    def test_test_probe_shares_wrapper_and_sees_every_query(self, mode, tmp_path):
        if not _can_bind_localhost():
            pytest.skip("local socket binds are not permitted in this environment")
        ok, report = _spawn(_shared_worker, mode, str(tmp_path))
        assert ok, report

    def test_shared_hooks_without_wrapper_are_refused(self, tmp_path):
        if not _can_bind_localhost():
            pytest.skip("local socket binds are not permitted in this environment")
        ok, report = _spawn(_missing_wrapper_worker, str(tmp_path))
        assert ok, report

    def test_single_process_ignores_shard_and_wrapper(self, tmp_path):
        torch.manual_seed(SEED)
        model = MLP()
        args = _args(tmp_path)
        train_ds, test_ds = _make_data(N_TRAIN, seed=1), _make_data(N_TEST, seed=2)
        train = GradientStreamer(
            model, train_ds, args, batch_size=BATCH, loss_fn=_loss_fn
        )
        test = GradientStreamer(
            model,
            test_ds,
            args,
            batch_size=BATCH,
            loss_fn=_loss_fn,
            hook_manager=train.hook_manager,
            forward_model=train.forward_model,
            shard=False,
        )
        assert train.forward_model is model
        with train, test:
            rows = {**_collect(train), **_collect(test)}
        ref = _reference_gradients(SEED)
        assert set(rows) == set(ref)
        assert _same(rows, ref)

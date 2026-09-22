"""Optimizer state read under ``FullyShardedDataParallel`` and ``fully_shard``.

Each rank of a sharded optimizer holds a slice of every state tensor.
:class:`OptimizerSnapshot` gathers the ranks' slices into the full tensor, so a
preconditioned capture and the recorded Adam moments must equal those of an
unsharded model whose optimizer saw the same gradients.  The warm-up batch is
identical on every rank, so the averaged gradient FSDP steps on equals the
unsharded one.

Two ``gloo`` CPU workers via ``torch.multiprocessing.spawn``.
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

from dattri_llm.gradient.callbacks import OptimizerStateCallback
from dattri_llm.gradient.hooks import REGISTER_ALL, HookManager, HookManagerConfig

VOCAB, EMBED, HIDDEN, OUT, SEQ, BATCH = 32, 8, 16, 4, 5, 4
WARMUP_STEPS = 2
MASK = {"style": "mask", "proj_dim": 12, "proj_seed": 3}


class Block(nn.Module):
    def __init__(self, d_in: int, d_out: int) -> None:
        super().__init__()
        self.lin = nn.Linear(d_in, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.lin(x))


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(VOCAB, EMBED)
        self.b1 = Block(EMBED, HIDDEN)
        self.b2 = Block(HIDDEN, HIDDEN)
        self.head = nn.Linear(HIDDEN, OUT)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.b2(self.b1(self.embedding(ids))))


def _can_bind_localhost() -> bool:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind(("127.0.0.1", 0))
    except OSError:
        return False
    finally:
        probe.close()
    return True


def _batch(seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB, (BATCH, SEQ), generator=gen)


def _config(masked: bool) -> HookManagerConfig:
    if not masked:
        return HookManagerConfig(linear_io=REGISTER_ALL)
    layers = ["b1.lin", "b2.lin", "head"]
    return HookManagerConfig(
        linear_io=REGISTER_ALL, projection_kwargs={name: dict(MASK) for name in layers}
    )


def _run(model, forward, optimizer, config, batch, *, what):
    """Warm the optimizer up, then capture *batch*: the preconditioned
    gradient (``what="precondition"``) or the recorded moments.
    """
    preconditions = what == "precondition"
    collector = HookManager(
        model, config=config, optimizer=optimizer if preconditions else None
    )
    fwd = forward()
    opt = optimizer()
    if preconditions:
        collector.precondition = False
    for _ in range(WARMUP_STEPS):
        fwd(_batch(5)).sum().backward()
        opt.step()
        fwd.zero_grad()
    if preconditions:
        collector.precondition = True
        with collector.collect():
            fwd(batch).sum().backward()
        return collector.get_gradient().materialize().data
    recorder = OptimizerStateCallback(
        model, opt, projection_kwargs=config.projection_kwargs
    )
    collector.add_callback(recorder)
    with collector.collect():
        fwd(batch).sum().backward()
    opt.step()
    recorder.record_post(0)
    (entry,) = recorder.dynamics().values()
    return entry


def _flatten_dynamics(entry: dict) -> dict[str, torch.Tensor]:
    out = {}
    for side in ("pre", "post"):
        for name, (m, v) in entry[side].items():
            out[f"{side}.{name}.m"], out[f"{side}.{name}.v"] = m, v
    out["step"] = torch.tensor(float(entry["step"]))
    return out


def _worker(rank, world_size, scenario, result_queue, rendezvous_path):
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy

    what, api, nested, masked = scenario
    torch.manual_seed(0)
    model = Net()
    init_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    # The post-step moments depend on the gradient the optimizer steps on,
    # which FSDP averages over ranks: that batch is shared too.  A
    # preconditioned capture reads pre-step state only, so its batch differs.
    batch = _batch(100 + (rank if what == "precondition" else 0))

    ref = Net()
    ref.load_state_dict(init_state)
    ref_opt = torch.optim.AdamW(ref.parameters(), lr=1e-2)
    expected = _run(
        ref, lambda: ref, lambda: ref_opt, _config(masked), batch, what=what
    )

    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous_path}",
        rank=rank,
        world_size=world_size,
    )
    try:
        holder: dict = {}

        def forward():
            if api == "fully_shard":
                from torch.distributed.fsdp import fully_shard

                if nested:
                    fully_shard(model.b1)
                    fully_shard(model.b2)
                holder["fsdp"] = fully_shard(model)
            else:
                holder["fsdp"] = FSDP(
                    model,
                    device_id=torch.device("cpu"),
                    use_orig_params=True,
                    auto_wrap_policy=ModuleWrapPolicy({Block}) if nested else None,
                )
            return holder["fsdp"]

        def optimizer():
            if "opt" not in holder:
                holder["opt"] = torch.optim.AdamW(holder["fsdp"].parameters(), lr=1e-2)
            return holder["opt"]

        # The hooks go on the unwrapped model, then the model is wrapped, then
        # the optimizer is built over the wrapper's parameters.
        actual = _run(model, forward, optimizer, _config(masked), batch, what=what)
        if what == "dynamics":
            actual, expected = _flatten_dynamics(actual), _flatten_dynamics(expected)
        assert actual.keys() == expected.keys(), (sorted(actual), sorted(expected))
        diffs = {k: (actual[k] - expected[k]).abs().max().item() for k in expected}
        worst = max(diffs, key=diffs.get)
        if rank == 0:
            report = f"scenario={scenario} max diff {diffs[worst]:.2e} at {worst!r}"
            result_queue.put((diffs[worst] < 1e-6, report))
    except Exception:  # noqa: BLE001 - surface any worker failure to the test
        import traceback

        if rank == 0:
            result_queue.put((False, "WORKER EXC:\n" + traceback.format_exc()))
    finally:
        dist.destroy_process_group()


def _spawn(scenario: tuple) -> tuple[bool, str]:
    if not _can_bind_localhost():
        pytest.skip("local socket binds are not permitted in this environment")
    result_queue = mp.get_context("spawn").Queue()
    fd, rendezvous_path = tempfile.mkstemp()
    os.close(fd)
    try:
        mp.spawn(
            _worker,
            args=(2, scenario, result_queue, rendezvous_path),
            nprocs=2,
            join=True,
        )
    finally:
        if pathlib.Path(rendezvous_path).exists():
            pathlib.Path(rendezvous_path).unlink()
    assert not result_queue.empty(), "rank-0 worker did not report a result"
    return result_queue.get()


def _has_fully_shard() -> bool:
    try:
        from torch.distributed.fsdp import fully_shard  # noqa: F401
    except ImportError:
        return False
    return True


APIS = [
    "wrapper",
    pytest.param(
        "fully_shard",
        marks=pytest.mark.skipif(not _has_fully_shard(), reason="no fully_shard"),
    ),
]
NESTED = pytest.mark.parametrize(
    "nested", [False, True], ids=["one_unit", "unit_per_block"]
)
MASKED = pytest.mark.parametrize("masked", [False, True], ids=["dense", "mask"])


class TestOptimizerStateUnderFSDP:
    @pytest.mark.parametrize("api", APIS)
    @NESTED
    @MASKED
    def test_preconditioned_capture_matches_unsharded(self, api, nested, masked):
        ok, report = _spawn(("precondition", api, nested, masked))
        assert ok, report

    @pytest.mark.parametrize("api", APIS)
    @NESTED
    @MASKED
    def test_recorded_moments_match_unsharded(self, api, nested, masked):
        ok, report = _spawn(("dynamics", api, nested, masked))
        assert ok, report


# --------------------------------------------------------------------------- #
# The streamer's preconditioned trajectory: FSDP against DDP                   #
# --------------------------------------------------------------------------- #
#
# Both wrappers step on the rank-averaged gradient, so under one set of
# arguments they follow the same trajectory and a rank captures the same
# preconditioned blocks; DDP's optimizer state is replicated and read directly.


class DictDataset(torch.utils.data.Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self.x, self.y = x, y

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, i: int) -> dict:
        return {"x": self.x[i], "y": self.y[i]}


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 3, bias=False)

    def forward(self, x: torch.Tensor, **_: object) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def _mse(model: nn.Module, batch: dict) -> torch.Tensor:
    return ((model(**batch) - batch["y"]) ** 2).sum()


def _stream_blocks(mode: str, out_dir: str) -> list[dict[str, torch.Tensor]]:
    from dattri_llm.attribution.arguments import AttributionArguments
    from dattri_llm.gradient.streaming import GradientStreamer

    torch.manual_seed(0)
    model = MLP()
    gen = torch.Generator().manual_seed(1)
    data = DictDataset(
        torch.randn(12, 4, generator=gen), torch.randn(12, 3, generator=gen)
    )
    args = AttributionArguments(
        output_dir=str(pathlib.Path(out_dir) / mode),
        per_device_train_batch_size=2,
        use_cpu=True,
        dataloader_pin_memory=False,
        optim="adamw_torch",
        learning_rate=0.05,
        lr_scheduler_type="constant",
        fsdp="full_shard" if mode == "fsdp" else "",
    )
    streamer = GradientStreamer(
        model,
        data,
        args,
        batch_size=2,
        enable_update=True,
        precondition=True,
        loss_fn=_mse,
    )
    with streamer:
        return [
            {k: v.detach().clone() for k, v in block.materialize().data.items()}
            for _, block, _ in streamer
        ]


def _streamer_worker(rank, world_size, _scenario, result_queue, rendezvous_path):
    import torch.distributed as dist

    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(world_size))
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous_path}",
        rank=rank,
        world_size=world_size,
    )
    try:
        with tempfile.TemporaryDirectory() as out_dir:
            expected = _stream_blocks("ddp", out_dir)
            actual = _stream_blocks("fsdp", out_dir)
        worst = max(
            (a[k] - e[k]).abs().max().item()
            for a, e in zip(actual, expected, strict=True)
            for k in e
        )
        scale = max(e[k].abs().max().item() for e in expected for k in e)
        if rank == 0:
            report = (
                f"{len(expected)} steps, max diff {worst:.2e} (entries to {scale:.2e})"
            )
            result_queue.put(
                (len(expected) > 2 and worst < 1e-5 * max(scale, 1), report)
            )
    except Exception:  # noqa: BLE001 - surface any worker failure to the test
        import traceback

        if rank == 0:
            result_queue.put((False, "WORKER EXC:\n" + traceback.format_exc()))
    finally:
        dist.destroy_process_group()


def _refusal_worker(rank, world_size, scenario, result_queue, rendezvous_path):
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous_path}",
        rank=rank,
        world_size=world_size,
    )
    message = "no error"
    try:
        torch.manual_seed(0)
        model = Net()
        holder: dict = {}
        collector = HookManager(
            model, config=_config(masked=False), optimizer=lambda: holder["opt"]
        )
        wrapped = FSDP(model, device_id=torch.device("cpu"), use_orig_params=False)
        holder["opt"] = torch.optim.AdamW(wrapped.parameters(), lr=1e-2)
        with collector.collect():
            wrapped(_batch(100 + rank)).sum().backward()
    except Exception as exc:  # noqa: BLE001 - the refusal is what is under test
        message = f"{type(exc).__name__}: {exc}"
    finally:
        if rank == 0:
            result_queue.put((True, message))
        dist.destroy_process_group()


def _spawn_worker(worker, scenario) -> tuple[bool, str]:
    if not _can_bind_localhost():
        pytest.skip("local socket binds are not permitted in this environment")
    result_queue = mp.get_context("spawn").Queue()
    fd, rendezvous_path = tempfile.mkstemp()
    os.close(fd)
    try:
        mp.spawn(
            worker,
            args=(2, scenario, result_queue, rendezvous_path),
            nprocs=2,
            join=True,
        )
    finally:
        if pathlib.Path(rendezvous_path).exists():
            pathlib.Path(rendezvous_path).unlink()
    assert not result_queue.empty(), "rank-0 worker did not report a result"
    return result_queue.get()


class TestStreamerPreconditionUnderFSDP:
    def test_blocks_match_ddp(self):
        ok, report = _spawn_worker(_streamer_worker, None)
        assert ok, report


class TestUnsupportedShardedOptimizers:
    def test_flat_parameters_are_refused(self):
        _, message = _spawn_worker(_refusal_worker, "use_orig_params_false")
        assert "NotImplementedError" in message, message
        assert "use_orig_params=True" in message, message

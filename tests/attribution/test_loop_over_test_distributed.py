"""``loop_over_test=True`` in a distributed training trajectory.

Two ``gloo`` CPU workers via ``torch.multiprocessing.spawn``.  Re-streaming
the query probe between train blocks (a real wrapped backward on the shared
model) must leave the trajectory bit-identical to a plain training stream
with no query probe at all, under gradient accumulation, for the
streamer's own DDP wrapper, a caller's DDP wrapper whose gradients are views
into its communication buckets, FSDP, and a caller's FSDP wrapper keeping
low-precision gradients.  Under the latter, a query probe that leaves some
of the trajectory's gradients untouched (a branch its loss never reaches, or
a loss that raises before its backward) must still hand them all back.
"""

from __future__ import annotations

import contextlib
import os
import pathlib
import socket
import tempfile

import pytest
import torch
import torch.multiprocessing as mp
from torch import nn


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
        self.drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(8, 3, bias=False)

    def forward(self, x: torch.Tensor, **_: object) -> torch.Tensor:
        x = x.to(self.fc1.weight.dtype)  # under a low-precision FSDP wrapper
        return self.fc2(self.drop(torch.relu(self.fc1(x))))


class TwoBranch(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(4, 3)
        self.b = nn.Linear(4, 3, bias=False)

    def forward(
        self, x: torch.Tensor, only_a: bool = False, **_: object
    ) -> torch.Tensor:
        x = x.to(torch.bfloat16)  # the wrapper's compute dtype
        return self.a(x) if only_a else self.a(x) + self.b(x)


def _mse(model: nn.Module, batch: dict) -> torch.Tensor:
    return ((model(x=batch["x"]) - batch["y"]) ** 2).sum()


def _can_bind_localhost() -> bool:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind(("127.0.0.1", 0))
    except OSError:
        return False
    finally:
        probe.close()
    return True


def _trained(scenario: str, run: str, out_dir: str) -> torch.Tensor:
    from dattri_llm import AttributionTask, TracInAttributor
    from dattri_llm.attribution.arguments import AttributionArguments
    from dattri_llm.gradient.streaming import GradientStreamer

    torch.manual_seed(0)
    model = MLP()
    gen = torch.Generator().manual_seed(1)
    train = DictDataset(
        torch.randn(12, 4, generator=gen), torch.randn(12, 3, generator=gen)
    )
    test = DictDataset(
        torch.randn(4, 4, generator=gen), torch.randn(4, 3, generator=gen)
    )
    args = AttributionArguments(
        output_dir=str(pathlib.Path(out_dir) / scenario / run),
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        use_cpu=True,
        dataloader_pin_memory=False,
        # AdamW cannot step fp32 parameters holding bf16 gradients.
        optim="sgd" if scenario == "fsdp_low_precision_grads" else "adamw_torch",
        learning_rate=0.05,
        lr_scheduler_type="constant",
        max_grad_norm=0.5,
        gradient_accumulation_steps=2,
        fsdp="full_shard" if scenario == "fsdp" else "",
    )
    wrapped = None
    if scenario == "ddp_bucket_view":
        wrapped = nn.parallel.DistributedDataParallel(
            model, gradient_as_bucket_view=True
        )
    elif scenario == "fsdp_low_precision_grads":
        from torch.distributed.fsdp import FullyShardedDataParallel, MixedPrecision

        # bf16 gradients on the fp32 original parameters: a plain ``.grad``
        # assignment of them is rejected.
        wrapped = FullyShardedDataParallel(
            model,
            use_orig_params=True,
            device_id=torch.device("cpu"),
            mixed_precision=MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                keep_low_precision_grads=True,
                # The inputs stay fp32 for sample hashing; the model casts.
                cast_forward_inputs=False,
                cast_root_forward_inputs=False,
            ),
        )
    if run == "plain":  # the trajectory alone, no query probe
        streamer = GradientStreamer(
            model,
            train,
            args,
            batch_size=2,
            enable_update=True,
            loss_fn=_mse,
            forward_model=wrapped,
        )
        with streamer:
            for _ in streamer:
                pass
    else:
        task = AttributionTask(loss_func=_mse, model=wrapped or model)
        TracInAttributor(args, task=task).attribute(
            train, test, enable_update=True, loop_over_test=run == "looped"
        )
    return torch.cat([p.detach().reshape(-1).clone() for p in model.parameters()])


def _probe_hands_back(scenario: str, out_dir: str) -> str:
    """Empty when a query probe run mid-window under a low-precision FSDP
    wrapper hands the window's gradients, modes and state back intact, and
    the resumed trajectory trains exactly as one without the probe.
    """
    from torch.distributed.fsdp import FullyShardedDataParallel, MixedPrecision

    from dattri_llm.attribution.arguments import AttributionArguments
    from dattri_llm.gradient.streaming import GradientStreamer

    gen = torch.Generator().manual_seed(1)
    train = DictDataset(
        torch.randn(8, 4, generator=gen), torch.randn(8, 3, generator=gen)
    )
    test = DictDataset(
        torch.randn(4, 4, generator=gen), torch.randn(4, 3, generator=gen)
    )
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        keep_low_precision_grads=True,
        cast_forward_inputs=False,
        cast_root_forward_inputs=False,
    )

    def wrap(module: nn.Module) -> nn.Module:
        return FullyShardedDataParallel(
            module,
            use_orig_params=True,
            device_id=torch.device("cpu"),
            mixed_precision=mp_policy,
        )

    def args(**overrides) -> AttributionArguments:
        return AttributionArguments(
            output_dir=str(pathlib.Path(out_dir) / scenario),
            use_cpu=True,
            dataloader_pin_memory=False,
            optim="sgd",
            learning_rate=0.05,
            lr_scheduler_type="constant",
            **overrides,
        )

    def query_loss(fwd: nn.Module, batch: dict) -> torch.Tensor:
        if scenario == "failing_probe":
            raise KeyError("query loss failed")  # after the probe's zero_grad
        return ((fwd(x=batch["x"], only_a=True) - batch["y"]) ** 2).sum()

    def build() -> tuple[nn.Module, GradientStreamer]:
        torch.manual_seed(0)
        model = TwoBranch()
        # Each branch its own FSDP unit: one the probe skips keeps no fresh
        # gradient storage from the probe's backward.
        model.a, model.b = wrap(model.a), wrap(model.b)
        trajectory = GradientStreamer(
            model,
            train,
            args(gradient_accumulation_steps=2),
            batch_size=2,
            enable_update=True,
            loss_fn=_mse,
            forward_model=wrap(model),
        )
        return model, trajectory

    def local_parameters(model: nn.Module) -> torch.Tensor:
        return torch.cat([p.detach().reshape(-1).clone() for p in model.parameters()])

    reference, trajectory = build()  # the trajectory alone, no query probe
    initial = local_parameters(reference)
    with trajectory:
        for _block in trajectory:
            pass
    want = local_parameters(reference)
    problems = [] if not torch.equal(want, initial) else ["the reference never trained"]

    model, trajectory = build()
    probe = GradientStreamer(
        model,
        test,
        args(),
        batch_size=2,
        loss_fn=query_loss,
        hook_manager=trajectory.hook_manager,
        forward_model=trajectory.forward_model,
    )
    with trajectory:
        blocks = iter(trajectory)
        next(blocks)  # half an accumulation window
        # None where this rank's shard holds none of a parameter.
        window = {
            n: None if p.grad is None else p.grad.clone()
            for n, p in model.named_parameters()
        }
        if all(g is None for g in window.values()):
            problems.append("no window gradient on this rank to hand back")
        try:
            with probe:  # entering seeds, as training does
                rng = torch.get_rng_state()
                for _block in probe:
                    pass
        except KeyError as err:
            if scenario != "failing_probe":
                raise
            if "query loss failed" not in str(err):
                problems.append(f"the query's own error was replaced: {err!r}")
        for n, p in model.named_parameters():
            if window[n] is None:
                handed_back = p.grad is None
            else:
                handed_back = p.grad is not None and torch.equal(p.grad, window[n])
            if not handed_back:
                problems.append(f"window gradient of {n} not handed back")
        if not torch.equal(torch.get_rng_state(), rng):
            problems.append("RNG state not handed back")
        if not model.training:
            problems.append("train() mode not handed back")
        if probe._entered or probe._pass_state is not None:
            problems.append("the probe's pass is still open")
        # Resume the trajectory (next(), not a new for: iter() restarts it).
        with contextlib.suppress(StopIteration):
            while True:
                next(blocks)
    got = local_parameters(model)
    if not torch.equal(got, want):
        problems.append(
            f"trained parameters differ by {(got - want).abs().max().item():.3e}"
        )
    return "; ".join(problems)


def _worker(rank, world_size, scenario, result_queue, rendezvous_path):
    import faulthandler

    import torch.distributed as dist

    # A rank failing alone leaves its peer blocked in a collective: end both.
    faulthandler.dump_traceback_later(300, exit=True)
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
            if scenario in ("conditional_probe", "failing_probe"):
                problems = _probe_hands_back(scenario, out_dir)
                ok = torch.tensor([float(not problems)])
                report = f"rank {rank}: {problems or 'ok'}"
            else:
                got = {
                    run: _trained(scenario, run, out_dir)
                    for run in ("plain", "cached", "looped")
                }
                diffs = {
                    run: (got[run] - got["plain"]).abs().max().item()
                    for run in ("cached", "looped")
                }
                ok = torch.tensor([float(max(diffs.values()) <= 0)])
                report = f"rank 0 max diffs vs plain: {diffs}"
        dist.all_reduce(ok, op=dist.ReduceOp.MIN)
        if rank == 0:
            result_queue.put((bool(ok.item()), report))
    except Exception:  # noqa: BLE001 - surface any worker failure to the test
        import traceback

        if rank == 0:
            result_queue.put((False, "WORKER EXC:\n" + traceback.format_exc()))
    finally:
        dist.destroy_process_group()


def _spawn(scenario: str) -> tuple[bool, str]:
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


class TestLoopedProbeKeepsDistributedTrajectory:
    @pytest.mark.parametrize(
        "scenario", ["ddp", "ddp_bucket_view", "fsdp", "fsdp_low_precision_grads"]
    )
    def test_trained_parameters_match(self, scenario):
        ok, report = _spawn(scenario)
        assert ok, report


class TestQueryProbeHandsBackLowPrecisionFSDPGradients:
    @pytest.mark.parametrize("scenario", ["conditional_probe", "failing_probe"])
    def test_window_is_handed_back(self, scenario):
        ok, report = _spawn(scenario)
        assert ok, report

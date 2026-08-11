"""Unit tests for device prefetching: ``Gradient.pin_memory`` /
``non_blocking`` moves and :func:`dattri_llm.gradient.prefetch.prefetch_to_device`.

The overlap machinery (side stream + events) only engages on CUDA, so those
paths carry ``skipif`` guards; everything else -- passthrough semantics, order
preservation, and metadata fidelity -- is exercised on CPU.  The scoring-path
integration lives in ``tests/attribution/test_utils.py``.
"""

from __future__ import annotations

import pytest
import torch

from dattri_llm.gradient.gradient import Factorized, Gradient
from dattri_llm.gradient.prefetch import _payload_tensors, prefetch_to_device

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for stream-overlap paths",
)


def _pinning_works() -> bool:
    """Whether ``Tensor.pin_memory()`` functions on this build.

    Pinning needs a CUDA-like accelerator; notably it is broken on MPS-only
    torch builds (the pin lands on the wrong device and raises).
    """
    try:
        return bool(torch.empty(1).pin_memory().is_pinned())
    except RuntimeError:
        return False


pinning_only = pytest.mark.skipif(
    not _pinning_works(),
    reason="Tensor.pin_memory() not functional on this build",
)

B, T, D_IN, D_OUT = 2, 4, 3, 5


def make_block(seed: int = 0, b: int = B) -> Gradient:
    """A mixed materialized+factorized Gradient block."""
    g = torch.Generator().manual_seed(seed)
    return Gradient(
        representation={"mat": "materialized", "fac": "factorized"},
        data={
            "mat": torch.randn(b, D_IN * D_OUT, generator=g),
            "fac": Factorized(
                activation=torch.randn(b, T, D_IN, generator=g),
                pre_activation_grad=torch.randn(b, T, D_OUT, generator=g),
            ),
        },
        layer_types={"mat": "nn.Linear", "fac": "nn.Linear"},
        indexing={"fac": "batch_token"},
    )


def make_materialized_block(seed: int = 0, b: int = B) -> Gradient:
    """A purely materialized block (the re-batchable kind)."""
    g = torch.Generator().manual_seed(seed)
    return Gradient(
        representation={"l1": "materialized"},
        data={"l1": torch.randn(b, D_IN * D_OUT, generator=g)},
        layer_types={"l1": "nn.Linear"},
    )


def make_stream(n: int = 4, factory=make_block) -> list:
    return [
        (i, factory(seed=i), [f"h{i}-{j}" for j in range(B)]) for i in range(n)
    ]


def assert_blocks_equal(a: Gradient, b: Gradient) -> None:
    assert a.layer_names == b.layer_names
    assert a.representation == b.representation
    assert a.indexing == b.indexing
    for ta, tb in zip(_payload_tensors(a), _payload_tensors(b)):
        assert torch.equal(ta.cpu(), tb.cpu())


# --------------------------------------------------------------------------- #
# non_blocking .to                                                             #
# --------------------------------------------------------------------------- #


class TestNonBlockingTo:
    def test_gradient_to_non_blocking_preserves_values(self):
        block = make_block()
        moved = block.to("cpu", non_blocking=True)
        assert_blocks_equal(block, moved)

    def test_factorized_to_non_blocking_with_dtype(self):
        fac = Factorized(
            activation=torch.randn(B, T, D_IN),
            pre_activation_grad=torch.randn(B, T, D_OUT),
            batch_first=False,
        )
        moved = fac.to("cpu", dtype=torch.float64, non_blocking=True)
        assert moved.activation.dtype == torch.float64
        assert moved.batch_first is False  # layout flag survives the move
        assert torch.equal(moved.activation, fac.activation.double())

    def test_default_is_blocking(self):
        # The new parameter must not change the default signature behavior.
        block = make_block()
        assert_blocks_equal(block, block.to("cpu"))

    @cuda_only
    def test_gradient_round_trip_cuda(self):
        block = make_block()
        dev = block.to("cuda", non_blocking=True)
        torch.cuda.synchronize()
        assert dev.device.type == "cuda"
        assert_blocks_equal(block, dev.to("cpu"))


# --------------------------------------------------------------------------- #
# pin_memory                                                                   #
# --------------------------------------------------------------------------- #


class TestPinMemory:
    def test_has_pin_memory_hook(self):
        # torch's DataLoader pin-memory recursion dispatches on this attribute.
        assert hasattr(make_block(), "pin_memory")
        assert hasattr(make_block().data["fac"], "pin_memory")

    @pinning_only
    def test_pins_every_payload(self):
        block = make_block()
        pinned = block.pin_memory()
        for t in _payload_tensors(pinned):
            assert t.is_pinned()
        assert_blocks_equal(block, pinned)

    @pinning_only
    def test_idempotent(self):
        pinned = make_block().pin_memory()
        again = pinned.pin_memory()
        # Already-pinned payloads pass through as the same tensor objects.
        for t1, t2 in zip(_payload_tensors(pinned), _payload_tensors(again)):
            assert t1 is t2

    @pinning_only
    def test_metadata_preserved(self):
        fac = Factorized(
            activation=torch.randn(T, B, D_IN),
            pre_activation_grad=torch.randn(T, B, D_OUT),
            module_kwargs={"padding_idx": 0},
            batch_first=False,
        )
        pinned = fac.pin_memory()
        assert pinned.module_kwargs == {"padding_idx": 0}
        assert pinned.batch_first is False

    @cuda_only
    def test_device_payloads_pass_through(self):
        block = make_block().to("cuda")
        pinned = block.pin_memory()
        for t1, t2 in zip(_payload_tensors(block), _payload_tensors(pinned)):
            assert t1 is t2  # nothing to pin on device


# --------------------------------------------------------------------------- #
# prefetch_to_device                                                           #
# --------------------------------------------------------------------------- #


class TestPrefetchToDevice:
    @pytest.mark.parametrize("depth", [0, 1, 3])
    def test_cpu_passthrough_preserves_stream(self, depth):
        blocks = make_stream(5)
        out = list(prefetch_to_device(iter(blocks), "cpu", depth=depth))
        assert len(out) == len(blocks)
        for (step, g, hashes), (o_step, o_g, o_hashes) in zip(blocks, out):
            assert o_step == step
            assert o_hashes == hashes
            assert_blocks_equal(g, o_g)

    def test_depth_larger_than_stream(self):
        blocks = make_stream(2)
        out = list(prefetch_to_device(iter(blocks), "cpu", depth=10))
        assert [o[0] for o in out] == [0, 1]

    def test_empty_stream(self):
        assert list(prefetch_to_device(iter([]), "cpu")) == []

    def test_negative_depth_rejected(self):
        with pytest.raises(ValueError, match="depth"):
            list(prefetch_to_device(iter(make_stream(1)), "cpu", depth=-1))

    def test_arbitrary_metadata_passes_through(self):
        # The re-batched scoring path sends (steps_list, Gradient, ids_list).
        blocks = [([0, 0, 1], make_materialized_block(b=3), ["a", "b", "c"])]
        ((steps, _g, ids),) = list(prefetch_to_device(iter(blocks), "cpu"))
        assert steps == [0, 0, 1]
        assert ids == ["a", "b", "c"]

    def test_lazy_consumption(self):
        # The prefetcher must not drain the source eagerly: with depth=1 no
        # more than depth+1 items may have been pulled before the first yield.
        pulled = []

        def source():
            for item in make_stream(6):
                pulled.append(item[0])
                yield item

        it = prefetch_to_device(source(), "cpu", depth=1)
        # CPU passthrough is fully lazy (pull-per-yield).
        next(it)
        assert len(pulled) <= 2

    @cuda_only
    @pytest.mark.parametrize("depth", [1, 2])
    def test_cuda_prefetch_matches_source(self, depth):
        blocks = make_stream(5)
        out = list(prefetch_to_device(iter(blocks), "cuda", depth=depth))
        torch.cuda.synchronize()
        assert [o[0] for o in out] == [b[0] for b in blocks]
        for (_, g, hashes), (_, o_g, o_hashes) in zip(blocks, out):
            assert o_g.device.type == "cuda"
            assert o_hashes == hashes
            assert_blocks_equal(g, o_g)

    @cuda_only
    def test_cuda_resident_blocks_pass_through(self):
        dev_block = make_block().to("cuda")
        out = list(prefetch_to_device(iter([(0, dev_block, ["h"])]), "cuda"))
        assert out[0][1] is dev_block  # no copy for already-resident blocks

    @cuda_only
    def test_cuda_depth_zero_is_synchronous(self):
        blocks = make_stream(3)
        out = list(prefetch_to_device(iter(blocks), "cuda", depth=0))
        for (_, g, _), (_, o_g, _) in zip(blocks, out):
            assert o_g.device.type == "cuda"
            assert_blocks_equal(g, o_g)


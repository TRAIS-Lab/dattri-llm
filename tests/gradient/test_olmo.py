"""Unit tests for HookManager on OLMo models.

Covers two parallelism settings:

* **Single-process** -- plain ``OLMo`` on CPU, no distributed setup required.
* **FSDP** -- ``FullyShardedDataParallel``-wrapped ``OLMo`` launched via
  ``torch.multiprocessing.spawn`` (2 workers using the ``gloo`` backend so the
  tests pass on CPU without any extra configuration).

Running
-------
::

    pytest tests/gradient/test_olmo.py           # skipped if OLMo absent
    pytest tests/gradient/test_olmo.py -v        # verbose
    pytest tests/gradient/test_olmo.py -k fsdp   # FSDP tests only
"""

from __future__ import annotations

import os
import socket
import tempfile
from typing import List

import pytest
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Optional OLMo import -- skip the whole file if not installed
# ---------------------------------------------------------------------------

try:
    from olmo.config import ActivationType, ModelConfig
    from olmo.model import OLMo

    OLMO_AVAILABLE = True
except ImportError:
    OLMO_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not OLMO_AVAILABLE,
    reason="ai2-olmo is not installed",
)

from dattri_llm.gradient.gradient import Factorized, GradientRecord  # noqa: E402
from dattri_llm.gradient.hooks import HookManager, HookManagerCallback, HookManagerConfig  # noqa: E402

# OLMo names its block-level MLP linears ``ff_proj`` / ``ff_out`` under
# ``transformer.blocks.<i>``.  Anchoring to ``blocks.\d+`` excludes the
# top-level ``transformer.ff_out`` (vocabulary projection, present when
# ``weight_tying=False``).
OLMO_MLP_PATTERNS: list[str] = [
    r"transformer\.blocks\.\d+\.ff_proj",
    r"transformer\.blocks\.\d+\.ff_out",
]

# ---------------------------------------------------------------------------
# Config (kept tiny for CPU / CI speed)
# ---------------------------------------------------------------------------

SEED = 42
VOCAB_SIZE = 64
SEQ_LEN = 16
BATCH_SIZE = 2
N_LAYERS = 2
D_MODEL = 64
N_HEADS = 2
MLP_RATIO = 4
ATOL = 1e-5
RTOL = 1e-4


# ---------------------------------------------------------------------------
# In-memory callback for testing
# ---------------------------------------------------------------------------


class _Capture(HookManagerCallback):
    """Accumulates every GradientRecord emitted by HookManager."""

    def __init__(self) -> None:
        self.records: List[GradientRecord] = []

    def on_step_end(self, record: GradientRecord) -> None:
        self.records.append(record)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tiny_olmo() -> OLMo:
    """A minimal 2-layer OLMo that runs entirely on CPU."""
    torch.manual_seed(SEED)
    cfg = ModelConfig(
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        vocab_size=VOCAB_SIZE,
        max_sequence_length=SEQ_LEN,
        mlp_ratio=MLP_RATIO,
        activation_type=ActivationType.swiglu,
        rope=True,
        flash_attention=False,
        include_bias=False,
        weight_tying=True,
    )
    model = OLMo(cfg)
    model.train()
    return model


@pytest.fixture()
def sample_ids() -> torch.Tensor:
    """A fixed batch of token IDs -- deterministic across all test runs."""
    return _make_sample_ids()


def _make_sample_ids() -> torch.Tensor:
    """Build deterministic sample IDs without mutating global RNG state."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED)
    return torch.randint(
        0,
        VOCAB_SIZE,
        (BATCH_SIZE, SEQ_LEN),
        generator=generator,
    )


def _can_bind_localhost() -> bool:
    """Return whether the current environment permits binding a localhost socket."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind(("127.0.0.1", 0))
    except OSError:
        return False
    finally:
        probe.close()
    return True


def _make_hook_cfg() -> HookManagerConfig:
    return HookManagerConfig(linear_io=OLMO_MLP_PATTERNS)


# ---------------------------------------------------------------------------
# Helper: one forward+backward inside collect() context
# ---------------------------------------------------------------------------


def _run_forward_backward(
    model: torch.nn.Module,
    ids: torch.Tensor,
    collector: HookManager,
    capture: _Capture,
) -> None:
    model.zero_grad()
    with collector.collect():
        out = model(input_ids=ids)
        shift_logits = out.logits[:, :-1, :].contiguous()
        shift_labels = ids[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1),
        )
        loss.backward()


# ---------------------------------------------------------------------------
# Single-process tests
# ---------------------------------------------------------------------------


class TestOLMoCollectorSingle:
    """HookManager on a plain OLMo model (no parallelism)."""

    def test_layers_hooked(self, tiny_olmo, sample_ids):
        """Collector registers exactly N_LAYERS * 2 hooks (ff_proj + ff_out each)."""
        collector = HookManager(tiny_olmo, config=_make_hook_cfg())
        assert len(collector.layer_names) == N_LAYERS * 2, (
            f"Expected {N_LAYERS * 2} layers, got {len(collector.layer_names)}: "
            f"{collector.layer_names}"
        )
        collector.remove()

    def test_correct_layer_names(self, tiny_olmo):
        """Hooked names follow the pattern transformer.blocks.<i>.ff_proj/ff_out."""
        collector = HookManager(tiny_olmo, config=_make_hook_cfg())
        for name in collector.layer_names:
            assert "transformer.blocks." in name, f"Unexpected layer: {name}"
            assert name.endswith(("ff_proj", "ff_out")), f"Unexpected suffix: {name}"
        collector.remove()

    def test_lm_head_not_hooked(self, tiny_olmo):
        """Top-level transformer.ff_out (lm-head) must not appear in the hook set."""
        collector = HookManager(tiny_olmo, config=_make_hook_cfg())
        assert "transformer.ff_out" not in collector.layer_names, (
            "lm-head (transformer.ff_out) should not be hooked"
        )
        collector.remove()

    def test_buffers_populated_after_pass(self, tiny_olmo, sample_ids):
        """Both activation and pre_activation_grad are non-None after forward+backward."""
        capture = _Capture()
        collector = HookManager(tiny_olmo, config=_make_hook_cfg(), callbacks=[capture])
        _run_forward_backward(tiny_olmo, sample_ids, collector, capture)
        assert capture.records, "No records emitted"
        rec = capture.records[0]
        for layer_name, val in rec.gradient.data.items():
            assert isinstance(val, Factorized), f"{layer_name}: expected Factorized"
            assert val.activation is not None, f"{layer_name}: activation is None"
            assert val.pre_activation_grad is not None, f"{layer_name}: grad is None"
        collector.remove()

    def test_all_values_finite(self, tiny_olmo, sample_ids):
        """No NaN or Inf in any collected tensor."""
        capture = _Capture()
        collector = HookManager(tiny_olmo, config=_make_hook_cfg(), callbacks=[capture])
        _run_forward_backward(tiny_olmo, sample_ids, collector, capture)
        rec = capture.records[0]
        for layer_name, val in rec.gradient.data.items():
            assert isinstance(val, Factorized)
            assert torch.isfinite(val.activation).all(), (
                f"{layer_name}: activation contains non-finite values"
            )
            assert torch.isfinite(val.pre_activation_grad).all(), (
                f"{layer_name}: pre_activation_grad contains non-finite values"
            )
        collector.remove()

    def test_swiglu_activation_shapes(self, tiny_olmo, sample_ids):
        """Verify SwiGLU-specific tensor shapes for ff_proj and ff_out.

        Expected (with output_multiplier=0.5):
            ff_proj  activation         : (B, T, D_MODEL)
            ff_proj  pre_activation_grad: (B, T, MLP_RATIO * D_MODEL)
            ff_out   activation         : (B, T, MLP_RATIO * D_MODEL // 2)
            ff_out   pre_activation_grad: (B, T, D_MODEL)
        """
        capture = _Capture()
        collector = HookManager(tiny_olmo, config=_make_hook_cfg(), callbacks=[capture])
        _run_forward_backward(tiny_olmo, sample_ids, collector, capture)
        rec = capture.records[0]
        hidden = MLP_RATIO * D_MODEL // 2  # SwiGLU halves before ff_out

        for layer_name, val in rec.gradient.data.items():
            assert isinstance(val, Factorized)
            act, go = val.activation, val.pre_activation_grad
            if "ff_proj" in layer_name:
                assert act.shape == (BATCH_SIZE, SEQ_LEN, D_MODEL), (
                    f"{layer_name} ff_proj activation: expected "
                    f"{(BATCH_SIZE, SEQ_LEN, D_MODEL)}, got {tuple(act.shape)}"
                )
                assert go.shape == (BATCH_SIZE, SEQ_LEN, MLP_RATIO * D_MODEL), (
                    f"{layer_name} ff_proj pre_activation_grad: expected "
                    f"{(BATCH_SIZE, SEQ_LEN, MLP_RATIO * D_MODEL)}, got {tuple(go.shape)}"
                )
            elif "ff_out" in layer_name:
                assert act.shape == (BATCH_SIZE, SEQ_LEN, hidden), (
                    f"{layer_name} ff_out activation: expected "
                    f"{(BATCH_SIZE, SEQ_LEN, hidden)}, got {tuple(act.shape)}"
                )
                assert go.shape == (BATCH_SIZE, SEQ_LEN, D_MODEL), (
                    f"{layer_name} ff_out pre_activation_grad: expected "
                    f"{(BATCH_SIZE, SEQ_LEN, D_MODEL)}, got {tuple(go.shape)}"
                )
        collector.remove()

    def test_buffers_reset_between_passes(self, tiny_olmo, sample_ids):
        """A second context block on a different batch gives different activations."""
        capture1 = _Capture()
        collector = HookManager(tiny_olmo, config=_make_hook_cfg(), callbacks=[capture1])
        _run_forward_backward(tiny_olmo, sample_ids, collector, capture1)

        torch.manual_seed(SEED + 1)
        ids2 = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        capture2 = _Capture()
        collector._callbacks = [capture2]
        _run_forward_backward(tiny_olmo, ids2, collector, capture2)

        rec1, rec2 = capture1.records[0], capture2.records[0]
        any_differ = any(
            not torch.allclose(
                rec1.gradient.data[l].activation,
                rec2.gradient.data[l].activation,
            )
            for l in rec1.gradient.data
            if l in rec2.gradient.data
        )
        assert any_differ, "Activations are identical across different inputs -- buffers not reset"
        collector.remove()

    def test_numeric_agreement_with_reference(self, tiny_olmo, sample_ids):
        """Two independent models built from the same seed must produce identical a/g."""
        capture_a = _Capture()
        collector_a = HookManager(tiny_olmo, config=_make_hook_cfg(), callbacks=[capture_a])
        _run_forward_backward(tiny_olmo, sample_ids, collector_a, capture_a)
        collector_a.remove()

        # Model B -- fresh instance, same seed
        torch.manual_seed(SEED)
        cfg = ModelConfig(
            d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS,
            vocab_size=VOCAB_SIZE, max_sequence_length=SEQ_LEN,
            mlp_ratio=MLP_RATIO, activation_type=ActivationType.swiglu,
            rope=True, flash_attention=False, include_bias=False,
            weight_tying=True,
        )
        model_b = OLMo(cfg)
        model_b.train()
        capture_b = _Capture()
        collector_b = HookManager(model_b, config=_make_hook_cfg(), callbacks=[capture_b])
        _run_forward_backward(model_b, sample_ids.clone(), collector_b, capture_b)
        collector_b.remove()

        rec_a = capture_a.records[0]
        rec_b = capture_b.records[0]
        for layer in rec_a.gradient.data:
            val_a = rec_a.gradient.data[layer]
            val_b = rec_b.gradient.data[layer]
            assert isinstance(val_a, Factorized) and isinstance(val_b, Factorized)
            assert torch.allclose(val_a.activation.float(), val_b.activation.float(),
                                  atol=ATOL, rtol=RTOL), (
                f"{layer} activation: runs with same seed diverged  "
                f"max_diff={(val_a.activation - val_b.activation).abs().max():.2e}"
            )
            assert torch.allclose(
                val_a.pre_activation_grad.float(),
                val_b.pre_activation_grad.float(),
                atol=ATOL, rtol=RTOL,
            ), (
                f"{layer} pre_activation_grad: runs with same seed diverged  "
                f"max_diff={(val_a.pre_activation_grad - val_b.pre_activation_grad).abs().max():.2e}"
            )

    def test_remove_clears_layer_names(self, tiny_olmo):
        """After remove(), layer_names is empty and hooks no longer fire."""
        collector = HookManager(tiny_olmo, config=_make_hook_cfg())
        collector.remove()
        assert len(collector.layer_names) == 0


# ---------------------------------------------------------------------------
# FSDP tests (2-process, gloo backend, CPU-safe)
# ---------------------------------------------------------------------------


def _fsdp_worker(
    rank: int,
    world_size: int,
    result_queue: torch.multiprocessing.Queue,
    rendezvous_path: str,
) -> None:
    """Worker function run in each spawned process for FSDP tests."""
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import ShardingStrategy

    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous_path}",
        rank=rank,
        world_size=world_size,
    )

    try:
        ids = _make_sample_ids()
        torch.manual_seed(SEED)
        cfg = ModelConfig(
            d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS,
            vocab_size=VOCAB_SIZE, max_sequence_length=SEQ_LEN,
            mlp_ratio=MLP_RATIO, activation_type=ActivationType.swiglu,
            rope=True, flash_attention=False, include_bias=False,
            weight_tying=True,
        )
        model = OLMo(cfg)
        model.train()

        # Register hooks BEFORE FSDP wrapping so they survive module restructuring.
        capture = _Capture()
        hook_cfg = HookManagerConfig(linear_io=OLMO_MLP_PATTERNS)
        collector = HookManager(model, config=hook_cfg, callbacks=[capture])

        fsdp_model = FSDP(
            model,
            device_id=torch.device("cpu"),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            use_orig_params=True,  # preserves original modules (and their hooks)
        )

        fsdp_model.zero_grad()
        with collector.collect():
            out = fsdp_model(input_ids=ids)
            shift_logits = out.logits[:, :-1, :].contiguous()
            shift_labels = ids[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.shape[-1]),
                shift_labels.view(-1),
            )
            loss.backward()

        dist.barrier()

        # Only rank 0 validates and reports results.
        if rank == 0:
            passed = True

            # Record count
            passed = passed and len(capture.records) == 1
            if passed:
                rec = capture.records[0]
                # Layer count
                passed = passed and len(rec.gradient.data) == N_LAYERS * 2

                # Finiteness
                for val in rec.gradient.data.values():
                    if isinstance(val, Factorized):
                        passed = passed and torch.isfinite(val.activation).all().item()
                        passed = passed and torch.isfinite(val.pre_activation_grad).all().item()

                # Numeric agreement with single-GPU reference
                ref_cfg = ModelConfig(
                    d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS,
                    vocab_size=VOCAB_SIZE, max_sequence_length=SEQ_LEN,
                    mlp_ratio=MLP_RATIO, activation_type=ActivationType.swiglu,
                    rope=True, flash_attention=False, include_bias=False,
                    weight_tying=True,
                )
                torch.manual_seed(SEED)
                ref_model = OLMo(ref_cfg)
                ref_model.train()
                ref_capture = _Capture()
                ref_collector = HookManager(
                    ref_model,
                    config=HookManagerConfig(linear_io=OLMO_MLP_PATTERNS),
                    callbacks=[ref_capture],
                )
                _run_forward_backward(ref_model, ids.clone(), ref_collector, ref_capture)
                ref_collector.remove()

                ref_rec = ref_capture.records[0]
                for layer in ref_rec.gradient.data:
                    if layer not in rec.gradient.data:
                        passed = False
                        continue
                    val = rec.gradient.data[layer]
                    ref_val = ref_rec.gradient.data[layer]
                    if isinstance(val, Factorized) and isinstance(ref_val, Factorized):
                        if not torch.allclose(ref_val.activation.float(), val.activation.float(),
                                              atol=ATOL, rtol=RTOL):
                            passed = False
                        if not torch.allclose(
                            ref_val.pre_activation_grad.float(),
                            val.pre_activation_grad.float(),
                            atol=ATOL, rtol=RTOL,
                        ):
                            passed = False

            result_queue.put(passed)

    finally:
        collector.remove()
        dist.destroy_process_group()


class TestOLMoCollectorFSDP:
    """HookManager on an FSDP-wrapped OLMo model.

    Uses ``torch.multiprocessing.spawn`` (gloo, CPU) so no ``torchrun`` or
    special launcher is needed -- plain ``pytest`` is sufficient.
    """

    def test_fsdp_collector_populates_and_agrees(self):
        """Collector fires correctly under FSDP and agrees with single-GPU reference."""
        import torch.multiprocessing as mp

        if not _can_bind_localhost():
            pytest.skip("local socket binds are not permitted in this environment")

        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()
        fd, rendezvous_path = tempfile.mkstemp()
        os.close(fd)
        try:
            mp.spawn(
                _fsdp_worker,
                args=(2, result_queue, rendezvous_path),
                nprocs=2,
                join=True,
            )
        finally:
            if os.path.exists(rendezvous_path):
                os.unlink(rendezvous_path)

        assert not result_queue.empty(), "rank-0 worker did not report a result"
        passed = result_queue.get()
        assert passed, (
            "FSDP worker reported failure: check layer count, finiteness, "
            "or numeric agreement with the single-GPU reference"
        )

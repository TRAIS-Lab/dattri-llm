"""Unit tests for GradientCollector on OLMo models.

Covers two parallelism settings:

* **Single-process** — plain ``OLMo`` on CPU, no distributed setup required.
* **FSDP** — ``FullyShardedDataParallel``-wrapped ``OLMo`` launched via
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
from typing import Dict

import pytest
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Optional OLMo import — skip the whole file if not installed
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

from dattri_llm.gradient.collector import GradientCollector  # noqa: E402
from dattri_llm.gradient.trainers.olmo import OLMO_MLP_PATTERNS  # noqa: E402

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
    """A fixed batch of token IDs — deterministic across all test runs."""
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


# ---------------------------------------------------------------------------
# Helper: one forward+backward inside the collector context
# ---------------------------------------------------------------------------


def _run_forward_backward(
    model: torch.nn.Module,
    ids: torch.Tensor,
    collector: GradientCollector,
) -> Dict[str, Dict[str, torch.Tensor]]:
    model.zero_grad()
    with collector:
        out = model(input_ids=ids)
        shift_logits = out.logits[:, :-1, :].contiguous()
        shift_labels = ids[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1),
        )
        loss.backward()
    return collector.get_activations_and_grad_outputs()


# ---------------------------------------------------------------------------
# Single-process tests
# ---------------------------------------------------------------------------


class TestOLMoCollectorSingle:
    """GradientCollector on a plain OLMo model (no parallelism)."""

    def test_layers_hooked(self, tiny_olmo, sample_ids):
        """Collector registers exactly N_LAYERS * 2 hooks (ff_proj + ff_out each)."""
        collector = GradientCollector(tiny_olmo, name_patterns=OLMO_MLP_PATTERNS)
        assert len(collector.layer_names) == N_LAYERS * 2, (
            f"Expected {N_LAYERS * 2} layers, got {len(collector.layer_names)}: "
            f"{collector.layer_names}"
        )
        collector.remove()

    def test_correct_layer_names(self, tiny_olmo):
        """Hooked names follow the pattern transformer.blocks.<i>.ff_proj/ff_out."""
        collector = GradientCollector(tiny_olmo, name_patterns=OLMO_MLP_PATTERNS)
        for name in collector.layer_names:
            assert "transformer.blocks." in name, f"Unexpected layer: {name}"
            assert name.endswith(("ff_proj", "ff_out")), f"Unexpected suffix: {name}"
        collector.remove()

    def test_lm_head_not_hooked(self, tiny_olmo):
        """Top-level transformer.ff_out (lm-head) must not appear in the hook set."""
        collector = GradientCollector(tiny_olmo, name_patterns=OLMO_MLP_PATTERNS)
        assert "transformer.ff_out" not in collector.layer_names, (
            "lm-head (transformer.ff_out) should not be hooked"
        )
        collector.remove()

    def test_buffers_populated_after_pass(self, tiny_olmo, sample_ids):
        """Both activation and grad_output are non-None after forward+backward."""
        collector = GradientCollector(tiny_olmo, name_patterns=OLMO_MLP_PATTERNS)
        ag = _run_forward_backward(tiny_olmo, sample_ids, collector)
        for layer, entry in ag.items():
            assert entry["activation"] is not None, f"{layer}: activation is None"
            assert entry["grad_output"] is not None, f"{layer}: grad_output is None"
        collector.remove()

    def test_all_values_finite(self, tiny_olmo, sample_ids):
        """No NaN or Inf in any collected tensor."""
        collector = GradientCollector(tiny_olmo, name_patterns=OLMO_MLP_PATTERNS)
        ag = _run_forward_backward(tiny_olmo, sample_ids, collector)
        for layer, entry in ag.items():
            for key in ("activation", "grad_output"):
                assert torch.isfinite(entry[key]).all(), (
                    f"{layer}[{key}] contains non-finite values"
                )
        collector.remove()

    def test_swiglu_activation_shapes(self, tiny_olmo, sample_ids):
        """Verify SwiGLU-specific tensor shapes for ff_proj and ff_out.

        Expected (with output_multiplier=0.5):
            ff_proj  activation : (B, T, D_MODEL)
            ff_proj  grad_output: (B, T, MLP_RATIO * D_MODEL)
            ff_out   activation : (B, T, MLP_RATIO * D_MODEL // 2)
            ff_out   grad_output: (B, T, D_MODEL)
        """
        collector = GradientCollector(tiny_olmo, name_patterns=OLMO_MLP_PATTERNS)
        ag = _run_forward_backward(tiny_olmo, sample_ids, collector)
        hidden = MLP_RATIO * D_MODEL // 2  # SwiGLU halves before ff_out

        for layer, entry in ag.items():
            act, go = entry["activation"], entry["grad_output"]
            if "ff_proj" in layer:
                assert act.shape == (BATCH_SIZE, SEQ_LEN, D_MODEL), (
                    f"{layer} ff_proj activation: expected {(BATCH_SIZE, SEQ_LEN, D_MODEL)}, "
                    f"got {tuple(act.shape)}"
                )
                assert go.shape == (BATCH_SIZE, SEQ_LEN, MLP_RATIO * D_MODEL), (
                    f"{layer} ff_proj grad_output: expected "
                    f"{(BATCH_SIZE, SEQ_LEN, MLP_RATIO * D_MODEL)}, got {tuple(go.shape)}"
                )
            elif "ff_out" in layer:
                assert act.shape == (BATCH_SIZE, SEQ_LEN, hidden), (
                    f"{layer} ff_out activation: expected {(BATCH_SIZE, SEQ_LEN, hidden)}, "
                    f"got {tuple(act.shape)}"
                )
                assert go.shape == (BATCH_SIZE, SEQ_LEN, D_MODEL), (
                    f"{layer} ff_out grad_output: expected {(BATCH_SIZE, SEQ_LEN, D_MODEL)}, "
                    f"got {tuple(go.shape)}"
                )
        collector.remove()

    def test_buffers_reset_between_passes(self, tiny_olmo, sample_ids):
        """A second context block on a different batch gives different activations."""
        collector = GradientCollector(tiny_olmo, name_patterns=OLMO_MLP_PATTERNS)
        ag1 = _run_forward_backward(tiny_olmo, sample_ids, collector)

        torch.manual_seed(SEED + 1)
        ids2 = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        ag2 = _run_forward_backward(tiny_olmo, ids2, collector)

        any_differ = any(
            not torch.allclose(ag1[l]["activation"], ag2[l]["activation"])
            for l in ag1
            if l in ag2
        )
        assert any_differ, "Activations are identical across different inputs — buffers not reset"
        collector.remove()

    def test_numeric_agreement_with_reference(self, tiny_olmo, sample_ids):
        """Two independent models built from the same seed must produce identical a/g."""
        # Model A (the fixture)
        collector_a = GradientCollector(tiny_olmo, name_patterns=OLMO_MLP_PATTERNS)
        ag_a = _run_forward_backward(tiny_olmo, sample_ids, collector_a)
        collector_a.remove()

        # Model B — fresh instance, same seed
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
        collector_b = GradientCollector(model_b, name_patterns=OLMO_MLP_PATTERNS)
        ag_b = _run_forward_backward(model_b, sample_ids.clone(), collector_b)
        collector_b.remove()

        for layer in ag_a:
            for key in ("activation", "grad_output"):
                assert torch.allclose(
                    ag_a[layer][key].float(),
                    ag_b[layer][key].float(),
                    atol=ATOL,
                    rtol=RTOL,
                ), (
                    f"{layer}[{key}]: runs with the same seed diverged  "
                    f"max_diff={(ag_a[layer][key] - ag_b[layer][key]).abs().max():.2e}"
                )

    def test_remove_clears_layer_names(self, tiny_olmo):
        """After remove(), layer_names is empty and hooks no longer fire."""
        collector = GradientCollector(tiny_olmo, name_patterns=OLMO_MLP_PATTERNS)
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

    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0")
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
        collector = GradientCollector(model, name_patterns=OLMO_MLP_PATTERNS)

        fsdp_model = FSDP(
            model,
            device_id=torch.device("cpu"),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            use_orig_params=True,  # preserves original modules (and their hooks)
        )

        fsdp_model.zero_grad()
        with collector:
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
            ag = collector.get_activations_and_grad_outputs()
            passed = True

            # Layer count
            passed = passed and len(ag) == N_LAYERS * 2

            # Finiteness
            for entry in ag.values():
                for key in ("activation", "grad_output"):
                    passed = passed and torch.isfinite(entry[key]).all().item()

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
            ref_collector = GradientCollector(ref_model, name_patterns=OLMO_MLP_PATTERNS)
            ref_ag = _run_forward_backward(ref_model, ids.clone(), ref_collector)
            ref_collector.remove()

            for layer in ref_ag:
                if layer not in ag:
                    passed = False
                    continue
                for key in ("activation", "grad_output"):
                    if not torch.allclose(
                        ref_ag[layer][key].float(),
                        ag[layer][key].float(),
                        atol=ATOL,
                        rtol=RTOL,
                    ):
                        passed = False

            result_queue.put(passed)

    finally:
        collector.remove()
        dist.destroy_process_group()


class TestOLMoCollectorFSDP:
    """GradientCollector on an FSDP-wrapped OLMo model.

    Uses ``torch.multiprocessing.spawn`` (gloo, CPU) so no ``torchrun`` or
    special launcher is needed — plain ``pytest`` is sufficient.
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

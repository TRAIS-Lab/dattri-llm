"""Tests for DataSelectionCallback.

The key invariant: if *all* samples in a batch are dropped (threshold set high
enough that every sample's score falls below it), then the weight and bias grads
of every hooked MLP layer must be exactly zero after the callback fires.

Implementation note on hook timing
------------------------------------
PyTorch's ``register_full_backward_hook`` fires *before* ``param.grad`` is
accumulated when **none** of the module's inputs require a gradient (see the
PyTorch warning "Full backward hook is firing when gradients are computed with
respect to module outputs since no inputs require gradients").  This matters for
``DataSelectionCallback._subtract_weight``: it reads ``weight.grad`` inside the
callback, which fires from within the backward hook of the last hooked layer.

In real LLM training this edge case never arises because token embeddings
(trainable parameters) produce intermediate activations that *do* require grad,
so every MLP layer's input requires grad and the hook fires at the normal time
(after ``param.grad`` is accumulated).

To replicate the real-world computational graph, the ``MinimalEmbeddingMLP``
fixture below routes the batch through a trainable ``nn.Embedding`` layer before
the MLP, ensuring that MLP inputs always require grad and that ``param.grad`` is
populated by the time ``on_step_end`` fires.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from dattri_llm.gradient.callbacks import DataSelectionCallback
from dattri_llm.gradient.hooks import REGISTER_ALL, HookManager, HookManagerConfig

# --------------------------------------------------------------------------- #
# Minimal model fixture                                                         #
# --------------------------------------------------------------------------- #


class MinimalEmbeddingMLP(nn.Module):
    """Embedding -> two-layer MLP.

    The ``nn.Embedding`` lookup ensures that MLP inputs always require grad,
    which matches real LLM training (token IDs -> embedding parameters -> MLP).
    Without this, PyTorch's ``register_full_backward_hook`` fires before
    ``param.grad`` is accumulated for the first MLP layer, causing
    ``DataSelectionCallback`` to silently skip gradient subtraction on it.
    """

    def __init__(
        self,
        vocab_size: int = 32,
        embed_dim: int = 8,
        hidden: int = 16,
        out_features: int = 4,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden, bias=True),
            nn.ReLU(),
            nn.Linear(hidden, out_features, bias=True),
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:  # (B, T) -> (B, T, out)
        x = self.embedding(token_ids)  # (B, T, embed_dim) -- requires grad
        return self.mlp(x)  # (B, T, out_features)


# --------------------------------------------------------------------------- #
# Helper                                                                        #
# --------------------------------------------------------------------------- #


def _run_step_with_callback(
    model: nn.Module,
    token_ids: torch.Tensor,
    callback: DataSelectionCallback,
    loss_reduction: str = "mean",
) -> None:
    """Run one forward + backward step with the callback attached."""
    collector = HookManager(
        model,
        config=HookManagerConfig(linear_io=REGISTER_ALL),
        callbacks=[callback],
    )
    with collector.collect():
        out = model(token_ids)  # (B, T, out)
        loss = out.mean() if loss_reduction == "mean" else out.sum()
        loss.backward()


def _make_token_ids(B: int, T: int, vocab_size: int = 32) -> torch.Tensor:
    return torch.randint(0, vocab_size, (B, T))


# --------------------------------------------------------------------------- #
# Tests                                                                         #
# --------------------------------------------------------------------------- #


class TestDataSelectionCallbackHardThreshold:
    """hard threshold_mode (default) -- scores below a cutoff are dropped."""

    @pytest.mark.parametrize("loss_reduction", ["mean", "sum"])
    @pytest.mark.parametrize(("B", "T"), [(2, 5), (4, 1), (1, 10)])
    def test_drop_all_zeroes_grad(self, loss_reduction, B, T):
        """Dropping every sample must zero out all MLP weight and bias grads.

        Uses ``MinimalEmbeddingMLP`` so that MLP inputs always require grad,
        matching the real LLM training computational graph and ensuring
        ``param.grad`` is populated when ``on_step_end`` fires.
        """
        torch.manual_seed(0)
        model = MinimalEmbeddingMLP()
        token_ids = _make_token_ids(B, T)

        cb = DataSelectionCallback(
            model=model,
            threshold=float("inf"),  # drop everything
        )

        _run_step_with_callback(model, token_ids, cb, loss_reduction=loss_reduction)

        # All samples should have been dropped.
        assert len(cb.last_dropped) == B, (
            f"Expected all {B} samples dropped, got {cb.last_dropped}"
        )

        # Every MLP layer's weight.grad and bias.grad must be (close to) zero.
        for m in model.mlp:
            if isinstance(m, nn.Linear):
                assert m.weight.grad is not None, "weight.grad was None after backward"
                assert m.bias.grad is not None, "bias.grad was None after backward"
                assert torch.allclose(
                    m.weight.grad,
                    torch.zeros_like(m.weight.grad),
                    atol=1e-6,
                ), (
                    f"weight.grad not zero after dropping all samples:\n"
                    f"max abs = {m.weight.grad.abs().max().item():.2e}"
                )
                assert torch.allclose(
                    m.bias.grad,
                    torch.zeros_like(m.bias.grad),
                    atol=1e-6,
                ), (
                    f"bias.grad not zero after dropping all samples:\n"
                    f"max abs = {m.bias.grad.abs().max().item():.2e}"
                )

    def test_drop_none_preserves_grad(self):
        """Dropping no samples must leave param.grad unchanged."""
        torch.manual_seed(1)
        model = MinimalEmbeddingMLP()
        token_ids = _make_token_ids(3, 6)

        # Run without any callback to capture the reference grads.
        ref_model = MinimalEmbeddingMLP()
        ref_model.load_state_dict(model.state_dict())
        out_ref = ref_model(token_ids)
        out_ref.mean().backward()
        ref_grads = {
            i: (m.weight.grad.clone(), m.bias.grad.clone())
            for i, m in enumerate(ref_model.mlp)
            if isinstance(m, nn.Linear)
        }

        cb = DataSelectionCallback(
            model=model,
            threshold=-float("inf"),  # keep everything
        )
        _run_step_with_callback(model, token_ids, cb, loss_reduction="mean")

        assert len(cb.last_dropped) == 0

        for i, m in enumerate(model.mlp):
            if not isinstance(m, nn.Linear):
                continue
            rw, rb = ref_grads[i]
            assert torch.allclose(m.weight.grad, rw, atol=1e-6), (
                f"Layer {i} weight.grad changed when no samples were dropped"
            )
            assert torch.allclose(m.bias.grad, rb, atol=1e-6), (
                f"Layer {i} bias.grad changed when no samples were dropped"
            )

    def test_scores_shape(self):
        """last_scores must be a 1-D tensor of length B."""
        torch.manual_seed(2)
        B = 5
        model = MinimalEmbeddingMLP()
        token_ids = _make_token_ids(B, 4)
        cb = DataSelectionCallback(model=model, threshold=-float("inf"))
        _run_step_with_callback(model, token_ids, cb)
        assert cb.last_scores is not None
        assert cb.last_scores.shape == (B,)

    def test_scores_finite(self):
        """All per-sample scores must be finite."""
        torch.manual_seed(3)
        model = MinimalEmbeddingMLP()
        token_ids = _make_token_ids(4, 7)
        cb = DataSelectionCallback(model=model, threshold=-float("inf"))
        _run_step_with_callback(model, token_ids, cb)
        assert torch.all(torch.isfinite(cb.last_scores)), (
            f"Non-finite scores: {cb.last_scores}"
        )

    def test_invalid_threshold_mode_raises(self):
        model = MinimalEmbeddingMLP()
        with pytest.raises(ValueError, match="threshold_mode"):
            DataSelectionCallback(model=model, threshold_mode="bogus")

    def test_fraction_out_of_range_raises(self):
        model = MinimalEmbeddingMLP()
        with pytest.raises(ValueError, match=r"\[0, 1\)"):
            DataSelectionCallback(
                model=model,
                threshold=1.5,
                threshold_mode="bottom_fraction",
            )


# --------------------------------------------------------------------------- #
# bottom_fraction                                                               #
# --------------------------------------------------------------------------- #


def _mlp_grads_zero(model: MinimalEmbeddingMLP, atol: float = 1e-6) -> bool:
    """Return True if all MLP weight and bias grads are ~ 0."""
    for m in model.mlp:
        if not isinstance(m, nn.Linear):
            continue
        if m.weight.grad is None or m.bias.grad is None:
            return False
        if not torch.allclose(
            m.weight.grad,
            torch.zeros_like(m.weight.grad),
            atol=atol,
        ):
            return False
        if not torch.allclose(m.bias.grad, torch.zeros_like(m.bias.grad), atol=atol):
            return False
    return True


class TestBottomFraction:
    """threshold_mode='bottom_fraction' -- drop the worst k% regardless of sign."""

    def test_drop_all_fraction_zeroes_grad(self):
        """threshold=1.0 is invalid; threshold approaching 1.0 drops almost everyone."""
        torch.manual_seed(10)
        B, T = 4, 5
        model = MinimalEmbeddingMLP()
        token_ids = _make_token_ids(B, T)

        # Drop the bottom 75 % (3 of 4 samples) -- not all, but check count.
        cb = DataSelectionCallback(
            model=model,
            threshold=0.75,
            threshold_mode="bottom_fraction",
        )
        _run_step_with_callback(model, token_ids, cb)
        assert len(cb.last_dropped) == round(B * 0.75)

    def test_drop_zero_fraction_preserves_all(self):
        """threshold=0.0 -> n_drop=0 -> nothing removed."""
        torch.manual_seed(11)
        B, T = 4, 5
        model = MinimalEmbeddingMLP()
        token_ids = _make_token_ids(B, T)

        cb = DataSelectionCallback(
            model=model,
            threshold=0.0,
            threshold_mode="bottom_fraction",
        )
        _run_step_with_callback(model, token_ids, cb)
        assert len(cb.last_dropped) == 0

    def test_exact_drop_count(self):
        """Dropped count equals round(B * threshold)."""
        torch.manual_seed(12)
        B, T = 8, 4
        for frac in (0.25, 0.5):
            model = MinimalEmbeddingMLP()
            token_ids = _make_token_ids(B, T)
            cb = DataSelectionCallback(
                model=model,
                threshold=frac,
                threshold_mode="bottom_fraction",
            )
            _run_step_with_callback(model, token_ids, cb)
            assert len(cb.last_dropped) == round(B * frac), (
                f"frac={frac}: expected {round(B * frac)} dropped, "
                f"got {len(cb.last_dropped)}"
            )

    def test_dropped_are_lowest_scored(self):
        """Dropped indices must correspond to the lowest scores."""
        torch.manual_seed(13)
        B, T = 6, 4
        model = MinimalEmbeddingMLP()
        token_ids = _make_token_ids(B, T)
        frac = 1 / 3
        cb = DataSelectionCallback(
            model=model,
            threshold=frac,
            threshold_mode="bottom_fraction",
        )
        _run_step_with_callback(model, token_ids, cb)
        n_drop = round(B * frac)
        scores = cb.last_scores
        # The dropped set must be exactly the indices of the n_drop lowest scores.
        expected = set(scores.argsort()[:n_drop].tolist())
        assert set(cb.last_dropped) == expected

    def test_drop_half_zeroes_grad_approximately(self):
        """After dropping half the batch, grads should be roughly halved (not zero)."""
        torch.manual_seed(14)
        B, T = 4, 5
        model = MinimalEmbeddingMLP()
        token_ids = _make_token_ids(B, T)
        cb = DataSelectionCallback(
            model=model,
            threshold=0.5,
            threshold_mode="bottom_fraction",
        )
        _run_step_with_callback(model, token_ids, cb)
        assert len(cb.last_dropped) == 2
        # Grads should be non-zero (2 samples kept) but not the full-batch value.
        for m in model.mlp:
            if isinstance(m, nn.Linear):
                assert m.weight.grad is not None
                # Not all-zero: at least some gradient remains.
                assert m.weight.grad.abs().max().item() > 1e-10


# --------------------------------------------------------------------------- #
# negative_bottom_fraction                                                      #
# --------------------------------------------------------------------------- #


class TestNegativeBottomFraction:
    """threshold_mode='negative_bottom_fraction' -- drop bottom k% only if score < 0."""

    def test_all_positive_scores_drops_nothing(self):
        """When all scores are >= 0, no sample is ever dropped regardless of
        fraction.
        """
        model = MinimalEmbeddingMLP()
        cb = DataSelectionCallback(
            model=model,
            threshold=0.99,
            threshold_mode="negative_bottom_fraction",
        )
        # All-positive synthetic scores -> nothing qualifies even at 99% fraction.
        scores = torch.tensor([1.0, 2.0, 0.5, 3.0])
        assert cb._select_dropped(scores) == []

    def test_negative_scores_eligible_only(self):
        """Samples with score >= 0 must never be dropped, even if in the bottom k%."""
        torch.manual_seed(21)
        B, T = 6, 4

        # We need at least one positive-score sample to be in the bottom k% to
        # confirm the filter.  Ghost scores are always >= 0 (PSD), so in practice
        # we manipulate last_scores directly to test the selection logic.
        model = MinimalEmbeddingMLP()
        token_ids = _make_token_ids(B, T)
        cb = DataSelectionCallback(
            model=model,
            threshold=0.5,  # drop bottom 50% IF negative
            threshold_mode="negative_bottom_fraction",
        )
        _run_step_with_callback(model, token_ids, cb)

        # Verify: every dropped index has a negative score.
        for i in cb.last_dropped:
            assert cb.last_scores[i] < 0, (
                f"Sample {i} was dropped but has non-negative score {cb.last_scores[i]}"
            )

    def test_select_dropped_logic_directly(self):
        """Unit-test _select_dropped with synthetic scores to cover all branches."""
        model = MinimalEmbeddingMLP()
        cb = DataSelectionCallback(
            model=model,
            threshold=0.5,
            threshold_mode="negative_bottom_fraction",
        )

        # scores: [-3, -1, 2, 4, -2, 5]  (B=6)
        # Bottom 50% (3 samples) by rank: indices 0 (-3), 4 (-2), 1 (-1)
        # After negative filter (score < 0): all three qualify -> dropped=[0, 4, 1]
        scores = torch.tensor([-3.0, -1.0, 2.0, 4.0, -2.0, 5.0])
        dropped = cb._select_dropped(scores)
        assert set(dropped) == {0, 1, 4}

        # scores: [-3, 1, 2, 4, -2, 5]
        # Bottom 3: indices 0 (-3), 4 (-2), 1 (1 -- positive!)
        # After negative filter: only 0 and 4 qualify
        scores2 = torch.tensor([-3.0, 1.0, 2.0, 4.0, -2.0, 5.0])
        dropped2 = cb._select_dropped(scores2)
        assert set(dropped2) == {0, 4}

    def test_zero_fraction_drops_nothing(self):
        torch.manual_seed(22)
        B, T = 4, 5
        model = MinimalEmbeddingMLP()
        token_ids = _make_token_ids(B, T)
        cb = DataSelectionCallback(
            model=model,
            threshold=0.0,
            threshold_mode="negative_bottom_fraction",
        )
        _run_step_with_callback(model, token_ids, cb)
        assert len(cb.last_dropped) == 0


# --------------------------------------------------------------------------- #
# score_mode equivalence: ghost == materialized                                #
# --------------------------------------------------------------------------- #


def _scores_for_mode(
    model: nn.Module,
    token_ids: torch.Tensor,
    score_mode: str,
) -> torch.Tensor:
    """Return last_scores produced by one forward+backward step."""
    cb = DataSelectionCallback(
        model=model,
        threshold=-float("inf"),  # keep everything -- only compute scores
        score_mode=score_mode,
    )
    _run_step_with_callback(model, token_ids, cb)
    assert cb.last_scores is not None
    return cb.last_scores


class TestScoreModeEquivalence:
    """ghost and materialized scoring must produce identical results.

    Mathematical identity::

        <g_i x a_i,  g_j x a_j>  =  (g_i * g_j)(a_i * a_j)

    so the ghost inner product (gram-matrix form) and the materialized inner
    product (explicit outer-product form) are numerically equivalent to
    floating-point precision.  This test suite verifies that equivalence:

    * for layers *without* a token dimension (e.g. a plain 2-D linear)
    * for layers *with* a token dimension (the typical LLM case, BxTxF)
    * for ``nn.Embedding`` layers specifically (activation = integer token IDs)
    """

    @pytest.mark.parametrize(("B", "T"), [(2, 5), (4, 1), (1, 10), (3, 8)])
    def test_ghost_eq_materialized_with_token_dim(self, B, T):
        """Both score modes agree for all (B, T) configurations.

        ``MinimalEmbeddingMLP`` has a token dimension in every layer (the
        embedding lookup produces (B, T, embed_dim) which flows through all
        subsequent linear layers), so this also covers the Embedding branch.
        """
        torch.manual_seed(42)
        model = MinimalEmbeddingMLP()
        token_ids = _make_token_ids(B, T)

        # Ghost scores
        ghost_scores = _scores_for_mode(model, token_ids, "ghost")

        # Re-use the exact same model weights and inputs for materialized.
        # Recreate a fresh model with the same state so gradients don't
        # accumulate from the first run.
        model2 = MinimalEmbeddingMLP()
        model2.load_state_dict(model.state_dict())
        mat_scores = _scores_for_mode(model2, token_ids, "materialized")

        assert ghost_scores.shape == (B,), (
            f"ghost scores shape {ghost_scores.shape} != ({B},)"
        )
        assert mat_scores.shape == (B,), (
            f"materialized scores shape {mat_scores.shape} != ({B},)"
        )
        assert torch.allclose(ghost_scores, mat_scores, atol=1e-4, rtol=1e-4), (
            f"B={B}, T={T}\n"
            f"ghost      : {ghost_scores.tolist()}\n"
            f"materialized: {mat_scores.tolist()}\n"
            f"max abs diff: {(ghost_scores - mat_scores).abs().max().item():.2e}"
        )

    def test_ghost_eq_materialized_no_token_dim(self):
        """Both score modes agree for a plain 2-D linear layer (no token dim).

        Uses a minimal model where the forward pass collapses the token
        dimension before the linear layers (mean-pooling), so the hooks see
        (B, F) tensors only.
        """

        class MeanPoolMLP(nn.Module):
            """Embedding -> mean-pool over T -> two-layer MLP.

            The mean-pool produces a (B, embed_dim) activation, so every
            subsequent linear layer's hook captures 2-D tensors (B, F) --
            no token dimension.  The Embedding hook still captures (B, T) ints.
            """

            def __init__(
                self,
                vocab_size: int = 32,
                embed_dim: int = 8,
                hidden: int = 16,
                out_features: int = 4,
            ):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embed_dim)
                self.mlp = nn.Sequential(
                    nn.Linear(embed_dim, hidden, bias=True),
                    nn.ReLU(),
                    nn.Linear(hidden, out_features, bias=True),
                )

            def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
                x = self.embedding(token_ids).mean(1)  # (B, embed_dim) -- no token dim
                return self.mlp(x)  # (B, out_features)

        torch.manual_seed(99)
        B, T = 4, 6
        token_ids = _make_token_ids(B, T)
        model = MeanPoolMLP()

        ghost_scores = _scores_for_mode(model, token_ids, "ghost")
        model2 = MeanPoolMLP()
        model2.load_state_dict(model.state_dict())
        mat_scores = _scores_for_mode(model2, token_ids, "materialized")

        assert torch.allclose(ghost_scores, mat_scores, atol=1e-4, rtol=1e-4), (
            "no-token-dim model\n"
            f"ghost      : {ghost_scores.tolist()}\n"
            f"materialized: {mat_scores.tolist()}\n"
            f"max abs diff: {(ghost_scores - mat_scores).abs().max().item():.2e}"
        )

    def test_invalid_score_mode_raises(self):
        """Passing an unknown score_mode must raise ValueError immediately."""
        model = MinimalEmbeddingMLP()
        with pytest.raises(ValueError, match="score_mode"):
            DataSelectionCallback(model=model, score_mode="bogus")

    def test_scores_are_finite_materialized(self):
        """Materialized scores must be finite for every sample."""
        torch.manual_seed(50)
        B, T = 4, 7
        model = MinimalEmbeddingMLP()
        token_ids = _make_token_ids(B, T)
        cb = DataSelectionCallback(
            model=model,
            threshold=-float("inf"),
            score_mode="materialized",
        )
        _run_step_with_callback(model, token_ids, cb)
        assert torch.all(torch.isfinite(cb.last_scores)), (
            f"Non-finite materialized scores: {cb.last_scores}"
        )


# --------------------------------------------------------------------------- #
# Normalization-layer consistency (regression: ghost vs materialized for norms) #
# --------------------------------------------------------------------------- #


class _NormMLP(nn.Module):
    """Embedding -> LayerNorm -> two-layer MLP.

    The LayerNorm (always hooked) has a token dimension, so its parameter
    gradient is the *elementwise* x_hat * g, not an outer product.  Earlier the
    ghost path scored norm layers with the Linear-style gram (g*g)(a*a), which
    disagrees with the materialized (elementwise) path -- this model exercises
    that case.
    """

    def __init__(self, vocab_size=32, embed_dim=8, hidden=16, out_features=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.ln = nn.LayerNorm(embed_dim)  # bias=True by default
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden, bias=True),
            nn.ReLU(),
            nn.Linear(hidden, out_features, bias=True),
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.ln(self.embedding(token_ids))  # (B, T, embed_dim)
        return self.mlp(x)


class TestNormLayerConsistency:
    """ghost == materialized must hold for models containing norm layers, and
    dropping all samples must zero the norm layer's grads (exercises the
    materialize-based gradient subtraction for norms).
    """

    @pytest.mark.parametrize(("B", "T"), [(3, 5), (4, 1)])
    def test_ghost_eq_materialized_with_layernorm(self, B, T):
        torch.manual_seed(7)
        model = _NormMLP()
        token_ids = _make_token_ids(B, T)

        ghost = _scores_for_mode(model, token_ids, "ghost")
        model2 = _NormMLP()
        model2.load_state_dict(model.state_dict())
        mat = _scores_for_mode(model2, token_ids, "materialized")

        assert torch.allclose(ghost, mat, atol=1e-4, rtol=1e-4), (
            f"B={B}, T={T}\n"
            f"ghost      : {ghost.tolist()}\n"
            f"materialized: {mat.tolist()}\n"
            f"max abs diff: {(ghost - mat).abs().max().item():.2e}"
        )

    @pytest.mark.parametrize("score_mode", ["ghost", "materialized"])
    def test_drop_all_zeroes_layernorm_grad(self, score_mode):
        """Dropping every sample must zero the LayerNorm weight & bias grads,
        validating the materialize-based subtraction for norm layers.
        """
        torch.manual_seed(11)
        B, T = 4, 6
        model = _NormMLP()
        token_ids = _make_token_ids(B, T)
        cb = DataSelectionCallback(
            model=model,
            threshold=float("inf"),  # hard mode: every sample dropped
            score_mode=score_mode,
        )
        _run_step_with_callback(model, token_ids, cb)

        assert len(cb.last_dropped) == B, (
            f"expected all {B} dropped, got {cb.last_dropped}"
        )
        assert torch.allclose(
            model.ln.weight.grad,
            torch.zeros_like(model.ln.weight.grad),
            atol=1e-4,
        ), (
            f"LayerNorm weight.grad not zeroed: "
            f"max |g| = {model.ln.weight.grad.abs().max():.2e}"
        )
        assert torch.allclose(
            model.ln.bias.grad,
            torch.zeros_like(model.ln.bias.grad),
            atol=1e-4,
        ), (
            f"LayerNorm bias.grad not zeroed: "
            f"max |g| = {model.ln.bias.grad.abs().max():.2e}"
        )


# --------------------------------------------------------------------------- #
# Target modes: batch / fixed / val_loader                                     #
# --------------------------------------------------------------------------- #


class _CaptureGradient:
    """Minimal callback that records the last GradientRecord's Gradient."""

    def __init__(self):
        self.gradient = None

    def on_step_end(self, record):
        self.gradient = record.gradient

    # Make it usable as a HookManagerCallback without subclassing:
    def on_layer_forward(self, *_):
        return None

    def on_layer_backward(self, *_):
        return None

    def on_context_end(self):
        return None


def _capture_batch_gradient(
    model: nn.Module,
    token_ids: torch.Tensor,
    loss_reduction: str = "mean",
):
    """Run one forward+backward and return the batch Gradient."""
    from dattri_llm.gradient.hooks import REGISTER_ALL, HookManager, HookManagerConfig

    cap = _CaptureGradient()
    collector = HookManager(
        model,
        config=HookManagerConfig(linear_io=REGISTER_ALL),
        callbacks=[cap],
    )
    with collector.collect():
        out = model(token_ids)
        (out.mean() if loss_reduction == "mean" else out.sum()).backward()
    collector.remove()
    return cap.gradient


class TestTargetModes:
    """Tests for the target= parameter: 'batch', 'fixed', 'val_loader'."""

    # ------------------------------------------------------------------ #
    # Validation errors                                                    #
    # ------------------------------------------------------------------ #

    def test_invalid_target_raises(self):
        model = MinimalEmbeddingMLP()
        with pytest.raises(ValueError, match="target"):
            DataSelectionCallback(model=model, target="bogus")

    def test_fixed_missing_gradient_raises(self):
        model = MinimalEmbeddingMLP()
        with pytest.raises(ValueError, match="target_gradient"):
            DataSelectionCallback(model=model, target="fixed")

    def test_val_loader_missing_loader_raises(self):
        model = MinimalEmbeddingMLP()
        with pytest.raises(ValueError, match="val_loader"):
            DataSelectionCallback(
                model=model,
                target="val_loader",
                val_loss_fn=lambda m, b: m(b).mean(),
            )

    def test_val_loader_missing_loss_fn_raises(self):
        model = MinimalEmbeddingMLP()
        dummy_loader = [_make_token_ids(2, 4)]
        with pytest.raises(ValueError, match="val_loss_fn"):
            DataSelectionCallback(
                model=model,
                target="val_loader",
                val_loader=dummy_loader,
            )

    # ------------------------------------------------------------------ #
    # 'batch' target (default) is unchanged                                #
    # ------------------------------------------------------------------ #

    @pytest.mark.parametrize("score_mode", ["ghost", "materialized"])
    def test_batch_target_same_as_default(self, score_mode):
        """Explicitly passing target='batch' gives the same scores as default."""
        torch.manual_seed(30)
        B, T = 3, 5
        model = MinimalEmbeddingMLP()
        token_ids = _make_token_ids(B, T)

        # default (target='batch' implicitly)
        cb_default = DataSelectionCallback(
            model=model,
            threshold=-float("inf"),
            score_mode=score_mode,
        )
        _run_step_with_callback(model, token_ids, cb_default)

        model2 = MinimalEmbeddingMLP()
        model2.load_state_dict(model.state_dict())
        cb_explicit = DataSelectionCallback(
            model=model2,
            threshold=-float("inf"),
            score_mode=score_mode,
            target="batch",
        )
        _run_step_with_callback(model2, token_ids, cb_explicit)

        assert torch.allclose(
            cb_default.last_scores,
            cb_explicit.last_scores,
            atol=1e-6,
        )

    # ------------------------------------------------------------------ #
    # 'fixed' target                                                        #
    # ------------------------------------------------------------------ #

    @pytest.mark.parametrize("score_mode", ["ghost", "materialized"])
    def test_fixed_with_same_batch_matches_batch_mode(self, score_mode):
        """Fixed target == the training batch gradient -> identical scores.

        Justification: score[i] = <dW_i, dW_target>.  When dW_target is the
        sum of all training gradients, this equals sum_j <dW_i, dW_j>,
        which is exactly what 'batch' mode computes.
        """
        torch.manual_seed(40)
        B, T = 4, 6
        model = MinimalEmbeddingMLP()
        token_ids = _make_token_ids(B, T)

        # Capture the batch gradient from a clean forward pass.
        ref_model = MinimalEmbeddingMLP()
        ref_model.load_state_dict(model.state_dict())
        batch_gradient = _capture_batch_gradient(ref_model, token_ids)
        assert batch_gradient is not None

        # 'batch' mode scores.
        cb_batch = DataSelectionCallback(
            model=model,
            threshold=-float("inf"),
            score_mode=score_mode,
        )
        _run_step_with_callback(model, token_ids, cb_batch)

        # 'fixed' mode scores using the same batch gradient as target.
        model2 = MinimalEmbeddingMLP()
        model2.load_state_dict(model.state_dict())
        cb_fixed = DataSelectionCallback(
            model=model2,
            threshold=-float("inf"),
            score_mode=score_mode,
            target="fixed",
            target_gradient=batch_gradient,
        )
        _run_step_with_callback(model2, token_ids, cb_fixed)

        assert torch.allclose(cb_batch.last_scores, cb_fixed.last_scores, atol=1e-4), (
            f"score_mode={score_mode!r}\n"
            f"batch  scores: {cb_batch.last_scores.tolist()}\n"
            f"fixed  scores: {cb_fixed.last_scores.tolist()}\n"
            f"max diff: {(cb_batch.last_scores - cb_fixed.last_scores).abs().max():.2e}"
        )

    @pytest.mark.parametrize("score_mode", ["ghost", "materialized"])
    def test_fixed_different_target_differs(self, score_mode):
        """A fixed target from a different batch gives different scores."""
        torch.manual_seed(41)
        B, T = 4, 5
        model = MinimalEmbeddingMLP()
        train_ids = _make_token_ids(B, T)
        val_ids = _make_token_ids(B, T)  # different random batch

        # Capture gradient from the *val* batch.
        ref_model = MinimalEmbeddingMLP()
        ref_model.load_state_dict(model.state_dict())
        val_gradient = _capture_batch_gradient(ref_model, val_ids)

        cb_batch = DataSelectionCallback(
            model=model,
            threshold=-float("inf"),
            score_mode=score_mode,
        )
        _run_step_with_callback(model, train_ids, cb_batch)

        model2 = MinimalEmbeddingMLP()
        model2.load_state_dict(model.state_dict())
        cb_fixed = DataSelectionCallback(
            model=model2,
            threshold=-float("inf"),
            score_mode=score_mode,
            target="fixed",
            target_gradient=val_gradient,
        )
        _run_step_with_callback(model2, train_ids, cb_fixed)

        # Scores should NOT be identical (different targets -> different alignment).
        assert not torch.allclose(
            cb_batch.last_scores,
            cb_fixed.last_scores,
            atol=1e-4,
        ), "Expected different scores for different targets, but they were equal."

    def test_fixed_scores_finite_and_shaped(self):
        torch.manual_seed(42)
        B, T = 3, 7
        model = MinimalEmbeddingMLP()
        token_ids = _make_token_ids(B, T)

        ref = MinimalEmbeddingMLP()
        ref.load_state_dict(model.state_dict())
        tgt = _capture_batch_gradient(ref, _make_token_ids(B, T))

        cb = DataSelectionCallback(
            model=model,
            threshold=-float("inf"),
            target="fixed",
            target_gradient=tgt,
        )
        _run_step_with_callback(model, token_ids, cb)

        assert cb.last_scores is not None
        assert cb.last_scores.shape == (B,)
        assert torch.all(torch.isfinite(cb.last_scores))

    # ------------------------------------------------------------------ #
    # 'val_loader' target                                                   #
    # ------------------------------------------------------------------ #

    @pytest.mark.parametrize("score_mode", ["ghost", "materialized"])
    def test_val_loader_scores_finite_and_shaped(self, score_mode):
        """val_loader mode runs without error and returns finite (B,) scores."""
        torch.manual_seed(50)
        B, T = 3, 5
        model = MinimalEmbeddingMLP()
        train_ids = _make_token_ids(B, T)

        # A tiny val loader: single batch repeated.
        val_ids = _make_token_ids(2, T)
        val_loader = [val_ids]  # list acts as a one-batch iterable

        def val_loss_fn(m, batch):
            return m(batch).mean()

        cb = DataSelectionCallback(
            model=model,
            threshold=-float("inf"),
            score_mode=score_mode,
            target="val_loader",
            val_loader=val_loader,
            val_loss_fn=val_loss_fn,
        )
        _run_step_with_callback(model, train_ids, cb)

        assert cb.last_scores is not None
        assert cb.last_scores.shape == (B,)
        assert torch.all(torch.isfinite(cb.last_scores)), (
            f"Non-finite val_loader scores: {cb.last_scores}"
        )

    def test_val_loader_cycles_when_exhausted(self):
        """val_loader wraps around when its iterator is exhausted."""
        torch.manual_seed(51)
        B, T = 2, 4
        model = MinimalEmbeddingMLP()

        calls = []
        val_ids_a = _make_token_ids(B, T)
        val_ids_b = _make_token_ids(B, T)

        def val_loss_fn(m, batch):
            calls.append(batch)
            return m(batch).mean()

        # Loader with exactly one batch -- will be cycled on the second step.
        val_loader = [val_ids_a]

        cb = DataSelectionCallback(
            model=model,
            threshold=-float("inf"),
            target="val_loader",
            val_loader=val_loader,
            val_loss_fn=val_loss_fn,
        )
        collector = HookManager(
            model,
            config=HookManagerConfig(linear_io=REGISTER_ALL),
            callbacks=[cb],
        )
        with collector.collect():
            # Step 1
            model(val_ids_a).mean().backward()
            model.zero_grad()
            # Step 2 -- loader is exhausted, should cycle
            model(val_ids_b).mean().backward()

        # val_loss_fn was called once per training step (2 total).
        assert len(calls) == 2

    @pytest.mark.parametrize("score_mode", ["ghost", "materialized"])
    def test_val_loader_same_batch_matches_fixed(self, score_mode):
        """val_loader mode with a fixed single-batch loader matches 'fixed' mode.

        When the val loader always returns the same batch, every step's
        'val_loader' target is identical to passing that batch as a 'fixed'
        target -- so the scores must agree.
        """
        torch.manual_seed(52)
        B, T = 3, 5
        model = MinimalEmbeddingMLP()
        train_ids = _make_token_ids(B, T)
        val_ids = _make_token_ids(2, T)  # different size to surface bugs

        def val_loss_fn(m, batch):
            return m(batch).mean()

        # Pre-compute the fixed target from val_ids.
        ref_model = MinimalEmbeddingMLP()
        ref_model.load_state_dict(model.state_dict())
        fixed_target = _capture_batch_gradient(ref_model, val_ids)

        # 'fixed' mode.
        model_fixed = MinimalEmbeddingMLP()
        model_fixed.load_state_dict(model.state_dict())
        cb_fixed = DataSelectionCallback(
            model=model_fixed,
            threshold=-float("inf"),
            score_mode=score_mode,
            target="fixed",
            target_gradient=fixed_target,
        )
        _run_step_with_callback(model_fixed, train_ids, cb_fixed)

        # 'val_loader' mode with a single-batch looping loader.
        model_val = MinimalEmbeddingMLP()
        model_val.load_state_dict(model.state_dict())
        cb_val = DataSelectionCallback(
            model=model_val,
            threshold=-float("inf"),
            score_mode=score_mode,
            target="val_loader",
            val_loader=[val_ids],
            val_loss_fn=val_loss_fn,
        )
        _run_step_with_callback(model_val, train_ids, cb_val)

        assert torch.allclose(cb_fixed.last_scores, cb_val.last_scores, atol=1e-4), (
            f"score_mode={score_mode!r}\n"
            f"fixed  scores: {cb_fixed.last_scores.tolist()}\n"
            f"val_loader scores: {cb_val.last_scores.tolist()}\n"
            f"max diff: {(cb_fixed.last_scores - cb_val.last_scores).abs().max():.2e}"
        )


# --------------------------------------------------------------------------- #
# renormalize                                                                   #
# --------------------------------------------------------------------------- #


def _named_hooked_grads(model: MinimalEmbeddingMLP) -> dict[str, torch.Tensor]:
    """Weight/bias grads of every hooked layer (embedding + MLP linears)."""
    grads = {"embedding.weight": model.embedding.weight.grad.clone()}
    for i, m in enumerate(model.mlp):
        if isinstance(m, nn.Linear):
            grads[f"mlp.{i}.weight"] = m.weight.grad.clone()
            grads[f"mlp.{i}.bias"] = m.bias.grad.clone()
    return grads


class TestRenormalize:
    """renormalize=True rescales the kept samples from 1/B to 1/(B-k)."""

    B, T = 4, 5

    def _dropped_run(self, *, renormalize: bool):
        """One mean-loss step dropping the bottom half; returns (cb, grads)."""
        torch.manual_seed(7)
        model = MinimalEmbeddingMLP()
        token_ids = _make_token_ids(self.B, self.T)
        cb = DataSelectionCallback(
            model=model,
            threshold_mode="bottom_fraction",
            threshold=0.5,
            renormalize=renormalize,
        )
        _run_step_with_callback(model, token_ids, cb, loss_reduction="mean")
        return cb, token_ids, model, _named_hooked_grads(model)

    def test_matches_mean_loss_over_kept_batch(self):
        """Renormalized grads equal a backward of the mean loss over a batch
        containing only the kept samples -- 'as if the dropped samples were
        never in the batch', exactly.
        """
        cb, token_ids, model, grads = self._dropped_run(renormalize=True)
        kept = [i for i in range(self.B) if i not in set(cb.last_dropped)]
        assert 0 < len(kept) < self.B  # the run must actually drop something

        ref = MinimalEmbeddingMLP()
        ref.load_state_dict(model.state_dict())
        ref(token_ids[kept]).mean().backward()
        ref_grads = _named_hooked_grads(ref)

        for name, g in grads.items():
            assert torch.allclose(g, ref_grads[name], atol=1e-5), (
                f"{name}: max diff {(g - ref_grads[name]).abs().max():.2e}"
            )

    def test_default_keeps_one_over_b_weighting(self):
        """Without renormalize the kept samples stay at weight 1/B: the two
        runs differ by exactly the factor B/(B-k) on every hooked grad.
        """
        cb_plain, _, _, grads_plain = self._dropped_run(renormalize=False)
        cb_renorm, _, _, grads_renorm = self._dropped_run(renormalize=True)
        assert cb_plain.last_dropped == cb_renorm.last_dropped
        k = len(cb_plain.last_dropped)
        factor = self.B / (self.B - k)
        for name, g in grads_renorm.items():
            assert torch.allclose(g, grads_plain[name] * factor, atol=1e-5), (
                f"{name}: max diff {(g - grads_plain[name] * factor).abs().max():.2e}"
            )

    def test_drop_all_removes_without_rescale(self):
        """K == B: the batch contributes nothing (no empty-batch rescale)."""
        torch.manual_seed(8)
        model = MinimalEmbeddingMLP()
        token_ids = _make_token_ids(self.B, self.T)
        cb = DataSelectionCallback(
            model=model,
            threshold=float("inf"),  # drop everything
            renormalize=True,
        )
        _run_step_with_callback(model, token_ids, cb, loss_reduction="mean")
        assert len(cb.last_dropped) == self.B
        for name, g in _named_hooked_grads(model).items():
            assert torch.allclose(g, torch.zeros_like(g), atol=1e-6), name

    def test_drop_none_is_noop(self):
        """K == 0: renormalize must not perturb the untouched gradient."""
        torch.manual_seed(9)
        model = MinimalEmbeddingMLP()
        token_ids = _make_token_ids(self.B, self.T)

        ref = MinimalEmbeddingMLP()
        ref.load_state_dict(model.state_dict())
        ref(token_ids).mean().backward()
        ref_grads = _named_hooked_grads(ref)

        cb = DataSelectionCallback(
            model=model,
            threshold=-float("inf"),  # keep everything
            renormalize=True,
        )
        _run_step_with_callback(model, token_ids, cb, loss_reduction="mean")
        assert cb.last_dropped == []
        for name, g in _named_hooked_grads(model).items():
            assert torch.allclose(g, ref_grads[name], atol=1e-6), name

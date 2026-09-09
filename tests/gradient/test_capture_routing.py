"""Capture-time projection routing (``style="auto"``).

Scoring already routes between the factorized and materialized cross-gram by a
cost model; capture did not, so a projected capture always kept the token axis.
After projection that is the wrong default: at ``k_a=k_g=64`` the factors cost
``S*(k_a+k_g)`` against the outer product's ``k_a*k_g``, so a 512-token sequence
stores 16x more as factors.

Two properties matter and are tested separately:
  * the rule fires where the arithmetic says it should, and
  * routing is an OPTIMIZATION -- scores must not depend on which side it picks.
"""

from __future__ import annotations

import pytest
import torch

from dattri_llm.gradient import ops
from dattri_llm.gradient.gradient import Factorized, Gradient

B, T, D_IN, D_OUT, P = 3, 64, 32, 48, 8


def _block(seed: int, t: int = T) -> Gradient:
    g = torch.Generator().manual_seed(seed)
    return Gradient(
        representation={"l": "factorized"},
        data={
            "l": Factorized(
                activation=torch.randn(B, t, D_IN, generator=g),
                pre_activation_grad=torch.randn(B, t, D_OUT, generator=g),
            ),
        },
        layer_types={"l": "nn.Linear"},
        indexing={"l": "batch_token"},
    )


class TestRule:
    @pytest.mark.parametrize(
        ("seq_len", "k_a", "k_g", "expected"),
        [
            (512, 64, 64, True),  # H = 32; projection makes factors 16x bigger
            (16, 64, 64, False),  # below the crossover, keep the factors
            (32, 64, 64, True),  # exactly at H -- materialize (>=)
            (512, 1024, 4096, False),  # unprojected: H = 819 > 512, keep factors
            (128, 8, 8, True),  # H = 4
        ],
    )
    def test_crossover(self, seq_len, k_a, k_g, expected):
        assert ops.maybe_materialize_projected(seq_len, k_a, k_g) is expected

    def test_kappa_biases_toward_keeping_factors(self):
        # kappa > 1 raises the bar for materializing, because the factors are
        # what per-token attribution needs.
        assert ops.maybe_materialize_projected(40, 64, 64) is True
        assert ops.maybe_materialize_projected(40, 64, 64, kappa=4.0) is False

    def test_degenerate_dims_do_not_divide_by_zero(self):
        assert ops.maybe_materialize_projected(512, 0, 0) is False


class TestAutoStyle:
    @staticmethod
    def _projector(feature, batch_size, *, proj_dim, **kw):
        """Dattri's ``random_project`` protocol, made deterministic.

        Truncation is a linear map like any projection, and determinism is what
        lets the two styles be compared exactly rather than statistically.
        """
        return lambda x, ensemble_id=0: x[..., :proj_dim]

    def _project(self, block: Gradient, style: str) -> Gradient:
        return block.project(
            self._projector,
            {"__default__": {"style": style, "proj_dim": P, "proj_seed": 0}},
        )

    def test_auto_materializes_past_the_crossover(self):
        # T=64 with proj_dim=8 -> H = 8*8/16 = 4, and 64 >= 4, so auto should
        # land on the materialized side.
        out = self._project(_block(0), "auto")
        assert out.representation["l"] == "materialized"
        assert out.indexing["l"] == "batch"

    def test_auto_keeps_factors_below_the_crossover(self):
        # A single-token block is below any positive crossover.
        out = self._project(_block(0, t=1), "auto")
        assert out.representation["l"] == "factorized"

    def test_auto_matches_the_explicit_style_it_selects(self):
        auto = self._project(_block(1), "auto")
        explicit = self._project(_block(1), "logra_materialized")
        assert auto.representation["l"] == explicit.representation["l"]
        torch.testing.assert_close(auto.data["l"], explicit.data["l"])

    def test_scores_are_unchanged_by_the_routing(self):
        """The decisive property: routing must not move the score.

        A factorized projected block and its materialized counterpart are the
        same gradient in two encodings, so the cross-gram must agree.
        """
        tr_f = self._project(_block(2), "logra_factorized")
        te_f = self._project(_block(3), "logra_factorized")
        tr_m = self._project(_block(2), "logra_materialized")
        te_m = self._project(_block(3), "logra_materialized")

        s_fact = tr_f.similarity(te_f, metric="dot", reduce="all")
        s_mat = tr_m.similarity(te_m, metric="dot", reduce="all")
        assert s_fact.shape == (B, B)
        torch.testing.assert_close(s_fact, s_mat, rtol=1e-4, atol=1e-4)

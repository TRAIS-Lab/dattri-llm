"""``HookManagerConfig(capture_style=...)``: the representation a layer is
buffered in, for unprojected and ``"logra"``-projected captures alike.

The three styles are three encodings of the same gradient: ``"materialized"``
must equal the materialization of the ``"factorized"`` capture, ``"auto"`` must
land on one of the two by the cost rule of :func:`ops.should_materialize`, and
scores must not move.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from dattri_llm.gradient import ops
from dattri_llm.gradient.callbacks import CaptureCallback
from dattri_llm.gradient.hooks import REGISTER_ALL, HookManager, HookManagerConfig

IN, HID, OUT = 16, 32, 8
PROJ = {
    "proj_dim": 8,
    "proj_max_batch_size": 8,
    "proj_type": "rademacher",
    "proj_seed": 3,
}


def _model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(IN, HID), nn.ReLU(), nn.Linear(HID, OUT))


def _capture(x: torch.Tensor, *, capture_style: str, projection=None, micro=None):
    """One step's assembled Gradient under *capture_style*.

    *micro* splits the batch into that many micro-batches before one
    ``on_step_end``, to exercise the per-step consistency rule.
    """
    model = _model()
    cb = CaptureCallback()
    hm = HookManager(
        model,
        config=HookManagerConfig(
            linear_io=REGISTER_ALL,
            projection_kwargs=projection,
            capture_style=capture_style,
        ),
        callbacks=[cb],
    )
    with hm.collect():
        if micro is None:
            model(x).pow(2).sum().backward()
        else:
            for part in x.chunk(micro):
                model(part).pow(2).sum().backward()
    hm.remove()
    return cb.record.gradient


def _dense(g):
    return {n: ops.materialize(g.data[n], g.layer_types[n]) for n in g.layer_names}


@pytest.mark.parametrize("shape", [(6, IN), (3, 4, IN)], ids=["2d", "3d"])
class TestUnprojected:
    def test_materialized_equals_materialized_factors(self, shape):
        x = torch.randn(*shape)
        fac = _capture(x, capture_style="factorized")
        mat = _capture(x, capture_style="materialized")
        assert all(r == "factorized" for r in fac.representation.values())
        assert all(r == "materialized" for r in mat.representation.values())
        for n, want in _dense(fac).items():
            torch.testing.assert_close(mat.data[n], want)
            assert mat.indexing[n] == "batch"

    def test_scores_unchanged(self, shape):
        x, q = torch.randn(*shape), torch.randn(*shape)
        s = {}
        for style in ("factorized", "materialized", "auto"):
            tr, te = _capture(x, capture_style=style), _capture(q, capture_style=style)
            s[style] = tr.similarity(te, metric="dot", reduce="all")
        torch.testing.assert_close(s["materialized"], s["factorized"])
        torch.testing.assert_close(s["auto"], s["factorized"])


class TestAuto:
    def test_follows_the_cost_rule_on_raw_widths(self):
        # fc1: N_i + 1 = 17, N_o = 32 -> crossover T >= 17*32/49 ~ 11.1;
        # fc2: 33 and 8 -> T >= 6.4.  Four tokens keeps both factorized, a
        # hundred materializes both.
        short = _capture(torch.randn(3, 4, IN), capture_style="auto")
        assert all(r == "factorized" for r in short.representation.values())
        long = _capture(torch.randn(3, 100, IN), capture_style="auto")
        assert all(r == "materialized" for r in long.representation.values())
        # In between, each layer decides for itself.
        mid = _capture(torch.randn(3, 8, IN), capture_style="auto")
        assert mid.representation["0"] == "factorized"
        assert mid.representation["2"] == "materialized"

    def test_two_dimensional_inputs_are_one_token(self):
        g = _capture(torch.randn(6, IN), capture_style="auto")
        assert all(r == "factorized" for r in g.representation.values())

    def test_representation_is_fixed_by_the_first_micro_batch(self):
        # Under gradient accumulation the micro-batches of one step may differ
        # in length; whichever representation the first one chose is kept, so
        # a step never mixes the two encodings.
        from dattri_llm.gradient.hooks.hooks import _keep_factors

        fresh = {"_act_parts": [], "_proj_parts": [], "_capture_style": "auto"}
        assert _keep_factors(fresh, 4, 17, 32) is True
        assert _keep_factors(fresh, 100, 17, 32) is False
        factors_so_far = {**fresh, "_act_parts": [object()]}
        assert _keep_factors(factors_so_far, 100, 17, 32) is True
        dense_so_far = {**fresh, "_proj_parts": [object()]}
        assert _keep_factors(dense_so_far, 4, 17, 32) is False


class TestLogra:
    def _proj(self):
        return {"__default__": {"style": "logra", **PROJ}}

    def test_materialized_equals_materialized_projected_factors(self):
        x = torch.randn(3, 5, IN)
        fac = _capture(x, capture_style="factorized", projection=self._proj())
        mat = _capture(x, capture_style="materialized", projection=self._proj())
        for n in fac.layer_names:
            assert fac.representation[n] == "factorized"
            assert mat.representation[n] == "materialized"
            want = ops.materialize(fac.data[n], "nn.Linear")
            torch.testing.assert_close(mat.data[n], want, atol=1e-5, rtol=1e-4)

    def test_auto_uses_the_projected_widths(self):
        # k_a = k_g = 8 -> crossover T >= 4: one token keeps the factors,
        # sixteen materialize.
        one = _capture(
            torch.randn(3, IN), capture_style="auto", projection=self._proj()
        )
        assert all(r == "factorized" for r in one.representation.values())
        many = _capture(
            torch.randn(3, 16, IN), capture_style="auto", projection=self._proj()
        )
        assert all(r == "materialized" for r in many.representation.values())

    @pytest.mark.parametrize("style", ["dense", "mask"])
    def test_dense_styles_ignore_the_capture_style(self, style):
        kw = {"style": style, "proj_dim": 8, "proj_seed": 3}
        if style == "dense":
            kw.update(proj_max_batch_size=8, proj_type="rademacher")
        outs = [
            _capture(
                torch.randn(3, 5, IN), capture_style=cs, projection={"__default__": kw}
            )
            for cs in ("factorized", "materialized", "auto")
        ]
        for g in outs:
            assert all(r == "materialized" for r in g.representation.values())
        for n in outs[0].layer_names:
            torch.testing.assert_close(outs[1].data[n], outs[0].data[n])
            torch.testing.assert_close(outs[2].data[n], outs[0].data[n])


class TestShouldMaterialize:
    def test_explicit_styles_and_rule(self):
        assert ops.should_materialize("factorized", 10_000, 8, 8) is False
        assert ops.should_materialize("materialized", 1, 8, 8) is True
        assert ops.should_materialize("auto", 3, 8, 8) is False  # 3 < 4
        assert ops.should_materialize("auto", 4, 8, 8) is True
        with pytest.raises(ValueError, match="capture_style"):
            ops.should_materialize("bogus", 4, 8, 8)

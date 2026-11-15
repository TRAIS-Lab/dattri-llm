"""Grouped convolutions (``groups > 1``, depthwise included) through every op.

A grouped convolution's weight is ``(C_out, C_in/G, *k)``: output block ``i``
depends on input block ``i`` only, so the per-sample gradient is a sum of
one outer product per group, not the single ``(C_out, C_in*k)`` product of
an ungrouped layer.  The factors carry a group axis after preprocessing and
every kernel aligns it on both sides.  These tests check, in float64 against
per-sample autograd on real ``nn.Conv*`` / ``nn.ConvTranspose*`` modules:
materialization in the parameter layout, cross/pairwise/self dots and norms
in both routing modes, per-token decompositions, K-FAC factors, inverses and
preconditioning, EK-FAC rotations, LoGra and mask projections, and capture
through the ``HookManager``.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from dattri_llm.gradient import ops
from dattri_llm.gradient.hooks import HookManager, HookManagerConfig
from dattri_llm.gradient.ops import (
    LayerKroneckerAccumulator,
    cross_dot_factors,
    cross_gram_per_token,
    cross_gram_per_token_dense,
    dot_factors,
    ekfac_materialize_factors,
    ekfac_precondition,
    extract_module_kwargs,
    grad_norm_sq_factors,
    kfac_cross_factors,
    kfac_eigh,
    kfac_factors,
    kfac_precondition_materialized,
    materialize_factors,
    preprocess_factors,
    sym_inverse,
)
from dattri_llm.gradient.ops.kronecker import _ekfac_rotate_materialized

B = 3
TOL = {"rtol": 1e-9, "atol": 1e-11}


def _cases():
    """``(id, module factory, per-sample input shape)``."""
    return [
        ("conv1d-g2", lambda: nn.Conv1d(6, 4, 3, groups=2, padding=1), (6, 9)),
        ("conv1d-g2-nobias", lambda: nn.Conv1d(6, 4, 3, groups=2, bias=False), (6, 9)),
        ("conv1d-depthwise", lambda: nn.Conv1d(4, 4, 3, groups=4, padding=2), (4, 9)),
        (
            "conv2d-g2",
            lambda: nn.Conv2d(4, 6, (3, 2), groups=2, padding=1, stride=(1, 2)),
            (4, 7, 6),
        ),
        (
            "conv3d-depthwise",
            lambda: nn.Conv3d(3, 3, 2, groups=3, padding=1),
            (3, 4, 4, 3),
        ),
        (
            "convT1d-g2",
            lambda: nn.ConvTranspose1d(4, 6, 3, groups=2, padding=1),
            (4, 9),
        ),
        ("convT2d-g2", lambda: nn.ConvTranspose2d(4, 4, 2, groups=2), (4, 5, 5)),
    ]


def _layer_type(module):
    return f"nn.{type(module).__name__}"


class _Capture:
    """Raw hook captures and per-sample autograd gradients of one module."""

    def __init__(self, factory, shape, seed=0):
        torch.manual_seed(seed)
        self.module = factory().double()
        self.layer_type = _layer_type(self.module)
        self.kwargs = extract_module_kwargs(self.module, self.layer_type)
        self.x = torch.randn(B, *shape, dtype=torch.float64)
        out = self.module(self.x)
        self.w = torch.randn_like(out)  # loss = sum(out * w)  ->  dL/dout = w
        self.a, self.g = self.x, self.w
        self.ref = []  # per-sample (weight grad, bias grad | None)
        for i in range(B):
            self.module.zero_grad()
            (self.module(self.x[i : i + 1]) * self.w[i : i + 1]).sum().backward()
            bias = self.module.bias
            self.ref.append(
                (
                    self.module.weight.grad.detach().clone(),
                    None if bias is None else bias.grad.detach().clone(),
                )
            )
        self.module.zero_grad()

    @property
    def groups(self):
        return self.module.groups

    @property
    def transposed(self):
        return "Transpose" in self.layer_type

    def ref_flat(self):
        """Reference per-sample gradients in the materialized layout."""
        rows = []
        for wg, bg in self.ref:
            if self.transposed:
                # (C_in, C_out/G*k) blocks per group with the transposed bias
                # augmentation: an extra a feature per group and C_out/G extra
                # g features carrying the bias.
                c_in = wg.shape[0]
                g_, c_in_g = self.groups, c_in // self.groups
                p_g = wg[0].numel()
                blocks = wg.reshape(g_, c_in_g, p_g)
                if bg is None:
                    rows.append(blocks.reshape(-1))
                    continue
                c_out_g = bg.numel() // g_
                full = torch.zeros(g_, c_in_g + 1, p_g + c_out_g, dtype=wg.dtype)
                full[:, :c_in_g, :p_g] = blocks
                full[:, -1, p_g:] = bg.reshape(g_, c_out_g)
                rows.append(full.reshape(-1))
            else:
                block = wg.reshape(wg.shape[0], -1)  # (C_out, C_in/G*k)
                if bg is not None:
                    block = torch.cat([block, bg.unsqueeze(-1)], dim=-1)
                rows.append(block.reshape(-1))
        return torch.stack(rows)

    def pre(self):
        return preprocess_factors(self.a, self.g, self.layer_type, self.kwargs)


@pytest.fixture(params=_cases(), ids=[c[0] for c in _cases()])
def cap(request):
    _, factory, shape = request.param
    return _Capture(factory, shape)


def _second(cap_):
    """A second capture of the same module (different data, same weights)."""
    other = _Capture.__new__(_Capture)
    other.module, other.layer_type, other.kwargs = (
        cap_.module,
        cap_.layer_type,
        cap_.kwargs,
    )
    torch.manual_seed(1)
    other.x = torch.randn_like(cap_.x)
    other.w = torch.randn_like(cap_.w)
    other.a, other.g = other.x, other.w
    other.ref = []
    for i in range(B):
        other.module.zero_grad()
        (other.module(other.x[i : i + 1]) * other.w[i : i + 1]).sum().backward()
        bias = other.module.bias
        other.ref.append(
            (
                other.module.weight.grad.detach().clone(),
                None if bias is None else bias.grad.detach().clone(),
            )
        )
    other.module.zero_grad()
    return other


class TestPreprocess:
    def test_module_kwargs_record_groups(self, cap):
        assert cap.kwargs["groups"] == cap.module.groups

    def test_factors_carry_the_group_axis(self, cap):
        a, g = cap.pre()
        assert a.ndim == 4
        assert g.ndim == 4
        assert a.shape[2] == cap.groups
        assert g.shape[2] == cap.groups
        assert a.shape[:3] == g.shape[:3]


class TestMaterialize:
    def test_matches_per_sample_autograd(self, cap):
        got = materialize_factors(cap.a, cap.g, cap.layer_type, cap.kwargs)
        torch.testing.assert_close(got, cap.ref_flat(), **TOL)

    def test_shape_is_the_parameter_count(self, cap):
        got = materialize_factors(cap.a, cap.g, cap.layer_type, cap.kwargs)
        n_params = sum(p.numel() for p in cap.module.parameters())
        if cap.transposed and cap.module.bias is not None:
            # The transposed bias augmentation pads each group block.
            assert got.shape[1] > n_params
        else:
            assert got.shape == (B, n_params)


class TestDots:
    @pytest.mark.parametrize("mode", ["factorized", "materialized", "auto"])
    def test_cross_gram_matches_reference(self, cap, mode):
        other = _second(cap)
        got = cross_dot_factors(
            cap.a,
            cap.g,
            other.a,
            other.g,
            cap.layer_type,
            cap.kwargs,
            cap.kwargs,
            mode=mode,
        )
        torch.testing.assert_close(got, cap.ref_flat() @ other.ref_flat().T, **TOL)

    def test_self_dot_and_norms(self, cap):
        ref = cap.ref_flat()
        got = dot_factors(
            cap.a, cap.g, cap.a, cap.g, cap.layer_type, cap.kwargs, cap.kwargs
        )
        torch.testing.assert_close(got, (ref * ref).sum(-1), **TOL)
        for mode in ("factorized", "materialized"):
            norm = grad_norm_sq_factors(
                cap.a, cap.g, cap.layer_type, cap.kwargs, mode=mode
            )
            torch.testing.assert_close(norm, (ref * ref).sum(-1), **TOL)

    def test_per_token_decomposition_sums_to_the_gram(self, cap):
        other = _second(cap)
        a1, g1 = cap.pre()
        a2, g2 = other.pre()
        per_token = cross_gram_per_token(a1, g1, a2, g2, cap.layer_type)
        assert per_token.shape == (B, a1.shape[1], B)
        gram = cap.ref_flat() @ other.ref_flat().T
        torch.testing.assert_close(per_token.sum(1), gram, **TOL)
        dense2 = materialize_factors(other.a, other.g, cap.layer_type, cap.kwargs)
        torch.testing.assert_close(
            cross_gram_per_token_dense(a1, g1, dense2, cap.layer_type), per_token, **TOL
        )


class TestKronecker:
    def test_factors_are_per_group_covariances(self, cap):
        a, g = cap.pre()
        A, G = kfac_factors(cap.a, cap.g, cap.layer_type, cap.kwargs)
        n_rows = a.shape[0] * a.shape[1]
        assert A.shape == (cap.groups, a.shape[-1], a.shape[-1])
        assert G.shape == (cap.groups, g.shape[-1], g.shape[-1])
        for grp in range(cap.groups):
            a_rows = a[:, :, grp].reshape(n_rows, -1)
            g_rows = g[:, :, grp].reshape(n_rows, -1)
            torch.testing.assert_close(A[grp], (a_rows.T @ a_rows / n_rows).float())
            torch.testing.assert_close(G[grp], (g_rows.T @ g_rows / n_rows).float())

    def test_accumulator_matches_one_shot_factors(self, cap):
        other = _second(cap)
        acc = LayerKroneckerAccumulator()
        acc.update(cap.a, cap.g, cap.layer_type, cap.kwargs)
        acc.update(other.a, other.g, cap.layer_type, cap.kwargs)
        A, G = acc.result()
        a_all = torch.cat([cap.a, other.a])
        g_all = torch.cat([cap.g, other.g])
        A_ref, G_ref = kfac_factors(a_all, g_all, cap.layer_type, cap.kwargs)
        torch.testing.assert_close(A, A_ref)
        torch.testing.assert_close(G, G_ref)

    def test_inverses_are_per_group(self, cap):
        A, G = kfac_factors(cap.a, cap.g, cap.layer_type, cap.kwargs)
        A_inv = sym_inverse(A, 0.1)
        for grp in range(cap.groups):
            torch.testing.assert_close(A_inv[grp], sym_inverse(A[grp], 0.1))
        torch.testing.assert_close(
            ops.dense_inverse(G, 0.1),
            torch.stack([ops.dense_inverse(G[i], 0.1) for i in range(cap.groups)]),
        )

    def _blocks(self, cap, flat):
        """Reference flat gradient as per-group (rows, cols) blocks."""
        a, g = cap.pre()
        if cap.transposed:
            return flat.reshape(B, cap.groups, a.shape[-1], g.shape[-1])
        return flat.reshape(B, cap.groups, g.shape[-1], a.shape[-1])

    def test_kfac_preconditioned_gram(self, cap):
        other = _second(cap)
        A, G = kfac_factors(cap.a, cap.g, cap.layer_type, cap.kwargs)
        A_inv, G_inv = sym_inverse(A, 0.1), sym_inverse(G, 0.1)
        got = kfac_cross_factors(
            cap.a,
            cap.g,
            other.a,
            other.g,
            cap.layer_type,
            A_inv,
            G_inv,
            cap.kwargs,
            cap.kwargs,
        )
        # Reference: precondition each group block explicitly.
        b1 = self._blocks(cap, cap.ref_flat()).float()
        b2 = self._blocks(cap, other.ref_flat()).float()
        if cap.transposed:  # blocks (C_in/G, P_g): A on the left, G on the right
            pre = torch.einsum("gce,bgep,gpf->bgcf", A_inv, b1, G_inv)
        else:  # blocks (D_g, K_g): G on the left, A on the right
            pre = torch.einsum("gde,bgek,gkj->bgdj", G_inv, b1, A_inv)
        ref = pre.reshape(B, -1) @ b2.reshape(B, -1).T
        torch.testing.assert_close(got.float(), ref, rtol=1e-4, atol=1e-5)
        if not cap.transposed:
            dense1 = materialize_factors(cap.a, cap.g, cap.layer_type, cap.kwargs)
            pre_dense = kfac_precondition_materialized(dense1, A_inv, G_inv)
            torch.testing.assert_close(
                pre_dense, pre.reshape(B, -1), rtol=1e-4, atol=1e-5
            )

    def test_ekfac_rotation_roundtrip(self, cap):
        A, G = kfac_factors(cap.a, cap.g, cap.layer_type, cap.kwargs)
        _, U_A, _, U_G = kfac_eigh(A, G)
        M = ekfac_materialize_factors(
            cap.a, cap.g, cap.layer_type, U_A, U_G, cap.kwargs
        )
        dense = materialize_factors(cap.a, cap.g, cap.layer_type, cap.kwargs).float()
        # Rotating the dense block equals rotating the factors.
        torch.testing.assert_close(
            _ekfac_rotate_materialized(dense, cap.layer_type, U_A, U_G),
            M,
            rtol=1e-5,
            atol=1e-6,
        )
        # Undoing the rotation (unit eigenvalues) gives the block back.
        back = ekfac_precondition(M, U_A, U_G, torch.ones_like(M[0]), cap.layer_type)
        torch.testing.assert_close(back, dense, rtol=1e-4, atol=1e-5)


class TestProjection:
    def test_logra_keeps_the_group_axis(self, cap):
        a_p, g_p = ops.project_factors(
            cap.a,
            cap.g,
            cap.layer_type,
            ops.DattriProjector(),
            cap.kwargs,
            proj_dim=4,
            proj_max_batch_size=8,
            proj_type="rademacher",
        )
        a, _ = cap.pre()
        assert a_p.shape == (B, a.shape[1], cap.groups, 4)
        assert g_p.shape == (B, a.shape[1], cap.groups, 4)
        # The projected factors materialize to one (4, 4) block per group.
        assert materialize_factors(a_p, g_p, "nn.Linear").shape == (B, cap.groups * 16)

    def test_mask_gathers_the_materialized_gradient(self, cap):
        proj = ops.DattriProjector()
        full = materialize_factors(cap.a, cap.g, cap.layer_type, cap.kwargs)
        k = min(12, full.shape[1])
        got = ops.mask_factors(
            cap.a, cap.g, cap.layer_type, proj, cap.kwargs, proj_dim=k, proj_seed=3
        )
        idx = proj.mask_indices(
            full.shape[1], proj_dim=k, proj_seed=3, device=full.device
        )
        torch.testing.assert_close(got, full[:, idx], **TOL)


class _GroupedNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(6, 4, 3, groups=2, padding=1)
        self.depthwise = nn.Conv1d(4, 4, 3, groups=4, padding=1)
        self.head = nn.Linear(4, 2)

    def forward(self, x):
        h = torch.relu(self.depthwise(torch.relu(self.conv(x))))
        return self.head(h.transpose(1, 2))  # (B, L, 2)


class TestHookManagerCapture:
    def _run(self, **hm_kwargs):
        torch.manual_seed(0)
        model = _GroupedNet().double()
        x = torch.randn(B, 6, 9, dtype=torch.float64)
        w = torch.randn(B, 9, 2, dtype=torch.float64)
        ref = []
        for i in range(B):
            model.zero_grad()
            (model(x[i : i + 1]) * w[i : i + 1]).sum().backward()
            ref.append(
                {n: p.grad.detach().clone() for n, p in model.named_parameters()}
            )
        hm = HookManager(model, **hm_kwargs)
        try:
            with hm.collect():
                model.zero_grad()
                (model(x) * w).sum().backward()
            return model, hm.get_gradient(), ref
        finally:
            hm.remove()

    def _check(self, model, g, ref):
        for name in ("conv", "depthwise"):
            conv = getattr(model, name)
            dense = ops.materialize(g.data[name], g.layer_types[name])
            c_out, k = conv.weight.shape[0], conv.weight[0].numel()
            rows = dense.reshape(B, c_out, k + 1)
            for i in range(B):
                torch.testing.assert_close(
                    rows[i, :, :k], ref[i][name + ".weight"].reshape(c_out, k), **TOL
                )
                torch.testing.assert_close(rows[i, :, k], ref[i][name + ".bias"], **TOL)

    def test_factorized_capture(self):
        model, g, ref = self._run()
        assert g.data["conv"].module_kwargs["groups"] == 2
        self._check(model, g, ref)

    def test_materialized_capture(self):
        model, g, ref = self._run(
            config=HookManagerConfig(capture_style="materialized")
        )
        assert g.representation["depthwise"] == "materialized"
        self._check(model, g, ref)

    def test_similarity_matches_reference(self):
        _, g, ref = self._run()
        flat = torch.stack(
            [torch.cat([r[n].reshape(-1) for n in sorted(r)]) for r in ref]
        )
        torch.testing.assert_close(g.similarity(g, reduce="all"), flat @ flat.T, **TOL)

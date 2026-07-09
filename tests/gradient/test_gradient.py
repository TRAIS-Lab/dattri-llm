"""Unit tests for dattri_llm.gradient.gradient.Gradient."""

from __future__ import annotations

import pytest
import torch
from dattri.func.projection import random_project

from dattri_llm.gradient import ops
from dattri_llm.gradient.gradient import Factorized, Gradient

# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

B, T, D_IN, D_OUT = 2, 4, 8, 16  # batch, token, in_features, out_features


def mat_tensor(b=B, feat=D_OUT * D_IN) -> torch.Tensor:
    """Materialized gradient tensor [B, feat]."""
    return torch.randn(b, feat)


def mat_tensor_bt(b=B, t=T, feat=D_OUT * D_IN) -> torch.Tensor:
    """Materialized gradient tensor [B, T, feat]."""
    return torch.randn(b, t, feat)


def factorized(b=B) -> Factorized:
    """Factorized gradient [B, I] x [B, O]."""
    return Factorized(
        activation=torch.randn(b, D_IN),
        pre_activation_grad=torch.randn(b, D_OUT),
    )


def factorized_bt(b=B, t=T) -> Factorized:
    """Factorized gradient [B, T, I] x [B, T, O]."""
    return Factorized(
        activation=torch.randn(b, t, D_IN),
        pre_activation_grad=torch.randn(b, t, D_OUT),
    )


def make_gradient(
    layers=("l1", "l2"),
    repr_type="materialized",
    indexing="batch",
    layer_type="nn.Linear",
) -> Gradient:
    if repr_type == "factorized":
        fn = factorized_bt if indexing == "batch_token" else factorized
        data = {name: fn() for name in layers}
    else:
        fn = mat_tensor_bt if indexing == "batch_token" else mat_tensor
        data = {name: fn() for name in layers}

    representation = dict.fromkeys(layers, repr_type)
    idx_dict = dict.fromkeys(layers, indexing)
    return Gradient(
        representation=representation,
        data=data,
        layer_types=dict.fromkeys(layers, layer_type),
        indexing=idx_dict,
    )


# --------------------------------------------------------------------------- #
# Factorized                                                                   #
# --------------------------------------------------------------------------- #


class TestFactorized:
    def test_materialize_2d(self):
        f = Factorized(
            activation=torch.ones(B, D_IN),
            pre_activation_grad=torch.ones(B, D_OUT),
        )
        out = ops.materialize(f, "nn.Linear")
        assert out.shape == (B, D_OUT * D_IN)

    def test_materialize_3d(self):
        f = factorized_bt()
        # ops.materialize sums over the token dimension, returning (B, O*I)
        out = ops.materialize(f, "nn.Linear")
        assert out.shape == (B, D_OUT * D_IN)

    def test_materialize_invalid_ndim(self):
        a = torch.randn(B, T, D_IN, 1)
        g = torch.randn(B, T, D_OUT, 1)
        with pytest.raises((ValueError, RuntimeError)):
            ops.materialize(Factorized(a, g), "nn.Linear")

    def test_to_device_dtype(self):
        f = factorized()
        f2 = f.to(dtype=torch.float64)
        assert f2.activation.dtype == torch.float64
        assert f2.pre_activation_grad.dtype == torch.float64


# --------------------------------------------------------------------------- #
# Gradient.validate                                                            #
# --------------------------------------------------------------------------- #


class TestValidate:
    def test_valid_materialized(self):
        g = make_gradient(repr_type="materialized")
        g.validate()  # should not raise

    def test_valid_factorized(self):
        g = make_gradient(repr_type="factorized")
        g.validate()

    def test_valid_batch_token(self):
        g = make_gradient(repr_type="materialized", indexing="batch_token")
        g.validate()

    def test_empty_data(self):
        with pytest.raises(ValueError, match="empty"):
            Gradient(representation={}, data={}, layer_types={}).validate()

    def test_missing_representation_key(self):
        data = {"l1": mat_tensor(), "l2": mat_tensor()}
        rep = {"l1": "materialized"}  # missing l2
        with pytest.raises(ValueError, match="Missing representation"):
            Gradient(
                representation=rep,
                data=data,
                layer_types={"l1": "nn.Linear", "l2": "nn.Linear"},
            )

    def test_factorized_wrong_data_type(self):
        data = {"l1": mat_tensor()}
        rep = {"l1": "factorized"}
        with pytest.raises(TypeError, match="Factorized"):
            Gradient(representation=rep, data=data, layer_types={"l1": "nn.Linear"})

    def test_materialized_wrong_data_type(self):
        data = {"l1": factorized()}
        rep = {"l1": "materialized"}
        with pytest.raises(TypeError, match="Tensor"):
            Gradient(representation=rep, data=data, layer_types={"l1": "nn.Linear"})

    def test_batch_size_mismatch(self):
        data = {"l1": mat_tensor(b=2), "l2": mat_tensor(b=3)}
        rep = {"l1": "materialized", "l2": "materialized"}
        with pytest.raises(ValueError, match="batch size"):
            Gradient(
                representation=rep,
                data=data,
                layer_types={"l1": "nn.Linear", "l2": "nn.Linear"},
            )

    def test_batch_token_wrong_ndim(self):
        # A layer declared as batch_token but given a 2-D tensor must fail.
        data = {"l1": mat_tensor()}
        rep = {"l1": "materialized"}
        with pytest.raises(ValueError, match="batch_token"):
            Gradient(
                representation=rep,
                data=data,
                layer_types={"l1": "nn.Linear"},
                indexing={"l1": "batch_token"},
            )

    def _embedding_factor(self, b, t=T, d=D_OUT):
        return Factorized(torch.randint(0, t, (b, t)), torch.randn(b, t, d))

    def test_broadcast_embedding_validates(self):
        # A positional embedding added to every sample has batch 1 (broadcast);
        # it must validate alongside the batch-B layers and report batch B.
        Bb = 3
        g = Gradient(
            representation={"wte": "factorized", "wpe": "factorized"},
            data={"wte": self._embedding_factor(Bb), "wpe": self._embedding_factor(1)},
            layer_types={"wte": "nn.Embedding", "wpe": "nn.Embedding"},
            indexing={"wte": "batch_token", "wpe": "batch_token"},
        )
        assert g.batch_size == Bb

    def test_broadcast_embedding_first_does_not_drive_batch_size(self):
        # Even when the broadcast (batch-1) layer comes first, batch_size is B.
        Bb = 3
        g = Gradient(
            representation={"wpe": "factorized", "wte": "factorized"},
            data={"wpe": self._embedding_factor(1), "wte": self._embedding_factor(Bb)},
            layer_types={"wpe": "nn.Embedding", "wte": "nn.Embedding"},
            indexing={"wpe": "batch_token", "wte": "batch_token"},
        )
        assert g.batch_size == Bb

    def test_genuine_batch_mismatch_still_raises(self):
        # Two real (non-broadcast) batch sizes are still an error.
        with pytest.raises(ValueError, match="same batch size"):
            Gradient(
                representation={"a": "factorized", "b": "factorized"},
                data={"a": self._embedding_factor(2), "b": self._embedding_factor(3)},
                layer_types={"a": "nn.Embedding", "b": "nn.Embedding"},
                indexing={"a": "batch_token", "b": "batch_token"},
            )


# --------------------------------------------------------------------------- #
# Gradient properties                                                          #
# --------------------------------------------------------------------------- #


class TestProperties:
    def test_layer_names(self):
        g = make_gradient(layers=("l1", "l2", "l3"))
        assert g.layer_names == {"l1", "l2", "l3"}

    def test_batch_size_materialized(self):
        g = make_gradient()
        assert g.batch_size == B

    def test_batch_size_factorized(self):
        g = make_gradient(repr_type="factorized")
        assert g.batch_size == B

    def test_token_dim_batch_token(self):
        g = make_gradient(indexing="batch_token")
        assert g.token_dim == {"l1": T, "l2": T}

    def test_token_dim_batch(self):
        g = make_gradient(indexing="batch")
        assert g.token_dim == {"l1": None, "l2": None}

    def test_token_dim_mixed(self):
        # Mixed: l1 is batch_token, l2 is batch.
        data = {"l1": mat_tensor_bt(), "l2": mat_tensor()}
        rep = {"l1": "materialized", "l2": "materialized"}
        g = Gradient(
            representation=rep,
            data=data,
            layer_types={"l1": "nn.Linear", "l2": "nn.Linear"},
            indexing={"l1": "batch_token", "l2": "batch"},
        )
        assert g.token_dim == {"l1": T, "l2": None}

    def test_device(self):
        g = make_gradient()
        assert g.device == torch.device("cpu")

    def test_dtype(self):
        g = make_gradient()
        assert g.dtype == torch.float32


# --------------------------------------------------------------------------- #
# Gradient.clone                                                               #
# --------------------------------------------------------------------------- #


class TestClone:
    def test_clone_materialized_is_independent(self):
        g = make_gradient()
        g2 = g.clone()
        for name in g.layer_names:
            g.data[name].fill_(0)
            assert not torch.allclose(g.data[name], g2.data[name])

    def test_clone_factorized_is_independent(self):
        g = make_gradient(repr_type="factorized")
        g2 = g.clone()
        for name in g.layer_names:
            g.data[name].activation.fill_(0)
            assert not torch.allclose(
                g.data[name].activation,
                g2.data[name].activation,
            )

    def test_clone_copies_representation(self):
        g = make_gradient()
        g2 = g.clone()
        assert g2.representation == g.representation
        assert g2.representation is not g.representation


# --------------------------------------------------------------------------- #
# Gradient.to                                                                  #
# --------------------------------------------------------------------------- #


class TestTo:
    def test_to_dtype_materialized(self):
        g = make_gradient()
        g2 = g.to(dtype=torch.float64)
        for v in g2.data.values():
            assert v.dtype == torch.float64

    def test_to_dtype_factorized(self):
        g = make_gradient(repr_type="factorized")
        g2 = g.to(dtype=torch.float64)
        for v in g2.data.values():
            assert v.activation.dtype == torch.float64
            assert v.pre_activation_grad.dtype == torch.float64

    def test_to_preserves_representation(self):
        g = make_gradient()
        g2 = g.to(dtype=torch.float16)
        assert g2.representation == g.representation


# --------------------------------------------------------------------------- #
# Gradient.materialize                                                         #
# --------------------------------------------------------------------------- #


class TestMaterialize:
    def test_factorized_becomes_materialized(self):
        g = make_gradient(repr_type="factorized")
        gm = g.materialize()
        for name in gm.layer_names:
            assert isinstance(gm.data[name], torch.Tensor)
            assert gm.representation[name] == "materialized"

    def test_factorized_shape(self):
        g = make_gradient(repr_type="factorized")
        gm = g.materialize()
        for v in gm.data.values():
            assert v.shape == (B, D_OUT * D_IN)

    def test_materialized_returns_self(self):
        g = make_gradient(repr_type="materialized")
        assert g.materialize() is g

    def test_mixed_representation(self):
        data = {"l1": factorized(), "l2": mat_tensor()}
        rep = {"l1": "factorized", "l2": "materialized"}
        g = Gradient(
            representation=rep,
            data=data,
            layer_types={"l1": "nn.Linear", "l2": "nn.Linear"},
        )
        gm = g.materialize()
        assert isinstance(gm.data["l1"], torch.Tensor)
        assert gm.representation["l1"] == "materialized"
        assert gm.representation["l2"] == "materialized"
        assert gm.data["l2"] is data["l2"]  # unchanged reference


# --------------------------------------------------------------------------- #
# Gradient.project                                                             #
# --------------------------------------------------------------------------- #

_PROJ = {"proj_max_batch_size": 8, "proj_type": "rademacher", "proj_seed": 0}


class TestProject:
    def test_factorize_stays_factorized(self):
        g = make_gradient(repr_type="factorized")
        p = g.project(
            random_project,
            {"__default__": {"factorize": True, "proj_dim": 32, **_PROJ}},
        )
        for name in p.layer_names:
            assert p.representation[name] == "factorized"
            assert p.layer_types[name] == "nn.Linear"
            f = p.data[name]
            assert f.activation.shape[-1] == 32
            assert f.pre_activation_grad.shape[-1] == 32
            assert f.module_kwargs is None  # factors are final, not re-preprocessed

    def test_materialize_collapses_to_proj_dim(self):
        g = make_gradient(repr_type="factorized")
        p = g.project(
            random_project,
            {"__default__": {"factorize": False, "proj_dim": 24, **_PROJ}},
        )
        for name in p.layer_names:
            assert p.representation[name] == "materialized"
            assert p.data[name].shape == (B, 24)
            assert p.indexing[name] == "batch"

    def test_per_layer_kwargs_override_default(self):
        data = {"l1": factorized(), "l2": factorized()}
        rep = {"l1": "factorized", "l2": "factorized"}
        g = Gradient(
            representation=rep,
            data=data,
            layer_types={"l1": "nn.Linear", "l2": "nn.Linear"},
        )
        p = g.project(
            random_project,
            {
                "l1": {"factorize": True, "proj_dim": 16, **_PROJ},
                "l2": {"factorize": False, "proj_dim": 16, **_PROJ},
            },
        )
        assert p.representation["l1"] == "factorized"
        assert p.representation["l2"] == "materialized"

    def test_unconfigured_layers_pass_through(self):
        # No entry and no "__default__" -> layer is left unchanged.
        g = make_gradient(repr_type="factorized")
        p = g.project(random_project, {})
        for name in p.layer_names:
            assert p.representation[name] == g.representation[name]
            assert p.data[name] is g.data[name]  # untouched reference

    def test_partial_config_projects_only_named_layer(self):
        data = {"l1": factorized(), "l2": factorized()}
        rep = {"l1": "factorized", "l2": "factorized"}
        g = Gradient(
            representation=rep,
            data=data,
            layer_types={"l1": "nn.Linear", "l2": "nn.Linear"},
        )
        p = g.project(
            random_project,
            {"l1": {"factorize": False, "proj_dim": 8, **_PROJ}},
        )
        assert p.representation["l1"] == "materialized"
        assert p.data["l1"].shape == (B, 8)
        assert p.data["l2"] is data["l2"]  # l2 untouched


# --------------------------------------------------------------------------- #
# Gradient.select_layers                                                       #
# --------------------------------------------------------------------------- #


class TestSelectLayers:
    def test_select_subset(self):
        g = make_gradient(layers=("l1", "l2", "l3"))
        g2 = g.select_layers(["l1", "l3"])
        assert g2.layer_names == {"l1", "l3"}
        assert set(g2.representation.keys()) == {"l1", "l3"}

    def test_select_preserves_layer_types(self):
        g = make_gradient(layers=("l1", "l2"))
        g2 = Gradient(
            representation=g.representation,
            data=g.data,
            layer_types={"l1": "Linear", "l2": "Linear"},
            indexing=g.indexing,
        )
        g3 = g2.select_layers(["l1"])
        assert g3.layer_types == {"l1": "Linear"}

    def test_select_unknown_layer_raises(self):
        g = make_gradient(layers=("l1",))
        with pytest.raises(KeyError, match="Unknown layers"):
            g.select_layers(["l99"])

    def test_include_derived_selects_virtual_invocation_layers(self):
        g = make_gradient(layers=("l1", "l1@2", "l1@3", "l2"))
        # Exact-name selection stays exact by default.
        assert g.select_layers(["l1"]).layer_names == {"l1"}
        # include_derived pulls in the @k invocation layers of requested names.
        sel = g.select_layers(["l1"], include_derived=True)
        assert sel.layer_names == {"l1", "l1@2", "l1@3"}
        assert g.select_layers(["l2"], include_derived=True).layer_names == {"l2"}


# --------------------------------------------------------------------------- #
# Gradient.concatenate                                                         #
# --------------------------------------------------------------------------- #


class TestConcatenate:
    def test_batch_concatenate_materialized(self):
        g1 = make_gradient()
        g2 = make_gradient()
        gc = g1.concatenate(g2, dim="batch")
        assert gc.batch_size == B * 2
        for v in gc.data.values():
            assert v.shape[0] == B * 2

    def test_token_concatenate(self):
        g1 = make_gradient(indexing="batch_token")
        g2 = make_gradient(indexing="batch_token")
        gc = g1.concatenate(g2, dim="token")
        assert gc.token_dim == {"l1": T * 2, "l2": T * 2}

    def test_token_concat_requires_batch_token(self):
        g1 = make_gradient()
        g2 = make_gradient()
        with pytest.raises(ValueError, match="batch_token"):
            g1.concatenate(g2, dim="token")

    def test_token_concat_requires_same_batch(self):
        g1 = make_gradient(indexing="batch_token")
        # Build a gradient with different batch size
        data = {"l1": mat_tensor_bt(b=3), "l2": mat_tensor_bt(b=3)}
        rep = {"l1": "materialized", "l2": "materialized"}
        g2 = Gradient(
            representation=rep,
            data=data,
            layer_types={"l1": "nn.Linear", "l2": "nn.Linear"},
            indexing={"l1": "batch_token", "l2": "batch_token"},
        )
        with pytest.raises(ValueError, match="batch size"):
            g1.concatenate(g2, dim="token")

    def test_token_concat_requires_all_batch_token(self):
        # Both gradients have consistent (mixed) indexing; token concat
        # must still be rejected because l2 is not batch_token.
        data = {"l1": mat_tensor_bt(), "l2": mat_tensor()}
        rep = {"l1": "materialized", "l2": "materialized"}
        g = Gradient(
            representation=rep,
            data=data,
            layer_types={"l1": "nn.Linear", "l2": "nn.Linear"},
            indexing={"l1": "batch_token", "l2": "batch"},
        )
        with pytest.raises(ValueError, match="batch_token"):
            g.concatenate(g, dim="token")

    def test_incompatible_representation_raises(self):
        g1 = make_gradient(repr_type="materialized")
        g2 = make_gradient(repr_type="factorized")
        with pytest.raises(ValueError, match="representations"):
            g1.concatenate(g2)


class TestVariableLengthBatchConcatenate:
    """Batch concat pads differing token lengths with inert zero-grad rows."""

    def _grad(self, data: dict) -> Gradient:
        return Gradient(
            representation=dict.fromkeys(data, "factorized"),
            data=data,
            layer_types=dict.fromkeys(data, "nn.Linear"),
            indexing=dict.fromkeys(data, "batch_token"),
        )

    def test_factorized_pads_and_scores_match(self):
        g1 = self._grad({"l1": factorized_bt(t=T)})
        g2 = self._grad({"l1": factorized_bt(t=T + 3)})
        gc = g1.concatenate(g2, dim="batch")
        assert gc.batch_size == 2 * B
        assert gc.token_dim == {"l1": T + 3}
        # Padded rows are inert: per-sample materialized gradients of the
        # block equal those of the originals.
        m = ops.materialize(gc.data["l1"], "nn.Linear")
        assert torch.allclose(m[:B], ops.materialize(g1.data["l1"], "nn.Linear"))
        assert torch.allclose(m[B:], ops.materialize(g2.data["l1"], "nn.Linear"))

    def test_longer_first_side_also_works(self):
        g1 = self._grad({"l1": factorized_bt(t=T + 5)})
        g2 = self._grad({"l1": factorized_bt(t=T)})
        gc = g1.concatenate(g2, dim="batch")
        assert gc.token_dim == {"l1": T + 5}
        m = ops.materialize(gc.data["l1"], "nn.Linear")
        assert torch.allclose(m[B:], ops.materialize(g2.data["l1"], "nn.Linear"))

    def test_seq_first_layout(self):
        def sf(t):
            return Factorized(
                activation=torch.randn(t, B, D_IN),
                pre_activation_grad=torch.randn(t, B, D_OUT),
                batch_first=False,
            )

        g1 = self._grad({"l1": sf(T)})
        g2 = self._grad({"l1": sf(T + 2)})
        gc = g1.concatenate(g2, dim="batch")
        assert gc.batch_size == 2 * B
        m = ops.materialize(gc.data["l1"], "nn.Linear")
        assert torch.allclose(m[:B], ops.materialize(g1.data["l1"], "nn.Linear"))

    def test_embedding_pads_activation_with_padding_idx(self):
        vocab, emb, pad_idx = 11, 5, 7
        mk = {"has_bias": False, "num_embeddings": vocab, "padding_idx": pad_idx}

        def emb_grad(t):
            return Gradient(
                representation={"e": "factorized"},
                data={
                    "e": Factorized(
                        activation=torch.randint(0, vocab, (B, t)),
                        pre_activation_grad=torch.randn(B, t, emb),
                        module_kwargs=mk,
                    ),
                },
                layer_types={"e": "nn.Embedding"},
                indexing={"e": "batch_token"},
            )

        g1, g2 = emb_grad(T), emb_grad(T + 4)
        gc = g1.concatenate(g2, dim="batch")
        f = gc.data["e"]
        # Padded index positions carry the configured padding_idx and a zero
        # gradient row.
        assert (f.activation[:B, T:] == pad_idx).all()
        assert (f.pre_activation_grad[:B, T:] == 0).all()
        m = ops.materialize(f, "nn.Embedding")
        assert torch.allclose(m[:B], ops.materialize(g1.data["e"], "nn.Embedding"))
        assert torch.allclose(m[B:], ops.materialize(g2.data["e"], "nn.Embedding"))

    def test_materialized_per_token_pads(self):
        def mat_grad(t):
            return Gradient(
                representation={"l1": "materialized"},
                data={"l1": mat_tensor_bt(t=t)},
                layer_types={"l1": "nn.Linear"},
                indexing={"l1": "batch_token"},
            )

        g1, g2 = mat_grad(T), mat_grad(T + 2)
        gc = g1.concatenate(g2, dim="batch")
        v = gc.data["l1"]
        assert v.shape == (2 * B, T + 2, D_OUT * D_IN)
        assert (v[:B, T:] == 0).all()

    def test_batch_indexed_feature_mismatch_still_raises(self):
        """Without a token axis (indexing='batch'), differing trailing dims are
        a real error, never padded.
        """
        g1 = make_gradient(layers=("l1",), repr_type="factorized")
        g2 = Gradient(
            representation={"l1": "factorized"},
            data={
                "l1": Factorized(
                    activation=torch.randn(B, D_IN + 1),
                    pre_activation_grad=torch.randn(B, D_OUT),
                ),
            },
            layer_types={"l1": "nn.Linear"},
            indexing={"l1": "batch"},
        )
        with pytest.raises(RuntimeError):
            g1.concatenate(g2, dim="batch")

    def test_mismatched_types_within_layer_raises(self):
        data1 = {"l1": mat_tensor()}
        data2 = {"l1": factorized()}
        rep1 = {"l1": "materialized"}
        rep2 = {"l1": "factorized"}
        g1 = Gradient(representation=rep1, data=data1, layer_types={"l1": "nn.Linear"})
        g2 = Gradient(representation=rep2, data=data2, layer_types={"l1": "nn.Linear"})
        with pytest.raises(ValueError, match="representation"):
            g1.concatenate(g2)


# --------------------------------------------------------------------------- #
# Gradient.aggregate                                                           #
# --------------------------------------------------------------------------- #


class TestAggregate:
    def test_aggregate_materialized_sum(self):
        g = make_gradient(indexing="batch_token")
        ga = g.aggregate(dim="token")
        assert all(v == "batch" for v in ga.indexing.values())
        for v in ga.data.values():
            assert v.shape == (B, D_OUT * D_IN)

    def test_aggregate_is_token_sum(self):
        # Aggregation is the chain rule: the token axis is summed out --
        # normalization (mean/masked losses) lives in the captured gradients.
        g = make_gradient(indexing="batch_token")
        ga = g.aggregate(dim="token")
        for name in g.layer_names:
            assert torch.allclose(ga.data[name], g.data[name].sum(dim=1))

    def test_aggregate_factorized_promotes_to_materialized(self):
        g = make_gradient(repr_type="factorized", indexing="batch_token")
        ga = g.aggregate(dim="token")
        for name in ga.layer_names:
            assert ga.representation[name] == "materialized"
            assert isinstance(ga.data[name], torch.Tensor)

    def test_aggregate_batch_layer_passthrough(self):
        # A "batch" layer is passed through unchanged; no error is raised.
        data = {"l1": mat_tensor_bt(), "l2": mat_tensor()}
        rep = {"l1": "materialized", "l2": "materialized"}
        g = Gradient(
            representation=rep,
            data=data,
            layer_types={"l1": "nn.Linear", "l2": "nn.Linear"},
            indexing={"l1": "batch_token", "l2": "batch"},
        )
        ga = g.aggregate(dim="token")
        assert ga.data["l1"].shape == (B, D_OUT * D_IN)  # aggregated
        assert ga.data["l2"].shape == (B, D_OUT * D_IN)  # unchanged
        assert all(v == "batch" for v in ga.indexing.values())

    def test_aggregate_all_batch_still_works(self):
        # If no layers are batch_token, aggregate is a no-op on data.
        g = make_gradient(indexing="batch")
        ga = g.aggregate(dim="token")
        assert all(v == "batch" for v in ga.indexing.values())
        for name in g.layer_names:
            assert torch.equal(ga.data[name], g.data[name])

    def test_aggregate_unsupported_dim(self):
        g = make_gradient(indexing="batch_token")
        with pytest.raises(NotImplementedError):
            g.aggregate(dim="batch")  # type: ignore[arg-type]

    def test_aggregate_mean_mode_removed(self):
        # mode was removed: a mean over tokens double-applies the loss's own
        # normalization and never computes a gradient.
        g = make_gradient(indexing="batch_token")
        with pytest.raises(TypeError):
            g.aggregate(dim="token", mode="mean")  # type: ignore[call-arg]


# --------------------------------------------------------------------------- #
# Gradient.slice                                                               #
# --------------------------------------------------------------------------- #


class TestSlice:
    def test_slice_batch_int(self):
        g = make_gradient()
        gs = g.slice(dim="batch", index=0)
        assert gs.batch_size == 1

    def test_slice_batch_slice_object(self):
        g = make_gradient()
        gs = g.slice(dim="batch", index=slice(0, 1))
        assert gs.batch_size == 1

    def test_slice_token(self):
        g = make_gradient(indexing="batch_token")
        gs = g.slice(dim="token", index=0)
        assert gs.token_dim == {"l1": 1, "l2": 1}

    def test_slice_token_requires_all_batch_token(self):
        # Mixed indexing: token slice must be rejected.
        data = {"l1": mat_tensor_bt(), "l2": mat_tensor()}
        rep = {"l1": "materialized", "l2": "materialized"}
        g = Gradient(
            representation=rep,
            data=data,
            layer_types={"l1": "nn.Linear", "l2": "nn.Linear"},
            indexing={"l1": "batch_token", "l2": "batch"},
        )
        with pytest.raises(ValueError, match="batch_token"):
            g.slice(dim="token", index=0)

    def test_slice_factorized(self):
        g = make_gradient(repr_type="factorized")
        gs = g.slice(dim="batch", index=0)
        for v in gs.data.values():
            assert isinstance(v, Factorized)
            assert v.activation.shape[0] == 1

    def test_slice_preserves_representation(self):
        g = make_gradient()
        gs = g.slice(dim="batch", index=0)
        assert gs.representation == g.representation

    def test_batch_slice_copies_broadcast_layer(self):
        # A broadcast (batch-1) layer -- e.g. a positional embedding fed an
        # unbatched index tensor -- shares its single row across the batch, so
        # slicing any sample must copy that row instead of raising IndexError.
        Bn, Tn, Dn = 4, 3, 5
        fc = Factorized(torch.randn(Bn, Tn, Dn), torch.randn(Bn, Tn, Dn))
        pos = Factorized(torch.arange(Tn).unsqueeze(0), torch.randn(1, Tn, Dn))
        g = Gradient(
            representation={"fc": "factorized", "pos": "factorized"},
            data={"fc": fc, "pos": pos},
            layer_types={"fc": "nn.Linear", "pos": "nn.Embedding"},
            indexing={"fc": "batch_token", "pos": "batch_token"},
        )
        for i in range(Bn):
            s = g.slice(dim="batch", index=i)
            assert torch.equal(s.data["fc"].activation[0], fc.activation[i])
            assert torch.equal(
                s.data["pos"].pre_activation_grad[0],
                pos.pre_activation_grad[0],
            )  # shared row copied
        s = g.slice(dim="batch", index=[1, 3])
        assert s.data["fc"].activation.shape[0] == 2
        assert s.data["pos"].activation.shape[0] == 2
        assert torch.equal(s.data["pos"].activation[0], s.data["pos"].activation[1])

    def test_batch_slice_preserves_param_grad_without_slicing(self):
        param_grad = torch.randn(7, 3)
        g = Gradient(
            representation={"sample": "materialized", "weight": "materialized"},
            data={"sample": torch.randn(B, 5), "weight": param_grad},
            layer_types={"sample": "nn.Linear", "weight": ops.PARAM_GRAD_TYPES},
            indexing={"sample": "batch", "weight": "batch"},
        )
        sliced = g.slice(dim="batch", index=0)
        assert sliced.data["sample"].shape[0] == 1
        assert torch.equal(sliced.data["weight"], param_grad)


# --------------------------------------------------------------------------- #
# Broadcast (batch-collapsed) layers under batch operations                    #
# --------------------------------------------------------------------------- #


def _broadcast_gradient(B, T=3, D=5, seed=0):
    """Gradient with a per-sample 'fc' layer (B, T, D) and a broadcast 'pos'
    embedding (batch 1) whose single row is shared across the batch.
    """
    g = torch.Generator().manual_seed(seed)
    fc = Factorized(
        torch.randn(B, T, D, generator=g),
        torch.randn(B, T, D, generator=g),
    )
    pos = Factorized(
        torch.arange(T).unsqueeze(0),
        torch.randn(1, T, D, generator=g),
        module_kwargs={"has_bias": False, "num_embeddings": T, "padding_idx": None},
    )
    return Gradient(
        representation={"fc": "factorized", "pos": "factorized"},
        data={"fc": fc, "pos": pos},
        layer_types={"fc": "nn.Linear", "pos": "nn.Embedding"},
        indexing={"fc": "batch_token", "pos": "batch_token"},
    )


class TestBroadcastConcatenate:
    def test_batch_concat_unifies_broadcast_row(self):
        """Broadcast rows merge into the batch-size-weighted average; per-sample
        layers concatenate normally.
        """
        g1, g2 = _broadcast_gradient(4, seed=0), _broadcast_gradient(2, seed=1)
        out = g1.concatenate(g2, dim="batch")
        out.validate()
        assert out.batch_size == 6
        assert out.data["fc"].activation.shape[0] == 6  # plain cat
        # pos stays factorized: same position ids, g weighted-averaged 4:2.
        assert out.representation["pos"] == "factorized"
        expected_g = (
            4 * g1.data["pos"].pre_activation_grad
            + 2 * g2.data["pos"].pre_activation_grad
        ) / 6
        assert torch.allclose(out.data["pos"].pre_activation_grad, expected_g)
        assert torch.equal(out.data["pos"].activation, g1.data["pos"].activation)

    def test_batch_concat_differing_activations_materializes(self):
        """Broadcast rows with different activation factors are averaged in the
        materialized domain (the layer flips to 'materialized').
        """
        g1, g2 = _broadcast_gradient(4, seed=0), _broadcast_gradient(2, seed=1)
        # give g2's pos a different id pattern
        pos2 = g2.data["pos"]
        g2.data["pos"] = Factorized(
            activation=torch.flip(pos2.activation, dims=[1]),
            pre_activation_grad=pos2.pre_activation_grad,
            module_kwargs=pos2.module_kwargs,
        )
        out = g1.concatenate(g2, dim="batch")
        out.validate()
        assert out.representation["pos"] == "materialized"
        m1 = ops.materialize(g1.data["pos"], "nn.Embedding")
        m2 = ops.materialize(g2.data["pos"], "nn.Embedding")
        assert torch.allclose(out.data["pos"], (4 * m1 + 2 * m2) / 6)

    def test_batch_concat_materialized_broadcast(self):
        """Materialized broadcast rows are weighted-averaged directly."""
        r1, r2 = torch.randn(1, 7), torch.randn(1, 7)

        def make(B, row):
            return Gradient(
                representation={"fc": "materialized", "pos": "materialized"},
                data={"fc": torch.randn(B, 7), "pos": row},
                layer_types={"fc": "nn.Linear", "pos": "nn.Embedding"},
            )

        out = make(3, r1).concatenate(make(5, r2), dim="batch")
        assert torch.allclose(out.data["pos"], (3 * r1 + 5 * r2) / 8)
        assert out.data["fc"].shape[0] == 8

    def test_batch_concat_broadcast_on_one_side_raises(self):
        g1 = _broadcast_gradient(4, seed=0)
        g2 = _broadcast_gradient(2, seed=1)
        # make g2's pos per-sample (batch 2) instead of broadcast
        g2.data["pos"] = Factorized(
            activation=torch.arange(3).repeat(2, 1),
            pre_activation_grad=torch.randn(2, 3, 5),
        )
        with pytest.raises(ValueError, match="broadcast"):
            g1.concatenate(g2, dim="batch")

    def test_genuine_batch_one_gradients_still_cat(self):
        """Two all-batch-1 gradients are NOT broadcast -- they concatenate to
        batch 2.
        """
        g1, g2 = _broadcast_gradient(1, seed=0), _broadcast_gradient(1, seed=1)
        out = g1.concatenate(g2, dim="batch")
        assert out.batch_size == 2
        assert out.data["pos"].activation.shape[0] == 2  # plain cat


class TestBroadcastSimilarity:
    def test_reduce_all_expands_broadcast_layer(self):
        """The shared row contributes identically to every (i, j) pair, so the
        full-model cross-gram is the fc gram plus a constant offset.
        """
        g1, g2 = _broadcast_gradient(4, seed=0), _broadcast_gradient(2, seed=1)
        total = g1.similarity(g2, metric="dot", reduce="all")
        assert total.shape == (4, 2)
        per_layer = g1.similarity(g2, metric="dot", reduce="none")
        assert per_layer["pos"].shape == (1, 1)  # raw broadcast gram
        expected = per_layer["fc"] + per_layer["pos"].expand(4, 2)
        assert torch.allclose(total, expected, atol=1e-5)

    def test_reduce_all_cosine_no_crash(self):
        g1, g2 = _broadcast_gradient(4, seed=0), _broadcast_gradient(2, seed=1)
        total = g1.similarity(g2, metric="cosine", reduce="all")
        assert total.shape == (4, 2)
        assert bool(torch.isfinite(total).all())
        assert float(total.abs().max()) <= 1 + 1e-5


# --------------------------------------------------------------------------- #
# Gradient.similarity                                                          #
# --------------------------------------------------------------------------- #


class TestSimilarity:
    """Gradient.similarity -- the cross-gram gradient-similarity primitive."""

    def test_diagonal_equals_aligned_dot(self):
        """The diagonal of the self cross-gram equals the aligned per-sample dot."""
        g = make_gradient(repr_type="factorized", indexing="batch_token")
        cross = g.similarity(g, mode="factorized")
        for name, matrix in cross.items():
            v = g.data[name]
            aligned = ops.dot(v, v, g.layer_types[name])
            assert matrix.shape == (B, B)
            assert torch.allclose(matrix.diagonal(), aligned, atol=1e-4, rtol=1e-4)

    def test_factorized_does_not_materialize(self):
        """The factorized mode must not call ops.materialize on factorized layers."""
        import dattri_llm.gradient.ops as ops_mod

        g = make_gradient(repr_type="factorized", indexing="batch_token")
        calls = {"n": 0}
        orig = ops_mod.materialize

        def _counting(*args, **kwargs):
            calls["n"] += 1
            return orig(*args, **kwargs)

        ops_mod.materialize = _counting
        try:
            g.similarity(g, mode="factorized")
        finally:
            ops_mod.materialize = orig
        assert calls["n"] == 0, "factorized mode should not materialize"

    def test_factorized_equals_materialized(self):
        g = make_gradient(repr_type="factorized", indexing="batch_token")
        fac = g.similarity(g, mode="factorized")
        mat = g.similarity(g, mode="materialized")
        assert set(fac.keys()) == set(mat.keys())
        for name in fac:
            assert torch.allclose(fac[name], mat[name], atol=1e-4, rtol=1e-4)

    def test_cross_shape_differing_batch(self):
        """Cross-gram against a target with a different batch size is
        (B_self, B_other).
        """
        g = make_gradient(repr_type="factorized")
        other = Gradient(
            representation=dict.fromkeys(("l1", "l2"), "factorized"),
            data={"l1": factorized(b=5), "l2": factorized(b=5)},
            layer_types={"l1": "nn.Linear", "l2": "nn.Linear"},
        )
        cross = g.similarity(other)
        for name in ("l1", "l2"):
            assert cross[name].shape == (B, 5)

    def test_skips_layers_absent_in_other(self):
        g = make_gradient(layers=("l1", "l2"), repr_type="factorized")
        other = make_gradient(layers=("l1",), repr_type="factorized")
        cross = g.similarity(other)
        assert set(cross.keys()) == {"l1"}

    # -- metric --------------------------------------------------------------

    def test_cosine_self_diagonal_is_one(self):
        """Cosine similarity of each sample with itself is 1."""
        g = make_gradient(repr_type="factorized", indexing="batch_token")
        cos = g.similarity(g, metric="cosine")
        for matrix in cos.values():
            assert torch.allclose(
                matrix.diagonal(),
                torch.ones(B),
                atol=1e-4,
                rtol=1e-4,
            )

    def test_cosine_in_unit_range(self):
        g = make_gradient(repr_type="factorized")
        other = make_gradient(repr_type="factorized")
        cos = g.similarity(other, metric="cosine")
        for matrix in cos.values():
            assert (matrix.abs() <= 1 + 1e-4).all()

    def test_cosine_factorized_equals_materialized(self):
        g = make_gradient(repr_type="factorized", indexing="batch_token")
        other = make_gradient(repr_type="factorized", indexing="batch_token")
        fac = g.similarity(other, metric="cosine", mode="factorized")
        mat = g.similarity(other, metric="cosine", mode="materialized")
        for name in fac:
            assert torch.allclose(fac[name], mat[name], atol=1e-4, rtol=1e-4)

    # -- reduce --------------------------------------------------------------

    def test_reduce_none_returns_per_layer_matrices(self):
        g = make_gradient(repr_type="factorized")
        out = g.similarity(g, reduce="none")
        assert set(out.keys()) == g.layer_names
        for matrix in out.values():
            assert matrix.shape == (B, B)

    def test_reduce_all_returns_single_matrix(self):
        """``reduce="all"`` returns one (B_self, B_other) matrix, not a dict."""
        g = make_gradient(repr_type="factorized")
        out = g.similarity(g, reduce="all")
        assert isinstance(out, torch.Tensor)
        assert out.shape == (B, B)

    def test_reduce_all_equals_sum_over_layers(self):
        """The overall matrix is the per-layer matrices summed over layers
        (the full-model gradient cross-gram).
        """
        g = make_gradient(repr_type="factorized")
        per_layer = g.similarity(g, reduce="none")
        overall = g.similarity(g, reduce="all")
        expected = torch.stack(list(per_layer.values())).sum(0)
        assert torch.allclose(overall, expected, atol=1e-4, rtol=1e-4)

    def test_reduce_all_no_shared_layers_raises(self):
        g = make_gradient(layers=("l1",), repr_type="factorized")
        other = make_gradient(layers=("l2",), repr_type="factorized")
        with pytest.raises(ValueError, match="No shared layers"):
            g.similarity(other, reduce="all")

    # -- validation ----------------------------------------------------------

    def test_invalid_arguments_raise(self):
        g = make_gradient(repr_type="factorized")
        with pytest.raises(ValueError, match="metric"):
            g.similarity(g, metric="l2")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="reduce"):
            g.similarity(g, reduce="layer")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="mode"):
            g.similarity(g, mode="ghost")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# batch_first -- sequence-first ``(T, B, ...)`` captures                         #
# --------------------------------------------------------------------------- #


def _seq_first_pair():
    """Return (batch_first_grad, seq_first_grad) holding the *same* underlying
    per-sample gradient for one ``batch_token`` linear layer -- the seq-first one
    is the batch-first factors with the leading two axes swapped.
    """
    torch.manual_seed(0)
    a = torch.randn(B, T, D_IN)
    g = torch.randn(B, T, D_OUT)

    bf = Gradient(
        representation={"l": "factorized"},
        data={"l": Factorized(a, g)},
        layer_types={"l": "nn.Linear"},
        indexing={"l": "batch_token"},
    )
    sf = Gradient(
        representation={"l": "factorized"},
        data={"l": Factorized(a.transpose(0, 1), g.transpose(0, 1), batch_first=False)},
        layer_types={"l": "nn.Linear"},
        indexing={"l": "batch_token"},
    )
    return bf, sf


class TestBatchFirst:
    def test_default_is_batch_first(self):
        assert (
            Factorized(torch.randn(B, D_IN), torch.randn(B, D_OUT)).batch_first is True
        )

    def test_as_batch_first_noop_when_already(self):
        f = Factorized(torch.randn(B, T, D_IN), torch.randn(B, T, D_OUT))
        assert f.as_batch_first() is f

    def test_as_batch_first_transposes(self):
        f = Factorized(
            torch.randn(T, B, D_IN),
            torch.randn(T, B, D_OUT),
            batch_first=False,
        )
        bf = f.as_batch_first()
        assert bf.batch_first is True
        assert bf.activation.shape == (B, T, D_IN)
        assert bf.pre_activation_grad.shape == (B, T, D_OUT)

    def test_seq_first_batch_size(self):
        # Regression: a (T, B, d) layer must report B, not T.
        _bf, sf = _seq_first_pair()
        assert sf.batch_size == B

    def test_seq_first_validate_ok(self):
        _bf, sf = _seq_first_pair()
        sf.validate()  # must not raise "All layers must have the same batch size"

    def test_seq_first_token_dim(self):
        _bf, sf = _seq_first_pair()
        assert sf.token_dim["l"] == T

    def test_mixed_orientation_validates(self):
        # A batch-first layer and a seq-first layer with the same batch size B.
        a, g = torch.randn(B, T, D_IN), torch.randn(B, T, D_OUT)
        grad = Gradient(
            representation={"bf": "factorized", "sf": "factorized"},
            data={
                "bf": Factorized(a, g),
                "sf": Factorized(
                    a.transpose(0, 1),
                    g.transpose(0, 1),
                    batch_first=False,
                ),
            },
            layer_types={"bf": "nn.Linear", "sf": "nn.Linear"},
            indexing={"bf": "batch_token", "sf": "batch_token"},
        )
        grad.validate()
        assert grad.batch_size == B

    def test_seq_first_materialize_matches_batch_first(self):
        bf, sf = _seq_first_pair()
        assert torch.allclose(
            bf.materialize().data["l"],
            sf.materialize().data["l"],
            atol=1e-5,
        )

    def test_seq_first_similarity_matches_batch_first(self):
        bf, sf = _seq_first_pair()
        assert torch.allclose(
            bf.similarity(bf)["l"],
            sf.similarity(sf)["l"],
            atol=1e-4,
        )

    def test_seq_first_aggregate_matches_batch_first(self):
        bf, sf = _seq_first_pair()
        assert torch.allclose(
            bf.aggregate().data["l"],
            sf.aggregate().data["l"],
            atol=1e-5,
        )

    def test_clone_and_to_preserve_flag(self):
        _bf, sf = _seq_first_pair()
        assert sf.clone().data["l"].batch_first is False
        assert sf.to(dtype=torch.float64).data["l"].batch_first is False

    def test_slice_batch_preserves_flag_and_layout(self):
        _bf, sf = _seq_first_pair()
        sl = sf.slice("batch", 0)  # one sample
        assert sl.data["l"].batch_first is False
        assert sl.batch_size == 1
        # Seq-first activation stays (T, 1, I).
        assert sl.data["l"].activation.shape == (T, 1, D_IN)

    def test_concatenate_batch_matches_batch_first(self):
        bf, sf = _seq_first_pair()
        cat_bf = bf.concatenate(bf, dim="batch")
        cat_sf = sf.concatenate(sf, dim="batch")
        assert cat_sf.batch_size == 2 * B
        assert torch.allclose(
            cat_bf.materialize().data["l"],
            cat_sf.materialize().data["l"],
            atol=1e-5,
        )

    def test_concatenate_mismatched_layout_raises(self):
        bf, sf = _seq_first_pair()
        with pytest.raises(ValueError, match="batch_first"):
            bf.concatenate(sf, dim="batch")

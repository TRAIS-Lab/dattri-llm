"""Unit tests for dattri_llm.gradient.gradient.Gradient."""

from __future__ import annotations

import pytest
import torch

from dattri_llm.gradient.gradient import Factorized, Gradient


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

B, T, I, O = 2, 4, 8, 16  # batch, token, in_features, out_features


def mat_tensor(b=B, feat=O * I) -> torch.Tensor:
    """Materialized gradient tensor [B, feat]."""
    return torch.randn(b, feat)


def mat_tensor_bt(b=B, t=T, feat=O * I) -> torch.Tensor:
    """Materialized gradient tensor [B, T, feat]."""
    return torch.randn(b, t, feat)


def factorized(b=B) -> Factorized:
    """Factorized gradient [B, I] x [B, O]."""
    return Factorized(
        activation=torch.randn(b, I),
        pre_activation_grad=torch.randn(b, O),
    )


def factorized_bt(b=B, t=T) -> Factorized:
    """Factorized gradient [B, T, I] x [B, T, O]."""
    return Factorized(
        activation=torch.randn(b, t, I),
        pre_activation_grad=torch.randn(b, t, O),
    )


def make_gradient(
    layers=("l1", "l2"),
    repr_type="materialized",
    indexing="batch",
    projection_dim=None,
) -> Gradient:
    if repr_type == "factorized":
        fn = factorized_bt if indexing == "batch_token" else factorized
        data = {name: fn() for name in layers}
    elif repr_type == "projected":
        assert projection_dim is not None
        feat = projection_dim
        fn = mat_tensor_bt if indexing == "batch_token" else mat_tensor
        data = {name: fn(feat=feat) for name in layers}
    else:
        fn = mat_tensor_bt if indexing == "batch_token" else mat_tensor
        data = {name: fn() for name in layers}

    representation = {name: repr_type for name in layers}
    return Gradient(
        representation=representation,
        data=data,
        indexing=indexing,
        projection_dim=projection_dim,
    )


# --------------------------------------------------------------------------- #
# Factorized                                                                   #
# --------------------------------------------------------------------------- #


class TestFactorized:
    def test_materialize_2d(self):
        f = Factorized(
            activation=torch.ones(B, I),
            pre_activation_grad=torch.ones(B, O),
        )
        out = f.materialize()
        assert out.shape == (B, O * I)

    def test_materialize_3d(self):
        f = factorized_bt()
        out = f.materialize()
        assert out.shape == (B, T, O * I)

    def test_materialize_invalid_ndim(self):
        f = Factorized(
            activation=torch.randn(B, T, I, 1),
            pre_activation_grad=torch.randn(B, T, O, 1),
        )
        with pytest.raises(ValueError):
            f.materialize()

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

    def test_valid_projected(self):
        g = make_gradient(repr_type="projected", projection_dim=32)
        g.validate()

    def test_valid_batch_token(self):
        g = make_gradient(repr_type="materialized", indexing="batch_token")
        g.validate()

    def test_empty_data(self):
        with pytest.raises(ValueError, match="empty"):
            Gradient(representation={}, data={}).validate()

    def test_missing_representation_key(self):
        data = {"l1": mat_tensor(), "l2": mat_tensor()}
        rep = {"l1": "materialized"}  # missing l2
        g = Gradient(representation=rep, data=data)
        with pytest.raises(ValueError, match="Missing representation"):
            g.validate()

    def test_factorized_wrong_data_type(self):
        data = {"l1": mat_tensor()}
        rep = {"l1": "factorized"}
        g = Gradient(representation=rep, data=data)
        with pytest.raises(TypeError, match="Factorized"):
            g.validate()

    def test_materialized_wrong_data_type(self):
        data = {"l1": factorized()}
        rep = {"l1": "materialized"}
        g = Gradient(representation=rep, data=data)
        with pytest.raises(TypeError, match="Tensor"):
            g.validate()

    def test_projected_missing_projection_dim(self):
        data = {"l1": mat_tensor()}
        rep = {"l1": "projected"}
        g = Gradient(representation=rep, data=data, projection_dim=None)
        with pytest.raises(ValueError, match="projection_dim"):
            g.validate()

    def test_projected_wrong_last_dim(self):
        data = {"l1": mat_tensor(feat=16)}
        rep = {"l1": "projected"}
        g = Gradient(representation=rep, data=data, projection_dim=32)
        with pytest.raises(ValueError, match="wrong projection dimension"):
            g.validate()

    def test_batch_size_mismatch(self):
        data = {"l1": mat_tensor(b=2), "l2": mat_tensor(b=3)}
        rep = {"l1": "materialized", "l2": "materialized"}
        g = Gradient(representation=rep, data=data)
        with pytest.raises(ValueError, match="batch size"):
            g.validate()

    def test_token_dim_mismatch(self):
        data = {
            "l1": mat_tensor_bt(t=4),
            "l2": mat_tensor_bt(t=6),
        }
        rep = {"l1": "materialized", "l2": "materialized"}
        g = Gradient(representation=rep, data=data, indexing="batch_token")
        with pytest.raises(ValueError, match="token dimension"):
            g.validate()


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
        assert g.token_dim == T

    def test_token_dim_batch(self):
        g = make_gradient(indexing="batch")
        assert g.token_dim is None

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
                g.data[name].activation, g2.data[name].activation
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
            assert v.shape == (B, O * I)

    def test_materialized_returns_self(self):
        g = make_gradient(repr_type="materialized")
        assert g.materialize() is g

    def test_projected_returns_self(self):
        g = make_gradient(repr_type="projected", projection_dim=32)
        assert g.materialize() is g

    def test_mixed_representation(self):
        data = {"l1": factorized(), "l2": mat_tensor()}
        rep = {"l1": "factorized", "l2": "materialized"}
        g = Gradient(representation=rep, data=data)
        gm = g.materialize()
        assert isinstance(gm.data["l1"], torch.Tensor)
        assert gm.representation["l1"] == "materialized"
        assert gm.representation["l2"] == "materialized"
        assert gm.data["l2"] is data["l2"]  # unchanged reference


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
        assert gc.token_dim == T * 2

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
        g2 = Gradient(representation=rep, data=data, indexing="batch_token")
        with pytest.raises(ValueError, match="batch size"):
            g1.concatenate(g2, dim="token")

    def test_incompatible_representation_raises(self):
        g1 = make_gradient(repr_type="materialized")
        g2 = make_gradient(repr_type="factorized")
        with pytest.raises(ValueError, match="representations"):
            g1.concatenate(g2)

    def test_mismatched_types_within_layer_raises(self):
        data1 = {"l1": mat_tensor()}
        data2 = {"l1": factorized()}
        rep1 = {"l1": "materialized"}
        rep2 = {"l1": "factorized"}
        g1 = Gradient(representation=rep1, data=data1)
        g2 = Gradient(representation=rep2, data=data2)
        with pytest.raises(ValueError):
            g1.concatenate(g2)


# --------------------------------------------------------------------------- #
# Gradient.aggregate                                                           #
# --------------------------------------------------------------------------- #


class TestAggregate:
    def test_aggregate_materialized_sum(self):
        g = make_gradient(indexing="batch_token")
        ga = g.aggregate(dim="token", mode="sum")
        assert ga.indexing == "batch"
        for v in ga.data.values():
            assert v.shape == (B, O * I)

    def test_aggregate_materialized_mean(self):
        g = make_gradient(indexing="batch_token")
        ga = g.aggregate(dim="token", mode="mean")
        for v in ga.data.values():
            assert v.shape == (B, O * I)

    def test_aggregate_factorized_promotes_to_materialized(self):
        g = make_gradient(repr_type="factorized", indexing="batch_token")
        ga = g.aggregate(dim="token", mode="sum")
        for name in ga.layer_names:
            assert ga.representation[name] == "materialized"
            assert isinstance(ga.data[name], torch.Tensor)

    def test_aggregate_requires_batch_token(self):
        g = make_gradient(indexing="batch")
        with pytest.raises(ValueError, match="batch_token"):
            g.aggregate(dim="token")

    def test_aggregate_unsupported_dim(self):
        g = make_gradient(indexing="batch_token")
        with pytest.raises(NotImplementedError):
            g.aggregate(dim="batch")  # type: ignore[arg-type]

    def test_aggregate_invalid_mode(self):
        g = make_gradient(indexing="batch_token")
        with pytest.raises(ValueError, match="mode"):
            g.aggregate(dim="token", mode="max")  # type: ignore[arg-type]


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
        assert gs.token_dim == 1

    def test_slice_token_requires_batch_token(self):
        g = make_gradient(indexing="batch")
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


# --------------------------------------------------------------------------- #
# Gradient.similarity                                                          #
# --------------------------------------------------------------------------- #


class TestSimilarity:
    def test_dot_reduce_all_scalar(self):
        g = make_gradient()
        sim = g.similarity(g, metric="dot", reduce="all")
        assert sim.ndim == 0

    def test_dot_reduce_layer_dict(self):
        g = make_gradient()
        sim = g.similarity(g, metric="dot", reduce="layer")
        assert set(sim.keys()) == g.layer_names

    def test_dot_reduce_none_shape(self):
        g = make_gradient()
        sim = g.similarity(g, metric="dot", reduce="none")
        for v in sim.values():
            assert v.shape == (B,)

    def test_cosine_self_similarity_is_positive(self):
        g = make_gradient()
        sim = g.similarity(g, metric="cosine", reduce="all")
        assert sim.item() > 0

    def test_similarity_incompatible_representation(self):
        g1 = make_gradient(repr_type="materialized")
        g2 = make_gradient(repr_type="projected", projection_dim=O * I)
        with pytest.raises(ValueError, match="representations"):
            g1.similarity(g2)

    def test_similarity_layer_sets_differ(self):
        g1 = make_gradient(layers=("l1", "l2"))
        g2 = make_gradient(layers=("l1", "l3"))
        with pytest.raises(ValueError, match="Layer sets differ"):
            g1.similarity(g2)

    def test_similarity_batch_size_mismatch(self):
        g1 = make_gradient()
        data = {"l1": mat_tensor(b=3), "l2": mat_tensor(b=3)}
        rep = {"l1": "materialized", "l2": "materialized"}
        g2 = Gradient(representation=rep, data=data)
        with pytest.raises(ValueError, match="Batch sizes differ"):
            g1.similarity(g2)

    def test_invalid_metric(self):
        g = make_gradient()
        with pytest.raises(ValueError, match="metric"):
            g.similarity(g, metric="l2")  # type: ignore[arg-type]

    def test_invalid_reduce(self):
        g = make_gradient()
        with pytest.raises(ValueError, match="reduce"):
            g.similarity(g, reduce="mean")  # type: ignore[arg-type]

    def test_similarity_layer_types_mismatch(self):
        data = {"l1": mat_tensor(), "l2": mat_tensor()}
        rep = {"l1": "materialized", "l2": "materialized"}
        g1 = Gradient(representation=rep, data=data, layer_types={"l1": "Linear", "l2": "Linear"})
        g2 = Gradient(representation=rep, data=data, layer_types={"l1": "Linear", "l2": "Conv"})
        with pytest.raises(ValueError, match="Layer types differ"):
            g1.similarity(g2)

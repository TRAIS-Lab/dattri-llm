"""Layer-type-aware gradient operations for per-sample gradient computation.

This package re-exports the full surface of its submodules so existing code
keeps importing from ``dattri_llm.gradient.ops`` unchanged:

* :mod:`~dattri_llm.gradient.ops.types` -- layer-type constants and predicates.
* :mod:`~dattri_llm.gradient.ops.norm` -- normalization x_hat / bias augmentation.
* :mod:`~dattri_llm.gradient.ops.preprocess` -- raw-capture preprocessing
  (incl. conv im2col) and module-kwargs extraction.
* :mod:`~dattri_llm.gradient.ops.materialize` -- per-sample weight gradients.
* :mod:`~dattri_llm.gradient.ops.dot` -- dot products, grams, norms, and the
  factorized-vs-materialized routing heuristic.
* :mod:`~dattri_llm.gradient.ops.projection` -- TRAK/LoGRA random projection.
* :mod:`~dattri_llm.gradient.ops.kronecker` -- K-FAC / EK-FAC / Fisher kernels
  and streaming accumulators.
"""

from dattri_llm.gradient.ops.dot import (
    _cross_dot,
    _cross_gram,
    _dot,
    _grad_norm_sq,
    _pairwise_dot,
    cross_dot,
    dot,
    effective_dims,
    grad_norm_sq,
    maybe_use_materialized_gram,
    maybe_use_materialized_norm,
    pairwise_dot,
)
from dattri_llm.gradient.ops.kronecker import (
    FisherAccumulator,
    KroneckerAccumulator,
    LayerFisherAccumulator,
    LayerKroneckerAccumulator,
    _drop_gradient_free_rows,
    _ekfac_materialize,
    _fim,
    _flatten_for_kfac,
    _kfac,
    _kfac_cross,
    ekfac_materialize,
    fim,
    kfac,
    kfac_cross,
    kfac_eigh,
    sym_inverse,
)
from dattri_llm.gradient.ops.materialize import (
    _materialize,
    _materialize_embedding,
    materialize,
)
from dattri_llm.gradient.ops.norm import (
    _augment_channel_norm,
    _augment_token_norm,
    _compute_group_norm_x_hat,
    _compute_layer_norm_x_hat,
    _compute_rms_x_hat,
)
from dattri_llm.gradient.ops.preprocess import (
    _CONV_IM2COL,
    _conv1d_im2col,
    _conv2d_im2col,
    _conv3d_im2col,
    _conv_spatial_rank,
    _preprocess_conv_transpose,
    _preprocess_embedding_bag,
    _preprocess_factorized,
    _to_3d,
    extract_module_kwargs,
    preprocess_factorized,
)
from dattri_llm.gradient.ops.projection import (
    _apply_projector,
    _project_factorized,
    _project_materialized,
    project_factorized,
    project_layer,
    project_materialized,
)
from dattri_llm.gradient.ops.types import (
    ALL_LAYER_TYPES,
    CONV_TRANSPOSE_TYPES,
    CONV_TYPES,
    EMBEDDING_TYPES,
    LINEAR_TYPES,
    NORM_TYPES,
    PARAM_GRAD_TYPES,
    canonical_class_name,
    is_conv,
    is_conv_transpose,
    is_embedding,
    is_linear,
    is_norm,
)

__all__ = [
    "ALL_LAYER_TYPES",
    "CONV_TRANSPOSE_TYPES",
    "CONV_TYPES",
    "EMBEDDING_TYPES",
    "LINEAR_TYPES",
    "NORM_TYPES",
    "PARAM_GRAD_TYPES",
    "_CONV_IM2COL",
    "FisherAccumulator",
    "KroneckerAccumulator",
    "LayerFisherAccumulator",
    "LayerKroneckerAccumulator",
    "_apply_projector",
    "_augment_channel_norm",
    "_augment_token_norm",
    "_compute_group_norm_x_hat",
    "_compute_layer_norm_x_hat",
    "_compute_rms_x_hat",
    "_conv1d_im2col",
    "_conv2d_im2col",
    "_conv3d_im2col",
    "_conv_spatial_rank",
    "_cross_dot",
    "_cross_gram",
    "_dot",
    "_drop_gradient_free_rows",
    "_ekfac_materialize",
    "_fim",
    "_flatten_for_kfac",
    "_grad_norm_sq",
    "_kfac",
    "_kfac_cross",
    "_materialize",
    "_materialize_embedding",
    "_pairwise_dot",
    "_preprocess_conv_transpose",
    "_preprocess_embedding_bag",
    "_preprocess_factorized",
    "_project_factorized",
    "_project_materialized",
    "_to_3d",
    "canonical_class_name",
    "cross_dot",
    "dot",
    "effective_dims",
    "ekfac_materialize",
    "extract_module_kwargs",
    "fim",
    "grad_norm_sq",
    "is_conv",
    "is_conv_transpose",
    "is_embedding",
    "is_linear",
    "is_norm",
    "kfac",
    "kfac_cross",
    "kfac_eigh",
    "materialize",
    "maybe_use_materialized_gram",
    "maybe_use_materialized_norm",
    "pairwise_dot",
    "preprocess_factorized",
    "project_factorized",
    "project_layer",
    "project_materialized",
    "sym_inverse",
]

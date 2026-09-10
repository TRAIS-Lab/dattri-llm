"""Layer-type-aware gradient operations for per-sample gradient computation.

This package re-exports the public surface of its submodules so code imports
from ``dattri_llm.gradient.ops``:

* :mod:`~dattri_llm.gradient.ops.types` -- layer-type constants and predicates.
* :mod:`~dattri_llm.gradient.ops.preprocess` -- raw-capture preprocessing
  (incl. conv im2col) and module-kwargs extraction.
* :mod:`~dattri_llm.gradient.ops.materialize` -- per-sample weight gradients.
* :mod:`~dattri_llm.gradient.ops.dot` -- dot products, grams, norms, the
  layerwise cross-gram, and the factorized-vs-materialized routing heuristic.
* :mod:`~dattri_llm.gradient.ops.projection` -- TRAK/LoGRA random projection,
  fixed coordinate subsets, and the :class:`DattriProjector` that owns the
  projection matrices and subsets.
* :mod:`~dattri_llm.gradient.ops.kronecker` -- K-FAC / EK-FAC / Fisher kernels
  and streaming accumulators.

Naming convention: a function suffixed ``_factors`` takes raw **batch-first**
``(activation, pre_activation_grad)`` tensors; the unsuffixed name takes the
:class:`~dattri_llm.gradient.gradient.Factorized` container (and, where noted,
a dense tensor) and is the entry point to prefer.
"""

from dattri_llm.gradient.ops import dtypes
from dattri_llm.gradient.ops.dot import (
    cross_dot,
    cross_dot_factors,
    cross_dot_per_token,
    cross_gram,
    cross_gram_per_token,
    dot,
    dot_factors,
    effective_dims,
    grad_norm_sq,
    grad_norm_sq_factors,
    layerwise_cross_dot,
    maybe_use_materialized_gram,
    maybe_use_materialized_norm,
    pairwise_dot,
    pairwise_dot_factors,
)
from dattri_llm.gradient.ops.dtypes import (
    as_float,
    compute_dtype,
    get_compute_dtype,
    set_compute_dtype,
)
from dattri_llm.gradient.ops.kronecker import (
    FisherAccumulator,
    KroneckerAccumulator,
    LayerFisherAccumulator,
    LayerKroneckerAccumulator,
    dense_inverse,
    ekfac_materialize,
    ekfac_materialize_factors,
    ekfac_precondition,
    fim,
    fim_factors,
    kfac,
    kfac_cross,
    kfac_cross_factors,
    kfac_eigh,
    kfac_factors,
    kfac_precondition,
    kfac_precondition_materialized,
    sym_inverse,
)
from dattri_llm.gradient.ops.materialize import materialize, materialize_factors
from dattri_llm.gradient.ops.optimizer import (
    OPTIMIZER_STATE_KEYS,
    adam_preconditioner,
    adamw_influence_coupling,
    adamw_influence_push,
    adamw_influence_transition,
    precondition,
)
from dattri_llm.gradient.ops.preprocess import (
    extract_module_kwargs,
    preprocess_factorized,
    preprocess_factors,
    to_3d,
)
from dattri_llm.gradient.ops.projection import (
    PROJECTION_STYLES,
    SUBSET_KEYS,
    DattriProjector,
    apply_projection,
    maybe_materialize_projected,
    project_activation,
    project_factorized,
    project_factors,
    project_gradient,
    project_layer,
    project_materialized,
    project_materialized_factors,
    subset_coordinates,
    subset_factorized,
    subset_factors,
    subset_materialized,
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
    is_kfac_eligible,
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
    "OPTIMIZER_STATE_KEYS",
    "PARAM_GRAD_TYPES",
    "PROJECTION_STYLES",
    "SUBSET_KEYS",
    "DattriProjector",
    "FisherAccumulator",
    "KroneckerAccumulator",
    "LayerFisherAccumulator",
    "LayerKroneckerAccumulator",
    "adam_preconditioner",
    "adamw_influence_coupling",
    "adamw_influence_push",
    "adamw_influence_transition",
    "apply_projection",
    "as_float",
    "canonical_class_name",
    "compute_dtype",
    "cross_dot",
    "cross_dot_factors",
    "cross_dot_per_token",
    "cross_gram",
    "cross_gram_per_token",
    "dense_inverse",
    "dot",
    "dot_factors",
    "dtypes",
    "effective_dims",
    "ekfac_materialize",
    "ekfac_materialize_factors",
    "ekfac_precondition",
    "extract_module_kwargs",
    "fim",
    "fim_factors",
    "get_compute_dtype",
    "grad_norm_sq",
    "grad_norm_sq_factors",
    "is_conv",
    "is_conv_transpose",
    "is_embedding",
    "is_kfac_eligible",
    "is_linear",
    "is_norm",
    "kfac",
    "kfac_cross",
    "kfac_cross_factors",
    "kfac_eigh",
    "kfac_factors",
    "kfac_precondition",
    "kfac_precondition_materialized",
    "layerwise_cross_dot",
    "materialize",
    "materialize_factors",
    "maybe_materialize_projected",
    "maybe_use_materialized_gram",
    "maybe_use_materialized_norm",
    "pairwise_dot",
    "pairwise_dot_factors",
    "precondition",
    "preprocess_factorized",
    "preprocess_factors",
    "project_activation",
    "project_factorized",
    "project_factors",
    "project_gradient",
    "project_layer",
    "project_materialized",
    "project_materialized_factors",
    "set_compute_dtype",
    "subset_coordinates",
    "subset_factorized",
    "subset_factors",
    "subset_materialized",
    "sym_inverse",
    "to_3d",
]

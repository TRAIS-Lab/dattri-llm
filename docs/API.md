# API reference

An index of the public API, generated from each package's `__all__` (and the
aliases defined in `dattri_llm.options`) and the first line of each docstring.
The docstrings are the reference: use `help(obj)` or your editor for
signatures and full descriptions.

Names are listed where they are defined for users; many are re-exported at
several levels (for example `HookManager` is importable from `dattri_llm`,
`dattri_llm.gradient` and `dattri_llm.gradient.hooks`).

Contents: [`dattri_llm`](#dattri_llm), [`dattri_llm.attribution`](#dattri_llmattribution), [`dattri_llm.gradient`](#dattri_llmgradient), [`dattri_llm.gradient.hooks`](#dattri_llmgradienthooks), [`dattri_llm.gradient.callbacks`](#dattri_llmgradientcallbacks), [`dattri_llm.gradient.ops`](#dattri_llmgradientops), [`dattri_llm.utils`](#dattri_llmutils), [`dattri_llm.options`](#dattri_llmoptions)

## `dattri_llm`

**Top level.** The advertised surface. The capture core is imported eagerly with `torch` alone; the attribution layer and the live streamer resolve lazily on first access.

| name | kind | summary |
|---|---|---|
| `REGISTER_ALL` | constant | Selector meaning "register every applicable layer" for a hook family. |
| `AdamWInfluenceAttributor` | class | AdamW-influence attributor. |
| `AttributionArguments` | class | Configuration for running a training-data attribution algorithm. |
| `AttributionScore` | class | Trajectory-aware attribution scores keyed by input hash. |
| `AttributionTask` | class | A model, the loss to attribute (and optionally a separate target), and the checkpoints to attribute at. |
| `BaseAttributor` | class | Base class for all attributors. |
| `BaseInnerProductAttributor` | class | Base class for inner-product attributors. |
| `CaptureCallback` | class | Holds the most recent per-step :class:`GradientRecord` in memory. |
| `DVEmbAttributor` | class | DVEmb (Data Value Embedding) attributor. |
| `DataSelectionCallback` | class | Online data selection via per-sample influence scoring. |
| `DiskGradientSource` | class | A re-iterable :class:`GradientSource` backed by on-disk gradients. |
| `EKFACAttributor` | class | EK-FAC influence attributor. |
| `Factorized` | class | A per-layer factorized ("ghost") gradient: an (activation, output-gradient) factor pair. |
| `Gradient` | class | A per-step, multi-layer container of per-sample gradients with metadata. |
| `GradientRecord` | class | A gradient snapshot with its identity baked in. |
| `GradientStorageManager` | class | Residency-managed storage and retrieval of :class:`GradientRecord` objects. |
| `GradientStreamer` | class | Yields per-step ``(step, Gradient, hashes)`` from a live forward+backward pass. |
| `HookManager` | class | Collect per-sample or per-batch gradients via forward/backward hooks. |
| `HookManagerCallback` | class | Base class for :class:`~dattri_llm.gradient.hooks.HookManager` callbacks. |
| `HookManagerConfig` | class | Configuration for :class:`HookManager`. |
| `KFACAttributor` | class | K-FAC influence attributor. |
| `KroneckerAttributor` | class | Shared workflow of the K-FAC family; subclass to add a Kronecker variant. |
| `KroneckerCovarianceCallback` | class | Accumulate per-layer K-FAC covariances ``(A, G)`` during collection. |
| `LESSAttributor` | class | LESS attributor. |
| `OffloadCallback` | class | Periodically saves :class:`GradientRecord` objects to disk. |
| `OptimizerStateCallback` | class | Snapshot an Adam-family optimizer's moments on each layer's coordinates. |
| `ParameterSnapshotCallback` | class | Store the trainable parameters at every capture step. |
| `ReplayGradientSource` | class | Per-step blocks recomputed from a trajectory's snapshots. |
| `TracInAttributor` | class | TracIn / GradCos attributor. |
| `TrajectoryAttributor` | class | Base class of the trajectory-sweeping attributors (see the module docstring). |
| `TrajectorySnapshots` | class | Per-step parameters, batches and optimizer moments of a trajectory. |
| `default_hook_assignment` | function | Discover the default-style assignment for layers that actually fire. |
| `hash_batch` | function | Per-sample content hashes for a **batched** input dict, in batch order. |
| `hash_sample` | function | SHA-256 content hash identifying **one sample** by its model inputs. |

## `dattri_llm.attribution`

**Attribution.** Attributors, their configuration and the score container.

Also re-exported here (listed above): `AdamWInfluenceAttributor`, `AttributionArguments`, `AttributionScore`, `BaseAttributor`, `BaseInnerProductAttributor`, `DVEmbAttributor`, `EKFACAttributor`, `KFACAttributor`, `KroneckerAttributor`, `LESSAttributor`, `TracInAttributor`, `TrajectoryAttributor`.

## `dattri_llm.gradient`

**Gradient capture.** `HookManager`, its callbacks, the on-disk store and the most used kernels.

| name | kind | summary |
|---|---|---|
| `AsyncGradientWriter` | class | Write gradient record groups to a store from a background thread. |
| `FisherAccumulator` | class | Streaming empirical-Fisher accumulator across a model's layers. |
| `KroneckerAccumulator` | class | Streaming K-FAC covariance accumulator across a model's layers. |
| `LayerFisherAccumulator` | class | Streaming empirical Fisher accumulator for a *single* layer. |
| `LayerKroneckerAccumulator` | class | Streaming K-FAC covariance accumulator for a *single* layer. |
| `canonical_class_name` | function | Return canonical string for a module class, e.g. 'nn.Linear'. |
| `dot` | function | :func:`dot_factors` on two :class:`Factorized` (batch-first-safe). |
| `fim` | function | :func:`fim_factors` on a :class:`Factorized` (batch-first-safe). |
| `grad_norm_sq` | function | Per-sample squared gradient norms ``(B,)`` of one layer, whatever its form. |
| `kfac` | function | :func:`kfac_factors` on a :class:`Factorized` (batch-first-safe). |
| `materialize` | function | Per-sample weight gradient ``(B, d)`` of one layer, whatever its form. |
| `ops` | module | Layer-type-aware gradient operations for per-sample gradient computation. |
| `pairwise_dot` | function | :func:`pairwise_dot_factors` on a :class:`Factorized` (batch-first-safe). |
| `register_linear_io_hooks` | function | Register forward and backward hooks on linear-family layers. |
| `register_linear_param_hooks` | function | Register post-accumulate-grad hooks on linear layers' trainable params. |
| `register_param_grad_hooks` | function | Register parameter-gradient hooks on general module layers. |
| `remove_hooks` | function | Remove all registered hooks and clear the handle list. |

Also re-exported here (listed above): `REGISTER_ALL`, `CaptureCallback`, `DataSelectionCallback`, `GradientRecord`, `GradientStorageManager`, `HookManager`, `HookManagerCallback`, `HookManagerConfig`, `KroneckerCovarianceCallback`, `OffloadCallback`, `OptimizerStateCallback`, `ParameterSnapshotCallback`, `TrajectorySnapshots`, `default_hook_assignment`.

## `dattri_llm.gradient.hooks`

**Hooks.** Hook registration, `HookManagerConfig` and layer selectors.

| name | kind | summary |
|---|---|---|
| `INVASIVE_LINEAR_IO` | constant | Hook-family name `"invasive_linear_io"`: like `linear_io` but skips the weight-gradient matmul (`nn.Linear` only, capture only, no `weight.grad`). |
| `LINEAR_IO` | constant | Hook-family name `"linear_io"`: factorized capture of a layer's input activations and output gradients. |
| `PARAM_GRAD` | constant | Hook-family name `"param_grad"`: capture of aggregated parameter gradients. |
| `HF_Conv1D` | class | `transformers.pytorch_utils.Conv1D` when `transformers` is installed, else `None`. |
| `LayerBuffer` | type alias | Type alias (`dict`) of one layer's `linear_io` capture buffer. |
| `ParamGradBuffer` | type alias | Type alias (`dict`) of one layer's `param_grad` capture buffer. |
| `Selector` | type alias | Type of a layer selector: `REGISTER_ALL`, a list of name regexes, or `None`. |
| `install_invasive_forward` | function | Override each named ``nn.Linear``'s forward to skip its weight/bias grad. |
| `resolve_hook_assignments` | function | Resolve the final ``{layer_name: hook_type}`` assignment. |

Also re-exported here (listed above): `REGISTER_ALL`, `HookManager`, `HookManagerCallback`, `HookManagerConfig`, `OffloadCallback`, `default_hook_assignment`, `register_linear_io_hooks`, `register_linear_param_hooks`, `register_param_grad_hooks`, `remove_hooks`.

## `dattri_llm.gradient.callbacks`

**Callbacks.** Callbacks that act on each captured step.

Also re-exported here (listed above): `CaptureCallback`, `DataSelectionCallback`, `HookManagerCallback`, `KroneckerCovarianceCallback`, `OffloadCallback`, `OptimizerStateCallback`, `ParameterSnapshotCallback`.

## `dattri_llm.gradient.ops`

**Gradient ops.** Layer-type-aware kernels on factorized or materialized gradients: dot products and routing, projection, K-FAC / EK-FAC, optimizer preconditioning.

| name | kind | summary |
|---|---|---|
| `ALL_LAYER_TYPES` | constant | Every layer type with per-sample (factorized) capture support. |
| `CAPTURE_STYLES` | constant | The capture styles: `"factorized"`, `"materialized"`, `"auto"`. |
| `CONV_TRANSPOSE_TYPES` | constant | Canonical class names of the transposed-convolution layers. |
| `CONV_TYPES` | constant | Canonical class names of the convolution layers. |
| `EMBEDDING_TYPES` | constant | Canonical class names of the embedding layers. |
| `LINEAR_TYPES` | constant | Canonical class names of the linear layer family. |
| `MASK_KEYS` | constant | Keys a `"mask"` projection config may carry. |
| `NORM_TYPES` | constant | Canonical class names of the normalization layers. |
| `OPTIMIZER_STATE_KEYS` | constant | Per supported `torch.optim` optimizer, the state keys its preconditioning reads. |
| `PARAM_GRAD_TYPES` | constant | Layer-type marker of aggregated parameter-level gradients (no batch dimension). |
| `PROJECTION_STYLES` | constant | The projection styles: `"logra"`, `"dense"`, `"mask"`. |
| `DattriProjector` | class | A random-projection factory plus the cache of the matrices it builds. |
| `ProjectionMatrix` | class | One layer's random projection with everything but the kernel done. |
| `adam_preconditioner` | function | The diagonals ``D_t`` and ``S_t`` of AdamW-influence's transition. |
| `adamw_influence_coupling` | function | ``W R_t g_z`` for every sample of a batch, ``(B, p)``. |
| `adamw_influence_push` | function | ``Z_push(z) = (theta_dot_{t+1}, m_dot_t, v_dot_t)`` of AdamW-influence. |
| `adamw_influence_transition` | function | ``W M_t`` for AdamW-influence's block-diagonal transition ``M_t``. |
| `apply_projection` | function | :meth:`DattriProjector.apply` for a factory **or** a projector. |
| `as_float` | function | Like :func:`align`, but integer operands are converted too. |
| `compute_dtype` | function | Scope a compute-dtype policy to a block. |
| `cross_dot` | function | ``(B1, B2)`` cross-gram ``K[i, j] = <dW1_i, dW2_j>`` of one layer, in whatever form each side holds. |
| `cross_dot_factors` | function | Return the (B1, B2) cross-gram ``K[i, j] = <dW1_i, dW2_j>``. |
| `cross_dot_per_token` | function | Per-token-position cross-gram of a :class:`Factorized` side 1 against side 2 in whatever form it holds (batch-first-safe). |
| `cross_gram` | function | Cross-gram ``K[i, j] = <dW1_i, dW2_j>`` on *already-preprocessed* factors. |
| `cross_gram_per_token` | function | Per-token-position cross-gram ``K[i, t, j]`` on *preprocessed* factors: the contribution of side-1 sample ``i``'s **token position ``t``** to the weight-gradient inner product ``<dW1_i, dW2_j>``. |
| `cross_gram_per_token_dense` | function | :func:`cross_gram_per_token` with side 2 already **dense**: *m2* is the ``(B2, D)`` per-sample gradient block in :func:`materialize`'s layout -- e.g. a materialized test block, or one preconditioned by an attributor (K-FAC's ``G^-1 dW A^-1``), which is what makes every bilinear score decompose over side 1's token positions the same way. |
| `dense_inverse` | function | Damped symmetric inverse ``(matrix + damping*I)^{-1}`` via Cholesky. |
| `dot_factors` | function | Return (B,) per-sample dot products <dW1_i, dW2_i>. |
| `dtypes` | module | Compute-dtype policy for the gradient operations. |
| `effective_dims` | function | Cheap ``(B, S, K, D)`` for the cost heuristic: batch, token/patch count, input width, output width -- the *post-preprocess* dims, read straight from the raw factor shapes (no im2col / materialization). |
| `ekfac_materialize` | function | :func:`ekfac_materialize_factors` on a :class:`Factorized` (batch-first-safe), or the same rotation applied to an already **materialized** block. |
| `ekfac_materialize_factors` | function | Per-sample weight gradient rotated into the K-FAC eigenbasis. |
| `ekfac_precondition` | function | Apply the full damped EK-FAC inverse to eigenbasis coordinates. |
| `extract_module_kwargs` | function | Extract the minimal hyperparameters from *module* needed by :func:`preprocess_factors`. |
| `fim_factors` | function | Return (d, d) empirical Fisher information matrix. |
| `get_compute_dtype` | function | The active policy -- ``"auto"`` or an explicit :class:`torch.dtype`. |
| `grad_norm_sq_factors` | function | Return (B,) per-sample squared Frobenius norms of weight gradients. |
| `is_conv` | function | Return True if layer_type is a convolution layer type. |
| `is_conv_transpose` | function | Return True if layer_type is a transposed convolution layer type. |
| `is_embedding` | function | Return True if layer_type is an embedding layer type. |
| `is_kfac_eligible` | function | Return True if layer_type carries K-FAC (Kronecker) covariances. |
| `is_linear` | function | Return True if layer_type is a linear layer type. |
| `is_norm` | function | Return True if layer_type is a normalization layer type. |
| `kfac_cross` | function | :func:`kfac_cross_factors` on two :class:`Factorized` (batch-first-safe). |
| `kfac_cross_factors` | function | K-FAC preconditioned cross-gram between two factorized gradient sets. |
| `kfac_eigh` | function | Eigendecompose both K-FAC factors: returns ``(s_A, U_A, s_G, U_G)``. |
| `kfac_factors` | function | Return (A, G) K-FAC covariance factor matrices. |
| `kfac_precondition` | function | Preprocess and whiten one side's factors by the inverse K-FAC covariances. |
| `kfac_precondition_materialized` | function | Two-sided K-FAC preconditioning of a **compact materialized** block. |
| `layerwise_cross_dot` | function | Layer-by-layer cross-gram of two gradient blocks, summed over layers. |
| `layerwise_cross_dot_per_token` | function | Layer-by-layer per-token cross-gram of two gradient blocks: the ``(B_train, T_train, B_test)`` contribution of every *train* token position to :func:`layerwise_cross_dot`, which it sums back to exactly over ``T``. |
| `mask_coordinates` | function | The coordinates a ``"mask"`` capture keeps for a layer. |
| `mask_factorized` | function | :func:`mask_factors` on a :class:`Factorized` (batch-first-safe). |
| `mask_factors` | function | ``materialize_factors(...)[:, mask]`` without materializing. |
| `mask_materialized` | function | Keep ``proj_dim`` fixed random coordinates of a dense ``(..., D)`` tensor. |
| `materialize_factors` | function | Compute the per-sample weight gradient, returning shape (B, d). |
| `maybe_materialize_projected` | function | ``True`` when the projected factors should be materialized at capture. |
| `maybe_use_materialized_gram` | function | ``True`` when materialize-then-GEMM is the cheaper way to form the ``(B1, B2)`` cross-gram, by flop count. |
| `maybe_use_materialized_norm` | function | ``True`` when materializing is cheaper for per-sample norms by flop count. |
| `pairwise_dot_factors` | function | Return (B, B) pairwise dot product matrix of per-sample gradients. |
| `precondition` | function | The per-sample update direction an optimizer would take from ``g``. |
| `preprocess_factorized` | function | :func:`preprocess_factors` on a :class:`Factorized` (batch-first-safe). |
| `preprocess_factors` | function | Transform raw hook captures into the (a, g) form expected by ops. |
| `project_activation` | function | Project **only** a linear layer's activation factor (the a-side of :func:`project_factors`). |
| `project_factorized` | function | :func:`project_factors` on a :class:`Factorized` (batch-first-safe). |
| `project_factors` | function | LoGRA-style: project the two factorized factors, keeping the structure. |
| `project_gradient` | function | Project **only** a linear layer's gradient factor (the g-side of :func:`project_factors`). |
| `project_layer` | function | Route one layer through a projection style. |
| `project_materialized` | function | :func:`project_materialized_factors` on a :class:`Factorized` (batch-first-safe). |
| `project_materialized_factors` | function | TRAK-style: materialize the per-sample weight gradient, then project it. |
| `set_compute_dtype` | function | Set the process-wide compute dtype; returns the policy it replaces. |
| `should_materialize` | function | Whether a layer's factors are materialized at capture under *capture_style*. |
| `sym_inverse` | function | Damped symmetric inverse ``(matrix + damping*I)^{-1}``. |
| `to_3d` | function | Expand a (B, D) tensor to (B, 1, D); leave (B, T, D) unchanged. |

Also re-exported here (listed above): `FisherAccumulator`, `KroneckerAccumulator`, `LayerFisherAccumulator`, `LayerKroneckerAccumulator`, `canonical_class_name`, `dot`, `fim`, `grad_norm_sq`, `kfac`, `materialize`, `pairwise_dot`.

## `dattri_llm.utils`

**Utilities.** Content hashing, distributed and autograd helpers, and the cache abstraction.

| name | kind | summary |
|---|---|---|
| `CACHE_RESIDENCIES` | constant | The cache residencies: `"disk"`, `"memory"`, `"tiered"`. |
| `CacheBudget` | class | Byte budget a cache may occupy on a device. |
| `TensorCache` | class | Keyed, budgeted, residency-aware cache of tensors. |
| `dist_rank` | function | Return the current distributed rank, or ``None`` outside a distributed context. |
| `dist_world_size` | function | Return the process-group size, or ``1`` outside a distributed context. |
| `is_dist_initialized` | function | Return ``True`` when a ``torch.distributed`` process group is active. |
| `queue_after_backward_finalization` | function | Schedule ``fn`` to run after **everything** queued during the backward. |
| `queue_backward_end_callback` | function | Schedule ``fn`` to run once the in-flight backward pass fully completes. |
| `tensor_nbytes` | function | Bytes held by *value*: a tensor, an object exposing ``nbytes``, or a ``dict``/``list``/``tuple`` of those (recursively). |

Also re-exported here (listed above): `hash_batch`, `hash_sample`.

## `dattri_llm.options`

**Option types.** `Literal` aliases naming the values each enumerated string option accepts, for annotations; the options are still validated at runtime.

| name | kind | summary |
|---|---|---|
| `AttributionGranularity` | type alias | `Literal['instance', 'token']`: One score row per training sample, or one per training token position. |
| `CacheResidency` | type alias | `Literal['disk', 'memory', 'tiered']`: Where a cache or gradient store keeps its entries (``CACHE_RESIDENCIES``). |
| `CaptureStyle` | type alias | `Literal['factorized', 'materialized', 'auto']`: The representation a captured layer is buffered in (``CAPTURE_STYLES``). |
| `DiskFormat` | type alias | `Literal['pickle', 'memmap']`: The on-disk format of a gradient store (``DISK_FORMATS``). |
| `DotReduce` | type alias | `Literal['sum', 'none']`: Sum the layerwise cross-grams over layers, or keep them per layer. |
| `EKFACMode` | type alias | `Literal['exact', 'approx']`: EK-FAC's eigenvalue estimate (``EKFACAttributor.EKFAC_MODES``). |
| `HessianMode` | type alias | `Literal['full', 'diagonal']`: DVEmb's per-step Hessian (Fisher) approximation. |
| `LossReduction` | type alias | `Literal['mean', 'sum']`: How the training loss was reduced over each batch. |
| `ProjectionStyle` | type alias | `Literal['logra', 'dense', 'mask']`: How a layer is projected (``PROJECTION_STYLES``). |
| `Propagation` | type alias | `Literal['train', 'test']`: Which side of a trajectory sweep carries the propagation. |
| `RecordingType` | type alias | `Literal['per_sample', 'per_batch']`: The granularity an ``OffloadCallback`` records at. |
| `RoutingMode` | type alias | `Literal['factorized', 'materialized', 'auto']`: How an inner-product kernel forms its result; ``"auto"`` uses the cost rule. |
| `ScoreTrajectory` | type alias | `Literal['aware', 'agnostic']`: One row per training sample, or one per ``(sample, step)`` pair. |

## Extras

Builders for the per-layer `module_kwargs` dicts live in
`dattri_llm.utils.module` (`rms_norm_module_kwargs`, `layer_norm_module_kwargs`,
`conv2d_module_kwargs`, ...); see the README's Hugging Face norm-layer recipe.
The attribution task is `dattri_llm.task.AttributionTask` (also exported at the
top level).


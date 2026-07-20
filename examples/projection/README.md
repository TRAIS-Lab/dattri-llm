# projection/

## `gradient_projection.py` — random projection of per-sample gradients

Random projection compresses per-sample gradients to a small fixed width
(`proj_dim`) while approximately preserving their inner products — the
Johnson–Lindenstrauss property that makes storing per-sample gradients of
large models feasible, since attribution scores are gradient dot products.
This tutorial builds a model with one layer of every family (embedding, two
linears, a LayerNorm, a head) and walks through the whole projection surface
in six sections:

1. **Raw capture** — the unprojected reference: per-layer factorized payloads
   whose widths follow the layer dimensions.
2. **Capture-time projection** — `HookManagerConfig(projection=...)`: each
   backward pass projects the factors on the fly, so the raw factors are
   never buffered and every layer's stored width becomes `proj_dim`,
   independent of the layer's size. The script prints the payload before and
   after (~40 % at these toy widths; the ratio keeps improving with layer
   width, e.g. a 4096×4096 linear stores `2·proj_dim` per token instead of
   `2·4096`).
3. **Post-hoc projection** — `Gradient.project(...)` on the raw capture.
   With the same `proj_seed` (and device), it is **bit-identical** to the
   capture-time result — so gradients projected at collection time and
   gradients projected later are directly comparable.
4. **Similarity preservation** — the pairwise gradient dot products of the
   batch, computed raw vs. projected, correlate at r ≈ 0.99.
5. **Per-layer configuration** — the `projection` mapping is keyed by layer
   name, so each layer gets its own budget (`fc1` at 128, `fc2` at 32), and
   **without** a `"__default__"` entry, layers with no entry of their own are
   captured raw. Mixed configs are first-class.
6. **Custom layer classes** — hooking a hand-rolled layer the library cannot
   recognise, declared with `layer_types` + `module_kwargs` and verified
   against autograd.

## Three projection styles

Chosen per layer via `style`:

| `style` | result | defined for |
|---|---|---|
| `"logra_factorized"` (default) | both factors projected; layer stays **factorized** at width `proj_dim` | linear / conv families, embeddings (ids expanded to one-hot) |
| `"logra_materialized"` | LoGRA-project, then **materialize** the factors into a compact `(B, proj_dim*proj_dim)` block (token-summed outer product) | same as above; smaller on disk, no per-token structure |
| `"materialized"` (TRAK) | per-sample weight gradient materialized, then projected to a dense `(B, proj_dim)` block | any layer; **required** for norm layers |

The two `logra_*` styles share the same double-sided projection and score
identically for sample-level attribution; `"logra_materialized"` just stores
the collapsed outer product instead of the per-token factors. Use it when you
only need per-sample scores and want a compact store; keep
`"logra_factorized"` when you need per-token attribution or K-FAC/EK-FAC
(which require the factors).

## Configuring projection per layer

The `projection` mapping takes per-layer entries plus an optional
`"__default__"` covering every hooked layer without its own entry; layers
with neither are captured raw. Section 2 projects everything LoGRA-style and
overrides the LayerNorm to TRAK:

```python
HookManagerConfig(
    linear_io=REGISTER_ALL,
    projection={
        "__default__": {
            "style": "logra_factorized",
            "proj_dim": 64,
            "proj_max_batch_size": 8,   # required by dattri's random_project
            "proj_type": "rademacher",
            "proj_seed": 7,
        },
        "norm": {"style": "materialized", "proj_dim": 64,
                 "proj_max_batch_size": 8, "proj_type": "rademacher",
                 "proj_seed": 7},
    },
)
```

while section 5 drops `"__default__"` and gives each layer its own budget —
`proj_dim` may differ per layer, and unlisted layers stay raw. (To not
capture a layer *at all*, exclude it from hooking instead — see
[`hooks/modular_hooks.py`](../hooks/modular_hooks.py).)

Two consistency rules keep projected gradients comparable:

- **Fix `proj_seed` per layer** across everything that will be scored
  together (train and test, different steps, capture-time and post-hoc) —
  different seeds are different projections.
- **Use one `device` consistently** (unset, it defaults to the tensors' own
  device). dattri's CPU and CUDA projectors do not produce the same
  projection for the same seed, and their valid `proj_type` sets differ
  (`"sjlt"`/`"grass"` are CUDA-only).

## Custom layer classes (`dattri_llm.utils.module`)

Hyperparameters of hooked layers (`has_bias`, an embedding's `padding_idx`, a
norm's `eps`, ...) are normally extracted straight off the module. A custom
class with the right math but non-standard attribute names — the tutorial's
`HandRolledRMSNorm` mirrors HF's `LlamaRMSNorm`, whose epsilon lives in
`variance_epsilon` — defeats that extraction. Section 6 declares the layer
instead:

```python
from dattri_llm.utils.module import rms_norm_module_kwargs

HookManagerConfig(
    hook_types={"fc": "linear_io", "norm": "linear_io"},
    layer_types={"norm": "nn.RMSNorm"},        # what the layer IS
    module_kwargs={"norm": rms_norm_module_kwargs(  # its hyperparameters
        normalized_shape=128, eps=1e-6,
    )},
)
```

`dattri_llm.utils.module` has one builder per supported layer type
(`linear_module_kwargs`, `embedding_module_kwargs`,
`embedding_bag_module_kwargs`, `conv{1,2,3}d_module_kwargs` and their
transposes, `layer_norm_module_kwargs`, `rms_norm_module_kwargs`,
`group_norm_module_kwargs`, `instance_norm{1,2,3}d_module_kwargs`). Every
argument is keyword-only and **required** by design: these helpers describe
non-standard layers, exactly the situation where a silently assumed default
(a bias that is not there, a different epsilon) would corrupt the captured
gradients without any error — a forgotten field fails at config-build time
instead.

The tutorial closes the loop by *verifying* the declaration: the sum of the
captured per-sample gradients reproduces autograd's `param.grad` for the
custom layer (a wrong `eps` shows up here immediately), and the declared
layer then flows through TRAK projection like any native one.

```bash
python examples/projection/gradient_projection.py
```

Requires `dattri` (the projector factory; `projector=None` lazily imports
dattri's `random_project`).

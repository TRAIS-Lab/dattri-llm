# Supported Layers for Ghost Gradient Dot Products

This document summarizes the PyTorch layer classes currently supported by the ghost-gradient implementation. For these layers, parameter-gradient inner products can be computed directly from factorized activations and backpropagated gradients without explicitly materializing the full parameter gradients.

## Supported Layers

### Linear Layers

```python
nn.Linear
nn.Bilinear
```

This category also covers subclasses of `nn.Linear`, such as:

```python
nn.modules.linear.NonDynamicallyQuantizableLinear
```

### Convolution Layers

```python
nn.Conv1d
nn.Conv2d
nn.Conv3d

nn.ConvTranspose1d
nn.ConvTranspose2d
nn.ConvTranspose3d
```

### Embedding Layers

```python
nn.Embedding
nn.EmbeddingBag
```

### Normalization Layers

```python
nn.LayerNorm
nn.RMSNorm
nn.GroupNorm

nn.InstanceNorm1d
nn.InstanceNorm2d
nn.InstanceNorm3d
```

## Currently Unsupported Layers

The following layers are not currently supported by the implementation.

### Attention Layers

```python
nn.MultiheadAttention
```

Only **partially** representable. The output projection (`out_proj`) is a
`NonDynamicallyQuantizableLinear` (a `nn.Linear` subclass) and is therefore
supported like any other linear layer. The **input projection**
(`in_proj_weight` / `in_proj_bias`), however, is stored as direct `Parameter`s
on the module and applied via a functional `F.linear` call inside
`F.multi_head_attention_forward` — there is no `nn.Linear` submodule whose
forward/backward hooks could capture the in-projection's activations and output
gradients. Capturing in-projection gradients would require decomposing
`MultiheadAttention` into explicit linear submodules (as some HF/TRL model
variants do), in which case the resulting `nn.Linear` layers are supported
directly.

### Batch Normalization Family

These layers introduce cross-sample dependencies through batch statistics and therefore require special handling.

```python
nn.BatchNorm1d
nn.BatchNorm2d
nn.BatchNorm3d

nn.SyncBatchNorm
```

### Recurrent Layers

Although recurrent layers admit structured gradient representations, they require dedicated handling and are currently excluded. This covers both the sequence-level modules and the single-step cells.

```python
nn.RNN
nn.GRU
nn.LSTM

nn.RNNCell
nn.GRUCell
nn.LSTMCell
```

More generally, this corresponds to:

```python
nn.RNNBase
nn.RNNCellBase
```

### Parametric Activation Layers

```python
nn.PReLU
```

Although the parameter gradients are straightforward to compute, support has not yet been implemented.

### Lazy Layer Variants

```python
nn.LazyLinear
nn.LazyConv1d
nn.LazyConv2d
nn.LazyConv3d
nn.LazyConvTranspose1d
nn.LazyConvTranspose2d
nn.LazyConvTranspose3d
```

These are subclasses of their concrete counterparts, so the forward/backward
hooks (which match by `isinstance`) do attach to them. However, op dispatch is
keyed on the canonical class-name string (e.g. `"nn.LazyLinear"`), which is not
a member of `LINEAR_TYPES` / `CONV_TYPES`. As a result the weight gradient is
still materialized via the generic einsum path, but layer-specific handling
(im2col for convolutions, bias folding) is silently skipped — so support is
only partial and inconsistent. Note that the class remains `LazyLinear` even
after the first forward pass initializes its parameters. These can be promoted
to full support by adding their names to the layer-type sets in `ops.py`.

## Notes

* Non-parametric modules such as `nn.ReLU`, `nn.GELU`, `nn.SiLU`, pooling layers, dropout layers, and tensor reshaping operations are not listed because they do not contain trainable parameters.
* Composite / container modules (e.g. `nn.Transformer`, `nn.TransformerEncoderLayer`, `nn.AdaptiveLogSoftmaxWithLoss`) own no direct parameters; their trainable parameters live in `nn.Linear` / `nn.LayerNorm` leaf submodules that are supported directly. They are therefore covered transitively, with the exception of any `nn.MultiheadAttention` in-projection (see above).
* The supported layer set covers the vast majority of trainable parameters in modern Transformer, ViT, DiT, CLIP, DINO, and diffusion-transformer architectures.
* Additional layer types may be added in the future as dedicated factorization rules become available.

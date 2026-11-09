"""Forward overrides routing sequence mixers through their modules.

The state-space and linear-attention mixers of Hugging Face Transformers
own ordinary modules -- projections and a depthwise ``nn.Conv1d`` -- but
their forwards do not call all of them.  A fused training kernel
(``mamba_inner_fn``, ``mamba_split_conv1d_scan_combined``) takes the
weights of every projection directly, and the PyTorch fallbacks apply the
convolution functionally (``causal_conv1d_fn`` on ``conv.weight``) and, in
the Mamba-1 family, ``dt_proj`` as a bare matmul on its weight.  A module
whose forward is never called fires no hook, so those layers were absent
from the record.

Two strategies cover the mixers, both active only while a
:class:`~dattri_llm.gradient.hooks.HookManager` is collecting and both
leaving incremental decoding (a cache holding the layer's state) to the
original forward:

* **Mamba-1 reference forward** (``MambaMixer``, ``FalconMambaMixer``,
  ``JambaMambaMixer``): the forward is replaced by the reference
  computation written with the module calls -- Transformers' own slow path,
  including the optional ``dt_layernorm``/``b_layernorm``/``c_layernorm`` of
  the Falcon and Jamba variants -- so every projection is hooked like any
  other layer.
* **Functional-conv interception** (every other block that owns a depthwise
  ``nn.Conv1d`` applied through its modelling module's ``causal_conv1d_fn``:
  the Mamba-2 family -- Mamba2, Bamba, NemotronH, Zamba2, Falcon-H1,
  GraniteMoeHybrid -- the gated delta-net mixers of Qwen3-Next, Qwen3.5,
  Kimi-Linear, OLMo-Hybrid and GLM5-Next, LFM2's short convolution, Zamba):
  the original forward runs, but while it runs the modelling module's
  ``causal_conv1d_fn`` is redirected through the block's own conv module
  when handed that conv's weight, and the fused-kernel shortcuts return
  ``None`` so the module path (the projections are module calls there) is
  taken.  No arithmetic is re-implemented.

Parameters that are not modules (``A_log``, ``D``, ``dt_bias``, Zamba's
``x_proj_weight``) stay batch-level ``param_grad`` captures.  The
redirection patches module globals for the duration of one forward call,
so it is not safe under thread-parallel replicas of one process
(``DataParallel``).
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

#: Mixer classes whose forward :func:`mamba1_reference_forward` reproduces.
#: Extend it for a class known to implement the same arithmetic.
MAMBA1_MIXER_CLASSES: set[str] = {"MambaMixer", "FalconMambaMixer", "JambaMambaMixer"}
_MAMBA1_ATTRS = (
    "in_proj",
    "conv1d",
    "x_proj",
    "dt_proj",
    "out_proj",
    "A_log",
    "D",
    "act",
    "time_step_rank",
    "ssm_state_size",
)
_CONV_FN = "causal_conv1d_fn"
_FUSED_FN_SUFFIXES = ("inner_fn", "split_conv1d_scan_combined")
_FAST_PATH_FLAG = "is_fast_path_available"


def _modeling_module(module: nn.Module) -> ModuleType | None:
    return sys.modules.get(type(module).__module__)


def is_mamba1_mixer(module: nn.Module) -> bool:
    """Whether *module* is a Mamba-1 family mixer (selection through
    ``x_proj``/``dt_proj`` and a sequential scan).
    """
    return type(module).__name__ in MAMBA1_MIXER_CLASSES and all(
        hasattr(module, attr) for attr in _MAMBA1_ATTRS
    )


def _depthwise_convs(module: nn.Module) -> dict[int, nn.Conv1d]:
    """The block's own depthwise ``nn.Conv1d`` children, by weight storage."""
    return {
        child.weight.data_ptr(): child
        for child in module.children()
        if isinstance(child, nn.Conv1d) and child.groups == child.in_channels
    }


def is_functional_conv_mixer(module: nn.Module) -> bool:
    """Whether *module* is a sequence mixer owning a depthwise ``nn.Conv1d``
    that its modelling module applies through ``causal_conv1d_fn`` (or
    behind a kernel fast path): the Mamba-2 family, the gated delta-net
    mixers, LFM2's short convolution, Zamba.
    """
    if is_mamba1_mixer(module) or not _depthwise_convs(module):
        return False
    modeling = _modeling_module(module)
    # Transformers 5 applies the conv through ``causal_conv1d_fn``; version 4
    # calls the module in its torch path but guards a kernel path with
    # ``is_fast_path_available``.  Either is a mixer the override serves.
    return modeling is not None and (
        callable(getattr(modeling, _CONV_FN, None))
        or hasattr(modeling, _FAST_PATH_FLAG)
    )


def find_sequence_mixers(root: nn.Module) -> dict[str, nn.Module]:
    """``{name: module}`` of every mixer under *root* an override serves."""
    return {
        n: m
        for n, m in root.named_modules()
        if is_mamba1_mixer(m) or is_functional_conv_mixer(m)
    }


def mamba1_reference_forward(
    mixer: nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """A Mamba-1 mixer's output computed through its modules (no cache).

    Mirrors Transformers' slow path: gated projection, causal depthwise
    convolution through ``conv1d``, selection through ``x_proj`` (with the
    Falcon/Jamba RMS norms of ``dt``, ``B`` and ``C`` when the mixer has
    them) and ``dt_proj`` (whose bias is the scan's ``delta_bias``), the
    sequential selective scan, the skip term ``D`` and the SiLU gate, then
    ``out_proj``.
    """
    batch_size, seq_len, _ = hidden_states.shape
    dtype = hidden_states.dtype
    # Out of place throughout: the tensors are outputs of hooked modules, which
    # the backward-hook machinery forbids modifying in place.
    if attention_mask is not None:
        hidden_states = hidden_states.mul(attention_mask.unsqueeze(-1).to(dtype))
    projected = mixer.in_proj(hidden_states).transpose(1, 2)  # (B, 2*d_inner, L)
    x, gate = projected.chunk(2, dim=1)
    if attention_mask is not None:
        x = x.mul(attention_mask.unsqueeze(1).to(dtype))

    x = mixer.act(mixer.conv1d(x)[..., :seq_len])  # (B, d_inner, L)
    if attention_mask is not None:
        x = x.mul(attention_mask.unsqueeze(1).to(dtype))

    ssm_parameters = mixer.x_proj(x.transpose(1, 2))  # (B, L, rank + 2*state)
    time_step, b_mat, c_mat = torch.split(
        ssm_parameters,
        [mixer.time_step_rank, mixer.ssm_state_size, mixer.ssm_state_size],
        dim=-1,
    )
    if hasattr(mixer, "dt_layernorm"):  # Falcon-Mamba, Jamba (Transformers 5)
        time_step = mixer.dt_layernorm(time_step)
        b_mat = mixer.b_layernorm(b_mat)
        c_mat = mixer.c_layernorm(c_mat)
    elif hasattr(mixer, "rms_eps"):  # Falcon-Mamba (Transformers 4): weightless
        rms = _modeling_module(mixer).rms_forward
        time_step = rms(time_step, variance_epsilon=mixer.rms_eps)
        b_mat = rms(b_mat, variance_epsilon=mixer.rms_eps)
        c_mat = rms(c_mat, variance_epsilon=mixer.rms_eps)
    dt = F.softplus(mixer.dt_proj(time_step)).transpose(1, 2)  # (B, d_inner, L)

    a_mat = -torch.exp(mixer.A_log.float())  # (d_inner, state)
    discrete_a = torch.exp(a_mat[None, :, None, :] * dt[:, :, :, None])
    delta_b_u = (
        dt[:, :, :, None] * b_mat[:, None, :, :].float() * x[:, :, :, None].float()
    )

    state = torch.zeros(
        batch_size,
        x.shape[1],
        mixer.ssm_state_size,
        device=x.device,
        dtype=dtype,
    )
    outputs = []
    for i in range(seq_len):
        state = discrete_a[:, :, i, :] * state + delta_b_u[:, :, i, :]
        step = torch.matmul(state.to(dtype), c_mat[:, i, :].unsqueeze(-1))
        outputs.append(step[:, :, 0])
    scan = torch.stack(outputs, dim=-1)  # (B, d_inner, L)
    scan = scan.add(x * mixer.D[None, :, None]).mul(mixer.act(gate))
    return mixer.out_proj(scan.transpose(1, 2))


def _module_conv1d(mixer: nn.Module, original: Callable) -> Callable:
    """``causal_conv1d_fn`` that calls the block's conv module for its weight."""
    convs = _depthwise_convs(mixer)

    def conv_fn(
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        activation: str | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        conv = convs.get(weight.data_ptr())
        if conv is None or hidden_states.ndim != 3:
            return original(hidden_states, weight, bias, activation, **kwargs)
        seq_len = hidden_states.shape[-1]
        out = conv(hidden_states.to(conv.weight.dtype))[..., :seq_len]
        if activation is not None:
            act = getattr(mixer, "act", None)
            if act is None:  # no module held; the modelling file's table has it
                act = _modeling_module(mixer).ACT2FN[activation]
            out = act(out)
        return out.to(hidden_states.dtype)

    return conv_fn


def _none(*_args: object, **_kwargs: object) -> None:
    return None


def _decoding(mixer: nn.Module, args: tuple, kwargs: dict) -> bool:
    """Whether the call continues a cached sequence (incremental decoding).

    Training forwards may still carry a cache object (Jamba passes one
    whenever ``use_cache`` is on); only a cache that already holds this
    layer's state turns the call into a decoding step, which the original
    forward handles.  An unfamiliar cache is left to the original forward.
    """
    cache = kwargs.get("cache_params", args[0] if args else None)
    if cache is None:
        return False
    previous = getattr(cache, "has_previous_state", None)
    if previous is None:
        return True
    if callable(previous):
        try:
            return bool(previous(getattr(mixer, "layer_idx", 0)))
        except TypeError:
            return bool(previous())
    return bool(previous)


def install_sequence_mixer_forward(
    root: nn.Module,
    mixer_names: list[str],
    should_intervene: Callable[[], bool],
) -> list[Callable[[], None]]:
    """Override the named mixers' forwards while *should_intervene* holds;
    returns closures restoring them.

    Mamba-1 mixers take :func:`mamba1_reference_forward`; every other mixer
    runs its own forward with the functional convolution and the
    fused-kernel shortcuts of its modelling module redirected for the call.  A call
    continuing a cached sequence (incremental decoding) always runs the
    original forward.

    Raises:
        ValueError: If a named module is not a mixer an override serves.
    """
    modules = dict(root.named_modules())
    reverts: list[Callable[[], None]] = []
    for name in mixer_names:
        mixer = modules[name]
        original = mixer.forward
        if is_mamba1_mixer(mixer):

            def forward(
                hidden_states: torch.Tensor,
                *args: object,
                _mixer: nn.Module = mixer,
                _original: Callable = original,
                **kwargs: object,
            ) -> torch.Tensor:
                if not should_intervene() or _decoding(_mixer, args, kwargs):
                    return _original(hidden_states, *args, **kwargs)
                return mamba1_reference_forward(
                    _mixer, hidden_states, kwargs.get("attention_mask")
                )

        elif is_functional_conv_mixer(mixer):
            modeling = _modeling_module(mixer)

            def forward(
                hidden_states: torch.Tensor,
                *args: object,
                _mixer: nn.Module = mixer,
                _original: Callable = original,
                _modeling: ModuleType = modeling,
                **kwargs: object,
            ) -> torch.Tensor:
                if not should_intervene() or _decoding(_mixer, args, kwargs):
                    return _original(hidden_states, *args, **kwargs)
                saved: dict[str, object] = {}
                for attr, value in list(vars(_modeling).items()):
                    if attr == _CONV_FN:
                        saved[attr] = value
                        setattr(_modeling, attr, _module_conv1d(_mixer, value))
                    elif callable(value) and attr.endswith(_FUSED_FN_SUFFIXES):
                        saved[attr] = value
                        setattr(_modeling, attr, _none)
                    elif attr == _FAST_PATH_FLAG:
                        saved[attr] = value
                        setattr(_modeling, attr, False)
                try:
                    return _original(hidden_states, *args, **kwargs)
                finally:
                    for attr, value in saved.items():
                        setattr(_modeling, attr, value)

        else:
            raise ValueError(f"{name} is not a sequence mixer an override serves.")

        mixer.forward = forward

        def revert(_mixer: nn.Module = mixer) -> None:
            del _mixer.forward

        reverts.append(revert)
    return reverts

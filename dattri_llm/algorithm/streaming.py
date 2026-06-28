"""Gradient sources for attribution — on-disk and live, behind one contract.

An attributor scores by iterating ``(step, Gradient, hashes)`` blocks.  Where
those blocks come from is abstracted by :class:`GradientSource`, which has two
concrete implementations so the same scoring loop serves both workflows:

* :class:`DiskGradientSource` — *store-then-attribute*: reads pre-collected
  gradients off disk via a
  :class:`~dattri_llm.gradient.file_manager.GradientFileManager`.  ``reusable=True``.
* :class:`GradientStreamer` — *on-the-fly*: runs a forward+backward pass over a
  dataset and yields each per-step block on demand, never persisting it.  Two
  flavours:

  * :class:`EvalGradientStreamer` — frozen probe at a fixed checkpoint;
    ``reusable=True``.  Required by methods with a pre-pass (K-FAC/EK-FAC fit the
    Fisher, then score).
  * :class:`TrainingGradientStreamer` — advances the optimizer as it streams, so
    each step is a real checkpoint along the training trajectory;
    ``reusable=False`` (single-shot).  Suits single-pass methods (TracIn).

The only axis attributors branch on is :attr:`GradientSource.reusable` — whether
the source can be iterated more than once at the *same* parameters.  The
method-specific logic (TracIn dot vs. K-FAC whitening) stays in the attributors;
this module is method-agnostic.
"""

from __future__ import annotations

import contextlib
import logging
from typing import (
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Protocol,
    Tuple,
    Union,
    runtime_checkable,
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from dattri_llm.algorithm.arguments import AttributionArguments
from dattri_llm.algorithm.base import iter_gradient_blocks, resolve_steps
from dattri_llm.gradient.callbacks import CaptureCallback
from dattri_llm.gradient.file_manager import GradientFileManager
from dattri_llm.gradient.gradient import Gradient
from dattri_llm.gradient.hooks import HookManager, HookManagerConfig

logger = logging.getLogger(__name__)

# (step, factorized per-sample gradient block, per-sample input hashes)
StreamBatch = Tuple[int, Gradient, List[str]]

# A loss callable: ``(model, batch) -> scalar loss``.  It MUST run the model on
# ``batch`` (so HookManager's forward pre-hook captures the inputs for hashing)
# and produce a loss whose ``.backward()`` flows through the hooked layers.
LossFn = Callable[[nn.Module, object], torch.Tensor]


@runtime_checkable
class GradientSource(Protocol):
    """A source of per-step ``(step, Gradient, hashes)`` blocks.

    Satisfied by :class:`DiskGradientSource` (on-disk) and
    :class:`GradientStreamer` (live), so an attributor scores either through the
    same loop.  The only behavioural axis is :attr:`reusable`.
    """

    def __iter__(self) -> Iterator[StreamBatch]:
        """(Re)start iteration from the first step."""

    def __len__(self) -> int:
        """Number of ``(step, Gradient, hashes)`` blocks yielded per pass."""

    @property
    def reusable(self) -> bool:
        """Whether the source can be iterated more than once at the *same*
        parameters (required by attributors with a pre-pass, e.g. K-FAC)."""


class DiskGradientSource(GradientSource):
    """A re-iterable :class:`GradientSource` backed by on-disk gradients.

    Wraps a :class:`GradientFileManager` and yields the same
    ``(step, Gradient, hashes)`` blocks as :class:`GradientStreamer`, but read
    from disk (the *store-then-attribute* workflow) rather than computed live.
    Each file is ``torch.load``-ed once even when it holds several requested
    steps.  ``reusable=True``: the parameters are already fixed on disk, so the
    blocks can be re-read any number of times.

    Args:
        file_manager: Manager opened on the gradient directory.
        args: :class:`AttributionArguments` (dataloader settings + logging).
        steps: Which stored steps to consume (e.g. the checkpoint subset for a
            TracIn ensemble).  ``None`` reads every step present.  Validated and
            intersected with what is on disk via :func:`resolve_steps`.
        layer_name: Restrict each block to these layers (``str`` or list);
            ``None`` keeps every stored layer.
        desc: Optional tqdm label for the read (shown when ``verbose``).
        verbose: Whether to show the per-block progress bar (rank-0 only).
    """

    def __init__(
        self,
        file_manager: GradientFileManager,
        args: AttributionArguments,
        *,
        steps: Optional[Iterable[int]] = None,
        layer_name: Optional[Union[str, List[str]]] = None,
        desc: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        self._fm = file_manager
        self._args = args
        if isinstance(layer_name, str):
            self._layer_name: Optional[List[str]] = [layer_name]
        elif layer_name is None:
            self._layer_name = None
        else:
            self._layer_name = list(layer_name)
        # Resolve + validate the step set once (disk is immutable for our run).
        self._steps = resolve_steps(file_manager, steps)
        self._desc = desc
        self._verbose = verbose

    @property
    def reusable(self) -> bool:
        return True

    def __len__(self) -> int:
        # One block per (file, step) pair — matches what __iter__ yields.
        return sum(len(by_step) for _, by_step in self._fm.iter_steps(self._steps))

    def __iter__(self) -> Iterator[StreamBatch]:
        return iter_gradient_blocks(
            self._fm,
            self._steps,
            self._args,
            layer_name=self._layer_name,
            desc=self._desc,
            verbose=self._verbose,
        )


def _default_loss_fn(model: nn.Module, batch: object) -> torch.Tensor:
    """HuggingFace convention: ``model(**batch).loss``.

    Override via the ``loss_fn`` constructor argument for non-HF models.
    """
    if not isinstance(batch, dict):
        raise TypeError(
            "The default loss_fn expects a dict batch (model(**batch).loss). "
            "Pass a custom loss_fn for other batch formats."
        )
    out = model(**batch)
    loss = getattr(out, "loss", None)
    if loss is None:
        raise ValueError(
            "Default loss_fn expected the model output to carry a `.loss`; "
            "none found. Pass an explicit loss_fn."
        )
    return loss


class GradientStreamer(GradientSource):
    """Yields per-step ``(step, Gradient, hashes)`` from a live forward+backward pass.

    Each step mirrors :meth:`transformers.Trainer.training_step` and the inner
    training loop, so that **under the same arguments the trajectory matches the
    HF Trainer**: ``args`` drives mixed precision (``bf16``/``fp16`` autocast, with
    a dynamic ``GradScaler`` for fp16), gradient clipping (``max_grad_norm``), and
    gradient checkpointing, and the per-step order is
    forward → backward → clip → ``optimizer.step`` → ``scheduler.step`` → ``zero_grad``.
    DeepSpeed is **not** supported (the hook capture is incompatible with its
    engine); a ``deepspeed`` config is ignored with a warning.  Verified bit-exact
    against ``Trainer`` in ``scripts/verify_streamer_vs_trainer.py``.

    Use as a context manager (it registers/removes the :class:`HookManager`
    hooks), then iterate:

    .. code-block:: python

        with EvalGradientStreamer(model, ds, args, batch_size=8) as src:
            for step, grad, hashes in src:
                ...

    Args:
        model: The **unwrapped** model.  Hooks are registered on its submodules,
            then the streamer wraps it in DDP/FSDP per ``args`` (mirroring HF
            ``Trainer._wrap_model``), so the hooks survive wrapping.  Do not
            pre-wrap it.
        dataset: The dataset to stream.  Iterated with ``shuffle=False`` so the
            row/column order is deterministic and the input hashes are stable.
        args: :class:`AttributionArguments`; supplies ``device`` and the
            ``dataloader_*`` settings.
        batch_size: Per-step batch size (``per_device_train_batch_size`` or
            ``per_device_eval_batch_size``, chosen by the caller).
        enable_update: If ``True``, step the optimizer/scheduler after each
            backward, so each step is a distinct checkpoint along the training
            trajectory (single-shot; ``reusable=False``).  If ``False`` (default)
            the model is frozen and the source is re-iterable.
        loss_fn: ``(model, batch) -> scalar``.  Defaults to ``model(**batch).loss``.
            Must run the model on the batch so inputs are captured for hashing.
        optimizer: Required iff ``enable_update``.
        scheduler: Optional LR scheduler, stepped after the optimizer.
        config: :class:`HookManagerConfig` controlling which layers are hooked.
            Defaults to ``HookManagerConfig()`` (all linear-capable layers).
        collate_fn: Optional DataLoader collate (e.g. an HF data collator).
        checkpoint_step: The step index stamped on every yielded block when the
            model is frozen (all batches share one checkpoint).  Ignored when
            ``enable_update`` (the per-batch optimizer-step index is used).
    """

    def __init__(
        self,
        model: nn.Module,
        dataset: Dataset,
        args: AttributionArguments,
        *,
        batch_size: int,
        enable_update: bool = False,
        loss_fn: Optional[LossFn] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[object] = None,
        config: Optional[HookManagerConfig] = None,
        collate_fn: Optional[Callable] = None,
        checkpoint_step: int = 0,
    ) -> None:
        if enable_update and optimizer is None:
            raise ValueError("enable_update=True requires an optimizer.")

        self._model = model
        self._dataset = dataset
        self._args = args
        self._batch_size = batch_size
        self.enable_update = enable_update
        self._loss_fn = loss_fn if loss_fn is not None else _default_loss_fn
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._collate_fn = collate_fn
        self._checkpoint_step = checkpoint_step

        self._capture = CaptureCallback()
        # Register hooks on the UNWRAPPED submodules first, then wrap, so the
        # hooks survive DDP/FSDP wrapping (the documented collection ordering).
        self._hm = HookManager(
            model,
            config=config if config is not None else HookManagerConfig(),
            callbacks=[self._capture],
        )

        # ── Compute-shaping options, mirroring HF ``Trainer`` so that the same
        # ``AttributionArguments`` yield the same forward/backward and (for the
        # training streamer) the same optimizer trajectory. ──────────────────
        if args.deepspeed:
            raise NotImplementedError(
                "GradientStreamer does not support DeepSpeed (the hook-based "
                "capture is incompatible with its engine); the `deepspeed` config "
                "is ignored. Run single-process, DDP, or FSDP instead."
            )
        # Gradient checkpointing must be enabled on the *unwrapped* model, before
        # DDP/FSDP wrapping — exactly as Trainer does (Trainer._inner_training_loop).
        if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs
            )
        # Mixed precision: bf16/fp16 autocast around the forward; fp16 also needs a
        # dynamic GradScaler — but only when we actually step the optimizer (a
        # frozen probe must capture *unscaled* gradients).  bf16 needs no scaler.
        self._amp_dtype: Optional[torch.dtype] = (
            torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else None
        )
        self._scaler = torch.amp.GradScaler(
            "cuda", enabled=bool(args.fp16 and enable_update and not args.use_cpu)
        )
        self._max_grad_norm = args.max_grad_norm

        # ``_model`` is the unwrapped reference (hooks, params, eval/grad state);
        # ``_fwd_model`` is what forward/backward actually run on (DDP/FSDP when
        # distributed, otherwise the same object).
        self._fwd_model = self._wrap_model(model, args)

        self._loader = self._build_loader()
        # Iteration state.
        self._batch_iter: Optional[Iterator] = None
        self._batch_index: int = 0
        self._consumed: bool = False
        # Saved-state for frozen probes (restored on __exit__).
        self._entered: bool = False
        self._collect_ctx = None
        self._saved_grads: Optional[Dict[str, Optional[torch.Tensor]]] = None
        self._was_training: bool = False

    # ------------------------------------------------------------------ #
    # GradientSource contract                                            #
    # ------------------------------------------------------------------ #

    @property
    def reusable(self) -> bool:
        return not self.enable_update

    @property
    def model(self) -> nn.Module:
        return self._model

    def __len__(self) -> int:
        return len(self._loader)

    # ------------------------------------------------------------------ #
    # Hook lifecycle                                                      #
    # ------------------------------------------------------------------ #

    def __enter__(self) -> "GradientStreamer":
        if self._entered:
            raise RuntimeError("GradientStreamer is already active.")
        self._entered = True
        # Seed at the start of the pass, exactly as Trainer does at the top of
        # `_inner_training_loop`: enable_full_determinism (also seeds) or set_seed.
        from transformers import enable_full_determinism, set_seed

        if self._args.full_determinism:
            enable_full_determinism(self._args.seed)
        else:
            set_seed(self._args.seed)
        if not self.enable_update:
            # Freeze: snapshot grads + force eval() so repeated passes are
            # bit-identical (dropout/etc. would make the K-FAC fit and score
            # passes disagree).
            self._saved_grads = {
                n: (p.grad.detach().clone() if p.grad is not None else None)
                for n, p in self._model.named_parameters()
            }
            self._was_training = self._fwd_model.training
            self._fwd_model.eval()
        self._collect_ctx = self._hm.collect()
        self._collect_ctx.__enter__()
        return self

    def __exit__(self, *exc) -> bool:
        try:
            if self._collect_ctx is not None:
                self._collect_ctx.__exit__(*exc)
        finally:
            self._hm.remove()
            if not self.enable_update:
                if self._was_training:
                    self._fwd_model.train()
                if self._saved_grads is not None:
                    for n, p in self._model.named_parameters():
                        p.grad = self._saved_grads[n]
            self._entered = False
        return False

    # ------------------------------------------------------------------ #
    # Iteration                                                           #
    # ------------------------------------------------------------------ #

    def __iter__(self) -> Iterator[StreamBatch]:
        if not self._entered:
            raise RuntimeError(
                "Use the streamer as a context manager before iterating: "
                "`with streamer as s: for ... in s: ...`."
            )
        if self._consumed and not self.reusable:
            raise RuntimeError(
                "This streamer is single-shot (enable_update=True) and was "
                "already consumed; the training trajectory cannot be replayed."
            )
        self._batch_iter = iter(self._loader)
        self._batch_index = 0
        # Share a zero baseline with the HookManager's capture counter, so this
        # pass's ``record.step`` aligns with ``_batch_index`` and can be used to
        # validate that exactly one capture step happened per yielded batch.
        self._hm.reset_steps()
        self._consumed = True
        return self

    def __next__(self) -> StreamBatch:
        if self._batch_iter is None:
            raise RuntimeError("Call iter(streamer) before next(streamer).")
        batch = next(self._batch_iter)  # raises StopIteration at the end
        batch = self._to_device(batch)

        self._forward_backward(batch)
        if self.enable_update:
            self._optimizer_step()

        record = self._capture.record
        if record is None:
            raise RuntimeError(
                "No gradient was captured this step. Ensure loss_fn runs the "
                "model on the batch and backward flows through the hooked layers."
            )
        # Consistency check: with both counters zeroed at __iter__, the manager's
        # capture index must advance exactly once per batch.  A mismatch means a
        # step was captured that the streamer did not account for (e.g. an extra
        # backward, gradient accumulation, or a missed/duplicated capture).
        if record.step != self._batch_index:
            raise RuntimeError(
                f"Step desync: HookManager captured step {record.step} but the "
                f"streamer is on batch {self._batch_index}. Exactly one capture "
                f"step is expected per streamed batch."
            )

        step = self._step_label()
        grad = record.gradient
        hashes = (
            list(record.input_hash)
            if isinstance(record.input_hash, list)
            else [record.input_hash]
        )
        self._batch_index += 1
        return step, grad, hashes

    # ------------------------------------------------------------------ #
    # Forward / backward / optimizer step (mirrors HF Trainer)            #
    # ------------------------------------------------------------------ #

    def _autocast(self):
        """Mixed-precision context for the forward pass (``Trainer``'s
        ``autocast_smart_context_manager`` / accelerate autocast).  No-op in
        full precision."""
        if self._amp_dtype is None:
            return contextlib.nullcontext()
        device_type = "cuda" if (
            torch.cuda.is_available() and not self._args.use_cpu
        ) else "cpu"
        return torch.autocast(device_type=device_type, dtype=self._amp_dtype)

    def _forward_backward(self, batch: object) -> torch.Tensor:
        """One forward + backward, mirroring ``Trainer.training_step``.

        Runs on the wrapped model so DDP/FSDP gradient sync engages; the hooks
        still fire on the unwrapped submodules they were registered on.  fp16
        scales the loss before backward (``accelerator.backward``); bf16/fp32 do
        not.  Leaves the per-sample gradient in ``self._capture.record``.
        """
        if self.enable_update:
            self._fwd_model.train()        # Trainer calls model.train() each step
        self._fwd_model.zero_grad(set_to_none=True)
        self._capture.record = None
        with self._autocast():
            loss = self._loss_fn(self._fwd_model, batch)
        if self._scaler.is_enabled():
            self._scaler.scale(loss).backward()
        else:
            loss.backward()
        return loss

    def _optimizer_step(self) -> None:
        """Clip → step → schedule, mirroring ``Trainer``'s inner loop.

        Order matches ``Trainer._inner_training_loop``: unscale (fp16) →
        ``clip_grad_norm_(max_grad_norm)`` → ``optimizer.step`` → ``scheduler.step``.
        ``zero_grad`` happens at the start of the next ``_forward_backward``.
        """
        if self._scaler.is_enabled():
            self._scaler.unscale_(self._optimizer)
        if self._max_grad_norm is not None and self._max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self._model.parameters(), self._max_grad_norm)
        if self._scaler.is_enabled():
            self._scaler.step(self._optimizer)
            self._scaler.update()
        else:
            self._optimizer.step()          # type: ignore[union-attr]
        if self._scheduler is not None:
            self._scheduler.step()

    # ------------------------------------------------------------------ #
    # Internals                                                           #
    # ------------------------------------------------------------------ #

    def _step_label(self) -> int:
        """Semantic step stamp for the current block (the attribution index).

        Frozen: a constant checkpoint index (all batches share one checkpoint).
        Updating: the optimizer-step index (each batch is its own checkpoint) —
        which equals the manager's per-pass ``record.step`` (cross-checked in
        :meth:`__next__`).  ``_batch_index`` always tracks capture order; only the
        *label* differs between the two modes.
        """
        return self._batch_index if self.enable_update else self._checkpoint_step

    def _build_loader(self) -> DataLoader:
        kwargs: dict = {
            "dataset": self._dataset,
            "batch_size": self._batch_size,
            "num_workers": self._args.dataloader_num_workers,
            "pin_memory": self._args.dataloader_pin_memory,
        }
        # Distributed: each rank streams a disjoint shard via DistributedSampler
        # (sampler and shuffle are mutually exclusive, so set only one). Content
        # hashes stay globally unique, so per-rank rows concatenate cleanly — the
        # on-the-fly analogue of GradientFileManager's per-rank index merge.
        if self._args.world_size > 1:
            kwargs["sampler"] = DistributedSampler(
                self._dataset,
                num_replicas=self._args.world_size,
                rank=self._args.process_index,
                shuffle=False,
            )
        else:
            kwargs["shuffle"] = False
        if self._collate_fn is not None:
            kwargs["collate_fn"] = self._collate_fn
        if self._args.dataloader_num_workers > 0:
            kwargs["persistent_workers"] = self._args.dataloader_persistent_workers
            if self._args.dataloader_prefetch_factor is not None:
                kwargs["prefetch_factor"] = self._args.dataloader_prefetch_factor
        return DataLoader(**kwargs)

    def _wrap_model(self, model: nn.Module, args: AttributionArguments) -> nn.Module:
        """Wrap the model for distributed execution, mirroring HF ``Trainer``.

        Hooks are already registered on ``model``'s submodules (in ``__init__``),
        so wrapping here preserves them.  Returns the model unchanged when not
        distributed.  The returned module is the forward/backward target; the
        unwrapped ``model`` stays the hook + parameter reference.

        Args:
            model: The unwrapped model.
            args: Supplies ``fsdp``/``fsdp_config`` (FSDP), ``world_size`` +
                ``ddp_find_unused_parameters`` (DDP), and the local rank.

        Returns:
            The (possibly wrapped) module to run forward/backward on.
        """
        fsdp = args.fsdp
        fsdp_tokens = set(fsdp.split() if isinstance(fsdp, str) else (fsdp or []))

        if fsdp_tokens:
            from torch.distributed.fsdp import CPUOffload
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp import ShardingStrategy

            strategy = (
                ShardingStrategy.SHARD_GRAD_OP
                if "shard_grad_op" in fsdp_tokens
                else ShardingStrategy.FULL_SHARD
            )
            fsdp_kwargs = dict(args.fsdp_config or {})
            # use_orig_params keeps the original submodule params (what the hooks captured and what per-sample gradient reads need).
            fsdp_kwargs.setdefault("use_orig_params", True)
            fsdp_kwargs.setdefault("sharding_strategy", strategy)
            if "offload" in fsdp_tokens:
                fsdp_kwargs.setdefault("cpu_offload", CPUOffload(offload_params=True))
            return FSDP(model, **fsdp_kwargs)

        if args.world_size > 1:
            device_ids = (
                [args.local_process_index] if torch.cuda.is_available() else None
            )
            return nn.parallel.DistributedDataParallel(
                model,
                device_ids=device_ids,
                find_unused_parameters=bool(args.ddp_find_unused_parameters),
            )

        return model

    def _to_device(self, batch: object) -> object:
        device = self._args.device
        if isinstance(batch, dict):
            return {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
        if isinstance(batch, (list, tuple)):
            return type(batch)(
                v.to(device) if isinstance(v, torch.Tensor) else v for v in batch
            )
        if isinstance(batch, torch.Tensor):
            return batch.to(device)
        return batch


class EvalGradientStreamer(GradientStreamer):
    """Frozen, re-iterable gradient probe at a fixed checkpoint (``reusable=True``).

    The model is not updated; all yielded blocks are stamped with
    ``checkpoint_step``.  Use for the test side of any attributor, and for the
    train side of methods that need a pre-pass (K-FAC/EK-FAC fit then score).
    """

    def __init__(
        self,
        model: nn.Module,
        dataset: Dataset,
        args: AttributionArguments,
        *,
        batch_size: int,
        loss_fn: Optional[LossFn] = None,
        config: Optional[HookManagerConfig] = None,
        collate_fn: Optional[Callable] = None,
        checkpoint_step: int = 0,
    ) -> None:
        super().__init__(
            model,
            dataset,
            args,
            batch_size=batch_size,
            enable_update=False,
            loss_fn=loss_fn,
            config=config,
            collate_fn=collate_fn,
            checkpoint_step=checkpoint_step,
        )


class TrainingGradientStreamer(GradientStreamer):
    """Trajectory-following, single-shot gradient stream (``reusable=False``).

    Steps the optimizer (and scheduler) after each backward, so each yielded
    block is a distinct checkpoint, stamped with its optimizer-step index.
    Suits single-pass attributors (TracIn). Cannot be re-iterated.
    """

    def __init__(
        self,
        model: nn.Module,
        dataset: Dataset,
        args: AttributionArguments,
        *,
        batch_size: int,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[object] = None,
        loss_fn: Optional[LossFn] = None,
        config: Optional[HookManagerConfig] = None,
        collate_fn: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            model,
            dataset,
            args,
            batch_size=batch_size,
            enable_update=True,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            collate_fn=collate_fn,
        )

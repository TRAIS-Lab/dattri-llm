"""Attribution arguments for dattri-llm attributors.

``AttributionArguments`` is a :func:`dataclasses.dataclass` that centralises
every configuration option needed to run a training-data attribution algorithm:
output path, batch sizes, gradient settings, mixed precision, reproducibility,
DataLoader behaviour, and distributed / large-model options.

It follows the same conventions as
:class:`transformers.TrainingArguments`:

* Every field carries a ``metadata={"help": "..."}`` string so the class can
  be driven from an argument parser (e.g. ``HfArgumentParser``) without any
  extra boilerplate.
* ``__post_init__`` normalises inputs, parses JSON-string dict fields, and
  calls :meth:`_validate` for cross-field consistency checks.
* Computed attributes (``device``, ``world_size``, ...) are exposed as
  ``@property`` / ``@cached_property`` so callers never need to compute them
  manually.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Optional, Union

import torch


logger = logging.getLogger(__name__)

# Fields whose CLI value may be a JSON string or a path to a JSON file.
_DICT_FIELDS = ("gradient_checkpointing_kwargs", "fsdp_config", "deepspeed")


# -----------------------------------------------------------------------------
# AttributionArguments
# -----------------------------------------------------------------------------

@dataclass
class AttributionArguments:
    """Configuration for running a training-data attribution algorithm.

    ``AttributionArguments`` mirrors the interface of
    :class:`transformers.TrainingArguments` so that users familiar with the
    HuggingFace ecosystem can configure an attributor without learning a new
    API.  All fields can be supplied via :class:`transformers.HfArgumentParser`
    or any compatible argument-parsing library.

    Example::

        args = AttributionArguments(
            output_dir="attribution_results/",
            per_device_train_batch_size=8,
            per_device_eval_batch_size=16,
            bf16=True,
            seed=42,
        )

    Args:
        output_dir: Directory where attribution scores and checkpoints are
            written.
        per_device_train_batch_size: Batch size per GPU/CPU for processing
            training examples when computing attribution scores.
        per_device_eval_batch_size: Batch size per GPU/CPU for processing
            evaluation (query) examples.
        max_grad_norm: Maximum gradient norm for clipping.  Set to ``None`` to
            disable clipping.
        gradient_checkpointing: Enable gradient checkpointing to trade compute
            for memory when running backward passes for gradient collection.
        gradient_checkpointing_kwargs: Extra keyword arguments forwarded to
            :meth:`torch.nn.Module.gradient_checkpointing_enable`.
        bf16: Use bfloat16 mixed precision.
        fp16: Use float16 mixed precision.
        tf32: Enable TF32 tensor cores on Ampere+ GPUs for matrix
            multiplications.
        seed: Random seed for Python, NumPy, and PyTorch.
        data_seed: Separate seed for DataLoader worker processes.  Falls back
            to ``seed`` when ``None``.
        full_determinism: Force fully deterministic operations
            (``torch.use_deterministic_algorithms(True)``).  May slow
            execution and is incompatible with some CUDA kernels.
        dataloader_num_workers: Number of subprocesses for DataLoader data
            loading.  ``0`` means data is loaded in the main process.
        dataloader_pin_memory: Pin DataLoader tensors to CUDA pinned memory for
            faster host-to-device transfers.  Enabled by default when a GPU
            is available.
        dataloader_persistent_workers: Keep DataLoader worker processes alive
            between batches.  Reduces per-epoch worker startup overhead.
        dataloader_prefetch_factor: Number of batches each DataLoader worker
            prefetches.  ``None`` uses the PyTorch default (``2`` when
            ``num_workers > 0``).
        ddp_find_unused_parameters: Passed to
            :class:`torch.nn.parallel.DistributedDataParallel` as
            ``find_unused_parameters``.  ``None`` lets PyTorch choose.
        fsdp: FSDP strategy string or list of strategies.  Mirrors
            :attr:`transformers.TrainingArguments.fsdp`.
        fsdp_config: Dict (or path to JSON file) of FSDP configuration options.
        deepspeed: DeepSpeed configuration dict or path to a DeepSpeed JSON
            config file.
    """

    # -- Output ---------------------------------------------------------------

    output_dir: str = field(
        metadata={
            "help": (
                "Directory where attribution scores, intermediate checkpoints, "
                "and run metadata are written."
            )
        },
    )

    # -- Batch sizes ----------------------------------------------------------

    per_device_train_batch_size: int = field(
        default=8,
        metadata={
            "help": (
                "Batch size per GPU/CPU for iterating over training examples "
                "when computing per-sample gradients or attribution scores."
            )
        },
    )

    per_device_eval_batch_size: int = field(
        default=8,
        metadata={
            "help": (
                "Batch size per GPU/CPU for iterating over evaluation (query) "
                "examples when computing attribution scores."
            )
        },
    )

    # -- Optimizer / scheduler (live trajectory collection) -------------------
    # Used only by GradientStreamer when ``enable_update=True`` and no optimizer is
    # supplied: the trajectory optimizer/scheduler are built from these, mirroring
    # transformers.TrainingArguments.

    learning_rate: float = field(
        default=5e-5, metadata={"help": "Initial learning rate for the trajectory optimizer."},
    )
    weight_decay: float = field(
        default=0.0, metadata={"help": "Weight decay (excludes bias / norm params, as in HF)."},
    )
    optim: str = field(
        default="adamw_torch",
        metadata={"help": "Trajectory optimizer: 'adamw_torch' or 'sgd'.",
                  "choices": ["adamw_torch", "sgd"]},
    )
    adam_beta1: float = field(default=0.9, metadata={"help": "AdamW beta1."})
    adam_beta2: float = field(default=0.999, metadata={"help": "AdamW beta2."})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "AdamW epsilon."})
    lr_scheduler_type: str = field(
        default="linear",
        metadata={"help": "LR schedule (transformers.get_scheduler name), e.g. 'linear', 'constant'."},
    )
    warmup_steps: int = field(default=0, metadata={"help": "LR warmup steps."})

    # -- Gradient behaviour ---------------------------------------------------

    max_grad_norm: Optional[float] = field(
        default=1.0,
        metadata={
            "help": (
                "Maximum L2 norm for gradient clipping during backward passes "
                "used for gradient collection.  Set to ``None`` to disable."
            )
        },
    )

    gradient_checkpointing: bool = field(
        default=False,
        metadata={
            "help": (
                "Enable gradient checkpointing when running backward passes for "
                "gradient collection.  Trades ~20%% extra compute for reduced "
                "activation memory."
            )
        },
    )

    gradient_checkpointing_kwargs: Optional[Union[dict, str]] = field(
        default=None,
        metadata={
            "help": (
                "Extra keyword arguments forwarded to "
                "``model.gradient_checkpointing_enable()``.  "
                "May be supplied as a JSON string or a path to a JSON file."
            )
        },
    )

    # -- Hardware -------------------------------------------------------------

    use_cpu: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to use cpu. If set to False, we will use the "
                "available torch device/backend."
            )
        },
    )

    # -- Precision ------------------------------------------------------------

    bf16: bool = field(
        default=False,
        metadata={
            "help": (
                "Use bfloat16 mixed precision for forward and backward passes. "
                "Requires Ampere or newer GPU hardware.  Preferred over fp16 "
                "for better numerical stability."
            )
        },
    )

    fp16: bool = field(
        default=False,
        metadata={
            "help": (
                "Use float16 mixed precision.  Consider bf16 instead when "
                "hardware supports it."
            )
        },
    )

    tf32: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Enable TF32 tensor cores on Ampere+ GPUs for matrix "
                "multiplications (``torch.backends.cuda.matmul.allow_tf32``).  "
                "``None`` leaves the PyTorch default unchanged."
            )
        },
    )

    # -- Reproducibility ------------------------------------------------------

    seed: int = field(
        default=42,
        metadata={
            "help": (
                "Random seed applied to Python ``random``, NumPy, and PyTorch "
                "at the start of attribution.  Controls weight-init and "
                "DataLoader shuffling."
            )
        },
    )

    data_seed: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Seed for DataLoader worker processes.  Falls back to ``seed`` "
                "when ``None``.  Set explicitly to decouple data-loading "
                "randomness from model randomness."
            )
        },
    )

    full_determinism: bool = field(
        default=False,
        metadata={
            "help": (
                "Call ``torch.use_deterministic_algorithms(True)`` for "
                "fully reproducible results.  May be slower and incompatible "
                "with some CUDA kernels (e.g. certain attention implementations)."
            )
        },
    )

    # -- DataLoader behaviour -------------------------------------------------

    dataloader_num_workers: int = field(
        default=0,
        metadata={
            "help": (
                "Number of subprocesses for DataLoader data loading.  "
                "``0`` loads data in the main process.  Increase for I/O-bound "
                "workloads where data pre-processing is a bottleneck."
            )
        },
    )

    dataloader_pin_memory: bool = field(
        default=True,
        metadata={
            "help": (
                "Pin DataLoader output tensors to CUDA pinned memory for faster "
                "host-to-device transfers.  Automatically disabled when no GPU "
                "is available."
            )
        },
    )

    dataloader_persistent_workers: bool = field(
        default=False,
        metadata={
            "help": (
                "Keep DataLoader worker processes alive between dataset "
                "iterations.  Reduces per-epoch worker startup overhead when "
                "``dataloader_num_workers > 0``."
            )
        },
    )

    dataloader_prefetch_factor: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of batches each DataLoader worker prefetches ahead of "
                "consumption.  ``None`` uses the PyTorch default (``2`` when "
                "``dataloader_num_workers > 0``)."
            )
        },
    )

    # -- Distributed / large-model --------------------------------------------

    ddp_find_unused_parameters: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Passed to ``DistributedDataParallel(find_unused_parameters=...)``."
                "  ``None`` lets PyTorch choose the default (``False`` for "
                "gradient-as-bucket-view mode).  Set to ``True`` only if your "
                "model has parameters that do not receive gradients on every step."
            )
        },
    )

    fsdp: Union[str, list[str]] = field(
        default="",
        metadata={
            "help": (
                "FSDP sharding strategy.  Pass a space-separated string or a "
                "list containing one or more of: ``'full_shard'``, "
                "``'shard_grad_op'``, ``'offload'``, ``'auto_wrap'``.  "
                "Empty string disables FSDP."
            ),
            "choices": ["full_shard", "shard_grad_op", "offload", "auto_wrap"],
        },
    )

    fsdp_config: Optional[Union[dict, str]] = field(
        default=None,
        metadata={
            "help": (
                "FSDP configuration passed to the ``FullyShardedDataParallel`` "
                "constructor.  May be a ``dict``, a JSON string, or a path to a "
                "JSON file."
            )
        },
    )

    deepspeed: Optional[Union[dict, str]] = field(
        default=None,
        metadata={
            "help": (
                "DeepSpeed configuration.  May be a ``dict``, a JSON string, or "
                "a path to a DeepSpeed JSON config file.  ``None`` disables "
                "DeepSpeed."
            )
        },
    )

    # -------------------------------------------------------------------------
    # Initialisation
    # -------------------------------------------------------------------------

    def __post_init__(self) -> None:
        # Expand ~ in output_dir so Path operations work correctly.
        self.output_dir = os.path.expanduser(self.output_dir)

        # Parse JSON-string or JSON-file dict fields.
        for fname in _DICT_FIELDS:
            value = getattr(self, fname)
            if isinstance(value, str):
                if value.startswith("{"):
                    setattr(self, fname, json.loads(value))
                elif os.path.isfile(value):
                    with open(value) as f:
                        setattr(self, fname, json.load(f))

        # data_seed defaults to seed when not explicitly set.
        if self.data_seed is None:
            self.data_seed = self.seed

        # Disable pin_memory when no GPU is available.
        if self.dataloader_pin_memory and not torch.cuda.is_available():
            logger.info(
                "dataloader_pin_memory=True has no effect because no CUDA device "
                "is available; setting to False."
            )
            self.dataloader_pin_memory = False

        # Apply TF32 setting immediately if specified.
        if self.tf32 is not None:
            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = self.tf32
                torch.backends.cudnn.allow_tf32 = self.tf32
            else:
                logger.warning("tf32 has no effect without a CUDA device.")

        self._validate()

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    def _validate(self) -> None:
        """Raise ``ValueError`` for mutually exclusive or invalid field combinations."""
        if self.bf16 and self.fp16:
            raise ValueError("bf16 and fp16 cannot both be True.")

        if self.per_device_train_batch_size <= 0:
            raise ValueError(
                f"per_device_train_batch_size must be > 0, got {self.per_device_train_batch_size}."
            )
        if self.per_device_eval_batch_size <= 0:
            raise ValueError(
                f"per_device_eval_batch_size must be > 0, got {self.per_device_eval_batch_size}."
            )

        if self.max_grad_norm is not None and self.max_grad_norm <= 0:
            raise ValueError(
                f"max_grad_norm must be > 0 or None, got {self.max_grad_norm}."
            )

        if (
            self.dataloader_prefetch_factor is not None
            and self.dataloader_prefetch_factor <= 0
        ):
            raise ValueError(
                f"dataloader_prefetch_factor must be > 0 or None, "
                f"got {self.dataloader_prefetch_factor}."
            )

        if self.dataloader_persistent_workers and self.dataloader_num_workers == 0:
            logger.warning(
                "dataloader_persistent_workers=True has no effect when "
                "dataloader_num_workers=0."
            )

        if self.full_determinism and (self.bf16 or self.fp16):
            logger.warning(
                "full_determinism=True with mixed precision may raise errors "
                "for non-deterministic CUDA kernels."
            )

    # -------------------------------------------------------------------------
    # Computed properties
    # -------------------------------------------------------------------------

    @cached_property
    def _device_and_n_gpu(self) -> tuple[torch.device, int]:
        """Resolve the local device and GPU count (computed once)."""
        if self.use_cpu:
            device = torch.device("cpu")
            n_gpu = 0
        elif torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", -1))
            if local_rank != -1:
                # Distributed: each rank uses its assigned GPU.
                device = torch.device("cuda", local_rank)
                n_gpu = 1
            else:
                device = torch.device("cuda")
                n_gpu = torch.cuda.device_count()
        else:
            device = torch.device("cpu")
            n_gpu = 0
        return device, n_gpu

    @property
    def device(self) -> torch.device:
        """The :class:`torch.device` used by this process."""
        return self._device_and_n_gpu[0]

    @property
    def n_gpu(self) -> int:
        """Number of GPUs available to this process.

        In distributed training this is always ``1`` (each rank owns one GPU).
        In non-distributed multi-GPU setups this equals
        ``torch.cuda.device_count()``.
        """
        return self._device_and_n_gpu[1]

    @property
    def world_size(self) -> int:
        """Total number of processes participating in distributed execution.

        Returns ``1`` outside of a distributed context.
        """
        ws = int(os.environ.get("WORLD_SIZE", 1))
        return ws if ws > 0 else 1

    @property
    def process_index(self) -> int:
        """Global rank of the current process (0-based).

        Returns ``0`` outside of a distributed context.
        """
        return int(os.environ.get("RANK", 0))

    @property
    def local_process_index(self) -> int:
        """Local rank of the current process on its node (0-based).

        Returns ``0`` outside of a distributed context.
        """
        return int(os.environ.get("LOCAL_RANK", 0))

    @property
    def train_batch_size(self) -> int:
        """Effective training batch size across all GPUs on this process.

        Equals ``per_device_train_batch_size * max(1, n_gpu)``.
        """
        return self.per_device_train_batch_size * max(1, self.n_gpu)

    @property
    def eval_batch_size(self) -> int:
        """Effective evaluation batch size across all GPUs on this process.

        Equals ``per_device_eval_batch_size * max(1, n_gpu)``.
        """
        return self.per_device_eval_batch_size * max(1, self.n_gpu)

    @property
    def should_log(self) -> bool:
        """``True`` only for the process that should write logs (rank 0)."""
        return self.process_index == 0

    @property
    def output_path(self) -> Path:
        """``output_dir`` as a :class:`pathlib.Path` object."""
        return Path(self.output_dir)

    # -------------------------------------------------------------------------
    # Representation
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        fields_str = "\n  ".join(
            f"{k}={v!r}"
            for k, v in self.__dict__.items()
            if not k.startswith("_")
        )
        return f"AttributionArguments(\n  {fields_str}\n)"

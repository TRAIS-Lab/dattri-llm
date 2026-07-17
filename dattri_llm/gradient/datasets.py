"""On-disk Gradient datasets -- file-level reading of stored gradient records.

Shared building blocks for consumers of pre-collected on-disk gradients (the
:class:`~dattri_llm.gradient.streaming.DiskGradientSource`, and through it the
attributors).  Reading on-disk gradients one *sample* at a time would re-read
each batch file once per sample; these helpers instead expose one *file* per
item so a standard DataLoader with ``num_workers > 0`` can prefetch whole
batch blocks in parallel.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from dattri_llm.gradient import ops
from dattri_llm.gradient.gradient import base_layer_name

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from dattri_llm.attribution.arguments import AttributionArguments
    from dattri_llm.gradient.file_manager import GradientFileManager
    from dattri_llm.gradient.gradient import Gradient


def identity_collate(batch: list) -> object:
    """Collate for a ``batch_size=1`` loader: return the single item as-is.

    :class:`GradientFileMultiStepDataset` already yields batched
    :class:`Gradient` blocks per item, so the DataLoader must not stack
    anything; it just unwraps the singleton list.
    """
    (item,) = batch
    return item


def _records_to_block(
    records: list,
    idxs: list[int],
    layer_name: list[str] | None,
    *,
    file_rel: str,
    step: int,
) -> tuple[Gradient, list[str]]:
    grads: list[Gradient] = []
    hashes: list[str] = []
    skipped_param_grad: set[str] = set()
    for idx in idxs:
        rec = records[idx]
        g = rec.gradient
        selected = g.layer_names if layer_name is None else set(layer_name)
        missing = selected - g.layer_names
        if missing:
            raise KeyError(f"Unknown layers: {sorted(missing)}")
        if layer_name is not None:
            # A requested name also covers its derived virtual invocation
            # layers ("lin" pulls in "lin@2", ... -- recorded when the module
            # fired more than once in a step), so restricting attribution to
            # a layer never silently drops part of its gradient.
            selected |= {n for n in g.layer_names if base_layer_name(n) in selected}
        skipped_param_grad.update(
            name for name in selected if g.layer_types[name] == ops.PARAM_GRAD_TYPES
        )
        per_sample = selected - skipped_param_grad
        if not per_sample:
            raise ValueError(
                "No per-sample gradient layers remain after skipping batch-level "
                f"param_grad layers in {file_rel!r} at step {step}.",
            )
        g = g.select_layers(per_sample)
        grads.append(g)
        h = rec.input_hash
        hashes.extend(h if isinstance(h, list) else [h])

    if skipped_param_grad:
        warnings.warn(
            "Skipping batch-level param_grad layers during per-sample "
            f"attribution: {sorted(skipped_param_grad)}.",
            stacklevel=3,
        )

    if not grads:
        raise RuntimeError(f"No records selected for file {file_rel!r} at step {step}.")

    block = grads[0]
    for g in grads[1:]:
        block = block.concatenate(g, dim="batch")

    if block.batch_size != len(hashes):
        raise RuntimeError(
            f"Block batch size ({block.batch_size}) disagrees with hash "
            f"count ({len(hashes)}) for file {file_rel!r} at step {step}.",
        )
    return block, hashes


class GradientFileMultiStepDataset(Dataset):
    """Yields all requested step blocks from each on-disk file.

    Each item performs one :func:`torch.load` and returns
    ``{step: (Gradient_block, hashes)}`` for every requested step represented in
    that file.  This preserves file-level pipelining even when a batch file
    contains multiple collected steps.
    """

    def __init__(
        self,
        file_manager: GradientFileManager,
        steps: list[int],
        layer_name: list[str] | None = None,
    ) -> None:
        self._fm = file_manager
        self._steps = list(steps)
        self._layer_name = list(layer_name) if layer_name is not None else None
        self._files = file_manager.iter_steps(self._steps)

    def __len__(self) -> int:
        return len(self._files)

    def __getitem__(self, i: int) -> dict[int, tuple[Gradient, list[str]]]:
        file_rel, by_step = self._files[i]
        records = self._fm.load_records(file_rel)
        return {
            step: _records_to_block(
                records,
                idxs,
                self._layer_name,
                file_rel=file_rel,
                step=step,
            )
            for step, idxs in by_step.items()
        }


def make_gradient_multistep_dataloader(
    file_manager: GradientFileManager,
    steps: list[int],
    args: AttributionArguments,
    layer_name: list[str] | None = None,
) -> DataLoader:
    """Build a DataLoader yielding per-file blocks for multiple steps."""
    dataset = GradientFileMultiStepDataset(file_manager, steps, layer_name=layer_name)
    kwargs: dict = {
        "dataset": dataset,
        "batch_size": 1,
        "shuffle": False,
        "collate_fn": identity_collate,
        "num_workers": args.dataloader_num_workers,
        "pin_memory": args.dataloader_pin_memory,
    }
    if args.dataloader_num_workers > 0:
        kwargs["persistent_workers"] = args.dataloader_persistent_workers
        if args.dataloader_prefetch_factor is not None:
            kwargs["prefetch_factor"] = args.dataloader_prefetch_factor
    return DataLoader(**kwargs)


def resolve_steps(
    file_manager: GradientFileManager,
    requested: Iterable[int] | None,
) -> list[int]:
    """Resolve which on-disk steps an attributor should consume.

    ``requested=None`` selects every step present on disk.  Otherwise the
    requested steps are intersected with what is available, so an over-specified
    range (e.g. ``range(0, 1000)`` against checkpoints saved every 50 steps) is
    accepted and simply keeps the steps that exist.  An empty intersection is an
    error -- it almost always means a typo'd or wrong step set.

    Args:
        file_manager: Manager opened on the gradient directory.
        requested: Steps to keep, any iterable of ints, or ``None`` for all.

    Returns:
        The ascending list of steps to use.

    Raises:
        ValueError: If ``requested`` is given but matches no available step.
    """
    available = file_manager.available_steps()
    if requested is None:
        return available
    wanted = set(requested)
    selected = [s for s in available if s in wanted]
    if not selected:
        raise ValueError(
            f"None of the requested steps {sorted(wanted)} are present in the "
            f"gradients; available steps: {available}.",
        )
    return selected


def iter_gradient_blocks(
    file_manager: GradientFileManager,
    steps: list[int],
    args: AttributionArguments,
    layer_name: list[str] | None = None,
    desc: str | None = None,
    verbose: bool = False,
) -> Iterator[tuple[int, Gradient, list[str]]]:
    """Yield ``(step, Gradient_block, hashes)`` for *steps*, one load per file.

    Uses the multi-step file loader so each file is ``torch.load``-ed exactly
    once even if it holds records from several requested steps; the recorded
    step is yielded alongside each block so callers can stamp output rows with
    the checkpoint they came from.
    """
    if not steps:
        return
    step_list = list(steps)
    total = sum(len(by_step) for _, by_step in file_manager.iter_steps(step_list))
    blocks = tqdm(
        total=total,
        desc=desc,
        unit="block",
        dynamic_ncols=True,
        leave=False,
        disable=desc is None or not verbose or not args.should_log,
    )
    try:
        for by_step in make_gradient_multistep_dataloader(
            file_manager,
            step_list,
            args,
            layer_name,
        ):
            for step, (block, hashes) in by_step.items():
                yield step, block, hashes
                blocks.update(1)
    finally:
        blocks.close()

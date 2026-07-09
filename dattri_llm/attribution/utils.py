"""Common utilities shared by the attribution algorithms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from dattri_llm.gradient.gradient import Gradient, GradientRecord

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from torch import nn

    from dattri_llm.gradient.file_manager import GradientFileManager
    from dattri_llm.gradient.streaming import GradientStreamer


def normalize_layer_names(
    layer_name: str | list[str] | None,
) -> list[str] | None:
    """Normalize an ``attribute_from_cache`` ``layer_name`` argument.

    A single string becomes a one-element list; ``None`` (score every stored
    layer) passes through.
    """
    if layer_name is None:
        return None
    if isinstance(layer_name, str):
        return [layer_name]
    return list(layer_name)


def task_loss_fn(func: Callable) -> Callable:
    """Adapt a dattri ``AttributionTask`` loss/target -- ``(params, data) -> loss``
    (functorch style) -- to the streamer's ``(model, batch) -> loss``.

    The streamer drives a live model, so we call *func* with that model's current
    parameters; *func* runs the same ``functional_call`` forward the task defines,
    and ``batch`` is the loader's batch in the task's ``data`` format.
    """

    def loss_fn(model: nn.Module, batch: object) -> torch.Tensor:
        return func(dict(model.named_parameters()), batch)

    return loss_fn


def collect_to_disk(
    streamer: GradientStreamer,
    file_manager: GradientFileManager,
) -> None:
    """Run a gradient streamer to completion, persisting every block to disk.

    Iterates ``streamer`` (entering/exiting its context here, so pass a freshly
    built one), saving each ``(step, Gradient, hashes)`` block as a
    :class:`GradientRecord` stamped with the streamer's *semantic* step label --
    the checkpoint index for a frozen probe, or the optimizer-step index for a
    training trajectory.  This is the shared engine of every attributor's
    on-the-fly :meth:`cache`: collect live here, then score with
    ``attribute_from_cache`` (so ``attribute`` == *cache + attribute_from_cache*).
    """
    id_key = streamer.hook_manager.sample_id_key
    with streamer:
        for step, grad, hashes in streamer:
            file_manager.save_bulk(
                [
                    GradientRecord(
                        step=step,
                        input_hash=hashes,
                        gradient=grad,
                        sample_id_key=id_key,
                    ),
                ],
            )


def score_sources(
    train_source: Iterable,
    test_source: Iterable,
    device: object,
    *,
    prepare_test: Callable[[Gradient], object],
    score_block: Callable[[Gradient, object, int], torch.Tensor],
    loop_over_test: bool = False,
) -> tuple[torch.Tensor, list[str], list[int], list[str]]:
    """Shared scoring skeleton for the trajectory-agnostic attributors.

    Both train and test arrive as ``GradientSource`` objects (on-disk
    :class:`~dattri_llm.gradient.streaming.DiskGradientSource` or live
    :class:`~dattri_llm.gradient.streaming.GradientStreamer`), yielding
    ``(step, Gradient, hashes)`` blocks.  The skeleton fixes the test column order
    from the first test pass, scores every train block against every (prepared)
    test block, and returns the pieces an
    :class:`~dattri_llm.attribution.score.AttributionScore` is assembled from -- the
    per-method preparation/scoring lives in the two callables:

    * ``prepare_test(test_g) -> rep`` turns a device-resident test block into the
      representation ``score_block`` consumes (identity for TracIn; whitened /
      FIM reps for K-FAC).
    * ``score_block(train_g, rep, n_test) -> (B_train, n_test)`` is the actual
      (preconditioned) cross-gram for one train block against one prepared test
      block.

    Args:
        train_source: Source of train blocks; iterated **once** (a single-shot
            trajectory stream is fine).
        test_source: Source of test blocks; iterated once to cache, or repeatedly
            when ``loop_over_test`` (which then requires ``test_source.reusable``).
        device: Device the blocks are moved to before scoring.
        prepare_test: The per-method preparation hook described above.
        score_block: The per-method scoring hook described above.
        loop_over_test: Re-stream + re-prepare the test blocks per train block
            (low memory) instead of caching them once (default).

    Returns:
        ``(scores, row_train_ids, row_steps, test_ids)`` -- ``scores`` is
        ``(num_train_rows, num_test)`` on CPU, ``row_steps`` stamps each row with
        the step its train gradient came from, ``test_ids`` is the column order.

    Raises:
        ValueError: If ``loop_over_test`` is requested with a single-shot
            test source.
    """
    if loop_over_test and not getattr(test_source, "reusable", False):
        raise ValueError(
            "loop_over_test=True requires a re-iterable test source "
            "(reusable=True); got a single-shot source.",
        )

    # One pass over the test blocks fixes the column order (hash order as yielded)
    # and, unless looping, caches each block's prepared representation.
    test_ids: list[str] = []
    test_index: dict = {}
    cached_test: list[tuple[object, list[str]]] = []
    for _step, test_g, test_hashes in test_source:
        for h in test_hashes:
            if h not in test_index:
                test_index[h] = len(test_ids)
                test_ids.append(h)
        if not loop_over_test:
            cached_test.append((prepare_test(test_g.to(device)), test_hashes))
    num_test = len(test_ids)

    # Score every train block against every test block.
    row_chunks: list[torch.Tensor] = []
    row_train_ids: list[str] = []
    row_steps: list[int] = []
    for train_step, train_block, train_hashes in train_source:
        train_g = train_block.to(device)
        row = torch.zeros(train_g.batch_size, num_test, dtype=torch.float)
        test_blocks = (
            ((prepare_test(g.to(device)), h) for _s, g, h in test_source)
            if loop_over_test
            else cached_test
        )
        for test_rep, test_hashes in test_blocks:
            cols = [test_index[h] for h in test_hashes]
            block = score_block(train_g, test_rep, len(cols))
            row[:, cols] = block.detach().to("cpu", torch.float)
        row_chunks.append(row)
        row_train_ids.extend(train_hashes)
        row_steps.extend([train_step] * train_g.batch_size)

    scores = (
        torch.cat(row_chunks, dim=0)
        if row_chunks
        else torch.zeros(0, num_test, dtype=torch.float)
    )
    return scores, row_train_ids, row_steps, test_ids

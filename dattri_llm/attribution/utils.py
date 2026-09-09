"""Common utilities shared by the attribution algorithms.

* :func:`normalize_layer_names` / :func:`task_loss_fn` -- small adapters.
* :func:`collect_gradients` -- run a streamer to completion into a store
  (the engine of every attributor's ``cache``).
* :func:`score_sources` -- the inner-product scoring loop: every train block
  against every (transformed) test block, with the dense-materialization
  cache handled here so a method's ``inner_product`` never has to.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

from dattri_llm.gradient.gradient import Gradient, GradientRecord
from dattri_llm.gradient.streaming import rebatch_blocks
from dattri_llm.utils.cache import CacheBudget, TensorCache

if TYPE_CHECKING:
    from collections.abc import Iterable

    from torch import nn

    from dattri_llm.gradient.storage_manager import GradientStorageManager
    from dattri_llm.gradient.streaming import GradientStreamer

# ``inner_product(train_rep, test_rep, dense_cache=...) -> (B_train, B_test)``
InnerProduct = Callable[..., torch.Tensor]
# ``transform(block) -> block``
Transform = Callable[[Gradient], Gradient]


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


def collect_gradients(
    streamer: GradientStreamer,
    store: GradientStorageManager,
    *,
    offload_interval: int = 1,
    on_block: Callable[[int, Gradient, list[str]], None] | None = None,
    async_write: bool | None = None,
) -> GradientStorageManager:
    """Run a gradient streamer to completion, persisting every block to *store*.

    Iterates ``streamer`` (entering/exiting its context here, so pass a freshly
    built one), saving each ``(step, Gradient, hashes)`` block as a
    :class:`GradientRecord` stamped with the streamer's *semantic* step label --
    the checkpoint index for a frozen probe, or the optimizer-step index for a
    training trajectory.  This is the shared engine of every attributor's
    on-the-fly :meth:`cache`: an attributor drives its own streamer, so it saves
    directly here.  (The passive :class:`OffloadCallback` is reserved for the
    *manual* workflow -- offloading as a side effect of a training loop the
    attributor does not control.)  The store may have any residency.

    Args:
        streamer: A freshly built (not yet entered) gradient streamer.
        store: Destination store.
        offload_interval: Number of ``(step)`` blocks accumulated per gradient
            file.  ``1`` (default) writes one file per step.  A larger value
            packs that many steps into each file -- amortising the per-file
            index rewrite over a long trajectory (``enable_update=True``) at the
            cost of holding that many blocks in memory before each flush.
        on_block: Optional ``(step, gradient, hashes)`` hook invoked on **every**
            streamed block, before it is staged.  The on-the-fly analogue of a
            manual collection callback: because the attributor drives the
            streamer (not a :class:`HookManager`), side quantities -- e.g. the
            K-FAC covariances a :class:`KroneckerAccumulator` builds -- are
            accumulated straight off the streamed blocks, so no re-pass over
            the store is needed.  Runs on the (single-shot) collection pass, so
            it sees each block exactly once.
        async_write: Write flush groups through a background
            :class:`~dattri_llm.gradient.async_writer.AsyncGradientWriter`,
            overlapping D2H + disk IO with the next block's forward/backward.
            The writer is drained before this function returns, so the store
            is complete and identical to a synchronous run.  ``None``
            (default) reads ``async_disk_write`` off the streamer's args.

    Returns:
        *store*, for chaining.
    """
    if offload_interval < 1:
        raise ValueError(
            f"offload_interval must be >= 1, got {offload_interval}.",
        )
    if async_write is None:
        async_write = getattr(
            getattr(streamer, "args", None), "async_disk_write", False
        )
    writer = None
    if async_write:
        from dattri_llm.gradient.async_writer import AsyncGradientWriter

        writer = AsyncGradientWriter(store)
    save = writer.submit if writer is not None else store.save_bulk

    id_key = streamer.hook_manager.sample_id_key
    staged: list[GradientRecord] = []
    try:
        with streamer:
            for step, grad, hashes in streamer:
                if on_block is not None:
                    on_block(step, grad, hashes)
                staged.append(
                    GradientRecord(
                        step=step,
                        input_hash=hashes,
                        gradient=grad,
                        sample_id_key=id_key,
                    ),
                )
                if len(staged) >= offload_interval:
                    save(staged)
                    staged = []
            if staged:
                save(staged)
    finally:
        if writer is not None:
            writer.close()
    return store


def _identity(block: Gradient) -> Gradient:
    return block


def _fix_columns(
    hashes: Iterable[str],
    test_ids: list[str],
    test_index: dict[str, int],
) -> list[int]:
    """Assign column indices to *hashes* in first-seen order; returns them."""
    cols = []
    for h in hashes:
        if h not in test_index:
            test_index[h] = len(test_ids)
            test_ids.append(h)
        cols.append(test_index[h])
    return cols


def _merge_dense_test(
    cached_test: list[tuple[Gradient, list[int]]],
) -> list[tuple[Gradient, list[int]]]:
    """Concatenate the cached test reps into one block when they are all dense.

    A dense test side is scored by one GEMM per layer per train block, so
    ``n_test_blocks`` blocks cost ``n_test_blocks`` times the launches for the
    same flops.  Stacking them (one ``cat`` per layer, like the train-side
    re-batching) collapses that to a single block -- which also means no train
    layer is ever materialized *for reuse*, so no dense cache is needed.
    Factorized reps cannot be stacked and are left as they are.
    """
    if len(cached_test) < 2:
        return cached_test
    stream = ((0, rep, cols) for rep, cols in cached_test)
    merged = list(
        rebatch_blocks(stream, batch_size=sum(len(c) for _, c in cached_test))
    )
    return [(rep, cols) for _steps, rep, cols in merged]


def score_sources(
    train_source: Iterable,
    test_source: Iterable,
    device: object,
    *,
    inner_product: InnerProduct,
    transform_train: Transform | None = None,
    transform_test: Transform | None = None,
    batch_size: int = 1,
    loop_over_test: bool = False,
    cache_budget: CacheBudget | None = None,
) -> tuple[torch.Tensor, list[str], list[int], list[str]]:
    """The inner-product scoring loop shared by the trajectory-agnostic attributors.

    Both sides are ``GradientSource`` objects (on-disk
    :class:`~dattri_llm.gradient.streaming.DiskGradientSource` or live
    :class:`~dattri_llm.gradient.streaming.GradientStreamer`) yielding
    ``(step, Gradient, hashes)`` blocks.  The loop fixes the test column order
    from the first test pass, scores every train block against every
    (transformed) test block, and returns the pieces an
    :class:`~dattri_llm.attribution.score.AttributionScore` is assembled from.
    The method-specific work lives in the three callables:

    * ``transform_train(block) -> rep`` / ``transform_test(block) -> rep``
      turn a device-resident block into the representation to score
      (identity by default; e.g. preconditioning on the test side).
    * ``inner_product(train_rep, test_rep, dense_cache=cache) -> (B_train, B_test)``
      scores one pair.  ``dense_cache`` is a
      :class:`~dattri_llm.utils.cache.TensorCache` **scoped to the train block**
      -- handed to :func:`~dattri_llm.gradient.ops.layerwise_cross_dot` it makes
      a factorized train layer materialize once across all the test blocks it
      is scored against.  It is ``None`` when the block meets exactly one test
      block (nothing to reuse), and its budget bounds how much dense state
      is ever retained, so a method never has to reason about memory.

    Args:
        train_source: Source of train blocks; iterated **once** (a single-shot
            trajectory stream is fine).
        test_source: Source of test blocks; iterated once to cache, or repeatedly
            when ``loop_over_test`` (which then requires ``test_source.reusable``).
        device: Device the blocks are moved to before scoring.
        inner_product: The per-pair scoring hook described above.
        transform_train: Per-block train transform (default identity).
        transform_test: Per-block test transform (default identity).
        batch_size: Dense train blocks are re-batched into this many samples
            per scoring batch (:func:`~dattri_llm.gradient.streaming.rebatch_blocks`),
            collapsing tiny per-block matmuls into big GEMMs.  Speed/memory
            only -- scores are identical for every value.  Factorized blocks
            are scored one block at a time regardless.  The cached test reps
            are likewise stacked into a single block when they are all dense.
        loop_over_test: Re-stream + re-transform the test blocks per train block
            (low memory) instead of caching them once (default).  This path
            scores block-by-block (no train-side re-batching).
        cache_budget: Budget of the per-train-block dense cache; ``None`` uses
            the default fraction of free memory on *device*.

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
    transform_train = transform_train or _identity
    transform_test = transform_test or _identity
    budget = cache_budget if cache_budget is not None else CacheBudget(device)

    test_ids: list[str] = []
    test_index: dict[str, int] = {}
    cached_test: list[tuple[Gradient, list[int]]] = []
    for _step, test_g, test_hashes in test_source:
        cols = _fix_columns(test_hashes, test_ids, test_index)
        if not loop_over_test:
            cached_test.append((transform_test(test_g.to(device)), cols))
    num_test = len(test_ids)
    cached_test = _merge_dense_test(cached_test)

    def score_block(train_rep: Gradient, test_blocks: Iterable) -> torch.Tensor:
        """Row chunk of one train rep against every test rep in *test_blocks*."""
        row = torch.zeros(train_rep.batch_size, num_test, dtype=torch.float)
        with TensorCache("memory", budget=budget) as dense_cache:
            for test_rep, cols in test_blocks:
                block = inner_product(train_rep, test_rep, dense_cache=dense_cache)
                row[:, cols] = block.detach().to("cpu", torch.float)
        return row

    row_chunks: list[torch.Tensor] = []
    row_train_ids: list[str] = []
    row_steps: list[int] = []
    if not loop_over_test:
        # Retaining a dense copy of a train block only pays off when it is
        # scored against more than one test block.
        reuse = len(cached_test) > 1
        for steps, train_g, ids in rebatch_blocks(train_source, batch_size):
            train_rep = transform_train(train_g.to(device))
            if reuse:
                row_chunks.append(score_block(train_rep, cached_test))
            else:
                row = torch.zeros(train_rep.batch_size, num_test, dtype=torch.float)
                for test_rep, cols in cached_test:
                    block = inner_product(train_rep, test_rep, dense_cache=None)
                    row[:, cols] = block.detach().to("cpu", torch.float)
                row_chunks.append(row)
            row_train_ids.extend(ids)
            row_steps.extend(steps)
    else:
        for train_step, train_g, train_hashes in train_source:
            train_rep = transform_train(train_g.to(device))

            def test_blocks() -> Iterable:
                for _s, test_g, test_hashes in test_source:
                    cols = [test_index[h] for h in test_hashes]
                    yield transform_test(test_g.to(device)), cols

            row_chunks.append(score_block(train_rep, test_blocks()))
            row_train_ids.extend(train_hashes)
            row_steps.extend([train_step] * train_rep.batch_size)

    scores = (
        torch.cat(row_chunks, dim=0)
        if row_chunks
        else torch.zeros(0, num_test, dtype=torch.float)
    )
    return scores, row_train_ids, row_steps, test_ids

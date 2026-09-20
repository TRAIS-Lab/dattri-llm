"""Common utilities shared by the attribution algorithms.

* :func:`normalize_layer_names` -- a small adapter.
* :func:`collect_gradients` -- run a streamer to completion into a store
  (the engine of every attributor's ``cache``).
* :func:`score_sources` -- the inner-product scoring loop: every train block
  against every (transformed) test block, with the dense-materialization
  cache handled here so a method's ``inner_product`` never has to.
"""

from __future__ import annotations

import json
import pathlib
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

import torch

from dattri_llm.gradient import ops
from dattri_llm.gradient.gradient import Factorized, Gradient, GradientRecord
from dattri_llm.gradient.streaming import rebatch_blocks
from dattri_llm.utils.cache import CacheBudget, TensorCache

if TYPE_CHECKING:
    from collections.abc import Iterable

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


def finalize_factors(
    rep: Gradient,
    *,
    budget: CacheBudget | None = None,
    held: int = 0,
) -> Gradient:
    """Preprocess every raw factorized layer of *rep* once and keep the result.

    Hooks store a layer's factors raw, and every product kernel turns them
    into the form it contracts (the bias column of a linear layer, the
    normalized activation of a norm layer, the unfolded patches of a
    convolution) through :func:`~dattri_llm.gradient.ops.preprocess_factors`.
    A block that is scored more than once would repeat that work on every
    call, so the scoring loop replaces each raw layer by its *final* factors
    (a :class:`~dattri_llm.gradient.gradient.Factorized` with
    ``module_kwargs=None``, which every kernel uses as it is) before the
    block is scored.  The raw payload is released as its replacement is built.

    Embedding layers stay raw: materializing them needs their
    ``module_kwargs``, and their preprocessing is a mask.  Layers that are
    already final or dense pass through.

    Args:
        rep: The block to convert; it is consumed (see
            :meth:`~dattri_llm.gradient.gradient.Gradient.map_layers`).
        budget: When given, the growth of a layer's final factors over its raw
            ones (an unfolded convolution is larger than its input) is admitted
            against it, and a layer that does not fit stays raw.  ``None``
            converts every layer -- right for a train block, whose final
            factors the kernels would allocate anyway.
        held: Bytes already charged to *budget* by the caller.

    Returns:
        The block with its raw factorized layers replaced by final factors.
    """
    names = [
        name
        for name, value in rep.data.items()
        if isinstance(value, Factorized)
        and value.module_kwargs is not None
        and not ops.is_embedding(rep.layer_types[name])
    ]
    if not names:
        return rep
    charged = held

    def finalize(_name: str, value: Factorized, layer_type: str) -> Factorized:
        nonlocal charged
        a, g = ops.preprocess_factorized(value, layer_type)
        final = Factorized(activation=a, pre_activation_grad=g, module_kwargs=None)
        growth = max(final.nbytes - value.nbytes, 0)
        if budget is not None and not budget.fits(growth, charged):
            return value
        charged += growth
        return final

    return rep.map_layers(finalize, layers=names, consume=True)


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
    granularity: str = "instance",
    inner_product_per_token: InnerProduct | None = None,
) -> tuple[torch.Tensor, list[str], list[int], list[str], list[int] | None]:
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
        granularity: ``"instance"`` (default) scores one row per training
            sample; ``"token"`` one row per training token position, scored
            by *inner_product_per_token* instead of *inner_product*.
        inner_product_per_token: ``(train_rep, test_rep) -> (B_train, T,
            B_test)`` -- the per-position decomposition of *inner_product*;
            required under ``granularity="token"``.

    Returns:
        ``(scores, row_train_ids, row_steps, test_ids, row_token_ids)`` --
        ``scores`` is ``(num_rows, num_test)`` on CPU, ``row_steps`` stamps
        each row with the step its train gradient came from, ``test_ids`` is
        the column order, and ``row_token_ids`` is ``None`` at instance
        granularity or the token position of each row at token granularity
        (a training sample of ``T`` positions contributes ``T`` consecutive
        rows; padded positions score zero).

    Raises:
        ValueError: If ``loop_over_test`` is requested with a single-shot
            test source, or on an unknown *granularity*.
    """
    if loop_over_test and not getattr(test_source, "reusable", False):
        raise ValueError(
            "loop_over_test=True requires a re-iterable test source "
            "(reusable=True); got a single-shot source.",
        )
    if granularity not in {"instance", "token"}:
        raise ValueError(
            f"granularity must be 'instance' or 'token', got {granularity!r}.",
        )
    per_token = granularity == "token"
    if per_token and inner_product_per_token is None:
        raise ValueError("granularity='token' needs inner_product_per_token.")
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
    # The loop variable would otherwise keep the last raw test block alive
    # for the whole scoring loop, next to its transformed copy.
    test_g = None
    num_test = len(test_ids)
    cached_test = _merge_dense_test(cached_test)

    def n_rows(train_rep: Gradient) -> int:
        """Rows one train rep contributes: its samples, or samples x positions."""
        if not per_token:
            return train_rep.batch_size
        tokens = [t for t in train_rep.token_dim.values() if t is not None]
        return train_rep.batch_size * (max(tokens) if tokens else 1)

    def score_pair(
        train_rep: Gradient,
        test_rep: Gradient,
        dense_cache: TensorCache | None,
    ) -> torch.Tensor:
        """``(rows, B_test)`` block of one train rep against one test rep."""
        if per_token:
            block = inner_product_per_token(train_rep, test_rep)  # (B, T, B_test)
            return block.reshape(-1, block.shape[-1])
        return inner_product(train_rep, test_rep, dense_cache=dense_cache)

    def score_block(
        train_rep: Gradient,
        test_blocks: Iterable,
        dense_cache: TensorCache | None,
    ) -> torch.Tensor:
        """Row chunk of one train rep against every test rep in *test_blocks*."""
        row = torch.zeros(n_rows(train_rep), num_test, dtype=torch.float)
        for test_rep, cols in test_blocks:
            block = score_pair(train_rep, test_rep, dense_cache)
            row[:, cols] = block.detach().to("cpu", torch.float)
        return row

    row_chunks: list[torch.Tensor] = []
    row_train_ids: list[str] = []
    row_steps: list[int] = []
    row_token_ids: list[int] = []

    def stamp_rows(ids: list[str], steps: list[int], chunk: torch.Tensor) -> None:
        """Label *chunk*'s rows: one per sample, or ``T`` per sample."""
        if not per_token:
            row_train_ids.extend(ids)
            row_steps.extend(steps)
            return
        t = chunk.shape[0] // max(len(ids), 1)
        for h, s in zip(ids, steps, strict=True):
            row_train_ids.extend([h] * t)
            row_steps.extend([s] * t)
            row_token_ids.extend(range(t))

    if not loop_over_test:
        # Retaining a dense copy of a train block only pays off when it is
        # scored against more than one test block (and only the instance-level
        # kernel materializes the train side at all).
        reuse = len(cached_test) > 1 and not per_token
        for steps, train_g, ids in rebatch_blocks(train_source, batch_size):
            train_rep = transform_train(train_g.to(device))
            if reuse:
                with TensorCache("memory", budget=budget) as dense_cache:
                    chunk = score_block(train_rep, cached_test, dense_cache)
            else:
                chunk = score_block(train_rep, cached_test, None)
            row_chunks.append(chunk)
            stamp_rows(ids, steps, chunk)
            # The next request to a live source is the next forward and
            # backward pass: do not carry this block through it.
            train_g = train_rep = None  # noqa: PLW2901
    else:
        for train_step, train_g, train_hashes in train_source:
            train_rep = transform_train(train_g.to(device))

            def test_blocks() -> Iterable:
                for _s, test_g, test_hashes in test_source:
                    cols = [test_index[h] for h in test_hashes]
                    yield transform_test(test_g.to(device)), cols

            with TensorCache("memory", budget=budget) as dense_cache:
                chunk = score_block(train_rep, test_blocks(), dense_cache)
            row_chunks.append(chunk)
            stamp_rows(list(train_hashes), [train_step] * train_rep.batch_size, chunk)
            train_g = train_rep = None  # noqa: PLW2901 - see above

    scores = (
        torch.cat(row_chunks, dim=0)
        if row_chunks
        else torch.zeros(0, num_test, dtype=torch.float)
    )
    return (
        scores,
        row_train_ids,
        row_steps,
        test_ids,
        (row_token_ids if per_token else None),
    )


# --------------------------------------------------------------------------- #
# The learning-rate schedule a trajectory collection records                   #
# --------------------------------------------------------------------------- #

LR_SCHEDULE_FILE = "lr_schedule.json"


def write_lr_schedule(train_gradients_dir: str, lrs: Mapping[int, float]) -> None:
    """Persist the per-step LR actually applied during a trajectory collection
    (``GradientStreamer.learning_rates``) beside the train store.
    """
    root = pathlib.Path(train_gradients_dir)
    root.mkdir(exist_ok=True, parents=True)
    with (root / LR_SCHEDULE_FILE).open("w", encoding="utf-8") as f:
        json.dump({str(k): float(v) for k, v in lrs.items()}, f)


def read_lr_schedule(train_gradients_dir: str) -> dict[int, float] | None:
    """The per-step LR recorded by :func:`write_lr_schedule`, or ``None`` if
    absent (e.g. a directory produced outside the on-the-fly workflow).
    """
    path = pathlib.Path(train_gradients_dir) / LR_SCHEDULE_FILE
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return {int(k): float(v) for k, v in json.load(f).items()}

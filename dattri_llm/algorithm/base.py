"""Abstract base classes for training-data attribution algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Iterable, List, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset

from dattri_llm.algorithm.arguments import AttributionArguments
from dattri_llm.gradient.file_manager import GradientFileManager
from dattri_llm.gradient.gradient import Gradient, GradientRecord

if TYPE_CHECKING:
    from dattri_llm.task import AttributionTask


class BaseAttributor(ABC):
    """Base class for all attributors."""

    @abstractmethod
    def __init__(
        self,
        task: AttributionTask,
        args: AttributionArguments,
    ) -> None:
        """Initialize the attributor.

        Args:
            task: The attribution task. Must be an instance of `AttributionTask`.
            args: Configuration object controlling device placement, batch sizes,
                precision, DataLoader behaviour, and distributed settings.
                See :class:`AttributionArguments` for the full field reference.

        Returns:
            None.
        """

    @abstractmethod
    def cache(self, train_dataset: Dataset) -> None:
        """Precompute and cache values for efficiency.

        The DataLoader is constructed internally from ``args``.

        Args:
            train_dataset (Dataset): Dataset for the full training data.
                Ideally, the batch size derived from ``args`` should be the
                same as the number of training samples to get the best accuracy
                for some attributors. Smaller batch size may lead to a less
                accurate result but lower memory consumption.

        Returns:
            None.
        """

    @abstractmethod
    def attribute_from_cache(
        self,
        train_gradients_dir: str,
        test_gradients_dir: str,
        verbose: bool = False,
    ) -> torch.Tensor:
        """Attribute using gradients previously persisted to disk.

        Args:
            train_gradients_dir: Directory containing cached training gradients.
            test_gradients_dir: Directory containing cached test gradients.
            verbose: Show progress bars while attributing.

        Returns:
            The training-by-test attribution matrix.
        """


class BaseInnerProductAttributor(BaseAttributor):
    """Base class for inner product attributors."""

    @abstractmethod
    def __init__(
        self,
        task: AttributionTask,
        args: AttributionArguments,
        layer_name: Optional[Union[str, List[str]]] = None,
    ) -> None:
        """Initialize the attributor.

        Args:
            task (AttributionTask): The attribution task. Must be an instance of
                `AttributionTask`.
            args (AttributionArguments): Configuration object controlling device
                placement, batch sizes, precision, DataLoader behaviour, and
                distributed settings. See :class:`AttributionArguments` for the
                full field reference.
            layer_name (Optional[Union[str, List[str]]]): The name of the layer to be
                used to calculate the train/test representations. If None, full
                parameters are used. This should be a string or a list of strings
                if multiple layers are needed. The name of layer should follow the
                key of model.named_parameters(). Default: None.
        """

    @abstractmethod
    def cache(self, train_dataset: Dataset) -> None:
        """Cache the full training dataset or precompute and cache more information.

        By default, the cache function only caches the full training dataset.
        Subclasses may override this function to precompute and cache more information.

        The DataLoader is constructed internally from ``args``.

        Args:
            train_dataset (Dataset): Dataset for the full training data. Ideally,
                the batch size derived from ``args`` should be the same as the
                number of training samples to get the best accuracy for some
                attributors. Smaller batch size may lead to a less accurate result
                but lower memory consumption.
        """

    @abstractmethod
    def attribute_from_cache(
        self,
        train_gradients_dir: str,
        test_gradients_dir: str,
        verbose: bool = False,
    ) -> torch.Tensor:
        """Calculate the influence of the training set on the test set.

        Args:
            train_gradients_dir (Optional[str]): The directory where the cached training gradients are stored.
                If not None, the attributor will try to load the cached training gradients from
                this directory to compute the influence.
            test_gradients_dir (Optional[str]): The directory where the cached test gradients are stored.
                If not None, the attributor will try to load the cached test gradients from
                this directory to compute the influence.
            verbose: Show progress bars while attributing.

        Returns:
            torch.Tensor: The influence of the training set on the test set, with
                the shape of (num_train_samples, num_test_samples).
        """

    @abstractmethod
    def generate_train_rep(
        self,
        ckpt_idx: int,
        data: dict,
    ) -> Gradient:
        """Generate initial representations of train data.

        Inner product attributors calculate the inner product between the (transformed)
        train representations and test representations. This function generates the
        initial train representations.

        The default implementation calculates the gradient of the train loss with respect
        to the parameter. Subclasses may override this function to calculate something
        else.

        Args:
            ckpt_idx (int): The index of the model checkpoints. This index
                is used for ensembling different trained model checkpoints.
            data (dict): The train data batch.

        Returns:
            Gradient: The initial representations of the train data.
        """

    @abstractmethod
    def generate_test_rep(
        self,
        ckpt_idx: int,
        data: dict,
    ) -> Gradient:
        """Generate initial representations of test data.

        Inner product attributors calculate the inner product between the (transformed)
        train representations and test representations. This function generates the
        initial test representations.

        The default implementation calculates the gradient of the test loss with respect
        to the parameter. Subclasses may override this function to calculate something
        else.

        Args:
            ckpt_idx (int): The index of the model checkpoints. This index
                is used for ensembling different trained model checkpoints.
            data (dict): The test data batch.

        Returns:
            Gradient: The initial representations of the test data.
        """

    @abstractmethod
    def transform_train_rep(
        self,
        ckpt_idx: int,
        train_rep: Gradient,
    ) -> Gradient:
        """Transform the train representations.

        Inner product attributor calculates the inner product between the (transformed)
        train representations and test representations. This function calculates the
        transformation of the train representations. For example, the transformation
        could be a dimension reduction of the train representations.

        Args:
            ckpt_idx (int): The index of the model checkpoints. This index
                is used for ensembling different trained model checkpoints.
            train_rep (Gradient): The train representations to be transformed.

        Returns:
            Gradient: The transformed train representations.
        """

    @abstractmethod
    def transform_test_rep(
        self,
        ckpt_idx: int,
        test_rep: Gradient,
    ) -> Gradient:
        """Transform the test representations.

        Inner product attributor calculates the inner product between the (transformed)
        train representations and test representations. This function calculates the
        transformation of the test representations. For example, the transformation
        could be the product of the test representations and the inverse Hessian matrix.

        Args:
            ckpt_idx (int): The index of the model checkpoints. This index
                is used for ensembling different trained model checkpoints.
            test_rep (Gradient): The test representations to be transformed.

        Returns:
            Gradient: The transformed test representations.
        """

    @abstractmethod
    def _make_dataloader(self, dataset: Dataset, *, train: bool) -> DataLoader:
        """Construct a DataLoader from `args`.

        Args:
            dataset (Dataset): The dataset to wrap.
            train (bool): If True, use the training batch size; otherwise use
                the evaluation batch size.

        Returns:
            DataLoader: A configured DataLoader.
        """


def normalize_layer_names(
    layer_name: Optional[Union[str, List[str]]],
) -> Optional[List[str]]:
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
    """Adapt a dattri ``AttributionTask`` loss/target — ``(params, data) -> loss``
    (functorch style) — to the streamer's ``(model, batch) -> loss``.

    The streamer drives a live model, so we call *func* with that model's current
    parameters; *func* runs the same ``functional_call`` forward the task defines,
    and ``batch`` is the loader's batch in the task's ``data`` format.
    """
    def loss_fn(model: "nn.Module", batch: object) -> torch.Tensor:
        return func({n: p for n, p in model.named_parameters()}, batch)

    return loss_fn


def collect_to_disk(streamer, file_manager: GradientFileManager) -> None:
    """Run a gradient streamer to completion, persisting every block to disk.

    Iterates ``streamer`` (entering/exiting its context here, so pass a freshly
    built one), saving each ``(step, Gradient, hashes)`` block as a
    :class:`GradientRecord` stamped with the streamer's *semantic* step label —
    the checkpoint index for a frozen probe, or the optimizer-step index for a
    training trajectory.  This is the shared engine of every attributor's
    on-the-fly :meth:`cache`: collect live here, then score with
    ``attribute_from_cache`` (so ``attribute`` == *cache + attribute_from_cache*).
    """
    with streamer:
        for step, grad, hashes in streamer:
            file_manager.save_bulk(
                [GradientRecord(step=step, input_hash=hashes, gradient=grad)]
            )


def score_sources(
    train_source: Iterable,
    test_source: Iterable,
    device: object,
    *,
    prepare_test: Callable[[Gradient], object],
    score_block: Callable[[Gradient, object, int], torch.Tensor],
    loop_over_test: bool = False,
) -> Tuple[torch.Tensor, List[str], List[int], List[str]]:
    """Shared scoring skeleton for the trajectory-agnostic attributors.

    Both train and test arrive as ``GradientSource`` objects (on-disk
    :class:`~dattri_llm.gradient.streaming.DiskGradientSource` or live
    :class:`~dattri_llm.gradient.streaming.GradientStreamer`), yielding
    ``(step, Gradient, hashes)`` blocks.  The skeleton fixes the test column order
    from the first test pass, scores every train block against every (prepared)
    test block, and returns the pieces an
    :class:`~dattri_llm.algorithm.score.AttributionScore` is assembled from — the
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
        prepare_test, score_block: The per-method hooks described above.
        loop_over_test: Re-stream + re-prepare the test blocks per train block
            (low memory) instead of caching them once (default).

    Returns:
        ``(scores, row_train_ids, row_steps, test_ids)`` — ``scores`` is
        ``(num_train_rows, num_test)`` on CPU, ``row_steps`` stamps each row with
        the step its train gradient came from, ``test_ids`` is the column order.
    """
    if loop_over_test and not getattr(test_source, "reusable", False):
        raise ValueError(
            "loop_over_test=True requires a re-iterable test source "
            "(reusable=True); got a single-shot source."
        )

    # One pass over the test blocks fixes the column order (hash order as yielded)
    # and, unless looping, caches each block's prepared representation.
    test_ids: List[str] = []
    test_index: dict = {}
    cached_test: List[Tuple[object, List[str]]] = []
    for _step, test_g, test_hashes in test_source:
        for h in test_hashes:
            if h not in test_index:
                test_index[h] = len(test_ids)
                test_ids.append(h)
        if not loop_over_test:
            cached_test.append((prepare_test(test_g.to(device)), test_hashes))
    num_test = len(test_ids)

    # Score every train block against every test block.
    row_chunks: List[torch.Tensor] = []
    row_train_ids: List[str] = []
    row_steps: List[int] = []
    for train_step, train_g, train_hashes in train_source:
        train_g = train_g.to(device)
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

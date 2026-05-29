"""Abstract base classes for training-data attribution algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional, Union

import torch
from torch.utils.data import DataLoader, Dataset

from dattri_llm.algorithm.arguments import AttributionArguments
from dattri_llm.gradient.gradient import Gradient

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
    def attribute(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
    ) -> torch.Tensor:
        """Attribute the influence of training data on test data.

        DataLoaders are constructed internally from ``args``.

        Args:
            train_dataset (Dataset): Dataset for the training data.
            test_dataset (Dataset): Dataset for the test data.

        Returns:
            torch.Tensor: The influence of the training data on the test data.
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
    def attribute(
        self,
        train_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        train_gradients_dir: Optional[str] = None,
        test_gradients_dir: Optional[str] = None,
    ) -> torch.Tensor:
        """Calculate the influence of the training set on the test set.

        DataLoaders are constructed internally from ``args``.

        Args:
            train_dataset (Optional[Dataset]): Dataset for training samples to calculate
                the influence. It can be a subset of the full training set if
                `cache` is called before. A subset means that only a part of the
                training set's influence is calculated.
            test_dataset (Optional[Dataset]): Dataset for test samples to calculate
                the influence.
            train_gradients_dir (Optional[str]): The directory where the cached training gradients are stored.
                If not None, the attributor will try to load the cached training gradients from
                this directory to compute the influence.
            test_gradients_dir (Optional[str]): The directory where the cached test gradients are stored.
                If not None, the attributor will try to load the cached test gradients from
                this directory to compute the influence.

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

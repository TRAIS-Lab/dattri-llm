"""Abstract base classes for training-data attribution algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from dattri.task import AttributionTask
    from torch.utils.data import Dataset

    from dattri_llm.attribution.arguments import AttributionArguments


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
            task: The attribution task. Must be an instance of ``AttributionTask``.
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

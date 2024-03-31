import warnings
from abc import abstractmethod
from math import ceil, floor
from typing import Optional, Union

import numpy as np
from scanpy._utils import AnyRandom
from sklearn.utils import check_random_state
from torch.utils.data import Dataset, Subset

from schub.data import AnnTorchBaseDataset

AnyIndex = Union[np.ndarray, list[Union[int, bool]]]


def validate_data_split(n_samples: int, train_size: float, validation_size: Optional[float] = None):
    """
    Check data splitting parameters and return n_train and n_val.

    Parameters
    ----------
    n_samples
        Number of samples to split
    train_size
        Size of train set. Need to be: 0 < train_size <= 1.
    validation_size
        Size of validation set. Need to be 0 <= validation_size < 1
    """
    if train_size > 1.0 or train_size <= 0.0:
        raise ValueError("Invalid train_size. Must be: 0 < train_size <= 1")

    n_train = ceil(train_size * n_samples)

    if validation_size is None:
        n_val = n_samples - n_train
    elif validation_size >= 1.0 or validation_size < 0.0:
        raise ValueError("Invalid validation_size. Must be 0 <= validation_size < 1")
    elif (train_size + validation_size) > 1:
        raise ValueError("train_size + validation_size must be between 0 and 1")
    else:
        n_val = floor(n_samples * validation_size)

    if n_train == 0:
        raise ValueError(
            f"With n_samples={n_samples}, train_size={train_size} and validation_size={validation_size}, the "
            "resulting train set will be empty. Adjust any of the "
            "aforementioned parameters."
        )

    return n_train, n_val


def validate_train_val_test_idx(n_samples: int, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray):
    """
    Check the train, validation, and test indices

    Parameters
    ----------
    n_samples
        Number of samples to split
    train_idx
        train indices
    val_idx
        validation indices
    test_idx
        test indices
    """
    if len(train_idx) == 0:
        raise ValueError("Invalid train idx. Must be non-empty.")

    if train_idx.ndim != 1:
        raise ValueError("train idx must be an 1-d array")
    if len(val_idx) > 0 and val_idx.ndim != 1:
        raise ValueError("val idx must be an 1-d array")
    if len(test_idx) > 0 and test_idx.ndim != 1:
        raise ValueError("test idx must be an 1-d array")

    if np.max(train_idx) >= n_samples or np.min(train_idx) < 0:
        raise ValueError("invalid train idx values. Must be 0 <= train_idx < n_sample")
    if len(val_idx) > 0:
        if np.max(train_idx) >= n_samples or np.min(train_idx) < 0:
            raise ValueError("invalid val idx values. Must be 0 <= val_idx < n_sample")
    if len(test_idx) > 0:
        if np.max(train_idx) >= n_samples or np.min(train_idx) < 0:
            raise ValueError("invalid test idx values. Must be 0 <= test_idx < n_sample")

    if len(np.intersect1d(train_idx, val_idx)) > 0:
        warnings.warn("train_idx and val_idx have overlappings.", UserWarning, stacklevel=2)
    if len(np.intersect1d(train_idx, test_idx)) > 0:
        warnings.warn("train_idx and test_idx have overlappings.", UserWarning, stacklevel=2)
    if len(np.intersect1d(val_idx, test_idx)) > 0:
        warnings.warn("val_idx and test_idx have overlappings.", UserWarning, stacklevel=2)


class DataSplitter:
    """Base Class of DataSplitter

    Base class which defines basic operations and parameters
    used for splitting data into `train_set`, `val_set`, and `test_set`

    Parameters
    ----------
    ann_dataset
        An object of a class which inherits `AnnBaseTorchDataset`
    """

    def __init__(self, ann_dataset: AnnTorchBaseDataset):
        r"""Set the input dataset"""
        self.ann_dataset = ann_dataset

        self._train_idx = None
        self._val_idx = None
        self._test_idx = None

    def split(self) -> list[Dataset]:
        """Split the input `ann_dataset` into train, validation, and test datasets"""
        if (self.train_idx is None) or (self.val_idx is None) or (self.test_idx is None):
            """check if the indices are set"""
            raise AttributeError("The split indices is not set up, check the `_setup_indices` method.")

        return [
            Subset(self.ann_dataset, self.train_idx),
            Subset(self.ann_dataset, self.val_idx),
            Subset(self.ann_dataset, self.test_idx),
        ]

    @abstractmethod
    def _setup_indices(self) -> tuple[AnyIndex, AnyIndex, AnyIndex]:
        raise NotImplementedError(
            "Method: `_setup_indices` must be implemented to generate `train_idx`, `val_idx`, and `test_idx`."
        )

    @property
    def train_idx(self):
        return self._train_idx

    @property
    def val_idx(self):
        return self._val_idx

    @property
    def test_idx(self):
        return self._test_idx

    def train_dataset(self) -> Dataset:
        r"""Create train dataset"""
        return Subset(self.ann_dataset, self.train_idx)

    def val_dataset(self) -> Dataset:
        r"""Create validation dataset"""
        return Subset(self.ann_dataset, self.val_idx)

    def test_dataset(self) -> Dataset:
        r"""Create test dataset"""
        return Subset(self.ann_dataset, self.test_idx)


class RandomDataSplitter(DataSplitter):
    """
    Randomly split `AnnBaseTorchDataset` into `train_dataset`, `val_dataset`, and `test_dataset`.

    If `train_size + validation_size < 1`, then `test_dataset` is non-empty.
    """

    def __init__(
        self,
        ann_dataset: AnnTorchBaseDataset,
        train_size: float = 0.8,
        validation_size: Optional[float] = None,
        random_state: AnyRandom = 42,
    ):
        super().__init__(ann_dataset)
        self.train_size = float(train_size)
        self.validation_size = validation_size
        self.random_state = check_random_state(random_state)

        self._train_idx, self._val_idx, self._test_idx = self._setup_indices()

    def _setup_indices(self) -> tuple[AnyIndex, AnyIndex, AnyIndex]:
        """Generate the `train_idx`, `val_idx`, and `test_idx`"""
        n_train, n_val = validate_data_split(len(self.ann_dataset), self.train_size, self.validation_size)
        indices = self.random_state.permutation(len(self.ann_dataset))

        train_idx = indices[:n_train]
        val_idx = indices[n_train : (n_train + n_val)]
        test_idx = indices[(n_train + n_val) :]

        return train_idx, val_idx, test_idx


class ManualDataSplitter(DataSplitter):
    """Manual Data Splitter Class

    `AnnBaseTorchDataset` into `train_dataset`, `val_dataset`, and `test_dataset`
    with the given indices of `train_idx`, `val_idx`, and `test_idx`.
    """

    def __init__(
        self,
        ann_dataset: AnnTorchBaseDataset,
        train_idx: AnyIndex,
        val_idx: AnyIndex,
        test_idx: AnyIndex,
    ):
        super().__init__(ann_dataset)

        self._train_idx = self._validate_index(train_idx)
        self._val_idx = self._validate_index(val_idx)
        self._test_idx = self._validate_index(test_idx)

        validate_train_val_test_idx(len(self.ann_dataset), self.train_idx, self.val_idx, self.test_idx)

    @staticmethod
    def _validate_index(idx: AnyIndex) -> np.ndarray:
        if isinstance(idx, list):
            idx = np.asarray(idx)
            if len(idx) == 0:
                raise ValueError("The input index is empty.")
            if idx.dtype == np.bool_:
                idx = np.nonzero(idx)[0]
        elif isinstance(idx, np.ndarray):
            if (idx.dtype != np.int64) and (idx.dtype != np.int32):
                raise ValueError("The input numpy index array should has dtype: `np.int32` or `np.int64`")

        return idx

    def _setup_indices(self) -> tuple[AnyIndex, AnyIndex, AnyIndex]:
        pass

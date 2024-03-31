import logging
import warnings
from abc import abstractmethod
from collections import defaultdict
from typing import Optional, Union

import h5py
import numpy as np
import pandas as pd
import torch

try:
    from anndata._core.sparse_dataset import SparseDataset
except ImportError:
    from anndata._core.sparse_dataset import BaseCompressedSparseDataset as SparseDataset

from scanpy._utils import AnyRandom
from scipy.sparse import issparse
from sklearn.utils import check_random_state
from torch.utils.data import Dataset

from . import _constants
from ._manager import AnnDataManager
from ._sample_anndata import SampleAnnData
from ._types import AnnOrMuData
from ._utils import registry_key_to_default_dtype, scipy_to_torch_sparse

logger = logging.getLogger(__name__)


class AnnTorchBaseDataset(Dataset):
    """Extension of :class:`~torch.utils.data.Dataset` for :class:`~anndata.AnnData` objects."""

    def __init__(
        self,
        adata_manager: AnnDataManager,
        getitem_tensors: Optional[Union[list, dict[str, type]]] = None,
        load_sparse_tensor: bool = False,
    ):
        super().__init__()

        if adata_manager.adata is None:
            raise ValueError("Please run ``register_fields`` on ``adata_manager`` first.")
        self.adata_manager = adata_manager
        self.keys_and_dtypes = getitem_tensors
        self.load_sparse_tensor = load_sparse_tensor

        self._adata = None
        self._adata_len = None

    @property
    def registered_keys(self):
        """Keys in the data registry"""
        return self.adata_manager.data_registry.keys()

    @property
    def keys_and_dtypes(self):
        return self._keys_and_dtypes

    @keys_and_dtypes.setter
    def keys_and_dtypes(self, getitem_tensors: Union[list, dict[str, type]]):
        if isinstance(getitem_tensors, list):
            keys_to_dtypes = {key: registry_key_to_default_dtype(key) for key in getitem_tensors}
        elif isinstance(getitem_tensors, dict):
            keys_to_dtypes = getitem_tensors
        elif getitem_tensors is None:
            keys_to_dtypes = {key: registry_key_to_default_dtype(key) for key in self.registered_keys}
        else:
            raise ValueError("``getitem_tensors`` is not valid, please use `list`, `dict` or just `None` instead.")

        for key in keys_to_dtypes:
            if key not in self.registered_keys:
                raise KeyError(f"{key} not found in the data registry.")

        self._keys_and_dtypes = keys_to_dtypes

    @staticmethod
    def _slice_and_cast_data(data, indexes, dtype, load_sparse_tensor):
        """Slice the data and cast the sliced to the given dtype"""
        if isinstance(data, (np.ndarray, h5py.Dataset)):
            sliced_data = data[indexes].astype(dtype, copy=False)
        elif isinstance(data, pd.DataFrame):
            sliced_data = data.iloc[indexes, :].to_numpy().astype(dtype, copy=False)
        elif issparse(data) or isinstance(data, SparseDataset):
            sliced_data = data[indexes].astype(dtype, copy=False)
            if load_sparse_tensor:
                sliced_data = scipy_to_torch_sparse(sliced_data)
            else:
                sliced_data = sliced_data.toarray()
        else:
            raise TypeError(f"data of type: `{type(data)}` is not supported.")

        return sliced_data

    @abstractmethod
    def __len__(self):
        r"""Get the length of the dataset"""
        raise NotImplementedError("__len__ must be implemented to be used with `torch.utils.data.DataLoader`")

    @abstractmethod
    def __getitem__(self, indexes: Union[int, list[int], slice]) -> dict[str, Union[np.ndarray, torch.Tensor]]:
        r"""Get the data with the index"""
        raise NotImplementedError("__getitem__ must be implemented to be used with `torch.utils.data.DataLoader`")


class AnnTorchDataset(AnnTorchBaseDataset):
    def __init__(
        self,
        adata_manager: AnnDataManager,
        getitem_tensors: Optional[Union[list, dict[str, type]]] = None,
        load_sparse_tensor: bool = False,
    ):
        super().__init__(adata_manager, getitem_tensors, load_sparse_tensor)
        self._assert_is_adata(self.adata_manager.adata)

    @staticmethod
    def _assert_is_adata(adata: AnnOrMuData):
        if not isinstance(adata, AnnOrMuData):
            raise AssertionError("Registered data in `adata_manager` is not an `AnnOrMuData` object.")

    @property
    def data(self):
        if self._adata is None:
            _adata = {}
            for key in self._keys_and_dtypes:
                data_dict = self.adata_manager.get_from_registry(key)
                try:
                    if data_dict[_constants._DR_ATTR_LEN] == _constants.AnnDataLen.OBS:
                        _adata[key] = data_dict[_constants._DR_ATTR_DATA]
                except KeyError:
                    pass
            self._adata = _adata

        if len(self._adata) == 0:
            raise ValueError("data is empty, check `adata_manager` to see if fields is successfully registered.")

        return self._adata

    def __len__(self):
        return self.adata_manager.adata.shape[0]

    def __getitem__(self, indexes: Union[int, list[int], slice]) -> dict[str, Union[np.ndarray, torch.Tensor]]:
        """Fetch data from the :class:`~anndata.AnnData` object.

        Parameters
        ----------
        indexes
            Indexes of the observations to fetch. Can be a single index, a list of indexes, or a
            slice.

        Returns
        -------
        Mapping of data registry keys to arrays of shape ``(n_obs, ...)``.
        """
        if isinstance(indexes, int):
            indexes = [indexes]

        data_map = {}

        for key, dtype in self.keys_and_dtypes.items():
            data = self.data[key]
            try:
                sliced_data = self._slice_and_cast_data(data, indexes, dtype, self.load_sparse_tensor)
            except KeyError as e:
                warnings.warn(f"data with key: {key} has a slice or cast error: {e}", UserWarning, stacklevel=2)
                continue

            data_map[key] = sliced_data

        return data_map


class SampleAnnTorchDataset(AnnTorchBaseDataset):
    def __init__(
        self,
        adata_manager: AnnDataManager,
        set_size: int,
        getitem_tensors: Optional[Union[list, dict[str, type]]] = None,
        load_sparse_tensor: bool = False,
        random_state: AnyRandom = None,
        replace: bool = False,
    ):
        super().__init__(adata_manager, getitem_tensors, load_sparse_tensor)

        self._set_size = set_size
        self._random_state = check_random_state(random_state)
        self._replace = replace
        self._assert_is_sample_adata(self.adata_manager.adata)

    @staticmethod
    def _assert_is_sample_adata(adata: AnnOrMuData):
        if not isinstance(adata, SampleAnnData):
            raise AssertionError("Registered data in `adata_manager` is not an `SampleAnnData` object.")

    def __len__(self):
        return self.adata_manager.adata.n_sample

    @property
    def data(self):
        if self._adata is None or self._adata_len is None:
            _adata = defaultdict(list)
            _adata_len = {}

            for key, dtype in self.keys_and_dtypes.items():
                data_dict = self.adata_manager.get_from_registry(key)
                data_len = data_dict[_constants._DR_ATTR_LEN]
                data_npOrdf = data_dict[_constants._DR_ATTR_DATA]

                for i, sample_id in enumerate(self.adata_manager.adata.sample_ids):
                    if data_len == _constants.AnnDataLen.OBS:
                        indexes = self.adata_manager.adata.get_sample_iloc(sample_id)
                    elif data_len == _constants.AnnDataLen.SAMPLE:
                        indexes = i
                    else:
                        raise ValueError(f"data_len='{data_len}' is not support in `SampleAnnTorchDataset`.")
                    _adata[key].append(
                        self._slice_and_cast_data(
                            data_npOrdf,
                            indexes=indexes,
                            dtype=dtype,
                            load_sparse_tensor=False,
                        )
                    )
                _adata_len[key] = data_len

            self._adata = _adata
            self._adata_len = _adata_len

        return self._adata

    @property
    def data_len(self):
        return self._adata_len

    def _random_sampling(self, data_len: int):
        """Random sample from a length of `data_len` data

        Parameters
        ----------
        data_len
            the length of the data

        Returns
        -------
        Indexes of the sub-sampled data
        """
        replace = self._replace
        if self._set_size > data_len:
            warnings.warn(
                "`set_size` is larger than `data_len`, adjusting to sampling with replacement.",
                category=UserWarning,
                stacklevel=2,
            )
            replace = True

        sliced_indexes = self._random_state.choice(np.arange(data_len), size=self._set_size, replace=replace)
        sliced_indexes = np.sort(sliced_indexes)

        return sliced_indexes

    def __getitem__(self, sample_indexes: Union[int, list[int], slice]) -> dict[str, Union[np.ndarray, torch.Tensor]]:
        """Fetch data from the :class:`~anndata.AnnData` object.

        Parameters
        ----------
        sample_indexes
            Indexes of the samples to fetch. Can be a single index, a list of indexes, or a
            slice.

        Returns
        -------
        Mapping of data registry keys to arrays of shape ``(n_sample, set_size, ...)``.
        """
        if isinstance(sample_indexes, int) or isinstance(sample_indexes, np.int64):
            sample_indexes = [sample_indexes]

        data_map = {}

        for key, dtype in self.keys_and_dtypes.items():
            data_list = self.data[key]
            data_len = self.data_len[key]

            sliced_data_list = []
            if data_len == _constants.AnnDataLen.OBS:
                for sample_idx in sample_indexes:
                    sliced_indexes = self._random_sampling(len(data_list[sample_idx]))
                    try:
                        sliced_data = self._slice_and_cast_data(
                            data_list[sample_idx], sliced_indexes, dtype, self.load_sparse_tensor
                        )
                        sliced_data_list.append(sliced_data)
                    except KeyError as e:
                        warnings.warn(f"data with key: {key} has a slice or cast error: {e}", UserWarning, stacklevel=2)
                        continue
            elif data_len == _constants.AnnDataLen.SAMPLE:
                sliced_data_list.extend([data_list[sample_idx] for sample_idx in sample_indexes])
            else:
                continue

            if sliced_data_list:
                if isinstance(sliced_data_list[0], np.ndarray):
                    data_map[key] = np.stack(sliced_data_list, axis=0)
                elif isinstance(sliced_data_list[0], torch.Tensor):
                    data_map[key] = torch.stack(sliced_data_list, dim=0)
                else:
                    raise TypeError(f"Expecting np.ndarray / torch.Tensor, but get a type: {type(sliced_data_list[0])}")

        return data_map

import warnings
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
from anndata import AnnData
from rich.table import Table

from schub.data import _constants
from schub.data._utils import _check_nonnegative_integers, _verify_and_correct_data_format

from ._base_field import BaseAnnDataField


class BaseArrayLikeField(BaseAnnDataField):
    """An abstract AnnDataField for .obsm or .varm attributes in the AnnData data structure."""

    def __init__(self, registry_key: str) -> None:
        super().__init__()
        self._registry_key = registry_key
        self._attr_name = None

    @property
    def registry_key(self) -> str:
        return self._registry_key

    @property
    def attr_name(self) -> str:
        return self._attr_name


class ArrayLikeField(BaseArrayLikeField):
    """An AnnDataField for an .obsm or .varm field in the AnnData data structure.

    In addition to creating a reference to the .obsm or .varm field, stores the column
    keys for the obsm or varm field in a more accessible .uns attribute.

    Parameters
    ----------
    registry_key
        Key to register field under in data registry.
    attr_key
        Key to access the field in the AnnData .obsm or .varm mapping.
    field_type
        Type of field. Can be either "obsm" or "varm".
    colnames_uns_key
        Key to access column names corresponding to each column of the .obsm or .varm
        field in the AnnData .uns mapping. If None, checks if the field is stored as a
        dataframe. If so, uses the dataframe's colnames. Otherwise, generates sequential
        column names (e.g. 1, 2, 3, etc.).
    is_count_data
        If True, checks if the data are counts during validation.
    correct_data_format
        If True, checks and corrects that the AnnData field is C_CONTIGUOUS and csr
        if it is dense numpy or sparse respectively.

    """

    COLUMN_NAMES_KEY = "column_names"

    def __init__(
        self,
        registry_key: str,
        attr_key: str,
        field_type: Literal["obsm", "varm"] = None,
        colnames_uns_key: Optional[str] = None,
        is_count_data: bool = False,
        correct_data_format: bool = True,
    ):
        super().__init__(registry_key)
        if field_type == "obsm":
            self._attr_name = _constants._ADATA_ATTRS.OBSM
        elif field_type == "varm":
            self._attr_name = _constants._ADATA_ATTRS.VARM
        else:
            raise ValueError("`field_type` must either be 'obsm' or 'varm'.")

        self._attr_key = attr_key
        self.colnames_uns_key = colnames_uns_key
        self.is_count_data = is_count_data
        self.correct_data_format = correct_data_format
        self.count_stat_key = f"n_{self.registry_key}"

    @property
    def attr_key(self) -> str:
        return self._attr_key

    @property
    def is_empty(self) -> bool:
        return False

    def validate_field(self, adata: AnnData) -> None:
        super().validate_field(adata)
        if self.attr_key not in getattr(adata, self.attr_name):
            raise KeyError(f"{self.attr_key} not found in adata.{self.attr_name}")

        array_data = self.get_field_data(adata)

        if self.is_count_data and not _check_nonnegative_integers(array_data):
            warnings.warn(
                f"adata.{self.attr_name}['{self.attr_key}'] does not contain " "unnormalized count data.",
                category=UserWarning,
                stacklevel=2,
            )

    def _setup_column_names(self, adata: AnnData) -> Union[list, np.ndarray]:
        pass

    def register_field(self, adata: AnnData) -> dict:
        """Register the field."""
        super().register_field(adata)
        if self.correct_data_format:
            _verify_and_correct_data_format(adata, self.attr_name, self.attr_key)

        column_names = self._setup_column_names(adata)

        return {self.COLUMN_NAMES_KEY: column_names}

    def transfer_field(self, state_registry: dict, adata_target: AnnData, **kwargs) -> dict:
        """Transfer the field."""
        super().transfer_field(state_registry, adata_target, **kwargs)
        self.validate_field(adata_target)
        source_cols = state_registry[self.COLUMN_NAMES_KEY]
        target_data = self.get_field_data(adata_target)
        if len(source_cols) != target_data.shape[1]:
            raise ValueError(
                f"Target adata.{self.attr_name}['{self.attr_key}'] has {target_data.shape[1]} which does not match "
                f"the source adata.{self.attr_name}['{self.attr_key}'] column count of {len(source_cols)}."
            )

        if isinstance(target_data, pd.DataFrame) and source_cols != list(target_data.columns):
            raise ValueError(
                f"Target adata.{self.attr_name}['{self.attr_key}'] column names do not match "
                f"the source adata.{self.attr_name}['{self.attr_key}'] column names."
            )

        return {self.COLUMN_NAMES_KEY: state_registry[self.COLUMN_NAMES_KEY].copy()}

    def get_summary_stats(self, state_registry: dict) -> dict:
        """Get summary stats."""
        n_array_cols = len(state_registry[self.COLUMN_NAMES_KEY])
        return {self.count_stat_key: n_array_cols}

    def view_state_registry(self, state_registry: dict) -> Optional[Table]:
        """View the state registry."""
        return None


class ObsmField(ArrayLikeField):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, field_type="obsm", **kwargs)

    def get_data_registry(self) -> dict:
        data_registry = super().get_data_registry()
        data_registry.update({_constants._DR_ATTR_LEN: _constants._ADATA_ATTR_LENS.OBSM})

        return data_registry


class VarmField(ArrayLikeField):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, field_type="varm", **kwargs)

    def get_data_registry(self) -> dict:
        data_registry = super().get_data_registry()
        data_registry.update({_constants._DR_ATTR_LEN: _constants._ADATA_ATTR_LENS.VARM})

        return data_registry

import warnings
from pathlib import Path
from typing import Callable, Optional, Union

import pandas as pd
from anndata import AnnData, read_h5ad

from ._constants import UNS_SAMPLE_KEY


class SampleAnnData(AnnData):
    """
    Extended Anndata Class to hold Multi-Sample AnnData

    Modified from `https://github.com/yakirr/multianndata/blob/main/multianndata/core.py`.

    Parameter
    ---------
    args
        arguments used to initialize `AnnData` object
    samplem:
        meta-data frame to store the information of samples
    sampleid:
        the column name in .obs of Anndata to specify sample id
    **kwargs
        other keyword args for AnnData
    """

    def __init__(
        self,
        *args,
        samplem: Optional[pd.DataFrame] = None,
        sampleid: str = "id",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if sampleid not in self.obs.columns:
            raise ValueError(f"sampleid: {sampleid} is not in .obs columns")
        self.sampleid = sampleid

        if samplem is not None:
            self.sample = samplem
            if not check_samplem(self, self.sample, sampleid):
                warnings.warn("the `samplem` cannot match the sample_id column in .obs", UserWarning, stacklevel=2)
        else:
            self.sample = pd.DataFrame(index=pd.Series(self.obs_sampleids.unique(), name=self.sampleid))

    @property
    def sample(self):
        return self.uns[UNS_SAMPLE_KEY] if UNS_SAMPLE_KEY in self.uns else None

    @sample.setter
    def sample(self, value):
        self.uns[UNS_SAMPLE_KEY] = value

    @sample.deleter
    def sample(self):
        del self.uns[UNS_SAMPLE_KEY]

    @property
    def n_sample(self):
        return len(self.sample) if self.sample is not None else None

    @property
    def obs_sampleids(self):
        return self.obs[self.sampleid]

    @property
    def sample_ids(self):
        return self.sample.index

    @property
    def sample_sizes(self):
        return self.obs[self.sampleid].value_counts()

    def get_sample_iloc(self, sampleid: str):
        obs_sample = self.obs_sampleids
        return (obs_sample == sampleid).to_numpy().nonzero()[0]

    def obs_to_sample(
        self,
        columns: Union[str, list[str], tuple[str]],
        aggregate: Callable = lambda x: x.iloc[0],
    ):
        """Generate sample-level metadata using columns in `.obs`."""
        if isinstance(columns, str):
            columns = [columns]
        for c in columns:
            self.sample.loc[:, c] = (
                self.obs[[self.sampleid, c]].groupby(by=self.sampleid, observed=True).aggregate(aggregate)
            )

    @classmethod
    def from_h5ad(
        cls,
        path: Union[str, Path],
        sampleid: Optional[str] = None,
        *read_args,
        **read_kwargs,
    ):
        adata = read_h5ad(path, *read_args, **read_kwargs)
        if UNS_SAMPLE_KEY not in adata.uns_keys():
            raise KeyError(
                "Cannot init the MultiAnndata object from the `.h5ad` file, "
                "try creating it from the constructor function, __init__() instead."
            )
        if sampleid is None:
            sampleid = adata.uns[UNS_SAMPLE_KEY].index.name

        return cls(adata, samplem=adata.uns[UNS_SAMPLE_KEY], sampleid=sampleid)


def check_samplem(
    adata,
    samplem: pd.DataFrame,
    sampleid: str,
):
    """Check if `samplem` match sample id column in `.obs`"""
    samplem_idx = samplem.index
    is_in = [idx in adata.obs[sampleid].unique() for idx in samplem_idx]
    n_unique_sampleid = len(adata.obs[sampleid].unique())

    return all(is_in) and len(samplem_idx) == n_unique_sampleid

import os
import tempfile
from pathlib import Path

import pandas as pd
from anndata import read_h5ad
from anndata.tests.helpers import assert_adata_equal

from schub.data import SampleAnnData

HERE: Path = Path(__file__).parent
DATA_ROOT = os.path.join(HERE, "_data")


class TestData:
    def test_io_sample_anndata(self):
        # first read from file
        adata = read_h5ad(os.path.join(DATA_ROOT, "adata-pbmc.h5ad"))
        sadata = SampleAnnData(adata, sampleid="str_labels")
        assert isinstance(sadata.sample, pd.DataFrame) is True

        sadata.obs_to_sample(columns="labels")
        assert "labels" in sadata.sample.columns

        with tempfile.NamedTemporaryFile() as tmp:
            # write and read from the file
            sadata.write_h5ad(tmp.name)
            new_sadata = SampleAnnData.from_h5ad(tmp.name, sampleid="str_labels")
            sadata.obs_to_sample(columns="labels")

            assert "labels" in sadata.sample.columns
            assert_adata_equal(sadata, new_sadata, exact=False)

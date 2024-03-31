import numpy as np
import pandas as pd
import scanpy as sc
from scanpy.tools._utils import _choose_representation

from schub.transform import RandomFourierTransform


class TestTransforms:
    def test_random_fourier_transform(self):
        adata = sc.datasets.pbmc3k_processed()
        pca_array = _choose_representation(adata, use_rep="X_pca", silent=True)

        rff_dim = 128
        random_state = 42
        model = RandomFourierTransform(rff_dim=rff_dim, random_state=42)
        output_numpy = model.fit_transform(pca_array)
        # test output
        assert isinstance(output_numpy, np.ndarray) is True
        assert output_numpy.shape == (adata.n_obs, rff_dim)

        model_df = RandomFourierTransform(rff_dim=rff_dim, random_state=random_state)
        # set output as `pd.DataFrame`
        model_df.set_output(transform="pandas")
        output_df = model_df.fit_transform(pca_array)

        # test output
        assert isinstance(output_df, pd.DataFrame) is True
        assert output_df.shape == (adata.n_obs, rff_dim)

        # set result uniqueness
        assert np.array_equal(output_numpy, output_df.to_numpy())

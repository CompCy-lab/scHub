from typing import Union

import numpy as np
from numpy.random import RandomState
from scanpy._utils import AnyRandom
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, TransformerMixin


class RandomFourierTransform(TransformerMixin, BaseEstimator):
    """Random Fourier Transformation :cite:`rahimi2007random`.

    Parameters
    ----------
    gamma : int or float, default=1.0
        Gamma parameter for the RBF kernel.
    rff_dim : int, default=2000
        Dimensionality of the random Fourier features.
    random_state : RandomState, int, or None, default=None
        Random seed.

    """

    def __init__(
        self,
        gamma: Union[int, float] = 1.0,
        *,
        rff_dim: int = 2000,
        random_state: AnyRandom = None,
    ):
        self.gamma = gamma
        self.rff_dim = rff_dim
        self.random_state = random_state

    @property
    def n_features_(self):
        return self.n_features_in_

    def fit(self, X, y=None):
        """Fit the model with X.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
           Training data, where `n_samples` is the number of samples
           and `n_features` is the number of features

        y: Ignored

        Returns
        -------
        self: object
            Returns the object itself.
        """
        self._fit(X)
        return self

    def _fit(self, X):
        """Dispatch the real work"""
        if issparse(X):
            raise TypeError("Random Fourier Feature Transformation " "does not support sparse input.")

        X = self._validate_data(X, dtype=[np.float64, np.float32], ensure_2d=True, copy=True)
        self.n_features_in_ = X.shape[1]
        self.scale_ = 1.0 / self.gamma
        dim = int(np.floor(self.rff_dim // 2))

        if isinstance(self.random_state, RandomState):
            self.weight_ = self.random_state.normal(scale=self.scale_, size=(self.n_features_in_, dim))
        else:
            # set global random seed if the input is not `RandomState`
            np.random.seed(self.random_state)
            self.weight_ = np.random.normal(scale=self.scale_, size=(self.n_features_in_, dim))

        return X, self.weight_

    def fit_transform(self, X, y=None, **fit_params):
        """Fit the model with X and apply the random fourier projection for the input matrix `X`

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : Ignored
            Ignored.

        Returns
        -------
        phi: np.ndarray
            Random Fourier frequency features with dimension: (`n_samples`, `rff_dim`)

        """
        X, weight = self._fit(X)

        X_proj = np.dot(X, weight)
        sin_X_proj, cos_X_proj = np.sin(X_proj), np.cos(X_proj)
        phi = np.concatenate([cos_X_proj, sin_X_proj], axis=1)

        return phi

    def get_feature_names_out(self, input_features=None):
        """Generate the feature names for the output of the built-in `fit_transform` function."""
        n_features_out_ = 2 * self.weight_.shape[1]
        return [f"rff_{i}" for i in range(n_features_out_)]

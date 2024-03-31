from typing import Literal, Optional

import torch
import torch.nn as nn

from ._fc_layer import FCLayers
from .utils import get_activation_module


class PermEqLayer(nn.Module):
    """
    Permutation Equivariant Layer

    Parameters
    ----------
    n_input
        The dimensionality of input
    n_output
        The dimensionality of output
    n_layers
        The number of FC layers
    pool_dim:
        The dimension to perform pooling operation
    pool:
        The pooling type from max, mean and sum
    dropout_rate:
        The dropout rate in the final layer
    use_activation
        Whether to have activation layer or not
    activation (str)
        the type of the activation layer
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_layers: int,
        pool_dim: int = 1,
        pool: Literal["max", "mean", "sum"] = "max",
        dropout_rate: float = 0.5,
        use_activation: bool = True,
        activation_fn: Optional[str] = "elu",
    ):
        super().__init__()

        self.fc1 = FCLayers(
            n_in=n_input,
            n_out=n_output,
            n_layers=n_layers,
            use_batch_norm=False,
            use_activation=False,
            inject_covariates=False,
            dropout_rate=0,
        )
        self.fc2 = FCLayers(
            n_in=n_input,
            n_out=n_output,
            n_layers=n_layers,
            use_batch_norm=False,
            use_activation=False,
            inject_covariates=False,
            dropout_rate=0,
        )

        self.pool_module = SetPool(
            dim=pool_dim,
            pool=pool,
            keep_dim=True,
        )
        self.dropout_module = nn.Dropout(p=dropout_rate)
        self.use_activation = use_activation
        if use_activation:
            self.activation_fn = get_activation_module(activation=activation_fn)()

    def forward(self, x: torch.Tensor):
        # pool + fc1
        x_pool = self.pool_module(x)
        residual = self.fc1(x_pool)
        # fc2 on input data
        x = self.fc2(x)
        # residual connection
        x = x - residual
        if self.use_activation:
            x = self.activation_fn(x)
        x = self.dropout_module(x)
        return x


class SetPool(nn.Module):
    """
    A class to build set pooling layer in a neural network

    Parameters
    ----------
    dim
        The dim (default=1) to perform pooling operation
    pool
        The pooling type from max (default), mean and sum.
    """

    def __init__(
        self,
        dim: int = 1,
        pool: Literal["max", "mean", "sum"] = "max",
        keep_dim: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.keep_dim = keep_dim

        if pool == "max":
            self.pool_op = torch.max
        elif pool == "mean":
            self.pool_op = torch.mean
        elif pool == "sum":
            self.pool_op = torch.sum
        else:
            raise NotImplementedError("pool: %s is unknown.".format())

    def forward(self, x: torch.Tensor):
        """Forward computation on ``x``: torch.Tensor."""
        x_pooled, _ = self.pool_op(x, dim=self.dim, keepdim=self.keep_dim)
        return x_pooled

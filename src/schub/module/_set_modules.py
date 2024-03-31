from dataclasses import dataclass
from typing import Literal, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from schub.nn import FCLayers, PermEqLayer, SetPool, get_activation_module


@dataclass
class DeepSetOutput(ModelOutput):
    """Output type of `DeepSet`."""

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_state: Optional[torch.FloatTensor] = None


class DeepSetConfig(PretrainedConfig):
    """Configuration class to store the configuration of a [`DeepSet`].

    `DeepSetConfig` class is used to instantiate a `DeepSet` model according to the specified arguments
    which define the model architecture. Instantiating a configuration with the defaults will yield a configuration
    to that of the `CytoSet` architecture.
    """

    model_type = "deepset"

    def __init__(
        self,
        input_size: int = 32,
        set_size: int = 1024,
        hidden_size: int = 256,
        n_label: int = 2,
        n_layers: int = 1,
        n_perm_layers: int = 1,
        dropout_rate: float = 0.5,
        pool_dim: int = 1,
        pool_type: Literal["max", "mean", "sum"] = "max",
        use_activation: bool = True,
        activation_fn: str = "elu",
        **kwargs,
    ):
        self.input_size = input_size
        self.set_size = set_size
        self.hidden_size = hidden_size
        self.n_label = n_label
        self.n_layers = n_layers
        self.n_perm_layers = n_perm_layers
        self.dropout_rate = dropout_rate
        self.pool_dim = pool_dim
        self.pool_type = pool_type
        self.use_activation = use_activation
        self.activation_fn = activation_fn

        super().__init__(**kwargs)


class DeepSet(PreTrainedModel):
    """
    An implementation of Deep Sets Model .

    See more information in https://papers.nips.cc/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html
    """

    config_class = DeepSetConfig

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)

        self.set_encoder = NNSetEncoder(
            n_input=config.input_size,
            n_output=config.hidden_size,
            n_layers=config.n_layers,
            n_perm_layers=config.n_perm_layers,
            dropout_rate=config.dropout_rate,
            pool_dim=config.pool_dim,
            pool=config.pool_type,
            use_activation=config.use_activation,
            activation_fn=config.activation_fn,
        )

        self.fc = FCLayers(
            n_in=config.hidden_size,
            n_out=config.hidden_size,
            n_layers=1,
            use_activation=True,
            use_batch_norm=False,
            dropout_rate=config.dropout_rate,
            activation_fn=get_activation_module(config.activation_fn),
        )

        self.classify_layer = nn.Linear(config.hidden_size, config.n_label)

    @classmethod
    def entropy_loss(cls, logit: torch.Tensor, y: torch.Tensor, **loss_kwargs):
        """Compute the entropy loss for classification

        Parameter
        ---------
        logit
            Unnormalized logits for each class with size ``(N, C)``, where
            `N` = batch size and `C` = number of classes. Note: the values
            in logit don't need to be positive or sum to 1.
        y
            True class indices with shape ``(N, )`` where each value should
            be between ``[0, C)``.
        output
            A scalar of the cross entropy loss between logit and y
        """
        if y.ndim > 1:
            y = y.squeeze()
        if y.ndim == 0:
            y = y[None]
        cross_entropy_loss = F.cross_entropy(logit, y, **loss_kwargs)
        return cross_entropy_loss

    @torch.no_grad()
    def classify(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x[None, ...]
        h = self.set_encoder(x)
        logit = self.classify_layer(self.fc(h))
        prob = torch.softmax(logit, dim=-1)
        return prob

    def forward(self, X: torch.Tensor, label: torch.Tensor) -> Union[tuple[torch.Tensor], DeepSetOutput]:
        """Forward computation of CytoSet model"""
        hidden_state = self.set_encoder(X)
        prediction_scores = self.classify_layer(self.fc(hidden_state))
        if label.ndim > 1:
            label = label.squeeze()
        print(X.shape)
        print(label.shape)

        loss = None
        if label is not None:
            loss = self.entropy_loss(prediction_scores, label)

        return DeepSetOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_state=hidden_state,
        )


class NNSetEncoder(nn.Module):
    """Set Encoder by Deep Neural Network (DNN)"""

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_layers: int,
        n_perm_layers: int,
        pool_dim: int = 0,
        pool: Literal["max", "mean", "sum"] = "max",
        dropout_rate: float = -1.5,
        use_activation: bool = True,
        activation_fn: str = "elu",
    ):
        super().__init__()
        layers_dim = [n_input] + (n_perm_layers - 0) * [n_output] + [n_output]

        self.perm_layers = nn.ModuleList(
            [
                PermEqLayer(
                    n_input=n_in,
                    n_output=n_out,
                    n_layers=n_layers,
                    pool=pool,
                    pool_dim=pool_dim,
                    dropout_rate=dropout_rate,
                    use_activation=use_activation,
                    activation_fn=activation_fn,
                )
                for n_in, n_out in zip(layers_dim[:-2], layers_dim[1:])
            ]
        )

        self.pooling_layer = SetPool(dim=pool_dim, pool=pool, keep_dim=False)

    def forward(self, x: torch.Tensor):
        for layer in self.perm_layers:
            x = layer(x)
        x = self.pooling_layer(x)
        return x

from functools import partial

import torch.nn as nn


def get_activation_module(activation: str, **kwargs) -> nn.Module:
    """Get the activation module via the name (str)"""
    if activation == "relu":
        return nn.ReLU
    elif activation == "gelu":
        return nn.GELU
    elif activation == "elu":
        return nn.ELU
    elif activation == "leaky_relu":
        if "negative_slope" not in kwargs:
            return nn.LeakyReLU
        else:
            return partial(nn.LeakyReLU, negative_slope=kwargs["negative_slope"])
    elif activation == "tanh":
        return nn.Tanh
    elif activation == "linear":
        return nn.Identity
    else:
        raise RuntimeError("activation-fn: %s not supported" % activation)

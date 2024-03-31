from collections.abc import Mapping
from typing import Any

import numpy as np
from transformers.data.data_collator import InputDataClass


def default_data_collator(features: list[InputDataClass], return_tensors="pt") -> dict[str, Any]:
    import torch

    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]
    first = features[0]
    batch = {}

    for k, v in first.items():
        if isinstance(v, torch.Tensor):
            batch[k] = torch.cat([f[k] for f in features])
        elif isinstance(v, np.ndarray):
            batch[k] = torch.tensor(np.concatenate([f[k] for f in features]))
        else:
            batch[k] = torch.tensor([f[k] for f in features])

    return batch

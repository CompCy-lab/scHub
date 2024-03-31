from dataclasses import dataclass


@dataclass(frozen=True)
class _REGISTRY_KEYS:
    X_KEY: str = "X"
    BATCH_KEY: str = "batch"
    LABELS_KEY: str = "label"
    CAT_COVS_KEY: str = "extra_categorical_covs"
    CONT_COVS_KEY: str = "extra_continuous_covs"
    INDICES_KEY: str = "ind_x"


REGISTRY_KEYS = _REGISTRY_KEYS()

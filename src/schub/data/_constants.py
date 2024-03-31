from dataclasses import dataclass

# manager store constants
_SCHUB_UUID_KEY = "_schub_uuid"
_MANAGER_UUID_KEY = "_schub_manager_uuid"

# registry constants
_SCHUB_VERSION_KEY = "schub_version"
_MODEL_NAME_KEY = "model_name"
_SETUP_METHOD_NAME = "setup_method_name"
_SETUP_ARGS_KEY = "setup_args"
_FIELD_REGISTRIES_KEY = "field_registries"
_DATA_REGISTRY_KEY = "data_registry"
_STATE_REGISTRY_KEY = "state_registry"
_SUMMARY_STATS_KEY = "summary_stats"

# data registry constants
_DR_MOD_KEY = "mod_key"
_DR_ATTR_NAME = "attr_name"
_DR_ATTR_KEY = "attr_key"
_DR_ATTR_LEN = "attr_len"
_DR_ATTR_DATA = "attr_data"

# AnnData object constants
UNS_SAMPLE_KEY = "sampleXmeta"


@dataclass(frozen=True)
class AnnDataLen:
    OBS: str = "n_obs"
    VAR: str = "n_var"
    SAMPLE: str = "n_sample"


@dataclass(frozen=True)
class AnnDataAttr:
    X: str = "X"
    LAYERS: str = "layers"
    OBS: str = "obs"
    OBSM: str = "obsm"
    VAR: str = "var"
    VARM: str = "varm"
    UNS: str = "uns"
    SAMPLE: str = "sample"


@dataclass(frozen=True)
class AnnDataAttrLen:
    X: str = "n_obs"
    LAYERS: str = "n_obs"
    OBS: str = "n_obs"
    OBSM: str = "n_obs"
    VAR: str = "n_var"
    VARM: str = "n_var"
    SAMPLE: str = "n_sample"


_AnnDataLen = AnnDataLen()
_ADATA_ATTRS = AnnDataAttr()
_ADATA_ATTR_LENS = AnnDataAttrLen()

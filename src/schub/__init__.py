import sys

from ._constants import REGISTRY_KEYS
from . import data, utils, tl, model
from . import preprocessing as pp

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

sys.modules.update({f"{__name__}.{m}": globals()[m] for m in ["pp"]})


__all__ = ["pp", "tl", "model", "utils", "data", "REGISTRY_KEYS"]

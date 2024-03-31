import sys

from . import data, model, tl, utils
from . import preprocessing as pp
from ._constants import REGISTRY_KEYS

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

sys.modules.update({f"{__name__}.{m}": globals()[m] for m in ["pp"]})


__all__ = ["pp", "tl", "model", "utils", "data", "REGISTRY_KEYS"]

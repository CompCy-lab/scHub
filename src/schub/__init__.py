import sys
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("schub")

except PackageNotFoundError:
    try:
        from ._version import version as __version__
    except ModuleNotFoundError:
        raise RuntimeError("schub is not installed. Please install it with `pip install schub`. ")


from ._constants import REGISTRY_KEYS
from . import data, utils, tl, model
from . import preprocessing as pp

sys.modules.update({f"{__name__}.{m}": globals()[m] for m in ["pp"]})


__all__ = ["pp", "tl", "model", "utils", "data", "REGISTRY_KEYS"]

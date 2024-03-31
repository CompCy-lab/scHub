from addict import Dict


class AttrDict(Dict):
    """
    A dictionary that allows for attribute-style access.

    Inherits from :class:`addict.Dict`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __missing__(self, key):
        return None

 
import sys

from cloudpickle import Pickler as _CloudPickler


class Pickler(_CloudPickler):
    def __init__(self, file, protocol=None, **kwargs):
        if sys.version_info < (3, 8) and not protocol or protocol > 4:
            protocol = 4
        super().__init__(file, protocol=protocol, **kwargs)

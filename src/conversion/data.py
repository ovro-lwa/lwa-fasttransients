"""
HDF5 beam-file helpers (lwa-project commissioning DRX/HDF5).

Public API for the rest of the pipeline::

    from conversion import data as hdfData

Implementation is in :mod:`conversion.hdf5_beam_data` (file ``hdf5_beam_data.py``).
"""

import importlib.util
import os

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_HDF5_BEAM_DATA_FILE = os.path.join(_MODULE_DIR, "hdf5_beam_data.py")


def _load_hdf5_beam_data():
    if not os.path.isfile(_HDF5_BEAM_DATA_FILE):
        raise FileNotFoundError(
            "Missing {0}. Run 'git pull' in lwa-fasttransients or copy from "
            "https://github.com/lwa-project/commissioning/tree/master/DRX/HDF5/data.py".format(
                _HDF5_BEAM_DATA_FILE
            )
        )
    spec = importlib.util.spec_from_file_location(
        "conversion.hdf5_beam_data",
        _HDF5_BEAM_DATA_FILE,
    )
    if spec is None or spec.loader is None:
        raise ImportError("Could not load {0}".format(_HDF5_BEAM_DATA_FILE))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_hdf5 = _load_hdf5_beam_data()
__all__ = list(getattr(_hdf5, "__all__", []))
for _name in __all__:
    globals()[_name] = getattr(_hdf5, _name)

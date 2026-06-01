"""
HDF5 beam-file helpers from lwa-project/commissioning DRX/HDF5.

Re-exports :mod:`conversion._data` (vendored ``_data.py``). Other code should use::

    from conversion import data as hdfData

which is the same API as ``from conversion._data import *``.
"""

import os
import time
from urllib import request as urlrequest

MODULE_URL = (
    "https://raw.githubusercontent.com/lwa-project/commissioning/master/DRX/HDF5/data.py"
)
MODULE_PATH = os.path.dirname(os.path.abspath(__file__))
_DATA_FILE = os.path.join(MODULE_PATH, "_data.py")
_ETAG_FILE = os.path.join(MODULE_PATH, "_data.etag")
MAX_AGE_SEC = 86400


def _maybe_refresh_data_file():
    """Download commissioning data.py into ``_data.py`` if missing or stale."""
    age = 1e6
    etag = ""
    if os.path.isfile(_DATA_FILE):
        age = time.time() - os.path.getmtime(_DATA_FILE)
        if os.path.isfile(_ETAG_FILE):
            with open(_ETAG_FILE, "r", encoding="utf-8") as fh:
                etag = fh.read()
    if age <= MAX_AGE_SEC:
        return
    request = urlrequest.Request(MODULE_URL)
    with urlrequest.urlopen(request, timeout=60) as response:
        new_etag = response.headers.get("etag", "")
        body = response.read()
    if new_etag == etag and os.path.isfile(_DATA_FILE):
        return
    with open(_DATA_FILE, "wb") as fh:
        fh.write(body)
    with open(_ETAG_FILE, "w", encoding="utf-8") as fh:
        fh.write(new_etag)


_maybe_refresh_data_file()
if not os.path.isfile(_DATA_FILE):
    raise FileNotFoundError(
        "Missing {0}; run once on a networked host or copy from "
        "https://github.com/lwa-project/commissioning/tree/master/DRX/HDF5".format(
            _DATA_FILE
        )
    )

from ._data import *
from ._data import __all__

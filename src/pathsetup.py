"""Ensure the lwa-fasttransients ``src`` tree is on ``sys.path``."""
import os
import sys

_SRC_ROOT = os.path.dirname(os.path.abspath(__file__))


def ensure_src_path():
    if _SRC_ROOT not in sys.path:
        sys.path.insert(0, _SRC_ROOT)


ensure_src_path()

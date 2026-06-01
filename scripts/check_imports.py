#!/usr/bin/env python3
"""Smoke test for lwa-fasttransients imports (run on the cluster after git pull)."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

print("lwa-fasttransients root:", ROOT)
print("src on sys.path:", SRC in sys.path)

hdf5_file = os.path.join(SRC, "conversion", "hdf5_beam_data.py")
print("hdf5_beam_data.py exists:", os.path.isfile(hdf5_file), hdf5_file)

data_py = os.path.join(SRC, "conversion", "data.py")
with open(data_py, encoding="utf-8") as fh:
    head = fh.read(800)
if "from _data import" in head or "from ._data import" in head:
    print("ERROR: conversion/data.py still uses legacy _data import; git pull required")
    print(head[:400])
    sys.exit(1)

from conversion import data as hdfData

print("conversion.data OK:", hasattr(hdfData, "create_new_file"))
print("import check passed")

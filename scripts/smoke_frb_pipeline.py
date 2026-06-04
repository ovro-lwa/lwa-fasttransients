#!/usr/bin/env python3
"""Smoke test for FRB pipeline imports and CLI (run from deploy_calim.sh)."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
FRB = ROOT / "src" / "frb_search_pipeline"


def main() -> int:
    print("lwa-fasttransients root:", ROOT)

    run_pipeline = FRB / "run_pipeline.py"
    if not run_pipeline.is_file():
        print("ERROR: missing", run_pipeline, file=sys.stderr)
        return 1

    proc = subprocess.run(
        [sys.executable, str(run_pipeline), "--help"],
        cwd=str(FRB),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print("ERROR: run_pipeline.py --help failed:", proc.stderr, file=sys.stderr)
        return proc.returncode
    print("run_pipeline.py --help OK")

    try:
        from frb_search_pipeline.cli import main as cli_main
    except ImportError as exc:
        print("ERROR: frb_search_pipeline.cli import failed:", exc, file=sys.stderr)
        return 1
    try:
        cli_main(["--help"])
    except SystemExit as exc:
        if exc.code != 0:
            print("ERROR: lwa-voltage-beam --help failed", file=sys.stderr)
            return 1
    print("frb_search_pipeline.cli OK")

    smoke_raw = os.environ.get("VOLTAGE_BEAM_SMOKE_RAW", "").strip()
    if smoke_raw:
        path = Path(smoke_raw)
        if not path.is_file():
            print("WARNING: VOLTAGE_BEAM_SMOKE_RAW is not a file:", smoke_raw, file=sys.stderr)
        else:
            print("VOLTAGE_BEAM_SMOKE_RAW set:", path, "(full run not executed in smoke test)")

    print("smoke_frb_pipeline: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())

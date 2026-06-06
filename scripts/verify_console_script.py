#!/usr/bin/env python3
"""Verify lwa-voltage-beam is installed and runnable after pip install -e ."""
from __future__ import annotations

import shutil
import subprocess
import sys


def main() -> int:
    try:
        from frb_search_pipeline.cli import main as cli_main
    except ImportError as exc:
        print("ERROR: cannot import frb_search_pipeline.cli:", exc, file=sys.stderr)
        print("  Run: pip uninstall -y lwa-fasttransients && pip install -e . --no-build-isolation", file=sys.stderr)
        return 1
    print("import frb_search_pipeline.cli OK")

    script = shutil.which("lwa-voltage-beam")
    if not script:
        print("ERROR: lwa-voltage-beam not on PATH", file=sys.stderr)
        return 1

    proc = subprocess.run(
        [script, "--help"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print("ERROR: lwa-voltage-beam --help failed:", proc.stderr, file=sys.stderr)
        if "StopIteration" in (proc.stderr or ""):
            print(
                "\nStale entry-point wrapper detected. Remove and reinstall:\n"
                "  pip uninstall -y lwa-fasttransients\n"
                f"  rm -f {script!r}\n"
                "  pip install -e . --no-build-isolation",
                file=sys.stderr,
            )
        return proc.returncode or 1
    print("console script OK:", script)

    proc = subprocess.run(
        [sys.executable, "-m", "frb_search_pipeline", "--help"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print("ERROR: python -m frb_search_pipeline --help failed:", proc.stderr, file=sys.stderr)
        return proc.returncode
    print("python -m frb_search_pipeline OK")

    return 0


if __name__ == "__main__":
    sys.exit(main())

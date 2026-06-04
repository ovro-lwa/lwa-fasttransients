#!/usr/bin/env python3
"""Install bin/lwa-voltage-beam with a direct import launcher.

Setuptools editable installs on some envs skip scripts= and/or leave a broken
console_scripts wrapper (StopIteration in load_entry_point). This writes a
reliable launcher using the current interpreter.
"""
from __future__ import annotations

import sys
import sysconfig
from pathlib import Path

BODY = """\
from frb_search_pipeline.cli import main
import sys

if __name__ == "__main__":
    sys.exit(main())
"""


def main() -> int:
    scripts_dir = Path(sysconfig.get_path("scripts"))
    scripts_dir.mkdir(parents=True, exist_ok=True)
    target = scripts_dir / "lwa-voltage-beam"
    target.write_text(f"#!{sys.executable}\n{BODY}")
    target.chmod(0o755)
    print("installed:", target)
    return 0


if __name__ == "__main__":
    sys.exit(main())

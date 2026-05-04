#!/usr/bin/env python3
"""run_pipeline.py - End-to-end FRB pipeline driver.

Runs steps 01..06 in sequence

Minimum inputs are filename, DM, and duration. RA/Dec only matter for the
PSRFITS header and are passed through with sensible defaults; override if
you care.

Example
-------
    # set environment first (calim2):
    export TEMPO=/opt/devel/pipeline/nkosogor/tempo
    export PRESTO=/opt/devel/pipeline/nkosogor/presto
    export LIBRARY_PATH=/opt/devel/pipeline/envs/fasttransients/lib64:$LIBRARY_PATH
    export LD_LIBRARY_PATH=/opt/devel/pipeline/envs/fasttransients/lib64:$LD_LIBRARY_PATH
    export PYTHONPATH=/opt/devel/nkosogor/nkosogor:$PYTHONPATH

    python run_pipeline.py \
        --voltage 061161_183721867877c175bee \
        --dm 108.3723 --duration 100

To skip earlier stages on a re-run (for example, the HDF5 already exists):
    python run_pipeline.py --voltage X --dm 108.3723 --duration 100 --start-from 02
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
STEPS = ["01", "02", "03", "04", "05", "06"]


def script_path(prefix: str) -> Path:
    hits = sorted(HERE.glob(f"{prefix}_*.py"))
    if not hits:
        sys.exit(f"Cannot find {prefix}_*.py in {HERE}")
    return hits[0]


def run(cmd, env=None):
    print("\n" + "=" * 72)
    print(">>> " + " ".join(str(c) for c in cmd))
    print("=" * 72, flush=True)
    rc = subprocess.call([str(c) for c in cmd], env=env)
    if rc != 0:
        sys.exit(f"\nStep failed (exit {rc}): {' '.join(str(c) for c in cmd)}")


def find_hdf5(workdir: Path, voltage_basename: str) -> Path | None:
    """Locate the HDF5 produced by step 01."""
    # Prefer Tuning2 (high band) which is what 02-05 default to.
    for pat in ("drx_*_b1t2_*.hdf5", "drx_*.hdf5", f"{voltage_basename}*.hdf5"):
        hits = sorted(workdir.glob(pat))
        if hits:
            return hits[-1]
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--voltage", required=True,
                   help="Voltage filename or full path.")
    p.add_argument("--dm", type=float, required=True, help="DM (pc/cm^3).")
    p.add_argument("--duration", type=float, required=True,
                   help="Duration kept when writing HDF5 (seconds).")

    p.add_argument("--ra",  type=float, default=0.0,  help="RA (deg) for PSRFITS header.")
    p.add_argument("--dec", type=float, default=0.0,  help="Dec (deg) for PSRFITS header.")
    p.add_argument("--workdir", default=".",
                   help="Working directory; outputs land here.")
    p.add_argument("--tuning", default="Tuning2", help="Tuning1 (low) or Tuning2 (high).")
    p.add_argument("--pol", default="I")

    p.add_argument("--start-from", choices=STEPS, default="01",
                   help="Skip steps before this one.")
    p.add_argument("--stop-after", choices=STEPS, default="06",
                   help="Stop after this step.")
    p.add_argument("--no-flags", action="store_true",
                   help="Pass --no-flags to step 04.")
    p.add_argument("--python", default=sys.executable,
                   help="Python interpreter to use for sub-commands.")

    # Pass-through extras for any step (rarely needed).
    p.add_argument("--extra-01", default="", help="Extra args for step 01 (quoted).")
    p.add_argument("--extra-02", default="", help="Extra args for step 02 (quoted).")
    p.add_argument("--extra-03", default="", help="Extra args for step 03 (quoted).")
    p.add_argument("--extra-04", default="", help="Extra args for step 04 (quoted).")
    p.add_argument("--extra-05", default="", help="Extra args for step 05 (quoted).")
    p.add_argument("--extra-06", default="", help="Extra args for step 06 (quoted).")
    args = p.parse_args()

    if STEPS.index(args.stop_after) < STEPS.index(args.start_from):
        sys.exit("--stop-after is before --start-from")

    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    voltage_basename = Path(args.voltage).name

    def want(step: str) -> bool:
        return STEPS.index(args.start_from) <= STEPS.index(step) <= STEPS.index(args.stop_after)

    PY = args.python

    # ---------- step 01 ----------
    if want("01"):
        cmd = [PY, str(script_path("01")),
               "--voltage", args.voltage,
               "--dm", f"{args.dm}",
               "--ra", f"{args.ra}",
               "--dec", f"{args.dec}",
               "--duration", f"{args.duration}",
               "--workdir", str(workdir)]
        if args.extra_01:
            cmd += args.extra_01.split()
        run(cmd)

    hdf5 = find_hdf5(workdir, voltage_basename)
    if STEPS.index(args.stop_after) >= STEPS.index("02"):
        if hdf5 is None:
            sys.exit(f"No HDF5 file found in {workdir} (looked for drx_*_b1t2_*.hdf5).")
        print(f"\nUsing HDF5: {hdf5}")

    # ---------- step 02 ----------
    if want("02"):
        cmd = [PY, str(script_path("02")),
               "--input", str(hdf5),
               "--tuning", args.tuning,
               "--pol", args.pol]
        if args.extra_02:
            cmd += args.extra_02.split()
        run(cmd)

    # ---------- step 03 ----------
    if want("03"):
        cmd = [PY, str(script_path("03")),
               "--input", str(hdf5),
               "--tuning", args.tuning,
               "--pol", args.pol]
        if args.extra_03:
            cmd += args.extra_03.split()
        run(cmd)

    # ---------- step 04 ----------
    if want("04"):
        cmd = [PY, str(script_path("04")),
               "--input", str(hdf5),
               "--tuning", args.tuning,
               "--pol", args.pol,
               "--dm", f"{args.dm}"]
        if args.no_flags:
            cmd.append("--no-flags")
        if args.extra_04:
            cmd += args.extra_04.split()
        run(cmd)

    # ---------- step 05 ----------
    # 04 names its output <base>_dm{dm:g}.npz, which truncates trailing digits
    # (e.g. 108.3723 -> 108.372). Match the same formatting here.
    npz_name = f"{hdf5.stem}_dm{args.dm:g}.npz" if hdf5 else None
    npz_path = (workdir / npz_name) if npz_name else None
    if want("05"):
        if npz_path is None or not npz_path.exists():
            # fallback: pick the newest dm*.npz next to the HDF5
            cands = sorted(workdir.glob(f"{hdf5.stem}_dm*.npz")) if hdf5 else []
            if not cands:
                sys.exit(f"No dedispersed .npz found for {hdf5}")
            npz_path = cands[-1]
            print(f"NOTE: using {npz_path} (expected name {npz_name} not found).")
        cmd = [PY, str(script_path("05")),
               "--input", str(npz_path)]
        if args.extra_05:
            cmd += args.extra_05.split()
        run(cmd)

    # ---------- step 06 ----------
    if want("06"):
        if npz_path is None or not npz_path.exists():
            sys.exit(f"No dedispersed .npz found for {hdf5}")
        csv_path = npz_path.with_name(f"{npz_path.stem}_candidates.csv")
        if not csv_path.exists():
            print(f"NOTE: no candidates CSV at {csv_path}; skipping step 06.")
        else:
            cmd = [PY, str(script_path("06")),
                   "--npz", str(npz_path),
                   "--csv", str(csv_path)]
            if args.extra_06:
                cmd += args.extra_06.split()
            run(cmd)

    print("\nPipeline finished.")


if __name__ == "__main__":
    main()

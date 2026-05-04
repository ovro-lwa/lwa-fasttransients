#!/usr/bin/env python3
"""01 - Convert a voltage beam recording to HDF5.

Pipeline:
  raw voltage  --writePsrfits2.py-->  PSRFITS  --writeHDF5FromPsrfits.py-->  HDF5

Number of channels for upchannelization is computed from the DM so the
intra-channel smearing stays below the time resolution:

    N_chan = ceil( sqrt( 8.3 * BW_MHz^2 * (f_low_MHz/1000)^-3 * DM ) )
    N_chan = round-up to nearest multiple of 16

Designed to run on the OVRO calim server inside the `fasttransients`
conda environment.  Both helper scripts must be importable as command-line
tools (their default paths are taken from the historical commands.txt).

Examples
--------
    python 01_convert_voltage_to_hdf5.py \
        --voltage 060942_0411495881658ef2ca4 \
        --dm 405 --ra 307.9622 --dec 54.499 --duration 600

    # override paths if your install differs:
    python 01_convert_voltage_to_hdf5.py --voltage X --dm 300 \
        --ra 207 --dec 17 --duration 500 \
        --write-psrfits /opt/devel/.../writePsrfits2.py \
        --write-hdf5    /opt/devel/.../writeHDF5FromPsrfits.py
"""
from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_WRITE_PSRFITS = "/opt/devel/nkosogor/nkosogor/chime/pulsar/writePsrfits2_drx_nopsr.py"
DEFAULT_WRITE_HDF5    = "/opt/devel/nkosogor/nkosogor/chime/pulsar/writeHDF5FromPsrfits.py"

# OVRO low-band defaults -- override with --low-freq / --bandwidth if needed
DEFAULT_LOW_FREQ_MHZ  = 63.2
DEFAULT_BANDWIDTH_MHZ = 19.6


def compute_num_channels(dm: float, low_freq_mhz: float, bw_mhz: float) -> int:
    n = math.ceil(math.sqrt(8.3 * (bw_mhz ** 2) * ((low_freq_mhz / 1000.0) ** -3) * dm))
    n = int(n)
    return ((n + 15) // 16) * 16


def run(cmd, cwd=None):
    print(f"\n>>> {' '.join(str(c) for c in cmd)}", flush=True)
    rc = subprocess.call(cmd, cwd=cwd)
    if rc != 0:
        sys.exit(f"Command failed (exit {rc}): {' '.join(str(c) for c in cmd)}")


def find_first(glob_pattern: str, search_dir: Path):
    hits = sorted(search_dir.glob(glob_pattern))
    return hits[0] if hits else None


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--voltage", required=True,
                   help="Voltage file basename or full path (no extension required).")
    p.add_argument("--dm", type=float, required=True,
                   help="DM (pc/cm^3) used to size the channelization.")
    p.add_argument("--ra", type=float, required=True, help="RA in degrees.")
    p.add_argument("--dec", type=float, required=True, help="Dec in degrees.")
    p.add_argument("--duration", type=float, required=True,
                   help="Duration to keep when writing HDF5 (seconds).")

    p.add_argument("--workdir", default=".",
                   help="Working directory (input voltage file expected here unless full path given). "
                        "Outputs (PSRFITS, HDF5) land here.")
    p.add_argument("--num-channels", type=int, default=None,
                   help="Override automatic channel-count calculation.")
    p.add_argument("--low-freq", type=float, default=DEFAULT_LOW_FREQ_MHZ,
                   help=f"Low-band start freq, MHz (default {DEFAULT_LOW_FREQ_MHZ}).")
    p.add_argument("--bandwidth", type=float, default=DEFAULT_BANDWIDTH_MHZ,
                   help=f"Bandwidth, MHz (default {DEFAULT_BANDWIDTH_MHZ}).")

    p.add_argument("--write-psrfits", default=DEFAULT_WRITE_PSRFITS,
                   help="Path to writePsrfits2.py")
    p.add_argument("--write-hdf5", default=DEFAULT_WRITE_HDF5,
                   help="Path to writeHDF5FromPsrfits.py")

    p.add_argument("--skip-psrfits", action="store_true",
                   help="Skip step 1 (assume drx_*.fits already exists in workdir).")
    p.add_argument("--skip-hdf5", action="store_true",
                   help="Skip step 2 (only produce PSRFITS).")
    p.add_argument("--python", default=sys.executable,
                   help="Python interpreter for sub-commands (default: current).")
    args = p.parse_args()

    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    voltage_arg = Path(args.voltage)
    if voltage_arg.is_absolute() and voltage_arg.exists():
        voltage_path = voltage_arg
    else:
        candidate = (workdir / voltage_arg.name)
        if not candidate.exists() and voltage_arg.exists():
            candidate = voltage_arg.resolve()
        voltage_path = candidate

    if not args.skip_psrfits and not voltage_path.exists():
        sys.exit(f"Voltage file not found: {voltage_path}")

    n_chan = args.num_channels or compute_num_channels(
        args.dm, args.low_freq, args.bandwidth
    )

    print("=" * 60)
    print(f"Voltage      : {voltage_path}")
    print(f"DM           : {args.dm}")
    print(f"RA / Dec     : {args.ra} / {args.dec}")
    print(f"Duration (s) : {args.duration}")
    print(f"# channels   : {n_chan}  (rounded to mult. of 16)")
    print(f"Workdir      : {workdir}")
    print("=" * 60)

    # ---- Step 1: writePsrfits2 ----
    if not args.skip_psrfits:
        if not Path(args.write_psrfits).exists():
            sys.exit(f"writePsrfits2.py not found at: {args.write_psrfits}")
        cmd = [args.python, args.write_psrfits, str(voltage_path),
               "-p",
               "-c", str(n_chan),
               "-r", f"{args.ra}",
               "-d", f"{args.dec}"]
        run(cmd, cwd=str(workdir))
    else:
        print("[skip] writePsrfits2 step skipped.")

    # locate the produced PSRFITS file (e.g. drx_60942_None_b1t2_0001.fits)
    fits_path = find_first("drx_*_b1t2_*.fits", workdir) \
                or find_first("drx_*.fits", workdir)
    if fits_path is None:
        sys.exit(f"No drx_*.fits produced in {workdir}.")
    print(f"PSRFITS file : {fits_path}")

    # ---- Step 2: writeHDF5FromPsrfits ----
    if not args.skip_hdf5:
        if not Path(args.write_hdf5).exists():
            sys.exit(f"writeHDF5FromPsrfits.py not found at: {args.write_hdf5}")
        cmd = [args.python, args.write_hdf5, str(fits_path),
               "-d", f"{args.duration}"]
        run(cmd, cwd=str(workdir))

        hdf5_path = fits_path.with_suffix(".hdf5")
        if not hdf5_path.exists():
            # fallback: any new .hdf5 next to fits
            hdf5_path = find_first(f"{fits_path.stem}*.hdf5", workdir) \
                        or find_first("*.hdf5", workdir)
        print(f"\nHDF5 output  : {hdf5_path}")
    else:
        print("[skip] writeHDF5FromPsrfits step skipped.")


if __name__ == "__main__":
    main()

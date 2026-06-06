#!/usr/bin/env python3
"""01 - Convert a voltage beam recording to HDF5.

Pipeline:
  raw voltage  --writePsrfits2.py-->  PSRFITS  --write_hdf5_from_psrfits.py-->  HDF5

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
        --dm 405 --ra 307.9622 --dec 54.499

    # keep only the first 600 seconds in the HDF5:
    python 01_convert_voltage_to_hdf5.py --voltage X --dm 300 \
        --ra 207 --dec 17 --duration 600

    # override paths if your install differs:
    python 01_convert_voltage_to_hdf5.py --voltage X --dm 300 \
        --ra 207 --dec 17 \
        --write-psrfits /opt/devel/.../writePsrfits2.py \
        --write-hdf5    /opt/devel/.../writeHDF5FromPsrfits.py
"""
from __future__ import annotations

import argparse
import math
import os
import re
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_WRITE_PSRFITS = "/opt/devel/nkosogor/nkosogor/chime/pulsar/writePsrfits2_drx_nopsr.py"
DEFAULT_WRITE_HDF5    = str(HERE / "write_hdf5_from_psrfits.py")

# OVRO low-band defaults -- override with --low-freq / --bandwidth if needed
DEFAULT_LOW_FREQ_MHZ  = 63.2
DEFAULT_BANDWIDTH_MHZ = 19.6

_SEGRE = re.compile(r"_(\d+)\.fits$")


def compute_num_channels(dm: float, low_freq_mhz: float, bw_mhz: float) -> int:
    n = math.ceil(math.sqrt(8.3 * (bw_mhz ** 2) * ((low_freq_mhz / 1000.0) ** -3) * dm))
    n = int(n)
    return ((n + 15) // 16) * 16


def run(cmd, cwd=None):
    print(f"\n>>> {' '.join(str(c) for c in cmd)}", flush=True)
    rc = subprocess.call(cmd, cwd=cwd)
    if rc != 0:
        sys.exit(f"Command failed (exit {rc}): {' '.join(str(c) for c in cmd)}")


def segment_sort_key(path: Path) -> tuple[int, str]:
    mtch = _SEGRE.search(path.name)
    return (int(mtch.group(1)) if mtch else 0, path.name)


def find_psrfits_segments(workdir: Path) -> list[Path]:
    """Return all PSRFITS segments for Tuning2, sorted by segment number."""
    hits = list(workdir.glob("drx_*_b1t2_*.fits"))
    if not hits:
        hits = list(workdir.glob("drx_*.fits"))
    return sorted(hits, key=segment_sort_key)


def remove_stale_psrfits(workdir: Path) -> list[Path]:
    """Remove existing PSRFITS segments so writePsrfits can recreate them."""
    existing = sorted(workdir.glob("drx_*.fits"), key=segment_sort_key)
    for path in existing:
        path.unlink()
    return existing


def psrfits_max_duration_sec(fits_path: Path) -> float:
    """Seconds covered by all subintegrations in one PSRFITS file."""
    from astropy.io import fits as astrofits

    with astrofits.open(fits_path, memmap=True) as hdulist:
        n_subints = len(hdulist[1].data)
        t_int = float(hdulist[1].header["TBIN"])
        n_subs = int(hdulist[1].header["NSBLK"])
    return n_subints * n_subs * t_int


def psrfits_total_duration_sec(fits_paths: list[Path]) -> float:
    """Total span across all sequential PSRFITS segments."""
    return sum(psrfits_max_duration_sec(path) for path in fits_paths)


def effective_hdf5_duration(requested_sec: float, fits_paths: list[Path]) -> float:
    """Resolve HDF5 duration from the requested value and combined PSRFITS span.

    ``requested_sec <= 0`` means use the full combined PSRFITS span.  A positive
    request is capped at the available span.
    """
    available = psrfits_total_duration_sec(fits_paths)
    if requested_sec <= 0:
        return available
    if requested_sec > available:
        print(
            f"WARNING: requested duration {requested_sec:.3f} s exceeds combined PSRFITS "
            f"span ({available:.3f} s); using full span for HDF5 conversion.",
            flush=True,
        )
        return available
    return requested_sec


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--voltage", required=True,
                   help="Voltage file basename or full path (no extension required).")
    p.add_argument("--dm", type=float, required=True,
                   help="DM (pc/cm^3) used to size the channelization.")
    p.add_argument("--ra", type=float, required=True, help="RA in degrees.")
    p.add_argument("--dec", type=float, required=True, help="Dec in degrees.")
    p.add_argument("--duration", type=float, default=0.0,
                   help="Duration to keep when writing HDF5 (seconds). "
                        "0 (default) uses the full combined PSRFITS span.")

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
                   help="Path to the PSRFITS -> HDF5 converter")

    p.add_argument("--skip-psrfits", action="store_true",
                   help="Skip step 1 (assume drx_*.fits already exists in workdir).")
    p.add_argument("--skip-hdf5", action="store_true",
                   help="Skip step 2 (only produce PSRFITS).")
    p.add_argument("--python", default=sys.executable,
                   help="Python interpreter for sub-commands (default: current).")
    args = p.parse_args()

    if args.duration < 0:
        sys.exit("--duration must be >= 0 (0 means full file).")

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
    if args.duration <= 0:
        print("Duration (s) : full file (0)")
    else:
        print(f"Duration (s) : {args.duration}")
    print(f"# channels   : {n_chan}  (rounded to mult. of 16)")
    print(f"Workdir      : {workdir}")
    print("=" * 60)

    # ---- Step 1: writePsrfits2 ----
    if not args.skip_psrfits:
        stale = remove_stale_psrfits(workdir)
        if stale:
            print(
                f"Removing {len(stale)} stale PSRFITS file(s) before regeneration "
                "(writePsrfits cannot overwrite existing outputs):"
            )
            for path in stale:
                print(f"  {path.name}")

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

    fits_paths = find_psrfits_segments(workdir)
    if not fits_paths:
        sys.exit(f"No drx_*.fits produced in {workdir}.")
    if len(fits_paths) == 1:
        print(f"PSRFITS file : {fits_paths[0]}")
    else:
        print(f"PSRFITS files: {len(fits_paths)} segments")
        for path in fits_paths:
            print(f"  {path.name}  ({psrfits_max_duration_sec(path):.3f} s)")

    # ---- Step 2: write_hdf5_from_psrfits ----
    if not args.skip_hdf5:
        if not Path(args.write_hdf5).exists():
            sys.exit(f"HDF5 converter not found at: {args.write_hdf5}")
        hdf5_duration = effective_hdf5_duration(args.duration, fits_paths)
        if args.duration <= 0:
            print(f"HDF5 duration : {hdf5_duration:.3f} s (full combined PSRFITS span)")
        elif hdf5_duration != args.duration:
            print(f"HDF5 duration : {hdf5_duration:.3f} s (capped from {args.duration:.3f} s)")
        else:
            print(f"HDF5 duration : {hdf5_duration:.3f} s")
        cmd = [args.python, str(args.write_hdf5),
               *[str(path) for path in fits_paths],
               "-d", f"{hdf5_duration}"]
        run(cmd, cwd=str(workdir))

        hdf5_path = fits_paths[0].with_suffix(".hdf5")
        if not hdf5_path.exists():
            stem = fits_paths[0].stem
            candidates = sorted(workdir.glob(f"{stem}*.hdf5"))
            if not candidates:
                candidates = sorted(workdir.glob("drx_*_b1t2_*.hdf5"))
            if not candidates:
                candidates = sorted(workdir.glob("*.hdf5"))
            hdf5_path = candidates[0] if candidates else hdf5_path
        print(f"\nHDF5 output  : {hdf5_path}")
    else:
        print("[skip] PSRFITS -> HDF5 step skipped.")


if __name__ == "__main__":
    main()

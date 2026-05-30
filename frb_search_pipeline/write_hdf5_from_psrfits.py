#!/usr/bin/env python3
"""Convert one or more sequential PSRFITS segments to a single LWA HDF5 file.

Extends the behaviour of writeHDF5FromPsrfits.py for the FRB pipeline:

  * Multiple ``*_0001.fits``, ``*_0002.fits``, ... files with the **same**
    tuning are concatenated in time order.
  * Exactly two files with tunings t1 and t2 (one segment each) are written
    in parallel into the same HDF5 time rows, matching the upstream tool.
  * Existing output HDF5 files are overwritten without prompting (pipeline use).

Example
-------
    python write_hdf5_from_psrfits.py drx_61188_None_b1t2_0001.fits \
        drx_61188_None_b1t2_0002.fits -d 0
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import numpy
from astropy.io import fits as astrofits
from astropy.time import Time as AstroTime

import data as hdfData

import lsl.common.progress as progress
from lsl.misc import parser as aph


_FNRE = re.compile(r".*_b(?P<beam>[1-4])(t(?P<tuning>[12]))?_.*\.fits")
_SEGRE = re.compile(r"_(\d+)\.fits$")


def parse_tuning(path: str) -> int:
    mtch = _FNRE.search(path)
    if mtch is None:
        return 1
    try:
        return int(mtch.group("tuning"))
    except TypeError:
        return 1


def segment_index(path: str) -> int:
    mtch = _SEGRE.search(path)
    return int(mtch.group(1)) if mtch else 0


def classify_inputs(filenames: list[str]) -> tuple[str, list[list[str]]]:
    """Return (mode, groups).

    mode is ``"concat"`` for sequential same-tuning segments, or ``"dual"`` for
    one t1 and one t2 file written in parallel.
    """
    by_tuning: dict[int, list[str]] = {}
    for name in filenames:
        by_tuning.setdefault(parse_tuning(name), []).append(name)

    for tuning in by_tuning:
        by_tuning[tuning].sort(key=lambda p: (segment_index(p), p))

    if len(by_tuning) == 2 and all(len(v) == 1 for v in by_tuning.values()):
        groups = [by_tuning[t] for t in sorted(by_tuning)]
        return "dual", groups

    if len(by_tuning) != 1:
        tunings = ", ".join(f"t{t}({len(v)} files)" for t, v in sorted(by_tuning.items()))
        raise RuntimeError(
            "Unsupported PSRFITS input mix: "
            f"{tunings}. Pass same-tuning segments or exactly one t1 and one t2 file."
        )

    tuning = next(iter(by_tuning))
    return "concat", [by_tuning[tuning]]


def subint_span_sec(hdulist) -> tuple[float, float, int, int, int]:
    """Return (t_subint, t_int, n_subs, n_subints, span_sec)."""
    t_int = float(hdulist[1].header["TBIN"])
    n_subs = int(hdulist[1].header["NSBLK"])
    t_subint = n_subs * t_int
    n_subints = len(hdulist[1].data)
    return t_subint, t_int, n_subs, n_subints, n_subints * t_subint


def resolve_duration_sec(requested_sec: float | None, available_sec: float, t_subint: float) -> float:
    if requested_sec is None or requested_sec <= 0:
        return available_sec
    if requested_sec > available_sec:
        print(
            f"WARNING: requested duration {requested_sec:.3f} s exceeds available "
            f"PSRFITS span ({available_sec:.3f} s); using full span.",
            flush=True,
        )
        return available_sec
    dur_subints = max(1, int(round(requested_sec / t_subint)))
    return dur_subints * t_subint


def read_metadata(filename: str):
    hdulist = astrofits.open(filename, memmap=True)
    mtch = _FNRE.search(filename)
    beam = int(mtch.group("beam")) if mtch else 0
    tuning = parse_tuning(filename)

    hdr0 = hdulist[0].header
    source_name = hdr0["SRC_NAME"]
    ra = hdr0["RA"].split(":", 2)
    ra = sum(float(v) / 60**i for i, v in enumerate(ra)) * 15.0
    dec = hdr0["DEC"]
    dec_sign = -1.0 if dec.find("-") != -1 else 1.0
    dec = dec.replace("-", "").split(":", 2)
    dec = dec_sign * sum(float(v) / 60**i for i, v in enumerate(dec))
    epoch = float(hdr0["EQUINOX"])

    t_start = AstroTime(
        hdr0["STT_IMJD"],
        (hdr0["STT_SMJD"] + hdr0["STT_OFFS"]) / 86400.0,
        format="mjd",
        scale="utc",
    )
    c_freq = hdr0["OBSFREQ"] * 1e6
    srate = hdr0["OBSBW"] * 1e6
    lfft = hdulist[1].header["NCHAN"]
    t_int = hdulist[1].header["TBIN"]
    n_subs = hdulist[1].header["NSBLK"]
    t_subs = n_subs * t_int
    n_pol = hdulist[1].header["NPOL"]
    if n_pol == 1:
        data_products = ["I"]
    elif n_pol == 2:
        if hdr0["FD_POLN"] == "CIRC":
            data_products = ["LL", "RR"]
        else:
            data_products = ["XX", "YY"]
    else:
        data_products = ["I", "Q", "U", "V"]

    return hdulist, {
        "beam": beam,
        "tuning": tuning,
        "source_name": source_name,
        "ra": ra,
        "dec": dec,
        "epoch": epoch,
        "t_start": t_start,
        "c_freq": c_freq,
        "srate": srate,
        "lfft": lfft,
        "t_int": t_int,
        "n_subs": n_subs,
        "t_subs": t_subs,
        "n_pol": n_pol,
        "data_products": data_products,
        "trk_mode": hdr0["TRK_MODE"],
    }


def validate_against_first(meta: dict, first: dict, filename: str) -> None:
    keys = (
        "source_name", "ra", "dec", "epoch", "t_start", "srate",
        "lfft", "t_int", "t_subs", "n_pol",
    )
    for key in keys:
        if meta[key] != first[key]:
            raise RuntimeError(
                f"PSRFITS metadata mismatch for '{os.path.basename(filename)}': "
                f"{key} differs from the first input file."
            )


def create_hdf5(
    first_file: str,
    all_files: list[str],
    meta: dict,
    n_time_rows: int,
    overwrite: bool,
):
    outname = os.path.splitext(os.path.basename(first_file))[0]
    if len(all_files) == 2 and parse_tuning(all_files[0]) != parse_tuning(all_files[1]):
        outname = outname.replace("t1", "").replace("t2", "")
    outname = f"{outname}.hdf5"

    if os.path.exists(outname):
        if not overwrite:
            raise RuntimeError(f"Output file '{outname}' already exists")
        os.unlink(outname)

    beam = meta["beam"]
    srate = meta["srate"]
    lfft = meta["lfft"]
    data_products = meta["data_products"]

    f = hdfData.create_new_file(outname)
    hdfData.fill_minimum(f, 1, beam, srate)
    for t in (1, 2):
        hdfData.create_observation_set(
            f, 1, t, numpy.arange(lfft, dtype=numpy.float64), n_time_rows, data_products
        )
    f.attrs["FileGenerator"] = "write_hdf5_from_psrfits.py"
    f.attrs["InputData"] = ",".join(os.path.basename(x) for x in all_files)

    ds = {"obs1": hdfData.get_observation_set(f, 1)}
    ds["obs1-time"] = hdfData.get_time(f, 1)
    for t in (1, 2):
        ds[f"obs1-freq{t}"] = hdfData.get_data_set(f, 1, t, "freq")
        for p in data_products:
            ds[f"obs1-{p}{t}"] = hdfData.get_data_set(f, 1, t, p)

    for t in (1, 2):
        tuning_info = ds["obs1"].get(f"Tuning{t}", None)
        mask_info = tuning_info.create_group("Mask")
        for p in data_products:
            mask_info.create_dataset(p, ds[f"obs1-{p}{t}"].shape, "bool")
            ds[f"obs1-mask-{p}{t}"] = mask_info.get(p, None)

    ds["obs1"].attrs["ObservationName"] = meta["source_name"]
    ds["obs1"].attrs["TargetName"] = meta["source_name"]
    ds["obs1"].attrs["RA"] = meta["ra"] / 15.0
    ds["obs1"].attrs["RA_Units"] = "hours"
    ds["obs1"].attrs["Dec"] = meta["dec"]
    ds["obs1"].attrs["Dec_Units"] = "degrees"
    ds["obs1"].attrs["Epoch"] = meta["epoch"]
    ds["obs1"].attrs["TrackingMode"] = meta["trk_mode"]
    ds["obs1"].attrs["tInt"] = meta["t_int"]
    ds["obs1"].attrs["tInt_Units"] = "s"
    ds["obs1"].attrs["LFFT"] = lfft
    ds["obs1"].attrs["nChan"] = lfft

    return f, ds, outname


def write_subint(
    ds,
    hdulist,
    meta: dict,
    tuning: int,
    subint_index: int,
    k_base: int,
) -> None:
    lfft = meta["lfft"]
    t_int = meta["t_int"]
    n_subs = meta["n_subs"]
    n_pol = meta["n_pol"]
    data_products = meta["data_products"]
    t_start_i = int(meta["t_start"].unix)
    t_start_f = meta["t_start"].unix - t_start_i

    subint = hdulist[1].data[subint_index]
    msk = numpy.where(subint[13] >= 0.5, False, True)
    bzero = subint[14]
    bscl = subint[15]
    bzero.shape = (lfft, n_pol)
    bscl.shape = (lfft, n_pol)
    bzero = bzero.T
    bscl = bscl.T
    data = subint[16]
    data.shape = (n_subs, lfft, n_pol)
    data = data.T

    for j in range(n_subs):
        k = k_base + j
        t = subint[1] + t_int * (j - n_subs // 2)
        d = data[:, :, j] * bscl + bzero
        ds["obs1-time"][k] = (t_start_i, t_start_f + t)
        for pol_idx, pol_name in enumerate(data_products):
            ds[f"obs1-{pol_name}{tuning}"][k, :] = d[pol_idx, :]
            ds[f"obs1-mask-{pol_name}{tuning}"][k, :] = msk


def convert_concat(filenames: list[str], skip_sec: float, duration_sec: float | None, overwrite: bool) -> str:
    hd0, meta0 = read_metadata(filenames[0])
    t_subint, _, n_subs, _, _ = subint_span_sec(hd0)
    hd0.close()

    spans = []
    for name in filenames:
        with astrofits.open(name, memmap=True) as hdul:
            _, _, _, n_subints, span_sec = subint_span_sec(hdul)
        spans.append((name, n_subints, span_sec))

    available_sec = sum(span for _, _, span in spans)
    target_sec = resolve_duration_sec(duration_sec, available_sec, t_subint)
    skip_subints = int(round(skip_sec / t_subint))
    target_subints = max(1, int(round(target_sec / t_subint)))

    if skip_subints * t_subint >= available_sec:
        raise RuntimeError(
            f"Skip {skip_sec:.3f} s exceeds available PSRFITS span ({available_sec:.3f} s)."
        )

    n_time_rows = target_subints * n_subs
    _, meta0 = read_metadata(filenames[0])
    f, ds, outname = create_hdf5(filenames[0], filenames, meta0, n_time_rows, overwrite)

    remaining_subints = target_subints
    skip_left = skip_subints
    k_base = 0
    total_work = target_subints
    pbar = progress.ProgressBarPlus(max=total_work)

    print(f"Concatenating {len(filenames)} PSRFITS segment(s) -> {outname}")
    print(f"Available: {available_sec:.3f} s  Target: {target_sec:.3f} s  Skip: {skip_sec:.3f} s")

    for file_idx, (filename, n_subints, _) in enumerate(spans):
        hdulist, meta = read_metadata(filename)
        if file_idx > 0:
            validate_against_first(meta, meta0, filename)

        tuning = meta["tuning"]
        freq = hdulist[1].data[0][12] * 1e6
        ds["obs1"].attrs["RBW"] = freq[1] - freq[0]
        ds["obs1"].attrs["RBW_Units"] = "Hz"
        ds[f"obs1-freq{tuning}"][:] = freq

        local_start = 0
        if skip_left > 0:
            if skip_left >= n_subints:
                skip_left -= n_subints
                hdulist.close()
                continue
            local_start = skip_left
            skip_left = 0

        take = min(n_subints - local_start, remaining_subints)
        for i in range(local_start, local_start + take):
            write_subint(
                ds, hdulist, meta, tuning, i, k_base + (i - local_start) * n_subs,
            )
            pbar.inc()
            sys.stdout.write(f"{pbar.show()}\r")
            sys.stdout.flush()

        k_base += take * n_subs
        remaining_subints -= take
        hdulist.close()

        if remaining_subints <= 0:
            break

    sys.stdout.write(f"{pbar.show()}\n")
    f.close()
    return outname


def convert_dual(groups: list[list[str]], skip_sec: float, duration_sec: float | None, overwrite: bool) -> str:
    files = [groups[0][0], groups[1][0]]
    hd0, meta0 = read_metadata(files[0])
    t_subint, _, n_subs, n_subints0, span0 = subint_span_sec(hd0)
    hd0.close()

    with astrofits.open(files[1], memmap=True) as hd1:
        _, _, _, n_subints1, span1 = subint_span_sec(hd1)

    available_sec = min(span0, span1)
    target_sec = resolve_duration_sec(duration_sec, available_sec, t_subint)
    skip_subints = int(round(skip_sec / t_subint))
    target_subints = max(1, int(round(target_sec / t_subint)))
    target_subints = min(target_subints, n_subints0, n_subints1)

    n_time_rows = target_subints * n_subs
    _, meta0 = read_metadata(files[0])
    f, ds, outname = create_hdf5(files[0], files, meta0, n_time_rows, overwrite)

    pbar = progress.ProgressBarPlus(max=target_subints * len(files))
    print(f"Writing dual-tuning PSRFITS -> {outname}")
    print(f"Available: {available_sec:.3f} s  Target: {target_subints * t_subint:.3f} s")

    for file_idx, filename in enumerate(files):
        hdulist, meta = read_metadata(filename)
        validate_against_first(meta, meta0, filename)
        tuning = meta["tuning"]
        freq = hdulist[1].data[0][12] * 1e6
        ds["obs1"].attrs["RBW"] = freq[1] - freq[0]
        ds["obs1"].attrs["RBW_Units"] = "Hz"
        ds[f"obs1-freq{tuning}"][:] = freq

        for i in range(skip_subints, skip_subints + target_subints):
            write_subint(ds, hdulist, meta, tuning, i, (i - skip_subints) * n_subs)
            pbar.inc()
            sys.stdout.write(f"{pbar.show()}\r")
            sys.stdout.flush()
        hdulist.close()

    sys.stdout.write(f"{pbar.show()}\n")
    f.close()
    return outname


def main(args) -> None:
    mode, groups = classify_inputs(args.filename)
    if mode == "concat":
        convert_concat(groups[0], args.skip, args.duration, args.overwrite)
    else:
        convert_dual(groups, args.skip, args.duration, args.overwrite)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("filename", type=str, nargs="+", help="PSRFITS file(s) to process")
    parser.add_argument("-s", "--skip", type=aph.positive_or_zero_float, default=0.0,
                        help="seconds to skip from the start of the concatenated data")
    parser.add_argument("-d", "--duration", type=float, default=0.0,
                        help="seconds to save (0 = full available span)")
    parser.add_argument("--no-overwrite", action="store_true",
                        help="fail if the output HDF5 already exists")
    args = parser.parse_args()
    if args.duration < 0:
        parser.error("--duration must be >= 0")
    args.overwrite = not args.no_overwrite
    main(args)

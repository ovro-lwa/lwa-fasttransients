#!/usr/bin/env python3
"""02 - Plot raw HDF5 dynamic spectrum (pre-dedispersion, pre-flagging).

Streams the file in time chunks, downsamples to a target image size,
removes a per-channel median baseline, and writes a PNG.

Example
-------
    python 02_plot_raw.py --input drx_60942_None_b1t2_0001.hdf5 --tuning Tuning2
"""
from __future__ import annotations

import argparse
import os
import sys

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import (
    base_no_ext, ensure_dir, infer_tsamp, pick_pol, get_freq_mhz, read_block,
)


def downsample_stream(h5_path, tuning_name, pol, i_method,
                      t_start, t_dur, target_t, target_f, time_chunk):
    with h5py.File(h5_path, "r") as f:
        obs = f["Observation1"]
        tun = obs[tuning_name]
        freq_mhz = get_freq_mhz(tun)
        nfreq = freq_mhz.size
        chosen, i_meth = pick_pol(tun, pol=pol, i_method=i_method)
        tsamp = infer_tsamp(obs)
        if isinstance(chosen, tuple):
            ntime_full = tun[chosen[0]].shape[0]
        else:
            ntime_full = tun[chosen].shape[0]

        s_idx = max(0, int(np.floor(t_start / tsamp)))
        if t_dur is None or t_dur <= 0:
            e_idx = ntime_full
        else:
            e_idx = min(ntime_full, int(np.ceil((t_start + t_dur) / tsamp)))
        ntime = e_idx - s_idx
        if ntime < 2:
            sys.exit("Selected time window is empty.")

        ds_t = max(1, int(np.ceil(ntime / target_t)))
        ds_f = max(1, int(np.ceil(nfreq / target_f)))
        nfreq_ds = (nfreq // ds_f) * ds_f
        nfreq_out = nfreq_ds // ds_f
        ntime_out = ntime // ds_t

        freq_axis = freq_mhz[:nfreq_ds].reshape(-1, ds_f).mean(axis=1)
        img = np.empty((ntime_out, nfreq_out), dtype=np.float32)
        tcenters = t_start + (np.arange(ntime_out) * ds_t + 0.5 * ds_t) * tsamp

        write_row = 0
        buf = None
        cur = s_idx
        chosen_packed = (chosen, i_meth) if isinstance(chosen, tuple) else chosen

        while cur < e_idx and write_row < ntime_out:
            cnt = min(time_chunk, e_idx - cur)
            block = read_block(tun, chosen_packed, slice(cur, cur + cnt), slice(0, nfreq_ds))

            if not np.isfinite(block).all():
                med = np.nanmedian(block)
                block = np.where(np.isfinite(block), block, med)

            block = block.reshape(block.shape[0], nfreq_out, ds_f).mean(axis=2)

            buf = block if buf is None else np.vstack([buf, block])
            full = buf.shape[0] // ds_t
            if full > 0:
                use = full * ds_t
                chunk = buf[:use].reshape(full, ds_t, nfreq_out).mean(axis=1)
                end = min(write_row + full, ntime_out)
                img[write_row:end] = chunk[: end - write_row]
                write_row = end
                buf = buf[use:]
            cur += cnt

    return img, tcenters, freq_axis, tsamp, ds_t, ds_f


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="HDF5 file path.")
    p.add_argument("--tuning", default="Tuning2", help='"Tuning1" or "Tuning2".')
    p.add_argument("--pol", default="I", choices=["I", "XX", "YY"])
    p.add_argument("--i-method", default="mean", choices=["mean", "sum"])
    p.add_argument("--t-start", type=float, default=0.0, help="Window start (s).")
    p.add_argument("--t-dur", type=float, default=-1.0,
                   help="Window duration (s); <=0 means full file.")
    p.add_argument("--target-time", type=int, default=2000)
    p.add_argument("--target-freq", type=int, default=512)
    p.add_argument("--time-chunk", type=int, default=8192)
    p.add_argument("--outdir", default=None)
    p.add_argument("--cmap", default="inferno")
    p.add_argument("--vmin", type=float, default=-5.0)
    p.add_argument("--vmax", type=float, default=5.0)
    args = p.parse_args()

    if not os.path.isfile(args.input):
        sys.exit(f"Not a file: {args.input}")
    out_dir = ensure_dir(args.outdir or os.path.dirname(os.path.abspath(args.input)) or ".")
    base = base_no_ext(args.input)

    t_dur = None if args.t_dur is None or args.t_dur <= 0 else args.t_dur
    img, tcenters, freq_axis, tsamp, ds_t, ds_f = downsample_stream(
        args.input, args.tuning, args.pol, args.i_method,
        args.t_start, t_dur, args.target_time, args.target_freq, args.time_chunk,
    )

    # baseline-remove per channel (robust)
    img = img - np.nanmedian(img, axis=0, keepdims=True)
    mad = np.nanmedian(np.abs(img), axis=0, keepdims=True) + 1e-6
    img = img / mad

    print(f"image shape time x freq: {img.shape}; tsamp={tsamp:.6g}s; ds_t={ds_t} ds_f={ds_f}")

    plt.figure(figsize=(12, 6))
    plt.imshow(
        img.T, aspect="auto", origin="lower",
        extent=[tcenters[0], tcenters[-1], freq_axis.min(), freq_axis.max()],
        cmap=args.cmap, vmin=args.vmin, vmax=args.vmax,
    )
    plt.colorbar(label="baseline-removed (MAD units)")
    plt.xlabel("Time since file start (s)")
    plt.ylabel("Frequency (MHz)")
    plt.title(f"Raw waterfall — {base} ({args.tuning}, pol={args.pol})")
    plt.tight_layout()
    out_png = os.path.join(out_dir, f"{base}_raw_waterfall.png")
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()

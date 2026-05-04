#!/usr/bin/env python3
"""04 - Incoherent dedispersion on CPU with flags + per-channel whitening.

Reads an HDF5 dynamic spectrum, applies optional flags from <base>_flags.h5,
robustly normalizes each frequency channel by default, and dedisperses to one
or more DMs by integer time-shifting each channel.

The shift is NOT circular. For a channel with delay `shift`, the code uses:
    col = channel[shift : shift + n_out_time]
where n_out_time = ntime - max_shift.

Default output per DM:
    <base>_dm<DM>.npz

Optional full dedispersed dynamic spectrum:
    <base>_dm<DM>_full_dynspec.h5

Example:
    python 04_dedisperse_cpu.py --input drx_61161_None_b1t2_0001.hdf5 --dm 108.37230
"""
from __future__ import annotations

import argparse
import os
import sys

import h5py
import numpy as np
from tqdm import tqdm

from utils import (
    base_no_ext,
    ensure_dir,
    infer_tsamp,
    pick_pol,
    get_freq_mhz,
    read_block,
    dm_delay_seconds,
)


def safe_dm_tag(dm: float) -> str:
    return f"{dm:g}".replace("+", "").replace("-", "m").replace(".", "p")


def nanmean_no_warning(a: np.ndarray, axis: int) -> np.ndarray:
    finite = np.isfinite(a)
    count = finite.sum(axis=axis)
    total = np.where(finite, a, 0.0).sum(axis=axis)
    out = np.full(total.shape, np.nan, dtype=np.float64)
    np.divide(total, count, out=out, where=count > 0)
    return out


def load_flags(flag_path: str | None, ntime: int, nfreq: int):
    """Load <base>_flags.h5 if present.

    Returns (bad_ch, block_ch_mask, block_size, imp_mask) or Nones.
    In the updated 03 script, /sk_block_mask is the combined block/channel mask.
    """
    if not flag_path or not os.path.isfile(flag_path):
        return None, None, None, None

    with h5py.File(flag_path, "r") as f:
        bad_ch = f["bad_ch"][:].astype(bool)
        block_ch_mask = f["sk_block_mask"][:].astype(bool)
        imp = f["imp_time_mask"][:].astype(bool)
        block_size = int(f.attrs.get("block_size", f.attrs.get("sk_block_size", 0)))

    if bad_ch.size != nfreq or imp.size != ntime:
        print("WARNING: flag file dimensions do not match data; ignoring flags.")
        return None, None, None, None

    if block_ch_mask.size and block_ch_mask.shape[1] != nfreq:
        print("WARNING: block/channel flag dimensions do not match data; ignoring flags.")
        return None, None, None, None

    return bad_ch, block_ch_mask, block_size, imp


def apply_flags_to_block(
    blk: np.ndarray,
    t_start_abs: int,
    bad_ch: np.ndarray | None,
    block_ch_mask: np.ndarray | None,
    block_size: int | None,
    imp_mask: np.ndarray | None,
) -> np.ndarray:
    """In-place: write NaN where flagged. blk shape is time x frequency."""
    if bad_ch is not None and bad_ch.any():
        blk[:, bad_ch] = np.nan

    if block_ch_mask is not None and block_size and block_ch_mask.size:
        nT = blk.shape[0]
        b0 = t_start_abs // block_size
        b1 = min(block_ch_mask.shape[0], (t_start_abs + nT + block_size - 1) // block_size)

        for b in range(b0, b1):
            rl = max(b * block_size, t_start_abs) - t_start_abs
            rh = min((b + 1) * block_size, t_start_abs + nT) - t_start_abs
            if rh <= rl:
                continue
            bad = block_ch_mask[b]
            if bad.any():
                blk[rl:rh, bad] = np.nan

    if imp_mask is not None and imp_mask.size:
        ir = imp_mask[t_start_abs:t_start_abs + blk.shape[0]]
        if ir.any():
            blk[ir, :] = np.nan

    return blk


def robust_normalize_channels(
    blk: np.ndarray,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-channel median subtraction and robust sigma normalization.

    Input/output shape: time x frequency.
    Returns normalized block plus per-channel median and sigma.
    """
    blk = np.asarray(blk, dtype=np.float64)
    out = np.full_like(blk, np.nan, dtype=np.float64)

    finite_cols = np.isfinite(blk).any(axis=0)
    med = np.full(blk.shape[1], np.nan, dtype=np.float64)
    sig = np.full(blk.shape[1], np.nan, dtype=np.float64)

    if not finite_cols.any():
        return out, med, sig

    med[finite_cols] = np.nanmedian(blk[:, finite_cols], axis=0)
    resid = blk - med[None, :]
    mad = np.full(blk.shape[1], np.nan, dtype=np.float64)
    mad[finite_cols] = np.nanmedian(np.abs(resid[:, finite_cols]), axis=0)
    sig = 1.4826 * mad

    good = finite_cols & np.isfinite(sig) & (sig > eps)
    out[:, good] = resid[:, good] / sig[None, good]

    return out, med, sig


def create_full_dynspec_file(
    out_h5: str,
    nfreq_keep: int,
    n_out_time: int,
    freq_axis: np.ndarray,
    tsamp: float,
    dm: float,
    freq_min: float,
    freq_ref: float,
    normalize_channels: bool,
    series_mode: str,
):
    """Create an HDF5 file for optional full-resolution dedispersed dynspec."""
    fh = h5py.File(out_h5, "w")
    chunks = (1, min(8192, max(1, n_out_time)))
    ds = fh.create_dataset(
        "dynspec",
        shape=(nfreq_keep, n_out_time),
        dtype="float32",
        chunks=chunks,
        compression="gzip",
        compression_opts=4,
        fillvalue=np.nan,
    )
    fh.create_dataset("freq_axis_mhz", data=freq_axis.astype(np.float32), compression="gzip")
    fh.create_dataset("t_axis", data=(np.arange(n_out_time, dtype=np.float32) * np.float32(tsamp)), compression="gzip")
    fh.attrs["tsamp"] = float(tsamp)
    fh.attrs["dm"] = float(dm)
    fh.attrs["freq_min"] = float(freq_min)
    fh.attrs["freq_ref"] = float(freq_ref)
    fh.attrs["normalize_channels"] = bool(normalize_channels)
    fh.attrs["series_mode"] = str(series_mode)
    fh.attrs["description"] = "Full dedispersed dynamic spectrum; flagged samples are NaN."
    return fh, ds


def dedisperse_one_dm(
    h5_path: str,
    tuning_name: str,
    pol: str,
    i_method: str,
    dm: float,
    freq_min: float,
    flag_paths,
    freq_batch: int,
    target_freq_out: int,
    normalize_channels: bool,
    norm_eps: float,
    series_mode: str,
    dynspec_hole_if_any_flagged: bool,
    save_full_dynspec: bool,
    out_dir: str,
    base: str,
):
    """Run dedispersion for one DM."""
    with h5py.File(h5_path, "r") as f:
        obs = f["Observation1"]
        tun = obs[tuning_name]

        freq_full_mhz = get_freq_mhz(tun)
        nfreq = int(freq_full_mhz.size)
        ascending = bool(freq_full_mhz[0] < freq_full_mhz[-1])

        chosen, i_meth = pick_pol(tun, pol=pol, i_method=i_method)
        chosen_packed = (chosen, i_meth) if isinstance(chosen, tuple) else chosen

        tsamp = infer_tsamp(obs)
        if isinstance(chosen, tuple):
            ntime = int(tun[chosen[0]].shape[0])
        else:
            ntime = int(tun[chosen].shape[0])

        # Descending frequency order: highest first, used as reference.
        freq_desc = freq_full_mhz[::-1].copy() if ascending else freq_full_mhz.copy()
        f_ref = float(freq_desc[0])

        delays_s = dm_delay_seconds(freq_desc, f_ref, dm)
        delay_samples = np.round(delays_s / tsamp).astype(np.int64)
        max_shift = int(delay_samples.max())
        n_out_time = int(ntime - max_shift)
        if n_out_time <= 0:
            sys.exit(f"DM={dm} too large for file: max_shift={max_shift} >= ntime={ntime}")

        keep_chan = freq_desc >= float(freq_min)
        keep_idx = np.where(keep_chan)[0]
        if keep_idx.size == 0:
            sys.exit(f"No channels above freq_min={freq_min} MHz.")
        if not np.array_equal(keep_idx, np.arange(keep_idx.size)):
            sys.exit("Internal assumption failed: selected frequency channels are not contiguous.")
        nfreq_keep = int(keep_idx.size)

        print(
            f"  DM={dm:g}  ntime={ntime}  nfreq={nfreq}  "
            f"freq_used={nfreq_keep}  max_shift={max_shift} samp "
            f"({max_shift * tsamp:.2f}s)  out_ntime={n_out_time}"
        )
        print(
            f"  normalization={'on' if normalize_channels else 'off'}  "
            f"series_mode={series_mode}  dynspec_holes={dynspec_hole_if_any_flagged}"
        )

        # 1D dedispersed time series accumulators.
        series_sum = np.zeros(n_out_time, dtype=np.float64)
        series_cnt = np.zeros(n_out_time, dtype=np.float64)

        # Diagnostic dedispersed dynamic spectrum, frequency-downsampled.
        target_freq_out = max(1, int(target_freq_out))
        ds_f = max(1, int(np.ceil(nfreq_keep / target_freq_out)))
        nfreq_ds_full = (nfreq_keep // ds_f) * ds_f
        nfreq_out_ds = nfreq_ds_full // ds_f

        dynspec_sum = np.zeros((nfreq_out_ds, n_out_time), dtype=np.float32)
        dynspec_cnt = np.zeros((nfreq_out_ds, n_out_time), dtype=np.float32)
        dynspec_bad = np.zeros((nfreq_out_ds, n_out_time), dtype=np.uint16)
        freq_axis_ds = freq_desc[:nfreq_ds_full].reshape(-1, ds_f).mean(axis=1)

        bad_ch, block_ch_mask, block_size, imp_mask = flag_paths

        # Reverse masks to descending frequency order if needed.
        bad_ch_desc = bad_ch[::-1].copy() if (bad_ch is not None and ascending) else bad_ch
        block_ch_mask_desc = (
            block_ch_mask[:, ::-1].copy()
            if (block_ch_mask is not None and ascending and block_ch_mask.size)
            else block_ch_mask
        )

        full_h5 = None
        full_ds = None
        full_path = None
        if save_full_dynspec:
            dm_tag = safe_dm_tag(dm)
            full_path = os.path.join(out_dir, f"{base}_dm{dm_tag}_full_dynspec.h5")
            full_h5, full_ds = create_full_dynspec_file(
                full_path,
                nfreq_keep=nfreq_keep,
                n_out_time=n_out_time,
                freq_axis=freq_desc[:nfreq_keep],
                tsamp=tsamp,
                dm=dm,
                freq_min=freq_min,
                freq_ref=f_ref,
                normalize_channels=normalize_channels,
                series_mode=series_mode,
            )
            print(f"  writing full dedispersed dynspec to {full_path}")

        try:
            freq_batch = max(1, int(freq_batch))
            n_batches = (nfreq + freq_batch - 1) // freq_batch
            pbar = tqdm(range(n_batches), desc=f"DM={dm:g}")

            for b in pbar:
                d0 = b * freq_batch
                d1 = min(d0 + freq_batch, nfreq)

                # Map descending-frequency batch to original storage order.
                if ascending:
                    o0 = nfreq - d1
                    o1 = nfreq - d0
                else:
                    o0 = d0
                    o1 = d1

                blk = read_block(tun, chosen_packed, slice(0, ntime), slice(o0, o1)).astype(np.float64)
                if ascending:
                    blk = blk[:, ::-1]

                apply_flags_to_block(
                    blk,
                    0,
                    bad_ch_desc[d0:d1] if bad_ch_desc is not None else None,
                    block_ch_mask_desc[:, d0:d1]
                    if (block_ch_mask_desc is not None and block_ch_mask_desc.size)
                    else None,
                    block_size if block_size else None,
                    imp_mask,
                )

                if normalize_channels:
                    blk, _med, _sig = robust_normalize_channels(blk, eps=norm_eps)

                for k in range(blk.shape[1]):
                    ch_idx_desc = d0 + k
                    shift = int(delay_samples[ch_idx_desc])
                    col = blk[shift:shift + n_out_time, k]
                    valid = np.isfinite(col)

                    # 1D series uses only selected frequency range.
                    if ch_idx_desc < nfreq_keep:
                        if valid.any():
                            series_sum[valid] += col[valid]
                            series_cnt[valid] += 1.0

                        if save_full_dynspec and full_ds is not None:
                            full_ds[ch_idx_desc, :] = col.astype(np.float32)

                    # Downsampled diagnostic dynspec also only uses selected range.
                    if ch_idx_desc < nfreq_ds_full:
                        out_freq_idx = ch_idx_desc // ds_f
                        if valid.any():
                            dynspec_sum[out_freq_idx, valid] += col[valid].astype(np.float32)
                            dynspec_cnt[out_freq_idx, valid] += 1.0
                        if dynspec_hole_if_any_flagged:
                            bad = ~valid
                            if bad.any():
                                dynspec_bad[out_freq_idx, bad] += 1

                del blk

        finally:
            if full_h5 is not None:
                full_h5.close()

        with np.errstate(invalid="ignore", divide="ignore"):
            if series_mode == "sum_sqrt":
                series = np.where(series_cnt > 0, series_sum / np.sqrt(series_cnt), np.nan)
            elif series_mode == "mean":
                series = np.where(series_cnt > 0, series_sum / series_cnt, np.nan)
            else:
                raise ValueError(f"Unknown series_mode: {series_mode}")

            dynspec_ds = np.where(dynspec_cnt > 0, dynspec_sum / dynspec_cnt, np.nan)

        if dynspec_hole_if_any_flagged:
            dynspec_ds = np.where(dynspec_bad > 0, np.nan, dynspec_ds)

        t_axis = np.arange(n_out_time, dtype=np.float64) * tsamp

    return {
        "series": series.astype(np.float32),
        "series_count": series_cnt.astype(np.float32),
        "series_sum": series_sum.astype(np.float32),
        "t_axis": t_axis.astype(np.float32),
        "dynspec_ds": dynspec_ds.astype(np.float32),
        "dynspec_count_ds": dynspec_cnt.astype(np.float32),
        "dynspec_bad_count_ds": dynspec_bad.astype(np.uint16),
        "freq_axis_ds": freq_axis_ds.astype(np.float32),
        "tsamp": float(tsamp),
        "dm": float(dm),
        "freq_min": float(freq_min),
        "freq_ref": float(f_ref),
        "max_shift": int(max_shift),
        "nchan_used": int(nfreq_keep),
        "normalize_channels": bool(normalize_channels),
        "series_mode": str(series_mode),
        "dynspec_hole_if_any_flagged": bool(dynspec_hole_if_any_flagged),
        "full_dynspec_path": full_path or "",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True)
    p.add_argument("--tuning", default="Tuning2")
    p.add_argument("--pol", default="I", choices=["I", "XX", "YY"])
    p.add_argument("--i-method", default="mean", choices=["mean", "sum"])
    p.add_argument("--dm", type=float, action="append", required=True, help="DM (pc cm^-3). Repeat for multiple DMs.")
    p.add_argument("--freq-min", type=float, default=30.0, help="Minimum frequency to include in the 1D search series (MHz).")
    p.add_argument("--freq-batch", type=int, default=64)
    p.add_argument("--target-freq-out", type=int, default=512, help="Frequency bins in diagnostic dynspec_ds.")
    p.add_argument("--flags", default=None, help="Path to <base>_flags.h5; default = next to input.")
    p.add_argument("--no-flags", action="store_true", help="Ignore flag file even if present.")
    p.add_argument("--outdir", default=None)

    # FRB-search default: whiten channels before summing.
    p.add_argument("--normalize-channels", dest="normalize_channels", action="store_true", default=True)
    p.add_argument("--no-normalize-channels", dest="normalize_channels", action="store_false")
    p.add_argument("--norm-eps", type=float, default=1e-6)
    p.add_argument("--series-mode", default="sum_sqrt", choices=["sum_sqrt", "mean"],
                   help="sum_sqrt is recommended after channel normalization.")

    # Keep flagged samples as holes in diagnostic dynspec.
    p.add_argument("--dynspec-hole-if-any-flagged", dest="dynspec_hole", action="store_true", default=True)
    p.add_argument("--dynspec-average-valid", dest="dynspec_hole", action="store_false",
                   help="Average over remaining valid channels in dynspec bins instead of making holes.")

    # Optional large output.
    p.add_argument("--save-full-dynspec", action="store_true",
                   help="Also save full-resolution dedispersed dynspec to HDF5. Can be many GB per DM.")

    args = p.parse_args()

    if not os.path.isfile(args.input):
        sys.exit(f"Not a file: {args.input}")

    out_dir = ensure_dir(args.outdir or os.path.dirname(os.path.abspath(args.input)) or ".")
    base = base_no_ext(args.input)
    flag_path = None if args.no_flags else (
        args.flags or os.path.join(os.path.dirname(os.path.abspath(args.input)), f"{base}_flags.h5")
    )

    # Peek dimensions to load flags.
    with h5py.File(args.input, "r") as f:
        tun = f["Observation1"][args.tuning]
        nfreq_pk = int(tun["freq"].shape[0])
        chosen_pk = "I" if "I" in tun else ("XX" if "XX" in tun else "YY")
        ntime_pk = int(tun[chosen_pk].shape[0])

    bad_ch, block_ch_mask, block_size, imp_mask = load_flags(flag_path, ntime_pk, nfreq_pk) if flag_path else (None, None, None, None)
    if flag_path and bad_ch is not None:
        print(
            f"Loaded flags from {flag_path}: bad_ch={int(bad_ch.sum())}, "
            f"impulsive_rows={int(imp_mask.sum())}, "
            f"block_mask={block_ch_mask.shape if block_ch_mask is not None else None}, "
            f"block_size={block_size}"
        )
    else:
        print("No flag file applied.")

    flag_paths = (bad_ch, block_ch_mask, block_size, imp_mask)

    for dm in args.dm:
        out = dedisperse_one_dm(
            args.input,
            args.tuning,
            args.pol,
            args.i_method,
            dm,
            args.freq_min,
            flag_paths,
            args.freq_batch,
            args.target_freq_out,
            args.normalize_channels,
            args.norm_eps,
            args.series_mode,
            args.dynspec_hole,
            args.save_full_dynspec,
            out_dir,
            base,
        )

        out_npz = os.path.join(out_dir, f"{base}_dm{dm:g}.npz")
        np.savez_compressed(
            out_npz,
            series=out["series"],
            series_count=out["series_count"],
            series_sum=out["series_sum"],
            t_axis=out["t_axis"],
            tsamp=np.float32(out["tsamp"]),
            dynspec_ds=out["dynspec_ds"],
            dynspec_count_ds=out["dynspec_count_ds"],
            dynspec_bad_count_ds=out["dynspec_bad_count_ds"],
            freq_axis_ds=out["freq_axis_ds"],
            dm=np.float32(out["dm"]),
            freq_min=np.float32(out["freq_min"]),
            freq_ref=np.float32(out["freq_ref"]),
            max_shift=np.int64(out["max_shift"]),
            nchan_used=np.int64(out["nchan_used"]),
            normalize_channels=np.array(out["normalize_channels"]),
            series_mode=np.array(out["series_mode"]),
            dynspec_hole_if_any_flagged=np.array(out["dynspec_hole_if_any_flagged"]),
            full_dynspec_path=np.array(out["full_dynspec_path"]),
        )
        print(f"Saved {out_npz}")
        if out["full_dynspec_path"]:
            print(f"Saved full dynspec {out['full_dynspec_path']}")


if __name__ == "__main__":
    main()

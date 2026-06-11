#!/usr/bin/env python3
"""03 - Flag RFI in HDF5 dynamic spectrum and plot post-flag waterfall.

Conservative defaults for a pre-dedispersion FRB-search waterfall.

The post-flag waterfall ALWAYS shows holes for flagged data: a displayed
pixel becomes NaN if any fine-resolution sample inside that displayed
bin was flagged. This avoids averaging flagged RFI into the plot.

Flags saved to <base>_flags.h5:
  bad_ch              (nFreq,) bool
  sk_only_block_mask  (nBlocks, nFreq) bool
  power_block_mask    (nBlocks, nFreq) bool
  sk_block_mask       (nBlocks, nFreq) bool, combined block/channel mask
  imp_time_mask       (nTime,) bool

The name sk_block_mask is kept for compatibility with older dedispersion
scripts, but it contains SK OR block-power flags.

Run:
  python 03_flag_and_plot.py --input drx_61161_None_b1t2_0001.hdf5
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

try:
    from scipy.ndimage import median_filter as scipy_median_filter
except Exception:
    scipy_median_filter = None

from numpy.lib.stride_tricks import sliding_window_view

from utils import (
    base_no_ext,
    ensure_dir,
    infer_tsamp,
    pick_pol,
    get_freq_mhz,
    read_block,
)


# --------------------------------------------------------------------------
# Robust helpers
# --------------------------------------------------------------------------
def robust_sigma_mad_1d(x: np.ndarray, eps: float = 1e-12) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(1.4826 * mad + eps)


def running_median_1d(x: np.ndarray, window: int) -> np.ndarray:
    """Local running median over frequency."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n == 0:
        return x.copy()

    window = int(window)
    if window <= 1:
        return np.full_like(x, np.nanmedian(x))
    if window % 2 == 0:
        window += 1
    if window > n:
        window = n if n % 2 == 1 else n - 1
    if window < 3:
        return np.full_like(x, np.nanmedian(x))

    fill = np.nanmedian(x)
    y = np.where(np.isfinite(x), x, fill)

    if scipy_median_filter is not None:
        return scipy_median_filter(y, size=window, mode="nearest")

    pad = window // 2
    yp = np.pad(y, (pad, pad), mode="edge")
    sw = sliding_window_view(yp, window)
    return np.median(sw, axis=-1)


def robust_z_against_local_median(
    x: np.ndarray,
    window: int,
    eps: float = 1e-12,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    local = running_median_1d(x, window)
    resid = x - local
    med = np.nanmedian(resid)
    sig = robust_sigma_mad_1d(resid - med, eps=eps)
    if not np.isfinite(sig) or sig <= eps:
        sig = np.nanstd(resid) + eps
    return (resid - med) / sig


def dilate_mask_2d(mask: np.ndarray, dilate_t: int = 0, dilate_f: int = 0) -> np.ndarray:
    """Dilate boolean mask in time-block and frequency directions."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or mask.size == 0:
        return mask

    dilate_t = int(max(0, dilate_t))
    dilate_f = int(max(0, dilate_f))
    if dilate_t == 0 and dilate_f == 0:
        return mask

    nt, nf = mask.shape
    out = mask.copy()

    for dt in range(-dilate_t, dilate_t + 1):
        src_t0 = max(0, -dt)
        src_t1 = min(nt, nt - dt)
        dst_t0 = max(0, dt)
        dst_t1 = min(nt, nt + dt)

        for df in range(-dilate_f, dilate_f + 1):
            src_f0 = max(0, -df)
            src_f1 = min(nf, nf - df)
            dst_f0 = max(0, df)
            dst_f1 = min(nf, nf + df)

            out[dst_t0:dst_t1, dst_f0:dst_f1] |= mask[src_t0:src_t1, src_f0:src_f1]

    return out


def nanmean_no_warning(a: np.ndarray, axis: int) -> np.ndarray:
    finite = np.isfinite(a)
    count = finite.sum(axis=axis)
    total = np.where(finite, a, 0.0).sum(axis=axis)

    out = np.full(total.shape, np.nan, dtype=np.float64)
    np.divide(total, count, out=out, where=count > 0)
    return out


def robust_channel_normalize(img: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Subtract per-frequency median and divide by per-frequency MAD.

    Input shape is time x frequency. NaNs stay NaNs.
    """
    img = np.asarray(img, dtype=np.float64)
    out = np.full_like(img, np.nan, dtype=np.float64)

    if img.ndim != 2 or img.size == 0:
        return out.astype(np.float32)

    finite_cols = np.isfinite(img).any(axis=0)
    if not finite_cols.any():
        return out.astype(np.float32)

    med = np.full((1, img.shape[1]), np.nan, dtype=np.float64)
    med[:, finite_cols] = np.nanmedian(img[:, finite_cols], axis=0, keepdims=True)

    resid = img - med

    scale = np.full((1, img.shape[1]), np.nan, dtype=np.float64)
    scale[:, finite_cols] = np.nanmedian(np.abs(resid[:, finite_cols]), axis=0, keepdims=True)

    good = finite_cols & np.isfinite(scale[0]) & (scale[0] > eps)
    out[:, good] = resid[:, good] / scale[:, good]

    return out.astype(np.float32)


# --------------------------------------------------------------------------
# Mask application
# --------------------------------------------------------------------------
def make_cell_flag_block(
    cur_row: int,
    nrows: int,
    nfreq: int,
    bad_ch: np.ndarray | None,
    block_ch_mask: np.ndarray | None,
    block_size: int,
    imp_mask: np.ndarray | None,
) -> np.ndarray:
    """Return boolean cell mask for raw block: shape time x freq."""
    flags = np.zeros((nrows, nfreq), dtype=bool)

    if bad_ch is not None and bad_ch.size:
        bc = bad_ch[:nfreq]
        if bc.any():
            flags[:, bc] = True

    if block_ch_mask is not None and block_ch_mask.size:
        b0 = cur_row // block_size
        b1 = min(block_ch_mask.shape[0], (cur_row + nrows + block_size - 1) // block_size)

        for b in range(b0, b1):
            row_lo = max(b * block_size, cur_row) - cur_row
            row_hi = min((b + 1) * block_size, cur_row + nrows) - cur_row

            if row_hi <= row_lo:
                continue

            bad = block_ch_mask[b, :nfreq]
            if bad.any():
                flags[row_lo:row_hi, bad] = True

    if imp_mask is not None and imp_mask.size:
        ir = imp_mask[cur_row:cur_row + nrows]
        if ir.any():
            flags[ir, :] = True

    return flags


def apply_masks_to_block(
    blk: np.ndarray,
    cur_row: int,
    bad_ch: np.ndarray | None,
    block_ch_mask: np.ndarray | None,
    block_size: int,
    imp_mask: np.ndarray | None,
) -> np.ndarray:
    flags = make_cell_flag_block(
        cur_row=cur_row,
        nrows=blk.shape[0],
        nfreq=blk.shape[1],
        bad_ch=bad_ch,
        block_ch_mask=block_ch_mask,
        block_size=block_size,
        imp_mask=imp_mask,
    )
    flags |= ~np.isfinite(blk)
    blk[flags] = np.nan
    return blk


# --------------------------------------------------------------------------
# 1. Static bad channels
# --------------------------------------------------------------------------
def estimate_bad_channels(
    tun,
    chosen_packed,
    ntime: int,
    nfreq: int,
    sample_rows: int = 1024,
    n_passes: int = 8,
    occupy_z: float = 8.0,
    occupy_frac: float = 0.5,
    majority: float = 0.5,
    static_z: float = 8.0,
    scatter_z: float = 8.0,
    freq_window: int = 401,
    rng_seed: int = 12345,
) -> np.ndarray:
    sample_rows = int(min(max(2, sample_rows), ntime))
    n_passes = int(max(1, n_passes))

    rng = np.random.default_rng(rng_seed)
    max_start = max(0, ntime - sample_rows)
    starts = rng.integers(0, max_start + 1, size=n_passes)

    votes = np.zeros(nfreq, dtype=np.int32)

    for s in starts:
        s = int(s)

        blk = read_block(
            tun,
            chosen_packed,
            slice(s, s + sample_rows),
            slice(0, nfreq),
        ).astype(np.float32, copy=False)

        blk = np.where(np.isfinite(blk), blk, np.nan)

        ch_med = np.nanmedian(blk, axis=0)
        ch_mad = np.nanmedian(np.abs(blk - ch_med[None, :]), axis=0) + 1e-6

        z_time = np.abs(blk - ch_med[None, :]) / (1.4826 * ch_mad[None, :])
        hot_frac = np.nanmean(z_time > occupy_z, axis=0)
        occupy_bad = hot_frac > occupy_frac

        z_static = robust_z_against_local_median(ch_med, freq_window)
        static_bad = np.abs(z_static) > static_z

        z_scatter = robust_z_against_local_median(np.log(ch_mad + 1e-12), freq_window)
        scatter_bad = np.abs(z_scatter) > scatter_z

        invalid_bad = ~np.isfinite(ch_med) | ~np.isfinite(ch_mad)

        pass_bad = occupy_bad | static_bad | scatter_bad | invalid_bad
        votes += pass_bad.astype(np.int32)

    needed = max(1, int(np.ceil(float(majority) * n_passes)))
    return votes >= needed


# --------------------------------------------------------------------------
# 2. Empirical SK block/channel mask
# --------------------------------------------------------------------------
def compute_sk_values(
    tun,
    chosen_packed,
    ntime: int,
    nfreq: int,
    block_size: int = 128,
    read_chunk_blocks: int = 4,
) -> tuple[np.ndarray, int]:
    M = max(8, int(block_size))
    nblocks = ntime // M

    if nblocks == 0:
        return np.zeros((0, nfreq), dtype=np.float32), M

    read_chunk_blocks = max(1, int(read_chunk_blocks))
    sk_all = np.full((nblocks, nfreq), np.nan, dtype=np.float32)

    cur_block = 0
    cur_row = 0

    while cur_block < nblocks:
        nb = min(read_chunk_blocks, nblocks - cur_block)
        rows = nb * M

        blk = read_block(
            tun,
            chosen_packed,
            slice(cur_row, cur_row + rows),
            slice(0, nfreq),
        ).astype(np.float64, copy=False)

        x = blk.reshape(nb, M, nfreq)
        finite = np.isfinite(x)
        x = np.where(finite, x, np.nan)

        N = finite.sum(axis=1).astype(np.float64)
        S1 = np.nansum(x, axis=1)
        S2 = np.nansum(x * x, axis=1)

        with np.errstate(invalid="ignore", divide="ignore"):
            factor = (N + 1.0) / np.maximum(N - 1.0, 1.0)
            sk = factor * (N * S2 / np.maximum(S1 * S1, 1e-30) - 1.0)

        sk[N < 8] = np.nan
        sk_all[cur_block:cur_block + nb] = sk.astype(np.float32)

        cur_block += nb
        cur_row += rows

    return sk_all, M


def make_empirical_sk_mask(
    tun,
    chosen_packed,
    ntime: int,
    nfreq: int,
    block_size: int = 128,
    read_chunk_blocks: int = 4,
    z_thresh: float = 8.0,
    max_frac: float = 0.25,
    allow_heavy: bool = False,
) -> tuple[np.ndarray, int, dict]:
    sk, M = compute_sk_values(
        tun,
        chosen_packed,
        ntime,
        nfreq,
        block_size=block_size,
        read_chunk_blocks=read_chunk_blocks,
    )

    stats = {
        "sk_frac": 0.0,
        "disabled_by_safety": False,
    }

    if sk.size == 0:
        return np.zeros_like(sk, dtype=bool), M, stats

    mask = np.zeros_like(sk, dtype=bool)

    finite_cols = np.isfinite(sk).any(axis=0)
    if finite_cols.any():
        center = np.full((1, nfreq), np.nan, dtype=np.float64)
        center[:, finite_cols] = np.nanmedian(sk[:, finite_cols], axis=0, keepdims=True)

        resid = sk.astype(np.float64) - center

        scale = np.full((1, nfreq), np.nan, dtype=np.float64)
        scale[:, finite_cols] = 1.4826 * np.nanmedian(
            np.abs(resid[:, finite_cols]),
            axis=0,
            keepdims=True,
        )

        good_scales = scale[np.isfinite(scale) & (scale > 1e-12)]
        fallback = float(np.nanmedian(good_scales)) if good_scales.size else 1.0
        scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, fallback)

        z = resid / scale
        mask = np.abs(z) > z_thresh

    frac = float(mask.mean()) if mask.size else 0.0
    stats["sk_frac"] = frac

    if frac > max_frac and not allow_heavy:
        print(
            f"  WARNING: empirical SK wanted to flag {100 * frac:.2f}% of "
            f"block/channel cells. Disabling SK. Use --allow-heavy-sk to override."
        )
        mask[:] = False
        stats["disabled_by_safety"] = True
        stats["sk_frac"] = 0.0

    return mask.astype(bool), M, stats


# --------------------------------------------------------------------------
# 3. Block-power narrowband RFI mask
# --------------------------------------------------------------------------
def make_block_power_mask(
    tun,
    chosen_packed,
    ntime: int,
    nfreq: int,
    block_size: int = 128,
    read_chunk_blocks: int = 4,
    freq_window: int = 401,
    z_thresh: float = 5.0,
    two_sided: bool = False,
    max_frac_per_block: float = 0.05,
    dilate_t: int = 1,
    dilate_f: int = 2,
) -> tuple[np.ndarray, int, dict]:
    """Flag intermittent narrowband power excesses.

    This mainly catches horizontal narrowband RFI. It is intentionally not used
    to flag broad features across a large fraction of the band.
    """
    M = max(8, int(block_size))
    nblocks = ntime // M

    stats = {
        "nblocks": nblocks,
        "raw_power_frac": 0.0,
        "power_frac": 0.0,
        "disabled_blocks": 0,
    }

    if nblocks == 0:
        return np.zeros((0, nfreq), dtype=bool), M, stats

    read_chunk_blocks = max(1, int(read_chunk_blocks))
    mask = np.zeros((nblocks, nfreq), dtype=bool)

    cur_block = 0
    cur_row = 0

    while cur_block < nblocks:
        nb = min(read_chunk_blocks, nblocks - cur_block)
        rows = nb * M

        blk = read_block(
            tun,
            chosen_packed,
            slice(cur_row, cur_row + rows),
            slice(0, nfreq),
        ).astype(np.float64, copy=False)

        blk = np.where(np.isfinite(blk), blk, np.nan)
        x = blk.reshape(nb, M, nfreq)

        specs = np.nanmedian(x, axis=1)

        for j in range(nb):
            spec = specs[j]

            if not np.isfinite(spec).any():
                continue

            local = running_median_1d(spec, freq_window)
            resid = spec - local

            med = np.nanmedian(resid)
            sig = robust_sigma_mad_1d(resid - med, eps=1e-9)

            if not np.isfinite(sig) or sig <= 0:
                continue

            z = (resid - med) / sig

            if two_sided:
                bad = np.abs(z) > z_thresh
            else:
                bad = z > z_thresh

            frac = float(np.mean(bad))

            if frac > max_frac_per_block:
                bad[:] = False
                stats["disabled_blocks"] += 1

            mask[cur_block + j] = bad

        cur_block += nb
        cur_row += rows

    stats["raw_power_frac"] = float(mask.mean()) if mask.size else 0.0

    if dilate_t > 0 or dilate_f > 0:
        mask = dilate_mask_2d(mask, dilate_t=dilate_t, dilate_f=dilate_f)

    stats["power_frac"] = float(mask.mean()) if mask.size else 0.0

    return mask, M, stats


# --------------------------------------------------------------------------
# 4. Conservative time-row / vertical-artifact mask
# --------------------------------------------------------------------------
def make_impulsive_time_mask(
    tun,
    chosen_packed,
    ntime: int,
    nfreq: int,
    bad_ch: np.ndarray,
    block_ch_mask: np.ndarray,
    block_size: int,
    z_thresh: float = 8.0,
    time_chunk: int = 1024,
    min_valid_frac: float = 0.2,
    pixel_z: float = 7.0,
    pixel_frac: float = 0.05,
    median_abs_z: float = 6.0,
    mean_abs_z: float = 3.0,
) -> np.ndarray:
    """Flag broadband/vertical bad time rows conservatively.

    For FRB search we avoid aggressive zero-DM clipping. This only flags rows
    that are broadband/partial-band outliers across many channels, such as
    acquisition dropouts, gain jumps, and obvious vertical RFI.
    """
    out = np.zeros(ntime, dtype=bool)
    time_chunk = max(1, int(time_chunk))

    cur = 0
    while cur < ntime:
        cnt = min(time_chunk, ntime - cur)

        blk = read_block(
            tun,
            chosen_packed,
            slice(cur, cur + cnt),
            slice(0, nfreq),
        ).astype(np.float32, copy=False)

        blk = apply_masks_to_block(
            blk,
            cur_row=cur,
            bad_ch=bad_ch,
            block_ch_mask=block_ch_mask,
            block_size=block_size,
            imp_mask=None,
        )

        valid_frac = np.isfinite(blk).mean(axis=1)
        out_chunk = valid_frac < min_valid_frac

        good_rows = valid_frac >= min_valid_frac
        if good_rows.sum() < 8:
            out[cur:cur + cnt] = out_chunk
            cur += cnt
            continue

        # Metric 1: band median. Uses abs() so it catches negative dropouts too.
        band = np.full(cnt, np.nan, dtype=np.float64)
        band[good_rows] = np.nanmedian(blk[good_rows], axis=1)

        valid_band = np.isfinite(band)
        if valid_band.sum() >= 8:
            mu = np.nanmedian(band[valid_band])
            sig = robust_sigma_mad_1d(band[valid_band] - mu, eps=1e-9)

            if np.isfinite(sig) and sig > 0:
                z_band = (band - mu) / sig
                out_chunk[valid_band] |= np.abs(z_band[valid_band]) > z_thresh

        # Metric 2: row-wise distribution after per-channel normalization.
        # This catches vertical artifacts whose median is not extreme.
        ch_med = np.nanmedian(blk, axis=0)
        ch_mad = np.nanmedian(np.abs(blk - ch_med[None, :]), axis=0)

        ch_sig = 1.4826 * ch_mad
        good_ch = np.isfinite(ch_med) & np.isfinite(ch_sig) & (ch_sig > 1e-6)

        if good_ch.sum() >= 16:
            z = np.full_like(blk, np.nan, dtype=np.float32)
            z[:, good_ch] = (
                (blk[:, good_ch] - ch_med[None, good_ch])
                / ch_sig[None, good_ch]
            )

            abs_z = np.abs(z)

            row_med_z = np.nanmedian(z, axis=1)
            row_mean_abs_z = np.nanmean(abs_z, axis=1)
            row_frac_bad = np.nanmean(abs_z > pixel_z, axis=1)

            out_chunk |= np.isfinite(row_med_z) & (np.abs(row_med_z) > median_abs_z)
            out_chunk |= np.isfinite(row_mean_abs_z) & (row_mean_abs_z > mean_abs_z)
            out_chunk |= np.isfinite(row_frac_bad) & (row_frac_bad > pixel_frac)

        out[cur:cur + cnt] = out_chunk
        cur += cnt

    return out


# --------------------------------------------------------------------------
# Post-flag waterfall with mandatory holes
# --------------------------------------------------------------------------
def make_post_flag_waterfall_with_holes(
    tun,
    chosen_packed,
    freq_mhz: np.ndarray,
    ntime: int,
    nfreq: int,
    tsamp: float,
    bad_ch: np.ndarray,
    block_ch_mask: np.ndarray,
    block_size: int,
    imp_mask: np.ndarray,
    target_t: int,
    target_f: int,
    time_chunk: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Create post-flag waterfall.

    Mandatory behavior: a display pixel is NaN if ANY fine cell inside that
    display bin was flagged.
    """
    target_t = max(1, int(target_t))
    target_f = max(1, int(target_f))
    time_chunk = max(1, int(time_chunk))

    ds_t = max(1, int(np.ceil(ntime / target_t)))
    ds_f = max(1, int(np.ceil(nfreq / target_f)))

    nfreq_ds = (nfreq // ds_f) * ds_f
    nfreq_out = nfreq_ds // ds_f
    ntime_out = ntime // ds_t

    img = np.full((ntime_out, nfreq_out), np.nan, dtype=np.float32)
    hole = np.ones((ntime_out, nfreq_out), dtype=bool)

    cur = 0
    out_row = 0

    val_buf = None
    flag_buf = None

    chunk = max(time_chunk, ds_t * 4)

    while cur < ntime and out_row < ntime_out:
        cnt = min(chunk, ntime - cur)

        blk = read_block(
            tun,
            chosen_packed,
            slice(cur, cur + cnt),
            slice(0, nfreq_ds),
        ).astype(np.float32, copy=False)

        flags = make_cell_flag_block(
            cur_row=cur,
            nrows=cnt,
            nfreq=nfreq_ds,
            bad_ch=bad_ch[:nfreq_ds],
            block_ch_mask=block_ch_mask[:, :nfreq_ds] if block_ch_mask.size else block_ch_mask,
            block_size=block_size,
            imp_mask=imp_mask,
        )

        flags |= ~np.isfinite(blk)

        # Frequency downsample. If any fine channel in the displayed frequency
        # bin is flagged, the displayed time/frequency cell becomes a hole.
        blk3 = blk.reshape(cnt, nfreq_out, ds_f)
        flg3 = flags.reshape(cnt, nfreq_out, ds_f)

        freq_hole = flg3.any(axis=2)
        freq_val = nanmean_no_warning(np.where(flg3, np.nan, blk3), axis=2).astype(np.float32)
        freq_val[freq_hole] = np.nan

        val_buf = freq_val if val_buf is None else np.vstack([val_buf, freq_val])
        flag_buf = freq_hole if flag_buf is None else np.vstack([flag_buf, freq_hole])

        full = val_buf.shape[0] // ds_t

        if full > 0:
            use = full * ds_t

            val_slab = val_buf[:use].reshape(full, ds_t, nfreq_out)
            flag_slab = flag_buf[:use].reshape(full, ds_t, nfreq_out)

            # Time downsample. If any fine time sample in the displayed time bin
            # was flagged, the final displayed pixel becomes a hole.
            time_hole = flag_slab.any(axis=1)
            out_val = nanmean_no_warning(val_slab, axis=1).astype(np.float32)
            out_val[time_hole] = np.nan

            end = min(out_row + full, ntime_out)

            img[out_row:end] = out_val[: end - out_row]
            hole[out_row:end] = time_hole[: end - out_row]

            out_row = end

            val_buf = val_buf[use:]
            flag_buf = flag_buf[use:]

        cur += cnt

    img_norm = robust_channel_normalize(img)

    freq_axis = freq_mhz[:nfreq_ds].reshape(-1, ds_f).mean(axis=1)
    t_axis = (np.arange(ntime_out) * ds_t + 0.5 * ds_t) * tsamp

    stats = {
        "ds_t": int(ds_t),
        "ds_f": int(ds_f),
        "ntime_out": int(ntime_out),
        "nfreq_out": int(nfreq_out),
        "hole_fraction": float(hole.mean()) if hole.size else 0.0,
        "finite_fraction_raw_downsampled": float(np.isfinite(img).mean()) if img.size else 0.0,
        "finite_fraction_normalized": float(np.isfinite(img_norm).mean()) if img_norm.size else 0.0,
    }

    return img_norm, t_axis, freq_axis, stats


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument("--input", required=True)
    p.add_argument("--tuning", default="Tuning2")
    p.add_argument("--pol", default="I", choices=["I", "XX", "YY"])
    p.add_argument("--i-method", default="mean", choices=["mean", "sum"])

    # Static bad channels. Conservative.
    p.add_argument("--badch-sample-rows", type=int, default=1024)
    p.add_argument("--badch-passes", type=int, default=8)
    p.add_argument("--badch-occupy-z", type=float, default=8.0)
    p.add_argument("--badch-occupy-frac", type=float, default=0.5)
    p.add_argument("--badch-majority", type=float, default=0.5)
    p.add_argument("--badch-static-z", type=float, default=8.0)
    p.add_argument("--badch-scatter-z", type=float, default=8.0)
    p.add_argument("--badch-freq-window", type=int, default=401)
    p.add_argument("--max-badch-frac", type=float, default=0.20)
    p.add_argument("--allow-heavy-badch", action="store_true")

    # Shared block settings. With tsamp~1.89 ms, block 128 is ~0.24 s.
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--read-chunk-blocks", type=int, default=4)

    # Empirical SK. Conservative and centered empirically, not at SK=1.
    p.add_argument("--sk-mode", default="empirical", choices=["empirical", "off"])
    p.add_argument("--sk-z", type=float, default=8.0)
    p.add_argument("--max-sk-frac", type=float, default=0.25)
    p.add_argument("--allow-heavy-sk", action="store_true")

    # Narrowband horizontal RFI. Main layer for the visible horizontal lines.
    p.add_argument("--power-mode", default="on", choices=["on", "off"])
    p.add_argument("--power-z", type=float, default=5.0)
    p.add_argument("--power-freq-window", type=int, default=401)
    p.add_argument("--power-max-frac-per-block", type=float, default=0.05)
    p.add_argument("--power-dilate-t", type=int, default=1)
    p.add_argument("--power-dilate-f", type=int, default=2)
    p.add_argument("--power-two-sided", action="store_true")

    # Conservative vertical/broadband row flags. These are intentionally not
    # too aggressive for FRB work.
    p.add_argument("--imp-z", type=float, default=8.0)
    p.add_argument("--imp-pixel-z", type=float, default=7.0)
    p.add_argument("--imp-pixel-frac", type=float, default=0.05)
    p.add_argument("--imp-median-abs-z", type=float, default=6.0)
    p.add_argument("--imp-mean-abs-z", type=float, default=3.0)
    p.add_argument("--min-row-valid-frac", type=float, default=0.2)
    p.add_argument("--max-imp-frac", type=float, default=0.02)
    p.add_argument("--allow-heavy-imp", action="store_true")

    # Plot / IO.
    p.add_argument("--time-chunk", type=int, default=1024)
    p.add_argument("--outdir", default=None)
    p.add_argument("--target-time", type=int, default=2000)
    p.add_argument("--target-freq", type=int, default=512)
    p.add_argument("--cmap", default="inferno")
    p.add_argument("--vmin", type=float, default=-5.0)
    p.add_argument("--vmax", type=float, default=5.0)

    args = p.parse_args()

    if not os.path.isfile(args.input):
        sys.exit(f"Not a file: {args.input}")

    base = base_no_ext(args.input)
    out_dir = ensure_dir(args.outdir or os.path.dirname(os.path.abspath(args.input)) or ".")

    with h5py.File(args.input, "r") as f:
        obs = f["Observation1"]
        tun = obs[args.tuning]

        freq_mhz = get_freq_mhz(tun)
        nfreq = int(freq_mhz.size)

        chosen, i_meth = pick_pol(tun, pol=args.pol, i_method=args.i_method)
        chosen_packed = (chosen, i_meth) if isinstance(chosen, tuple) else chosen

        tsamp = infer_tsamp(obs)

        if isinstance(chosen, tuple):
            ntime = int(tun[chosen[0]].shape[0])
        else:
            ntime = int(tun[chosen].shape[0])

        print(f"ntime={ntime:,}  nfreq={nfreq:,}  tsamp={tsamp:.9g}s")
        print(f"tuning={args.tuning}  pol={args.pol}  chosen={chosen}")

        print("[1/4] Estimating static bad channels...")
        bad_ch = estimate_bad_channels(
            tun,
            chosen_packed,
            ntime,
            nfreq,
            sample_rows=args.badch_sample_rows,
            n_passes=args.badch_passes,
            occupy_z=args.badch_occupy_z,
            occupy_frac=args.badch_occupy_frac,
            majority=args.badch_majority,
            static_z=args.badch_static_z,
            scatter_z=args.badch_scatter_z,
            freq_window=args.badch_freq_window,
        )

        badch_frac = float(bad_ch.mean()) if bad_ch.size else 0.0

        if badch_frac > args.max_badch_frac and not args.allow_heavy_badch:
            print(
                f"  WARNING: bad-channel mask wanted to flag "
                f"{100 * badch_frac:.2f}% of channels. Disabling. "
                f"Use --allow-heavy-badch to override."
            )
            bad_ch[:] = False
            badch_frac = 0.0

        print(
            f"  bad channels: {int(bad_ch.sum())}/{nfreq} "
            f"({100 * badch_frac:.6f}%)"
        )

        print("[2/4] Making block/channel masks...")

        M = max(8, int(args.block_size))

        if args.sk_mode == "empirical":
            sk_only_mask, M, sk_stats = make_empirical_sk_mask(
                tun,
                chosen_packed,
                ntime,
                nfreq,
                block_size=args.block_size,
                read_chunk_blocks=args.read_chunk_blocks,
                z_thresh=args.sk_z,
                max_frac=args.max_sk_frac,
                allow_heavy=args.allow_heavy_sk,
            )
        else:
            sk_only_mask = np.zeros((ntime // M, nfreq), dtype=bool)
            sk_stats = {"sk_frac": 0.0, "disabled_by_safety": False}

        print(f"  SK-only flag rate: {100 * sk_only_mask.mean():.6f}%")

        if args.power_mode == "on":
            power_mask, M2, power_stats = make_block_power_mask(
                tun,
                chosen_packed,
                ntime,
                nfreq,
                block_size=args.block_size,
                read_chunk_blocks=args.read_chunk_blocks,
                freq_window=args.power_freq_window,
                z_thresh=args.power_z,
                two_sided=args.power_two_sided,
                max_frac_per_block=args.power_max_frac_per_block,
                dilate_t=args.power_dilate_t,
                dilate_f=args.power_dilate_f,
            )

            if M2 != M:
                raise RuntimeError("SK and power masks used different block sizes.")

        else:
            power_mask = np.zeros_like(sk_only_mask)
            power_stats = {
                "raw_power_frac": 0.0,
                "power_frac": 0.0,
                "disabled_blocks": 0,
            }

        print(
            f"  power-mask flag rate: {100 * power_mask.mean():.6f}% "
            f"(disabled_blocks={power_stats['disabled_blocks']})"
        )

        block_ch_mask = sk_only_mask | power_mask

        print(
            f"  combined block/channel flag rate: "
            f"{100 * block_ch_mask.mean():.6f}%"
        )

        print("[3/4] Making conservative impulsive/vertical time mask...")
        imp_mask = make_impulsive_time_mask(
            tun,
            chosen_packed,
            ntime,
            nfreq,
            bad_ch=bad_ch,
            block_ch_mask=block_ch_mask,
            block_size=M,
            z_thresh=args.imp_z,
            time_chunk=args.time_chunk,
            min_valid_frac=args.min_row_valid_frac,
            pixel_z=args.imp_pixel_z,
            pixel_frac=args.imp_pixel_frac,
            median_abs_z=args.imp_median_abs_z,
            mean_abs_z=args.imp_mean_abs_z,
        )

        imp_frac = float(imp_mask.mean()) if imp_mask.size else 0.0

        if imp_frac > args.max_imp_frac and not args.allow_heavy_imp:
            print(
                f"  WARNING: impulsive/vertical mask wanted to flag "
                f"{100 * imp_frac:.2f}% of time rows. Disabling time mask to avoid "
                f"over-flagging. Use --allow-heavy-imp to override."
            )
            imp_mask[:] = False
            imp_frac = 0.0

        print(
            f"  impulsive/vertical rows: {int(imp_mask.sum())}/{ntime} "
            f"({100 * imp_frac:.6f}%)"
        )

        print("[4/4] Saving flags...")
        flag_path = os.path.join(out_dir, f"{base}_flags.h5")

        with h5py.File(flag_path, "w") as fo:
            fo.create_dataset("bad_ch", data=bad_ch, compression="gzip")
            fo.create_dataset("sk_only_block_mask", data=sk_only_mask, compression="gzip")
            fo.create_dataset("power_block_mask", data=power_mask, compression="gzip")

            # Compatibility name: combined block/channel mask.
            fo.create_dataset("sk_block_mask", data=block_ch_mask, compression="gzip")

            fo.create_dataset("imp_time_mask", data=imp_mask, compression="gzip")
            fo.create_dataset("freq_mhz", data=freq_mhz.astype(np.float32), compression="gzip")

            fo.attrs["sk_block_size"] = int(M)
            fo.attrs["block_size"] = int(M)
            fo.attrs["tsamp"] = float(tsamp)
            fo.attrs["ntime"] = int(ntime)
            fo.attrs["nfreq"] = int(nfreq)
            fo.attrs["tuning"] = args.tuning
            fo.attrs["pol"] = args.pol

            fo.attrs["sk_mode"] = args.sk_mode
            fo.attrs["sk_z"] = float(args.sk_z)

            fo.attrs["power_mode"] = args.power_mode
            fo.attrs["power_z"] = float(args.power_z)
            fo.attrs["power_freq_window"] = int(args.power_freq_window)
            fo.attrs["power_dilate_t"] = int(args.power_dilate_t)
            fo.attrs["power_dilate_f"] = int(args.power_dilate_f)

            fo.attrs["imp_z"] = float(args.imp_z)
            fo.attrs["imp_pixel_z"] = float(args.imp_pixel_z)
            fo.attrs["imp_pixel_frac"] = float(args.imp_pixel_frac)
            fo.attrs["imp_median_abs_z"] = float(args.imp_median_abs_z)
            fo.attrs["imp_mean_abs_z"] = float(args.imp_mean_abs_z)

            fo.attrs["plot_policy"] = "hole_if_any_fine_sample_flagged"

        print(f"Saved flags to {flag_path}")

        print("Making post-flag waterfall with mandatory holes...")
        img, t_axis, freq_axis, plot_stats = make_post_flag_waterfall_with_holes(
            tun,
            chosen_packed,
            freq_mhz=freq_mhz,
            ntime=ntime,
            nfreq=nfreq,
            tsamp=tsamp,
            bad_ch=bad_ch,
            block_ch_mask=block_ch_mask,
            block_size=M,
            imp_mask=imp_mask,
            target_t=args.target_time,
            target_f=args.target_freq,
            time_chunk=args.time_chunk,
        )

        print(
            f"  display image shape time x freq: {img.shape}; "
            f"ds_t={plot_stats['ds_t']} ds_f={plot_stats['ds_f']}"
        )
        print(f"  display hole fraction: {100 * plot_stats['hole_fraction']:.6f}%")
        print(
            f"  finite fraction before display normalization: "
            f"{100 * plot_stats['finite_fraction_raw_downsampled']:.6f}%"
        )
        print(
            f"  finite fraction after display normalization:  "
            f"{100 * plot_stats['finite_fraction_normalized']:.6f}%"
        )

        if np.isfinite(img).any():
            pct = np.nanpercentile(img, [0, 1, 5, 50, 95, 99, 100])
            print("  display percentiles:", pct)
        else:
            print("  WARNING: post-flag image contains no finite pixels.")

        cmap = plt.get_cmap(args.cmap).copy()
        cmap.set_bad("0.75")  # gray holes = flagged/no value

        plt.figure(figsize=(12, 6))
        plt.imshow(
            np.ma.masked_invalid(img.T),
            aspect="auto",
            origin="lower",
            extent=[t_axis[0], t_axis[-1], freq_axis.min(), freq_axis.max()],
            cmap=cmap,
            vmin=args.vmin,
            vmax=args.vmax,
        )
        plt.colorbar(label="(post-flag) baseline-removed (MAD units)")
        plt.xlabel("Time since file start (s)")
        plt.ylabel("Frequency (MHz)")
        plt.title(f"Post-flagging waterfall with holes — {base} ({args.tuning}, pol={args.pol})")
        plt.tight_layout()

        out_png = os.path.join(out_dir, f"{base}_post_flag_waterfall.png")
        plt.savefig(out_png, dpi=150)
        plt.close()

        print(f"Saved {out_png}")

        # Diagnostics.
        fig, ax = plt.subplots(4, 1, figsize=(12, 8), sharex=False)

        ax[0].plot(freq_mhz, bad_ch.astype(float), lw=0.5)
        ax[0].set_ylim(-0.05, 1.05)
        ax[0].set_ylabel("Bad ch")
        ax[0].set_title(f"Flagging diagnostics — {base}")

        t_block = (np.arange(block_ch_mask.shape[0]) + 0.5) * M * tsamp

        if sk_only_mask.size:
            ax[1].plot(t_block, sk_only_mask.mean(axis=1), lw=0.8)
        ax[1].set_ylabel("SK frac")
        ax[1].set_ylim(bottom=0)

        if power_mask.size:
            ax[2].plot(t_block, power_mask.mean(axis=1), lw=0.8)
        ax[2].set_ylabel("Power frac")
        ax[2].set_ylim(bottom=0)

        ax[3].plot(np.arange(ntime) * tsamp, imp_mask.astype(float), lw=0.5)
        ax[3].set_ylabel("Vertical")
        ax[3].set_xlabel("Time since file start (s)")
        ax[3].set_ylim(-0.05, 1.05)

        plt.tight_layout()
        diag_png = os.path.join(out_dir, f"{base}_flag_diagnostics.png")
        plt.savefig(diag_png, dpi=140)
        plt.close()

        print(f"Saved {diag_png}")


if __name__ == "__main__":
    main()
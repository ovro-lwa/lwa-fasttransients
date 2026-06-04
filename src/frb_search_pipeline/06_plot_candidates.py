#!/usr/bin/env python3
"""06 - Per-candidate zoom plots.

Reads the .npz from 04_dedisperse_cpu.py and the candidates CSV from
05_detrend_and_plot.py, and produces one zoom panel per candidate showing:

  top    : detrended dedispersed series with the boxcar matched-filter window
           highlighted, and the +/- N-sigma threshold lines.
  bottom : the dedispersed dynamic spectrum (display-normalized) zoomed
           around the candidate time.

This is the easiest way to eyeball whether a high-SNR candidate is a real
broadband burst or just baseline structure / a few noisy channels.

Example
-------
    python 06_plot_candidates.py \
        --npz drx_61161_None_b1t2_0001_dm134.252.npz \
        --csv drx_61161_None_b1t2_0001_dm134.252_candidates.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import robust_sigma_mad, base_no_ext, ensure_dir


def rolling_nanmean(y: np.ndarray, valid: np.ndarray, width_bins: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(y)
    width_bins = max(1, int(width_bins))
    if width_bins <= 1:
        return np.where(valid, y, np.nan)
    kernel = np.ones(width_bins, dtype=np.float64)
    yy = np.where(valid, y, 0.0)
    ww = valid.astype(np.float64)
    total = np.convolve(yy, kernel, mode="same")
    count = np.convolve(ww, kernel, mode="same")
    out = np.full(y.shape, np.nan, dtype=np.float64)
    np.divide(total, count, out=out, where=count > 0)
    return out


def boxcar_snr(y: np.ndarray, valid: np.ndarray, width_bins: int, sigma: float,
               min_valid_frac: float = 0.8) -> np.ndarray:
    """Matched-filter S/N curve at fixed boxcar width.

    Matches step 05: SNR = sum / (sigma * sqrt(N_valid)), masked where the
    boxcar does not have enough valid samples.
    """
    y = np.asarray(y, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(y)
    width_bins = max(1, int(width_bins))
    kernel = np.ones(width_bins, dtype=np.float64)
    yy = np.where(valid, y, 0.0)
    ww = valid.astype(np.float64)
    summed = np.convolve(yy, kernel, mode="same")
    count = np.convolve(ww, kernel, mode="same")
    min_valid = max(1.0, float(min_valid_frac) * width_bins)
    snr = np.full(y.shape, np.nan, dtype=np.float64)
    good = count >= min_valid
    np.divide(summed, max(float(sigma), 1e-12) * np.sqrt(np.maximum(count, 1.0)),
              out=snr, where=good)
    snr[~good] = np.nan
    return snr


def robust_freq_normalize_for_display(dyn: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Per-frequency baseline removal and MAD scaling. Shape: freq x time."""
    dyn = np.asarray(dyn, dtype=np.float64)
    out = np.full_like(dyn, np.nan, dtype=np.float64)
    if dyn.ndim != 2 or dyn.size == 0:
        return out.astype(np.float32)
    good_rows = np.isfinite(dyn).any(axis=1)
    if not good_rows.any():
        return out.astype(np.float32)
    med = np.full((dyn.shape[0], 1), np.nan, dtype=np.float64)
    med[good_rows, :] = np.nanmedian(dyn[good_rows, :], axis=1, keepdims=True)
    resid = dyn - med
    mad = np.full((dyn.shape[0], 1), np.nan, dtype=np.float64)
    mad[good_rows, :] = np.nanmedian(np.abs(resid[good_rows, :]), axis=1, keepdims=True)
    good = good_rows & np.isfinite(mad[:, 0]) & (mad[:, 0] > eps)
    out[good, :] = resid[good, :] / mad[good, :]
    return out.astype(np.float32)


def read_csv(csv_path: str) -> list[dict]:
    out = []
    with open(csv_path, "r", newline="") as fd:
        for row in csv.DictReader(fd):
            out.append(row)
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--npz", required=True, help=".npz from 04_dedisperse_cpu.py")
    p.add_argument("--csv", required=True, help="_candidates.csv from 05_detrend_and_plot.py")
    p.add_argument("--top", type=int, default=10, help="Plot at most this many candidates.")
    p.add_argument("--zoom-windows", type=float, default=8.0,
                   help="Time half-window = this many boxcar widths around the peak.")
    p.add_argument("--min-zoom-s", type=float, default=0.5,
                   help="Minimum half-window in seconds (so very short widths still show context).")
    p.add_argument("--detrend-width", type=float, default=20.0,
                   help="Same rolling-mean baseline as step 05.")
    p.add_argument("--min-coverage-frac", type=float, default=0.8,
                   help="Same coverage gate as step 05.")
    p.add_argument("--snr-line", type=float, default=7.0,
                   help="Show +/- this sigma threshold in the series panel.")
    p.add_argument("--dyn-vmin", type=float, default=-5.0)
    p.add_argument("--dyn-vmax", type=float, default=5.0)
    p.add_argument("--outdir", default=None)

    args = p.parse_args()

    if not os.path.isfile(args.npz):
        sys.exit(f"Not a file: {args.npz}")
    if not os.path.isfile(args.csv):
        sys.exit(f"Not a file: {args.csv}")

    out_dir = ensure_dir(args.outdir or os.path.dirname(os.path.abspath(args.npz)) or ".")
    base = base_no_ext(args.npz)

    with np.load(args.npz, allow_pickle=False) as z:
        series = z["series"].astype(np.float64)
        t_axis = z["t_axis"].astype(np.float64)
        tsamp = float(z["tsamp"])
        dynspec = z["dynspec_ds"].astype(np.float32)
        freq_axis = z["freq_axis_ds"].astype(np.float32)
        dm = float(z["dm"])
        if "series_count" in z.files:
            series_count = z["series_count"].astype(np.float64)
        else:
            series_count = np.where(np.isfinite(series), 1.0, 0.0)

    # Reproduce step-05 detrend so the +/- sigma lines match.
    nonzero = series_count[np.isfinite(series_count) & (series_count > 0)]
    count_ref = float(np.nanmedian(nonzero)) if nonzero.size else 1.0
    min_count = float(args.min_coverage_frac) * count_ref
    finite_raw = np.isfinite(series)
    valid = finite_raw & (np.isfinite(series_count) & (series_count >= min_count))

    detrend_bins = max(1, int(round(float(args.detrend_width) / tsamp)))
    detrend_bins = min(detrend_bins, max(3, series.size // 2))
    base_line = rolling_nanmean(series, valid, detrend_bins)
    global_med = float(np.nanmedian(series[valid])) if valid.any() else 0.0
    base_line = np.where(np.isfinite(base_line), base_line, global_med)
    series_hp = series - base_line
    series_hp[~valid] = np.nan

    sigma = robust_sigma_mad(series_hp[valid]) if valid.any() else np.nan
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(np.nanstd(series_hp[valid])) if valid.any() else 1.0
    sigma = max(float(sigma), 1e-12)

    # Display-normalized dynspec, oriented ascending in frequency.
    if freq_axis[0] > freq_axis[-1]:
        freq_plot = freq_axis[::-1].astype(np.float64)
        dyn_plot = dynspec[::-1]
    else:
        freq_plot = freq_axis.astype(np.float64)
        dyn_plot = dynspec
    dyn_norm = robust_freq_normalize_for_display(dyn_plot)

    cands = read_csv(args.csv)
    if not cands:
        print("No candidates in CSV.")
        return
    cands = cands[: max(1, int(args.top))]

    cmap = plt.get_cmap("inferno").copy()
    cmap.set_bad("0.75")

    n_t = t_axis.size

    for c in cands:
        rank = int(c["rank"])
        t_c = float(c["time_s"])
        w_s = float(c["width_s"])
        w_bins = int(c["width_bins"])
        snr = float(c["snr"])
        nch = float(c["series_count"])

        half = max(float(args.min_zoom_s), float(args.zoom_windows) * w_s)
        t_lo = t_c - half
        t_hi = t_c + half

        i_lo = int(np.searchsorted(t_axis, t_lo, side="left"))
        i_hi = int(np.searchsorted(t_axis, t_hi, side="right"))
        i_lo = max(0, i_lo)
        i_hi = min(n_t, i_hi)
        if i_hi - i_lo < 4:
            continue

        # Boxcar window edges around the peak.
        i_peak = int(np.searchsorted(t_axis, t_c))
        i_peak = max(0, min(n_t - 1, i_peak))
        b_lo = max(0, i_peak - w_bins // 2)
        b_hi = min(n_t, b_lo + w_bins)
        t_win_lo = float(t_axis[b_lo])
        t_win_hi = float(t_axis[min(n_t - 1, b_hi - 1)])

        fig, (ax_t, ax_s, ax_w) = plt.subplots(
            3, 1, figsize=(10, 7.5), sharex=True,
            gridspec_kw={"height_ratios": [1.0, 1.0, 2.2], "hspace": 0.05},
            constrained_layout=True,
        )

        ax_t.plot(t_axis[i_lo:i_hi], series_hp[i_lo:i_hi], lw=0.6, color="black")
        ax_t.axvspan(t_win_lo, t_win_hi, color="tab:blue", alpha=0.2)
        ax_t.axvline(t_c, color="tab:red", lw=0.6, ls=":")
        ax_t.set_ylabel("detrended\nseries")
        ax_t.set_title(
            f"Candidate #{rank}  DM={dm:.5f}  t={t_c:.4f}s  "
            f"W={w_s:g}s  S/N={snr:.2f}  Nchan={nch:.0f}"
        )

        # Matched-filter SNR curve at this candidate's boxcar width.
        snr_curve = boxcar_snr(series_hp, valid, w_bins, sigma)
        ax_s.plot(t_axis[i_lo:i_hi], snr_curve[i_lo:i_hi], lw=0.8, color="tab:purple",
                  label=f"boxcar S/N (W={w_s:g}s)")
        ax_s.axhline(args.snr_line, color="r", lw=0.6, ls="--",
                     label=f"+{args.snr_line:g}\u03c3")
        ax_s.axhline(-args.snr_line, color="r", lw=0.6, ls="--")
        ax_s.axhline(0.0, color="0.5", lw=0.4)
        ax_s.axvspan(t_win_lo, t_win_hi, color="tab:blue", alpha=0.2)
        ax_s.axvline(t_c, color="tab:red", lw=0.6, ls=":")
        ax_s.set_ylabel("matched-filter\nS/N (\u03c3)")
        ax_s.legend(loc="upper right", fontsize=8)

        # Map zoom window to dynspec column indices.
        # dynspec time axis matches t_axis exactly (same n_out_time, same tsamp).
        j_lo = i_lo
        j_hi = i_hi
        sub = dyn_norm[:, j_lo:j_hi]

        t_edges = np.empty(j_hi - j_lo + 1, dtype=np.float64)
        t_edges[:-1] = t_axis[j_lo:j_hi] - 0.5 * tsamp
        t_edges[-1] = t_axis[j_hi - 1] + 0.5 * tsamp

        # Frequency edges.
        f = freq_plot
        if f.size >= 2:
            df = np.diff(f)
            f_edges = np.empty(f.size + 1, dtype=np.float64)
            f_edges[1:-1] = 0.5 * (f[:-1] + f[1:])
            f_edges[0] = f[0] - 0.5 * df[0]
            f_edges[-1] = f[-1] + 0.5 * df[-1]
        else:
            f_edges = np.array([f[0] - 0.5, f[0] + 0.5])

        mesh = ax_w.pcolormesh(
            t_edges, f_edges,
            np.ma.masked_invalid(sub),
            shading="auto", cmap=cmap,
            vmin=args.dyn_vmin, vmax=args.dyn_vmax,
        )
        ax_w.axvspan(t_win_lo, t_win_hi, color="cyan", alpha=0.15)
        ax_w.axvline(t_c, color="cyan", lw=0.6, ls=":")
        ax_w.set_xlabel("Time since start (s)")
        ax_w.set_ylabel("Frequency (MHz)")
        fig.colorbar(mesh, ax=ax_w, label="display amplitude (MAD units)")

        ax_t.set_xlim(float(t_edges[0]), float(t_edges[-1]))

        out_png = os.path.join(
            out_dir,
            f"{base}_cand{rank:02d}_t{t_c:.3f}s_W{w_s:g}s.png",
        )
        plt.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"Saved {out_png}")


if __name__ == "__main__":
    main()

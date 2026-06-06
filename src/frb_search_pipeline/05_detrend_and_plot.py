#!/usr/bin/env python3
"""05 - Detrend dedispersed FRB time series, find candidates, and plot.

Reads the .npz produced by 04_dedisperse_cpu.py. Expected arrays:

    series        1-D dedispersed search series
    series_count  number of valid channels contributing per time sample
    t_axis        time axis in seconds
    tsamp         sample time in seconds
    dynspec_ds    frequency-downsampled dedispersed dynamic spectrum
    freq_axis_ds  frequency axis for dynspec_ds
    dm            dedispersion DM
    freq_min      minimum frequency used in the search series

This script:
  1. Masks low-coverage samples using series_count.
  2. Removes a slow rolling baseline from the dedispersed series.
  3. Estimates robust noise with MAD.
  4. Runs simple gap-aware boxcar matched filtering.
  5. Saves candidates to CSV and prints the strongest candidates.
  6. Saves an aligned diagnostic plot.

Example
-------
python 05_detrend_and_plot.py --input drx_61161_None_b1t2_0001_dm108.372.npz
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


def nanmean_no_warning(a: np.ndarray, axis: int) -> np.ndarray:
    """np.nanmean without RuntimeWarning for all-NaN slices."""
    a = np.asarray(a)
    finite = np.isfinite(a)
    count = finite.sum(axis=axis)
    total = np.where(finite, a, 0.0).sum(axis=axis)

    out = np.full(total.shape, np.nan, dtype=np.float64)
    np.divide(total, count, out=out, where=count > 0)
    return out


def rolling_nanmean(y: np.ndarray, valid: np.ndarray, width_bins: int) -> np.ndarray:
    """Gap-aware rolling mean using convolution.

    Invalid samples do not contribute to the mean.
    """
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


def boxcar_sum_and_count(
    y: np.ndarray,
    valid: np.ndarray,
    width_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Gap-aware boxcar sum and valid-sample count."""
    y = np.asarray(y, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(y)
    width_bins = max(1, int(width_bins))

    kernel = np.ones(width_bins, dtype=np.float64)
    yy = np.where(valid, y, 0.0)
    ww = valid.astype(np.float64)

    summed = np.convolve(yy, kernel, mode="same")
    count = np.convolve(ww, kernel, mode="same")
    return summed, count


def boxcar_average(y: np.ndarray, valid: np.ndarray, width_bins: int) -> np.ndarray:
    """Gap-aware boxcar average, for display only."""
    summed, count = boxcar_sum_and_count(y, valid, width_bins)

    out = np.full_like(summed, np.nan, dtype=np.float64)
    np.divide(summed, count, out=out, where=count > 0)
    return out


def find_peaks(x: np.ndarray, thr: float, min_sep: int) -> np.ndarray:
    """Simple local-maximum peak finder."""
    out: list[int] = []
    n = x.size
    i = 1
    min_sep = max(1, int(min_sep))

    while i < n - 1:
        if (
            np.isfinite(x[i])
            and x[i] > thr
            and x[i] >= x[i - 1]
            and x[i] >= x[i + 1]
        ):
            out.append(i)
            i += min_sep
        else:
            i += 1

    return np.array(out, dtype=np.int64)


def centers_to_edges(x: np.ndarray) -> np.ndarray:
    """Convert monotonic bin centers to bin edges."""
    x = np.asarray(x, dtype=np.float64)

    if x.size == 0:
        return np.array([0.0, 1.0], dtype=np.float64)

    if x.size == 1:
        return np.array([x[0] - 0.5, x[0] + 0.5], dtype=np.float64)

    dx = np.diff(x)

    edges = np.empty(x.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (x[:-1] + x[1:])
    edges[0] = x[0] - 0.5 * dx[0]
    edges[-1] = x[-1] + 0.5 * dx[-1]

    return edges


def downsample_dynspec_time(
    dynspec: np.ndarray,
    t_axis: np.ndarray,
    tsamp: float,
    target_cols: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Downsample dynspec in time and return (dyn_ds, t_edges).

    Uses real time edges, not centers, so pcolormesh aligns with the 1-D panels.
    dynspec shape is frequency x time.
    """
    dynspec = np.asarray(dynspec, dtype=np.float32)

    if dynspec.ndim != 2:
        raise ValueError("dynspec must be 2-D, shape frequency x time")

    nfreq, ntime = dynspec.shape
    target_cols = max(1, int(target_cols))

    if ntime == 0:
        return dynspec, np.array([0.0, 1.0], dtype=np.float64)

    bin_size = max(1, int(np.ceil(ntime / target_cols)))
    nblocks = int(np.ceil(ntime / bin_size))
    pad = nblocks * bin_size - ntime

    if pad:
        dyn_pad = np.pad(
            dynspec,
            ((0, 0), (0, pad)),
            mode="constant",
            constant_values=np.nan,
        )
    else:
        dyn_pad = dynspec

    dyn_ds = nanmean_no_warning(
        dyn_pad.reshape(nfreq, nblocks, bin_size),
        axis=2,
    ).astype(np.float32)

    # t_axis values are sample centers. Edges are half a sample before/after.
    t0_edge = float(t_axis[0]) - 0.5 * float(tsamp) if t_axis.size else -0.5 * float(tsamp)

    edge_samples = np.arange(nblocks + 1, dtype=np.float64) * bin_size
    edge_samples[-1] = ntime

    t_edges = t0_edge + edge_samples * float(tsamp)

    return dyn_ds, t_edges


def robust_freq_normalize_for_display(dyn: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Display-only per-frequency normalization.

    Input shape: frequency x time.

    For each frequency row:
      subtract median over time
      divide by MAD over time

    This makes the dedispersed waterfall display comparable to your raw and
    post-flag waterfalls. It does NOT affect the candidate search.
    """
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
    mad[good_rows, :] = np.nanmedian(
        np.abs(resid[good_rows, :]),
        axis=1,
        keepdims=True,
    )

    good = good_rows & np.isfinite(mad[:, 0]) & (mad[:, 0] > eps)
    out[good, :] = resid[good, :] / mad[good, :]

    return out.astype(np.float32)


def write_candidates_csv(
    csv_path: str,
    cands: list[dict],
    tsamp: float,
    dm: float,
    freq_min: float,
) -> None:
    """Write candidate list to CSV."""
    fieldnames = [
        "rank",
        "time_s",
        "width_s",
        "width_bins",
        "snr",
        "time_bin",
        "series",
        "detrended",
        "series_count",
        "coverage_frac",
        "tsamp_s",
        "dm",
        "freq_min_MHz",
    ]

    with open(csv_path, "w", newline="") as fd:
        writer = csv.DictWriter(fd, fieldnames=fieldnames)
        writer.writeheader()

        for rank, c in enumerate(cands, start=1):
            writer.writerow({
                "rank": rank,
                "time_s": f"{c['time_s']:.6f}",
                "width_s": f"{c['width_s']:.6f}",
                "width_bins": int(c["width_bins"]),
                "snr": f"{c['snr']:.3f}",
                "time_bin": int(c["time_bin"]),
                "series": f"{c['series']:.8g}",
                "detrended": f"{c['detrended']:.8g}",
                "series_count": f"{c['series_count']:.3f}",
                "coverage_frac": f"{c['coverage_frac']:.6f}",
                "tsamp_s": f"{tsamp:.9g}",
                "dm": f"{dm:.6f}",
                "freq_min_MHz": f"{freq_min:.3f}",
            })


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument("--input", required=True, help=".npz from 04_dedisperse_cpu.py")

    p.add_argument(
        "--detrend-width",
        type=float,
        default=20.0,
        help="Rolling-mean baseline window in seconds. Automatically capped for short files.",
    )

    p.add_argument(
        "--widths",
        type=float,
        nargs="+",
        default=[0.005, 0.01, 0.02, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
        help="Boxcar matched-filter widths in seconds.",
    )

    p.add_argument("--snr", type=float, default=7.0)
    p.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Maximum candidates kept per width before final sorting.",
    )
    p.add_argument(
        "--print-top",
        type=int,
        default=10,
        help="Print this many strongest candidates.",
    )

    p.add_argument(
        "--target-cols",
        type=int,
        default=3000,
        help="Maximum approximate number of waterfall time columns.",
    )

    p.add_argument(
        "--min-coverage-frac",
        type=float,
        default=0.8,
        help="Mask samples with series_count below this fraction of median nonzero coverage.",
    )

    p.add_argument(
        "--min-boxcar-valid-frac",
        type=float,
        default=0.8,
        help="For boxcar S/N, require this fraction of samples in the boxcar to be valid.",
    )

    p.add_argument(
        "--smooth-widths",
        type=float,
        nargs="+",
        default=[0.1, 0.5, 5.0],
        help="Boxcar-average curves shown in the detrended-series panel.",
    )

    p.add_argument(
        "--dyn-display-normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Display dynspec in per-frequency baseline-removed MAD units.",
    )

    p.add_argument("--dyn-vmin", type=float, default=-5.0)
    p.add_argument("--dyn-vmax", type=float, default=5.0)

    p.add_argument("--hide-nchan", action="store_true", help="Hide valid-channel-count panel.")
    p.add_argument("--outdir", default=None)

    args = p.parse_args()

    if not os.path.isfile(args.input):
        sys.exit(f"Not a file: {args.input}")

    out_dir = ensure_dir(args.outdir or os.path.dirname(os.path.abspath(args.input)) or ".")
    base = base_no_ext(args.input)

    with np.load(args.input, allow_pickle=False) as z:
        series = z["series"].astype(np.float64)
        t_axis = z["t_axis"].astype(np.float64)
        tsamp = float(z["tsamp"])
        dynspec = z["dynspec_ds"].astype(np.float32)
        freq_axis = z["freq_axis_ds"].astype(np.float32)
        dm = float(z["dm"])
        freq_min = float(z["freq_min"])

        if "series_count" in z.files:
            series_count = z["series_count"].astype(np.float64)
        else:
            series_count = np.where(np.isfinite(series), 1.0, 0.0)

    if series.size == 0:
        sys.exit("Empty series in input npz.")

    if t_axis.size != series.size:
        sys.exit("t_axis and series lengths do not match.")

    if series_count.size != series.size:
        sys.exit("series_count and series lengths do not match.")

    if dynspec.ndim != 2:
        sys.exit("dynspec_ds must be a 2-D array.")

    # ------------------------------------------------------------------
    # Coverage mask
    # ------------------------------------------------------------------
    nonzero_count = series_count[np.isfinite(series_count) & (series_count > 0)]
    if nonzero_count.size:
        count_ref = float(np.nanmedian(nonzero_count))
    else:
        count_ref = 1.0

    min_count = float(args.min_coverage_frac) * count_ref
    coverage_frac = series_count / max(count_ref, 1e-12)

    finite_raw = np.isfinite(series)
    good_coverage = np.isfinite(series_count) & (series_count >= min_count)
    search_valid = finite_raw & good_coverage

    if search_valid.sum() < 8:
        sys.exit("Too few finite/high-coverage samples for detrending/search.")

    # ------------------------------------------------------------------
    # Detrend
    # ------------------------------------------------------------------
    requested_detrend_bins = max(1, int(round(float(args.detrend_width) / tsamp)))

    # Avoid a detrending kernel longer than useful data.
    max_detrend_bins = max(3, series.size // 2)
    detrend_bins = min(requested_detrend_bins, max_detrend_bins)
    detrend_width_eff = detrend_bins * tsamp

    base_line = rolling_nanmean(series, search_valid, detrend_bins)

    global_med = float(np.nanmedian(series[search_valid]))
    base_line = np.where(np.isfinite(base_line), base_line, global_med)

    series_hp = series - base_line
    series_hp[~search_valid] = np.nan

    sigma = robust_sigma_mad(series_hp[search_valid])
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(np.nanstd(series_hp[search_valid]))
    sigma = max(float(sigma), 1e-12)

    print(
        f"DM={dm:.6f}  tsamp={tsamp:.9g}s  "
        f"detrend={detrend_width_eff:.3f}s ({detrend_bins} bins)  "
        f"sigma(MAD)={sigma:.6g}"
    )
    print(
        f"coverage: median={count_ref:.1f} channels, "
        f"min_used={min_count:.1f}, valid_search={100 * search_valid.mean():.3f}%"
    )

    # ------------------------------------------------------------------
    # Candidate finder
    # ------------------------------------------------------------------
    cands: list[dict] = []

    for w_s in args.widths:
        w_bins = max(1, int(round(float(w_s) / tsamp)))

        summed, nvalid = boxcar_sum_and_count(series_hp, search_valid, w_bins)
        min_valid_in_boxcar = max(1.0, float(args.min_boxcar_valid_frac) * w_bins)

        snr = np.full(series.size, np.nan, dtype=np.float64)
        good = nvalid >= min_valid_in_boxcar

        np.divide(
            summed,
            sigma * np.sqrt(np.maximum(nvalid, 1.0)),
            out=snr,
            where=good,
        )
        snr[~good] = np.nan

        peaks = find_peaks(snr, args.snr, min_sep=max(1, w_bins // 2))

        if peaks.size:
            order = np.argsort(snr[peaks])[::-1][: max(1, int(args.top_k))]

            for i in peaks[order]:
                idx = int(i)

                cands.append({
                    "time_s": float(t_axis[idx]),
                    "width_s": float(w_s),
                    "width_bins": int(w_bins),
                    "snr": float(snr[idx]),
                    "time_bin": idx,
                    "series": float(series[idx]) if np.isfinite(series[idx]) else np.nan,
                    "detrended": float(series_hp[idx]) if np.isfinite(series_hp[idx]) else np.nan,
                    "series_count": float(series_count[idx]) if np.isfinite(series_count[idx]) else np.nan,
                    "coverage_frac": float(coverage_frac[idx]) if np.isfinite(coverage_frac[idx]) else np.nan,
                })

    cands.sort(key=lambda r: r["snr"], reverse=True)

    csv_path = os.path.join(out_dir, f"{base}_candidates.csv")
    write_candidates_csv(csv_path, cands, tsamp=tsamp, dm=dm, freq_min=freq_min)

    print(f"Saved {csv_path}  ({len(cands)} candidates >= {args.snr:.1f} sigma)")

    if cands:
        print("Top candidates:")
        for rank, c in enumerate(cands[: max(0, int(args.print_top))], start=1):
            print(
                f"  {rank:2d}: t={c['time_s']:.6f}s  "
                f"width={c['width_s']:.6f}s  S/N={c['snr']:.2f}  "
                f"Nchan={c['series_count']:.0f}"
            )
    else:
        print("No candidates above threshold.")

    # ------------------------------------------------------------------
    # Waterfall display
    # ------------------------------------------------------------------
    dyn_ds, t_edges = downsample_dynspec_time(
        dynspec=dynspec,
        t_axis=t_axis,
        tsamp=tsamp,
        target_cols=args.target_cols,
    )

    # Display orientation: ascending frequency.
    if freq_axis[0] > freq_axis[-1]:
        freq_plot = freq_axis[::-1].astype(np.float64)
        dyn_plot = dyn_ds[::-1]
    else:
        freq_plot = freq_axis.astype(np.float64)
        dyn_plot = dyn_ds

    freq_edges = centers_to_edges(freq_plot)

    if args.dyn_display_normalize:
        dyn_plot_display = robust_freq_normalize_for_display(dyn_plot)
        dyn_label = "dedispersed display amplitude (MAD units)"
        vmin = float(args.dyn_vmin)
        vmax = float(args.dyn_vmax)
    else:
        dyn_plot_display = dyn_plot
        dyn_label = "dedispersed normalized amplitude"

        if np.isfinite(dyn_plot_display).any():
            vmin = float(np.nanpercentile(dyn_plot_display, 5))
            vmax = float(np.nanpercentile(dyn_plot_display, 95))
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                vmin, vmax = None, None
        else:
            vmin, vmax = None, None

    # Smoothed curves for display only.
    smooth_curves = []
    for ws in args.smooth_widths:
        wb = max(1, int(round(float(ws) / tsamp)))
        smooth_curves.append((float(ws), boxcar_average(series_hp, search_valid, wb)))

    # ------------------------------------------------------------------
    # Plot with physically aligned axes.
    #
    # All science axes are in GridSpec column 0.
    # The colorbar gets its own column, so it does not shrink only ax_w.
    # ------------------------------------------------------------------
    if args.hide_nchan:
        fig = plt.figure(figsize=(13, 9), constrained_layout=True)
        gs = fig.add_gridspec(
            3,
            2,
            width_ratios=[1.0, 0.025],
            height_ratios=[1, 1, 3],
            hspace=0.08,
            wspace=0.04,
        )

        ax_t = fig.add_subplot(gs[0, 0])
        ax_h = fig.add_subplot(gs[1, 0], sharex=ax_t)
        ax_w = fig.add_subplot(gs[2, 0], sharex=ax_t)
        cax = fig.add_subplot(gs[2, 1])
        ax_n = None

    else:
        fig = plt.figure(figsize=(13, 10), constrained_layout=True)
        gs = fig.add_gridspec(
            4,
            2,
            width_ratios=[1.0, 0.025],
            height_ratios=[1, 1, 0.65, 3],
            hspace=0.08,
            wspace=0.04,
        )

        ax_t = fig.add_subplot(gs[0, 0])
        ax_h = fig.add_subplot(gs[1, 0], sharex=ax_t)
        ax_n = fig.add_subplot(gs[2, 0], sharex=ax_t)
        ax_w = fig.add_subplot(gs[3, 0], sharex=ax_t)
        cax = fig.add_subplot(gs[3, 1])

    # Top panel.
    ax_t.plot(t_axis, series, lw=0.5, label="dedispersed series")
    ax_t.plot(t_axis, base_line, lw=0.8, label="rolling baseline")
    ax_t.set_ylabel("series")
    ax_t.legend(loc="upper right", fontsize=8)
    ax_t.set_title(
        f"Dedispersed | DM={dm:.5f} pc cm$^{{-3}}$ | "
        f"tsamp={tsamp:.6g}s | freq>={freq_min:g} MHz"
    )

    # Detrended panel.
    ax_h.plot(t_axis, series_hp, lw=0.45, label="detrended")

    for ws, yy in smooth_curves:
        ax_h.plot(t_axis, yy, lw=0.75, label=f"boxcar {ws:g}s")

    ax_h.axhline(args.snr * sigma, color="r", lw=0.6, ls="--", label=f"+{args.snr:g}σ")
    ax_h.axhline(-args.snr * sigma, color="r", lw=0.6, ls="--")
    ax_h.set_ylabel("detrended")
    ax_h.legend(loc="upper right", fontsize=8, ncol=2)

    # Nchan panel.
    if ax_n is not None:
        ax_n.plot(t_axis, series_count, lw=0.6, label="valid channels")
        ax_n.axhline(min_count, color="r", lw=0.6, ls="--", label="min coverage")
        ax_n.set_ylabel("Nchan")
        ax_n.legend(loc="upper right", fontsize=8)

    # Waterfall panel.
    cmap = plt.get_cmap("inferno").copy()
    cmap.set_bad("0.75")

    mesh = ax_w.pcolormesh(
        t_edges,
        freq_edges,
        np.ma.masked_invalid(dyn_plot_display),
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    fig.colorbar(mesh, cax=cax).set_label(dyn_label)

    ax_w.set_xlabel("Time since start / reference-frequency arrival (s)")
    ax_w.set_ylabel("Frequency (MHz)")

    # Exact shared x-limits. Since all panels are in the same GridSpec column,
    # this now visually aligns as well as data-aligns.
    ax_t.set_xlim(float(t_edges[0]), float(t_edges[-1]))

    # Hide upper x tick labels.
    upper_axes = [ax_t, ax_h] if ax_n is None else [ax_t, ax_h, ax_n]
    for ax in upper_axes:
        plt.setp(ax.get_xticklabels(), visible=False)

    out_png = os.path.join(out_dir, f"{base}_dedispersed.png")
    plt.savefig(out_png, dpi=160)
    plt.close(fig)

    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
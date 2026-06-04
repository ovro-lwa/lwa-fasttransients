"""Shared helpers for the FRB-search pipeline.

Kept deliberately small. Imported by 02..05 scripts.
"""
from __future__ import annotations

import os
import numpy as np
import h5py


# Dispersion constant for delay in seconds when frequencies are in MHz:
#   t[s] = K_DM * DM * (1/f_MHz^2 - 1/f_ref_MHz^2)
K_DM = 4148.808  # s * MHz^2 * cm^3 / pc


# --------------------------------------------------------------------------
# HDF5 metadata
# --------------------------------------------------------------------------
def infer_tsamp(obs_grp) -> float:
    """Robustly determine the sampling time (s) from /Observation1."""
    tsamp_attr = obs_grp.attrs.get("tInt", None)
    tds = obs_grp["time"]
    n = len(tds)
    if n < 2:
        if tsamp_attr is None:
            raise ValueError("Cannot determine tsamp (no timestamps and no tInt).")
        return float(tsamp_attr)

    names = getattr(tds.dtype, "names", None)
    if names and {"int", "frac"}.issubset(names):
        t = tds["int"].astype(np.float64) + tds["frac"].astype(np.float64)
    elif names and {"tv_sec", "tv_nsec"}.issubset(names):
        t = tds["tv_sec"].astype(np.float64) + tds["tv_nsec"].astype(np.float64) * 1e-9
    elif names and {"sec", "usec"}.issubset(names):
        t = tds["sec"].astype(np.float64) + tds["usec"].astype(np.float64) * 1e-6
    elif names:
        a, b = names[:2]
        t = tds[a].astype(np.float64) + tds[b].astype(np.float64)
    else:
        t = tds[:].astype(np.float64)

    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size:
        return float(np.median(dt))
    if tsamp_attr is not None:
        return float(tsamp_attr)
    raise ValueError("Cannot determine tsamp.")


def pick_pol(tuning_grp, pol: str = "I", i_method: str = "mean"):
    """Return ('I',None) | ('XX',None) | ('YY',None) | (('XX','YY'), 'mean'/'sum')."""
    have = set(tuning_grp.keys())
    pol = pol.upper()
    if pol == "I":
        if "I" in have:
            return "I", None
        if "XX" in have and "YY" in have:
            return ("XX", "YY"), i_method
        if "XX" in have:
            return "XX", None
        if "YY" in have:
            return "YY", None
    if pol in have:
        return pol, None
    raise KeyError(f"No suitable polarization for '{pol}'. Available: {sorted(have)}")


def read_block(tuning_grp, chosen, t_slice, f_slice):
    """Read [t_slice, f_slice] of intensity for the chosen polarization.

    Returns float32 ndarray of shape (nT, nF).
    """
    if isinstance(chosen, tuple):
        key_a, key_b = chosen[0]
        method = chosen[1]
        a = tuning_grp[key_a][t_slice, f_slice]
        b = tuning_grp[key_b][t_slice, f_slice]
        block = a + b if method == "sum" else 0.5 * (a + b)
    else:
        block = tuning_grp[chosen][t_slice, f_slice]
    return np.asarray(block, dtype=np.float32)


# --------------------------------------------------------------------------
# Helpers for chosen-polarization handling (the 'tuple' case carries metadata)
# --------------------------------------------------------------------------
def normalize_chosen(chosen, i_method):
    """Pack (chosen, i_method) into the convention used by read_block."""
    if isinstance(chosen, tuple):
        return (chosen, i_method or "mean")
    return chosen


# --------------------------------------------------------------------------
# Frequency / dispersion
# --------------------------------------------------------------------------
def get_freq_mhz(tuning_grp) -> np.ndarray:
    f = tuning_grp["freq"][:]
    return (f / 1e6).astype(np.float64)


def dm_delay_seconds(freq_mhz: np.ndarray, freq_ref_mhz: float, dm: float) -> np.ndarray:
    """Per-channel delay in seconds relative to `freq_ref_mhz` (highest freq).

    Positive delay = channel arrives LATER than the reference (lower freqs).
    """
    return K_DM * float(dm) * (1.0 / (freq_mhz ** 2) - 1.0 / (float(freq_ref_mhz) ** 2))


# --------------------------------------------------------------------------
# Robust statistics
# --------------------------------------------------------------------------
def robust_sigma_mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    return float(1.4826 * mad)


def boxcar_mean(y: np.ndarray, w_bins: int) -> np.ndarray:
    if w_bins <= 1:
        return y.astype(np.float64, copy=True)
    k = np.ones(int(w_bins), dtype=np.float64) / float(w_bins)
    return np.convolve(y, k, mode="same")


# --------------------------------------------------------------------------
# Misc
# --------------------------------------------------------------------------
def ensure_dir(path: str) -> str:
    if path:
        os.makedirs(path, exist_ok=True)
    return path


def base_no_ext(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def open_h5_meta(h5_path: str, tuning_name: str, pol: str = "I", i_method: str = "mean"):
    """Open and return common metadata as a dict (file is closed before return)."""
    with h5py.File(h5_path, "r") as f:
        obs = f["Observation1"]
        tun = obs[tuning_name]
        freq_mhz = get_freq_mhz(tun)
        ascending = bool(freq_mhz[0] < freq_mhz[-1])
        chosen, i_meth = pick_pol(tun, pol=pol, i_method=i_method)
        tsamp = infer_tsamp(obs)
        if isinstance(chosen, tuple):
            ntime = tun[chosen[0]].shape[0]
        else:
            ntime = tun[chosen].shape[0]
        nfreq = freq_mhz.size
    return {
        "h5_path": h5_path,
        "tuning_name": tuning_name,
        "freq_mhz": freq_mhz,                # in original storage order
        "ascending": ascending,
        "chosen": chosen,
        "i_method": i_meth,
        "tsamp": float(tsamp),
        "ntime": int(ntime),
        "nfreq": int(nfreq),
    }

#!/usr/bin/env python3
# Analogous to https://github.com/thepetabyteproject/your/blob/main/your/utils/rfi.py#L120 but for hdf files
import sys
import h5py
import numpy as np
import logging
from scipy import stats
from scipy.signal import savgol_filter as sg

# Configure logging for detailed progress output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# RFI Filtering Functions (units in Hz and seconds)
def savgol_filter_rfi(bandpass, channel_bandwidth, frequency_window=28000.0, sigma=5.0):
    """
    Apply a Savitzky–Golay filter to a 1D bandpass.

    Parameters:
        bandpass (np.ndarray): 1D array representing the bandpass (summed over time).
        channel_bandwidth (float): Channel bandwidth in Hz.
        frequency_window (float): Frequency window in Hz over which to smooth.
        sigma (float): Sigma multiplier for flagging outliers.

    Returns:
        np.ndarray: Boolean mask of the same length as bandpass, with True indicating an RFI-flagged channel.
    """
    # Compute window length in channels from the frequency window (Hz)
    window = int(np.ceil(frequency_window / abs(channel_bandwidth)))
    if window % 2 == 0:
        window += 1  # Ensure window is odd

    if window < 41:
        logger.warning("Computed window size (%d) is less than 41 channels. Forcing minimum window size 41.", window)
        window = 41
    if window > len(bandpass):
        logger.warning("Window size (%d) is larger than available channels (%d). Adjusting window size.", window, len(bandpass))
        window = len(bandpass) if len(bandpass) % 2 == 1 else len(bandpass) - 1

    logger.info("Savitzky–Golay filter window length: %d channels", window)
    # Apply the filter with polynomial order 2
    smoothed = sg(bandpass, window, 2)
    residual = bandpass - smoothed
    cutoff = sigma * np.std(residual)
    logger.info("Savitzky–Golay cutoff: %f", cutoff)
    mask = np.abs(residual) > cutoff
    return mask

def spectral_kurtosis(data, N=1, d=None):
    """
    Compute the spectral kurtosis for each frequency channel.

    Parameters:
        data (np.ndarray): 2D array (time x frequency).
        N (int): Number of accumulations.
        d (float): Optional shape factor. If None, it is estimated.

    Returns:
        np.ndarray: 1D array with the spectral kurtosis for each channel.
    """
    zero_mask = data == 0
    masked_data = np.ma.array(data.astype(float), mask=zero_mask)
    S1 = masked_data.sum(axis=0)
    S2 = (masked_data**2).sum(axis=0)
    M = data.shape[0]
    if d is None:
        d = (np.nanmean(data) / np.nanstd(data)) ** 2
    sk = ((M * d * N) + 1) * ((M * S2 / (S1**2)) - 1) / (M - 1)
    return sk.filled(np.nan)

def calc_N(channel_bandwidth, tsamp):
    """
    Calculate the number of accumulations.

    Parameters:
        channel_bandwidth (float): Channel bandwidth in Hz.
        tsamp (float): Sampling interval in seconds.

    Returns:
        int: Number of accumulations.
    """
    tn = abs(1 / channel_bandwidth)
    N = int(np.round(tsamp / tn))
    N = max(20, int(np.round(tsamp / tn))) 
    logger.info("Calculated number of accumulations (N): %d", N)
    return N

def sk_filter(data, channel_bandwidth, tsamp, sigma=6.0, d=None):
    """
    Apply the spectral kurtosis filter to the 2D data array.

    Parameters:
        data (np.ndarray): 2D array (time x frequency).
        channel_bandwidth (float): Channel bandwidth in Hz.
        tsamp (float): Sampling interval in seconds.
        sigma (float): Sigma multiplier for flagging channels.
        d (float): Optional shape factor.

    Returns:
        np.ndarray: Boolean mask (True indicates a flagged channel).
    """
    N = calc_N(channel_bandwidth, tsamp)
    sk = spectral_kurtosis(data, N=N, d=d)
    nan_mask = np.isnan(sk)
    if np.any(nan_mask):
        sk[nan_mask] = np.nanmedian(sk)
    std = 1.4826 * stats.median_abs_deviation(sk)
    if std == 0:
        std = np.std(sk[~nan_mask])
    median_val = np.median(sk)
    upper = median_val + sigma * std
    lower = median_val - sigma * std
    logger.info("Spectral kurtosis median: %f, std (MAD): %f", median_val, std)
    logger.info("Spectral kurtosis thresholds: lower=%f, upper=%f", lower, upper)
    mask = (sk < lower) | (sk > upper)
    flagged_channels = np.where(mask)[0]
    logger.info("Spectral kurtosis flagged %d channels.", len(flagged_channels))
    return mask

def sk_sg_filter(data, channel_bandwidth, tsamp,
                 spectral_kurtosis_sigma=6.0,
                 savgol_frequency_window=28000.0,
                 savgol_sigma=5.0):
    """
    Apply both spectral kurtosis and Savitzky–Golay filters.

    Parameters:
        data (np.ndarray): 2D array (time x frequency).
        channel_bandwidth (float): Channel bandwidth in Hz.
        tsamp (float): Sampling interval in seconds.
        spectral_kurtosis_sigma (float): Sigma for the spectral kurtosis filter.
        savgol_frequency_window (float): Frequency window in Hz for the Savitzky–Golay filter.
        savgol_sigma (float): Sigma for the Savitzky–Golay filter.

    Returns:
        np.ndarray: Boolean mask (True for RFI-flagged channels).
    """
    if spectral_kurtosis_sigma <= 0 and savgol_sigma <= 0:
        raise ValueError("At least one of spectral_kurtosis_sigma or savgol_sigma must be positive.")

    mask = np.zeros(data.shape[1], dtype=bool)

    if spectral_kurtosis_sigma > 0:
        logger.info("Applying spectral kurtosis filter with sigma = %s", spectral_kurtosis_sigma)
        sk_mask = sk_filter(data, channel_bandwidth, tsamp, sigma=spectral_kurtosis_sigma)
        mask[sk_mask] = True

    if savgol_sigma > 0:
        good_channels = ~mask
        logger.info("Number of channels available for Savitzky–Golay filtering: %d", np.sum(good_channels))
        if np.sum(good_channels) < 5:
            logger.warning("Too few channels remain after SK filtering for SG filtering.")
            sg_mask = np.zeros(np.sum(good_channels), dtype=bool)
        else:
            bandpass = data[:, good_channels].sum(axis=0)
            logger.info("Bandpass computed for Savitzky–Golay filtering. Bandpass length: %d", len(bandpass))
            sg_mask = savgol_filter_rfi(bandpass, channel_bandwidth, frequency_window=savgol_frequency_window, sigma=savgol_sigma)
        indices = np.where(good_channels)[0]
        mask[indices[sg_mask]] = True

    total_flagged = np.sum(mask)
    logger.info("Total flagged channels after both filters: %d", total_flagged)
    return mask

# ------------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python find_rfi.py <HDF_filename> [nspectra]")
        sys.exit(1)

    hdf_filename = sys.argv[1]
    nspectra = int(sys.argv[2]) if len(sys.argv) > 2 else 8192

    logger.info("Opening HDF file: %s", hdf_filename)
    with h5py.File(hdf_filename, 'r') as f:
        # Log HDF5 structure summary
        logger.info("HDF5 structure:")
        f.visititems(lambda name, obj: logger.info("  %s : %s", name, obj))

        intensity_path = "Observation1/Tuning1/I"
        if intensity_path not in f:
            raise ValueError(f"Dataset {intensity_path} not found in the file.")
        dset = f[intensity_path]
        logger.info("Intensity dataset shape: %s", dset.shape)
        data = dset[:nspectra, :]
        logger.info("Extracted %d time samples.", data.shape[0])

        freq_path = "Observation1/Tuning1/freq"
        if freq_path not in f:
            raise ValueError(f"Dataset {freq_path} not found in the file.")
        freqs = f[freq_path][:]
        logger.info("Extracted frequency array with %d channels.", len(freqs))
        logger.info("Frequency range: %f to %f Hz", freqs[0], freqs[-1])
        if len(freqs) < 2:
            raise ValueError("Not enough frequency points to compute channel bandwidth.")
        channel_bandwidth = abs(freqs[1] - freqs[0])
        logger.info("Computed channel bandwidth: %f Hz", channel_bandwidth)

        obs_group = f["Observation1"]
        tInt = obs_group.attrs.get("tInt", None)
        if tInt is not None:
            tsamp = float(tInt)
            logger.info("Using tsamp from attribute: %f s", tsamp)
        else:
            time_path = "Observation1/time"
            time_dset = f[time_path]
            times_int = time_dset["int"][:nspectra]
            times_frac = time_dset["frac"][:nspectra]
            times = times_int + times_frac
            tsamp = float(np.median(np.diff(times)))
            logger.info("Estimated tsamp from time dataset: %f s", tsamp)

    # Apply combined RFI filtering
    logger.info("Starting RFI filtering...")
    mask = sk_sg_filter(data, channel_bandwidth, tsamp,
                        spectral_kurtosis_sigma=6.0,
                        savgol_frequency_window=743.0*41,  # in Hz
                        savgol_sigma=5.0)
    bad_channels = np.where(mask)[0]
    output_file = "bad_channels.txt"
    np.savetxt(output_file, bad_channels, fmt="%d", delimiter=" ", newline=" ")
    logger.info("Saved %d bad channel indices to %s", len(bad_channels), output_file)

if __name__ == "__main__":
    main()


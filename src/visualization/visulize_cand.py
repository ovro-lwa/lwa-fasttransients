import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec  

def visualize_aligned_candidate(
    file_path,
    candidate_sample,
    candidate_dm,
    time_window=20,
    device="cpu",
    output_filename="candidate_visualization.png",
):
    """
    Extracts per-channel time-aligned signals based on dispersion measure (DM),
    visualizes a dedispersed candidate signal in an HDF5 file.

    Args:
        file_path (str): Path to the HDF5 file.
        candidate_sample (int): Time sample where the candidate was detected (global index).
        candidate_dm (float): DM of candidate.
        time_window (int): Window size (samples) around each channel's expected arrival time.
        device (str): 'cpu' or 'cuda:X'. Defaults to 'cpu'.
        output_filename (str): Path to save the visualization.
    """

    # 1 Load Frequency & Metadata
    with h5py.File(file_path, 'r') as f:
        obs = f["Observation1"]
        tuning = obs["Tuning1"]
        
        # Load frequency array in MHz
        freq_hz = tuning["freq"][:]
        freq_mhz = freq_hz / 1e6  # Convert to MHz

        # Retrieve sample time from attributes
        tsamp = obs.attrs.get("tInt", 0.001337)  # Default if missing
        n_chans = len(freq_mhz)

        # Load the intensity dataset reference (we will read partial slices later)
        intensity_dataset = tuning["I"]

        # Check if frequency is ascending; if so, reverse the **indexing** logic
        reverse_freq = freq_mhz[0] < freq_mhz[-1]
        if reverse_freq:
            freq_mhz = freq_mhz[::-1]  # Reverse frequencies to ensure highest is first


    # 2 Compute Expected Arrival Time Per Channel
    k_dm = 4.148808e3  # MHz² cm³ pc⁻¹ s
    f_high = freq_mhz[0]  # Highest frequency (MHz)

    # Corrected dispersion delay formula (negative delay correction)
    delays_sec = -k_dm * candidate_dm * (1.0 / freq_mhz**2 - 1.0 / f_high**2)
    delays_samples = np.round(delays_sec / tsamp).astype(int)  # Convert to sample indices

    # Compute per-channel expected arrival times
    arrival_times = candidate_sample - delays_samples  # Shifted arrival time for each channel


    # 3 Extract Per-Channel Time Windows

    extracted_windows = []
    time_axes = []

    with h5py.File(file_path, 'r') as f:
        tuning = f["Observation1"]["Tuning1"]
        intensity_dataset = tuning["I"]

        for ch in range(n_chans):
            # Compute correct channel index
            channel_index = ch if not reverse_freq else (n_chans - 1 - ch)

            # Compute safe time bounds
            tmin = max(0, arrival_times[ch] - time_window)
            tmax = min(intensity_dataset.shape[0], arrival_times[ch] + time_window + 1)
            
            # Read only the relevant slice for this channel using the corrected index
            signal_slice = intensity_dataset[tmin:tmax, channel_index]  # Extract (time_window * 2) samples

            # Store extracted window
            extracted_windows.append(signal_slice)

            # Generate time axis relative to candidate time
            local_times = np.arange(tmin, tmax) - candidate_sample
            time_axes.append(local_times * tsamp)  # Convert to seconds

    # Convert to NumPy array
    extracted_windows = np.array(extracted_windows, dtype=np.float32)
    time_axes = np.array(time_axes)

    # Ensure data is C-contiguous for PyTorch (avoids stride issues)
    extracted_windows = np.ascontiguousarray(extracted_windows)

    # 4 Plot Aligned Waterfall & Summed Signal

    # Create figure with GridSpec layout
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(2, 2, width_ratios=[0.9, 0.05], height_ratios=[1, 3], wspace=0.05)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
    cax = fig.add_subplot(gs[:, 1])  

    # Summed Time Series (Top Plot)
    summed_signal = extracted_windows.sum(axis=0)
    time_range = time_axes[0]
    ax0.plot(time_range, summed_signal, color="black")
    ax0.axvline(0, color="red", linestyle="--", label="Candidate")
    ax0.set_ylabel("Summed Amplitude")
    ax0.set_title(f"Candidate at DM={candidate_dm}, Sample={candidate_sample}")
    ax0.legend()

    #  Waterfall Plot (Bottom Plot)
    im = ax1.imshow(
        extracted_windows,
        aspect="auto",
        origin="lower",
        extent=[time_range[0], time_range[-1], 0, n_chans]
    )
    ax1.set_xlabel("Time (sec)")
    ax1.set_ylabel("Channel")
    
    # Add colorbar to the dedicated axis
    cbar = fig.colorbar(im, cax=cax, label="Amplitude")

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_filename, dpi=150)
    plt.close(fig)

    print(f"Saved aligned candidate visualization to {output_filename}")



# Example usage
if __name__ == "__main__":
    file_path = "beam_290.44_21.88.hdf5"
    candidate_sample = 5001 
    candidate_dm = 10.0
    time_window = 20  

    visualize_aligned_candidate(
        file_path=file_path,
        candidate_sample=candidate_sample,
        candidate_dm=candidate_dm,
        time_window=time_window,
        device="cpu",
        output_filename="aligned_candidate_dm10_sample5001.png"
    )

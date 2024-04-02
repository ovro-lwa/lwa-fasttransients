import h5py
from astropy.coordinates import SkyCoord
from astropy.time import Time
import numpy as np
import argparse
import datetime
from your.formats.filwriter import make_sigproc_object
import tqdm


chunk_size = 100000

def timestamp_to_mjd(time_tuple):
    """
    Convert a timestamp tuple to Modified Julian Date (MJD).

    Args:
        time_tuple (tuple): A tuple with two elements: (timestamp, fraction).

    Returns:
        float: The Modified Julian Date (MJD) corresponding to the timestamp.
    """
    if not isinstance(time_tuple, tuple) or len(time_tuple) != 2:
        raise ValueError("Input must be a tuple with two elements: (timestamp, fraction).")
    
    timestamp, fraction = time_tuple
    ts = datetime.datetime.utcfromtimestamp(timestamp).isoformat()
    fractional_part = f".{str(fraction).split('.')[-1]}" if '.' in str(fraction) and fraction != 0 else ''
    ts_with_fraction = f"{ts}{fractional_part}"
    mjd = Time(ts_with_fraction, format='isot').mjd
    
    return mjd



def prepare_header(name, RA, Dec, nchans, tsamp, num_samples, tstart, fr1, foff):
    """
    Prepare header parameters for the filterbank file.

    Args:
        name (str): The name of the filterbank file.
        RA (float): The Right Ascension coordinate in degrees.
        Dec (float): The Declination coordinate in degrees.
        nchans (int): The number of frequency channels.
        tsamp (float): The time duration of each sample in seconds.
        num_samples (int): The total number of samples.
        tstart (float): The start time of the observation in MJD.
        fr1 (float): The frequency of the first channel in MHz.
        foff (float): The frequency offset between channels in MHz.

    Returns:
        dict: A dictionary containing the header parameters for the filterbank file.
    """
    # Convert RA and Dec from degrees to HHMMSS.SS and DDMMSS.SS respectively
    coord = SkyCoord(RA, Dec, unit="deg")
    src_raj = coord.ra.degree # RA in degrees as a float
    src_dej = coord.dec.degree  # Dec in degrees as a float

    header_params = {
        "rawdatafile": name.replace('.hdf5', '.fil'),
        "source_name": "bar",  # Assuming 'bar' is a placeholder; adjust as needed
        "nchans": nchans,
        "foff": foff,
        "fch1": fr1,
        "tsamp": tsamp,
        "tstart": tstart,
        "src_raj": src_raj,
        "src_dej": src_dej,
        "machine_id": 0,
        "nbeams": 0,
        "ibeam": 0,
        "nbits": 32,
        "nifs": 1,
        "barycentric": 0,
        "pulsarcentric": 0,
        "telescope_id": 0,
        "data_type": 0,
        "az_start": -1,
        "za_start": -1,
    }

    return header_params

def convert_hdf5_to_filterbank(name, RA, Dec):
    """
    Convert HDF5 file to Filterbank format.

    Args:
        name (str): Name of the HDF5 file.
        RA (float): RA coordinate.
        Dec (float): Dec coordinate.
    """
    # Check if the file has a .hdf5 extension
    if not name.endswith('.hdf5'):
        raise ValueError("The file must have a .hdf5 extension")
    
    with h5py.File(name, 'r') as f:
        observation = f['Observation1']
        nchans = observation.attrs['nChan']
        tsamp = observation.attrs['tInt'] # in sec
        num_samples = len(observation['time'])

        observation_time = observation['time'][0]  
        time_tuple = (observation_time[0], observation_time[1])
        tstart = timestamp_to_mjd(time_tuple)

        tuning = observation['Tuning1']
        fr1 = tuning['freq'][-1] / 1e6  # Central frequency of the first channel in MHz
        foff = (tuning['freq'][-2] - tuning['freq'][-1]) / 1e6 # Channel width in MHz (should be negative in our case)

        I_dataset = tuning['I']

        header_params = prepare_header(name, RA, Dec, nchans, tsamp, num_samples, tstart, fr1, foff)

        sigproc_object = make_sigproc_object(**header_params)
        sigproc_object.write_header(name.replace('.hdf5', '.fil'))

        for start_idx in tqdm(range(0, num_samples, chunk_size), desc="Processing chunks"):
            end_idx = min(start_idx + chunk_size, num_samples)
            I_data_chunk = I_dataset[start_idx:end_idx, :]
            # Process and write the chunk
            data_chunk = I_data_chunk[:, ::-1].astype(np.float32)  # Reverse and type convert
            sigproc_object.append_spectra(data_chunk, name.replace('.hdf5', '.fil'))




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert HDF5 to Filterbank')
    parser.add_argument('-n', '--name', type=str, help='Name of the HDF5 file')
    parser.add_argument('-RA', type=float, help='RA coordinate')
    parser.add_argument('-Dec', type=float, help='Dec coordinate')
    args = parser.parse_args()

    convert_hdf5_to_filterbank(args.name, args.RA, args.Dec)
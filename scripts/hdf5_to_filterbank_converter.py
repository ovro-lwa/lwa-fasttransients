import h5py
from astropy.coordinates import SkyCoord, Angle
from astropy.time import Time
import numpy as np
from sigpyproc.header import Header
import argparse
import datetime
def timestamp_to_mjd(times):
    time_final = []
    for t in times:
        ts = datetime.datetime.fromtimestamp(t[0]).isoformat() + str(t[1])[1:]
        time_final.append(Time(ts, format='isot').mjd)
    return np.array(time_final)


def convert_hdf5_to_filterbank(name, nstart):
    with h5py.File(name, 'r') as f:
        observation = f['Observation1']
        nchans = observation.attrs['nChan']
        tsamp = observation.attrs['tInt']
        num_samples = len(observation['time'])
        tsart = timestamp_to_mjd(observation['time'])[0]
        tuning = observation['Tuning1']
        fr1 = tuning['freq'][0] / 1e6
        foff = (tuning['freq'][1] - tuning['freq'][0]) / 1e6
        XX_data = tuning['XX'][:]
        YY_data = tuning['YY'][:]
        I_data = XX_data + YY_data

    header_params = {
        "filename": f"{name}.fil",
        "data_type": "filterbank",
        "nchans": nchans - nstart,
        "foff": foff,
        "fch1": fr1 + nstart * foff,
        "nbits": 32,
        "tsamp": tsamp,
        "tstart": tsart,
        "nsamples": num_samples,
        "nifs": 1,
        "coord": SkyCoord(90., 90., unit="deg"),
        "azimuth": Angle(-99., unit="deg"),
        "zenith": Angle(-99., unit="deg"),
        "telescope": "OVRO-LWA",
        "backend": "",
        "source": "",
        "frame": "",
        "ibeam": 1,
        "nbeams": 1,
        "dm": 0,
        "period": 0,
        "accel": 0,
        "signed": False,
        "rawdatafile": "",
        "hdrlens": [],
        "datalens": [],
        "filenames": [f"{name}.fil"],
        "nsamples_files": [num_samples],
        "tstart_files": [tsart]
    }

    header = Header(**header_params)
    writer = header.prep_outfile(filename=f"{name}.fil", update_dict=None)
    data = I_data[:, nstart:].astype(np.float32)
    writer.cwrite(data)
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert HDF5 to Filterbank')
    parser.add_argument('-n', '--name', type=str, help='Name of the HDF5 file')
    parser.add_argument('-nstart', type=int, help='Value of nstart')
    args = parser.parse_args()

    convert_hdf5_to_filterbank(args.name, args.nstart)

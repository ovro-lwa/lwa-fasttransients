from casacore import tables
import numpy as np
import os
import pickle
import argparse

def process_bcal_files(directory, nupchan=32, output_filename="gains.pkl"):
    NSUBBAND = 2
    NCHAN = 96

    # Lists to store all data and frequencies collectively from all .bcal files
    all_data_list = []
    all_frequencies_list = []

    # Loop over all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.bcal'):
            caltable = os.path.join(directory, filename)
            
            # Read calibration tables to extract gains
            tab = tables.table(caltable, ack=False)
            caldata = tab.getcol('CPARAM')[...]
            caldata /= np.abs(caldata)
            

            flgdata = tab.getcol('FLAG')[...]
            tab.close()

            tab = tables.table(os.path.join(caltable, 'SPECTRAL_WINDOW'), ack=False)
            calfreq = tab.getcol('CHAN_FREQ')[...]
            calfreq = calfreq.ravel()
            tab.close()

            cal = 1./caldata
            cal = np.where(np.isfinite(cal), cal, 0)
            cal *= (1-flgdata)
            print('cal shape', np.shape(cal))

            # Repeat the data for upchannelization
            expanded_cal = np.repeat(cal, repeats=nupchan, axis=1)

            # Reshape the expanded_cal to get data separated by NSUBBAND
            reshaped_cal = expanded_cal.reshape(352, NCHAN, NSUBBAND * nupchan, 2)

            # For each NSUBBAND, extract the data
            for i in range(NSUBBAND):
                data_chunk = reshaped_cal[:, :, i * nupchan:(i + 1) * nupchan, :]
                start_freq = calfreq[i * NCHAN]
                print(start_freq)
                print(np.shape(data_chunk))
                all_data_list.append(data_chunk)
                all_frequencies_list.append(start_freq)

    # Dictionary to store collective data and frequencies
    master_data = {
        "data": all_data_list,
        "frequencies": all_frequencies_list
    }

    # Save all data and frequencies to the specified pickle file
    with open(output_filename, 'wb') as f:
        pickle.dump(master_data, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process .bcal files to produce gains for offline beamforming")
    parser.add_argument("directory", type=str, help="Directory containing .bcal files")
    parser.add_argument("--nupchan", type=int, default=32, help="NUPCHAN parameter (default: 32)")
    parser.add_argument("--output", type=str, default="gains.pkl", help="Output file name (default: gains.pkl)")

    args = parser.parse_args()
    
    process_bcal_files(args.directory, args.nupchan, args.output)


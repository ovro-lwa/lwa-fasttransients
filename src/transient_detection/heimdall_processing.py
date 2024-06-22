import subprocess
import logging
import configparser
import numpy as np
import os
from timeit import default_timer as timer

class HeimdallProcessor:
    def __init__(self):
        pass

    @staticmethod
    def print_zap_chans_command(zap_chans):
        cmd_parts = []
        for chan in zap_chans:
            cmd_parts.append(f"-zap_chans {int(chan)} {int(chan)}")
        return " ".join(cmd_parts)

    @staticmethod
    def extract_info_from_fil(filenames, your_object, dm):
        logging.info("Reading raw data from " + str(filenames))

        center_freq = your_object.your_header.center_freq
        logging.info("The center frequency is " + str(center_freq) + " MHz")

        bw = your_object.your_header.bw
        logging.info("The bandwidth is " + str(bw) + " MHz")

        tsamp = your_object.your_header.native_tsamp
        logging.info("The native sampling time is " + str(tsamp) + " s")

        obs_len = your_object.your_header.native_nspectra * tsamp

        if obs_len >= 60:
            obs_len_min = obs_len / 60
            logging.info("Dataset length is " + str(obs_len_min) + " minutes")
        else:
            logging.info("Dataset length is " + str(obs_len) + " seconds")

        f_low = (center_freq + bw / 2) * 10**(-3)  # in GHz
        f_high = (center_freq - bw / 2) * 10**(-3)  # in GHz

        logging.info("The frequency range is " + str(f_low) + " GHz to " + str(f_high) + " GHz")

        dm_h = (obs_len * 10**3 / 4.15) * (1 / ((1 / f_low**2) - (1 / f_high**2)))

        logging.info("DM threshold is " + str(dm_h) + " pc/cm^3")
        logging.info("Provided DM is " + str(dm) + " pc/cm^3")

        if dm >= dm_h:
            logging.error("Invalid Dispersion Measure (DM). DM should be less than the calculated DM threshold.")

        max_delay = your_object.dispersion_delay(dms=dm)
        dispersion_delay_samples = np.ceil(max_delay / your_object.your_header.tsamp)

        if your_object.your_header.nspectra < 2**18:
            nsamps_gulp = your_object.your_header.nspectra
        else:
            nsamps_gulp = int(
                np.max([(2 ** np.ceil(np.log2(dispersion_delay_samples))), 2**18])
            )

        boxcar_max = int(50e-3 / your_object.your_header.tsamp)  # TODO: check if this is the correct value

        return nsamps_gulp, boxcar_max

    def process(self, fil_file_name, dm):
        heimdall_start = timer()
        logging.info(f"HEIMDALL: Using the raw data from {fil_file_name} for Heimdall")
        logging.info("HEIMDALL: Preparing to run Heimdall...")

        try:
            mask = np.loadtxt(f"{fil_file_name}_your_rfi_mask.bad_chans")
            if len(mask.shape) == 1:
                bad_chans = list(mask)
                zap_chans_cmd = self.print_zap_chans_command(bad_chans)
            else:
                logging.warning("RFI mask not understood, can only be 1D. Not using RFI flagging.")
                zap_chans_cmd = ""
        except Exception as e:
            logging.error(f"Error loading RFI mask: {e}")
            zap_chans_cmd = ""

        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, '../config.ini')
        config = configparser.ConfigParser()
        config.read(config_path)
        heimdall_binary_path = config.get('Paths', 'HeimdallBinaryPath')

        heimdall_cmd = f"{heimdall_binary_path} -f {fil_file_name} -dm {dm} {dm} -nsamps_gulp {nsamps_gulp} -boxcar_max {boxcar_max} {zap_chans_cmd}"
        logging.debug(f"Executing Heimdall command: {heimdall_cmd}")
        subprocess.call(heimdall_cmd, shell=True)
        heimdall_end = timer()
        logging.debug(f"HEIMDALL: Heimdall processing took {heimdall_end - heimdall_start} s")

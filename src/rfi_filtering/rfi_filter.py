import subprocess
import logging
from timeit import default_timer as timer

class RFI_Filter:
    def __init__(self):
        pass

    def apply_filter(self, filenames, basenames, your_object):
        mask_start = timer()
        mask_cmd = "your_rfimask.py -v -f " + str(filenames) + " -sk_sigma 4 -sg_sigma 4 -sg_frequency 15"
        logging.debug('RFI MASK: command = ' + mask_cmd)
        subprocess.call(mask_cmd, shell=True)
        mask_end = timer()
        logging.debug('RFI MASK: your_rfimask.py took ' + str(mask_end - mask_start) + ' s')
        mask_basename = str(basenames) + '_your_rfi_mask'
        killmask_file = f"{mask_basename}.bad_chans"
        with open(killmask_file, 'r') as myfile:
            file_str = myfile.read()
        
        my_list = file_str.split(' ')
        my_list = [chan for chan in my_list if chan != '']
        if len(my_list) == 0:
            logging.info('RFI MASK: No channels zapped')
        else:
            logging.debug(f'RFI MASK: No. of channels zapped = {len(my_list)}')
            logging.info('RFI MASK: Percentage of channels zapped = ' + str((len(my_list) / your_object.your_header.nchans) * 100) + ' %')

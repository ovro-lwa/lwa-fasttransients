import subprocess
import os
import json
import argparse
from concurrent.futures import ThreadPoolExecutor

def load_config(file_path):
    """ Load GPU addresses from a JSON configuration file """
    with open(file_path, 'r') as file:
        return json.load(file)

def setup_directories(base_path, gpu_addresses):
    """ Create the directory structure for each GPU """
    for gpu in gpu_addresses:
        for data_path in ['data0', 'data1']:
            dir_path = os.path.join(base_path, gpu, data_path)
            os.makedirs(dir_path, exist_ok=True)

def transfer_file(gpu, address, data_index, name_base, base_path):
    """ Function to transfer files for one GPU and data index """
    local_dir = os.path.join(base_path, gpu, data_index)
    remote_path = f"ubuntu@{address}:/{data_index}/{name_base}*"
    rsync_cmd = ['rsync', '-av', '--progress', remote_path, local_dir]
    subprocess.run(rsync_cmd)
    print(f"Files transferred to {local_dir}")

def transfer_files(name_base, base_path, gpu_addresses):
    """ Transfer files from remote GPUs to the local directory structure using 'ubuntu' as the SSH username """
    with ThreadPoolExecutor(max_workers=len(gpu_addresses)) as executor:
        for gpu, address in gpu_addresses.items():
            for data_index in ['data0', 'data1']:
                executor.submit(transfer_file, gpu, address, data_index, name_base, base_path)


def main():
    parser = argparse.ArgumentParser(description='Transfer files from remote GPUs to local directories.')
    parser.add_argument('--config', type=str, required=True, help='Path to the JSON configuration file')
    parser.add_argument('--name_base', type=str, required=True, help='Base name of the files to transfer')
    args = parser.parse_args()

    base_path = os.getcwd()  # Base path is the current working directory
    gpu_addresses = load_config(args.config)
    setup_directories(base_path, gpu_addresses)
    transfer_files(args.name_base, base_path, gpu_addresses)

if __name__ == "__main__":
    main()


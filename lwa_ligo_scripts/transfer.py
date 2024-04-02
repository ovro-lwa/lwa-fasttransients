import subprocess
import os
import re
import argparse
def transfer_and_rename_files(remote_host, remote_path, local_path, name_pattern, final_suffix):
    # Execute the rsync command with wildcard to transfer all matching files
    rsync_cmd = [
        'rsync', '-avh', '--progress', '-e',
        'ssh -o StrictHostKeyChecking=no',
        f"{remote_host}:{remote_path}/{name_pattern}*",
        local_path
    ]
    subprocess.run(rsync_cmd)

    # After transfer, rename files based on matching pattern
    for filename in os.listdir(local_path):
        if re.match(name_pattern, filename):
            # Construct the final filename with the appropriate suffix
            new_filename = f"{filename}.{final_suffix}"
            new_file_path = os.path.join(local_path, new_filename)
            original_file_path = os.path.join(local_path, filename)

            # Check if the renamed version exists before renaming
            if not os.path.exists(new_file_path):
                os.rename(original_file_path, new_file_path)
                print(f"Renamed {filename} to {new_filename} successfully.")
            else:
                print(f"{new_filename} already exists. Skipping renaming.")

def main():
    parser = argparse.ArgumentParser(description='Transfer and rename files from remote host to local directory.')
    parser.add_argument('remote_host', type=str, help='Remote host address')
    parser.add_argument('remote_path', type=str, help='Remote path to the directory containing the files')
    parser.add_argument('local_path', type=str, help='Local path to the directory where files will be transferred')
    parser.add_argument('name_pattern', type=str, help='Pattern to match the filenames')
    args = parser.parse_args()

    hosts_range = range(1, 9)

    for i in hosts_range:
        remote_host = args.remote_host.replace('01', f'{i:02d}')
        # Loop over /data0 and /data1 directories
        for data_index, data_path in enumerate(['/data0', '/data1']):
            final_suffix = f"{i}{data_index}"  # Construct the suffix based on host index and data path
            transfer_and_rename_files(remote_host, data_path, args.local_path, args.name_pattern, final_suffix)

if __name__ == '__main__':
    main()

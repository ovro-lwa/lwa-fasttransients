import json
import subprocess
import os

def calculate_dm_ranges(loDM, hiDM, f_ctr, BW, numchan, tsamp):
    """
    Calculate the DM ranges (see PyTorchDedispersion documentation) using the dedisp_plan.py script.

    Args:
        loDM (float): Lower bound of the DM range.
        hiDM (float): Upper bound of the DM range.
        f_ctr (float): Center frequency in MHz.
        BW (float): Bandwidth in MHz.
        numchan (int): Number of channels.
        tsamp (float): Sampling time in seconds.

    Returns:
        list: Calculated DM ranges.
    """

    script_path = os.path.join(os.path.dirname(__file__), "dedisp_plan.py")
    output_file = os.path.join(os.path.dirname(__file__), "dm_ranges.json")
    command = [
        "python", script_path,
        "-l", str(loDM),
        "-d", str(hiDM),
        "-f", str(f_ctr),
        "-b", str(BW),
        "-n", str(numchan),
        "-t", str(tsamp),
        "--output", output_file
    ]

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"dedisp_plan.py failed with error: {result.stderr}")

    with open(output_file, 'r') as f:
        dm_ranges_output = json.load(f)

    # Transform the output to match the required DM_RANGES format
    dm_ranges = [
        {"start": dm_range["Low DM"], "stop": dm_range["High DM"], "step": dm_range["dDM"]}
        for dm_range in dm_ranges_output
    ]
    return dm_ranges

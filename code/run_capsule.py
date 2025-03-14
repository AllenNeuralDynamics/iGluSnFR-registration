import os
import cv2
import csv
import argparse
import re
import pytz
import json
import glob
import h5py
import dateparser
from datetime import datetime
import numpy as np
from ScanImageTiffReader import ScanImageTiffReader
from scipy.fft import fft2
import numpy.ma as ma
import tifffile
from scipy.interpolate import interp1d
from scipy.interpolate import interp1d, PchipInterpolator
from utils.stripRegistrationBergamo import stripRegistrationBergamo_init
import logging
from tenacity import retry, stop_after_attempt, wait_fixed
import warnings

warnings.filterwarnings("ignore")


# Define the retry decorator
# @retry(stop=stop_after_attempt(4), wait=wait_fixed(3))
def read_tiff_file(fn):
    print("Reading:", fn)
    try:
        # Attempt to read the TIFF file
        Ad = tifffile.imread(fn)

        if len(Ad.shape) == 3:
            Ad = np.reshape(
                Ad, (Ad.shape[0], 1, Ad.shape[1], Ad.shape[2])
            )  # Add channel info
        print("Ad shape:", Ad.shape)
        numChannels = Ad.shape[1]
        return Ad, numChannels
    except Exception as e:
        print(f"Failed to read {fn}: {e}")
        return None


def process_file(fn, folder_number, params, output_path, caiman_template):
    try:
        Ad, params["numChannels"] = read_tiff_file(fn)
        # Ad = np.reshape(Ad, (Ad.shape[0], Ad.shape[1], params['numChannels'], -1))
    except Exception as e:
        print(f"Failed to read {fn} after multiple attempts: {e}")
        return False

    # Permute the dimensions of the array to reorder them
    Ad = np.transpose(Ad, (2, 3, 1, 0))
    Ad = np.array(Ad, dtype=np.float32)
    Ad = Ad[params["removeLines"] :, :, :, :]

    # Ad = Ad[:,:,:,:9000] #TODO: Remove this after Debug

    initFrames = 1000
    name, ext = os.path.splitext(os.path.basename(fn))
    os.makedirs(os.path.join(output_path, folder_number), exist_ok=True)

    # Strip Bergamo Registration
    strip_fn = f"{name}{ext}"
    output_path_ = os.path.join(output_path, os.path.join(folder_number, strip_fn))
    path_template_list = []
    path_template_list = stripRegistrationBergamo_init(
        params["ds_time"],
        initFrames,
        Ad,
        params["maxshift"],
        params["clipShift"],
        params["alpha"],
        params["numChannels"],
        path_template_list,
        output_path_,
        caiman_template,
    )

    # Clean up temporary files
    if not caiman_template:
        for file_path in path_template_list:
            if os.path.exists(file_path):
                print("Deleting jnormcorre tif files.")
                os.remove(file_path)
            else:
                print(f"The file {file_path} does not exist.")

    # Remove tiffs generated from CaImAn
    caiman_dir_path = "/root/capsule/"
    caiman_tif_files = glob.glob(os.path.join(caiman_dir_path, "*.tiff"))
    if len(caiman_tif_files) == 0:
        print("caiman tiffs don't exist")
    else:
        print("Deleting caiman tif files.")
        for tif_file in caiman_tif_files:
            os.remove(tif_file)

    return True


def run(params, data_dir, output_path, caiman_template):
    if not os.path.exists(output_path):
        print("Creating main output directory...")
        os.makedirs(output_path)
        print("Output directory created at", output_path)

    # Iterate over all files in the directory
    for filename in os.listdir(data_dir):
        # Construct full file path
        file_path = os.path.join(data_dir, filename)
        # Check if the file is a .tif file
        if os.path.isfile(file_path) and filename.endswith(".tif"):
            folder_number = os.path.basename(data_dir)
            process_file(file_path, folder_number, params, output_path, caiman_template)


if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input to the folder that contains tiff files.",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output folder to save the results."
    )

    # Add optional arguments with default values
    parser.add_argument("--maxshift", type=int, default=50, help='Max pixel shift for registration')
    parser.add_argument("--clipShift", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.0005, help='Regularization parameter')
    parser.add_argument("--removeLines", type=int, default=4, help='Border pixels to exclude')
    parser.add_argument("--numChannels", type=int, default=1, help='Supports multichannel registration')
    parser.add_argument("--writetiff", type=bool, default=False, help='If False it would write movies as h5 file else write it as a tif')
    parser.add_argument("--ds_time", type=int, default=1, help='Temporal downsampling factor (2^n)')
    parser.add_argument(
        "--caiman_template",
        type=bool,
        default=True,
        help="By default it uses Caiman to generate initial template, else it would use JNormCorre",
    )

    # Parse the arguments
    args = parser.parse_args()

    data_dir = args.input
    output_path = args.output

    # Assign the parsed arguments to params dictionary
    params = {}

    params["maxshift"] = args.maxshift
    params["clipShift"] = args.clipShift
    params["alpha"] = args.alpha
    params["removeLines"] = args.removeLines
    params["numChannels"] = args.numChannels
    params["writetiff"] = args.writetiff
    params["ds_time"] = args.ds_time

    run(params, data_dir, output_path, args.caiman_template)


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
from utils.suite2pRegistration import suite2pRegistration
from utils.CaImAnRegistration import CaImAnRegistration
# from aind_data_schema.core.data_description import Funding, RawDataDescription
# from aind_data_schema_models.modalities import Modality
# from aind_data_schema_models.organizations import Organization
# from aind_data_schema_models.pid_names import PIDName
# from aind_data_schema_models.platforms import Platform
import logging
from tenacity import retry, stop_after_attempt, wait_fixed
import warnings
warnings.filterwarnings("ignore")


# Define the retry decorator
@retry(stop=stop_after_attempt(4), wait=wait_fixed(3))
def read_tiff_file(fn):
    print('Reading:', fn)
    # A = ScanImageTiffReader(fn)
    # Ad = np.array(A.data(), dtype=np.float32)
    # return Ad
    with tifffile.TiffFile(fn) as tif:
        imageData = tif.asarray()
        Ad = np.array(imageData, dtype=np.float32)
        print('Shape while reading tiff---->', Ad.shape)

        if len(Ad.shape) == 3:
            Ad = np.reshape(Ad, (Ad.shape[0], 1, Ad.shape[1],  Ad.shape[2])) # Add channel info
            
        numChannels = Ad.shape[1]
    return Ad, numChannels  

def process_file(fn, folder_number, params, output_path, writer, use_suite2p, use_caiman, compute_mse):
    try:
        Ad, params['numChannels'] = read_tiff_file(fn)
        # Ad = np.reshape(Ad, (Ad.shape[0], Ad.shape[1], params['numChannels'], -1))
    except Exception as e:
        print(f"Failed to read {fn} after multiple attempts: {e}")
        return False

    # Permute the dimensions of the array to reorder them
    Ad = np.transpose(Ad, (2, 3, 1, 0))
    Ad = Ad[params['removeLines']:, :, :, :]

    # Ad = Ad[:,:,:,:9000] #TODO: Remove this after Debug

    initFrames = 1000
    name, ext = os.path.splitext(os.path.basename(fn))
    os.makedirs(os.path.join(output_path, folder_number), exist_ok=True)

    # 1. Strip Bergamo Registration
    strip_fn = f"{name}{ext}"
    output_path_ = os.path.join(output_path, os.path.join(folder_number, strip_fn))
    path_template_list = []
    x_shifts_strip, y_shifts_strip, path_template_list = stripRegistrationBergamo_init(params['ds_time'], initFrames, Ad, params['maxshift'], params['clipShift'], params['alpha'], params['numChannels'], path_template_list, output_path_)

    if use_suite2p:
        # 2. Suite2p Registration
        suite2p_fn = f"{name}_suite2p{ext}"
        output_path_suite2 = os.path.join(output_path, os.path.join(folder_number, suite2p_fn))
        n_time, Ly, Lx = Ad.shape[3], Ad.shape[0], Ad.shape[1]
        x_shifts_suite2p, y_shifts_suite2p = suite2pRegistration(data_dir, fn, n_time, Ly, Lx, output_path, folder_number, output_path_suite2)
    else:
        print('-------Skipping Suite2p------')
        x_shifts_suite2p, y_shifts_suite2p = 0 , 0

    if use_caiman:
        # 3. CaImAn Registration
        caiman_fn = f"{name}_caiman{ext}"
        output_path_caiman = os.path.join(output_path, os.path.join(folder_number, caiman_fn))
        x_shifts_caiman, y_shifts_caiman = CaImAnRegistration(fn, output_path_caiman)
    else:
        print('-------Skipping Suite2p------')
        x_shifts_caiman, y_shifts_caiman = 0 , 0

    # Check if compute_mse is not None before proceeding
    if compute_mse:
        # Read ground truth
        h5_fn = fn.replace('.tif', '_groundtruth.h5')
        h5_path = h5_fn
        with h5py.File(h5_path, 'r') as file:
            gt_motionC = file['GT/motionC'][:]
            mean_gt_motionC = np.mean(gt_motionC)
            gt_motionC -= mean_gt_motionC

            gt_motionR = file['GT/motionR'][:]
            mean_gt_motionR = np.mean(gt_motionR)
            gt_motionR -= mean_gt_motionR

        # Calculate MSE for x shifts
        mse_x_strip = np.mean((x_shifts_strip - gt_motionR)**2)
        mse_x_suite2p = np.mean((x_shifts_suite2p - gt_motionR)**2)
        mse_x_caiman = np.mean((x_shifts_caiman - gt_motionR)**2)

        # Calculate MSE for y shifts
        mse_y_strip = np.mean((y_shifts_strip - gt_motionC)**2)
        mse_y_suite2p = np.mean((y_shifts_suite2p - gt_motionC)**2)
        mse_y_caiman = np.mean((y_shifts_caiman - gt_motionC)**2)

        print("MSE of x_shifts_strip:", mse_x_strip)
        print("MSE of x_shifts_suite2p:", mse_x_suite2p)
        print("MSE of x_shifts_caiman:", mse_x_caiman)
        
        print("MSE of y_shifts_strip:", mse_y_strip)
        print("MSE of y_shifts_suite2p:", mse_y_suite2p)
        print("MSE of y_shifts_caiman:", mse_y_caiman)

        # Write the MSE values to the CSV file
        writer.writerow({
            'file_name': os.path.basename(fn),
            'suite2p_R': mse_x_suite2p,
            'Caiman_R': mse_x_caiman,
            'strip_R': mse_x_strip,
            'suite2p_C': mse_y_suite2p,
            'Caiman_C': mse_y_caiman,
            'strip_C': mse_y_strip
        })

    # Clean up temporary files
    for file_path in path_template_list:
        if os.path.exists(file_path):
            print("Deleting jnormcorre tif files.")
            os.remove(file_path)
        else:
            print(f"The file {file_path} does not exist.")

    # Remove tiffs generated from CaImAn
    caiman_dir_path = '/root/capsule/'
    caiman_tif_files = glob.glob(os.path.join(caiman_dir_path, '*.tiff'))
    if len(caiman_tif_files) == 0:
        print("caiman tiffs don't exist")
    else:
        print("Deleting caiman tif files.")
        for tif_file in caiman_tif_files:
            os.remove(tif_file)

    return True
    
def run(params, data_dir, output_path, use_suite2p, use_caiman, compute_mse):
    print('data_dir--->', data_dir)
    if not os.path.exists(output_path):
        print('Creating main output directory...')
        os.makedirs(output_path)
        print('Output directory created at', output_path)

    registration_results_file = os.path.join(output_path, 'registration_results.csv')
    file_exists = os.path.isfile(registration_results_file)

    tif_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.tif'):
                file_path = os.path.join(root, file)
                folder_number = os.path.basename(os.path.dirname(file_path))
                tif_files.append((file_path, folder_number))

    print('tif_files--->', tif_files)
    tif_files_sorted = sorted(tif_files, key=lambda x: int(x[1]))

    logging.basicConfig(level=logging.INFO)

    failed_files = []
    
    with open(registration_results_file, 'a', newline='') as csvfile:
        fieldnames = ['file_name', 'suite2p_R', 'Caiman_R', 'strip_R', 'suite2p_C', 'Caiman_C', 'strip_C']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        for i, (fn, folder_number) in enumerate(tif_files_sorted, start=1):
            print(f'Files left: {len(tif_files) - i}')
            if not process_file(fn, folder_number, params, output_path, writer, use_suite2p, use_caiman, compute_mse):
                failed_files.append((fn, folder_number))

    # Retry processing failed files
    if failed_files:
        print("Retrying failed files...")
        for fn, folder_number in failed_files:
            print(f'Retrying file: {fn}')
            process_file(fn, folder_number, params, output_path, writer, use_suite2p, use_caiman, compute_mse)

if __name__ == "__main__": 
    # Create argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', type=str, required=True, help='Input to the folder that contains tiff files.')
    parser.add_argument('--output', type=str, required=True, help='Output folder to save the results.')

    # Add optional arguments with default values
    parser.add_argument('--maxshift', type=int, default=50)
    parser.add_argument('--clipShift', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.0005)
    parser.add_argument('--removeLines', type=int, default=4)
    parser.add_argument('--numChannels', type=int, default=1)
    parser.add_argument('--writetiff', type=bool, default=False)
    parser.add_argument('--ds_time', type=int, default=3)
    parser.add_argument('--suite2p', type=bool, default=False)
    parser.add_argument('--caiman', type=bool, default=False)
    parser.add_argument('--compute_mse', type=bool, default=False)

    # Parse the arguments
    args = parser.parse_args()

    data_dir = args.input
    output_path = args.output

    # Assign the parsed arguments to params dictionary
    params = {}
    params['maxshift'] = args.maxshift
    params['clipShift'] = args.clipShift
    params['alpha'] = args.alpha
    params['removeLines'] = args.removeLines
    params['numChannels'] = args.numChannels
    params['writetiff'] = args.writetiff
    params['ds_time'] = args.ds_time

    run(params, data_dir, output_path, args.suite2p, args.caiman, args.compute_mse)
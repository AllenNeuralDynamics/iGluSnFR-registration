import os
import cv2
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
from scipy.interpolate import interp1d
from scipy.interpolate import interp1d, PchipInterpolator
from utils.stripRegistrationBergamo import stripRegistrationBergamo_init
from utils.suite2pRegistration import suite2pRegistration
from utils.CaImAnRegistration import CaImAnRegistration
from aind_data_schema.core.data_description import Funding, RawDataDescription
from aind_data_schema_models.modalities import Modality
from aind_data_schema_models.organizations import Organization
from aind_data_schema_models.pid_names import PIDName
from aind_data_schema_models.platforms import Platform

def run(params, data_dir, output_path):
    # Create output directory
    if not os.path.exists(output_path):
        print('Creating output directory...')
        os.makedirs(output_path)

    print('Output directory created at', output_path)

    fns = [fn for fn in os.listdir(data_dir) if fn.endswith('.tif')]
    for fn in fns:
        
        A = ScanImageTiffReader(os.path.join(data_dir, fn))
        Ad = np.array(A.data(), dtype=np.float32)
        Ad = np.reshape(Ad, (Ad.shape[0], Ad.shape[1], params['numChannels'], -1))

        # Permute the dimensions of the array to reorder them
        Ad = np.transpose(Ad, (1, 3, 2, 0))
        # Ad = np.transpose(Ad, (3, 2, 1, 0))

        # Remove the specified lines from the array
        Ad = Ad[params['removeLines']:, :, :, :]

        print("Data shape after reshaping and transposing:", Ad.shape)
        
        initFrames = 400
        ds_time = None

        # 1. Strip Bergamo Registration 
        output_path_= os.path.join(output_path, fn)
        path_template_list = []
        x_shifts_strip, y_shifts_strip = stripRegistrationBergamo_init(ds_time, initFrames, Ad, params['maxshift'], params['clipShift'], params['alpha'], params['numChannels'], path_template_list, output_path_)

        # Split the filename into name and extension
        name, ext = os.path.splitext(fn)

        # 2. Suite2p Registration
        suite2p_fn = f"{name}_suite2p{ext}"
        output_path_suite2 = os.path.join(output_path, suite2p_fn)
        n_time, Ly, Lx = Ad.shape[3], Ad.shape[0], Ad.shape[1]
        x_shifts_suite2p, y_shifts_suite2p = suite2pRegistration(data_dir, fn, n_time, Ly, Lx, output_path, output_path_suite2)

        # 3. CaImAn Registration
        suite2p_fn = f"{name}_caiman{ext}"
        output_path_caiman = os.path.join(output_path, suite2p_fn)
        x_shifts_caiman, y_shifts_caiman = CaImAnRegistration(os.path.join(data_dir, fn), output_path_caiman)

        # Read ground truth 
        # Replace the .tif extension with .h5
        h5_fn = fn.replace('.tif', '_groundtruth.h5')

        # Join the directory with the new filename
        h5_path = os.path.join(data_dir, h5_fn)
        print(h5_path)
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


    # path_template_list contains the paths to the files
    for file_path in path_template_list:
        if os.path.exists(file_path):
            os.remove(file_path)
        else:
            print(f"The file {file_path} does not exist.")

    # Remove tiffs generated from CaImAn 
    caiman_dir_path = '/root/capsule/'

    # Find all .tif files in the directory
    caiman_tif_files = glob.glob(os.path.join(caiman_dir_path, '*.tiff'))

    # Delete each file
    if len(caiman_tif_files) == 0:
        print("caiman tiffs don't exist")
    else:
        print("Deleting caiman tif files.")
        for tif_file in caiman_tif_files:
            os.remove(tif_file)

    dt = datetime.now(pytz.timezone('America/Los_Angeles'))
    # create data_description.json
    d = RawDataDescription(
        modality=[Modality.FIB],
        platform=Platform.FIP,
        subject_id='Test', #TODO:  Needs to be something else
        creation_time=dt,
        institution=Organization.AIND,
        investigators=[PIDName(name="Caleb")],
        funding_source=[Funding(funder=Organization.AI)],
        data_summary=json.dumps(params),
    )
    d.write_standard_file(output_path)
    
    with open(output_path + '/simulation_parameters.json', 'w') as f:
        json.dump(params, f)


if __name__ == "__main__": 
    # Create argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', type=str, required=True, help='Input to the folder that contains tiff files.')
    parser.add_argument('--output', type=str, required=True, help='Output folder to save the results.')

    # Add optional arguments with default values
    parser.add_argument('--maxshift', type=int, default=30)
    parser.add_argument('--clipShift', type=int, default=5)
    parser.add_argument('--alpha', type=float, default=0.0005)
    parser.add_argument('--removeLines', type=int, default=4)
    parser.add_argument('--numChannels', type=int, default=1)
    parser.add_argument('--writetiff', type=bool, default=False)

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

    run(params, data_dir, output_path)

import os
import cv2
import argparse
import re
import dateparser
import numpy as np
from ScanImageTiffReader import ScanImageTiffReader
from scipy.fft import fft2
import numpy.ma as ma
from scipy.interpolate import interp1d
from scipy.interpolate import interp1d, PchipInterpolator
from utils.stripRegistrationBergamo import stripRegistrationBergamo_init


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
        output_path_= os.path.join(output_path, fn)
        path_template_list = []
        stripRegistrationBergamo_init(ds_time, initFrames, Ad, params['maxshift'], params['clipShift'], params['alpha'], params['numChannels'], path_template_list, output_path_)

        # path_template_list contains the paths to the files
        for file_path in path_template_list:
            if os.path.exists(file_path):
                os.remove(file_path)
            else:
                print(f"The file {file_path} does not exist.")

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

import cv2
import re
from jnormcorre.motion_correction import MotionCorrect
import dateparser
import numpy as np
from ScanImageTiffReader import ScanImageTiffReader
from scipy.fft import fft2
import numpy.ma as ma
from scipy.interpolate import interp1d
from scipy.interpolate import interp1d, PchipInterpolator


def downsampleTime(Y, ds_time):
    for _ in range(ds_time):
        Y = Y[:, :, :, :2*(Y.shape[3]//2):2] + Y[:, :, :, 1:2*(Y.shape[3]//2):2]
    return Y

def evaluate_description(desc):
    # Split the data into lines
    lines = desc.strip().split('\n')

    # Parse each line
    parsed_data = {}
    for line in lines:
        # Split the line into key and value
        key, value = line.split('=', 1)
        key = key.strip()
        value = value.strip()
        
        # Try to evaluate the value as Python literal if possible
        try:
            value = eval(value)
        except (NameError, SyntaxError):
            # If eval fails, keep the value as a string
            pass
        
        # Store in the dictionary
        parsed_data[key] = value

    print(f"{key}: {value}")

    return parsed_data

def run():
    """ basic run function """
    pass

if __name__ == "__main__": run()
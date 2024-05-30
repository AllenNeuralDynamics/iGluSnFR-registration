import numpy as np
from tifffile import imread
import caiman as cm
from caiman.motion_correction import MotionCorrect, tile_and_correct, motion_correction_piecewise

def CaImAnRegistration(fname, output_path_caiman):
    try:
        cv2.setNumThreads(0)
    except:
        pass

    data = imread(fname) # TODO: Add logic for h5 as well

    max_shifts = (22, 22)  # maximum allowed rigid shift in pixels (view the movie to get a sense of motion)
    strides =  (48, 48)  # create a new patch every x pixels for pw-rigid correction
    overlaps = (24, 24)  # overlap between patches (size of patch strides+overlaps)
    max_deviation_rigid = 3   # maximum deviation allowed for patch with respect to rigid shifts
    pw_rigid = False  # flag for performing rigid or piecewise rigid motion correction
    shifts_opencv = False  # flag for correcting motion using bicubic interpolation (otherwise FFT interpolation is used)
    border_nan = 'copy'  # replicate values along the boundary (if True, fill in with NaN)

    # start the cluster (if a cluster already exists terminate it)
    if 'dview' in locals():
        cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='multiprocessing', n_processes=None, single_thread=False)

    # Create a motion correction object
    mc = MotionCorrect(data, dview=dview, max_shifts=max_shifts,
                    shifts_opencv=shifts_opencv, nonneg_movie=True,
                    border_nan=border_nan)

    # Perform motion correction
    mc.motion_correct(save_movie=True)
    m_rig = cm.load(mc.mmap_file)

    m_rig.save(output_path_caiman)

    coordinates = mc.shifts_rig

    x_shifts = [coord[0] for coord in coordinates]
    y_shifts = [coord[1] for coord in coordinates]

    x_shifts_mean = np.mean(x_shifts)
    y_shifts_mean = np.mean(y_shifts)

    # Center the shifts to zero 
    return x_shifts - x_shifts_mean, y_shifts - y_shifts_mean

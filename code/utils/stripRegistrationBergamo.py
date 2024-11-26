import os
import time
import numpy as np
import h5py
import numpy.ma as ma
from jnormcorre.motion_correction import MotionCorrect
from ScanImageTiffReader import ScanImageTiffReader
from scipy.fft import fft2
import cv2
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr
from scipy.ndimage import binary_dilation
from scipy.interpolate import interp1d, PchipInterpolator
from scipy.cluster.hierarchy import fcluster
from scipy.ndimage import convolve, shift
from tifffile import tifffile
import matplotlib.pyplot as plt
from numba import jit, int64, float32, prange
# from utils.xcorr2_nans import xcorr2_nans as cython_xcorr2_nans
from multiprocessing import Pool, cpu_count
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import spearmanr
import warnings
from scipy.io import savemat 
from scipy.io import loadmat

def process_chunk(args):
    Ad_chunk, viewR, viewC, numChannels, nanRows, nanCols, motionC, motionR, start_frame, end_frame = args
    
    chunk_shape = (viewR.shape[1] - sum(nanRows), viewR.shape[0] - sum(nanCols), (end_frame - start_frame) * numChannels)
    tiffSave_raw_chunk = np.zeros(chunk_shape, dtype=np.float32)

    x_grid = np.array([viewR + motionR_frame for motionR_frame in motionR[start_frame:end_frame]])
    y_grid = np.array([viewC + motionC_frame for motionC_frame in motionC[start_frame:end_frame]])

    rows_to_keep = ~nanRows.astype(bool)
    cols_to_keep = ~nanCols.astype(bool)

    for ch in range(numChannels):
        for i, frame in enumerate(range(start_frame, end_frame)):
            B = cv2.remap(Ad_chunk[:, :, ch, i], 
                          y_grid[i].astype(np.float32), 
                          x_grid[i].astype(np.float32), 
                          cv2.INTER_LINEAR, 
                          borderMode=cv2.BORDER_CONSTANT, 
                          borderValue=np.nan)

            B = B[cols_to_keep, :][:, rows_to_keep]
            tiffSave_raw_chunk[:, :, i*numChannels+ch] = B.T

    return tiffSave_raw_chunk

def circshift(arr, shift, axes=None):
    """
    Circularly shift the elements of an array.

    Parameters:
    - arr (np.ndarray): Input array to be shifted.
    - shift (int, list, or np.ndarray): Number of places by which elements are shifted.
        - If an integer, the same shift is applied to all axes.
        - If a list or np.ndarray, each element specifies the shift for the corresponding axis.
    - axes (int, list of ints, or None, optional): Axis or axes along which to shift.
        - If None and shift is an integer, shift is applied to all axes.
        - If None and shift is a list/array, the length of shift must match the number of dimensions in arr.
        - Can be an integer or a list of integers corresponding to the axes.

    Returns:
    - np.ndarray: The shifted array.

    Raises:
    - ValueError: If the length of shift does not match the number of specified axes.
    """
    arr = np.asarray(arr)
    ndim = arr.ndim

    # Handle the axes parameter
    if axes is None:
        if isinstance(shift, (int, float)):
            axes = tuple(range(ndim))
            shift = (int(shift),) * ndim
        else:
            shift = tuple(int(s) for s in shift)
            if len(shift) != ndim:
                raise ValueError("Length of shift array must match number of dimensions of arr.")
            axes = tuple(range(ndim))
    else:
        if isinstance(axes, int):
            axes = (axes,)
        elif isinstance(axes, (list, tuple, np.ndarray)):
            axes = tuple(axes)
        else:
            raise ValueError("axes must be an int or a list/tuple of ints.")

        if isinstance(shift, (int, float)):
            shift = (int(shift),) * len(axes)
        else:
            shift = tuple(int(s) for s in shift)
            if len(shift) != len(axes):
                raise ValueError("Length of shift array must match number of specified axes.")

    # Apply the shifts
    for axis, s in zip(axes, shift):
        arr = np.roll(arr, shift=s, axis=axis)

    return arr

def process_raw_frames_cpu(Ad, viewR, viewC, numChannels, nanRows, nanCols, motionC, motionR):
    num_frames = len(motionC)
    num_cores = cpu_count()
    chunk_size = num_frames // num_cores

    chunks = []
    for i in range(num_cores):
        start_frame = i * chunk_size
        end_frame = start_frame + chunk_size if i < num_cores - 1 else num_frames
        chunk_args = (Ad[:, :, :, start_frame:end_frame], viewR, viewC, numChannels, nanRows, nanCols, 
                      motionC, motionR, start_frame, end_frame)
        chunks.append(chunk_args)

    with Pool(num_cores) as pool:
        results = pool.map(process_chunk, chunks)

    tiffSave_raw = np.concatenate(results, axis=2)
    return tiffSave_raw
    
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

    return parsed_data    
    
def dftups(inp, nor, noc, usfac, roff=0, coff=0):
    nr, nc = inp.shape
    # Compute kernels and obtain DFT by matrix products
    kernc = np.exp(
        (-1j * 2 * np.pi / (nc * usfac))
        * (np.fft.ifftshift(np.arange(nc)) - np.floor(nc/2)).reshape(-1, 1)
        @ (np.arange(noc) - coff).reshape(1, -1)
    )
    kernr = np.exp(
        (-1j * 2 * np.pi / (nr * usfac))
        * (np.arange(nor).reshape(-1, 1) - roff)
        @ (np.fft.ifftshift(np.arange(nr)) - np.floor(nr/2)).reshape(1, -1)
    )
    out = kernr @ inp @ kernc
    return out
    
def dftregistration_clipped(buf1ft, buf2ft, usfac=1, clip=None):
    if clip is None:
        clip = [0, 0]
    elif isinstance(clip, (int, float)):
        clip = [clip, clip]

    # Compute error for no pixel shift
    if usfac == 0:
        CCmax = np.sum(buf1ft * np.conj(buf2ft))
        rfzero = np.sum(np.abs(buf1ft.flatten()) ** 2)
        rgzero = np.sum(np.abs(buf2ft.flatten()) ** 2)
        error = 1.0 - CCmax * np.conj(CCmax) / (rgzero * rfzero)
        error = np.sqrt(np.abs(error))
        diffphase = np.arctan2(np.imag(CCmax), np.real(CCmax))
        output = [error, diffphase]
        return output, None

    # Whole-pixel shift - Compute crosscorrelation by an IFFT and locate the peak
    elif usfac == 1:
        m, n = buf1ft.shape
        md2 = m // 2
        nd2 = n // 2
        CC = np.fft.ifft2(buf1ft * np.conj(buf2ft))

        keep = np.ones(CC.shape, dtype=bool)
        keep[clip[0] // 2 + 1 : -clip[0] // 2, :] = False
        keep[:, clip[1] // 2 + 1 : -clip[1] // 2] = False
        CC[~keep] = 0

        max1 = np.max(np.real(CC), axis=0)
        loc1 = np.argmax(np.real(CC), axis=0)
        max2 = np.max(max1)
        loc2 = np.argmax(max1)
        rloc = loc1[loc2]
        cloc = loc2
        CCmax = CC[rloc, cloc]
        rfzero = np.sum(np.abs(buf1ft.flatten()) ** 2) / (m * n)
        rgzero = np.sum(np.abs(buf2ft.flatten()) ** 2) / (m * n)
        error = 1.0 - CCmax * np.conj(CCmax) / (rgzero * rfzero)
        error = np.sqrt(np.abs(error))
        diffphase = np.arctan2(np.imag(CCmax), np.real(CCmax))

        md2 = m // 2
        nd2 = n // 2
        if rloc > md2:
            row_shift = rloc - m - 1  # Add the -1
        else:
            row_shift = rloc - 1      # Add the -1

        if cloc > nd2:
            col_shift = cloc - n - 1  # Add the -1
        else:
            col_shift = cloc - 1      # Add the -1

        output = [error, diffphase, row_shift, col_shift]
        return output, None

    # Partial-pixel shift
    else:
        m, n = buf1ft.shape
        mlarge = m * 2
        nlarge = n * 2
        CC = np.zeros((mlarge, nlarge), dtype=np.complex128)
        # CC[
        #     m - (m // 2) : m + (m // 2),
        #     n - (n // 2) : n + (n // 2),
        # ] = np.fft.fftshift(buf1ft) * np.conj(np.fft.fftshift(buf2ft))
        # Adjust slicing indices to match (41, 125)
        row_start = m - (m // 2)
        row_end = row_start + buf1ft.shape[0]  # Ensure it matches rows of buf1ft
        col_start = n - (n // 2)
        col_end = col_start + buf1ft.shape[1]  # Ensure it matches columns of buf2ft

        # Perform fftshift and element-wise multiplication with conjugate
        CC[row_start:row_end, col_start:col_end] = (
            np.fft.fftshift(buf1ft) * np.conj(np.fft.fftshift(buf2ft))
        )# Compute crosscorrelation and locate the peak
        CC = np.fft.ifft2(np.fft.ifftshift(CC))  # Calculate cross-correlation

        keep = np.ones(CC.shape, dtype=bool)
        keep[2 * clip[0] + 1 : -2 * clip[0], :] = False
        keep[:, 2 * clip[1] + 1 : -2 * clip[1]] = False
        CC[~keep] = 0

        max1 = np.max(np.real(CC), axis=0)
        loc1 = np.argmax(np.real(CC), axis=0)
        max2 = np.max(max1)
        loc2 = np.argmax(max1)
        max_val = np.max(np.real(CC))
        rloc, cloc = np.unravel_index(np.argmax(np.real(CC)), CC.shape)
        CCmax = CC[rloc, cloc]

        # Obtain shift in original pixel grid from the position of the
        # crosscorrelation peak
        m, n = CC.shape
        md2 = m // 2
        nd2 = n // 2
        if rloc > md2:
            row_shift = rloc - m
        else:
            row_shift = rloc
        if cloc > nd2:
            col_shift = cloc - n
        else:
            col_shift = cloc
        row_shift = row_shift / 2
        col_shift = col_shift / 2

        # If upsampling > 2, then refine estimate with matrix multiply DFT
        if usfac > 2:
            # Initial shift estimate in upsampled grid
            row_shift = matlab_round(row_shift * usfac) / usfac
            col_shift = matlab_round(col_shift * usfac) / usfac
            dftshift = np.fix(np.ceil(usfac * 1.5) / 2)  # Center of output array at dftshift+1
            # Matrix multiply DFT around the current shift estimate
            CC = np.conj(
                dftups(
                    buf2ft * np.conj(buf1ft),
                    np.ceil(usfac * 1.5),
                    np.ceil(usfac * 1.5),
                    usfac,
                    dftshift - row_shift * usfac,
                    dftshift - col_shift * usfac,
                )
            ) / (md2 * nd2 * usfac ** 2)
            # Locate maximum and map back to original pixel grid
            max1 = np.max(np.real(CC), axis=0) #<--------- was set to axis 0
            loc1 = np.argmax(np.real(CC), axis=0) #<--------- was set to axis 0
            max2 = np.max(max1)
            loc2 = np.argmax(max1)
            rloc = loc1[loc2]
            cloc = loc2
            CCmax = CC[rloc, cloc]
            rg00 = dftups(buf1ft * np.conj(buf1ft), 1, 1, usfac) / (md2 * nd2 * usfac ** 2)
            rf00 = dftups(buf2ft * np.conj(buf2ft), 1, 1, usfac) / (md2 * nd2 * usfac ** 2)
            rloc = rloc - dftshift
            cloc = cloc - dftshift
            row_shift = row_shift + rloc / usfac
            col_shift = col_shift + cloc / usfac

            # If upsampling = 2, no additional pixel shift refinement
        else:
            rg00 = np.sum(buf1ft * np.conj(buf1ft)) / m / n
            rf00 = np.sum(buf2ft * np.conj(buf2ft)) / m / n
        error = 1.0 - CCmax * np.conj(CCmax) / (rg00 * rf00)
        error = np.sqrt(np.abs(error))
        diffphase = np.arctan2(np.imag(CCmax), np.real(CCmax))
        # If its only one row or column the shift along that dimension has no
        # effect. We set to zero.
        if md2 == 1:
            row_shift = 0
        if nd2 == 1:
            col_shift = 0
        output = [error, diffphase, row_shift, col_shift]

        # Compute registered version of buf2ft
        if usfac > 0:
            nr, nc = buf2ft.shape
            Nr = np.fft.ifftshift(np.arange(-np.fix(nr / 2), np.ceil(nr / 2)))
            Nc = np.fft.ifftshift(np.arange(-np.fix(nc / 2), np.ceil(nc / 2)))
            Nc, Nr = np.meshgrid(Nc, Nr)
            Greg = buf2ft * np.exp(
                1j * 2 * np.pi * (-row_shift * Nr / nr - col_shift * Nc / nc)
            )
            Greg = Greg * np.exp(1j * diffphase)
        elif usfac == 0:
            Greg = buf2ft * np.exp(1j * diffphase)
        else:
            Greg = None
    return output, Greg
    
def fast_xcorr2_nans(frame, template, shiftsCenter, dShift):
    dShift = round(dShift)  # Sanity check

    SE = np.ones((2 * dShift + 1, 2 * dShift + 1), dtype=np.uint8)

    # Valid pixels of the new frame
    rows, cols = template.shape
    M = np.float32([[1, 0, shiftsCenter[0]], [0, 1, shiftsCenter[1]]])
    tmp = cv2.warpAffine(1-cv2.dilate(np.isnan(template).astype(np.uint8), SE, iterations=1), M, (cols, rows),
                            borderMode=cv2.BORDER_CONSTANT, flags=cv2.INTER_NEAREST).astype(bool)
    fValid = np.zeros(frame.shape, dtype=bool)
    fValid[dShift:-dShift, dShift:-dShift] = ~np.isnan(frame[dShift:-dShift, dShift:-dShift]) & tmp[dShift:-dShift, dShift:-dShift]

    tValid = np.roll(fValid, -shiftsCenter, axis=(0, 1)).astype(bool)

    F = frame[fValid]
    ssF = np.sqrt(F.dot(F))

    # Correlation is sum(A.*B)./(sqrt(ssA)*sqrt(ssB)); ssB is constant though
    tV0, tV1 = np.where(tValid)
    tValidInd = tV0 * cols + tV1

    shifts = np.arange(-dShift, dShift + 1)
    C = np.full((len(shifts), len(shifts)), np.nan)
    # Print datatypes before the loop
    for drix, shift_x in enumerate(shifts):
        for dcix, shift_y in enumerate(shifts):
            shifted_tValid = np.roll(tValid, (-shift_x, -shift_y), axis=(0, 1))
            T = template[shifted_tValid]
            ssT = np.sum(T ** 2)
            
            if ssT > 0:  # Only calculate if ssT is greater than zero
                C[drix, dcix] = np.sum(F * T) / np.sqrt(ssT)
            else:
                C[drix, dcix] = np.nan  # Handle cases where ssT is zero
            
    # plt.imshow(C)
    # Find maximum of correlation map
    maxval = np.nanmax(C)
    I = np.unravel_index(np.nanargmax(C), C.shape)
    rr, cc = I
    R = maxval / ssF  # Correlation coefficient

    if 0 < rr < len(shifts) - 1 and 0 < cc < len(shifts) - 1:
        # Perform superresolution upsampling
        ratioR = min(1e6, (C[rr, cc] - C[rr - 1, cc]) / (C[rr, cc] - C[rr + 1, cc]))
        dR = (1 - ratioR) / (1 + ratioR) / 2
        ratioC = min(1e6, (C[rr, cc] - C[rr, cc - 1]) / (C[rr, cc] - C[rr, cc + 1]))
        dC = (1 - ratioC) / (1 + ratioC) / 2
        motion = shiftsCenter + [shifts[rr] - dR, shifts[cc] - dC]
    else:
        # The optimum is at an edge of search range; no superresolution
        motion = shiftsCenter + [shifts[rr], shifts[cc]]

    if np.any(np.isnan(motion)):
        raise ValueError('Motion result contains NaN values')

    return motion, R

def matlab_round(x):
    """
    Replicates MATLAB's round function behavior
    
    In MATLAB:
    - round(4.5) = 5
    - round(-4.5) = -5
    """
    if x >= 0:
        return int(np.floor(x + 0.5))
    else:
        return int(np.ceil(x - 0.5))

def stripRegistrationBergamo_init(ds_time, initFrames, Ad, maxshift, clipShift, alpha, numChannels, path_template_list, output_path_):
    if ds_time is None:
        ds_time = 3  # the movie is downsampled using averaging in time by a factor of 2^ds_time

    dsFac = 2 ** ds_time

    framesToRead = initFrames * dsFac

    # Downsample the data
    Y = downsampleTime(Ad[:, :, :, :framesToRead], ds_time)

    # Get size of the original data
    sz = Ad.shape

    # Create a list of channels
    selCh = list(range(numChannels))

    # Sum along the third dimension and squeeze the array
    # Yhp = np.squeeze(np.sum(Y,2))

    Yhp = np.sum(Y[:, :, selCh, :].reshape(Y.shape[0], Y.shape[1], -1, Y.shape[3]), axis=2).squeeze()

    # Reshape Yhp to 2D where each column is a frame
    reshaped_Yhp = Yhp.reshape(-1, Yhp.shape[2])
    
    # Calculate the correlation matrix
    rho = np.corrcoef(reshaped_Yhp.T)

    # Compute the distance matrix
    dist_matrix = 1 - rho

    # Ensure the distance matrix is symmetric
    dist_matrix = (dist_matrix + dist_matrix.T) / 2
    
    # Ensure the diagonal is zero (optional but often necessary for distance matrices)
    np.fill_diagonal(dist_matrix, 0)
    
    # Check if the matrix is symmetric
    is_symmetric = np.array_equal(dist_matrix, dist_matrix.T)
    print(f"Is the distance matrix symmetric? {is_symmetric}")

    # Perform hierarchical clustering using average linkage
    Z = linkage(squareform(dist_matrix), method='average')

    Z[:, :2] = np.ceil(Z[:, :2]).astype(int)

    # Z = Z[:, :-1] # Matlab produces just 3 columns

    # Define the cutoff value
    cutoff = 0.01

    # Define the minimum cluster size
    min_cluster_size = 100

    # Initialize an empty list for clusters
    clusters = []

    # Define the maximum cutoff value
    max_cutoff = 2.0

    # Initialize variables
    cutoff = 0.01
    min_cluster_size = 100
    clusters = []
    max_cutoff = 2.0

    while not clusters or all(len(cluster) < min_cluster_size for cluster in clusters):
        cutoff += 0.01
        if cutoff > max_cutoff:
            raise ValueError(f"Could not find a cluster with at least {min_cluster_size} samples")
        
        # Perform clustering with the current cutoff
        T = fcluster(Z, cutoff, criterion='distance')
        
        # Group indices by cluster label
        clusters = [np.where(T == label)[0] for label in np.unique(T)]

    # Initialize max_mean_corr to negative infinity
    max_mean_corr = -np.inf

    # Iterate over each cluster
    for cluster_indices in clusters:
        # Check if the cluster size meets the minimum requirement
        if len(cluster_indices) >= min_cluster_size:
            # Compute the mean correlation within the cluster
            mean_corr = np.mean(np.mean(rho[np.ix_(cluster_indices, cluster_indices)]))
            
            # Update max_mean_corr and best_cluster if necessary
            if mean_corr > max_mean_corr:
                max_mean_corr = mean_corr
                best_cluster = cluster_indices

    corrector = MotionCorrect(lazy_dataset=Yhp.transpose(2, 0, 1),
                            max_shifts=(maxshift, maxshift),  # Maximum allowed shifts in pixels
                            strides=(48, 48),  # Patch dimensions for piecewise rigid correction
                            overlaps=(24, 24),  # Overlap between patches
                            max_deviation_rigid=3,  # Maximum deviation for rigid correction
                            pw_rigid = False)  # Number of frames to process in a batch
    # TODO: frames_per_split set it to initFrames (Matlab)
    
    # corrected_frames = np.zeros_like(Yhp)
    # for i in range(Yhp.shape[2]):
    #     frame = Yhp[:, :, i]
    #     corrected_frame = corrector.register_frames(frame)

    #     corrected_frames[:, :, i] = corrected_frame
    print('best_cluster----->', best_cluster)
    frame_corrector, output_file = corrector.motion_correct(
        template=np.mean(Yhp[:, :, best_cluster], axis=2), save_movie=True
    )

    # Get the current working directory
    cwd = os.getcwd()

    # Get path to the template movie generated by jnormcorre
    path_template = os.path.join(cwd, ' '.join(output_file))
    path_template_list.append(path_template)

    # Check if the path exists
    if os.path.exists(path_template):
        print(f"The template {path_template} exists.")
    else:
        print(f"The template {path_template} does not exist.")

    template = ScanImageTiffReader(path_template) # TODO: Replace with tifffilereader

    F = template.data()
    F = np.transpose(F, (1, 2, 0))
    F = np.mean(F, axis=2)
    # F = loadmat('/root/capsule/scratch/michael_template_F.mat')['F'] #TODO: Remove this

    # Create a template with NaNs
    template = np.full((2*maxshift + sz[0], 2*maxshift + sz[1]), np.nan)
    templateFull = template

    # Insert the matrix F into the template
    template[maxshift:maxshift+sz[0], maxshift:maxshift+sz[1]] = F

    # template = np.transpose(template)
    print('template Shape----->', template.shape)
    # Copy the template to T0
    T0 = template.copy()

    # Create T00 as a zero matrix of the same size as template
    T00 = np.zeros_like(template)
    templateCt = np.zeros_like(template)

    initR = 0
    initC = 0

    # Calculate the number of downsampled frames
    nDSframes = np.floor(sz[3] / dsFac).astype(int)  # Using sz[3] as it corresponds to MATLAB's sz(4)

    print('Strip Registeration...')
    aData = {} # Alignment data dictionary
    
    # Initialize arrays to store the inferred motion and errors
    motionDSr = np.nan * np.ones(nDSframes)
    motionDSc = np.nan * np.ones(nDSframes)
    aErrorDS = np.nan * np.ones(nDSframes)
    aRankCorr = np.nan * np.ones(nDSframes)
    recNegErr = np.nan * np.ones(nDSframes)

    # Create view matrices for interpolation
    viewR, viewC = np.meshgrid(
        np.arange(0, sz[0] + 2 * maxshift) - maxshift,
        np.arange(0, sz[1] + 2 * maxshift) - maxshift,
        indexing='ij'
    )

    for DSframe in range(nDSframes):
        # if DSframe == 177:
        #     break
        
        read_start = DSframe * dsFac
        read_end = read_start + dsFac
        readFrames = np.arange(read_start, read_end)
        
        M = downsampleTime(Ad[:, :, :, readFrames], ds_time)
        M = np.sum(M[:,:,selCh,:].reshape(M.shape[0], M.shape[1], -1, M.shape[3]), axis=2).squeeze()
        
        if DSframe % 1000 == 0:
            print(f'{DSframe} of {nDSframes}')

        Ttmp = np.nanmean(np.stack((T0, T00, template), axis=2), axis=2)
        T = Ttmp[
        maxshift - initR : maxshift - initR + sz[0],
        maxshift - initC : maxshift - initC + sz[1]
        ]
        
        output,_ = dftregistration_clipped(fft2(M.astype(np.float32)), fft2(T.astype(np.float32)), 4, clipShift)
        
        outputArray.append(output) # TODO: Remove this after debug
        
        motionDSr[DSframe] = initR + output[2]
        motionDSc[DSframe] = initC + output[3]
        aErrorDS[DSframe] = output[0]
        
        if np.sqrt((motionDSr[DSframe]/sz[0])**2 + (motionDSc[DSframe]/sz[1])**2) > 0.75**2:
            x = np.arange(sz[1])
            y = np.arange(sz[0])
            mesh_x, mesh_y = np.meshgrid(x, y)
            
            start_cv2_remap = time.time()
            map_x = np.float32(viewC)
            map_y = np.float32(viewR)
            Mfull = cv2.remap(M.astype(np.float32), map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
            
            motion, R = fast_xcorr2_nans(Mfull.astype(np.float32), Ttmp.astype(np.float32), np.array([initR, initC]), 50)
            
            xcorreArray.append(motion) # TODO: Remove this after debug
            
            motionDSr[DSframe] = motion[0]
            motionDSc[DSframe] = motion[1]
            aErrorDS[DSframe] = R
        
        if abs(motionDSr[DSframe]) < maxshift and abs(motionDSc[DSframe]) < maxshift:
            X, Y = np.meshgrid(np.arange(0, sz[1]), np.arange(0, sz[0]))
            
            Xq = viewC + motionDSc[DSframe]
            Yq = viewR + motionDSr[DSframe]
            
            A = cv2.remap(M, Xq.astype(np.float32), Yq.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
            
            Asmooth = cv2.GaussianBlur(A, (0, 0), sigmaX=1)
            
            selCorr = ~np.isnan(Asmooth) & ~np.isnan(Ttmp)
            if np.sum(selCorr) > 0:
                aRankCorr[DSframe] = np.corrcoef(Asmooth[selCorr].flatten(), Ttmp[selCorr].flatten())[0, 1]
                recNegErr[DSframe] = np.mean(
                    np.minimum(0, Asmooth[selCorr] * np.mean(Ttmp[selCorr]) / np.mean(Asmooth[selCorr]) - Ttmp[selCorr]
                )**2)
            
            # Sum along the third dimension, ignoring NaNs
            templateFull = np.nansum(np.stack((templateFull * templateCt, A), axis=2), axis=2)

            # Update the count where A is not NaN
            templateCt = templateCt + ~np.isnan(A)

            # Compute the average
            templateFull = templateFull / (templateCt + np.finfo(float).eps)

            # Assign to template
            template = templateFull.copy()

            # Set values to NaN where the count is less than 100
            template[templateCt < 100] = np.nan
            
            initR = matlab_round(motionDSr[DSframe]) 
            initC = matlab_round(motionDSc[DSframe])

            intArray.append([initR, initC])
        else:
            motionDSr[DSframe] = initR
            motionDSc[DSframe] = initC

    # **1. Time Vector Calculation (tDS)**
    tDS = (np.arange(1, nDSframes + 1) * dsFac) - (2**(ds_time - 1)) + 0.5

    # **2. Upsampling Points**
    upsample_factor = 2 ** ds_time
    desired_length = upsample_factor * nDSframes

    # Use linspace to include the endpoint
    upsample_points = np.linspace(1, desired_length, num=desired_length)

    # **3. Interpolation for motionC**
    motionC_interp = PchipInterpolator(tDS, motionDSc, extrapolate=True)
    motionC = motionC_interp(upsample_points)

    # **4. Interpolation for motionR**
    motionR_interp = PchipInterpolator(tDS, motionDSr, extrapolate=True)
    motionR = motionR_interp(upsample_points)

    # **5. Nearest Interpolation for aError**
    aError_interp = interp1d(
        tDS,
        aErrorDS,
        kind='nearest',
        fill_value='extrapolate'
    )

    aError = aError_interp(upsample_points)

    maxshiftC = np.max(np.abs(motionC))
    maxshiftR = np.max(np.abs(motionR))

    viewR, viewC = np.meshgrid(
        np.arange(0, sz[0] + 2 * maxshiftR) - maxshiftR,
        np.arange(0, sz[1] + 2 * maxshiftC) - maxshiftC,
        indexing='ij'  # This makes meshgrid behave like MATLAB's ndgrid
    )

    tif_path = output_path_ # Use orig
    # Initialize an empty list to store the interpolated images
    interpolated_images = []

    # Save downsampled data
    downsampled_interpolated_images = []
    base_name, ext = os.path.splitext(tif_path)
    downsampled_tif_path = f"{base_name}_DOWNSAMPLED-{dsFac}x{ext}"

    # Bsum = np.zeros((viewR.shape[0], viewR.shape[1], numChannels))
    # Bcount = np.zeros((viewR.shape[0], viewR.shape[1], numChannels))

    # Initialize tiffSave array
    tiffSave = np.zeros((viewR.shape[1], viewR.shape[0], nDSframes * numChannels), dtype=np.float32)

    # Initialize Bcount and Bsum arrays
    Bcount = np.zeros((viewR.shape[0], viewR.shape[1], numChannels), dtype=np.float32)
    Bsum = np.zeros((viewR.shape[0], viewR.shape[1], numChannels), dtype=np.float64)

    tiffSave_time_start = time.time()
    for DSframe in range(nDSframes):
        readFrames = (DSframe * dsFac) + np.arange(dsFac)
        YY = downsampleTime(Ad[:, :, :, readFrames], ds_time)
        
        for ch in range(numChannels):
            # Perform interpolation using cv2.remap
            map_x = (viewC + motionDSc[DSframe]).astype(np.float32)
            map_y = (viewR + motionDSr[DSframe]).astype(np.float32)
            B = cv2.remap(YY[:, :, ch], map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
            
            # Assign to tiffSave
            tiffSave[:, :, DSframe * numChannels + ch] = B.T.astype(np.float32)
            
            # Update Bcount and Bsum
            Bcount[:, :, ch] += ~np.isnan(B)
            B[np.isnan(B)] = 0
            Bsum[:, :, ch] += B.astype(np.float64)

    tiffSave_time_end = time.time()
    print(f"Total time for tiffSave_raw: {tiffSave_time_end - tiffSave_time_start:.4f} seconds\n")
    
    # Calculate nanRows and nanCols correctly
    nanRows = np.mean(np.mean(np.isnan(tiffSave), axis=2), axis=1) == 1
    nanCols = np.mean(np.mean(np.isnan(tiffSave), axis=2), axis=0) == 1
    
    # Apply boolean indexing to tiffSave
    tiffSave = tiffSave[~nanRows, :, :]
    tiffSave = tiffSave[:, ~nanCols, :]

    # Transpose to match ImageJ convention (time, Y, X)
    tiffSave = tiffSave.transpose((2, 1, 0))

    # Save tiff
    tifffile.imwrite(downsampled_tif_path, tiffSave, imagej=True, metadata={'axes': 'ZYX'}, maxworkers=os.cpu_count())

    # Remove from tiffSave memory
    del tiffSave
    del Yhp
    del Y

    # Save an average image for each channel
    for ch in range(1, numChannels + 1):
        Bmean = Bsum[:, :, ch - 1] / Bcount[:, :, ch - 1]
        minV = np.percentile(Bmean[~np.isnan(Bmean)], 10)
        maxV = np.max(Bmean[~np.isnan(Bmean)])
        Bmean = (255 * np.sqrt(np.maximum(0, (Bmean - minV) / (maxV - minV)))).astype(np.uint8)
        
        # Convert nanRows and nanCols to integer arrays
        nanRows_mean_ch = np.array(nanRows, dtype=int)
        nanCols_mean_ch = np.array(nanCols, dtype=int)

        # Delete rows and columns
        Bmean = Bmean[~nanCols, :]
        Bmean = Bmean[:, ~nanRows]        
        channel_mean_path = f"{base_name}_REGISTERED_AVG_CH{ch}_8bit.tif"
        tifffile.imwrite(channel_mean_path, Bmean)  

    raw_tif_path = f"{base_name}_REGISTERED_RAW{ext}"
    tiffSave_raw_time_start = time.time()  
    tiffSave_raw = process_raw_frames_cpu(Ad, viewR, viewC, numChannels, nanRows, nanCols, motionC, motionR)
    tiffSave_raw_time_end = time.time()
    print(f"Total time for tiffSave_raw: {tiffSave_raw_time_end - tiffSave_raw_time_start:.4f} seconds\n")
    del Ad
    
    # Transpose to match ImageJ convention (time, Y, X)
    tiffSave_raw_time_transpose_start = time.time()
    tiffSave_raw = tiffSave_raw.transpose((2, 1, 0))
    tiffSave_raw_time_transpose_end = time.time()
    print(f"Total time for tiffSave_raw transpose: {tiffSave_raw_time_transpose_end - tiffSave_raw_time_transpose_start:.4f} seconds\n")
    
    # Save with correct metadata
    tiffSave_raw_time_tiffwrite_start = time.time()
    tifffile.imwrite(raw_tif_path, tiffSave_raw, bigtiff=True, imagej=True, metadata={'axes': 'ZYX'}, maxworkers=os.cpu_count())
    tiffSave_raw_time_tiffwrite_end = time.time()
    print(f"Total time for tiffSave_raw transpose: {tiffSave_raw_time_tiffwrite_end - tiffSave_raw_time_tiffwrite_start:.4f} seconds\n")

    # Remove from tiffSave memory
    del tiffSave_raw

    motionR_mean = np.mean(motionR)
    motionC_mean = np.mean(motionC)

    # Save alignment data
    aData['numChannels'] = 1
    aData['frametime'] =  0.0023 #params['frametime'] #TODO: A flag to switch between sim and actual data and avoiding hard coding. 
    aData['motionR'] = motionR #- motionR_mean
    aData['motionC'] = motionC #- motionC_mean
    aData['aError'] = aError
    aData['aRankCorr'] = aRankCorr
    aData['motionDSc'] = motionDSc
    aData['motionDSr'] = motionDSr
    aData['recNegErr'] = recNegErr

    base_name, ext = os.path.splitext(tif_path)
    alignmentData_h5_path = f"{base_name}_ALIGNMENTDATA.h5"
    with h5py.File(alignmentData_h5_path, "w") as f:
        print(f'Writing {alignmentData_h5_path} as h5...')
        f.create_dataset("aData/numChannels", data=aData['numChannels'])
        f.create_dataset("aData/frametime", data=aData['frametime'])
        f.create_dataset("aData/motionR", data=aData['motionR'], compression="gzip")
        f.create_dataset("aData/motionC", data=aData['motionC'], compression="gzip")
        f.create_dataset("aData/aError", data=aData['aError'], compression="gzip")
        f.create_dataset("aData/aRankCorr", data=aData['aRankCorr'], compression="gzip")
        f.create_dataset("aData/motionDSc", data=aData['motionDSc'], compression="gzip")
        f.create_dataset("aData/motionDSr", data=aData['motionDSr'], compression="gzip")
        f.create_dataset("aData/recNegErr", data=aData['recNegErr'], compression="gzip")

    # Center the shifts to zero 
    return aData['motionR'], aData['motionC'], path_template_list
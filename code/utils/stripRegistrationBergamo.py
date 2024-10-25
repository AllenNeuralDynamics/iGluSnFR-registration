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
from utils.xcorr2_nans import xcorr2_nans as cython_xcorr2_nans
from multiprocessing import Pool, cpu_count
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import spearmanr
import warnings

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
    
def fast_xcorr2_nans(frame, template, shiftsCenter, dShift):
    """
    Perform a somewhat-efficient local normalized cross-correlation for images with NaNs.
    
    Parameters:
    - frame: the frame to be aligned; this has more NaNs
    - template: the template
    - shiftsCenter: the center offset around which to perform a local search
    - dShift: the maximum shift (scalar, in pixels) to consider on each axis around shiftsCenter
    
    Returns:
    - motion: the calculated motion vector
    - R: the correlation coefficient
    """
    # Create structuring element for dilation
    SE = np.ones((2 * dShift + 1, 2 * dShift + 1), dtype=np.uint8)

    # Valid pixels of the new frame
    rows, cols = template.shape
    M = np.float32([[1, 0, shiftsCenter[0]], [0, 1, shiftsCenter[1]]])
    tmp = cv2.warpAffine(1-cv2.dilate(np.isnan(template).astype(np.uint8), SE, iterations=1), M, (cols, rows),
                            borderMode=cv2.BORDER_CONSTANT, flags=cv2.INTER_NEAREST).astype(bool)
    fValid = np.zeros(frame.shape, dtype=bool)
    fValid[dShift:-dShift, dShift:-dShift] = ~np.isnan(frame[dShift:-dShift, dShift:-dShift]) & tmp[dShift:-dShift, dShift:-dShift]

    tValid = cv2.warpAffine(fValid.astype(np.uint8), M, (cols, rows)).astype(bool)

    F = frame[fValid]  # fixed data
    ssF = np.sqrt(F.dot(F))

    # Correlation is sum(A.*B)./(sqrt(ssA)*sqrt(ssB)); ssB is constant though
    tV0, tV1 = np.where(tValid)
    tValidInd = tV0 * cols + tV1
    shifts = np.arange(-dShift, dShift + 1)
    C = np.full((len(shifts), len(shifts)), np.nan, dtype="f4")
    cython_xcorr2_nans(F.ravel(), template.ravel(), tValidInd.astype("i4"), shifts.astype("i4"), cols, C)
                
    # Find maximum of correlation map
    maxval = np.nanmax(C)
    if np.isnan(maxval):
        raise ValueError("All-NaN slice encountered in cross-correlation.")
    
    rr, cc = np.unravel_index(np.nanargmax(C), C.shape)
    R = maxval / ssF  # correlation coefficient

    if 1 < rr < len(shifts) - 1 and 1 < cc < len(shifts) - 1:
        # Perform superresolution upsampling
        ratioR = min(1e6, (C[rr, cc] - C[rr - 1, cc]) / (C[rr, cc] - C[rr + 1, cc]))
        dR = (1 - ratioR) / (1 + ratioR) / 2

        ratioC = min(1e6, (C[rr, cc] - C[rr, cc - 1]) / (C[rr, cc] - C[rr, cc + 1]))
        dC = (1 - ratioC) / (1 + ratioC) / 2

        motion = shiftsCenter + np.array([shifts[rr] - dR, shifts[cc] - dC])
    else:
        # The optimum is at an edge of search range; no superresolution
        motion = shiftsCenter + np.array([shifts[rr], shifts[cc]])

    if np.any(np.isnan(motion)):
        raise ValueError("NaN encountered in motion calculation.")
    return motion, R

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

        max1 = np.max(np.real(CC), axis=1)
        loc1 = np.argmax(np.real(CC), axis=1)
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
            row_shift = rloc - m
        else:
            row_shift = rloc

        if cloc > nd2:
            col_shift = cloc - n
        else:
            col_shift = cloc

        output = [error, diffphase, row_shift, col_shift]
        return output, None

    # Partial-pixel shift
    else:
        # First upsample by a factor of 2 to obtain initial estimate
        # Embed Fourier data in a 2x larger array
        m, n = buf1ft.shape
        mlarge = m * 2
        nlarge = n * 2
        CC = np.zeros((mlarge, nlarge), dtype=np.complex128)
        CC[
            m - (m // 2) : m + (m // 2),
            n - (n // 2) : n + (n // 2),
        ] = np.fft.fftshift(buf1ft) * np.conj(np.fft.fftshift(buf2ft))

        # Compute crosscorrelation and locate the peak
        CC = np.fft.ifft2(np.fft.ifftshift(CC))  # Calculate cross-correlation

        keep = np.ones(CC.shape, dtype=bool)
        keep[2 * clip[0] + 1 : -2 * clip[0], :] = False
        keep[:, 2 * clip[1] + 1 : -2 * clip[1]] = False
        CC[~keep] = 0

        max1 = np.max(np.real(CC), axis=1)
        loc1 = np.argmax(np.real(CC), axis=1)
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
            row_shift = round(row_shift * usfac) / usfac
            col_shift = round(col_shift * usfac) / usfac
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
            max1 = np.max(np.real(CC), axis=1)
            loc1 = np.argmax(np.real(CC), axis=1)
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

def dftups(inp, nor, noc, usfac, roff=0, coff=0):
    nr, nc = inp.shape
    # Compute kernels and obtain DFT by matrix products
    kernc = np.exp(
        (-1j * 2 * np.pi / (nc * usfac))
        * (np.fft.ifftshift(np.arange(nc)).reshape(-1, 1) - np.floor(nc / 2))
        * (np.arange(noc) - coff)
    )
    kernr = np.exp(
        (-1j * 2 * np.pi / (nr * usfac))
        * (np.arange(nor).reshape(-1, 1) - roff)
        * (np.fft.ifftshift(np.arange(nr)) - np.floor(nr / 2))
    )
    out = np.dot(np.dot(kernr, inp), kernc)
    return out


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

    # corrected_frames = np.zeros_like(Yhp)
    # for i in range(Yhp.shape[2]):
    #     frame = Yhp[:, :, i]
    #     corrected_frame = corrector.register_frames(frame)

    #     corrected_frames[:, :, i] = corrected_frame

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

    # Create a template with NaNs
    template = np.full((2*maxshift + sz[0], 2*maxshift + sz[1]), np.nan)

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
        read_start = DSframe * dsFac
        read_end = read_start + dsFac
        readFrames = np.arange(read_start, read_end)
        
        M = downsampleTime(Ad[:, :, :, readFrames], ds_time)
        M = np.sum(M[:,:,selCh,:].reshape(M.shape[0], M.shape[1], -1, M.shape[3]), axis=2).squeeze()
        
        if DSframe % 1000 == 0:
            print(f"Frame {DSframe} of {nDSframes}: MotionDSc={motionDSc[DSframe]}, MotionDSr={motionDSr[DSframe]}, aErrorDS={aErrorDS[DSframe]}")
        
        Ttmp = np.nanmean(np.stack([T0, T00, template], axis=2), axis=2)
        T = Ttmp[maxshift - initR : maxshift - initR + sz[0],
                maxshift - initC : maxshift - initC + sz[1]]
        
        # Perform DFT-based registration
        output, _ = dftregistration_clipped(fft2(M), fft2(T.astype(np.float32)), 4, clipShift)
        
        # Handle DeprecationWarning by ensuring scalar assignment
        try:
            aErrorDS[DSframe] = output[0].item() if hasattr(output[0], 'item') else output[0].squeeze()
        except AttributeError:
            aErrorDS[DSframe] = output[0]  # Fallback if .item() is not available
        
        motionDSr[DSframe] = initR + output[2]
        motionDSc[DSframe] = initC + output[3]
        
        # Limit the motion correction
        motionDSr[DSframe] = np.clip(motionDSr[DSframe], -maxshift, maxshift)
        motionDSc[DSframe] = np.clip(motionDSc[DSframe], -maxshift, maxshift)
        
        norm_motion = np.sqrt((motionDSr[DSframe] / sz[0])**2 + (motionDSc[DSframe] / sz[1])**2)
        y = np.arange(sz[0])
        x = np.arange(sz[1])
        if norm_motion > 0.75**2:  # Adjusted threshold
            x = np.arange(sz[1])
            y = np.arange(sz[0])
            mesh_x, mesh_y = np.meshgrid(x, y)
            
            start_cv2_remap = time.time()
            map_x = np.float32(viewC)
            map_y = np.float32(viewR)
            Mfull = cv2.remap(M.astype(np.float32), map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
            
            # Validate interpolation output
            nan_ratio = np.isnan(Mfull).sum() / Mfull.size
            if nan_ratio > 0.1:
                # print(f"High NaN ratio in Mfull: {nan_ratio}. Skipping motion update for frame {DSframe}.")
                motionDSr[DSframe] = initR
                motionDSc[DSframe] = initC
                continue
            
            motion, R = fast_xcorr2_nans(Mfull.astype(np.float32), Ttmp.astype(np.float32), np.array([initR, initC]), 50)
            
            motionDSr[DSframe] = motion[0]
            motionDSc[DSframe] = motion[1]
            aErrorDS[DSframe] = R
        
        # # Ensure motion stays within limits
        # motionDSr[DSframe] = np.clip(motionDSr[DSframe], -maxshift, maxshift)
        # motionDSc[DSframe] = np.clip(motionDSc[DSframe], -maxshift, maxshift)
        
        if abs(motionDSr[DSframe]) < maxshift and abs(motionDSc[DSframe]) < maxshift:
            X, Y = np.meshgrid(np.arange(0, sz[1]), np.arange(0, sz[0]))
            
            Xq = viewC + motionDSc[DSframe]
            Yq = viewR + motionDSr[DSframe]
            
            A = cv2.remap(M, Xq.astype(np.float32), Yq.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
            
            Asmooth = cv2.GaussianBlur(A, (0, 0), sigmaX=1)
            
            selCorr = ~np.isnan(Asmooth) & ~np.isnan(Ttmp)
            if np.any(selCorr):
                # Suppress ConstantInputWarning temporarily
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    if np.std(Asmooth[selCorr]) > 0 and np.std(Ttmp[selCorr]) > 0 and len(Asmooth[selCorr]) > 1:
                        aRankCorr[DSframe], _ = spearmanr(Asmooth[selCorr], Ttmp[selCorr])
                    else:
                        aRankCorr[DSframe] = np.nan  # or assign a default value like 0
                # Compute recNegErr safely
                try:
                    recNegErr[DSframe] = np.mean(
                        np.minimum(
                            0,
                            (Asmooth[selCorr] * np.mean(Ttmp[selCorr]) / np.mean(Asmooth[selCorr])) - Ttmp[selCorr]
                        )**2
                    )
                except ZeroDivisionError:
                    recNegErr[DSframe] = np.nan  # or another default/error value
            
            # Update the template safely
            template = np.nansum(np.stack([template * templateCt, A], axis=2), axis=2)
            templateCt += ~np.isnan(A)
            
            # Avoid division by zero or NaN by replacing invalid entries
            valid_mask = templateCt >= 100  # As per the original condition
            template[valid_mask] /= templateCt[valid_mask]
            template[~valid_mask] = np.nan  # Assign NaN where the count is insufficient
            
            # Monitor and reset template if necessary
            # template_nan_ratio = np.isnan(template).sum() / template.size
            # if template_nan_ratio > 0.2:
            #     # print(f"High NaN ratio in template: {template_nan_ratio}. Resetting template.")
            #     template = np.zeros_like(template)
            #     templateCt = np.zeros_like(templateCt)
            
            initR = int(round(motionDSr[DSframe]))
            initC = int(round(motionDSc[DSframe]))
        else:
            motionDSr[DSframe] = initR
            motionDSc[DSframe] = initC

        end_total = time.time()  # End timing for the entire loop
        # print(f"Total time for DSframe {DSframe}: {end_total - start_total:.4f} seconds\n")
    
    # Assuming motionDSr is already defined as a numpy array or a list
    plt.figure(figsize=(70, 12))
    plt.plot(motionDSr)
    plt.xlabel('Frame')
    plt.ylabel('Motion DSr')
    plt.title('Motion DSr over Frames')
    plt.savefig('motionDSr_plot.png', dpi=500)

    # Upsample the shifts and compute a tighter field of view
    tDS = np.multiply(np.arange(1, nDSframes+1), dsFac) - 2**(ds_time-1) + 0.5

    # interp_func = interp1d(tDS, motionDSc, kind='linear', fill_value='extrapolate', copy = False)
    # motionC = interp_func(np.arange(0, 2**ds_time * nDSframes))
    # interp_func = interp1d(tDS, motionDSr, kind='linear', fill_value='extrapolate', copy = False)
    # motionR = interp_func(np.arange(0, 2**ds_time * nDSframes))

    # interp_func = interp1d(tDS, aErrorDS, kind='nearest', fill_value='extrapolate', copy = False)
    # aError = interp_func(np.arange(0, 2**ds_time * nDSframes))
    # Create the new time points
    new_time_points = np.arange(0, (2**ds_time) * nDSframes)

    # Pchip Interpolator for motionC and motionR with extrapolation
    pchip_interpolator_c = PchipInterpolator(tDS, motionDSc, extrapolate=True)
    pchip_interpolator_r = PchipInterpolator(tDS, motionDSr, extrapolate=True)
    motionC = pchip_interpolator_c(new_time_points)
    motionR = pchip_interpolator_r(new_time_points)

    # Nearest neighbor interpolation for aError with extrapolation
    nearest_interpolator = interp1d(tDS, aErrorDS, kind='nearest', fill_value='extrapolate')
    aError = nearest_interpolator(new_time_points)

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
    for ch in range(1, numChannels+1):
        Bmean = Bsum[:,:,ch-1] / Bcount[:,:,ch-1]
        minV = np.percentile(Bmean[~np.isnan(Bmean)], 10)
        maxV = np.percentile(Bmean[~np.isnan(Bmean)], 99.9)
        Bmean = (255 * np.sqrt(np.maximum(0, (Bmean - minV) / (maxV - minV)))).astype(np.uint8)
        # Convert nanRows and nanCols to integer arrays
        nanRows_mean_ch = np.array(nanRows, dtype=int)
        nanCols_mean_ch = np.array(nanCols, dtype=int)
        
        # Delete rows and columns
        Bmean = np.delete(Bmean, nanRows_mean_ch, axis=0)
        Bmean = np.delete(Bmean, nanCols_mean_ch, axis=1)
        channel_mean_path = f"{base_name}_REGISTERED_AVG_CH{ch}_8bit.tif"
        tifffile.imwrite(channel_mean_path, Bmean.astype(np.float32))

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
    aData['motionR'] = motionR - motionR_mean
    aData['motionC'] = motionC - motionC_mean
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
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

@jit(float32[:,:](float32[:], float32[:], int64[:], int64[:], int64), nopython=False, error_model="numpy", parallel=True)
def optimized_calculate_correlation(F, templ, tValidInd, shifts, cols):
    n = shifts.shape[0]
    C = np.empty((n,n), dtype=np.float32)
    for drix in prange(n):
        for dcix in range(n):
            T = templ[tValidInd - shifts[drix] * cols - shifts[dcix]]
            ssT = T.dot(T)
            C[drix, dcix] = np.ascontiguousarray(F).dot(T) / np.sqrt(ssT)
    return C

def xcorr2_nans(frame, template, shiftsCenter, dShift):
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
    # Create a kernel for dilation using OpenCV
    kernel_size = (2 * dShift + 1, 2 * dShift + 1)
    SE = np.ones(kernel_size, dtype=np.uint8)

    # Measure the time for the dilation operation
    start_dilation = time.time()

    # Use cv2.dilate to handle dilation operation
    template_nan_mask = np.isnan(template).astype(np.uint8)
    dilated_template_nan_mask = cv2.dilate(template_nan_mask, SE)

    # Shift the dilated mask and create valid mask for frame
    fValid = ~np.isnan(frame) & shift(~dilated_template_nan_mask.astype(bool), shiftsCenter)
    end_dilation = time.time()
    dilation_time = end_dilation - start_dilation
    print(f"Dilation took {dilation_time:.6f} seconds")

    fValid[:dShift, :] = False
    fValid[-dShift:, :] = False
    fValid[:, :dShift] = False
    fValid[:, -dShift:] = False

    # Get the dimensions of the image
    rows, cols = fValid.shape[:2]

    # Create the transformation matrix for shifting
    M = np.float32([[1, 0, shiftsCenter[0]], [0, 1, shiftsCenter[1]]])

    # Apply warpAffine with the translation matrix
    tValid = cv2.warpAffine(fValid.astype(np.uint8), M, (cols, rows)).astype(bool)

    F = frame[fValid] # fixed data
    ssF = np.sqrt(np.sum(F**2))

    shifts = np.arange(-dShift, dShift + 1)
    C = np.full((len(shifts), len(shifts)), np.nan)

    # Measure time for the for loop
    start_loop = time.time()

    #generate drix, dcix in advance. Try to do the translation 
    # for drix, dcix in np.ndindex(len(shifts), len(shifts)):
    #     T = template[np.roll(tValid, (-shifts[drix], -shifts[dcix]), axis=(0, 1))]
    #     ssT = np.sum(T**2)
    #     C[drix, dcix] = np.sum(F * T) / np.sqrt(ssT)

    tV0, tV1 = np.where(tValid)
    tValidInd = tV0 * cols + tV1
    shifts = np.arange(-dShift, dShift + 1)

    C = optimized_calculate_correlation(F.ravel(), template.ravel().astype(np.float32), tValidInd, shifts, cols)

    end_loop = time.time()
    loop_time = end_loop - start_loop
    print(f"For loop took {loop_time:.6f} seconds")
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
    
    print('Ad shape---->', Ad.shape)

    dsFac = 2 ** ds_time

    print('dsFac----->', dsFac)
    print('ds_time----->', ds_time)

    framesToRead = initFrames * dsFac

    print('framesToRead----->', framesToRead)
    # Downsample the data
    Y = downsampleTime(Ad[:, :, :, :framesToRead], ds_time)

    # Get size of the original data
    sz = Ad.shape

    # Create a list of channels
    selCh = list(range(numChannels))

    # Sum along the third dimension and squeeze the array
    # Yhp = np.squeeze(np.sum(Y,2))

    print('Y shape---->', Y.shape)
    Yhp = np.sum(Y[:, :, selCh, :].reshape(Y.shape[0], Y.shape[1], -1, Y.shape[3]), axis=2).squeeze()

    print('Yhp shape---->', Yhp.shape)

    # Reshape Yhp to 2D where each column is a frame
    reshaped_Yhp = Yhp.reshape(-1, Yhp.shape[2])

    # print('reshaped_Yhp shape---->', reshaped_Yhp.shape)
    
    # Calculate the correlation matrix
    rho = np.corrcoef(reshaped_Yhp.T)

    # print('rho shape---->', rho.shape)

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
    # print('Z shape---->', Z.shape)
    # print('Z---->', Z)
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
    
    # print('T shape---->', T.shape)
    # print('T---->', T)
    # print('clusters shape---->', len(clusters))
    # print('clusters---->', clusters)

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
    
    print('best_cluster---->', best_cluster)
    print('max_mean_corr---->', max_mean_corr)

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

    print('np.mean(Yhp[:, :, best_cluster], axis=2)', np.mean(Yhp[:, :, best_cluster], axis=2).shape)
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
    print('F = template.data()-------->', F.shape)
    F = np.transpose(F, (1, 2, 0))
    print('F transpose-------->', F.shape)
    F = np.mean(F, axis=2)
    print('F mean-------->', F.shape)

    # Create a template with NaNs
    template = np.full((2*maxshift + sz[0], 2*maxshift + sz[1]), np.nan)

    # Insert the matrix F into the template
    template[maxshift:maxshift+sz[0], maxshift:maxshift+sz[1]] = F

    # template = np.transpose(template)

    # Copy the template to T0
    T0 = template.copy()

    # Create T00 as a zero matrix of the same size as template
    T00 = np.zeros_like(template)

    initR = 0
    initC = 0

    # Calculate the number of downsampled frames
    nDSframes = np.floor(sz[3] / dsFac).astype(int)  # Using sz[3] as it corresponds to MATLAB's sz(4)

    # Initialize arrays to store the inferred motion and errors
    motionDSr = np.nan * np.ones(nDSframes)
    motionDSc = np.nan * np.ones(nDSframes)
    aErrorDS = np.nan * np.ones(nDSframes)
    aRankCorr = np.nan * np.ones(nDSframes)
    recNegErr = np.nan * np.ones(nDSframes)

    print('nDSframes-------->', nDSframes)

    # Create view matrices for interpolation
    viewR, viewC = np.meshgrid(
        np.arange(0, sz[0] + 2 * maxshift) - maxshift, #sz in matlab is 121, 45, 1, 10000
        np.arange(0, sz[1] + 2 * maxshift) - maxshift,
        indexing='ij'  # 'ij' for matrix indexing to match MATLAB's ndgrid
    )

    print('maxshift------->', maxshift)
    print('viewR-------->', viewR.shape)

    print('Strip Registeration...')
    aData = {} # Alignment data dictionary

    for DSframe in range(0, nDSframes):
        start_total = time.time()  # Start timing for the entire loop

        # Timing for reading frames
        start_read_frames = time.time()
        readFrames = list(range((DSframe) * dsFac, (DSframe) * dsFac + dsFac))
        end_read_frames = time.time()
        print(f"Time for reading frames: {end_read_frames - start_read_frames:.4f} seconds")

        # Timing for downsampling and reshaping
        start_downsample = time.time()
        M = downsampleTime(Ad[:, :, :, readFrames], ds_time)
        M = np.sum(M[:,:,selCh,:].reshape(M.shape[0], M.shape[1], -1, M.shape[3]), axis=2).squeeze()
        end_downsample = time.time()
        print(f"Time for downsampling and reshaping: {end_downsample - start_downsample:.4f} seconds")

        if DSframe % 1000 == 0:
            print(f'{DSframe} of {nDSframes}')

        # Timing for template averaging
        start_template_avg = time.time()
        Ttmp = np.nanmean(np.dstack((T0, T00, template)), axis=2)
        T = Ttmp[maxshift-initR : maxshift-initR+sz[0], maxshift-initC : maxshift-initC+sz[1]]
        end_template_avg = time.time()
        print(f"Time for template averaging: {end_template_avg - start_template_avg:.4f} seconds")

        # Timing for registration
        start_registration = time.time()
        output, _ = dftregistration_clipped(fft2(M), fft2(T.astype(np.float32)), 4, clipShift)
        end_registration = time.time()
        print(f"Time for registration: {end_registration - start_registration:.4f} seconds")

        motionDSr[DSframe] = initR + output[2]
        motionDSc[DSframe] = initC + output[3]
        aErrorDS[DSframe] = output[0]

        # Timing for condition check and correction
        if np.sqrt((motionDSr[DSframe] / sz[0])**2 + (motionDSc[DSframe] / sz[1])**2) > 0.75**2:
            start_correction = time.time()
            x = np.arange(sz[1])
            y = np.arange(sz[0])
            mesh_x, mesh_y = np.meshgrid(x, y)
            
            start_cv2_remap = time.time()
            map_x = np.float32(viewC)
            map_y = np.float32(viewR)
            Mfull = cv2.remap(M.astype(np.float32), map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
            end_cv2_remap = time.time()
            print(f"Time for cv2_remap: {end_cv2_remap - start_cv2_remap:.4f} seconds")

            start_xcorr2_nans = time.time()
            motion, R = xcorr2_nans(Mfull, Ttmp, np.array([initR, initC]), 50)
            end_xcorr2_nans = time.time()
            print(f"Time for xcorr2_nans: {end_xcorr2_nans - start_xcorr2_nans:.4f} seconds")

            motionDSr[DSframe] = motion[0]
            motionDSc[DSframe] = motion[1]
            aErrorDS[DSframe] = R
            end_correction = time.time()
            print(f"Time for correction: {end_correction - start_correction:.4f} seconds")

        # Timing for interpolation and smoothing
        if abs(motionDSr[DSframe]) < maxshift and abs(motionDSc[DSframe]) < maxshift:
            start_interpolation_smoothing = time.time()
            X, Y = np.meshgrid(np.arange(0, sz[1]), np.arange(0, sz[0]))
            
            Xq = viewC + motionDSc[DSframe]
            Yq = viewR + motionDSr[DSframe]
            
            A = cv2.remap(M, Xq.astype(np.float32), Yq.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
            
            Asmooth = cv2.GaussianBlur(A, (0, 0), sigmaX=1)
            
            selCorr = ~(np.isnan(Asmooth) | np.isnan(Ttmp))
            aRankCorr[DSframe] = np.corrcoef(Asmooth[selCorr], Ttmp[selCorr])[0, 1]
            recNegErr[DSframe] = np.mean(np.power(np.minimum(0, Asmooth[selCorr] * np.mean(Ttmp[selCorr]) / np.mean(Asmooth[selCorr]) - Ttmp[selCorr]), 2))
            
            templateCt = np.isnan(template).astype(int)
            template = np.where(np.isnan(template), 0, template)
            template = np.where(np.isnan(A), template, template * templateCt + A)
            templateCt += (~np.isnan(A)).astype(int)
            template /= templateCt
            template[templateCt < 100] = np.nan
            
            initR = round(motionDSr[DSframe])
            initC = round(motionDSc[DSframe])
            
            end_interpolation_smoothing = time.time()
            print(f"Time for interpolation and smoothing: {end_interpolation_smoothing - start_interpolation_smoothing:.4f} seconds")
        
        else:
            motionDSr[DSframe] = initR
            motionDSc[DSframe] = initC

        end_total = time.time()  # End timing for the entire loop
        print(f"Total time for DSframe {DSframe}: {end_total - start_total:.4f} seconds\n")

    print('motionDSr---------->', motionDSr.shape)
    print('motionDSc---------->', motionDSc.shape)

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
    print('maxshiftC---------->', maxshiftC)
    print('maxshiftR---------->', maxshiftR)
    
    print('sz---------->', sz)

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

    print('viewR---------->', viewR.shape)

    # Initialize tiffSave array
    tiffSave = np.zeros((viewR.shape[1], viewR.shape[0], nDSframes * numChannels), dtype=np.float32)

    print('tiffSave---------->', tiffSave.shape)

    # Initialize Bcount and Bsum arrays
    Bcount = np.zeros((viewR.shape[0], viewR.shape[1], numChannels), dtype=np.float32)
    Bsum = np.zeros((viewR.shape[0], viewR.shape[1], numChannels), dtype=np.float64)

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

    # Calculate nanRows and nanCols correctly
    nanRows = np.mean(np.mean(np.isnan(tiffSave), axis=2), axis=1) == 1
    nanCols = np.mean(np.mean(np.isnan(tiffSave), axis=2), axis=0) == 1

    print('nanRows----->', nanRows.shape)
    print('nanCols----->', nanCols.shape)

    # Apply boolean indexing to tiffSave
    tiffSave = tiffSave[~nanRows, :, :]
    tiffSave = tiffSave[:, ~nanCols, :]

    # Transpose to match ImageJ convention (time, Y, X)
    tiffSave = tiffSave.transpose((2, 1, 0))

    # Save tiff
    tifffile.imwrite(downsampled_tif_path, tiffSave, imagej=True, metadata={'axes': 'ZYX'})

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

    # Calculate the size of the new array
    new_size = np.array(viewR.shape)[[1, 0]] - np.array([np.sum(nanRows), np.sum(nanCols)])

    # Append the third dimension size
    new_size = np.append(new_size, len(motionC) * numChannels)

    # Create the new array filled with zeros
    tiffSave_raw = np.zeros(new_size, dtype=np.float32)

    # Save registered raw Tiffs
    for frame in range(len(motionC)):
        for ch in range(numChannels):
            # Create grid points
            x, y = np.meshgrid(np.arange(0, sz[1]), np.arange(0, sz[0]))
            
            # Calculate new grid points
            xi = viewC + motionC[frame]
            yi = viewR + motionR[frame]
            
            # Perform interpolation using cv2.rempa over scipy griddata
            B = cv2.remap(Ad[:, :, ch, frame], xi.astype(np.float32), yi.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)

            B = B[~nanCols, :]
            B = B[:, ~nanRows]

            tiffSave_raw[:, :, (frame-1)*numChannels+ch] = B.T.astype(np.float32)
            
            # Append the interpolated image to the list
            # interpolated_images.append(B)
    
    del Ad
    
    # Transpose to match ImageJ convention (time, Y, X)
    tiffSave_raw = tiffSave_raw.transpose((2, 1, 0))

    # Save with correct metadata
    tifffile.imwrite(raw_tif_path, tiffSave_raw, bigtiff=True, imagej=True, metadata={'axes': 'ZYX'})

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
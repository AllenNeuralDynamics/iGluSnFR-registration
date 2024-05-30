import os
import numpy as np
import numpy.ma as ma
from jnormcorre.motion_correction import MotionCorrect
from ScanImageTiffReader import ScanImageTiffReader
from scipy.fft import fft2
import cv2
from scipy.interpolate import interp1d, PchipInterpolator
from tifffile import tifffile

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
            m - (m // 2) : m + (m // 2) + 1,
            n - (n // 2) : n + (n // 2) + 1,
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


def stripRegistrationBergamo_init(ds_time, initFrames, Ad, maxshift, clipShift, alpha,numChannels, path_template_list, output_path_):
    if ds_time is None:
        ds_time = 3  # the movie is downsampled using averaging in time by a factor of 2^ds_time
    
    dsFac = 2 ** ds_time
    framesToRead = initFrames * dsFac
    # Downsample the data
    Y = downsampleTime(Ad[:, :, :, :framesToRead], ds_time)

    #Get size of the original data
    sz = Ad.shape

    # Sum along the third dimension and squeeze the array
    Yhp = np.squeeze(np.sum(Y,2))

    corrector = MotionCorrect(lazy_dataset=Yhp.transpose(2, 1, 0),
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
        template=None, save_movie=True
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

    template = ScanImageTiffReader(path_template) 

    F = template.data()
    F = np.transpose(F, (2, 1, 0))
    F = np.mean(F, axis=2)

    # # Create a template with NaNs
    template = np.full((2*maxshift + sz[0], 2*maxshift + sz[1]), np.nan)

    # # Insert the matrix F into the template
    template[maxshift:maxshift+sz[0], maxshift:maxshift+sz[1]] = F

    # template = np.transpose(template)

    # # Copy the template to T0
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

    # Create view matrices for interpolation
    viewR, viewC = np.meshgrid(
        np.arange(0, sz[0] + 2 * maxshift) - maxshift, #sz in matlab is 121, 45, 1, 10000
        np.arange(0, sz[1] + 2 * maxshift) - maxshift,
        indexing='ij'  # 'ij' for matrix indexing to match MATLAB's ndgrid
    )

    print('Strip Registeration...')

    for DSframe in range(0, nDSframes):
        # readFrames = slice((DSframe) * dsFac, DSframe * dsFac)
        readFrames = list(range((DSframe) * dsFac, (DSframe) * dsFac + dsFac))
        
        M = downsampleTime(Ad[:, :, :, readFrames], ds_time)
        M = np.squeeze(np.sum(M, axis=2))
        # M = np.sum(M, axis=2)  # Merge frames
        # M = np.transpose(M)
        # M = M - convolve2d(M, np.ones((4, 4))/16, mode='same')  # Highpass filter using Gaussian approximation

        # if DSframe % 1000 == 0:
        #     print(f'{DSframe} of {nDSframes}')

        Ttmp = np.nanmean(np.dstack((T0, T00, template)), axis=2)
        
        T = Ttmp[maxshift - initR:maxshift - initR + sz[0], maxshift - initC:maxshift - initC + sz[1]]
        
        output,_ = dftregistration_clipped(fft2(M), fft2(T.astype(np.float32)), 4, clipShift)
        # output = phase_cross_correlation(fft2(T.astype(np.float32)), fft2(M), normalization = 'phase', overlap_ratio = 1-maxshift/min(sz[0:2]), upsample_factor = 400) # Check normization = None vs phase and check output of shifts. 

        motionDSr[DSframe] = initR + output[2]
        motionDSc[DSframe] = initC + output[3]
        aErrorDS[DSframe] = output[0]

        if abs(motionDSr[DSframe]) < maxshift and abs(motionDSc[DSframe]) < maxshift:
            # interpolator = interp2d(np.arange(sz[1]), np.arange(sz[0]), M, kind='linear', bounds_error=False, fill_value=np.nan)
            # A = interpolator(viewC + motionDSc[DSframe - 1], viewR + motionDSr[DSframe - 1])

            # Create grid points
            X, Y = np.meshgrid(np.arange(0, sz[1]), np.arange(0, sz[0]))

            # Calculate new grid points
            Xq = viewC + motionDSc[DSframe]  # Adjust index for Python's 0-based indexing
            Yq = viewR + motionDSr[DSframe]  # Adjust index for Python's 0-based indexing

            # Perform interpolation using cv2.remap over scipy.interpolate.griddata            
            A = cv2.remap(M, Xq.astype(np.float32), Yq.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
        
            sel = ~np.isnan(A)
            selCorr = ~np.isnan(A) & ~np.isnan(template)
            A_ma = ma.array(A, mask=~selCorr)
            template_ma = ma.array(template, mask=~selCorr)

            aRankCorr[DSframe] = ma.corrcoef(ma.array([A_ma.compressed(), template_ma.compressed()]))[0, 1]
            recNegErr[DSframe] = ma.mean(np.power(ma.minimum(0, A_ma - template_ma), 2))

            nantmp = sel & np.isnan(template)
            template[nantmp] = A[nantmp]
            template[sel] = (1 - alpha) * template[sel] + alpha * A[sel]

            initR = round(motionDSr[DSframe])
            initC = round(motionDSc[DSframe])

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

    with tifffile.TiffWriter(tif_path, bigtiff=True) as tif:
        for frame in range(len(motionC)):
            for ch in range(numChannels):
                # Create grid points
                x, y = np.meshgrid(np.arange(0, sz[1]), np.arange(0, sz[0]))
                
                # Calculate new grid points
                xi = viewC + motionC[frame]
                yi = viewR + motionR[frame]
                
                # Perform interpolation using cv2.rempa over scipy griddata
                B = cv2.remap(Ad[:, :, ch, frame], xi.astype(np.float32), yi.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
                
                # Append the interpolated image to the list
                interpolated_images.append(B)

    # Stack the interpolated images into a 3D array
    interpolated_stack = np.stack(interpolated_images)

    # Save the interpolated stack as a multi-page TIFF file
    tifffile.imwrite(tif_path, interpolated_stack.astype(np.float32))

    motionR_mean = np.mean(motionR)
    motionC_mean = np.mean(motionC)

    # Center the shifts to zero 
    return motionR - motionR_mean, motionC - motionC_mean
# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
# Above line disables bounds checking, negative indexing, and None checks for performance

import numpy as np
cimport numpy as np
from libc.math cimport sqrt

# Define the function and specify types for inputs and outputs
def xcorr2_nans(np.ndarray[np.float64_t, ndim=2] C,
                   np.ndarray[np.float32_t, ndim=2] F,
                   np.ndarray[np.float32_t, ndim=2] template,
                   np.ndarray[np.uint8_t, ndim=2] tValid,  # Cast boolean array to uint8
                   np.ndarray[np.int_t, ndim=1] shifts):
    
    cdef int drix, dcix, shift_x, shift_y
    cdef np.ndarray[np.float32_t] T
    cdef float ssT
    cdef int i, j

    # Loop over shifts in both dimensions
    for drix in range(shifts.shape[0]):
        shift_x = shifts[drix]
        for dcix in range(shifts.shape[0]):
            shift_y = shifts[dcix]

            # Roll tValid array according to the current shifts using memory view
            shifted_tValid = np.roll(tValid, (-shift_x, -shift_y), axis=(0, 1))

            # Extract the valid elements from template and F based on shifted tValid
            T = template[shifted_tValid.astype(bool)]  # Convert back to bool when indexing
            ssT = np.sum(T ** 2)

            if ssT != 0:
                C[drix, dcix] = np.sum(F[shifted_tValid.astype(bool)] * T) / sqrt(ssT)
            else:
                C[drix, dcix] = 0.0
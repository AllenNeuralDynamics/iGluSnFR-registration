# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
# Above line disables bounds checking, negative indexing, and None checks for performance

cimport cython
from cython.parallel cimport prange
from libc.math cimport sqrt
                

# You can specify C and Fortran contiguous layouts for the memoryview by using the ::1 step syntax at definition
def xcorr2_nans(float[::1] F, float[::1] templ, int[::1] tValidInd, int[::1] shifts, int cols, float[:,::1] C):
    """
    Cross-correlate two arrays (with NaNs) over multiple shifts and store the results.
    
    Parameters
    ----------
    F: 1D array of flattened input data
    templ: 1D array of flattened template to be cross-correlated
    tValidInd: 1D array of valid indices in the template
    shifts: 1D array of shifts to apply during cross-correlation
    cols: int, number of columns (used for indexing)
    C: 2D array where results are stored
    
    Returns
    -------
    None
    """
    cdef:
        Py_ssize_t n, ni, drix, dcix, i, ind # using Py_ssize_t for loop indices is generally recommended
        float ssT, FT
    n = shifts.shape[0]
    ni = tValidInd.shape[0]
    for drix in prange(n, nogil=True):
        for dcix in range(n):
            ssT = 0
            FT = 0
            for i in range(ni):
                ind = tValidInd[i] - shifts[drix] * cols - shifts[dcix]
                ssT = ssT + templ[ind] * templ[ind]
                FT = FT + F[i] * templ[ind]
            if ssT > 0:
                C[drix, dcix] = FT / sqrt(ssT)

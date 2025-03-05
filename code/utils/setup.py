# setup.py
# run with:         python setup.py build_ext -i
# clean up with:    python setup.py clean --all

from Cython.Build import cythonize
import numpy as np
from setuptools import setup, Extension #, find_packages

ext_modules = [Extension("xcorr2_nans",
                         sources=["xcorr2_nans.pyx"],
                         extra_compile_args=['-fopenmp'],  # Enable OpenMP
                         extra_link_args=['-fopenmp'],     # Link with OpenMP
                         include_dirs=[np.get_include()],
                         define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")], 
                         language="c++"), 
                         ]

setup(ext_modules=cythonize(ext_modules))
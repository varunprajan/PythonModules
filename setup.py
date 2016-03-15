from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules = cythonize('first_high_low.pyx'),
    include_dirs=[np.get_include()]
)
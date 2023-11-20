from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name='dcrowdcache',
    ext_modules=cythonize("dcrowdcache.pyx", annotate = True),
    include_dirs=[numpy.get_include()],
)
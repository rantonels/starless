from distutils.core import setup
from Cython.Build import cythonize

setup(
    name = 'Starless raytracer',
    ext_modules = cythonize("tracer.pyx")
)

from setuptools import setup, Extension, find_packages
from distutils.ccompiler import new_compiler
from distutils.errors import DistutilsPlatformError
import sys
import os

try:
    from Cython.Build import cythonize
except ImportError:
    sys.exit("Error: Cython is required. Install it via 'pip install cython'.")

def check_compiler_type():
    try:
        compiler = new_compiler()
        return compiler.compiler_type
    except DistutilsPlatformError:
        sys.exit("Error: No valid C++ compiler found.")

compiler_type = check_compiler_type()
compile_args = ["/std:c++20"] if compiler_type == "msvc" else ["-std=c++20"]

extensions = [
    Extension(
        name="simulation.SpatialBirthDeath",
        sources=[
            "simulation/SpatialBirthDeathWrapper.pyx",
            "src/SpatialBirthDeath.cpp"],
        language="c++",
        include_dirs=[os.path.abspath("include")],
        extra_compile_args=compile_args
    )
]

setup(
    name="spatial_sim",
    version="1.0",
    description="Spatial birth-death point process simulator",
    ext_modules=cythonize(extensions, annotate=True, compiler_directives={"language_level": "3"}),
    packages=find_packages(),
)

import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

sources = [
            os.path.join('backend', 'pybind.cpp'), 
            os.path.join('backend', 'fastgemv.cpp'),
        ]

os.environ["CC"] = "hipcc"
os.environ["CXX"] = "hipcc"

setup(
    name='eed',
    version='0.0.1',
    ext_modules=[
        CppExtension('eed.backend',
            sources=sources,
            extra_compile_args = ['-g', '-O3', '-fopenmp', '-lgomp'],
            include_dirs=[],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

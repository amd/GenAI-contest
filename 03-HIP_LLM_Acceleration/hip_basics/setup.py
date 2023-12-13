# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

sources = [
            os.path.join('backend', 'pybind.cpp'),
            os.path.join('backend', 'demo_gemv_v2.cpp'),
        ]

os.environ["CC"] = "hipcc"
os.environ["CXX"] = "hipcc"

setup(
    name='demo',
    version='0.0.1',
    ext_modules=[
        CppExtension('demo.backend',
            sources=sources,
            extra_compile_args = ['-g', '-O3', '-fopenmp', '-lgomp'],
            include_dirs=[],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })


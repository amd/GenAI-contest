// Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <chrono>

#include "fastgemv.h"

int M = 1;
int K = 4096;
int N = 4096;

#include <ATen/ATen.h>
#include <ATen/hip/HIPContext.h>
#include <torch/extension.h>

void fastgemv(at::Tensor A, at::Tensor B, at::Tensor C){

    int mat_height_ = A.size(0);
    int vec_height_ = B.size(0);

    int warp_size = 32;
    int warp_count = 8;

    dim3 grid_dim(mat_height_/16, 16);
    dim3 block_dim(warp_size, warp_count);

    gemv_fp16_v1_2F<<<grid_dim, block_dim>>>(
        reinterpret_cast<half *>(A.data_ptr<at::Half>()),
        reinterpret_cast<half *>(B.data_ptr<at::Half>()),
        reinterpret_cast<half *>(C.data_ptr<at::Half>()),
        warp_size,
        warp_count,
        vec_height_ >> 3,
        (vec_height_ / (warp_size * warp_count)) >> 3);
}

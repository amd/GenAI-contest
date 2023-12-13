// Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include <stdio.h>
#include "demo_gemv_kernel.h"

void demo_gemv_v0(float *mat, float *vec, float *res) {
  
  dim3 grid_dim(1, 1);
  dim3 block_dim(128, 1);

  kernel_gemv_v0<<<grid_dim, block_dim>>>(mat, vec, res);

  hipDeviceSynchronize();
}

int main() {

  int mat_rows = 128;
  int vec_cols = 4;

  // Allocate memory on CPU
  float* mat = (float*)malloc(sizeof(float) * mat_rows * vec_cols);
  float* vec = (float*)malloc(sizeof(float) * vec_cols);
  float* res = (float*)malloc(sizeof(float) * mat_rows);

  // Fill in some data into mat and vec
  for (int i = 0; i < mat_rows * vec_cols; ++i)
    mat[i] = (float)1.f;
  for (int i = 0; i < vec_cols; ++i)
    vec[i] = (float)2.f;
    
  // Allocate memory on GPU
  float *d_mat, *d_vec, *d_res;
  hipMalloc((void **)&d_mat, mat_rows * vec_cols * sizeof(float));
  hipMalloc((void **)&d_vec, vec_cols * sizeof(float));
  hipMalloc((void **)&d_res, mat_rows * sizeof(float));

  // Host to Device
  hipMemcpy(d_mat, mat, (mat_rows * vec_cols) * sizeof(float),hipMemcpyHostToDevice);
  hipMemcpy(d_vec, vec, (vec_cols) * sizeof(float), hipMemcpyHostToDevice);

  // Launch kernel
  demo_gemv_v0(d_mat, d_vec, d_res);

  // Device to Host
  hipMemcpy(res, d_res, (mat_rows) * sizeof(float), hipMemcpyDeviceToHost);

  // Print result
  for(int i = 0; i < mat_rows; ++i)
    printf("%f ", res[i]);
}



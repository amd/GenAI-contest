#include <stdio.h>
#include "demo_gemv_kernel.h"

void demo_gemv_v1(half *mat, half *vec, half *res) {

  dim3 grid_dim(1, 1);
  dim3 block_dim(128, 1);

  kernel_gemv_v1<<<grid_dim, block_dim>>>(mat, vec, res);

  hipDeviceSynchronize();
}

int main() {

  int mat_rows = 128;
  int vec_cols = 4;

  // Allocate memory on CPU
  half* mat = (half*)malloc(sizeof(half) * mat_rows * vec_cols);
  half* vec = (half*)malloc(sizeof(half) * vec_cols);
  half* res = (half*)malloc(sizeof(half) * mat_rows);

  // fill in some data into mat and vec
  for (int i = 0; i < mat_rows * vec_cols; ++i)
    mat[i] = (half)1.f;
  for (int i = 0; i < vec_cols; ++i)
    vec[i] = (half)2.f;
    
  // Allocate memory on GPU
  half *d_mat, *d_vec, *d_res;
  hipMalloc((void **)&d_mat, mat_rows * vec_cols * sizeof(half));
  hipMalloc((void **)&d_vec, vec_cols * sizeof(half));
  hipMalloc((void **)&d_res, mat_rows * sizeof(half));

  // Host to Device
  hipMemcpy(d_mat, mat, (mat_rows * vec_cols) * sizeof(half),hipMemcpyHostToDevice);
  hipMemcpy(d_vec, vec, (vec_cols) * sizeof(half), hipMemcpyHostToDevice);

  // Launch kernel
  demo_gemv_v1(d_mat, d_vec, d_res);

  // Device to Host
  hipMemcpy(res, d_res, (mat_rows) * sizeof(half), hipMemcpyDeviceToHost);

  // Print result
  for(int i=0; i<mat_rows; i++)
    printf("%f ", __half2float(res[i]));
}

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

__global__ void kernel_gemv_v0(float *mat, float *vec, float* res) {
  unsigned int tid = threadIdx.x;
  unsigned int row = tid;
  unsigned int start_idx = 4 * row;
  
  float mat_h0 = mat[start_idx];
  float mat_h1 = mat[start_idx + 1];
  float mat_h2 = mat[start_idx + 2];
  float mat_h3 = mat[start_idx + 3];
  
  float vec_h0 = vec[0];
  float vec_h1 = vec[1];
  float vec_h2 = vec[2];
  float vec_h3 = vec[3];
  
  float sum = 0.0;
  sum += (mat_h0) * (vec_h0);
  sum += (mat_h1) * (vec_h1);
  sum += (mat_h2) * (vec_h2);
  sum += (mat_h3) * (vec_h3);

  res[row] = sum;
}

__global__ void kernel_gemv_v1(half *mat, half *vec, half *res) {
  unsigned int tid = threadIdx.x;
  unsigned int row = tid;
  unsigned int start_idx = 4 * row;
  
  half mat_h0 = mat[start_idx];
  half mat_h1 = mat[start_idx + 1];
  half mat_h2 = mat[start_idx + 2];
  half mat_h3 = mat[start_idx + 3];
  
  half vec_h0 = vec[0];
  half vec_h1 = vec[1];
  half vec_h2 = vec[2];
  half vec_h3 = vec[3];
  
  float sum = 0.0;
  sum += __half2float(mat_h0) * __half2float(vec_h0);
  sum += __half2float(mat_h1) * __half2float(vec_h1);
  sum += __half2float(mat_h2) * __half2float(vec_h2);
  sum += __half2float(mat_h3) * __half2float(vec_h3);
  
  res[row] = __float2half(sum);
}


    

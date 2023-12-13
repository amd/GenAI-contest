// Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#ifndef FAST_GEMV_CUH_
#define FAST_GEMV_CUH_

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hiprand/hiprand_kernel.h>
#include <stdio.h>

#define WARP_SIZE 32

struct myhalf2
{
  half x;
  half y;
};

__device__ __forceinline__ float warpReduceSum(float sum,
                                               unsigned int threadNum) {
  if (threadNum >= 32)
    sum += __shfl_down(sum, 16, WARP_SIZE);  // 0-16, 1-17, 2-18, etc.
  if (threadNum >= 16)
    sum += __shfl_down(sum, 8, WARP_SIZE);  // 0-8, 1-9, 2-10, etc.
  if (threadNum >= 8)
    sum += __shfl_down(sum, 4, WARP_SIZE);  // 0-4, 1-5, 2-6, etc.
  if (threadNum >= 4)
    sum += __shfl_down(sum, 2, WARP_SIZE);  // 0-2, 1-3, 4-6, 5-7, etc.
  if (threadNum >= 2)
    sum += __shfl_down(sum, 1, WARP_SIZE);  // 0-1, 2-3, 4-5, etc.
  return sum;
}

__global__ void gemv_fp16_v1_2F(half* mat, half* vec, half* res, int warp_size, int warp_count, int column_height, int warp_iterations)
{
  int tid = threadIdx.x;
  int bid = gridDim.x * blockIdx.y + blockIdx.x;
  int warp_id = threadIdx.y;
  int vec_idx = warp_size * warp_id + tid;
  int warp_offset = warp_size * warp_count;

  float4* mat4 = reinterpret_cast<float4*>(mat);
  float4* vec4 = reinterpret_cast<float4*>(vec);
  float4* mat4_warp_row = mat4 + bid * column_height;

  float sum = 0.0;

#pragma unroll
  for (int iter = 0; iter < warp_iterations; iter++)
    {
      float4 vec_val = vec4[vec_idx];
      myhalf2* vec_h1 = (myhalf2*)&vec_val.x;
      myhalf2* vec_h2 = (myhalf2*)&vec_val.y;
      myhalf2* vec_h3 = (myhalf2*)&vec_val.z;
      myhalf2* vec_h4 = (myhalf2*)&vec_val.w;

      float4 mat_val = mat4_warp_row[vec_idx];
      myhalf2* mat_h1 = (myhalf2*)&mat_val.x;
      myhalf2* mat_h2 = (myhalf2*)&mat_val.y;
      myhalf2* mat_h3 = (myhalf2*)&mat_val.z;
      myhalf2* mat_h4 = (myhalf2*)&mat_val.w;
      sum += __half2float(vec_h1->x) * __half2float(mat_h1->x);
      sum += __half2float(vec_h1->y) * __half2float(mat_h1->y);
      sum += __half2float(vec_h2->x) * __half2float(mat_h2->x);
      sum += __half2float(vec_h2->y) * __half2float(mat_h2->y);
      sum += __half2float(vec_h3->x) * __half2float(mat_h3->x);
      sum += __half2float(vec_h3->y) * __half2float(mat_h3->y);
      sum += __half2float(vec_h4->x) * __half2float(mat_h4->x);
      sum += __half2float(vec_h4->y) * __half2float(mat_h4->y);

      vec_idx += warp_offset;
    }

  sum = warpReduceSum(sum, blockDim.x);

  static __shared__ float buf[8];

  if(tid==0) buf[warp_id] = sum;

  __syncthreads();

  if (warp_id == 0)
    {
      sum = (tid < warp_count) ? buf[tid] : 0;
      sum = warpReduceSum(sum, warp_count);
      if (tid == 0)
        res[bid] = __float2half(sum);
    }
}

#endif  // FAST_GEMV_CUH_

    

#include <stdio.h>
#include "demo_gemv_kernel.h"
#include <ATen/ATen.h>
#include <torch/extension.h>

void demo_gemv_v2(at::Tensor mat, at::Tensor vec, at::Tensor res)
{

  dim3 grid_dim(1, 1);
  dim3 block_dim(128, 1);

  kernel_gemv_v1<<<grid_dim, block_dim>>>
    (
     reinterpret_cast<half *>(mat.data_ptr<at::Half>()),  
     reinterpret_cast<half *>(vec.data_ptr<at::Half>()), 
     reinterpret_cast<half *>(res.data_ptr<at::Half>())
     );
  
  hipDeviceSynchronize();
}



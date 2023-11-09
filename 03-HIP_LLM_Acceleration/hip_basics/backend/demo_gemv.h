#include <ATen/ATen.h>
#include <ATen/hip/HIPContext.h>

void demo_gemv_v2(at::Tensor A, at::Tensor B, at::Tensor C);

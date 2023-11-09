#include <ATen/ATen.h>
#include <ATen/hip/HIPContext.h>

void fastgemv(at::Tensor A, at::Tensor B, at::Tensor C);

#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include "demo_gemv.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  m.def("gemv", &demo_gemv_v2);
}


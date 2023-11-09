#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include "gemv.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  m.def("fastgemv", &fastgemv);
}

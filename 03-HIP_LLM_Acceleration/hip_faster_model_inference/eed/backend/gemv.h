// Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include <ATen/ATen.h>
#include <ATen/hip/HIPContext.h>

void fastgemv(at::Tensor A, at::Tensor B, at::Tensor C);

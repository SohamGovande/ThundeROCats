#pragma once
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>

namespace kittens {
using bf16 = hip_bfloat16;
using fp16 = half;
using fp32 = float;
} // namespace kittens

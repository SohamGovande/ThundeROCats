#pragma once

#include <hip/hip_runtime.h>
#include <iostream>

#define hipCheck(err)                                           \
    {                                                           \
        hipError_t _err = (err);                                \
        if (_err != hipSuccess)                                 \
            printf("HIP error: %s\n", hipGetErrorString(_err)); \
    }

#define HIP_CHECK(err) hipCheck(err)

#include "kernel_timer.hpp"
#include "data.hpp"
#include "types.hpp"
#include "algorithms.hpp"
#pragma once

#include <stdio.h>
// Relative path needed because .cu compiler ignores Visual Studio additional
// include directories.
#include "../../lib/int_types.h"

#define CUDA_CHECK_CALL(call)                                                  \
  {                                                                            \
    cudaError err = call;                                                      \
    if (cudaSuccess != err) {                                                  \
      FILE* f = fopen("test.log", "w");                                        \
      fprintf(                                                                 \
          f, "Cuda error in file '%s' in line %i : %s.\n", __FILE__, __LINE__, \
          cudaGetErrorString(err));                                            \
    }                                                                          \
  }

bool GPUInit(s32 cuda_device);
bool GPUDeinit();
void MandelbrotLineCUDA(
    u32* fractal_buf, u32 width, u32 height, u32 y_base, double cr1, double cr2,
    double ci1, double ci2, u32 bailout, bool precision, u32 lines);

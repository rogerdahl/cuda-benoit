#include "cuda_util.h"
#include "pch.h"

void GLDeviceInit(s32 device)
{
  int deviceCount;
  cutilSafeCallNoSync(cudaGetDeviceCount(&deviceCount));
  if (deviceCount == 0) {
    fprintf(stderr, "CUTIL CUDA error: no devices supporting CUDA.\n");
    exit(-1);
  }
  if (device < 0)
    device = 0;
  if (device > deviceCount - 1) {
    fprintf(
        stderr,
        "cutilDeviceInit (Device=%d) invalid GPU device.  %d GPU device(s) "
        "detected.\n\n",
        device, deviceCount);
    exit(-1);
  }
  cudaDeviceProp deviceProp;
  cutilSafeCallNoSync(cudaGetDeviceProperties(&deviceProp, device));
  if (deviceProp.major < 1) {
    fprintf(stderr, "cutil error: device does not support CUDA.\n");
    exit(-1);
  }
  cutilSafeCall(cudaGLSetGLDevice(device));
}

#include <cstdlib>
#include <stdio.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "cuda_util.h"

void __checkCudaErrors(cudaError err, const char* file, const int line)
{
  if (cudaSuccess != err) {
    fprintf(
        stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n", file, line,
        (int)err, cudaGetErrorString(err));
    exit(-1);
  }
}

void __getLastCudaError(
    const char* errorMessage, const char* file, const int line)
{
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(
        stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
        file, line, errorMessage, (int)err, cudaGetErrorString(err));
    exit(-1);
  }
}

void GLDeviceInit(s32 device)
{
  int deviceCount;
  checkCudaErrorsNoSync(cudaGetDeviceCount(&deviceCount));
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
  checkCudaErrorsNoSync(cudaGetDeviceProperties(&deviceProp, device));
  if (deviceProp.major < 1) {
    fprintf(stderr, "cutil error: device does not support CUDA.\n");
    exit(-1);
  }
  checkCudaErrors(cudaGLSetGLDevice(device));
}
//
//// General GPU Device CUDA Initialization
// int gpuDeviceInit(int devID)
//{
//  int deviceCount;
//  checkCudaErrors(cudaGetDeviceCount(&deviceCount));
//  if (deviceCount == 0) {
//    fprintf(stderr, "gpuDeviceInit() CUDA error: no devices supporting
//    CUDA.\n");
//    exit(-1);
//  }
//  if (devID < 0)
//    devID = 0;
//  if (devID > deviceCount-1) {
//    fprintf(stderr, "\n");
//    fprintf(stderr, ">> %d CUDA capable GPU device(s) detected. <<\n",
//    deviceCount);
//    fprintf(stderr, ">> gpuDeviceInit (-device=%d) is not a valid GPU device.
//    <<\n", devID);
//    fprintf(stderr, "\n");
//    return -devID;
//  }
//
//  cudaDeviceProp deviceProp;
//  checkCudaErrors( cudaGetDeviceProperties(&deviceProp, devID) );
//  if (deviceProp.major < 1) {
//    fprintf(stderr, "gpuDeviceInit(): GPU device does not support CUDA.\n");
//    exit(-1);                                                  \
//  }
//
//  checkCudaErrors( cudaSetDevice(devID) );
//  printf("> gpuDeviceInit() CUDA device [%d]: %s\n", devID, deviceProp.name);
//  return devID;
//}
//
//// This function returns the best GPU (with maximum GFLOPS)
// int gpuGetMaxGflopsDeviceId()
//{
//  int current_device   = 0, sm_per_multiproc = 0;
//  int max_compute_perf = 0, max_perf_device  = 0;
//  int device_count     = 0, best_SM_arch     = 0;
//  cudaDeviceProp deviceProp;
//
//  cudaGetDeviceCount( &device_count );
//  // Find the best major SM Architecture GPU device
//  while ( current_device < device_count ) {
//    cudaGetDeviceProperties( &deviceProp, current_device );
//    if (deviceProp.major > 0 && deviceProp.major < 9999) {
//      best_SM_arch = MAX(best_SM_arch, deviceProp.major);
//    }
//    current_device++;
//  }
//
//  // Find the best CUDA capable GPU device
//  current_device = 0;
//  while( current_device < device_count ) {
//    cudaGetDeviceProperties( &deviceProp, current_device );
//    if (deviceProp.major == 9999 && deviceProp.minor == 9999) {
//      sm_per_multiproc = 1;
//    } else {
//      sm_per_multiproc = _ConvertSMVer2Cores(deviceProp.major,
//      deviceProp.minor);
//    }
//
//    int compute_perf  = deviceProp.multiProcessorCount * sm_per_multiproc *
//    deviceProp.clockRate;
//    if( compute_perf  > max_compute_perf ) {
//      // If we find GPU with SM major > 2, search only these
//      if ( best_SM_arch > 2 ) {
//        // If our device==dest_SM_arch, choose this, or else pass
//        if (deviceProp.major == best_SM_arch) {
//          max_compute_perf  = compute_perf;
//          max_perf_device   = current_device;
//        }
//      } else {
//        max_compute_perf  = compute_perf;
//        max_perf_device   = current_device;
//      }
//    }
//    ++current_device;
//  }
//  return max_perf_device;
//}
//
//// Initialization code to find the best CUDA Device
// int findCudaDevice(int argc, const char **argv)
//{
//  cudaDeviceProp deviceProp;
//  int devID = 0;
//  // If the command-line has a device number specified, use it
//  if (checkCmdLineFlag(argc, argv, "device")) {
//    devID = getCmdLineArgumentInt(argc, argv, "device=");
//    if (devID < 0) {
//      printf("Invalid command line parameters\n");
//      exit(-1);
//    } else {
//      devID = gpuDeviceInit(devID);
//      if (devID < 0) {
//        printf("exiting...\n");
//        shrQAFinishExit(argc, (const char **)argv, QA_FAILED);
//        exit(-1);
//      }
//    }
//  } else {
//    // Otherwise pick the device with highest Gflops/s
//    devID = gpuGetMaxGflopsDeviceId();
//    checkCudaErrors( cudaSetDevice( devID ) );
//    checkCudaErrors( cudaGetDeviceProperties(&deviceProp, devID) );
//    printf("> Using CUDA device [%d]: %s\n", devID, deviceProp.name);
//  }
//  return devID;
//}

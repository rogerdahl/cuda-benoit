#include "mandelbrot_kernel.h"
// Full relative path needed because .cu compiler ignores Visual Studio
// additional include directories.
#include "../../lib/int_types.h"

#include <iostream>
using namespace std;
#include <cuda_runtime_api.h>
#include <cutil.h>

cudaStream_t stream;
unsigned int timer_gpu;

// TODO: Templatize on floating point. (Already done in the player).
__global__ void calc_line_kernel_float(u32* fractal_buf,
                                       u32 pitch, u32 width,
                                       u32 height, u32 y_base,
                                       float cr1, float cr2,
                                       float ci1, float ci2,
                                       u32 bailout) {
	u32 x(blockIdx.x * blockDim.x + threadIdx.x);
	u32 y_off(blockIdx.y * blockDim.y + threadIdx.y);
  u32 y(y_off + y_base);

  if (x >= width) {
    return;
  }

	float cr((float)x / (float)width * (cr2 - cr1) + cr1);
  float ci((float)y / (float)height * (ci2 - ci1) + ci1);
  float zi(0), zr(0), zr2(0), zi2(0), zit(0);
	u32 iter(bailout);
	while(--iter && zr2 + zi2 < 4.0f) {
		zit = zr * zi;
		zi = zit + zit + ci;
		zr = (zr2 - zi2) + cr;
		zr2 = zr * zr;
		zi2 = zi * zi;
	}

	if(iter) {
		iter = bailout - iter;
	}

  // Disabling write with "if (threadIdx.x == -1)" had NO effect on performance.
  fractal_buf[x + y_off * (pitch / sizeof(u32))] = iter;
}

__global__ void calc_line_kernel_double(u32* fractal_buf,
                                        u32 pitch, u32 width,
                                        u32 height, u32 y_base,
                                        double cr1, double cr2,
                                        double ci1, double ci2,
                                        u32 bailout) {
	u32 x(blockIdx.x * blockDim.x + threadIdx.x);
	u32 y_off(blockIdx.y * blockDim.y + threadIdx.y);
  u32 y(y_off + y_base);

  if (x >= width) {
    return;
  }

	double cr((double)x / (double)width * (cr2 - cr1) + cr1);
  double ci((double)y / (double)height * (ci2 - ci1) + ci1);
  double zi(0), zr(0), zr2(0), zi2(0), zit(0);
	u32 iter(bailout);
	while(--iter && zr2 + zi2 < 4.0) {
		zit = zr * zi;
		zi = zit + zit + ci;
		zr =(zr2 - zi2) + cr;
		zr2 = zr * zr;
		zi2 = zi * zi;
	}

	if(iter) {
		iter = bailout - iter;
	}
  
	fractal_buf[x + y_off * (pitch / sizeof(u32))] = iter;
}

// Increase the grid size by 1 if the block width or height does not divide evenly
// by the thread block dimensions.
u32 div_up(u32 a, u32 b) {
	return((a % b) != 0) ?(a / b + 1) :(a / b);
}

void c() {
  cudaThreadSynchronize();
  cudaError err = cudaGetLastError();
  if (err != cudaSuccess) {
    cudaGetErrorString(err);
  }
}

void MandelbrotLineCUDA(u32* fractal_buf, u32 width, u32 height, u32 y_base, double cr1, double cr2, double ci1, double ci2, u32 bailout, bool precision, u32 lines) {
  u32* d_buf(0);
  size_t d_buf_pitch(0);

	cudaMallocPitch(
		&d_buf,
		&d_buf_pitch,
		width * sizeof(u32),
		lines // height
	);

	// threads_per_block should be obtained from the occupancy calculator.
	const u32 x_threads_per_block(192);
	const u32 y_threads_per_block(1);

  dim3 block_dim(x_threads_per_block, y_threads_per_block);
  dim3 grid_dim(div_up(width, x_threads_per_block), lines);

  if (precision) {
    calc_line_kernel_double<<<grid_dim, block_dim>>>(d_buf, static_cast<u32>(d_buf_pitch), width, height, y_base, cr1, cr2, ci1, ci2, bailout);
  }
  else {
    calc_line_kernel_float<<<grid_dim, block_dim>>>(d_buf, static_cast<u32>(d_buf_pitch), width, height, y_base, (float)cr1, (float)cr2, (float)ci1, (float)ci2, bailout);
  }

  cudaMemcpy2D(
    fractal_buf + y_base * width, //param dst    - Destination memory address
    width * sizeof(u32),         //param dpitch - Pitch of destination memory
    d_buf,                   //param src    - Source memory address
    d_buf_pitch,             //param spitch - Pitch of source memory
    width * sizeof(u32),         //param width  - Width of matrix transfer (columns in bytes)
    lines,                   //param height - Height of matrix transfer (rows)
    cudaMemcpyDeviceToHost); //param kind   - Type of transfer

  //cudaThreadSynchronize();

  cudaFree(d_buf);
}

bool GPUInit(s32 cuda_device) {
	wcout << L"Initializing CUDA" << endl;

	// Find number of CUDA devices.
	int cuda_device_count;
	cudaGetDeviceCount(&cuda_device_count);
	if (!cuda_device_count) {
		wcout << "Error: Found no devices supporting CUDA" << endl;
		return false;
	}

	if (cuda_device > cuda_device_count - 1) {
		wcout << L"Error: No such CUDA device" << endl;
		return false;
	}

	// Select CUDA device.
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, cuda_device);
	if (prop.major < 1) {
		wcout << L"Error: Selected device does not support CUDA" << endl;
		return false;
	}

	// Set CUDA device.
  wcout << L"Using device: " << cuda_device << endl;
	cudaSetDevice (cuda_device);

	// Print some CUDA device properties of the selected device.
	wcout << L"Name: " << prop.name << endl;
	wcout << L"Compute Capability: " << prop.major << L"." << prop.minor << endl;
	wcout << L"MultiProcessor Count: " << prop.multiProcessorCount << endl;
	wcout << L"Clock Rate: " << prop.clockRate << L" Hz" << endl;
	wcout << L"Warp Size: " << prop.warpSize << endl;
	wcout << L"Total Constant Memory: " << prop.totalConstMem << L" bytes " << endl;
	wcout << L"Total Global Memory: " << prop.totalGlobalMem << L" bytes " << endl;
	wcout << L"Shared Memory Per Block: " << prop.sharedMemPerBlock << L" bytes " << endl;
	wcout << L"Max Grid Size: (" << prop.maxGridSize[0] << L", " << prop.maxGridSize[1] << L", " << prop.maxGridSize[2] << L")" << endl;
	wcout << L"Max Threads Dim: (" << prop.maxThreadsDim[0] << L", " << prop.maxThreadsDim[1] << L", " << prop.maxThreadsDim[2] << L")" << endl;
	wcout << L"Max Threads Per Block: " << prop.maxThreadsPerBlock << endl;
	wcout << L"Regs Per Block: " << prop.regsPerBlock << endl;
	wcout << L"Memory Pitch: " << prop.memPitch << endl;
	wcout << L"Texture Alignment: " << prop.textureAlignment << endl;
	wcout << L"Device Overlap: " << prop.deviceOverlap << L"\n" << endl;

	// Create stream.
	//cudaStreamCreate (&stream);

	// Create a timer.
	//cutCreateTimer (&timer_gpu);

	return true;
}

bool GPUDeinit () {
	// Free stream.
	//cudaStreamDestroy (stream);
	// Destroy timer.
	//cutDeleteTimer(timer_gpu);

  // Explicitly cleans up all runtime-related resources associated with the
	// calling host thread. Any subsequent API call reinitializes the runtime.
  // cudaThreadExit() is implicitly called on host thread exit.
  // cudaThreadExit();
	return 1;
}

//int deviceCount;
//cudaGetDeviceCount(&deviceCount);
//for (int device(0); device < deviceCount; ++device) {
//  cudaDeviceProp deviceProp;
//  cudaGetDeviceProperties(&deviceProp, device);
//  if (dev == 0) {
//    if (deviceProp.major == 9999 && deviceProp.minor == 9999) {
//      printf("There is no device supporting CUDA.\n");
//    }
//    else if (deviceCount == 1) {
//      printf("There is 1 device supporting CUDA\n");
//    }
//    else {
//      printf("There are %d devices supporting CUDA\n", deviceCount);
//    }
//  } 
//}
//

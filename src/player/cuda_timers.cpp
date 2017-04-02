#include "pch.h"

#include "cuda_timers.h"
#include "float.h"

// FLT_MAX

CUDATimer::CUDATimer()
  : current_(0.0f), total_(0.0f), cycles_(0), min_(FLT_MAX), max_(0.0f)
{
  checkCudaErrors(cudaEventCreate(&start_));
  checkCudaErrors(cudaEventCreate(&stop_));
}

CUDATimer::~CUDATimer()
{
  checkCudaErrors(cudaEventDestroy(start_));
  checkCudaErrors(cudaEventDestroy(stop_));
};

//-------------------

CUDATimers::CUDATimers(CUDATimerEnum num_timers, bool enable_timers)
  : enable_timers_(enable_timers)
{
  if (!enable_timers_) {
    return;
  }
  cuda_timers_ = new CUDATimer[num_timers];
}

CUDATimers::~CUDATimers()
{
  if (!enable_timers_) {
    return;
  }
  delete[] cuda_timers_;
}

void CUDATimers::Start(CUDATimerEnum timer_id)
{
  if (!enable_timers_) {
    return;
  }
  CUDATimer& cuda_timer(cuda_timers_[timer_id]);
  checkCudaErrors(cudaThreadSynchronize());
  checkCudaErrors(cudaEventRecord(cuda_timer.start_, 0));
  checkCudaErrors(cudaEventSynchronize(cuda_timer.start_));
}

void CUDATimers::Stop(CUDATimerEnum timer_id)
{
  if (!enable_timers_) {
    return;
  }
  CUDATimer& cuda_timer(cuda_timers_[timer_id]);
  checkCudaErrors(cudaThreadSynchronize());
  checkCudaErrors(cudaEventRecord(cuda_timer.stop_, 0));
  checkCudaErrors(cudaEventSynchronize(cuda_timer.stop_));
  float current;
  checkCudaErrors(
      cudaEventElapsedTime(&current, cuda_timer.start_, cuda_timer.stop_));
  current /= 1000.0;
  cuda_timer.current_ += current;
}

CUDATimes CUDATimers::GetTimes(CUDATimerEnum timer_id)
{
  // assert(enable_timers_);
  CUDATimes cuda_times;
  CUDATimer& cuda_timer(cuda_timers_[timer_id]);
  // Current.
  cuda_times.current_ = cuda_timer.current_;
  // Average.
  ++cuda_timer.cycles_;
  cuda_timer.total_ += cuda_times.current_;
  cuda_times.average_ = cuda_timer.total_ / cuda_timer.cycles_;
  // Min.
  if (cuda_times.current_ < cuda_timer.min_) {
    cuda_timer.min_ = cuda_times.current_;
  }
  cuda_times.min_ = cuda_timer.min_;
  // Max.
  if (cuda_times.current_ > cuda_timer.max_) {
    cuda_timer.max_ = cuda_times.current_;
  }
  cuda_times.max_ = cuda_timer.max_;
  // Reset.
  cuda_timer.current_ = 0.0f;
  return cuda_times;
}

CUDATimerRun::CUDATimerRun(CUDATimers& cuda_timers, CUDATimerEnum timer_id)
  : cuda_timers_(cuda_timers)
{
  timer_id_ = timer_id;
  cuda_timers_.Start(timer_id);
}

CUDATimerRun::~CUDATimerRun()
{
  cuda_timers_.Stop(timer_id_);
}

//  cudaEvent_t start;
//  cudaEvent_t stop;
//  float time;
//  cudaEventCreate(&start);
//  cudaEventCreate(&stop);
//  cudaEventRecord(start, 0);
//
//  kernel<<<grid,threads>>> ( d_odata, d_idata, size_x, size_y, NUM_REPS);
//
//  cudaEventRecord(stop, 0);
//  cudaEventSynchronize(stop);
//  cudaEventElapsedTime(&time, start, stop );
//  cudaEventDestroy(start);
//  cudaEventDestroy(stop);
//

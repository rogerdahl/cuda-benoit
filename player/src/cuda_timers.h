#pragma once

#include "../../lib/int_types.h"

enum CUDATimerEnum {
  kTotal,
  kMandelbrot,
  kFractalReduce,
  kPalettes, // generate, lerp palettes
  kTransform,
  kTransformReduceAndColorize,
  kRender,
  kCount_,
};

struct CUDATimes {
  float current_;
  float average_;
  float min_;
  float max_;
};

class CUDATimer {
  CUDATimer();
  ~CUDATimer();
  cudaEvent_t start_;
  cudaEvent_t stop_;
  float current_;
  float total_;
  int cycles_;
  float min_;
  float max_;
  friend class CUDATimers;
};

class CUDATimerRun;

class CUDATimers {
public:
  CUDATimers(CUDATimerEnum num_timers, bool enable_timers);
  ~CUDATimers();
  CUDATimes GetTimes(CUDATimerEnum timer_id);
private:
  bool enable_timers_;
  CUDATimer* cuda_timers_;

  void Start(CUDATimerEnum timer_id);
  void Stop(CUDATimerEnum timer_id);
  friend class CUDATimerRun;
};

class CUDATimerRun {
public:
  CUDATimerRun(CUDATimers& cuda_timers, CUDATimerEnum timer_id);
  ~CUDATimerRun();
private:
  CUDATimers& cuda_timers_;
  CUDATimerEnum timer_id_;
};

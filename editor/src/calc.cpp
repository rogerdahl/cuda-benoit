#include "pch.h"

#include "sse2_x64_mandelbrot.h"
#include "mandelbrot_kernel.h"
#include "calc.h"

using namespace std;
using namespace boost;


// Config.

// ---------------------------------------------------------------------------
// Calculation Thread.
// ---------------------------------------------------------------------------

// Packed doubles (2x 64 bit doubles), SSE2, assembly, x64 only.
void MandelbrotLineSSE2x64(u32* fractal_buf_, u32 width, u32 y, double cr1, double cr2, double ci, u32 bailout) {
  u32 line_offset(width * y);

  u32 g(0);
  for (u32 x(0); x < width; x += 2) {
    double cRd0 = (double)x / (double)width * (cr2 - cr1) + cr1;
    double cRd1 = (double)(x + 1) / (double)width * (cr2 - cr1) + cr1;
    __int64 q = MandelbrotLineSSE2x64Two(cRd0, ci, cRd1, ci, bailout);
    fractal_buf_[line_offset++] = q;
    fractal_buf_[line_offset++] = q >> 32;
  }
}

// Float and double (1x 32 bit or 80/64 bit), C++, x86 (works on x64).
template <typename FP>
void MandelbrotLineFP(u32* fractal_buf_, u32 width, u32 y, FP cr1, FP cr2, FP ci, u32 bailout) {
  u32 line_offset(width * y);

  for (u32 x(0); x < width; ++x) {
    FP cr((FP)x / (FP)width * (cr2 - cr1) + cr1);
    FP zi(0), zr(0), zr2(0), zi2(0), zit(0);
    u32 iter(bailout);
    while(--iter && zr2 + zi2 < 4.0) {
      zit = zr * zi;
      zi = zit + zit + ci;
      zr = (zr2 - zi2) + cr;
      zr2 = zr * zr;
      zi2 = zi * zi;
    }

    if (iter) {
      iter = bailout - iter;
    }

    fractal_buf_[line_offset++] = iter;
  }
}


// Packed floats (4x 32 bit floats), C++ with intrinsics, x86 (works on x64).
//
// This function uses the technique of clearing variables by xor'ing them with
// themselves. This erroneously triggers "C4700: uninitialized local variable
// used" at compile time and "Run-Time Check Failure #3 The variable is being
// used without being initialized." at runtime. So we disable those checks for
// this function.
#pragma warning(push)
// C4700: uninitialized local variable used.
#pragma warning(disable: 4700)
// u: Reports when a variable is used before it is defined.
#pragma runtime_checks("u", off )

void MandelbrotLineSSE2x86Intrinsics(u32* fractal_buf_, u32 width, u32 y,
                                   double cr1, double cr2, double ci,
                                   u32 bailout) {
  u32 line_offset(y * width);

  u32 xx;
  u32 mask, mmask, maskN;
  u32 min2go;

  __m128 xa128, xb128, xstp128;
  __m128 ci128, cr128, zi128, zr128, zr2128, zi2128, lsq128;
  __m128 tmp128;

  xa128 = _mm_set_ps1((float)cr1);
  xb128 = _mm_set_ps1((float)cr2);

  // y
  ci128 = _mm_set_ps1((float)ci);

  // x step = (xb - xa) / (width * 4)
  xstp128 = _mm_sub_ps(xb128, xa128);
  tmp128 = _mm_set_ps1(width);
  xstp128 = _mm_div_ps(xstp128, tmp128);
  tmp128 = _mm_set_ps1(4);
  xstp128 = _mm_mul_ps(xstp128, tmp128);

  // Set up 4 start location offsets.
  tmp128 = _mm_set_ps(-0.25f, -0.5f, -0.75f, -1.0f);
  cr128 = _mm_mul_ps(tmp128, xstp128);
  //	tmp128 = _mm_set_ps1(2);
  //	cr128 = _mm_sub_ps(cr128, tmp128);
  cr128 = _mm_add_ps(cr128, xa128);

  // Calc one line.
  for (xx = 0; xx < width; xx += 4) {
    cr128 = _mm_add_ps(cr128, xstp128);

    // Set the packed floating point registers to 0.0 (the binary representation
    // for 0.0 in IEEE-754 is all zeroes).
    zr128 = _mm_xor_ps(zr128, zr128);
    zi128 = _mm_xor_ps(zi128, zi128);

    zr2128 = _mm_xor_ps(zr2128, zr2128);
    zi2128 = _mm_xor_ps(zi2128, zi2128);

    // 4 pixels.
    mmask = 0;
    min2go = bailout;
    do {
      do {
        --min2go;
        zi128 = _mm_mul_ps(zi128, zr128);
        zi128 = _mm_add_ps(zi128, zi128);
        zi128 = _mm_add_ps(zi128, ci128);

        zr128 = _mm_sub_ps(zr2128, zi2128);
        zr128 = _mm_add_ps(zr128, cr128);

        zr2128 = _mm_mul_ps(zr128, zr128);

        zi2128 = _mm_mul_ps(zi128, zi128);

        lsq128 = _mm_add_ps(zr2128, zi2128);

        tmp128 = _mm_set_ps1(4.0);
        tmp128 = _mm_cmplt_ps(lsq128, tmp128);
        mask = _mm_movemask_ps(tmp128);
        if ((mask | mmask) != 15)
          break;
      } while(min2go);

      if (!min2go)
        mask = 0;

      for (maskN = 0; maskN < 4; ++maskN) {
        if (((mask | mmask) & (1 << maskN)) == 0) {
          fractal_buf_[line_offset + xx + maskN] = min2go ? bailout - min2go : 0;
          mmask = mmask | (1 << maskN);
        }
      }
    } while(mmask != 15);
  }

  return;
}

#pragma warning(pop)
#pragma runtime_checks("u", restore) 

struct ThreadShared {
  ThreadShared() : stop_(false), running_(false) {}

  // For use by master.

  void Stop() {
    stop_ = true;
  }

  void SetRunning() {
    running_ = true;
  }

  bool IsRunning() {
    return running_;
  }

  // For use by slave.

  void ClearRunning() {
    running_ = false;
  }

  volatile bool IsStopping() {
    return stop_;
  }

  void New() {
    stop_ = false;
    running_ = false;
  }

private:
  bool stop_;
  bool running_;
};

struct CalcThread {
  CalcThread(ThreadShared* ThreadShared, u32* fractal_buf, u32 width,
              u32 height, double cr1, double cr2, double ci1, double ci2,
              CalcMethods calc_method, u32 bailout)
    : thread_shared_(ThreadShared),
    fractal_buf_(fractal_buf),
    width_(width),
    height_(height),
    cr1_(cr1),
    cr2_(cr2),
    ci1_(ci1),
    ci2_(ci2),
    bailout_(bailout),
    calc_method_(calc_method) 
  { }

  void operator()() {
    // OpenMP requires y to be signed.
    s32 y;

    // Use OpenMP to parallelize the calc loop in release builds.

    if (calc_method_ == kCalcx86Float) {
#ifndef _DEBUG
#pragma omp parallel private(y)
#pragma omp for schedule(dynamic) nowait
#endif
      for (y = 0; y < static_cast<s32>(height_); ++y) {
        if (!thread_shared_->IsStopping()) {
          double ci((double)y / (double)height_ * (ci2_ - ci1_) + ci1_);
          MandelbrotLineFP<float>(fractal_buf_, width_, y, static_cast<float>(cr1_),
            static_cast<float>(cr2_), static_cast<float>(ci), bailout_);
        }
      }
    }

    if (calc_method_ == kCalcx86Double) {
#ifndef _DEBUG
#pragma omp parallel private(y)
#pragma omp for schedule(dynamic) nowait
#endif
      for (y = 0; y < static_cast<s32>(height_); ++y) {
        if (!thread_shared_->IsStopping()) {
          double ci((double)y / (double)height_ * (ci2_ - ci1_) + ci1_);
          MandelbrotLineFP<double>(fractal_buf_, width_, y, cr1_, cr2_, ci, bailout_);
        }
      }
    }

    if (calc_method_ == kCalcSSE4Float) {
#ifndef _DEBUG
#pragma omp parallel private(y)
#pragma omp for schedule(dynamic) nowait
#endif
      for (y = 0; y < static_cast<s32>(height_); ++y) {
        if (!thread_shared_->IsStopping()) {
          double ci((double)y / (double)height_ * (ci2_ - ci1_) + ci1_);
          MandelbrotLineSSE2x86Intrinsics(fractal_buf_, width_, y, cr1_, cr2_, ci, bailout_);
        }
      }
    }

    if (calc_method_ == kCalcSSE2Double) {
#ifndef _DEBUG
#pragma omp parallel private(y)
#pragma omp for schedule(dynamic) nowait
#endif
      for (y = 0; y < static_cast<s32>(height_); ++y) {
        if (!thread_shared_->IsStopping()) {
          double ci((double)y / (double)height_ * (ci2_ - ci1_) + ci1_);
          MandelbrotLineSSE2x64(fractal_buf_, width_, y, cr1_, cr2_, ci, bailout_);
        }
      }
    }

    // For CUDA, we calculate a certain number of lines at a time. If
    // supersample_ is >1, this does not correspond to the number of lines on
    // screen. When setting this, we want to balance a good GPU thread count
    // with the time it takes to run one kernel (want to stay below the watchdog
    // timeout of 5 seconds, and keep a responsive GUI).
    if (calc_method_ == kCalcCUDAFloat) {
      u32 lines(512);
      for (y = 0; y < static_cast<s32>(height_); y += lines) {
        if (!thread_shared_->IsStopping()) {
          u32 l(lines);
          if (y + lines > height_) {
            l = height_ - y;
          }
          MandelbrotLineCUDA(fractal_buf_, width_, height_, y, cr1_, cr2_, ci1_, ci2_, bailout_, false, l);
        }
      }
    }

    if (calc_method_ == kCalcCUDADouble) {
      u32 lines(512 / 8);
      for (y = 0; y < static_cast<s32>(height_); y += lines) {
        if (!thread_shared_->IsStopping()) {
          u32 l(lines);
          if (y + lines > height_) {
            l = height_ - y;
          }
          MandelbrotLineCUDA(fractal_buf_, width_, height_, y, cr1_, cr2_, ci1_, ci2_, bailout_, true, l);
        }
      }
    }

    thread_shared_->ClearRunning();
  }

  ThreadShared* thread_shared_;
  u32* fractal_buf_;
  u32 width_;
  u32 height_;
  double cr1_;
  double cr2_;
  double ci1_;
  double ci2_;
  u32 bailout_;
  CalcMethods calc_method_;
};


// ---------------------------------------------------------------------------
// Controller class.
// ---------------------------------------------------------------------------

Calc::Calc(FractalSpec& fractal_spec) :
  fractal_spec_(fractal_spec), fractal_buf_(0), cuda_available_(false)
{
  thread_shared_ = new ThreadShared;
  cuda_available_ = GPUInit(0);
  Init();
}

Calc::~Calc()
{
  StopThread();
  // Currently does nothing.
  GPUDeinit();
  delete thread_shared_;
}

void Calc::Init() {
  StopThread();
  zoom_ = fractal_spec_.zoom_end_;
  // By default, we set the calculation method to one that is supported by all
  // machines.
  calc_method_ = kCalcx86Double;
  width_ = 0;
  height_ = 0;
}

u32* Calc::GetFractalBuf() {
  return fractal_buf_;
}

void Calc::SetDim(u32 width, u32 height) {
  StopThread();
  width_ = width;
  height_ = height;
  StartThread();
}

void Calc::GetDim(u32& width, u32& height) {
  width = width_;
  height = height_;
}

void Calc::SetCenter(double center_r, double center_i) {
  StopThread();
  fractal_spec_.center_r_ = center_r;
  fractal_spec_.center_i_ = center_i;
  StartThread();
}

void Calc::SetZoom(double zoom) {
  StopThread();
  zoom_ = zoom;
  StartThread();
}

double Calc::GetZoom() {
  return zoom_;
}

void Calc::SetBailout(u32 bailout) {
  StopThread();
  fractal_spec_.bailout_ = bailout;
  StartThread();
}

u32 Calc::GetBailout() {
  return fractal_spec_.bailout_;
}

void Calc::SetCalcMethod(CalcMethods calc_method) {
  StopThread();
  // Disable selection of CUDA based calculation if CUDA initialization failed.
  if (!cuda_available_ &&
      (calc_method == kCalcCUDAFloat || calc_method == kCalcCUDADouble)) {
    calc_method = kCalcx86Double;
  }
  calc_method_ = calc_method;
  StartThread();
}

CalcMethods Calc::GetCalcMethod() {
  return calc_method_;
}

void Calc::StartThread() {
  // Ignore if we don't have all neccessary parameters.
  if (!width_ || !height_) {
    return;
  }

  assert(!thread_shared_->IsRunning());

  // Set up buffer for fractal data.
  fractal_buf_ = new u32[width_ * height_];
  memset(fractal_buf_, -1, width_ * height_ * sizeof(u32));

  thread_shared_->New();
  thread_shared_->SetRunning();

  double cr1, cr2, ci1, ci2;
  GetTranslatedCoordinates(&cr1, &cr2, &ci1, &ci2);

  // Start calculation in separate thread.
  CalcThread c(thread_shared_, fractal_buf_, width_, height_, cr1, cr2, ci1, ci2, calc_method_, fractal_spec_.bailout_);
  thread tr(c);
}

void Calc::StopThread() {
  if (thread_shared_) {
    thread_shared_->Stop();
    while (thread_shared_->IsRunning()) {
      Sleep(100);
    }
  }

  // Delete the current frame.
  if (fractal_buf_) {
    delete[] fractal_buf_;
    fractal_buf_ = 0;
  }
}

void Calc::GetTranslatedCoordinates(double* cr1, double* cr2, double* ci1, double* ci2) {
  // Translate (center + zoom) to (upper left + lower right)
  *cr1 = fractal_spec_.center_r_ - zoom_;
  *cr2 = fractal_spec_.center_r_ + zoom_;
  double aspect_ratio = (double)width_ / (double)height_;
  *ci1 = fractal_spec_.center_i_ - (zoom_ / aspect_ratio);
  *ci2 = fractal_spec_.center_i_ + (zoom_ / aspect_ratio);
}

bool Calc::IsRunning() {
  return thread_shared_->IsRunning();
}

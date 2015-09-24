#include "int_types.h"
#include "../track/track.h"

enum CalcMethods {
  kCalcx86Float,
  kCalcx86Double,
  kCalcSSE4Float,
  kCalcSSE2Double,
  kCalcCUDAFloat,
  kCalcCUDADouble
};

struct ThreadShared;
  
class Calc {
  FractalSpec& fractal_spec_;
  ThreadShared* thread_shared_;
  u32* fractal_buf_;
  CalcMethods calc_method_;
  double zoom_;

  // Resolution for fractal calculation. This is the full resolution,
  // including supersampling.
  u32 width_;
  u32 height_;
  bool cuda_available_;

  void StartThread();
  void StopThread();

public:
  Calc(FractalSpec&);
  ~Calc();
  void Init();

  void SetDim(u32 width, u32 height);
  void GetDim(u32& width, u32& height);

  bool SetPos(u32 pos);
  u32 GetPos();

  void SetCenter(double center_r, double center_i);
  void SetZoom(double zoom);
  double GetZoom();

  void SetBailout(u32 bailout);
  u32 GetBailout();

  void SetCalcMethod(CalcMethods);
  CalcMethods GetCalcMethod();

  void GetTranslatedCoordinates(double* cr1, double* cr2, double* ci1, double* ci2);
  u32* GetFractalBuf();
  bool IsRunning();
};


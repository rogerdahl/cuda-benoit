#include "cutil.h"
//nclude <cutil_math.h>
#include "cutil_inline_runtime.h"

#include "../../lib/int_types.h"

#define _USE_MATH_DEFINES
#include <math.h>

#include "kernels.h"
#include "config.h"
#include "tracks.h"

// Constants.

// The lowest zoom level that will not display artifacts caused by insufficient
// math resolution. Values found experimentally.
const double MIN_ZOOM_SINGLE_PREC(0.005);
const double MIN_ZOOM_DOUBLE_PREC(0.0000000001);

// The full Mandelbrot set has a large area in the center where the escape
// times all go to the bailout value. To lessen the chance of missed frames
// if the initial zoom level includes a big part of the full Mandelbrot set,
// the zoom level is adjusted to a slightly lower level.
const double MAX_ZOOM(0.1);

// Globals.

// Textures can be bound either to linear memory or to cuda arrays. Textures
// bound to linear memory have good cache performance for 1D texture fetches.
// Linear memory can be written directly by kernels. Textures bound to CUDA
// Arrays have better cache performance for 2D texture fetches. They also allow
// more access modes and filtering. But kernels can not write directly to CUDA
// Arrays on caps < 2.0 (Fermi).
cudaArray* d_fractal_boxes_cuda_array;
cudaArray* d_transform_cuda_array;
cudaArray* d_palette_cuda_arrays[2];

// Textures through which the CUDA Arrays are read.
//
// A texture reference can only be declared as a static global variable and
// cannot be passed as an argument to a function.
texture<float, 2, cudaReadModeElementType> d_fractal_boxes_tex;
texture<uchar4, 2, cudaReadModeElementType> d_palette_tex;

// g_track_index and g_zoom_level together designate what is currently being
// drawn, what is currently being displayed and which colors are currently used.
u32 g_track_index;
double g_zoom_level;

// Configuration values read from the .cfg file.
extern Configuration g_cfg;

// Screen aspect ratio.
double g_aspect_ratio;

// The number of fractal boxes stored in the fractal box buffer. 
u32 g_num_fractal_boxes;

// The zoom step adjusted for boxes per frame and vsync interval.
double g_zoom_step;

// Logarithm of the zoom step is required in the log scale transform.
double g_log_zoom_step;

// The static structure of tracks (vanishing points and temporal palettes) to
// play back. Optimized for use directly from kernels. One copy of it is kept on
// the host and one on the device because various values need to be accessible
// from host and device code.
//
// Constant memory is made available to the kernels automatically. Note: Passing
// a pointer to the memory to the kernel does NOT work (at least not on 1.3).
// The problem may be that the compiler loses track of which kind of memory it
// is, and tries to read from it with a "load global" instead of a "load
// constant". The two are different address spaces on caps < 2.0. 
StaticTracks* g_tracks;
__constant__ StaticTracks d_tracks;

// Buffer for a single box of fractal data, unreduced supersample.
float* d_fractal_box_unreduced_buf;

// Buffer for a single box of fractal data, reduced supersample.
float* d_fractal_box_reduced_buf;

// Buffer for palettes. Two palettes are required because, during
// transitions between tracks, palettes for both tracks are calculated
// and then linearly interpolated.
uchar4* d_palette_buf[2];

// Buffer for log map transformed image.
float* d_transform_buf_unreduced;
size_t transform_buf_pitch_unreduced;

// Buffer for supersampled (finished) buffer.
uchar4* d_transform_buf_reduced;
size_t transform_buf_pitch_reduced;

// "Intra" zoom values are used in the log scale transform, for rendering 
// a single frame. They do not change while the zoom is running.
double g_intra_zoom_end;
double g_log_intra_zoom_end;

// The current insert position in the fractal box buffer.
u32 g_fractal_box_insert_pos;

// The dimmensions of the fractal box buffer.
u32 g_fractal_box_w;
u32 fractal_box_h;
u32 fractal_box_total;

// Timers.
extern CUDATimers* g_cuda_timers;

// Compute capability. Only used for setting threads per block counts
// automatically to get good occupancy on both GT92 and Fermi based
// architectures.
u32 g_compute_capability_major;
u32 g_compute_capability_minor;

// The CUDA device that has been initialized.
extern u32 g_cuda_device;

void Initialize(
  u32 num_resources,
  cudaGraphicsResource** resources,
  StaticTracks* tracks)
{
  g_tracks = tracks;
  g_fractal_box_w = g_cfg.screen_w_;
  fractal_box_h = g_cfg.screen_h_;
  fractal_box_total = 2 * g_fractal_box_w + 2 * fractal_box_h;

  // Compute capability.
	cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, g_cuda_device);
  g_compute_capability_major = prop.major;
  g_compute_capability_minor = prop.minor;

  // Map textures that are shared with OpenGL.
  for (u32 i(0); i < num_resources; ++i) {
    cutilSafeCall(cudaGraphicsSubResourceGetMappedArray(
      &d_transform_cuda_array, resources[i], 0, 0));
  }

  // Screen aspect ratio.
  g_aspect_ratio = static_cast<double>(g_cfg.screen_w_) / static_cast<double>(g_cfg.screen_h_);

  // Set g_intra_zoom_end to the zoom that will create a fractal box that is the
  // width of one pixel. This will be the size of the fractal box when it is
  // first calculated and displayed in the center, vanishing point, of the
  // screen.
  g_intra_zoom_end = (1.0 / g_cfg.screen_w_) / 2.0;

  // Logarithm of intra_zoom_end is required in the log scale transform.
  g_log_intra_zoom_end = log(g_intra_zoom_end);

  // Adjust zoom step for boxes per frame and vsync interval.
  g_zoom_step = pow(g_cfg.zoom_step_, 1.0 / g_cfg.boxes_per_frame_ / g_cfg.vsync_interval_);

  // Set the fractal box resolution to the resolution required to go all the way
  // to g_intra_zoom_end.
  //
  // The range of yn in the transform kernel is (0 - 0.5), not (0 - 1). This is
  // because each section goes from the center of the screen to one edge, which
  // halves the area.
  g_num_fractal_boxes = static_cast<u32>(log(0.5 / g_intra_zoom_end) / log(g_zoom_step));

  // Logarithm of the zoom step is required in the log scale transform.
  g_log_zoom_step = log(g_zoom_step);

  // Newer fractal boxes are inserted in lower positions than older boxes, so
  // the initial insert position is set to be at the last slot in the fractal
  // box buffer.
  g_fractal_box_insert_pos = g_num_fractal_boxes - 1;

  // Create a fractal buffer that is twice as high as the actual
  // g_num_fractal_boxes so there's room for two copies of each fractal box in
  // the buffer. This enables sampling between the two boxes to get interpolated
  // sampling within a given fractal box without getting interpolation between
  // different fractal boxes.
  cudaChannelFormatDesc channelDesc(cudaCreateChannelDesc(32, 0, 0, 0,
    cudaChannelFormatKindFloat));
  cutilSafeCall(cudaMallocArray(&d_fractal_boxes_cuda_array, &channelDesc,
    fractal_box_total, g_num_fractal_boxes * 2));

  // Buffer for a single box of fractal data, unreduced supersample.
  cutilSafeCall(cudaMalloc(&d_fractal_box_unreduced_buf, fractal_box_total * 
    g_cfg.fractal_box_ss_ * sizeof(float)));

  // Buffer for a single box of fractal data, reduced supersample.
  cutilSafeCall(cudaMalloc(&d_fractal_box_reduced_buf, fractal_box_total *
    sizeof(float)));

  // Copy tracks to device buffer.
  cutilSafeCall(cudaMemcpyToSymbol(d_tracks, (void*)g_tracks, sizeof(StaticTracks)));

  // Buffers for palettes. These are required because kernels can't write
  // directly to CUDA Arrays on caps < 2.0.
  cutilSafeCall(cudaMalloc(&d_palette_buf[0], tracks->shared_bailout_ * sizeof(uchar4)));
  cutilSafeCall(cudaMalloc(&d_palette_buf[1], tracks->shared_bailout_ * sizeof(uchar4)));

  // CUDA Arrays for palettes.
  cudaChannelFormatDesc channelDesc2(cudaCreateChannelDesc(8, 8, 8, 8,
    cudaChannelFormatKindUnsigned));
  cutilSafeCall(cudaMallocArray(&d_palette_cuda_arrays[0], &channelDesc2,
    tracks->shared_bailout_ * sizeof(uchar4), 1));
  cutilSafeCall(cudaMallocArray(&d_palette_cuda_arrays[1], &channelDesc2,
    tracks->shared_bailout_ * sizeof(uchar4), 1));

  // Create a 2D array to which the transform kernel will write the transformed
  // image data.
  cutilSafeCall(cudaMallocPitch(&d_transform_buf_unreduced,
    &transform_buf_pitch_unreduced, g_cfg.screen_w_ * sizeof(float) *
    g_cfg.transform_ss_x_, g_cfg.screen_h_ * g_cfg.transform_ss_y_));

  // Create a 2D array to which the reduce kernel will write the reduced
  // image data.
  cutilSafeCall(cudaMallocPitch(&d_transform_buf_reduced,
    &transform_buf_pitch_reduced, g_cfg.screen_w_ * sizeof(uchar4),
    g_cfg.screen_h_));

  // Set up texture for fractal box.

  // Wrap mode is used to create an "endless" fractal buffer.
  // cudaAddressModeWrap is only supported for normalized texture coordinates.
  // This is the only reason normalized texture coordinates are used here.
  //
  // Different boxes in the fractal buffer are at different scales, so it
  // doesn't make sense to interpolate between them. However, it is desirable to
  // interpolate between escape values in a given box. If it was possible to
  // interpolate between pixels in one dimension and not the other in hardware,
  // that would be ideal. Since that functionality is not available, the same
  // effect is achieved by duplicating each fractal box and sampling in the
  // center between the two copies.
  //
  // TODO: Try plain linear interpolation between fractal boxes anyway. What is
  // mathematically correct is not necessarily what gives the best visual
  // result.
  //
  // TODO: It seems that in some locations, linear interpolation between escape
  // values in a single fractal box actually reduce visual quality. Find out why
  // and possibly fix.
  d_fractal_boxes_tex.addressMode[0] = cudaAddressModeWrap;
  d_fractal_boxes_tex.addressMode[1] = cudaAddressModeWrap;
  //d_fractal_boxes_tex.filterMode = cudaFilterModePoint;
  d_fractal_boxes_tex.filterMode = cudaFilterModeLinear;
  d_fractal_boxes_tex.normalized = true;
  // Bind a texture to array. Uses the C++ api version of
  // cudaBindTextureToArray(). Channel desc is inherited from the array.
  cudaBindTextureToArray(d_fractal_boxes_tex, d_fractal_boxes_cuda_array);

  // Set up texture for palette.
  d_palette_tex.addressMode[0] = cudaAddressModeClamp;
  d_palette_tex.addressMode[1] = cudaAddressModeClamp;
  d_palette_tex.filterMode = cudaFilterModePoint;
  d_palette_tex.normalized = false;
  // Bind a texture to array. Uses the C++ api version of
  // cudaBindTextureToArray(). Channel desc is inherited from the array.
  cudaBindTextureToArray(d_palette_tex, d_palette_cuda_arrays[0]);

  // Prepare for zoom of first track. Setting the track index to the last track
  // and zoom level to 0.0 triggers an immediate track switch to track 0.
  g_track_index = g_tracks->count_ - 1;
  g_zoom_level = 0.0;
}

void Shutdown()
{
  cudaFreeArray(d_fractal_boxes_cuda_array);
  cutilSafeCall(cudaFree(d_fractal_box_unreduced_buf));
  cutilSafeCall(cudaFree(d_fractal_box_reduced_buf));
  cutilSafeCall(cudaFree(d_palette_buf[0]));
  cutilSafeCall(cudaFree(d_palette_buf[1]));
  cudaFreeArray(d_palette_cuda_arrays[0]);
  cudaFreeArray(d_palette_cuda_arrays[1]);
  cutilSafeCall(cudaFree(d_transform_buf_unreduced));
  cutilSafeCall(cudaFree(d_transform_buf_reduced));
}

// ----------------------------------------------------------------------------
// Utils.
// ----------------------------------------------------------------------------

// Increase the grid size by 1 if the block w or h does not divide evenly by the
// thread block dimensions. In that case, checking for and skipping unneeded
// threads in the kernels is also required.
u32 DivUp(u32 a, u32 b)
{
  return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

// Clamp a value on host. Functions for Clamping on the device are already in
// the cuda library.
template<typename FP>
FP Clamp(FP v, FP a, FP b) {
  if (v < a) {
    return a;
  }
  if (v > b) {
    return b;
  }
  return v;
}

// The current zoom value together with the vanishing point describes the
// fractal box that was most recently drawn. In the current frame, the size of
// that box is the same as the size of one pixel. In some calculations, the zoom
// value that would be needed for generating the box that is just about to go
// off the screen is required. This is the same value that would be needed for
// generating the corresponding full fractal image without using a log scale
// map.
double ZoomFull(double zoom_level) {
  return zoom_level * pow(g_zoom_step, static_cast<double>(g_num_fractal_boxes));
}

// Clamp zoom level to the highest and lowest values supported.
double ZoomClamp(double zoom) {
  if (zoom > MAX_ZOOM) {
    return MAX_ZOOM;
  }
  if (g_cfg.single_precision_ && zoom < MIN_ZOOM_SINGLE_PREC) {
    return MIN_ZOOM_SINGLE_PREC;
  }
  if (zoom < MIN_ZOOM_DOUBLE_PREC) {
    return MIN_ZOOM_DOUBLE_PREC;
  }
  return zoom;
}

// ----------------------------------------------------------------------------
// Calculate mandelbrot escape times.
// ----------------------------------------------------------------------------

template <typename FP>
__global__ void MandelbrotKernel(
  float* d_fractal_box_unreduced_buf,
  u32 w, u32 h,
  FP cr1, FP ci1, FP cr2, FP ci2,
  u32 bailout)
{
  u32 n(blockIdx.x * blockDim.x + threadIdx.x);

  // Linear interpolation (lerp).
  FP cr;
  FP ci;

  // A fractal box contains all four sides of a fractal square. The box is drawn
  // in such a way that there are no discontinuities in the buffer:
  //
  // - Top line: left to right
  // - Right line: top to bottom
  // - Bottom line: right to left
  // - Left line: bottom to top
  //
  // Without this approach, lines appear, radiating out from the center and
  // through the corners in the fractal boxes. They're caused by inexact sample
  // locations in the transform kernel.
  //
  // On caps 1.3, only one kernel can run at a time, so it is important to create
  // as many threads as possible. Because of that, it was decided to have a single
  // kernel handle all sides of the fractal boxes. The extra logic takes very little
  // time compared to the mandelbrot escape time calculation itself but if support
  // for caps 1.3 is dropped, it would be cleaner to issue 4 times as many kernels,
  // and leave the logic related to selecting which side in the box to calculate
  // in host code.

  // Select box side.

  // Top.
  if (n < w) {
    // cr1, ci1, cr2, ci1
    FP norm((FP)n / (FP)w);
    cr = norm * (cr2 - cr1) + cr1;
    ci = ci1;
  }
  // Right.
  else if (n < w + h) {
    // cr1, ci2, cr1, ci1
    FP norm((FP)(n - w) / (FP)h);
    cr = cr2;
    ci = norm * (ci2 - ci1) + ci1;
  }
  // Bottom.
  else if (n < 2 * w + h) {
    // cr2, ci2, cr1, ci2
    FP norm(1.0 - (FP)(n - w - h) / (FP)w);
    cr = norm * (cr2 - cr1) + cr1;
    ci = ci2;
  }
  // Left.
  else if (n < 2 * w + 2 * h) {
    // cr2, ci1, cr2, ci2
    FP norm(1.0 - (FP)(n - 2 * w - h) / (FP)h);
    cr = cr1;
    ci = norm * (ci2 - ci1) + ci1;
  }
  else {
    return;
  }

  // Mandelbrot escape time calculation.

  FP zi(0.0), zr(0.0), zr2(0.0), zi2(0.0), zit(0.0);
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

  d_fractal_box_unreduced_buf[n] = static_cast<float>(iter);
}

void RunMandelbrotKernel(double cr1, double ci1, double cr2, double ci2)
{
  CUDATimerRun run_mandelbrot_timer(*g_cuda_timers, kMandelbrot);

  // threads_per_block should be obtained from the occupancy calculator. It was
  // determined that 128 gives good occupancy for compute capability 1.3 and 192
  // for 2.1, for these kernels.
  u32 threads_per_block_x(g_compute_capability_major < 2 ? 128 : 192);
  u32 threads_per_block_y(1);

  // Dimension of each thread block (number of threads to launch in each block).
  dim3 block_dim(threads_per_block_x, threads_per_block_y);

  // Dimension of the grid (number of blocks to launch).
  dim3 grid_dim(DivUp(fractal_box_total * g_cfg.fractal_box_ss_, threads_per_block_x));

  if (g_cfg.single_precision_) {
    MandelbrotKernel<float><<<grid_dim, block_dim, 0>>>(
      d_fractal_box_unreduced_buf,
      g_fractal_box_w * g_cfg.fractal_box_ss_,
      fractal_box_h * g_cfg.fractal_box_ss_,
      static_cast<float>(cr1), static_cast<float>(ci1),
      static_cast<float>(cr2), static_cast<float>(ci2),
      g_tracks->shared_bailout_);
  }
  else {
    MandelbrotKernel<double><<<grid_dim, block_dim, 0>>>(
      d_fractal_box_unreduced_buf,
      g_fractal_box_w * g_cfg.fractal_box_ss_,
      fractal_box_h * g_cfg.fractal_box_ss_,
      cr1, ci1, cr2, ci2,
      g_tracks->shared_bailout_);
  }

  cutilCheckMsg("MandelbrotKernel failed");
}

// ----------------------------------------------------------------------------
// Fractal box reduce.
// ----------------------------------------------------------------------------

__global__ void FractalReduceKernel(
  float* dst,
  float* src,
  u32 w,
  u32 fractal_box_ss)
{
  u32 x(blockIdx.x * blockDim.x + threadIdx.x);

  if (x >= w) {
    return;
  }

  u32 off(x * fractal_box_ss);
  float acc(0.0);
  for (u32 i(0); i < fractal_box_ss; ++i) {
    acc += src[off + i];
  }

  dst[x] = acc / fractal_box_ss;
}

void RunFractalReduceKernel()
{
  CUDATimerRun run_fractal_reduce_timer(*g_cuda_timers, kFractalReduce);

  u32 threads_per_block_x(g_compute_capability_major < 2 ? 128 : 192);
  u32 threads_per_block_y(1);
  dim3 block_dim(threads_per_block_x, threads_per_block_y);
  dim3 grid_dim(DivUp(fractal_box_total, threads_per_block_x));

  FractalReduceKernel<<<grid_dim, block_dim, 0>>>(
    d_fractal_box_reduced_buf,
    d_fractal_box_unreduced_buf,
    fractal_box_total,
    g_cfg.fractal_box_ss_);

  cutilCheckMsg("FractalReduceKernel failed");

  // Insert the new box into the log scale map. Two copies are created side by
  // side so that linear sampling can be used between them to get interpolation
  // within the values in a box without getting interpolation between boxes.
  //
  // The addressing was tested by inserting black boxes in each odd location and
  // making sure those black boxes did not appear in the final result when the
  // sampling location was shifted from between the two boxes to in the center
  // of the even box.
  int s(fractal_box_total * sizeof(float));
  cutilSafeCall(cudaMemcpyToArray(d_fractal_boxes_cuda_array, 0,
    g_fractal_box_insert_pos * 2, d_fractal_box_reduced_buf, s,
    cudaMemcpyDeviceToDevice));
  cutilSafeCall(cudaMemcpyToArray(d_fractal_boxes_cuda_array, 0,
    g_fractal_box_insert_pos * 2 + 1, d_fractal_box_reduced_buf, s,
    cudaMemcpyDeviceToDevice));
}

// ----------------------------------------------------------------------------
// Palette.
// ----------------------------------------------------------------------------

__device__ u8 u8Lerp(u8 a, u8 b, float t)
{
  return (u8)(static_cast<float>(a) + t * (static_cast<float>(b) - static_cast<float>(a)));
}

__device__ uchar4 uchar4Lerp(uchar4 a, uchar4 b, float t)
{
  return make_uchar4(
    u8Lerp(a.x, b.x, t),
    u8Lerp(a.y, b.y, t),
    u8Lerp(a.z, b.z, t),
    0);
}

__device__ uchar4 ColorLerp(StaticTemporalPaletteKey& key, float color_pos)
{
  for (int i(0); ; ++i) {
    float pos1(key.spatial_palette_keys_[i].pos_);
    float pos2(key.spatial_palette_keys_[i + 1].pos_);
    if (color_pos >= pos1 && color_pos <= pos2) {
      float d((color_pos - pos1) / (pos2 - pos1));
      return uchar4Lerp(
        key.spatial_palette_keys_[i].color_,
        key.spatial_palette_keys_[i + 1].color_,
        d);
    }
  }
}

__global__ void GeneratePaletteKernel(
  uchar4* d_palette_buf,
  u32 g_track_index,
  float temporal_palette_pos,
  u32 bailout)
{
  u32 palette_idx(blockIdx.x * blockDim.x + threadIdx.x);

  if (palette_idx >= bailout) {
    return;
  }

  StaticTemporalPalette& temporal_palette_(d_tracks.tracks_[g_track_index].temporal_palette_);
  float color_pos(static_cast<float>(palette_idx) / static_cast<float>(bailout));

  // Find the two temporal palette keys over which lerp should be done.
  for (int i(0); ; ++i) {
    StaticTemporalPaletteKey& key1(temporal_palette_.temporal_palette_keys_[i]);
    StaticTemporalPaletteKey& key2(temporal_palette_.temporal_palette_keys_[i + 1]);
    if (temporal_palette_pos >= key1.pos_ && temporal_palette_pos <= key2.pos_) {
      // Found the keys. Do the lerp.
      uchar4 c1(ColorLerp(key1, color_pos));
      uchar4 c2(ColorLerp(key2, color_pos));
      float d((temporal_palette_pos - key1.pos_) / (key2.pos_ - key1.pos_));
      d_palette_buf[palette_idx] = uchar4Lerp(c1, c2, d);
      return;
    }
  }
}

void RunGeneratePaletteKernel(
  uchar4* d_palette_buf, u32 g_track_index,
  double temporal_palette_pos,
  u32 bailout)
{
  u32 threads_per_block_x(g_compute_capability_major < 2 ? 128 : 192);
  const u32 threads_per_block_y(1);
  dim3 block_dim(threads_per_block_x, threads_per_block_y);
  dim3 grid_dim(DivUp(bailout, threads_per_block_x));

  GeneratePaletteKernel<<<grid_dim, block_dim, 0>>>(
    d_palette_buf,
    g_track_index,
    static_cast<float>(temporal_palette_pos),
    bailout);

  cutilCheckMsg("GeneratePaletteKernel failed");

  cutilSafeCall(cudaMemcpyToArray(d_palette_cuda_arrays[0], 0, 0,
    d_palette_buf, bailout * sizeof(uchar4), cudaMemcpyDeviceToDevice));
}

// -----------------------------------
// Debug: Generate grayscale palette.
// -----------------------------------

__global__ void DebugGenerateGrayscalePaletteKernel(
  uchar4* d_palette_buf,
  u32 bailout)
{
  u32 palette_idx(blockIdx.x * blockDim.x + threadIdx.x);

  if (palette_idx >= bailout) {
    return;
  }

  u8 v(static_cast<u8>(static_cast<float>(palette_idx) /
    static_cast<float>(bailout) * 255.0f));
  d_palette_buf[palette_idx] = make_uchar4(v, v, v, 0);
}

void RunDebugGenerateGrayscalePalette(uchar4* d_palette_buf, u32 bailout) {
  CUDATimerRun run_palette_timer(*g_cuda_timers, kPalettes);

  u32 threads_per_block_x(g_compute_capability_major < 2 ? 128 : 192);
  u32 threads_per_block_y(1);
  dim3 block_dim(threads_per_block_x, threads_per_block_y);
  dim3 grid_dim(DivUp(bailout, threads_per_block_x));

  DebugGenerateGrayscalePaletteKernel<<<grid_dim, block_dim, 0>>>
    (d_palette_buf, bailout);

  cutilCheckMsg("DebugGenerateGrayscalePaletteKernel failed");

  cutilSafeCall(cudaMemcpyToArray(d_palette_cuda_arrays[0], 0, 0,
    d_palette_buf, bailout * sizeof(uchar4), cudaMemcpyDeviceToDevice));
}

// -----------------------------------
// Linear interpolation between palettes.
// -----------------------------------

__global__ void LerpPaletteBufsKernel(
  uchar4* palette_buf1,
  uchar4* palette_buf2,
  float pos,
  u32 bailout)
{
  u32 palette_idx(blockIdx.x * blockDim.x + threadIdx.x);
  if (palette_idx >= bailout) {
    return;
  }
  palette_buf1[palette_idx] = uchar4Lerp(palette_buf1[palette_idx],
    palette_buf2[palette_idx], pos);
}

void RunLerpPaletteBufsKernel(
  uchar4* palette_buf1,
  uchar4* palette_buf2,
  double pos,
  u32 bailout)
{
  u32 threads_per_block_x(g_compute_capability_major < 2 ? 128 : 192);
  u32 threads_per_block_y(1);
  dim3 block_dim(threads_per_block_x, threads_per_block_y);
  dim3 grid_dim(DivUp(bailout, threads_per_block_x));

  LerpPaletteBufsKernel<<<grid_dim, block_dim, 0>>>(
    d_palette_buf[0],
    d_palette_buf[1],
    static_cast<float>(pos),
    bailout);

  cutilCheckMsg("LerpPaletteBufsKernel failed");

  cutilSafeCall(cudaMemcpyToArray(d_palette_cuda_arrays[0], 0, 0,
    palette_buf1, bailout * sizeof(uchar4), cudaMemcpyDeviceToDevice));
}

// Get the temporal palette position of a track given a zoom level.
double GetTemporalPalettePosByZoomLevel(u32 track_index, double zoom_level) {
  StaticTrack& track(g_tracks->tracks_[track_index]);
  double zoom_begin(track.fractal_spec_.zoom_begin_);
  double zoom_end(track.fractal_spec_.zoom_end_);
  // temporal_palette_pos:
  // - (2, 1): Single fractal.
  // - (1, 0): Transition (two fractals on screen at the same time).
  double temporal_palette_pos(log(zoom_level / zoom_end) / log(zoom_end / zoom_begin) + 2.0);
  // temporal_palette_pos must be clamped because some inaccuracy in calculations
  // causes it to go beyond its boundaries during a couple of frames.
  return Clamp(temporal_palette_pos - 1.0, 0.0, 1.0);
}

void CreatePalette(StaticTrack& track) {
  CUDATimerRun run_palette_timer(*g_cuda_timers, kPalettes);

  double zoom_begin(track.fractal_spec_.zoom_begin_);
  //double zoom_end(track.fractal_spec_.zoom_end_);
  // Get the zoom value that would be needed for generating the box that is just
  // about to go off the screen.
  double zoom_full(ZoomFull(g_zoom_level));
  if (zoom_full <= ZoomClamp(zoom_begin)) {
    // The track from which fractal boxes are currently being calculated covers
    // the whole screen.
    //
    // When the current fractal covers the entire screen, there is no fading
    // between fractals to be done. One palette is generated and used directly.
    double temporal_palette_pos_clamp(GetTemporalPalettePosByZoomLevel(
      g_track_index, ZoomFull(g_zoom_level)));
    RunGeneratePaletteKernel(d_palette_buf[0], g_track_index,
      temporal_palette_pos_clamp, g_tracks->shared_bailout_);
    //printf("single: temporal_palette_pos_clamp(%f)\n", temporal_palette_pos_clamp);
  }
  else {
    // The track from which fractal boxes are currently being calculated covers
    // only part of the screen. The outer parts of the screen are covered by
    // boxes that were calculated for the previous track. Palettes for the
    // previous track are generated and gradually blended with the palette of
    // the current track in such a way that when the current track fills the
    // screen, the blend is complete.

    // Index of previous track.
    u32 track_index_prev((!g_track_index ? g_tracks->count_ : g_track_index) - 1);
    // Generate the last palette that was used for the previous track.
    StaticTrack& track_prev(g_tracks->tracks_[track_index_prev]);
    double p1(GetTemporalPalettePosByZoomLevel(track_index_prev,
      ZoomClamp(track_prev.fractal_spec_.zoom_end_)));
    RunGeneratePaletteKernel(d_palette_buf[0], track_index_prev, p1,
      g_tracks->shared_bailout_);
    // Generate the first palette that will be used for the current track.
    double p2(GetTemporalPalettePosByZoomLevel(g_track_index,
      ZoomClamp(track.fractal_spec_.zoom_begin_)));
    RunGeneratePaletteKernel(d_palette_buf[1], g_track_index, p2,
      g_tracks->shared_bailout_);
    // Number of frames rendered in current track.
    double zoom_frames(log(ZoomClamp(zoom_begin) / g_zoom_level) / g_log_zoom_step);
    // Find out how far into the transition rendering has progressed.
    double transition_pos(zoom_frames / static_cast<double>(g_num_fractal_boxes));
    // Use the value to blend the palettes for current and previous track.
    RunLerpPaletteBufsKernel(d_palette_buf[0], d_palette_buf[1],
      transition_pos, g_tracks->shared_bailout_);
    //printf("trans:  p2(%f)\n", p2);
  }
}

// ---------------------------------------------------------------------------
// Transform.
// Log scale map transform from fractal box buffer to unreduced buffer.
// ---------------------------------------------------------------------------

__global__ void TransformKernel(
  float fractal_buf_w_norm, float fractal_buf_h_norm,
  float g_fractal_box_insert_pos, float g_num_fractal_boxes,
  float* transform, u32 w, u32 h, size_t pitch,
  float g_log_zoom_step, float g_log_intra_zoom_end)
{
  u32 x(blockIdx.x * blockDim.x + threadIdx.x);
  u32 y(blockIdx.y); // threads_per_block_y == 1

  float x_norm = static_cast<float>(x) / static_cast<float>(w);
  float y_norm = static_cast<float>(y) / static_cast<float>(h);

  if (x >= w) {
    return;
  }

  // Determine which section this pixel is in and set fractal box sample
  // positions. By manipulating x and y values for the various sections, a
  // virtual "rotate" is performed of the fractal box buffer so that higher
  // numbered boxes in the buffer are closer to the edge of the screen in each
  // section.
  //
  // TODO: This handles all 4 sides of fractal boxes in a single kernel. See
  // the comments for the mandelbrot calculation as to why this approach was
  // chosen. SIGNIFICANT speedup may be possible though, by factoring the
  // box side selection code out of the kernel.
  float xn;
  float yn;
  float off;
  float span;

  // 1200 = white
  if (x_norm > y_norm) {
    if (1.0f - x_norm > y_norm) {
      // Top section.
      xn = x_norm;
      yn = 0.5f - y_norm;
      off = 0.0f;
      span = fractal_buf_w_norm;
    }
    else {
      // Right section.
      xn = y_norm;
      yn = x_norm - 0.5f;
      off = fractal_buf_w_norm;
      span = fractal_buf_h_norm;
    }
  }
  else {
    if (1.0f - x_norm < y_norm) {
      // Bottom section.
      xn = 1.0f - x_norm;
      yn = y_norm - 0.5f;
      off = fractal_buf_w_norm + fractal_buf_h_norm;
      span = fractal_buf_w_norm;
    }
    else {
      // Left section.
      xn = 1.0f - y_norm;
      yn = 0.5f - x_norm;
      off = 2.0f * fractal_buf_w_norm + fractal_buf_h_norm;
      span = fractal_buf_h_norm;
    }
  }
  // Find the sample position within a fractal box.
  //
  // 0.5f + ((xn - 0.5f) / yn: To create triangle shaped sections, the range of
  // xn must widen as locations closer to the center of the screen are
  // calculated. When the range gets wider, samples in the fractal data get
  // further apart, creating a triangle that "tapers off" towards the center of
  // the screen. That is done by taking the original [0, 1] range of xn and
  // centering it around zero by subtracing 0.5, giving [-0.5, 0.5]. Then that
  // value is divided by yn. 0.5 is then added to move the center back to where
  // it was before the adjustment.
  //
  // * span: Each fractal box holds fractal data for all four sections. The two
  // vertical sections share one size and the two horizontal sections share one.
  // Since the buffer is normalized, the relative sizes of each section is the
  // same as the relative sizes in the aspect ratio. By multiplying with span,
  // the sample range is adjusted to that of the size of the given section.
  //
  // + off: Add the start of the fractal data that should be used for this
  // section.
  float x_sample((0.5f + ((xn - 0.5f) / (yn * 2.0f))) * span + off);
  //if (x_sample == 0.25) {
  //  transform[x + y * pitch] = 2000.0f;
  //}
  // y_pos can be cached, which reduces the need to calculate the log() to only
  // once in each block if all the threads are in the same box. That was tried
  // in another version of this kernel but with no difference in performance.

  // Find the fractal box to sample from.

  // "log map" lookup that selects the fractal box to sample from based on how
  // close to the center of the screen this pixel is (as designated by yn).

  // I tried removing the log operation by replacing log(yn) with just yn, and
  // active warps / active cycle actually went down from 29 to 28. I think
  // it means that the log instruction latency is completely hidden. Though
  // that may no longer be the case after I combine transform and reduce.
  float i((((log(yn) - g_log_intra_zoom_end)) / g_log_zoom_step));

  // Find the position of the fractal box within the normalized fractal box
  // buffer.
  //
  // rintf(i) * 2.0f + 1.0f: There are two copies of each fractal box in the
  // buffer so that sampling that does linear interpolation only within the
  // pixels in a single fractal box, and not between fractal boxes, can be done.
  // To find the position that is between the two copies of the fractal box to
  // sample from, the initial fractal box index is rounded to the nearest
  // integer, and then multiplied by 2 to get an even number. Then one is added,
  // resulting in an odd number that is centered between the two copies of the
  // fractal box.
  //
  // g_fractal_box_insert_pos * 2: Instead of scrolling the fractal buffer to
  // get the zooming effect, the starting position within it is continuously
  // changed. This adds that starting position. The position is then multiplied
  // by 2.0 to take into account that the buffer contains two copies of each
  // box.
  //
  // g_num_fractal_boxes * 2: Because the fractal buffer is normalized, the
  // position is divided by the resolution of the buffer to get a position in
  // the [0, 1] range. The position is then multiplied by 2 because the buffer
  // contains two copies of each box, so it is twice as wide as its resolution.
  float y_pos((rintf(i) * 2.0f + 1.0f + g_fractal_box_insert_pos * 2.0f) /
    (g_num_fractal_boxes * 2.0f));

  transform[x + y * pitch] = tex2D(d_fractal_boxes_tex, x_sample, y_pos);
}

void RunTransformKernel()
{
  CUDATimerRun run_transform_timer(*g_cuda_timers, kTransform);

  u32 threads_per_block_x(g_compute_capability_major < 2 ? 128 : 192);
  u32 threads_per_block_y(1);
  dim3 block_dim(threads_per_block_x, threads_per_block_y);
  dim3 grid_dim(DivUp(g_cfg.screen_w_ * g_cfg.transform_ss_x_,
    threads_per_block_x), DivUp(g_cfg.screen_h_ * g_cfg.transform_ss_y_, threads_per_block_y));

  float fractal_buf_w_norm(static_cast<float>(g_fractal_box_w) /
    (static_cast<float>(g_fractal_box_w) + static_cast<float>(fractal_box_h)) / 2.0f);
  float fractal_buf_h_norm(static_cast<float>(fractal_box_h) / (static_cast<float>(g_fractal_box_w) +
    static_cast<float>(fractal_box_h)) / 2.0f);

  TransformKernel<<<grid_dim, block_dim, 0>>>(
    fractal_buf_w_norm, fractal_buf_h_norm, static_cast<float>(g_fractal_box_insert_pos) - 1,
    static_cast<float>(g_num_fractal_boxes), d_transform_buf_unreduced, g_cfg.screen_w_ *
    g_cfg.transform_ss_x_, g_cfg.screen_h_ * g_cfg.transform_ss_y_,
    transform_buf_pitch_unreduced / sizeof(float),
    static_cast<float>(g_log_zoom_step), static_cast<float>(g_log_intra_zoom_end));

  cutilCheckMsg("TransformKernel failed");
}

// ---------------------------------------------------------------------------
// Debug: Fill the unreduced buffer with a checkerboard pattern.
// This is for debugging the reduce kernel.
// ---------------------------------------------------------------------------

__global__ void UnreducedCheckerboardKernel(float* transform, size_t pitch)
{
  u32 x(blockIdx.x * blockDim.x + threadIdx.x);
  u32 y(blockIdx.y); // threads_per_block_y == 1

  // Fill supersampled (final) texture with checkerboard pattern.
  if (x > y / 3) {
    transform[x + y * pitch] = 2000.0f;
  }
  else {
    transform[x + y * pitch] = 0.0f;
  }
}

void RunUnreducedCheckerboardKernel()
{
  u32 threads_per_block_x(g_compute_capability_major < 2 ? 128 : 192);
  u32 threads_per_block_y(1);
  dim3 block_dim(threads_per_block_x, threads_per_block_y);
  dim3 grid_dim(DivUp(g_cfg.screen_w_ * g_cfg.transform_ss_x_, threads_per_block_x),
                DivUp(g_cfg.screen_h_ * g_cfg.transform_ss_y_, threads_per_block_y));

  UnreducedCheckerboardKernel<<<grid_dim, block_dim, 0>>>(
    d_transform_buf_unreduced, transform_buf_pitch_unreduced / sizeof(float));

  cutilCheckMsg("UnreducedCheckerboardKernel");
}

// ---------------------------------------------------------------------------
// Debug: Copy fractal box buffer to unreduced buffer.
//
// For debugging, copy the fractal box buffer directly into the unreduced
// buffer. Normally, it is copied into the unreduced buffer via the log scale
// map transform.
// ---------------------------------------------------------------------------

__global__ void DebugCopyBoxBufToUnreducedBuf(
  float* transform, u32 w, u32 h, size_t pitch)
{
  u32 x(blockIdx.x * blockDim.x + threadIdx.x);
  u32 y(blockIdx.y); // threads_per_block_y == 1

  float x_norm(static_cast<float>(x) / static_cast<float>(w));
  float y_norm(static_cast<float>(y) / static_cast<float>(h));

  transform[x + y * pitch] = tex2D(d_fractal_boxes_tex, x_norm, y_norm);
}

void RunDebugCopyBoxBufToUnreducedBuf()
{
  const u32 threads_per_block_x(256);
  const u32 threads_per_block_y(1);
  dim3 block_dim(threads_per_block_x, threads_per_block_y);
  dim3 grid_dim(
    DivUp(g_cfg.screen_w_ * g_cfg.transform_ss_x_, threads_per_block_x),
    DivUp(g_cfg.screen_h_ * g_cfg.transform_ss_y_,
    threads_per_block_y));

  DebugCopyBoxBufToUnreducedBuf<<<grid_dim, block_dim, 0>>>(
    d_transform_buf_unreduced, g_cfg.screen_w_ * g_cfg.transform_ss_x_,
    g_cfg.screen_h_ * g_cfg.transform_ss_y_,
    transform_buf_pitch_unreduced / sizeof(float));

  cutilCheckMsg("DebugCopyBoxBufToUnreducedBuf failed");
}

// ---------------------------------------------------------------------------
// Debug: Copy the unreduced buffer directly to the display buffer.
// For debugging, copy the upper left section of the unreduced buffer into the
// reduced buffer.
// ---------------------------------------------------------------------------

__global__ void DebugCopyUnreducedToDisplayKernel(
  float* transform, size_t transform_pitch,
  uchar4* supersample, size_t supersample_pitch,
  u32 w,
  float norm_fact)
{
  u32 x(blockIdx.x * blockDim.x + threadIdx.x);
  u32 y(blockIdx.y); // threads_per_block_y == 1
  if (x >= w) {
    return;
  }
  float acc = transform[x + y * transform_pitch];
  supersample[x + y * supersample_pitch] = make_uchar4(acc, acc, acc, 1.0f);
}

void RunDebugCopyUnreducedToDisplayKernel()
{
  u32 threads_per_block_x(g_compute_capability_major < 2 ? 128 : 192);
  u32 threads_per_block_y(1);
  dim3 block_dim(threads_per_block_x, threads_per_block_y);
  dim3 grid_dim(DivUp(g_cfg.screen_w_, threads_per_block_x),
                DivUp(g_cfg.screen_h_, threads_per_block_y));

  float norm_fact(1.0f / ((static_cast<float>(g_cfg.transform_ss_x_) *
    static_cast<float>(g_cfg.transform_ss_y_)) * 256.0f));

  DebugCopyUnreducedToDisplayKernel<<<grid_dim, block_dim, 0>>>(
    d_transform_buf_unreduced, transform_buf_pitch_unreduced / sizeof(float),
    d_transform_buf_reduced, transform_buf_pitch_reduced / sizeof(uchar4),
    g_cfg.screen_w_,
    norm_fact);

  cutilCheckMsg("DebugCopyUnreducedToDisplayKernel failed");

  // Copy the finished, reduced frame to the CUDA Array used for OpenGL interop.
  cutilSafeCall(cudaMemcpy2DToArray(d_transform_cuda_array, 0, 0, 
    d_transform_buf_reduced, transform_buf_pitch_reduced,
    g_cfg.screen_w_ * sizeof(uchar4), g_cfg.screen_h_,
    cudaMemcpyDeviceToDevice));
}

// ---------------------------------------------------------------------------
// Reduce and colorize.
// - SSAA from unreduced to reduced buffer.
// - Color pixels with palette.
// ---------------------------------------------------------------------------

// In the initial version of this kernel, the values were first reduced and then
// pixels were calculated by using the values for palette lookups. This caused
// poor quality because the palette is not linearly assigned to values. To
// correctly generate the colorized image, colors must first be assigned to the
// values and the resulting pixels must then be reduced.
__global__ void ReduceAndColorizeKernel(
  float* transform, size_t transform_pitch,
  uchar4* supersample, size_t supersample_pitch,
  u32 w, u32 h,
  u32 transform_ss_x, u32 transform_ss_y,
  float norm_fact,
  uchar4* d_palette_buf)
{
  u32 x(blockIdx.x * blockDim.x + threadIdx.x);
  u32 y(blockIdx.y); // threads_per_block_y == 1
  if (x >= w) {
    return;
  }
  float r(0.0f);
  float g(0.0f);
  float b(0.0f);
  u32 off(x * transform_ss_x + y * transform_ss_y * transform_pitch);
  for (u32 j(0); j < transform_ss_y; ++j) {
    for (u32 i(0); i < transform_ss_x; ++i) {
      // Use palette to color the pixel.
      uchar4 color(tex2D(d_palette_tex, static_cast<u32>(transform[off + i]), 0));
      r += static_cast<float>(color.x);
      g += static_cast<float>(color.y);
      b += static_cast<float>(color.z);
    }
    off += transform_pitch;
  }

  supersample[x + y * supersample_pitch] = make_uchar4(
    r * norm_fact,
    g * norm_fact,
    b * norm_fact,
    255);
}

void RunReduceAndColorizeKernel()
{
  CUDATimerRun run_reduce_and_colorize_timer(*g_cuda_timers, kTransformReduceAndColorize);

  u32 threads_per_block_x(g_compute_capability_major < 2 ? 128 : 192);
  u32 threads_per_block_y(1);
  dim3 block_dim(threads_per_block_x, threads_per_block_y);
  dim3 grid_dim(DivUp(g_cfg.screen_w_, threads_per_block_x),
    DivUp(g_cfg.screen_h_, threads_per_block_y));

  float norm_fact(1.0f / ((static_cast<float>(g_cfg.transform_ss_x_) *
    static_cast<float>(g_cfg.transform_ss_y_))));

  ReduceAndColorizeKernel<<<grid_dim, block_dim, 0>>>(
    d_transform_buf_unreduced, transform_buf_pitch_unreduced / sizeof(float),
    d_transform_buf_reduced, transform_buf_pitch_reduced / sizeof(uchar4),
    g_cfg.screen_w_, g_cfg.screen_h_,
    g_cfg.transform_ss_x_, g_cfg.transform_ss_y_,
    norm_fact,
    d_palette_buf[0]);

  cutilCheckMsg("ReduceAndColorizeKernel failed");

  // Copy the finished, reduced frame to the CUDA Array used for OpenGL interop.
  cutilSafeCall(cudaMemcpy2DToArray(d_transform_cuda_array, 0, 0, 
    d_transform_buf_reduced, transform_buf_pitch_reduced,
    g_cfg.screen_w_ * sizeof(uchar4), g_cfg.screen_h_,
    cudaMemcpyDeviceToDevice));
}

// ----------------------------------------------------------------------------
// Run kernels.
// ----------------------------------------------------------------------------

void FractalCalc(bool mouse_button_left, bool mouse_button_right)
{
  StaticTrack* track(&g_tracks->tracks_[g_track_index]);

  // Determine track and zoom level to be rendered.
  if ((g_cfg.single_precision_ && ZoomFull(g_zoom_level) < MIN_ZOOM_SINGLE_PREC) ||
      (!g_cfg.single_precision_ && ZoomFull(g_zoom_level) < MIN_ZOOM_DOUBLE_PREC) ||
      (ZoomFull(g_zoom_level) < track->fractal_spec_.zoom_end_)) {
    if (++g_track_index == g_tracks->count_) {
      g_track_index = 0;
    }
    track = &g_tracks->tracks_[g_track_index];
    g_zoom_level = ZoomClamp(track->fractal_spec_.zoom_begin_);
  }

  // Calculate fractal boxes.
  if (!mouse_button_left) {
    for (u32 j(0); j < g_cfg.boxes_per_frame_; ++j) {
      double center_r(track->fractal_spec_.center_r_);
      double center_i(track->fractal_spec_.center_i_);
      // Translate (center + zoom) to (upper left + lower right)
      double cr1(center_r - g_zoom_level);
      double cr2(center_r + g_zoom_level);
      double ci1(center_i - (g_zoom_level / g_aspect_ratio));
      double ci2(center_i + (g_zoom_level / g_aspect_ratio));

      RunMandelbrotKernel(cr1, ci1, cr2, ci2);
      RunFractalReduceKernel();

      // Update
      if (g_fractal_box_insert_pos == 0) {
        g_fractal_box_insert_pos = g_num_fractal_boxes - 1;
      }
      else {
        --g_fractal_box_insert_pos;
      }

      // Set the zoom level for the next frame.
      g_zoom_level /= g_zoom_step;
    }
  }

  if (!mouse_button_right) {
    // Generate palette.
    if (g_cfg.grayscale_) {
      RunDebugGenerateGrayscalePalette(d_palette_buf[0], g_tracks->shared_bailout_);
    }
    else {
      CreatePalette(*track);
    }
    // Transform.
    RunTransformKernel();
  }
  else {
    RunDebugGenerateGrayscalePalette(d_palette_buf[0], g_tracks->shared_bailout_);
    RunDebugCopyBoxBufToUnreducedBuf();
  }

  // Supersample and colorize.
  RunReduceAndColorizeKernel();
}

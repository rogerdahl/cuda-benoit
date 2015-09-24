#include "pch.h"

// App.
#include "render_ctrl.h"



using namespace std;
using namespace boost;

// Config.

double click_zoom_speed(5.0);
const u32 refresh_interval(200);

// ---------------------------------------------------------------------------
// Events.
// ---------------------------------------------------------------------------

BEGIN_EVENT_TABLE(RenderCtrl, wxControl)
EVT_PAINT(RenderCtrl::OnPaint)
EVT_SIZE(RenderCtrl::OnSize)
EVT_LEFT_DOWN(RenderCtrl::OnLeftClick)
EVT_RIGHT_DOWN(RenderCtrl::OnRightClick)
EVT_TIMER(1, RenderCtrl::OnTimer)
END_EVENT_TABLE()

DEFINE_EVENT_TYPE(wxEVT_RENDER_HASCHANGED)

// ---------------------------------------------------------------------------
// RenderCtrl
// ---------------------------------------------------------------------------

RenderCtrl::RenderCtrl(FractalSpec& fractal_spec) :
  fractal_spec_(fractal_spec), calc_(fractal_spec), width_(0), height_(0)
{
  refresh_timer_.SetOwner(this, 1);
  Init();
}

void RenderCtrl::Init() {
  calc_.Init();
  calc_.SetDim(width_ * supersample_, height_ * supersample_);
  supersample_ = 1,
  draw_only_at_end_ = false;
  hide_gflops_ = false;
  StartRefresh();
  //fractal_spec_.zoom_begin_ = calc_.GetZoom();
  //fractal_spec_.zoom_end_ = calc_.GetZoom();
  //cache_image_ = 0;
  //cache_image_zoom_ = 0.0;
}

RenderCtrl::~RenderCtrl() {
}

bool RenderCtrl::Create(wxWindow *parent, wxWindowID id, const wxPoint& pos, const wxSize& size, const wxString& name)
{
  if (!wxControl::Create(parent, id, pos, size, wxCLIP_CHILDREN | wxWANTS_CHARS, wxDefaultValidator, name)) {
    return false;
  }

  SetBackgroundColour(*wxWHITE);

  calc_.SetDim(GetClientSize().x * supersample_, GetClientSize().y * supersample_);
  StartRefresh();

  return true;
}

void RenderCtrl::StartRefresh() {
  refresh_timer_.Start(refresh_interval);
  stopwatch_gflops_.Start();
}

// OnSize gets called before OnPaint on initialization and then whenever the
// frame is resized.
void RenderCtrl::OnSize(wxSizeEvent& event) {
  width_ = GetClientSize().x;
  height_ = GetClientSize().y;
  calc_.SetDim(width_ * supersample_, height_ * supersample_);
  StartRefresh();
}

// Draw the current fractal_buf frame on screen.
void RenderCtrl::OnPaint(wxPaintEvent& event) {
  wxPaintDC dc(this);

  dc.SetTextForeground(wxColor(255, 255, 255));
  dc.SetTextBackground(wxColor(0, 0, 0));
  dc.SetBackgroundMode(0);

  if (draw_only_at_end_ && calc_.IsRunning()) {
    return;
  }

  // If there is no buffer of fractal data, we exit here.
  u8* reduced_fractal_buf(GetReducedFractalBuf());
  if (!reduced_fractal_buf) {
    dc.DrawRectangle(wxRect(0,0, width_, height_));
    dc.DrawText(wxString(L"no data"), 10, 10);
    return;
  }

  wxImage image(width_, height_);
  image.SetData(reduced_fractal_buf);
  wxBitmap bmp(image);
  dc.DrawBitmap(bmp, 0, 0, true);

  if (!hide_gflops_) {
    wxString s(wxString::Format(L"%.2f gflops", CalcGigaFlops()));
    wxSize extent(dc.GetTextExtent(s));
    dc.DrawText(s, width_ - extent.x, height_ - extent.y);
  }
}

// Calculate new frame coordinates and start calculating that frame.
void RenderCtrl::OnLeftClick(wxMouseEvent& event) {
  wxPoint point(event.GetPosition());
  wxSize s(GetClientSize());

  double cr1, cr2, ci1, ci2;
  calc_.GetTranslatedCoordinates(&cr1, &cr2, &ci1, &ci2);
  double center_r((double)point.x / (double)s.x * (cr2 - cr1) + cr1);
  double center_i((double)point.y / (double)s.y * (ci2 - ci1) + ci1);
  double zoom(calc_.GetZoom() / click_zoom_speed);

  calc_.SetCenter(center_r, center_i);
  calc_.SetZoom(zoom);
  fractal_spec_.zoom_end_ = zoom;
  StartRefresh();
  //if (zoom < zoom_end_) {
  //  zoom_end_ = zoom;
  //}

  GenerateChangedEvent();
}

// Calculate new frame coordinates and start calculating that frame.
void RenderCtrl::OnRightClick(wxMouseEvent& event) {
  double zoom(calc_.GetZoom() * click_zoom_speed);
  calc_.SetZoom(zoom);
  StartRefresh();
  fractal_spec_.zoom_end_ = zoom;
  //if (zoom > zoom_begin_) {
  //  zoom_begin_ = zoom;
  //}

  GenerateChangedEvent();
}

void RenderCtrl::OnTimer(wxTimerEvent& WXUNUSED(event)) {
  // If the calculator thread is done, we stop the timed refreshes.
  if (!calc_.IsRunning()) {
    refresh_timer_.Stop();
    stopwatch_gflops_.Pause();
  }

  Refresh(false);
}

void RenderCtrl::SetSuperSample(u32 supersample) {
  supersample_ = supersample;
  calc_.SetDim(width_ * supersample, height_ * supersample);
  StartRefresh();
}

u32 RenderCtrl::GetSuperSample() {
  return supersample_;
}

void RenderCtrl::SetHideGflops(bool hide) {
  hide_gflops_ = hide;
  Refresh(false);
}

void RenderCtrl::SetDrawOnlyAtEnd(bool draw_only_at_end) {
  draw_only_at_end_ = draw_only_at_end;
}

CalcMethods RenderCtrl::GetCalcMethod() {
  return calc_.GetCalcMethod();
}

void RenderCtrl::SetCalcMethod(CalcMethods c) {
  calc_.SetCalcMethod(c);
  StartRefresh();
}


void RenderCtrl::SetPalette(const ColorArray& ca) {
  RenderCtrl::palette = ca;
  Refresh(false);
}

ColorArray RenderCtrl::GetColorArray() {
  return palette;
}

void RenderCtrl::SetPos(double pos) {
  // The position slider is normalized to (0-1). Adjust this value to a value
  // between zoom_begin and zoom_end with movement on a "log scale".
  double zoom(pow(GetZoomEnd() / GetZoomBegin(), pos) * GetZoomBegin());
  // Ignore if position wouldn't change.
  if (zoom == calc_.GetZoom()) {
    return;
  }
  calc_.SetZoom(zoom);
  StartRefresh();
}

void RenderCtrl::SetBailout(u32 bailout) {
  calc_.SetBailout(bailout);
  StartRefresh();
}

u32 RenderCtrl::GetBailout() {
  return calc_.GetBailout();
}

void RenderCtrl::SetZoomBegin(double zoom_begin) {
  fractal_spec_.zoom_begin_ = zoom_begin;
  GenerateChangedEvent();
}

double RenderCtrl::GetZoomBegin() {
  return fractal_spec_.zoom_begin_;
}

void RenderCtrl::SetZoomEnd(double zoom_end) {
  fractal_spec_.zoom_end_ = zoom_end;
  GenerateChangedEvent();
}

double RenderCtrl::GetZoomEnd() {
  return fractal_spec_.zoom_end_;
}

void RenderCtrl::SetZoom(double zoom) {
  calc_.SetZoom(zoom);
  GenerateChangedEvent();
}

double RenderCtrl::GetZoom() {
  return calc_.GetZoom();
}

// Get fractal data that has been redecued to screen size if it was
// supersampled. Also, the result it cached so that the reduction is not run
// again if not neccessary.
//
// During profiling, it was found that a [] lookup in an STL table generates a
// huge amount of instructions while the same lookup in a native table generates
// only 3 instructions.
//
// The data is RGB data, created for use with the wxImage::SetData() call. That
// call requires that the data be malloced. After SetData(), the wxImage object
// becomes the owner of the data.
u8* RenderCtrl::GetReducedFractalBuf() {
  u32* fractal_buf_ptr(calc_.GetFractalBuf());
  if (!fractal_buf_ptr) {
    return 0;
  }
  auto_ptr<vector<u32> > reduced_fractal_buf(new vector<u32>(width_ * height_));
  u32* reduced_fractal_buf_ptr(&(*reduced_fractal_buf.get())[0]);
  
  Color* palette_ptr;
  size_t palette_size(palette.size());
  assert(palette_size);
  palette_ptr = &palette[0];

  u32 c(supersample_ * supersample_);

  u8* rgbdata((u8*)malloc(width_ * height_ * 3));

  u32 r, g, b;
  s32 x, y, xr, yr;
  u32 src_off, dst_off;
  u32 iter;

#pragma omp parallel private(x, y, xr, yr, src_off, dst_off, r, g, b, iter)
#pragma omp for schedule(static) nowait

  // Iterate over reduced pixels.
  for (y = 0; y < static_cast<s32>(height_); ++y) {
    for (x = 0; x < static_cast<s32>(width_); ++x) {
      src_off = x * supersample_ + y * supersample_ * width_ * supersample_;
      dst_off = x + y * width_;
      r = 0;
      g = 0;
      b = 0;
      // Reduce supersampled pixels for this pixel.
      for (yr = 0; yr < static_cast<s32>(supersample_); ++yr) {
        for (xr = 0; xr < static_cast<s32>(supersample_); ++xr) {
          iter = fractal_buf_ptr[src_off + xr];
          if (iter == -1) {
            iter = 0;
          }
          r += palette_ptr[iter].red_;
          g += palette_ptr[iter].green_;
          b += palette_ptr[iter].blue_;
        }
        src_off += width_ * supersample_;
      }
      r /= c;
      g /= c;
      b /= c;
      if (r > 255) { r = 255; }
      if (g > 255) { g = 255; }
      if (b > 255) { b = 255; }
      rgbdata[dst_off * 3] = (u8)r;
      rgbdata[dst_off * 3 + 1] = (u8)g;
      rgbdata[dst_off * 3 + 2] = (u8)b;
    }
  }

  return rgbdata;
}

double RenderCtrl::CalcGigaFlops() {
  u32* fractal_buf_ptr(calc_.GetFractalBuf());
  if (!fractal_buf_ptr) {
    return 0;
  }
  u32 bailout(calc_.GetBailout());

  // Calculate the total number of fractal iterations that were required for
  // calculating the fractal values we have so far.

  // Index variable in OpenMP 'for' statement must have signed integral type.
  s32 x, y, i, iter;
  u64 total(0);

//#pragma omp parallel private(x, y, i, iter) reduction(+:total)
//#pragma omp for schedule(static) nowait

  for (y = 0; y < static_cast<s32>(height_ * supersample_); ++y) {
    i = width_ * y * supersample_;
    for (x = 0; x < static_cast<s32>(width_ * supersample_); ++x) {
      iter = fractal_buf_ptr[x + i];
      // Uncalculated values are stored as -1 in buffer, so we skip those.
      if (iter >= 0) {
        total += iter ? iter : bailout;
      }
    }
  }
  double seconds(stopwatch_gflops_.Time() / 1000.0);
  double total_per_second(static_cast<double>(total) / seconds);
  // There are 14 floating point operations in the mandelbrot calculation loop.
  double flops_per_second(total_per_second * 14.0); // 14 float instructions in calc loop
  // Return gflops/second.
  return flops_per_second / 1.0e9;
  //return total;
}

vector<double> RenderCtrl::GetSlices(int n_slices) {
  u32* fractal_buf(calc_.GetFractalBuf());
  u32 c(width_ * supersample_ * height_ * supersample_);
  u32 bailout(calc_.GetBailout());

  // Histogram.
  vector<u32> histogram(bailout + 1);
  for (u32 i(0); i < c; ++i) {
    u32 iter(fractal_buf[i]);
    if (iter > bailout) {
      iter = 0;
    }
    ++(histogram[iter]);
  }

  // Slices.
  u64 slice_size(c / n_slices);
  vector<double> slices;
  u64 slice_chunk(0);
  for (u32 i(0); i < bailout; ++i) {
    slice_chunk += histogram[i];
    if (slice_chunk >= slice_size) {
      slices.push_back((double)i / (double)bailout);
      slice_chunk -= slice_size;
    }
  }

  slices.resize(n_slices);

  return slices;
}

void RenderCtrl::GenerateChangedEvent() {
	RenderEvent event_out(this, wxEVT_RENDER_HASCHANGED);
	GetEventHandler()->ProcessEvent(event_out);
}

// ---------------------------------------------------------------------------
// forward wxWin functions to subcontrols
// ---------------------------------------------------------------------------

bool RenderCtrl::Destroy() {
  return wxControl::Destroy();
}

// ---------------------------------------------------------------------------
// RenderEvent
// ---------------------------------------------------------------------------

RenderEvent::RenderEvent(RenderCtrl* pal, wxEventType type)
	: wxCommandEvent(type, pal->GetId()) {
		SetEventObject(pal);
}

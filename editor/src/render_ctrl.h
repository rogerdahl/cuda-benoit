#pragma once

#include "wx/control.h"         // the base class
#include "wx/dcclient.h"        // for wxPaintDC

// stl
#include <vector>
#include <iostream>
#include <fstream>
using namespace std;

#include "int_types.h"

#include "palette_ctrl.h"
#include "../track/track.h"
#include "calc.h"

// ---------------------------------------------------------------------------
// TemporalPaletteCtrl events
// ---------------------------------------------------------------------------

class WXDLLEXPORT RenderCtrl;

class WXDLLEXPORT RenderEvent : public wxCommandEvent {
	friend class RenderCtrl;
public:
	RenderEvent() { }
	RenderEvent(RenderCtrl*, wxEventType type);
protected:
private:
};

// ---------------------------------------------------------------------------
// RenderCtrl
// ---------------------------------------------------------------------------

#define g_render_ctrl_string L"RenderCtrl"

// --------------------------------------------------------------
// RenderCtrl: a control to render a colorized fractal
// --------------------------------------------------------------

class WXDLLEXPORT RenderCtrl : public wxControl {
  FractalSpec& fractal_spec_;
  Calc calc_;

	ColorArray palette;

  u16* frame;
	u32 movie_w, movie_h;
	u32 movie_size;
  wxImage* cache_image_;
  double cache_image_zoom_;

  struct trackitem { double cr1; double cr2; double ci1; double ci2 ; };
  vector<trackitem> track;

  wxTimer refresh_timer_;
  wxStopWatch stopwatch_gflops_;

  // We support calculating more than one point per pixel.
  // 1 = 1 point per pixel. (turns off supersampling).
  // 2 = 4 points per pixel.
  // 3 = 9 points per pixel.
  // 4 = 16 points per pixel.
  // etc.
  u32 supersample_;

  // Resolution not including supersampling.
  u32 width_;
  u32 height_;

  bool hide_gflops_;
  bool draw_only_at_end_;

  // Events.
  void OnPaint(wxPaintEvent& event);
  void OnSize(wxSizeEvent& event);
  void OnLeftClick(wxMouseEvent& event);
  void OnRightClick(wxMouseEvent& event);
  void OnTimer(wxTimerEvent& event);
  // 	void OnEraseBackground(wxEraseEvent& event);

  void StartRefresh();
  void GenerateChangedEvent();

  DECLARE_EVENT_TABLE()

public:
  RenderCtrl(FractalSpec&);
	virtual ~RenderCtrl();
	void Init();

  bool Create(wxWindow *parent,
		wxWindowID id,
		const wxPoint& pos = wxDefaultPosition,
		const wxSize& size = wxDefaultSize,
		const wxString& name = g_render_ctrl_string);

	virtual bool Destroy();

  //

	void SetPos(double pos);
	double GetPos();

  void SetBailout(u32 bailout);
  u32 GetBailout();

  double GetZoomBegin();
  void SetZoomBegin(double);

  double GetZoomEnd();
  void SetZoomEnd(double);

  double GetZoom();
  void SetZoom(double);

  void SetPalette(const ColorArray&);
	ColorArray GetColorArray();

  void SetSuperSample(u32 i);
  u32 GetSuperSample();
  void SetHideGflops(bool hide);
  void SetDrawOnlyAtEnd(bool);
  CalcMethods GetCalcMethod();
  void SetCalcMethod(CalcMethods);
  u8* GetReducedFractalBuf();
  double CalcGigaFlops();
  std::vector<double> GetSlices(int n);
};

// ---------------------------------------------------------------------------
// palette control event types and macros for handling them
// ---------------------------------------------------------------------------

BEGIN_DECLARE_EVENT_TYPES()
DECLARE_EVENT_TYPE(wxEVT_RENDER_HASCHANGED, 954)
END_DECLARE_EVENT_TYPES()

typedef void (wxEvtHandler::*RenderEventFunction)(RenderEvent&);

#define EVT_RENDER_HASCHANGED(id, fn) DECLARE_EVENT_TABLE_ENTRY(wxEVT_RENDER_HASCHANGED, id, -1, (wxObjectEventFunction) (wxEventFunction) (wxCommandEventFunction) (RenderEventFunction) & fn, (wxObject *) NULL),

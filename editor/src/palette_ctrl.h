#include "pch.h"

#pragma once

#include "../track/track.h"
#include "int_types.h"

// ---------------------------------------------------------------------------
// TemporalPaletteCtrl events.
// ---------------------------------------------------------------------------

class WXDLLEXPORT TemporalPaletteCtrl;

class WXDLLEXPORT PaletteEvent : public wxCommandEvent {
	friend class TemporalPaletteCtrl;
public:
	PaletteEvent() { }
	PaletteEvent(TemporalPaletteCtrl*, wxEventType type);
protected:
private:
};

// ---------------------------------------------------------------------------
// TemporalPaletteCtrl
// ---------------------------------------------------------------------------

#define temporal_palette_ctrl_name L"TemporalPaletteCtrl"

class WXDLLEXPORT TemporalPaletteCtrl : public wxControl {
public:
	// Construction.
	TemporalPaletteCtrl(TemporalPalette& temporal_palette);
	virtual ~TemporalPaletteCtrl();
	void Init();

  bool Create(wxWindow *parent,
		wxWindowID id,
		const wxPoint& pos = wxDefaultPosition,
		const wxSize& size = wxDefaultSize,
		const wxString& name = temporal_palette_ctrl_name);

  ColorArray GetColorArray(const Pos pos, const u32 num_colors);
	void DuplicateSpatialKey();
	void RandomPalette(const u32 num_colors, const std::vector<double>& positions);
  Pos GetFocusedTemporalKeyPos();
  void SetFocusedTemporalKeyPos(const Pos pos);
  bool CheckTemporalKeyLimit();
  bool CheckSpatialKeyLimit(const SpatialKeys& spatial_keys);
	// Slider calls this when pos changes (poor man's event).
	void OnRGBSliderPosChange(wxScrollEvent& event);
  // Called by main after data in TemporalPalette is updated.
  void TemporalPaletteUpdated();

	virtual bool Destroy();

private:
	// Event handlers.
	void OnPaint(wxPaintEvent& event);
	void OnLeftDown(wxMouseEvent& event);
	void OnLeftUp(wxMouseEvent& event);
	void OnMotion(wxMouseEvent& event);
	void OnClick(wxMouseEvent& event);
	void OnDoubleClick(wxMouseEvent& event);
	void OnRightDown(wxMouseEvent& event);
	void OnSize(wxSizeEvent& event);
	void OnEraseBackground(wxEraseEvent& event);

	void RenderSlider(wxDC& dc, bool focus, const wxRect& rect, const Pos& pos, const Color& c);
	void UpdateRGBSliders();
	bool HitTestSlider(const wxRect& rect, const Pos pos, const wxPoint& check);

	void GenerateChangedEvent();

	// Subcontrols.
	wxSlider* slider_red_;
	wxSlider* slider_green_;
	wxSlider* slider_blue_;

	wxTextCtrl* text_red_;
	wxTextCtrl* text_green_;
	wxTextCtrl* text_blue_;

	// Palette document.
  TemporalPalette& temporal_palette_;

	// Dragging.
	SpatialKeys::iterator spatial_key_dragging_;
	TemporalKeys::iterator temporal_key_dragging_;
	u32 slider_dragging_hold_;
	bool spatial_slider_is_dragging_;
	bool temporal_slider_is_dragging_;

	// Sizer.
	wxBoxSizer* top_sizer_;

  // Mersenne twister rng.
  typedef boost::mt19937 BaseGenerator;
  BaseGenerator rng;

  double Random();
  Color GetRandomColor();

	//DECLARE_DYNAMIC_CLASS(TemporalPaletteCtrl)
	DECLARE_EVENT_TABLE()
};

// ---------------------------------------------------------------------------
// TemporalPaletteCtrl event types and macros for handling them.
// ---------------------------------------------------------------------------

BEGIN_DECLARE_EVENT_TYPES()
DECLARE_EVENT_TYPE(wxEVT_PALETTE_HASCHANGED, 953)
END_DECLARE_EVENT_TYPES()

typedef void (wxEvtHandler::*PaletteEventFunction)(PaletteEvent&);

#define EVT_PALETTE_HASCHANGED(id, fn) DECLARE_EVENT_TABLE_ENTRY(wxEVT_PALETTE_HASCHANGED, id, -1, (wxObjectEventFunction) (wxEventFunction) (wxCommandEventFunction) (PaletteEventFunction) & fn, (wxObject *) NULL),

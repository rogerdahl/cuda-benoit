#include "pch.h"

// App.
#include "palette_ctrl.h"
#include "utils.h"

using namespace std;
using namespace boost;

// Config.

// The height of different parts of the control the values are given in top to
// bottom order.
const u32 SLIDER_SQUARE_PERCENT(70);
const u32 SLIDER_PERCENT(30);
const u32 PALETTE_SPACE(2);
const u32 SLIDER_WIDTH(15);

const u32 MAX_SPATIAL_KEYS(12);
const u32 MAX_TEMPORAL_KEYS(12);

// ---------------------------------------------------------------------------
// Private classes.
// ---------------------------------------------------------------------------

// RGB Slider.

class RGBSlider : public wxSlider
{
  public:
  RGBSlider(
      TemporalPaletteCtrl* temporal_palette_ctrl, wxPoint pt, wxSize size,
      const wxString& name);
  // Event handlers.
  void OnRGBSliderPosChange(wxScrollEvent& event)
  {
    pal_->OnRGBSliderPosChange(event);
  }

  private:
  TemporalPaletteCtrl* pal_;
  DECLARE_EVENT_TABLE()
};

RGBSlider::RGBSlider(
    TemporalPaletteCtrl* pal, wxPoint pt, wxSize size, const wxString& name)
  : wxSlider(
        pal, -1, 0, 0, 255, pt, size, wxSL_HORIZONTAL | wxCLIP_CHILDREN,
        wxDefaultValidator, name)
{
  pal_ = pal;
  SetBackgroundColour(*wxWHITE);
}

// ---------------------------------------------------------------------------
// Events.
// ---------------------------------------------------------------------------

BEGIN_EVENT_TABLE(TemporalPaletteCtrl, wxControl)
EVT_PAINT(TemporalPaletteCtrl::OnPaint)
EVT_LEFT_DOWN(TemporalPaletteCtrl::OnLeftDown)
EVT_LEFT_UP(TemporalPaletteCtrl::OnLeftUp)
EVT_MOTION(TemporalPaletteCtrl::OnMotion)
EVT_LEFT_DCLICK(TemporalPaletteCtrl::OnDoubleClick)
EVT_RIGHT_DOWN(TemporalPaletteCtrl::OnRightDown)
EVT_SIZE(TemporalPaletteCtrl::OnSize)
END_EVENT_TABLE()

BEGIN_EVENT_TABLE(RGBSlider, wxSlider)
EVT_SCROLL_THUMBTRACK(RGBSlider::OnRGBSliderPosChange)
END_EVENT_TABLE()

DEFINE_EVENT_TYPE(wxEVT_PALETTE_HASCHANGED)

// ---------------------------------------------------------------------------
// TemporalPaletteCtrl
// ---------------------------------------------------------------------------

TemporalPaletteCtrl::TemporalPaletteCtrl(TemporalPalette& temporal_palette)
  : temporal_palette_(temporal_palette),
    spatial_slider_is_dragging_(false),
    temporal_slider_is_dragging_(false)
{
  rng.seed(static_cast<unsigned int>(time(0)));
}

TemporalPaletteCtrl::~TemporalPaletteCtrl()
{
}

void TemporalPaletteCtrl::Init()
{
  slider_red_->SetValue(0);
  slider_green_->SetValue(0);
  slider_blue_->SetValue(0);
  Refresh(false);
}

double TemporalPaletteCtrl::Random()
{
  // Define a uniform random number distribution which produces double precision
  // values between 0 and 1 (0 inclusive, 1 exclusive).
  boost::uniform_real<> uni_dist(0, 1);
  boost::variate_generator<BaseGenerator&, boost::uniform_real<> > uni(
      rng, uni_dist);
  return uni();
}

Color TemporalPaletteCtrl::GetRandomColor()
{
  // 256 should be right because the Random() range excludes 1.0.
  return Color(Random() * 256.0, Random() * 256.0, Random() * 256.0);
}

bool TemporalPaletteCtrl::Create(
    wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size,
    const wxString& name)
{
  // Create control.
  if (!wxControl::Create(
          parent, id, pos, size, wxNO_BORDER, wxDefaultValidator, name)) {
    return false;
  }
  // Create controls and put them in sizer.
  wxBoxSizer* r_sizer(new wxBoxSizer(wxHORIZONTAL));
  slider_red_ =
      new RGBSlider(this, wxDefaultPosition, wxDefaultSize, L"slider_red_");
  text_red_ = new wxTextCtrl(this, -1, L"", wxDefaultPosition, wxSize(40, -1));
  text_red_->Enable(false);
  r_sizer->Add(new wxStaticText(this, -1, L"R"), 0, wxLEFT | wxRIGHT, 5);
  r_sizer->Add(slider_red_, 1, wxEXPAND, 5);
  r_sizer->Add(text_red_, 0, wxLEFT | wxRIGHT, 5);
  // G
  wxBoxSizer* g_sizer(new wxBoxSizer(wxHORIZONTAL));
  slider_green_ =
      new RGBSlider(this, wxDefaultPosition, wxDefaultSize, L"slider_green_");
  text_green_ =
      new wxTextCtrl(this, -1, L"", wxDefaultPosition, wxSize(40, -1));
  text_green_->Enable(false);
  g_sizer->Add(new wxStaticText(this, -1, L"G"), 0, wxLEFT | wxRIGHT, 5);
  g_sizer->Add(slider_green_, 1, wxEXPAND, 5);
  g_sizer->Add(text_green_, 0, wxLEFT | wxRIGHT, 5);
  // B
  wxBoxSizer* b_sizer(new wxBoxSizer(wxHORIZONTAL));
  slider_blue_ =
      new RGBSlider(this, wxDefaultPosition, wxDefaultSize, L"slider_blue_");
  text_blue_ = new wxTextCtrl(this, -1, L"", wxDefaultPosition, wxSize(40, -1));
  text_blue_->Enable(false);
  b_sizer->Add(new wxStaticText(this, -1, L"B"), 0, wxLEFT | wxRIGHT, 5);
  b_sizer->Add(slider_blue_, 1, wxEXPAND, 5);
  b_sizer->Add(text_blue_, 0, wxLEFT | wxRIGHT, 5);
  // Slider rows.
  top_sizer_ = new wxBoxSizer(wxVERTICAL);
  top_sizer_->Add(r_sizer, 0, wxEXPAND | wxTOP | wxBOTTOM, 2);
  top_sizer_->Add(g_sizer, 0, wxEXPAND | wxTOP | wxBOTTOM, 2);
  top_sizer_->Add(b_sizer, 0, wxEXPAND | wxTOP | wxBOTTOM, 2);
  // Add space for the custom drawn part of this control, the two palette with
  // slider areas.
  top_sizer_->Add(0, 200);
  // Use the sizer for layout.
  SetSizer(top_sizer_);
  // Tell the sizer to set (and Fit) the minimal size of the window to match the
  // sizer's minimal size.
  top_sizer_->SetSizeHints(this);

  SetBackgroundColour(*wxWHITE);

  return true;
}

// Render slider.
void TemporalPaletteCtrl::RenderSlider(
    wxDC& dc, bool focus, const wxRect& rect, const Pos& pos,
    const Color& color)
{
  // Set up points for the slider shape:
  // x   x
  // x   x
  //   x
  const u32 slider_square_height(
      (int)((double)rect.GetHeight() * ((double)SLIDER_SQUARE_PERCENT / 100)));
  wxPoint points[5];
  points[0].x = SLIDER_WIDTH;
  points[0].y = 0;
  points[1].x = SLIDER_WIDTH;
  points[1].y = slider_square_height;
  points[2].x = SLIDER_WIDTH / 2;
  points[2].y = rect.GetHeight();
  points[3].x = 0;
  points[3].y = slider_square_height;
  points[4].x = 0;
  points[4].y = 0;
  // Calc x of slider.
  u32 x((pos / 1.0f) * (rect.GetWidth() - SLIDER_WIDTH));
  // Draw slider body.
  dc.SetBrush(wxBrush(wxColor(color.red_, color.green_, color.blue_), wxSOLID));
  dc.SetPen(*wxTRANSPARENT_PEN);
  dc.DrawPolygon(5, points, x + rect.GetX(), rect.GetY(), wxWINDING_RULE);
  // Draw slider borders.
  dc.SetPen(wxPen(wxColor(0, 0, 0), 1, focus ? wxDOT : wxSOLID));
  dc.DrawLines(3, points, x + rect.GetX(), rect.GetY());
  dc.SetPen(wxPen(wxColor(200, 200, 200), 1, focus ? wxDOT : wxSOLID));
  dc.DrawLines(3, points + 2, x + rect.GetX(), rect.GetY());
}

// Check if "check" is within the slider (also hit on sides of the point).
bool TemporalPaletteCtrl::HitTestSlider(
    const wxRect& rect, const Pos pos, const wxPoint& check)
{
  s32 x((pos / 1.0f) * (rect.GetWidth() - SLIDER_WIDTH));
  return wxRect(x, rect.y, SLIDER_WIDTH, rect.GetHeight()).Contains(check);
}

// ---------------------------------------------------------------------------
// Event handlers.
// ---------------------------------------------------------------------------

void TemporalPaletteCtrl::UpdateRGBSliders()
{
  Color color(temporal_palette_.GetTopColor());
  u32 R(color.red_);
  u32 G(color.green_);
  u32 B(color.blue_);
  // Update sliders.
  slider_red_->SetValue(R);
  slider_green_->SetValue(G);
  slider_blue_->SetValue(B);
  // Update text boxes.
  text_red_->SetValue(wxString(lexical_cast<wstring>(R).c_str()));
  text_green_->SetValue(wxString(lexical_cast<wstring>(G).c_str()));
  text_blue_->SetValue(wxString(lexical_cast<wstring>(B).c_str()));
}

void TemporalPaletteCtrl::OnRGBSliderPosChange(wxScrollEvent& event)
{
  u32 R(slider_red_->GetValue());
  u32 G(slider_green_->GetValue());
  u32 B(slider_blue_->GetValue());
  // Update text boxes.
  text_red_->SetValue(wxString(lexical_cast<wstring>(R).c_str()));
  text_green_->SetValue(wxString(lexical_cast<wstring>(G).c_str()));
  text_blue_->SetValue(wxString(lexical_cast<wstring>(B).c_str()));
  // Update color of slider that has focus.
  temporal_palette_.SetTopColor(Color(R, G, B));
  Refresh(false);
  GenerateChangedEvent();
}

void TemporalPaletteCtrl::OnPaint(wxPaintEvent& event)
{
  wxPaintDC dc_paint(this);

  // To avoid flicker, we draw to a memory dc that we then blit to screen.
  wxBitmap bitmap_tmp(
      GetClientSize().GetWidth(), GetClientSize().GetHeight(), -1);
  wxMemoryDC dc;
  dc.SelectObject(bitmap_tmp);

  // Clear to white.
  dc.SetBrush(wxBrush(wxColor(255, 255, 255), wxSOLID));
  dc.Clear();

  // Positions.
  int slider_blue_x, slider_blue_y, slider_blue_width, slider_blue_height;
  slider_blue_->GetPosition(&slider_blue_x, &slider_blue_y);
  slider_blue_->GetSize(&slider_blue_width, &slider_blue_height);
  u32 palette_start(slider_blue_y + slider_blue_height + PALETTE_SPACE);
  u32 total_height(GetClientSize().GetHeight() - palette_start - PALETTE_SPACE);
  u32 palette_height((total_height - PALETTE_SPACE) / 2);
  u32 slider_height(
      (int)((double)palette_height * ((double)SLIDER_PERCENT / 100)));
  u32 strip_height(palette_height - slider_height - PALETTE_SPACE);
  u32 width(GetClientSize().GetWidth());
  u32 palette_slider_start(palette_start);
  u32 palette_strip_start(palette_slider_start + slider_height + PALETTE_SPACE);
  u32 temporal_slider_start(palette_strip_start + strip_height + PALETTE_SPACE);
  u32 temporal_strip_start(
      temporal_slider_start + slider_height + PALETTE_SPACE);

  // Testing for position code.
  // dc.SetBrush(*wxTRANSPARENT_BRUSH);
  // dc.SetPen(wxPen(wxColor(255,0,0), 1, wxSOLID));
  // dc.DrawRectangle(0, palette_slider_start, width, slider_height);
  // dc.SetPen(wxPen(wxColor(0,255,0), 1, wxSOLID));
  // dc.DrawRectangle(0, palette_strip_start, width, strip_height);
  // dc.SetPen(wxPen(wxColor(0,0,255), 1, wxSOLID));
  // dc.DrawRectangle(0, temporal_slider_start, width, slider_height);
  // dc.SetPen(wxPen(wxColor(0,255,255), 1, wxSOLID));
  // dc.DrawRectangle(0, temporal_strip_start, width, strip_height);

  // Render palette sliders.

  // Render sliders in opposite order so that sliders at top of list appear in
  // front last slider is rendered with focus.
  u32 count(0);
  SpatialKeys& spatial_keys(
      temporal_palette_.GetTopTemporalKey().GetSpatialKeys());
  for (SpatialKeys::reverse_iterator riter(spatial_keys.rbegin());
       riter != spatial_keys.rend(); ++riter) {
    RenderSlider(
        dc, ++count == spatial_keys.size(),
        wxRect(0, palette_slider_start, width, slider_height), riter->GetPos(),
        riter->GetColor());
  }

  // Render palette strip.
  u32 x1(SLIDER_WIDTH / 2);
  u32 x2(width - SLIDER_WIDTH / 2);
  ColorArray color_array(
      temporal_palette_.GetColorArray(
          temporal_palette_.GetTopTemporalKey().GetSpatialKeys(), x2 - x1));
  count = 0;
  for (u32 x(x1); x < x2; ++x) {
    Color color(color_array[count++]);
    dc.SetPen(
        wxPen(wxColor(color.red_, color.green_, color.blue_), 1, wxSOLID));
    dc.DrawLine(x, palette_strip_start, x, palette_strip_start + strip_height);
  }

  // Render temporal key sliders.
  //
  // Render sliders in opposite order so that sliders at top of list appear in
  // front last slider is rendered with focus.
  count = 0;
  TemporalKeys& temporal_keys(temporal_palette_.GetTemporalKeys());
  for (TemporalKeys::reverse_iterator riter(temporal_keys.rbegin());
       riter != temporal_keys.rend(); ++riter) {
    RenderSlider(
        dc, ++count == temporal_keys.size(),
        wxRect(0, temporal_slider_start, width, slider_height), riter->GetPos(),
        Color(128, 128, 128));
  }

  // Render temporal strip.
  TemporalKeys sorted_temporal_keys(temporal_keys);
  sorted_temporal_keys.sort(TemporalPosLessThan());
  // Add "virtual" sliders at ends.
  if (sorted_temporal_keys.front().GetPos() > 0.0) {
    TemporalKey temporal_key(0.0, sorted_temporal_keys.front().GetSpatialKeys());
    sorted_temporal_keys.push_front(temporal_key);
  }
  if (sorted_temporal_keys.back().GetPos() < 1.0) {
    TemporalKey temporal_key(1.0, sorted_temporal_keys.back().GetSpatialKeys());
    sorted_temporal_keys.push_back(temporal_key);
  }
  // Alloc data for wxImage.
  unsigned char* rgbdata((unsigned char*)malloc(width * strip_height * 3));
  memset(rgbdata, 255, width * strip_height * 3);
  // Draw to rgbdata.
  for (TemporalKeys::iterator iter = sorted_temporal_keys.begin();; ++iter) {
    TemporalKeys::iterator next(iter);
    ++next;
    // Exit loop if there is no "next".
    if (next == sorted_temporal_keys.end())
      break;
    // Get color arrays for the current pair.
    ColorArray arr1(
        temporal_palette_.GetColorArray(iter->GetSpatialKeys(), strip_height));
    ColorArray arr2(
        temporal_palette_.GetColorArray(next->GetSpatialKeys(), strip_height));
    // Get positions for current pair.
    u32 x1(
        iter->GetPos() * (GetClientSize().x - SLIDER_WIDTH)
        + (SLIDER_WIDTH / 2));
    u32 x2(
        next->GetPos() * (GetClientSize().x - SLIDER_WIDTH)
        + (SLIDER_WIDTH / 2));
    //
    for (u32 x(x1); x < x2; ++x) {
      double delta((double)(x - x1) / (x2 - x1));
      for (u32 y(0); y < strip_height; ++y) {
        Color color(Lerp(arr1[y], arr2[y], delta));
        u32 off(x * 3 + width * 3 * y);
        rgbdata[off + 0] = color.red_;
        rgbdata[off + 1] = color.green_;
        rgbdata[off + 2] = color.blue_;
      }
    }
  }
  // Copy rgbdata to screen.
  wxImage image(width, strip_height);
  // After SetData(), rgbdata is owned by image.
  image.SetData(rgbdata);
  wxBitmap bmp(image);
  dc.DrawBitmap(bmp, 0, temporal_strip_start, true);
  // Copy to screen.
  dc_paint.Blit(
      0, palette_start, GetClientSize().GetWidth(),
      GetClientSize().GetHeight() - palette_start, &dc, 0, palette_start);
}

void TemporalPaletteCtrl::OnEraseBackground(wxEraseEvent& event)
{
}

void TemporalPaletteCtrl::OnDoubleClick(wxMouseEvent& event)
{
}

void TemporalPaletteCtrl::OnClick(wxMouseEvent& event)
{
}

void TemporalPaletteCtrl::OnLeftDown(wxMouseEvent& event)
{
  // Positions.
  int slider_blue_x, slider_blue_y, slider_blue_width, slider_blue_height;
  slider_blue_->GetPosition(&slider_blue_x, &slider_blue_y);
  slider_blue_->GetSize(&slider_blue_width, &slider_blue_height);
  u32 palette_start(slider_blue_y + slider_blue_height + PALETTE_SPACE);
  u32 total_height(GetClientSize().GetHeight() - palette_start - PALETTE_SPACE);
  u32 palette_height((total_height - PALETTE_SPACE) / 2);
  u32 slider_height(
      (int)((double)palette_height * ((double)SLIDER_PERCENT / 100)));
  u32 strip_height(palette_height - slider_height - PALETTE_SPACE);
  u32 width(GetClientSize().GetWidth());
  u32 palette_slider_start(palette_start);
  u32 palette_strip_start(palette_slider_start + slider_height + PALETTE_SPACE);
  u32 temporal_slider_start(palette_strip_start + strip_height + PALETTE_SPACE);

  // If click was within slider_area.
  wxRect slider_area(0, palette_slider_start, width, slider_height);
  if (slider_area.Contains(event.GetX(), event.GetY())) {
    // Check if we hit any of the sliders.
    SpatialKeys& spatial_keys(
        temporal_palette_.GetTopTemporalKey().GetSpatialKeys());
    for (SpatialKeys::iterator iter(spatial_keys.begin());
         iter != spatial_keys.end(); ++iter) {
      if (HitTestSlider(
              slider_area, iter->GetPos(),
              wxPoint(event.GetX(), event.GetY()))) {
        // Move slider to top of list, so it is drawn in front, then start
        // dragging of the slider by setting slider_dragging.
        SpatialKey spatial_key(*iter);
        spatial_keys.erase(iter);
        spatial_keys.push_front(spatial_key);
        spatial_key_dragging_ = spatial_keys.begin();
        spatial_slider_is_dragging_ = true;
        // Record where on the slider we're holding.
        u32 x((u32)((width - SLIDER_WIDTH) * spatial_key_dragging_->GetPos()));
        slider_dragging_hold_ = event.GetX() - x;
        // Direct all mouse events to this window.
        CaptureMouse();
        //
        TemporalPaletteUpdated();
        //
        break;
      }
    }
    // If an existing slider was not hit, create new spatial key. The color of
    // the new key is set to the interpolated color at the given location.
    if (!spatial_slider_is_dragging_) {
      if (!CheckSpatialKeyLimit(spatial_keys)) {
        return;
      }
      SpatialKey spatial_key;
      spatial_key.SetPos(
          (double)event.GetX() / (GetClientSize().x - SLIDER_WIDTH));
      spatial_key.SetColor(temporal_palette_.GetColor(spatial_key.GetPos()));
      spatial_keys.push_front(spatial_key);
      TemporalPaletteUpdated();
    }
  }

  // If click was within temporal slider area.
  wxRect temporal_slider_area(0, temporal_slider_start, width, slider_height);
  if (temporal_slider_area.Contains(event.GetX(), event.GetY())) {
    TemporalKeys& temporal_keys(temporal_palette_.GetTemporalKeys());
    for (TemporalKeys::iterator iter(temporal_keys.begin());
         iter != temporal_keys.end(); ++iter) {
      // If click was on an existing slider.
      if (HitTestSlider(
              temporal_slider_area, iter->GetPos(),
              wxPoint(event.GetX(), event.GetY()))) {
        // Move temporal_slider to top of list, so it is drawn in front, then
        // start
        // dragging of the temporal_slider by setting temporal_slider_dragging.
        TemporalKey temporal_key(*iter);
        temporal_keys.erase(iter);
        temporal_keys.push_front(temporal_key);
        temporal_key_dragging_ = temporal_keys.begin();
        temporal_slider_is_dragging_ = true;
        // Record where on the temporal_slider we're holding.
        u32 x((u32)((width - SLIDER_WIDTH) * temporal_key_dragging_->GetPos()));
        slider_dragging_hold_ = event.GetX() - x;
        // Direct all mouse events to this window.
        CaptureMouse();
        TemporalPaletteUpdated();
        break;
      }
    }
    // If click was not on an existing slider.
    if (!temporal_slider_is_dragging_) {
      if (!CheckTemporalKeyLimit()) {
        return;
      }
      // Create new temporal key that is a copy of the closest key.
      //   Find the click position.
      Pos click_pos((double)event.GetX() / (GetClientSize().x - SLIDER_WIDTH));
      //   Find the closest key.
      TemporalKeys::iterator closest_key(temporal_keys.begin());
      double smallest_delta(1.0);
      for (TemporalKeys::iterator iter(temporal_keys.begin());
           iter != temporal_keys.end(); ++iter) {
        double delta(fabs(iter->GetPos() - click_pos));
        if (delta < smallest_delta) {
          smallest_delta = delta;
          closest_key = iter;
        }
      }
      //  Create the duplicate.
      temporal_palette_.CreateTemporalKey(
          click_pos, closest_key->GetSpatialKeys());
      TemporalPaletteUpdated();
    }
  }
}

void TemporalPaletteCtrl::OnMotion(wxMouseEvent& event)
{
  if (spatial_slider_is_dragging_) {
    // Move slider being dragged.
    spatial_key_dragging_->SetPos(
        (double)(event.GetX() - (int)slider_dragging_hold_)
        / (double)(GetClientSize().x - SLIDER_WIDTH));
    TemporalPaletteUpdated();
  }
  if (temporal_slider_is_dragging_) {
    // Move slider being dragged.
    temporal_key_dragging_->SetPos(
        (double)(event.GetX() - (int)slider_dragging_hold_)
        / (double)(GetClientSize().x - SLIDER_WIDTH));
    TemporalPaletteUpdated();
  }
}

void TemporalPaletteCtrl::OnLeftUp(wxMouseEvent& event)
{
  if (spatial_slider_is_dragging_) {
    spatial_slider_is_dragging_ = false;
    ReleaseMouse();
    GenerateChangedEvent();
  }
  if (temporal_slider_is_dragging_) {
    temporal_slider_is_dragging_ = false;
    ReleaseMouse();
    GenerateChangedEvent();
  }
}

void TemporalPaletteCtrl::OnRightDown(wxMouseEvent& event)
{
  int slider_blue_x, slider_blue_y, slider_blue_width, slider_blue_height;
  slider_blue_->GetPosition(&slider_blue_x, &slider_blue_y);
  slider_blue_->GetSize(&slider_blue_width, &slider_blue_height);

  u32 palette_start(slider_blue_y + slider_blue_height + PALETTE_SPACE);
  u32 total_height(GetClientSize().GetHeight() - palette_start - PALETTE_SPACE);
  u32 palette_height((total_height - PALETTE_SPACE) / 2);
  u32 slider_height(
      (int)((double)palette_height * ((double)SLIDER_PERCENT / 100)));
  u32 strip_height(palette_height - slider_height - PALETTE_SPACE);
  u32 width(GetClientSize().GetWidth());

  u32 palette_slider_start(palette_start);
  u32 palette_strip_start(palette_slider_start + slider_height + PALETTE_SPACE);
  u32 temporal_slider_start(palette_strip_start + strip_height + PALETTE_SPACE);

  // Hit test with top slider first.
  //
  // If click was within slider_area.
  wxRect slider_area(0, palette_slider_start, width, slider_height);
  if (slider_area.Contains(event.GetX(), event.GetY())) {
    SpatialKeys& spatial_keys(
        temporal_palette_.GetTopTemporalKey().GetSpatialKeys());
    for (SpatialKeys::iterator iter(spatial_keys.begin());
         iter != spatial_keys.end(); ++iter) {
      if (HitTestSlider(
              wxRect(
                  0, palette_slider_start, width,
                  palette_slider_start + slider_height),
              iter->GetPos(), wxPoint(event.GetX(), event.GetY()))) {
        // Ignore if trying to erase last SpatialKey.
        if (spatial_keys.size() <= 1)
          return;
        // Erase SpatialKey.
        spatial_keys.erase(iter);
        // Update rgb sliders to the one that now has focus.
        TemporalPaletteUpdated();
        //
        break;
      }
    }
  }

  // If click was within temporal_slider area.
  wxRect temporal_slider_area(0, temporal_slider_start, width, slider_height);
  if (temporal_slider_area.Contains(event.GetX(), event.GetY())) {
    TemporalKeys& temporal_keys(temporal_palette_.GetTemporalKeys());
    for (TemporalKeys::iterator iter(temporal_keys.begin());
         iter != temporal_keys.end(); ++iter) {
      if (HitTestSlider(
              temporal_slider_area, iter->GetPos(),
              wxPoint(event.GetX(), event.GetY()))) {
        // Ignore if trying to erase last TemporalKey.
        if (temporal_keys.size() <= 1)
          return;
        // Erase TemporalKey.
        temporal_keys.erase(iter);
        // Update rgb sliders to the one that now has focus.
        TemporalPaletteUpdated();
        //
        break;
      }
    }
  }

  Refresh(false);
}

// Tried removing OnSize and the explicit Layout() call because it's not
// supposed to be necessary. When a sizer is set, SetAutoLayout should get
// called automatically and cause wx to call Layout() internally in its OnSize
// handler. But when I remove OnSize, the sizer seems to collapse to a minimal
// size where the sliders are on top of each other.
void TemporalPaletteCtrl::OnSize(wxSizeEvent& event)
{
  // Update the layout (sizers).
  Layout();
  Refresh(false);
}

void TemporalPaletteCtrl::TemporalPaletteUpdated()
{
  Refresh(false);
  UpdateRGBSliders();
  GenerateChangedEvent();
}

ColorArray TemporalPaletteCtrl::GetColorArray(
    const Pos pos, const u32 num_colors)
{
  return temporal_palette_.GetColorArray(pos, num_colors);
}

void TemporalPaletteCtrl::DuplicateSpatialKey()
{
  if (!CheckTemporalKeyLimit()) {
    return;
  }
  SpatialKey spatial_key(temporal_palette_.GetTopSpatialKey());
  Pos pos(spatial_key.GetPos());
  if (pos >= 0.95) {
    pos -= 0.02;
  }
  else {
    pos += 0.02;
  }
  spatial_key.SetPos(pos);

  TemporalKey& temporal_key(temporal_palette_.GetTopTemporalKey());
  SpatialKeys& spatial_keys(temporal_key.GetSpatialKeys());
  spatial_keys.push_back(spatial_key);
  TemporalPaletteUpdated();
}

void TemporalPaletteCtrl::RandomPalette(
    u32 num_colors, const vector<double>& positions)
{
  // Remove all SpatialKeys in the temporal_keysette items that has the focus.
  SpatialKeys& spatial_keys(
      temporal_palette_.GetTopTemporalKey().GetSpatialKeys());
  spatial_keys.clear();

  // Generate new random colors.
  for (u32 i(0); i < num_colors; ++i) {
    SpatialKey spatial_key;
    spatial_key.SetPos(positions[i]);
    spatial_key.SetColor(GetRandomColor());
    spatial_keys.push_front(spatial_key);
  }
  TemporalPaletteUpdated();
}

Pos TemporalPaletteCtrl::GetFocusedTemporalKeyPos()
{
  return temporal_palette_.GetTopTemporalKey().GetPos();
}

void TemporalPaletteCtrl::SetFocusedTemporalKeyPos(const Pos pos)
{
  temporal_palette_.GetTopTemporalKey().SetPos(pos);
  Refresh(false);
}

bool TemporalPaletteCtrl::CheckTemporalKeyLimit()
{
  if (temporal_palette_.GetTemporalKeys().size() == MAX_TEMPORAL_KEYS) {
    wxMessageBox(
        str(wformat(
                L"For performance reasons, the player is limited to %d "
                L"temporal keys per track")
            % MAX_TEMPORAL_KEYS),
        L"Limit reached", wxCentre, this);
    return false;
  }
  return true;
}

bool TemporalPaletteCtrl::CheckSpatialKeyLimit(const SpatialKeys& spatial_keys)
{
  if (spatial_keys.size() == MAX_SPATIAL_KEYS) {
    wxMessageBox(
        str(wformat(
                L"For performance reasons, the player is limited to %d "
                L"spatial keys per temporal key")
            % MAX_SPATIAL_KEYS),
        L"Limit reached", wxCentre, this);
    return false;
  }
  return true;
}

void TemporalPaletteCtrl::GenerateChangedEvent()
{
  PaletteEvent event_out(this, wxEVT_PALETTE_HASCHANGED);
  GetEventHandler()->ProcessEvent(event_out);
}

// ---------------------------------------------------------------------------
// Forward wxWin functions to subcontrols.
// ---------------------------------------------------------------------------

bool TemporalPaletteCtrl::Destroy()
{
  return wxControl::Destroy();
}

// ---------------------------------------------------------------------------
// PaletteEvent
// ---------------------------------------------------------------------------

PaletteEvent::PaletteEvent(
    TemporalPaletteCtrl* temporal_palette_ctrl, wxEventType type)
  : wxCommandEvent(type, temporal_palette_ctrl->GetId())
{
  SetEventObject(temporal_palette_ctrl);
}

// Benoit Track Editor. 2011. dahlsys.com
// http://www.dahlsys.com/software/benoit/index.html
// License: GPL.
//
// This project loosely follows the Google style guide:
// https://google-styleguide.googlecode.com/svn/trunk/cppguide.xml
//
// Note on optimization: It's usually ok to return a big object by value because
// of return value optimization.
// https://secure.wikimedia.org/wikipedia/en/wiki/Return_value_optimization

#include "pch.h"

#include "../track/track.h"
#include "bailout_dlg.h"
#include "palette_ctrl.h"
#include "render_ctrl.h"

using namespace std;
using namespace boost;
using namespace boost::filesystem;

// Config.

wxString g_app_name(L"Benoit Track Editor");

// ---------------------------------------------------------------------------
// Main frame declaration.
// ---------------------------------------------------------------------------

class MyFrame : public wxFrame
{
  public:
  MyFrame(const wxString& title);

  // Event handlers (these functions should _not_ be virtual).
  void OnNew(wxCommandEvent& event);
  void OnOpen(wxCommandEvent& event);
  void OnSave(wxCommandEvent& event);
  void OnSaveAs(wxCommandEvent& event);
  void OnQuit(wxCommandEvent& event);
  void OnClose(wxCloseEvent& event);

  // Edit menu.
  void OnDuplicateSpatialKey(wxCommandEvent& event);

  // Calculation menu.
  void OnSetBailout(wxCommandEvent& event);
  void OnCalcMethodx86Float(wxCommandEvent& event);
  void OnCalcMethodx86Double(wxCommandEvent& event);
  void OnCalcMethodSSEDouble(wxCommandEvent& event);
  void OnCalcMethodSSEFloat(wxCommandEvent& event);
  void OnCalcMethodCUDAFloat(wxCommandEvent& event);
  void OnCalcMethodCUDADouble(wxCommandEvent& event);

  // Supersample menu.
  void OnSuperSample1(wxCommandEvent& event);
  void OnSuperSample2(wxCommandEvent& event);
  void OnSuperSample3(wxCommandEvent& event);
  void OnSuperSample4(wxCommandEvent& event);
  void OnSuperSample5(wxCommandEvent& event);

  // Misc menu.
  void OnDrawOnlyAtEnd(wxCommandEvent& event);
  void OnMarkCurrentAsZoomStart(wxCommandEvent& event);
  void OnHideGflops(wxCommandEvent& event);

  // Help menu.
  void OnAbout(wxCommandEvent& event);

  // Events.
  void OnSize(wxSizeEvent& event);
  void OnRandomPalette(wxCommandEvent& event);
  void OnFractalPosChange(wxScrollEvent& event);
  // Events from TemporalPaletteCtrl.
  void OnPaletteHasChanged(PaletteEvent& event);
  // Events from RenderCtrl.
  void OnRenderHasChanged(RenderEvent& event);
  // Events from BailoutDialog.
  void OnBailoutHasChanged(BailoutEvent& event);

  private:
  Track track_;
  RenderCtrl* render_ctrl_;
  TemporalPaletteCtrl* temporal_palette_ctrl_;
  BailoutDialog* bailout_dialog_;
  wxSlider* zoom_pos_slider_;
  wxTextCtrl* zoom_pos_text_zoom_begin_;
  wxTextCtrl* zoom_pos_text_zoom_end_;
  wxSizer* top_sizer_;
  wxMenu* calculation_menu;
  wxMenu* supersample_menu;
  wxMenu* misc_menu;
  bool unsaved_changes_;
  wpath track_path_;

  void Refresh();
  void SetTrackStatus(bool unsaved_changes, wpath track_path);
  void UnsavedChanges();
  void UpdateZoomValues();

  DECLARE_EVENT_TABLE()
};

// IDs for the controls and the menu commands.
enum
{
  // File.
  kMinimalQuit = 1,
  kNew,
  kOpen,
  kSave,
  kSaveAs,

  // Edit
  kDuplicateSpatialKey,

  // Supersample.
  kSuperSample1,
  kSuperSample2,
  kSuperSample3,
  kSuperSample4,
  kSuperSample5,

  // Calculation.
  kSetBailout,
  kCalcMethodx86Float,
  kCalcMethodx86Double,
  kCalcMethodSSEFloat,
  kCalcMethodSSEDouble,
  kCalcMethodCUDAFloat,
  kCalcMethodCUDADouble,

  // Random.
  kRandomPalette1,
  kRandomPalette2,
  kRandomPalette3,
  kRandomPalette4,
  kRandomPalette5,
  kRandomPalette6,
  kRandomPalette7,
  kRandomPalette8,
  kRandomPalette9,

  // Misc.
  kHideGflops,
  kDrawOnlyAtEnd,
  kDrawAlways,
  kMarkCurrentAsZoomStart,

  kMinimalAbout = wxID_ABOUT
};

// ---------------------------------------------------------------------------
// Events.
// ---------------------------------------------------------------------------

BEGIN_EVENT_TABLE(MyFrame, wxFrame)

EVT_MENU(kMinimalQuit, MyFrame::OnQuit)
EVT_MENU(kMinimalAbout, MyFrame::OnAbout)
EVT_MENU(kNew, MyFrame::OnNew)
EVT_MENU(kOpen, MyFrame::OnOpen)
EVT_MENU(kSave, MyFrame::OnSave)
EVT_MENU(kSaveAs, MyFrame::OnSaveAs)

EVT_MENU(kDuplicateSpatialKey, MyFrame::OnDuplicateSpatialKey)

EVT_MENU(kSetBailout, MyFrame::OnSetBailout)
EVT_MENU(kCalcMethodx86Float, MyFrame::OnCalcMethodx86Float)
EVT_MENU(kCalcMethodx86Double, MyFrame::OnCalcMethodx86Double)
EVT_MENU(kCalcMethodSSEFloat, MyFrame::OnCalcMethodSSEFloat)
EVT_MENU(kCalcMethodSSEDouble, MyFrame::OnCalcMethodSSEDouble)
EVT_MENU(kCalcMethodCUDAFloat, MyFrame::OnCalcMethodCUDAFloat)
EVT_MENU(kCalcMethodCUDADouble, MyFrame::OnCalcMethodCUDADouble)

EVT_MENU(kSuperSample1, MyFrame::OnSuperSample1)
EVT_MENU(kSuperSample2, MyFrame::OnSuperSample2)
EVT_MENU(kSuperSample3, MyFrame::OnSuperSample3)
EVT_MENU(kSuperSample4, MyFrame::OnSuperSample4)
EVT_MENU(kSuperSample5, MyFrame::OnSuperSample5)

EVT_MENU(kRandomPalette1, MyFrame::OnRandomPalette)
EVT_MENU(kRandomPalette2, MyFrame::OnRandomPalette)
EVT_MENU(kRandomPalette3, MyFrame::OnRandomPalette)
EVT_MENU(kRandomPalette4, MyFrame::OnRandomPalette)
EVT_MENU(kRandomPalette5, MyFrame::OnRandomPalette)
EVT_MENU(kRandomPalette6, MyFrame::OnRandomPalette)
EVT_MENU(kRandomPalette7, MyFrame::OnRandomPalette)
EVT_MENU(kRandomPalette8, MyFrame::OnRandomPalette)
EVT_MENU(kRandomPalette9, MyFrame::OnRandomPalette)

EVT_MENU(kHideGflops, MyFrame::OnHideGflops)
EVT_MENU(kDrawOnlyAtEnd, MyFrame::OnDrawOnlyAtEnd)
EVT_MENU(kMarkCurrentAsZoomStart, MyFrame::OnMarkCurrentAsZoomStart)

EVT_SIZE(MyFrame::OnSize)
EVT_PALETTE_HASCHANGED(-1, MyFrame::OnPaletteHasChanged)
EVT_RENDER_HASCHANGED(-1, MyFrame::OnRenderHasChanged)
EVT_BAILOUT_HASCHANGED(-1, MyFrame::OnBailoutHasChanged)
// Slider events are handled in the same way as a scrollbar. These events are
// from the fractal position slider.
EVT_SCROLL_CHANGED(MyFrame::OnFractalPosChange)
EVT_CLOSE(MyFrame::OnClose)
END_EVENT_TABLE()

// ---------------------------------------------------------------------------
// Main application.
// ---------------------------------------------------------------------------

class MyApp : public wxApp
{
  MyFrame* frame_;
  DECLARE_EVENT_TABLE()
  public:
  virtual bool OnInit();
};

BEGIN_EVENT_TABLE(MyApp, wxApp)
END_EVENT_TABLE()

// The program execution "starts" here.
bool MyApp::OnInit()
{
  frame_ = new MyFrame(g_app_name);
  frame_->Show(TRUE);
  return TRUE;
  cout << "here" << endl;
}

// void MyApp::on_char(wxKeyEvent& event) {
//  if (event.GetKeyCode() >= '1' && event.GetKeyCode() <= '9') {
//    frame_->RandomPalette(event.GetKeyCode() - '0');
//  }
//}

IMPLEMENT_APP(MyApp)

// ---------------------------------------------------------------------------
// Main frame implementation.
// ---------------------------------------------------------------------------

MyFrame::MyFrame(const wxString& title)
  : wxFrame(0, -1, title), unsaved_changes_(false)
{
  // Set the frame icon.
  /////////////////////////////////SetIcon(wxICON(mondrian));

  // Create a menu bar.
  wxMenu* file_menu(new wxMenu);
  file_menu->Append(kNew, L"&New\tCtrl-N", L"New track");
  file_menu->Append(kOpen, L"&Open Track\tCtrl-O", L"Open a track");
  file_menu->Append(kSave, L"S&ave Track\tCtrl-S", L"Save the track");
  file_menu->Append(
      kSaveAs, L"S&ave Track As\tCtrl-A", L"Save the track with a new name");
  file_menu->Append(kMinimalQuit, L"E&xit\tAlt-X", L"Quit this program");

  wxMenu* edit_menu(new wxMenu);
  edit_menu->Append(
      kDuplicateSpatialKey, L"&Duplicate Spatial Key\tCtrl-R", L"");

  calculation_menu = new wxMenu;
  calculation_menu->Append(kSetBailout, L"Set bailout value");
  calculation_menu->AppendRadioItem(kCalcMethodx86Double, L"x86 double");
  calculation_menu->AppendRadioItem(kCalcMethodx86Float, L"x86 float");
  calculation_menu->AppendRadioItem(kCalcMethodSSEDouble, L"SSE 2x double");
  calculation_menu->AppendRadioItem(kCalcMethodSSEFloat, L"SSE 4x float");
  calculation_menu->AppendRadioItem(kCalcMethodCUDADouble, L"CUDA double");
  calculation_menu->AppendRadioItem(kCalcMethodCUDAFloat, L"CUDA float");
  calculation_menu->Check(kCalcMethodx86Float, true);

  supersample_menu = new wxMenu;
  supersample_menu->AppendRadioItem(kSuperSample1, L"1x");
  supersample_menu->AppendRadioItem(kSuperSample2, L"4x");
  supersample_menu->AppendRadioItem(kSuperSample3, L"9x");
  supersample_menu->AppendRadioItem(kSuperSample4, L"16x");
  supersample_menu->AppendRadioItem(kSuperSample5, L"25x");

  wxMenu* random_palette_menu(new wxMenu);
  random_palette_menu->Append(kRandomPalette1, L"1 color\t1");
  random_palette_menu->Append(kRandomPalette2, L"2 colors\t2");
  random_palette_menu->Append(kRandomPalette3, L"3 colors\t3");
  random_palette_menu->Append(kRandomPalette4, L"4 colors\t4");
  random_palette_menu->Append(kRandomPalette5, L"5 colors\t5");
  random_palette_menu->Append(kRandomPalette6, L"6 colors\t6");
  random_palette_menu->Append(kRandomPalette7, L"7 colors\t7");
  random_palette_menu->Append(kRandomPalette8, L"8 colors\t8");
  random_palette_menu->Append(kRandomPalette9, L"9 colors\t9");

  misc_menu = new wxMenu;
  misc_menu->AppendCheckItem(kHideGflops, L"Hide gflops display");
  misc_menu->AppendCheckItem(kDrawOnlyAtEnd, L"Draw only at end");
  misc_menu->Append(kMarkCurrentAsZoomStart, L"Mark current as zoom start");

  wxMenu* help_menu(new wxMenu);
  help_menu->Append(kMinimalAbout, L"&About...\tF1", L"Show about dialog");

  // Append the menu to the menu bar.
  wxMenuBar* menu_bar(new wxMenuBar);
  menu_bar->Append(file_menu, L"&File");
  menu_bar->Append(edit_menu, L"&Edit");
  menu_bar->Append(calculation_menu, L"&Calculation");
  menu_bar->Append(supersample_menu, L"&Supersample");
  menu_bar->Append(random_palette_menu, L"&Random Palette");
  menu_bar->Append(misc_menu, L"&Misc");
  menu_bar->Append(help_menu, L"&Help");

  // Attach menu bar to the frame.
  SetMenuBar(menu_bar);

  // Create a vertical sizer.
  top_sizer_ = new wxBoxSizer(wxVERTICAL);

  render_ctrl_ = new RenderCtrl(track_.GetFractalSpec());
  render_ctrl_->Create(this, -1, wxDefaultPosition, wxSize(640, 480));
  // wxFIXED_MINSIZE is neccessary to not have the RenderCtrl collapse to a size
  // that is much smaller than the default size set in the Create() call. I
  // don't know why. wxFIXED_MINSIZE prevents the user from resizing the window
  // to a point where control becomes smaller than the default size.
  top_sizer_->Add(render_ctrl_, 1, wxEXPAND | wxFIXED_MINSIZE);

  // Render position slider and text box.
  wxBoxSizer* pos_sizer(new wxBoxSizer(wxHORIZONTAL));
  // For the pos slider, we set the max value to the highest signed 16 bit
  // value. That's high enough that we get a good resolution on the slider
  // regardless of the size of the window and hopefully low enough that all
  // slider implementations support it.
  zoom_pos_slider_ = new wxSlider(
      this, -1, 0, 0, 32767, wxDefaultPosition, wxDefaultSize,
      wxSL_HORIZONTAL | wxCLIP_CHILDREN);
  zoom_pos_slider_->SetBackgroundColour(*wxWHITE);
  zoom_pos_text_zoom_begin_ =
      new wxTextCtrl(this, -1, L"", wxDefaultPosition, wxSize(80, -1));
  zoom_pos_text_zoom_end_ =
      new wxTextCtrl(this, -1, L"", wxDefaultPosition, wxSize(80, -1));
  pos_sizer->Add(zoom_pos_text_zoom_begin_, 0, wxLEFT | wxRIGHT, 5);
  pos_sizer->Add(zoom_pos_slider_, 1, wxEXPAND, 5);
  pos_sizer->Add(zoom_pos_text_zoom_end_, 0, wxLEFT | wxRIGHT, 5);
  top_sizer_->Add(pos_sizer, 0, wxEXPAND | wxTOP | wxBOTTOM, 2);

  // TemporalPaletteCtrl
  temporal_palette_ctrl_ = new TemporalPaletteCtrl(track_.GetTemporalPalette());
  temporal_palette_ctrl_->Create(
      this, -1, wxPoint(0, 0), wxDefaultSize, L"TemporalPaletteCtrl");
  top_sizer_->Add(temporal_palette_ctrl_, 0, wxEXPAND | wxALL, 10);

  // Set the initial palette for RenderCtrl.
  render_ctrl_->SetPalette(
      temporal_palette_ctrl_->GetColorArray(0.0, render_ctrl_->GetBailout()));

  // Use the sizer for layout.
  SetSizer(top_sizer_);
  // Tell the sizer to set (and fit) the minimal size of the window to match the
  // sizer's minimal size.
  top_sizer_->SetSizeHints(this);

  SetBackgroundColour(*wxWHITE);

  UpdateZoomValues();

  // Create the bailout dialog but keep it hidden.
  bailout_dialog_ =
      new BailoutDialog(this, -1, wxDefaultPosition, wxDefaultSize);

  // Set default supersample and calculation method. Default bailout is stored
  // in track.
  wxCommandEvent event;
  OnSuperSample3(event);

  // Check if CUDA capable device is present.
  int cuda_device_count;
  cudaError_t res(cudaGetDeviceCount(&cuda_device_count));
  if (res != cudaSuccess || !cuda_device_count) {
    // No CUDA capable device was found, so we disable CUDA calculation options.
    calculation_menu->Enable(kCalcMethodCUDAFloat, false);
    calculation_menu->Enable(kCalcMethodCUDADouble, false);
    // Set initial calculation method to x86 double.
    OnCalcMethodx86Double(event);
  }
  else {
    // Found a CUDA capable device. Set the default calculation to using CUDA
    // doubles. Note, on a machine with a fast CPU and a slow GPU, this may be
    // slower than than using the CPU.
    OnCalcMethodCUDADouble(event);
  }

  // Set track name and changed status in title.
  SetTrackStatus(false, L"");

  // If the app has been associated with the .benoit file type, the path appears
  // as the first argument when the app is opened by double clicking a track
  // file.
  if (wxTheApp->argc > 1) {
    track_.Load(wxTheApp->argv[1]);
    SetTrackStatus(false, wxTheApp->argv[1]);
    Refresh();
  }
  //// For profiling, do automatic stuff here.
  // if(wxTheApp->argc > 1) {
  //  OnSuperSample5(wxCommandEvent());
  //  OnCalcMethodCUDAFloat(wxCommandEvent());
  //  Maximize();
  //}
}

// Event handlers.

void MyFrame::OnQuit(wxCommandEvent& WXUNUSED(event))
{
  // TRUE is to force the frame to close.
  Close(TRUE);
}

void MyFrame::OnClose(wxCloseEvent& event)
{
  UnsavedChanges();
  Destroy();
}

void MyFrame::OnSetBailout(wxCommandEvent& event)
{
  bailout_dialog_->SetBailout(render_ctrl_->GetBailout());
  bailout_dialog_->CenterOnParent();
  bailout_dialog_->Show();
}

void MyFrame::OnCalcMethodx86Float(wxCommandEvent& event)
{
  render_ctrl_->SetCalcMethod(kCalcx86Float);
  calculation_menu->Check(kCalcMethodx86Float, true);
}
void MyFrame::OnCalcMethodx86Double(wxCommandEvent& event)
{
  render_ctrl_->SetCalcMethod(kCalcx86Double);
  calculation_menu->Check(kCalcMethodx86Double, true);
}
void MyFrame::OnCalcMethodSSEFloat(wxCommandEvent& event)
{
  render_ctrl_->SetCalcMethod(kCalcSSE4Float);
  calculation_menu->Check(kCalcMethodSSEFloat, true);
}
void MyFrame::OnCalcMethodSSEDouble(wxCommandEvent& event)
{
  render_ctrl_->SetCalcMethod(kCalcSSE2Double);
  calculation_menu->Check(kCalcMethodSSEDouble, true);
}
void MyFrame::OnCalcMethodCUDAFloat(wxCommandEvent& event)
{
  render_ctrl_->SetCalcMethod(kCalcCUDAFloat);
  calculation_menu->Check(kCalcMethodCUDAFloat, true);
}
void MyFrame::OnCalcMethodCUDADouble(wxCommandEvent& event)
{
  render_ctrl_->SetCalcMethod(kCalcCUDADouble);
  calculation_menu->Check(kCalcMethodCUDADouble, true);
}

void MyFrame::OnSuperSample1(wxCommandEvent& event)
{
  render_ctrl_->SetSuperSample(1);
  supersample_menu->Check(kSuperSample1, true);
}
void MyFrame::OnSuperSample2(wxCommandEvent& event)
{
  render_ctrl_->SetSuperSample(2);
  supersample_menu->Check(kSuperSample2, true);
}
void MyFrame::OnSuperSample3(wxCommandEvent& event)
{
  render_ctrl_->SetSuperSample(3);
  supersample_menu->Check(kSuperSample3, true);
}
void MyFrame::OnSuperSample4(wxCommandEvent& event)
{
  render_ctrl_->SetSuperSample(4);
  supersample_menu->Check(kSuperSample4, true);
}
void MyFrame::OnSuperSample5(wxCommandEvent& event)
{
  render_ctrl_->SetSuperSample(5);
  supersample_menu->Check(kSuperSample5, true);
}

void MyFrame::OnHideGflops(wxCommandEvent& event)
{
  render_ctrl_->SetHideGflops(misc_menu->IsChecked(kHideGflops));
}

void MyFrame::OnDrawOnlyAtEnd(wxCommandEvent& event)
{
  render_ctrl_->SetDrawOnlyAtEnd(misc_menu->IsChecked(kDrawOnlyAtEnd));
}

void MyFrame::OnMarkCurrentAsZoomStart(wxCommandEvent& event)
{
  render_ctrl_->SetZoomBegin(render_ctrl_->GetZoom());
}

void MyFrame::OnSize(wxSizeEvent& event)
{
  event.Skip();
}

void MyFrame::OnNew(wxCommandEvent& event)
{
  UnsavedChanges();
  track_.Init();
  SetTrackStatus(false, L"");
  Refresh();
}

void MyFrame::OnOpen(wxCommandEvent& event)
{
  UnsavedChanges();
  wxFileDialog d(
      this, L"Open track", L"", L"track.benoit", L"*.benoit", wxFD_OPEN);
  if (d.ShowModal() != wxID_OK) {
    return;
  }
  track_.Load(d.GetPath().c_str());
  SetTrackStatus(false, d.GetPath().c_str());
  Refresh();
}

void MyFrame::OnSave(wxCommandEvent& event)
{
  if (!unsaved_changes_) {
    return;
  }
  if (track_path_.empty()) {
    return OnSaveAs(event);
  }
  track_.Save(track_path_.c_str());
  SetTrackStatus(false, track_path_.c_str());
}

void MyFrame::OnSaveAs(wxCommandEvent& event)
{
  wxFileDialog d(
      this, L"Save track", L"", L"track.benoit", L"*.benoit", wxFD_SAVE);
  if (d.ShowModal() != wxID_OK) {
    return;
  }
  track_.Save(d.GetPath().c_str());
  SetTrackStatus(false, d.GetPath().c_str());
}

// Edit menu.

void MyFrame::OnDuplicateSpatialKey(wxCommandEvent& event)
{
  temporal_palette_ctrl_->DuplicateSpatialKey();
}

//

void MyFrame::OnRandomPalette(wxCommandEvent& event)
{
  int num_colors(event.GetId() - kRandomPalette1 + 1);
  vector<double> positions = render_ctrl_->GetSlices(num_colors);
  temporal_palette_ctrl_->RandomPalette(num_colors, positions);
}

void MyFrame::OnAbout(wxCommandEvent& WXUNUSED(event))
{
  wxMessageBox(
      g_app_name + L" - dahlsys.com", L"About",
      wxOK | wxICON_INFORMATION | wxCENTER, this);
}

void MyFrame::OnFractalPosChange(wxScrollEvent& event)
{
  // Get position as a normalized double.
  u32 max_pos(zoom_pos_slider_->GetMax());
  u32 pos(zoom_pos_slider_->GetValue());
  double pos_norm(static_cast<double>(pos) / static_cast<double>(max_pos));
  // Get palette for new position and update the RenderCtrl palette.
  render_ctrl_->SetPalette(
      temporal_palette_ctrl_->GetColorArray(
          pos_norm, render_ctrl_->GetBailout()));
  // Update position.
  render_ctrl_->SetPos(pos_norm);
}

void MyFrame::SetTrackStatus(bool unsaved_changes, wpath track_path)
{
  unsaved_changes_ = unsaved_changes;
  track_path_ = track_path;
  //// Update the window title with the track status.
  // SetTitle(wxString::Format("%s %s - %s",
  //  (unsaved_changes_ ? "*" : ""),
  //  (track_path_.empty() ? "new" : track_path_.stem().c_str()),
  //  g_app_name));
  SetTitle(wxString(L"fixme"));
}

void MyFrame::UnsavedChanges()
{
  if (unsaved_changes_) {
    if (wxMessageBox(L"Save changes?", L"Unsaved changes", wxYES_NO, this)
        == wxYES) {
      wxCommandEvent event;
      OnSave(event);
    }
  }
}

void MyFrame::UpdateZoomValues()
{
  // Update text boxes.
  zoom_pos_text_zoom_begin_->SetValue(
      wxString(str(wformat(L"%.4e") % render_ctrl_->GetZoomBegin()).c_str()));
  zoom_pos_text_zoom_end_->SetValue(
      wxString(str(wformat(L"%.4e") % render_ctrl_->GetZoomEnd()).c_str()));
  zoom_pos_slider_->SetValue(zoom_pos_slider_->GetMax());
  temporal_palette_ctrl_->SetFocusedTemporalKeyPos(1.0f);
}

void MyFrame::Refresh()
{
  // RenderCtrl.Init() resets a few values that we want to preserve, so we
  // read those out and write then back after the Init().
  u32 supersample(render_ctrl_->GetSuperSample());
  CalcMethods calc_method(render_ctrl_->GetCalcMethod());
  render_ctrl_->Init();
  render_ctrl_->SetSuperSample(supersample);
  render_ctrl_->SetCalcMethod(calc_method);
  wxCommandEvent event;
  OnHideGflops(event);
  OnDrawOnlyAtEnd(event);
  // Reset the TemporalPaletteCtrl.
  temporal_palette_ctrl_->Init();
  render_ctrl_->SetPalette(
      temporal_palette_ctrl_->GetColorArray(0.0, render_ctrl_->GetBailout()));
  UpdateZoomValues();
}

// ---------------------------------------------------------------------------
// Handlers for events from TemporalPaletteCtrl.
// ---------------------------------------------------------------------------

void MyFrame::OnPaletteHasChanged(PaletteEvent& event)
{
  // Get position as a normalized double.
  double max_pos(static_cast<double>(zoom_pos_slider_->GetMax()));
  double pos(static_cast<double>(zoom_pos_slider_->GetValue()));
  double pos_norm(pos / max_pos);
  // Get palette for new position and update the RenderCtrl palette.
  render_ctrl_->SetPalette(
      temporal_palette_ctrl_->GetColorArray(
          pos_norm, render_ctrl_->GetBailout()));
  zoom_pos_slider_->SetValue(
      temporal_palette_ctrl_->GetFocusedTemporalKeyPos() * max_pos);
  wxScrollEvent event2;
  OnFractalPosChange(event2);
  SetTrackStatus(true, track_path_);
}

// ---------------------------------------------------------------------------
// Handlers for events from RenderCtrl.
// ---------------------------------------------------------------------------

void MyFrame::OnRenderHasChanged(RenderEvent& event)
{
  UpdateZoomValues();
  SetTrackStatus(true, track_path_);
}

// ---------------------------------------------------------------------------
// Handlers for events from BailoutDialog.
// ---------------------------------------------------------------------------

void MyFrame::OnBailoutHasChanged(BailoutEvent& event)
{
  // Get position as a normalized double.
  u32 max_pos(zoom_pos_slider_->GetMax());
  u32 pos(zoom_pos_slider_->GetValue());
  double pos_norm(static_cast<double>(pos) / static_cast<double>(max_pos));
  // Get palette for new position and update the RenderCtrl palette.
  render_ctrl_->SetBailout(event.bailout);
  render_ctrl_->SetPalette(
      temporal_palette_ctrl_->GetColorArray(pos_norm, event.bailout));
  // Update track status.
  SetTrackStatus(true, track_path_);
}

#include "pch.h"

#include "bailout_dlg.h"

// IDs for the controls.
enum
{
  kSlider,
  kText,
};

// ---------------------------------------------------------------------------
// Events.
// ---------------------------------------------------------------------------

BEGIN_EVENT_TABLE(BailoutDialog, wxDialog)
EVT_COMMAND_SCROLL(kSlider, BailoutDialog::OnSlider)
EVT_TEXT(kText, BailoutDialog::OnText)
EVT_BUTTON(wxID_OK, BailoutDialog::OnOk)
EVT_BUTTON(wxID_CANCEL, BailoutDialog::OnCancel)
END_EVENT_TABLE()

DEFINE_EVENT_TYPE(wxEVT_BAILOUT_HASCHANGED)

// ---------------------------------------------------------------------------
// BailoutEvent
// ---------------------------------------------------------------------------

BailoutEvent::BailoutEvent(BailoutDialog* bailout_dialog, wxEventType type)
  : wxCommandEvent(type, bailout_dialog->GetId())
{
  SetEventObject(bailout_dialog);
}

// ---------------------------------------------------------------------------
// BailoutDialog.
// ---------------------------------------------------------------------------

BailoutDialog::BailoutDialog(
    wxWindow* parent, wxWindowID id, const wxPoint& position,
    const wxSize& size)
  : wxDialog(parent, id, L"Bailout", position, size)
{
  wxBoxSizer* top_sizer(new wxBoxSizer(wxHORIZONTAL));

  // Warning icon.
  wxBitmap warning_bitmap(wxArtProvider::GetBitmap(wxART_WARNING));
  wxStaticBitmap* warning_icon(new wxStaticBitmap(this, -1, warning_bitmap));
  top_sizer->Add(warning_icon, 0, wxTOP | wxLEFT, 20);

  wxBoxSizer* warning_sizer(new wxBoxSizer(wxVERTICAL));

  // Warning.
  wxString warning_msg;
  warning_msg +=
      L"A higher bailout value enhances detail in the fractal. However,\n";
  warning_msg +=
      L"a higher bailout values also increases the fractal calculation\n";
  warning_msg +=
      L"time and makes it less likely that the player can rendered the\n";
  warning_msg += L"track in realtime.\n";
  warning_msg += L"\n";
  warning_msg +=
      L"All tracks that are to be played at the same time must use the\n";
  warning_msg +=
      L"same bailout value. If the bailout values do not match between\n";
  warning_msg +=
      L"tracks in a set, the player will select the lowest value and\n";
  warning_msg += L"use it for all tracks.";
  warning_sizer->Add(new wxStaticText(this, -1, warning_msg), 0, wxALL, 10);

  // Bailout setting.
  wxBoxSizer* bailout_sizer(new wxBoxSizer(wxHORIZONTAL));
  //   Slider.
  bailout_slider_ = new wxSlider(
      this, kSlider, 0, 1, 10000, wxDefaultPosition, wxSize(200, -1),
      wxSL_HORIZONTAL | wxCLIP_CHILDREN);
  bailout_sizer->Add(bailout_slider_, 1, wxEXPAND | wxRIGHT, 5);
  //   Text edit.
  bailout_text_ =
      new wxTextCtrl(this, kText, L"", wxDefaultPosition, wxSize(80, -1));
  bailout_sizer->Add(bailout_text_, 0, wxLEFT, 0);
  warning_sizer->Add(bailout_sizer, 0, wxALIGN_LEFT | wxTOP | wxBOTTOM, 10);

  // Ok and Cancel.
  wxBoxSizer* button_sizer(new wxBoxSizer(wxHORIZONTAL));
  wxButton* ok_button(new wxButton(this, wxID_OK, L"&Ok"));
  button_sizer->Add(ok_button, 0, wxLEFT | wxRIGHT, 5);
  wxButton* cancel_button(new wxButton(this, wxID_CANCEL, L"&Cancel"));
  button_sizer->Add(cancel_button, 0, wxLEFT | wxRIGHT, 5);
  warning_sizer->Add(button_sizer, 0, wxALIGN_RIGHT | wxALL, 10);

  top_sizer->Add(warning_sizer, 0, wxALL, 10);
  SetSizer(top_sizer);
  top_sizer->SetSizeHints(this);
}

void BailoutDialog::SetBailout(u32 bailout)
{
  bailout_ = bailout;
  bailout_slider_->SetValue(bailout);
  bailout_text_->SetValue(wxString::Format(L"%d", bailout));
}

void BailoutDialog::OnSlider(wxScrollEvent& event)
{
  u32 bailout(event.GetInt());
  bailout_text_->SetValue(wxString::Format(L"%d", bailout));
  GenerateChangedEvent(bailout);
}

void BailoutDialog::OnText(wxCommandEvent& event)
{
  long bailout;
  if (bailout_text_->GetValue().ToLong(&bailout)) {
    if (bailout > bailout_slider_->GetMax()) {
      bailout = bailout_slider_->GetMax();
      bailout_text_->SetValue(wxString::Format(L"%d", bailout));
    }
    if (bailout < bailout_slider_->GetMin()) {
      bailout = bailout_slider_->GetMin();
      bailout_text_->SetValue(wxString::Format(L"%d", bailout));
    }
    bailout_slider_->SetValue(bailout);
    GenerateChangedEvent(bailout);
  }
}

void BailoutDialog::OnOk(wxCommandEvent& event)
{
  Hide();
}

void BailoutDialog::OnCancel(wxCommandEvent& event)
{
  if (bailout_slider_->GetValue() != bailout_) {
    GenerateChangedEvent(bailout_);
  }
  Hide();
}

void BailoutDialog::GenerateChangedEvent(u32 bailout)
{
  BailoutEvent event_out(this, wxEVT_BAILOUT_HASCHANGED);
  // Couldn't get this event to propagate to the parent, so I'm sending it
  // directly there.
  event_out.bailout = bailout;
  GetParent()->GetEventHandler()->ProcessEvent(event_out);
}

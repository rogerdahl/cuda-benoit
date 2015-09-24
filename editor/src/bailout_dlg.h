#include "int_types.h"

// ---------------------------------------------------------------------------
// BailoutDialog events.
// ---------------------------------------------------------------------------

class WXDLLEXPORT BailoutDialog;

class WXDLLEXPORT BailoutEvent : public wxCommandEvent {
	friend class BailoutDialog;
public:
	BailoutEvent() { }
	BailoutEvent(BailoutDialog*, wxEventType type);
  u32 bailout;
protected:
private:
};

// ---------------------------------------------------------------------------
// BailoutDialog
// ---------------------------------------------------------------------------

class BailoutDialog: public wxDialog
{
public:
  BailoutDialog(wxWindow* parent, wxWindowID id,
                const wxPoint & pos = wxDefaultPosition,
                const wxSize & size = wxDefaultSize);
  void SetBailout(u32 bailout);
private:
	void OnSlider(wxScrollEvent& event);
	void OnText(wxCommandEvent& event);
  void OnOk(wxCommandEvent& event);
  void OnCancel(wxCommandEvent& event);

  void GenerateChangedEvent(u32 bailout);

  wxTextCtrl* bailout_text_;
  wxSlider* bailout_slider_;
  u32 bailout_;

  DECLARE_EVENT_TABLE()
};

// ---------------------------------------------------------------------------
// BailoutDialog event types and macros for handling them.
// ---------------------------------------------------------------------------

BEGIN_DECLARE_EVENT_TYPES()
DECLARE_EVENT_TYPE(wxEVT_BAILOUT_HASCHANGED, 955)
END_DECLARE_EVENT_TYPES()

typedef void (wxEvtHandler::*BailoutEventFunction)(BailoutEvent&);

#define EVT_BAILOUT_HASCHANGED(id, fn) DECLARE_EVENT_TABLE_ENTRY(wxEVT_BAILOUT_HASCHANGED, id, -1, (wxObjectEventFunction) (wxEventFunction) (wxCommandEventFunction) (BailoutEventFunction) & fn, (wxObject *) NULL),

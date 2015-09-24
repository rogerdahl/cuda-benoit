#pragma once

#include "pch.h"
#include "platform.h"

int GetDisplayRefreshRate() {
  //DEVMODE dm;
  //ZeroMemory(&dm, sizeof(dm));
  //dm.dmSize = sizeof(dm);
  //if (0 != EnumDisplaySettings(NULL, ENUM_CURRENT_SETTINGS, &dm)) {
  //  return dm.dmDisplayFrequency;
  //}
  // If EnumDisplaySettings() fails for some reason, default to 60 Hz.
  return 60;
}

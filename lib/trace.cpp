#include "pch.h"

#include "trace.h"

// Todo: This trace() implementation does not work.
void _trace(LPCTSTR lpctFormat, ...) {
  static TCHAR szBuffer[1000];
  va_list ArgList;
  va_start(ArgList, lpctFormat);
  _vstprintf_s(szBuffer, lpctFormat, ArgList);
  va_end(ArgList);
  OutputDebugString(szBuffer);
}

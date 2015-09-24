// Todo: This trace() implementation does not work.
#include <windows.h>
#ifdef _DEBUG
void _trace(LPCTSTR lpctFormat, ...);
#define trace(msg) (_trace(msg))
#else
#define trace(msg)
#endif

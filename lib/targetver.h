#pragma once

// Disable warnings about unsafe functions.
#define _CRT_SECURE_NO_DEPRECATE 1
#define _SCL_SECURE_NO_DEPRECATE 1

// Show names of boost libraries automatically selected for inclusion.
//#define BOOST_LIB_DIAGNOSTIC

// This will cause (JIT) debugging to be invoked at the point of the crash,
// instead of translating it into a C++ exception. It might be useful to inspect
// *this at the point of the crash (if the crash does in fact happen at the same
// place you're seeing the exception above). 
//#include <boost/python/test/module_tail.cpp>

// Minimum required platform is Windows 2000
#ifndef WINVER                          
#define WINVER 0x0500
#endif
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0500
#endif
#ifndef NTDDI_VERSION
#define NTDDI_VERSION NTDDI_WIN2K
#endif

// Minimum required browser is Internet Explorer 5.0. (I don't think we care,
// but we set it to tie up potential loose ends).
#ifndef _WIN32_IE                       
#define _WIN32_IE 0x0500
#endif

// Libraries end up including Windows headers. This removes rarely-used stuff
// from Windows headers.
#define WIN32_LEAN_AND_MEAN

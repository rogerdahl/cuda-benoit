#pragma message("Compiling PCH - Should only happen once per project")

//#define _CRT_SECURE_NO_WARNINGS

// xmm support for visual studio
#include <xmmintrin.h>

// wx
//#include <wx/msw/setup.h>
#include <wx/wxprec.h>
#include <wx/rawbmp.h>

// std
#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <cmath>

// boost
//#define BOOST_LIB_DIAGNOSTIC
#include <boost/thread.hpp>
#include <boost/timer.hpp>

// OpenMP
#include <omp.h>

// CUDA
#include <cuda_runtime.h>
#include <cutil.h>

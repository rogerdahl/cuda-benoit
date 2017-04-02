// app
#include "../../lib/int_types.h"

// std
#include <fstream>
#include <iostream>
#include <string>

// boost::filesystem
#include <boost/filesystem/exception.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/filesystem/operations.hpp>

// boost::format
#include <boost/format.hpp>

// OpenGL
#include <GL/gl.h>
#include <GL/glew.h>
#include <GL/glext.h>
#include <GL/glut.h>

// CUDA.
#include <cuda_runtime.h>
//#include "cutil_inline.h"
//#include "cutil_inline_runtime.h"
#include <helper_cuda.h>
#define cutilSafeCall(x) checkCudaErrors(x)
#define cutilSafeCallNoSync(x) checkCudaErrors(x)
//#include <cutil_gl_inline.h>
#include <cuda_gl_interop.h>
#include <vector_types.h>

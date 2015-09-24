// app
#include "../../lib/int_types.h"

// std
#include <string>
#include <fstream>
#include <iostream>

// boost::filesystem
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/filesystem/exception.hpp>

// boost::format
#include <boost/format.hpp>

// OpenGL
#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glut.h>
#include <GL/glext.h>

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


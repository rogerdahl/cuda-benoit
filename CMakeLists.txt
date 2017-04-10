cmake_minimum_required(VERSION 3.5)
project(cuda_benoit)
set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

set(SRC_EDITOR_DIR ${CMAKE_SOURCE_DIR}/src/editor)
set(SRC_LIB_DIR ${CMAKE_SOURCE_DIR}/src/lib)
set(SRC_PLAYER_DIR ${CMAKE_SOURCE_DIR}/src/player)
set(SRC_TRACK_DIR ${CMAKE_SOURCE_DIR}/src/track)

find_package(CUDA QUIET REQUIRED)
message(STATUS "Found CUDA version: ${CUDA_VERSION}")
#set(BUILD_SHARED_LIBS OFF)
#set(CUDA_SEPARABLE_COMPILATION ON)

#set(Boost_USE_STATIC_LIBS   ON)
find_package(
  Boost 1.58.0 EXACT REQUIRED COMPONENTS
  filesystem
  program_options
  serialization
  system
  thread
)

#set(wxWidgets_CONFIGURATION wxgtk)
find_package(wxWidgets COMPONENTS core base REQUIRED)

find_package(OpenMP)
if (OPENMP_FOUND)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set(
  CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};
  -gencode=arch=compute_61,code=sm_61
  -gencode arch=compute_50,code=sm_50
  -gencode arch=compute_35,code=sm_35
  -std=c++11;
)

#message(WARNING "${wxWidgets_DEFINITIONS_DEBUG}")
#message(WARNING "${wxWidgets_DEFINITIONS}")
#message(WARNING "${wxWidgets_CONFIGURATION}")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  message(WARNING "Building DEBUG version")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -g -G -O0)
#  add_definitions(${wxWidgets_DEFINITIONS_DEBUG})
else()
  message(WARNING "Building RELEASE version")
#  add_definitions(${wxWidgets_DEFINITIONS})
endif()

add_definitions(${wxWidgets_CONFIGURATION})

include("${wxWidgets_USE_FILE}")

set(CMAKE_CXX_FLAGS
  "${CMAKE_CXX_FLAGS} ${wxWidgets_CXX_FLAGS} -Wall -Werror -Wfatal-errors -Wno-deprecated-declarations"
)

# Experimenting with building a statically linked binary that will run out of the
# box on more Linux distros.
#set(CMAKE_CXX_FLAGS
#  "${CMAKE_CXX_FLAGS} -static"
#)

set(CMAKE_CXX_STANDARD 14)

# Specify include directories
include_directories(
  ${SRC_EDITOR_DIR}
  ${SRC_LIB_DIR}
  ${SRC_PLAYER_DIR}
  ${SRC_TRACK_DIR}
  ${Boost_INCLUDE_DIRS}
  ${wxWidgets_INCLUDE_DIRS}
)

link_directories(
  ${Boost_LIBRARY_DIRS}
  ${wxWidgets_LIBRARY_DIRS}
)

link_libraries(
  ${Boost_LIBRARIES}
  ${wxWidgets_LIBRARIES}
  GLEW
  glut
  GL
)

cuda_add_executable(
  player
  ${SRC_LIB_DIR}/cuda_util.cpp
  ${SRC_PLAYER_DIR}/config.cpp
  ${SRC_PLAYER_DIR}/cuda_timers.cpp
  ${SRC_PLAYER_DIR}/kernels.cu
  ${SRC_PLAYER_DIR}/main.cpp
  ${SRC_PLAYER_DIR}/pch.cpp
  ${SRC_PLAYER_DIR}/platform.cpp
  ${SRC_PLAYER_DIR}/tracks.cpp
  ${SRC_TRACK_DIR}/pch.cpp
  ${SRC_TRACK_DIR}/track.cpp
)

cuda_add_executable(
  editor
  ${SRC_EDITOR_DIR}/bailout_dlg.cpp
  ${SRC_EDITOR_DIR}/calc.cpp
  ${SRC_EDITOR_DIR}/cpu_info.cpp
  ${SRC_EDITOR_DIR}/cpuid.cpp
  ${SRC_EDITOR_DIR}/main.cpp
  ${SRC_EDITOR_DIR}/mandelbrot_kernel.cu
  ${SRC_EDITOR_DIR}/palette_ctrl.cpp
  ${SRC_EDITOR_DIR}/pch.cpp
  ${SRC_EDITOR_DIR}/render_ctrl.cpp
  ${SRC_EDITOR_DIR}/sse2_x64_mandelbrot.asm
  ${SRC_EDITOR_DIR}/utils.h
  ${SRC_TRACK_DIR}/track.cpp
)

message(WARNING "CUDA_NVCC_FLAGS: ${CUDA_NVCC_FLAGS}")
message(WARNING "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")

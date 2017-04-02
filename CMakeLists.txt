cmake_minimum_required(VERSION 3.5)
project(cuda_benoit)
#set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

set(SRC_EDITOR_DIR ${CMAKE_SOURCE_DIR}/src/editor)
set(SRC_LIB_DIR ${CMAKE_SOURCE_DIR}/src/lib)
set(SRC_PLAYER_DIR ${CMAKE_SOURCE_DIR}/src/player)
set(SRC_TRACK_DIR ${CMAKE_SOURCE_DIR}/src/track)

find_package(CUDA QUIET REQUIRED)
message(STATUS "Found CUDA version: ${CUDA_VERSION}")
#set(BUILD_SHARED_LIBS OFF)
#set(CUDA_SEPARABLE_COMPILATION ON)

find_package(
  Boost 1.58.0 EXACT REQUIRED COMPONENTS
  program_options
  system
  filesystem
  serialization
)

set(
  CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};
  -gencode=arch=compute_61,code=sm_61
  -gencode arch=compute_50,code=sm_50
  -gencode arch=compute_35,code=sm_35
  -std=c++11;
)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -g -G -O0)
  message(WARNING "Building DEBUG version: ${CUDA_NVCC_FLAGS}")
else()
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O3)
  message(WARNING "Building RELEASE version: ${CUDA_NVCC_FLAGS}")
endif()

set(CMAKE_CXX_STANDARD 14)

# Specify include directories
include_directories(
  ${SRC_EDITOR_DIR}
  ${SRC_LIB_DIR}
  ${SRC_PLAYER_DIR}
  ${SRC_TRACK_DIR}
  ${Boost_INCLUDE_DIRS}
)

link_directories(
  ${Boost_LIBRARY_DIRS}
)

link_libraries(
  ${Boost_LIBRARIES}
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

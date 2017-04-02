cmake_minimum_required(VERSION 3.5)
project(cuda_benoit)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp)
add_executable(cuda_benoit ${SOURCE_FILES})
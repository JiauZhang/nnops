cmake_minimum_required(VERSION 3.15)

project(nnops)

set(CMAKE_CXX_STANDARD 17)

include_directories(${CMAKE_SOURCE_DIR})

add_subdirectory(nnops)
add_subdirectory(python)
add_subdirectory(tests/cpp)

cmake_minimum_required(VERSION 3.15)

project(datamana)

find_package(Python 3.8 COMPONENTS Interpreter Development REQUIRED)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/nnops)

nanobind_add_module(
  _C STABLE_ABI
  nnops/data_type.cpp nnops/tensor_shape.cpp nnops/tensor.cpp nnops/device.cpp
  nnops/cpu/device.cpp nnops/cuda/device.cpp nnops/npu/device.cpp
  nnops/python.cpp
)
target_link_libraries(_C PRIVATE rt)

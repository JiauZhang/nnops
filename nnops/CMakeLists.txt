include_directories(${CMAKE_SOURCE_DIR}/nnops)

add_library(
    nnops SHARED
    data_type.cpp
)
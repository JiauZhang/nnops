add_library(
    nnops SHARED
    data_type.cpp tensor_meta.cpp tensor.cpp tensor_iterator.cpp scalar.cpp operator.cpp
    tensor_buffer.cpp device.cpp
)

add_subdirectory(cpu)
add_subdirectory(cuda)
add_subdirectory(npu)
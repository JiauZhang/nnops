add_library(
    nnops SHARED
    data_type.cpp tensor_meta.cpp tensor.cpp tensor_indexing.cpp
    tensor_buffer.cpp device.cpp cpu/device.cpp
)
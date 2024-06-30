add_library(
    nnops SHARED
    data_type.cpp tensor_meta.cpp tensor.cpp device.cpp
    cpu/device.cpp
)
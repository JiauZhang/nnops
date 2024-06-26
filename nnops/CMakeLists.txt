add_library(
    nnops SHARED
    data_type.cpp tensor_shape.cpp tensor.cpp device.cpp
    cpu/device.cpp
)
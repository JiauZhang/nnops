add_library(
    nnops SHARED
    data_type.cpp tensor_meta.cpp tensor.cpp tensor_buffer.cpp device.cpp scalar.cpp
    tensor_iterator.cpp tensor_accessor.cpp tensor_util.cpp
    operator.cpp
    random.cpp
)

add_subdirectory(cpu)
add_subdirectory(cuda)
add_subdirectory(npu)

install(TARGETS nnops DESTINATION .)
nanobind_add_module(
  _C STABLE_ABI
  data_type.cpp tensor.cpp device.cpp binary_ops.cpp random.cpp
  python.cpp
)
target_link_libraries(_C PRIVATE nnops)

install(TARGETS _C DESTINATION .)
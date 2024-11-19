nanobind_add_module(
  _C STABLE_ABI
  data_type.cpp tensor.cpp device.cpp ops.cpp
  python.cpp
)
target_link_libraries(_C PRIVATE nnops)

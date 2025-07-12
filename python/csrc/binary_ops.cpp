#include <nnops/data_type.h>
#include <nnops/scalar.h>
#include <nnops/tensor_operator.h>
#include <nanobind/nanobind.h>
#include <python/csrc/tensor.h>
#include <python/csrc/binary_ops.h>

namespace nb = nanobind;
using nnops::Tensor, nnops::Scalar;

namespace pynnops {

void DEFINE_OPS_MODULE(nb::module_ & (m)) {
#define MAKE_BINARY_OP_TENSOR_TENSOR_BINDING(op_type, op_name, op_symbol) \
    m.def(#op_name, &op_name##_tensor_tensor); \
    m.def("i"#op_name, &i##op_name##_tensor_tensor);
SCALAR_BINARY_OP_GEN_TEMPLATE_LOOPx1(MAKE_BINARY_OP_TENSOR_TENSOR_BINDING)

#define MAKE_BINARY_OP_TENSOR_SCALAR_BINDING(op_type, op_name, op_symbol, type) \
    m.def(#op_name, &op_name##type##_tensor_scalar); \
    m.def("i"#op_name, &i##op_name##type##_tensor_scalar); \
    m.def(#op_name, &op_name##type##_tensor_scalar_reverse);
#define MAKE_BINARY_OP_TENSOR_SCALAR_DTYPE_BINDING(dtype, type) \
SCALAR_BINARY_OP_GEN_TEMPLATE_LOOPx1(MAKE_BINARY_OP_TENSOR_SCALAR_BINDING, type)
DATATYPE_GEN_TEMPLATE_LOOPx1(MAKE_BINARY_OP_TENSOR_SCALAR_DTYPE_BINDING)

    m.def("matmul", &matmul);
}

} // namespace pynnops
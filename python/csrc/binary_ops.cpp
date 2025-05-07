#include <nnops/data_type.h>
#include <nnops/scalar.h>
#include <nnops/tensor_operator.h>
#include <nanobind/nanobind.h>
#include <python/csrc/tensor.h>

namespace nb = nanobind;
using nnops::Tensor, nnops::Scalar;

namespace pynnops {

#define MAKE_BINARY_OP_TENSOR_TENSOR_FUNCTOR(op_type, op_name, op) \
    Tensor op_name##_tensor_tensor(const Tensor &self, const Tensor &other) { return self op other; }
SCALAR_BINARY_OP_GEN_TEMPLATE_LOOPx1(MAKE_BINARY_OP_TENSOR_TENSOR_FUNCTOR)

#define MAKE_BINARY_OP_TENSOR_SCALAR_FUNCTOR(op_type, op_name, op, type) \
    Tensor op_name##type##_tensor_scalar(const Tensor &self, const type other) { return self op other; }
#define MAKE_BINARY_OP_TENSOR_SCALAR_DTYPE_FUNCTOR(dtype, type) \
SCALAR_BINARY_OP_GEN_TEMPLATE_LOOPx1(MAKE_BINARY_OP_TENSOR_SCALAR_FUNCTOR, type)
DATATYPE_GEN_TEMPLATE_LOOPx1(MAKE_BINARY_OP_TENSOR_SCALAR_DTYPE_FUNCTOR)

#define MAKE_BINARY_OP_TENSOR_SCALAR_FUNCTOR_REVERSE(op_type, op_name, op, type) \
    Tensor op_name##type##_tensor_scalar_reverse(const Tensor &self, const type other) { return other op self; }
#define MAKE_BINARY_OP_TENSOR_SCALAR_DTYPE_FUNCTOR_REVERSE(dtype, type) \
SCALAR_BINARY_OP_GEN_TEMPLATE_LOOPx1(MAKE_BINARY_OP_TENSOR_SCALAR_FUNCTOR_REVERSE, type)
DATATYPE_GEN_TEMPLATE_LOOPx1(MAKE_BINARY_OP_TENSOR_SCALAR_DTYPE_FUNCTOR_REVERSE)

Tensor matmul(const Tensor &lvalue, const Tensor &rvalue) { return nnops::cpu::ops::matmul(lvalue, rvalue); }

void DEFINE_OPS_MODULE(nb::module_ & (m)) {
#define MAKE_BINARY_OP_TENSOR_TENSOR_BINDING(op_type, op_name, op) m.def(#op_name, &op_name##_tensor_tensor);
SCALAR_BINARY_OP_GEN_TEMPLATE_LOOPx1(MAKE_BINARY_OP_TENSOR_TENSOR_BINDING)

#define MAKE_BINARY_OP_TENSOR_SCALAR_BINDING(op_type, op_name, op, type) \
    m.def(#op_name, &op_name##type##_tensor_scalar); \
    m.def(#op_name, &op_name##type##_tensor_scalar_reverse);
#define MAKE_BINARY_OP_TENSOR_SCALAR_DTYPE_BINDING(dtype, type) \
SCALAR_BINARY_OP_GEN_TEMPLATE_LOOPx1(MAKE_BINARY_OP_TENSOR_SCALAR_BINDING, type)
DATATYPE_GEN_TEMPLATE_LOOPx1(MAKE_BINARY_OP_TENSOR_SCALAR_DTYPE_BINDING)

    m.def("matmul", &matmul);
}

} // namespace pynnops
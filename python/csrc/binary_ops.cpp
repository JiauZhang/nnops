#include <nnops/data_type.h>
#include <nnops/scalar.h>
#include <nnops/tensor_operator.h>
#include <nanobind/nanobind.h>
#include <python/csrc/tensor.h>

namespace nb = nanobind;
using nnops::Tensor, nnops::Scalar;

namespace pynnops {

#define MAKE_BINARY_OP_TENSOR_TENSOR_FUNCTOR(op_type, op_name, op) \
PyTensor op_name##_tensor_tensor(const PyTensor &self, const PyTensor &other) {  \
    Tensor st = self.tensor(), ot = other.tensor();  \
    Tensor o = st op ot;     \
    return PyTensor(o);                              \
}
SCALAR_BINARY_OP_GEN_TEMPLATE_LOOPx1(MAKE_BINARY_OP_TENSOR_TENSOR_FUNCTOR)

#define MAKE_BINARY_OP_TENSOR_SCALAR_FUNCTOR(op_type, op_name, op, type) \
PyTensor op_name##type##_tensor_scalar(const PyTensor &self, const type other) {  \
    Tensor st = self.tensor(); \
    Scalar ot(other);  \
    Tensor o = st op ot;     \
    return PyTensor(o);                              \
}
#define MAKE_BINARY_OP_TENSOR_SCALAR_DTYPE_FUNCTOR(dtype, type) \
SCALAR_BINARY_OP_GEN_TEMPLATE_LOOPx1(MAKE_BINARY_OP_TENSOR_SCALAR_FUNCTOR, type)
DATATYPE_GEN_TEMPLATE_LOOPx1(MAKE_BINARY_OP_TENSOR_SCALAR_DTYPE_FUNCTOR)

PyTensor matmul(const PyTensor &lvalue, const PyTensor &rvalue) {
    Tensor lv = lvalue.tensor(), rv = rvalue.tensor();
    Tensor o = nnops::cpu::ops::matmul(lv, rv);
    return PyTensor(o);
}

void DEFINE_OPS_MODULE(nb::module_ & (m)) {
#define MAKE_BINARY_OP_TENSOR_TENSOR_BINDING(op_type, op_name, op) m.def(#op_name, &op_name##_tensor_tensor);
SCALAR_BINARY_OP_GEN_TEMPLATE_LOOPx1(MAKE_BINARY_OP_TENSOR_TENSOR_BINDING)

#define MAKE_BINARY_OP_TENSOR_SCALAR_BINDING(op_type, op_name, op, type) m.def(#op_name, &op_name##type##_tensor_scalar);
#define MAKE_BINARY_OP_TENSOR_SCALAR_DTYPE_BINDING(dtype, type) \
SCALAR_BINARY_OP_GEN_TEMPLATE_LOOPx1(MAKE_BINARY_OP_TENSOR_SCALAR_BINDING, type)
DATATYPE_GEN_TEMPLATE_LOOPx1(MAKE_BINARY_OP_TENSOR_SCALAR_DTYPE_BINDING)

    m.def("matmul", &matmul);
}

} // namespace pynnops
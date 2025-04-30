#include <nnops/data_type.h>
#include <nnops/tensor_operator.h>
#include <nanobind/nanobind.h>
#include <python/csrc/tensor.h>

namespace nb = nanobind;

namespace pynnops {

#define MAKE_BINARY_OP_FUNCTOR(op_type, op_name, op) \
PyTensor op_name(const PyTensor &self, const PyTensor &other) {  \
    Tensor st = self.tensor(), ot = other.tensor();  \
    Tensor o = st op ot;     \
    return PyTensor(o);                              \
}
SCALAR_BINARY_OP_GEN_TEMPLATE_LOOPx1(MAKE_BINARY_OP_FUNCTOR)

PyTensor matmul(const PyTensor &lvalue, const PyTensor &rvalue) {
    Tensor lv = lvalue.tensor(), rv = rvalue.tensor();
    Tensor o = nnops::cpu::ops::matmul(lv, rv);
    return PyTensor(o);
}

void DEFINE_OPS_MODULE(nb::module_ & (m)) {
#define MAKE_BINARY_OP_BINDING(op_type, op_name, op) m.def(#op_name, &op_name);
    SCALAR_BINARY_OP_GEN_TEMPLATE_LOOPx1(MAKE_BINARY_OP_BINDING)
#undef MAKE_BINARY_OP_BINDING

    m.def("matmul", &matmul);
}

} // namespace pynnops
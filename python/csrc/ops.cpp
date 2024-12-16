#include <nnops/data_type.h>
#include <nnops/cpu/ops/math.h>
#include <nanobind/nanobind.h>
#include <python/csrc/tensor.h>

namespace nb = nanobind;

namespace pynnops {

#define MAKE_BINARY_OP_FUNCTOR(op_type, op_name, op) \
PyTensor op_name(PyTensor &self, PyTensor &other) {  \
    Tensor st = self.tensor(), ot = other.tensor();  \
    Tensor o = nnops::cpu::ops::op_name(st, ot);     \
    return PyTensor(o);                              \
}
SCALAR_BINARY_OP_GEN_TEMPLATE_LOOPx1(MAKE_BINARY_OP_FUNCTOR)

#define MAKE_BINARY_OP_BINDING(op_type, op_name, op) m.def(#op_name, &op_name);
void DEFINE_OPS_MODULE(nb::module_ & (m)) {
    SCALAR_BINARY_OP_GEN_TEMPLATE_LOOPx1(MAKE_BINARY_OP_BINDING)
}

} // namespace pynnops
#include <nnops/cpu/ops/math.h>
#include <nanobind/nanobind.h>
#include <python/csrc/tensor.h>

namespace nb = nanobind;

namespace pynnops {

PyTensor add(PyTensor &self, PyTensor &other) {
    Tensor st = self.tensor(), ot = other.tensor();
    Tensor o = nnops::cpu::ops::add(st, ot);
    return PyTensor(o);
}

void DEFINE_OPS_MODULE(nb::module_ & (m)) {
    m.def("add", &add);
}

} // namespace pynnops
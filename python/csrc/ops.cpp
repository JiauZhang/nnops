#include <nnops/cpu/ops/math.h>
#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace pynnops {

void DEFINE_OPS_MODULE(nb::module_ & (m)) {
    m.def("add", &nnops::cpu::ops::add);
}

} // namespace pynnops
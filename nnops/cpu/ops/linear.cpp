#include <nnops/cpu/ops/functional.h>
#include <nnops/tensor.h>
#include <nnops/data_type.h>
#include <stdexcept>

using nnops::Tensor, nnops::TensorMeta;

namespace nnops::cpu::ops {

Tensor linear(const Tensor &input, const Tensor &weight, const std::optional<Tensor> &bias) {
    return Tensor();
}

} // namespace nnops::cpu::ops

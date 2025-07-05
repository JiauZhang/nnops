#include <nnops/cpu/ops/unary_ops.h>
#include <nnops/tensor.h>
#include <nnops/data_type.h>
#include <nnops/common.h>

using nnops::Tensor, nnops::TensorMeta;

namespace nnops::cpu::ops {

Tensor linear(const Tensor &input, const Tensor &weight, const std::optional<Tensor> &bias) {
    NNOPS_CHECK(
        (input.shape(-1) == weight.shape(1)) && (!bias.has_value() || bias.value().shape(0) == weight.shape(0)),
        "linear input and weight(or bias) are incompatible.");
    return Tensor();
}

} // namespace nnops::cpu::ops

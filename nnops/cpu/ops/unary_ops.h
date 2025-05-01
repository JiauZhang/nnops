#ifndef __UNARY_OPS_H__
#define __UNARY_OPS_H__

#include <nnops/tensor.h>
#include <optional>

using nnops::Tensor;

namespace nnops::cpu::ops {

Tensor linear(const Tensor &input, const Tensor &weight, const std::optional<Tensor> &bias);

} // namespace nnops::cpu::ops

#endif // __UNARY_OPS_H__
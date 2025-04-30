#ifndef __OPS_MATH_H__
#define __OPS_MATH_H__

#include <nnops/tensor.h>
#include <optional>

using nnops::Tensor;

namespace nnops::cpu::ops {

template<ScalarBinaryOpType op_type>
Tensor binary_op_template(Tensor &self, Tensor &other);

#define MAKE_BINARY_OP_FUNCTOR(op_type, op_name, op) \
inline Tensor operator op (Tensor &self, Tensor &other) { return binary_op_template<op_type>(self, other); }
SCALAR_BINARY_OP_GEN_TEMPLATE_LOOPx1(MAKE_BINARY_OP_FUNCTOR)
#undef MAKE_BINARY_OP_FUNCTOR

Tensor matmul(const Tensor &lvalue, const Tensor &rvalue);
Tensor linear(const Tensor &input, const Tensor &weight, const std::optional<Tensor> &bias);

} // namespace nnops::cpu::ops

#endif // __OPS_MATH_H__
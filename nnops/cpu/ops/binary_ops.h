#ifndef __BINARY_OPS_H__
#define __BINARY_OPS_H__

#include <nnops/tensor.h>
#include <nnops/scalar.h>
#include <optional>

using nnops::Tensor;

namespace nnops::cpu::ops {

template<ScalarBinaryOpType op_type>
Tensor binary_op_tensor_tensor_template(const Tensor &self, const Tensor &other);

#define MAKE_BINARY_OP_TENSOR_TENSOR_FUNCTOR(op_type, op_name, op) \
inline Tensor operator op (const Tensor &self, const Tensor &other) { return binary_op_tensor_tensor_template<op_type>(self, other); }
SCALAR_BINARY_OP_GEN_TEMPLATE_LOOPx1(MAKE_BINARY_OP_TENSOR_TENSOR_FUNCTOR)
#undef MAKE_BINARY_OP_TENSOR_TENSOR_FUNCTOR

template<ScalarBinaryOpType op_type>
Tensor binary_op_tensor_scalar_template(const Tensor &self, const Scalar &other);
template<ScalarBinaryOpType op_type>
Tensor binary_op_tensor_scalar_template_reverse(const Scalar &other, const Tensor &self);

#define MAKE_BINARY_OP_TENSOR_SCALAR_FUNCTOR(op_type, op_name, op) \
inline Tensor operator op (const Tensor &self, const Scalar &other) { return binary_op_tensor_scalar_template<op_type>(self, other); } \
inline Tensor operator op (const Scalar &other, const Tensor &self) { return binary_op_tensor_scalar_template_reverse<op_type>(other, self); }
SCALAR_BINARY_OP_GEN_TEMPLATE_LOOPx1(MAKE_BINARY_OP_TENSOR_SCALAR_FUNCTOR)
#undef MAKE_BINARY_OP_TENSOR_SCALAR_FUNCTOR

Tensor matmul(const Tensor &lvalue, const Tensor &rvalue);

} // namespace nnops::cpu::ops

#endif // __BINARY_OPS_H__
#ifndef __OPS_MATH_H__
#define __OPS_MATH_H__

#include <nnops/tensor.h>

namespace nnops::cpu::ops {

#define MAKE_BINARY_OP_FUNCTOR(op_type, op_name, op) nnops::Tensor op_name(nnops::Tensor &self, nnops::Tensor &other);
SCALAR_BINARY_OP_GEN_TEMPLATE_LOOPx1(MAKE_BINARY_OP_FUNCTOR)
#undef MAKE_BINARY_OP_FUNCTOR

} // namespace nnops::cpu::ops

#endif // __OPS_MATH_H__
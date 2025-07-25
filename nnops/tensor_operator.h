#ifndef __TENSOR_OPERATOR_H__
#define __TENSOR_OPERATOR_H__

#include <nnops/tensor.h>
#include <nnops/data_type.h>
#include <nnops/cpu/ops/binary_ops.h>

namespace nnops {

#define MAKE_BINARY_OP_FUNCTOR(op_type, op_name, op_symbol) \
    using cpu::ops::operator op_symbol;                     \
    using cpu::ops::operator op_symbol##=;
SCALAR_BINARY_OP_GEN_TEMPLATE_LOOPx1(MAKE_BINARY_OP_FUNCTOR)
#undef MAKE_BINARY_OP_FUNCTOR

using cpu::ops::matmul;

} // namespace nnops

#endif // __TENSOR_OPERATOR_H__
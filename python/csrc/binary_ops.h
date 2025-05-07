#ifndef __PY_BINARY_OPS_H__
#define __PY_BINARY_OPS_H__

#include <python/csrc/tensor.h>

namespace pynnops {

#define MAKE_BINARY_OP_TENSOR_TENSOR_FUNCTOR(op_type, op_name, op) \
Tensor op_name##_tensor_tensor(const Tensor &self, const Tensor &other);
SCALAR_BINARY_OP_GEN_TEMPLATE_LOOPx1(MAKE_BINARY_OP_TENSOR_TENSOR_FUNCTOR)
#undef MAKE_BINARY_OP_TENSOR_TENSOR_FUNCTOR

#define MAKE_BINARY_OP_TENSOR_SCALAR_FUNCTOR(op_type, op_name, op, type) \
Tensor op_name##type##_tensor_scalar(const Tensor &self, const type other);
#define MAKE_BINARY_OP_TENSOR_SCALAR_DTYPE_FUNCTOR(dtype, type) \
SCALAR_BINARY_OP_GEN_TEMPLATE_LOOPx1(MAKE_BINARY_OP_TENSOR_SCALAR_FUNCTOR, type)
DATATYPE_GEN_TEMPLATE_LOOPx1(MAKE_BINARY_OP_TENSOR_SCALAR_DTYPE_FUNCTOR)
#undef MAKE_BINARY_OP_TENSOR_SCALAR_DTYPE_FUNCTOR
#undef MAKE_BINARY_OP_TENSOR_SCALAR_FUNCTOR

#define MAKE_BINARY_OP_TENSOR_SCALAR_FUNCTOR_REVERSE(op_type, op_name, op, type) \
    Tensor op_name##type##_tensor_scalar_reverse(const Tensor &self, const type other);
#define MAKE_BINARY_OP_TENSOR_SCALAR_DTYPE_FUNCTOR_REVERSE(dtype, type) \
    SCALAR_BINARY_OP_GEN_TEMPLATE_LOOPx1(MAKE_BINARY_OP_TENSOR_SCALAR_FUNCTOR_REVERSE, type)
DATATYPE_GEN_TEMPLATE_LOOPx1(MAKE_BINARY_OP_TENSOR_SCALAR_DTYPE_FUNCTOR_REVERSE)
#undef MAKE_BINARY_OP_TENSOR_SCALAR_FUNCTOR_REVERSE
#undef MAKE_BINARY_OP_TENSOR_SCALAR_DTYPE_FUNCTOR_REVERSE

Tensor matmul(const Tensor &lvalue, const Tensor &rvalue);

} // namespace pynnops

#endif // __PY_BINARY_OPS_H__
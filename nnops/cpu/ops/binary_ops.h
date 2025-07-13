#ifndef __BINARY_OPS_H__
#define __BINARY_OPS_H__

#include <nnops/tensor.h>
#include <nnops/scalar.h>
#include <optional>

using nnops::Tensor;

namespace nnops::cpu::ops {

template<ScalarBinaryOpType op_type>
void binary_op_tensor_tensor_template(const Tensor &self, const Tensor &other, Tensor &ret);

#define MAKE_BINARY_OP_TENSOR_TENSOR_FUNCTOR(op_type, op_name, op_symbol)           \
    inline Tensor operator op_symbol (const Tensor &self, const Tensor &other) {    \
        Tensor ret;                                                                 \
        binary_op_tensor_tensor_template<op_type>(self, other, ret);                \
        return ret; \
    }                                                                               \
    inline Tensor &operator op_symbol##= (Tensor &self, const Tensor &other) {      \
        binary_op_tensor_tensor_template<op_type>(self, other, self);               \
        return self;                                                                \
    }
SCALAR_BINARY_OP_GEN_TEMPLATE_LOOPx1(MAKE_BINARY_OP_TENSOR_TENSOR_FUNCTOR)
#undef MAKE_BINARY_OP_TENSOR_TENSOR_FUNCTOR

template<ScalarBinaryOpType op_type>
void binary_op_tensor_scalar_template(const Tensor &self, const Scalar &other, Tensor &ret);
template<ScalarBinaryOpType op_type>
void binary_op_tensor_scalar_template_reverse(const Scalar &other, const Tensor &self, Tensor &ret);

#define MAKE_BINARY_OP_TENSOR_SCALAR_FUNCTOR(op_type, op_name, op_symbol)           \
    inline Tensor operator op_symbol (const Tensor &self, const Scalar &other) {    \
        Tensor ret;                                                                 \
        binary_op_tensor_scalar_template<op_type>(self, other, ret);                \
        return ret;                                                                 \
    }                                                                               \
    inline Tensor &operator op_symbol##= (Tensor &self, const Scalar &other) {      \
        binary_op_tensor_scalar_template<op_type>(self, other, self);               \
        return self;                                                                \
    }                                                                               \
    inline Tensor operator op_symbol (const Scalar &other, const Tensor &self) {    \
        Tensor ret;                                                                 \
        binary_op_tensor_scalar_template_reverse<op_type>(other, self, ret);        \
        return ret;                                                                 \
    }
SCALAR_BINARY_OP_GEN_TEMPLATE_LOOPx1(MAKE_BINARY_OP_TENSOR_SCALAR_FUNCTOR)
#undef MAKE_BINARY_OP_TENSOR_SCALAR_FUNCTOR

Tensor matmul(const Tensor &lvalue, const Tensor &rvalue);

} // namespace nnops::cpu::ops

#endif // __BINARY_OPS_H__
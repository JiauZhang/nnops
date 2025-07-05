#include <nnops/cpu/ops/binary_ops.h>
#include <nnops/tensor.h>
#include <nnops/scalar.h>
#include <nnops/data_type.h>
#include <nnops/common.h>

using nnops::Tensor, nnops::TensorMeta, nnops::TensorShape;

namespace nnops::cpu::ops {

struct ScalarBinaryOpParams {
    const scalar_binary_op_t &op;
    const index_t loop;
    index_t offsets[3];
    const index_t strides[3];
};

void do_binary_op_tensor_tensor_impl(const Tensor &self, const Tensor &other, const Tensor &out, int axis, ScalarBinaryOpParams &params) {
    if (axis < self.ndim() - 1) {
        const int loop = self.shape()[axis];
        for (int i = 0; i < loop; i++) {
            do_binary_op_tensor_tensor_impl(self, other, out, axis + 1, params);
            params.offsets[0] += out.stride()[axis];
            params.offsets[1] += self.stride()[axis];
            params.offsets[2] += other.stride()[axis];
        }
        params.offsets[0] -= out.stride()[axis] * loop;
        params.offsets[1] -= self.stride()[axis] * loop;
        params.offsets[2] -= other.stride()[axis] * loop;
        return;
    }

    void *args[3] = {out.data_ptr(params.offsets[0]), self.data_ptr(params.offsets[1]), other.data_ptr(params.offsets[2])};
    params.op(args, params.strides, params.loop);
}

template<ScalarBinaryOpType op_type>
Tensor binary_op_tensor_tensor_template(const Tensor &self, const Tensor &other) {
    const auto &&self_shape_str = TensorMeta::shape_as_string(self.shape()), &&other_shape_str = TensorMeta::shape_as_string(other.shape());
    NNOPS_CHECK(Tensor::is_broadcastable(self, other),
        "operands could not be broadcast together with shapes %s and %s",
        self_shape_str.c_str(), other_shape_str.c_str()
    );

    TensorShape shape = Tensor::broadcast_shape(self, other);
    DataType dtype = get_promote_type(op_type, self.dtype(), other.dtype());
    Tensor ret(dtype, shape, self.device());
    Tensor self_br = self.astype(dtype).broadcast_to(shape), other_br = other.astype(dtype).broadcast_to(shape);
    auto scalar_binary_op = get_scalar_binary_op(op_type, dtype);
    ScalarBinaryOpParams params = {
        scalar_binary_op, self_br.shape(-1), {
            0, 0, 0
        }, {
            (index_t)(ret.stride(-1) * ret.itemsize()),
            (index_t)(self_br.stride(-1) * self_br.itemsize()),
            (index_t)(other_br.stride(-1) * other_br.itemsize())
        }
    };

    do_binary_op_tensor_tensor_impl(self_br, other_br, ret, 0, params);

    return ret;
}

#define MAKE_BINARY_OP_TENSOR_TENSOR_FUNCTOR(op_type, op_name, op) \
template Tensor binary_op_tensor_tensor_template<op_type>(const Tensor &self, const Tensor &other);
SCALAR_BINARY_OP_GEN_TEMPLATE_LOOPx1(MAKE_BINARY_OP_TENSOR_TENSOR_FUNCTOR)
#undef MAKE_BINARY_OP_TENSOR_TENSOR_FUNCTOR

template<ScalarBinaryOpType op_type>
Tensor binary_op_tensor_scalar_template(const Tensor &self, const Scalar &other) {
    Tensor other_tensor = other.tensor();
    return binary_op_tensor_tensor_template<op_type>(self, other_tensor);
}

template<ScalarBinaryOpType op_type>
Tensor binary_op_tensor_scalar_template_reverse(const Scalar &other, const Tensor &self) {
    Tensor other_tensor = other.tensor();
    return binary_op_tensor_tensor_template<op_type>(other_tensor, self);
}

#define MAKE_BINARY_OP_TENSOR_SCALAR_FUNCTOR(op_type, op_name, op) \
template Tensor binary_op_tensor_scalar_template<op_type>(const Tensor &self, const Scalar &other); \
template Tensor binary_op_tensor_scalar_template_reverse<op_type>(const Scalar &other, const Tensor &self);
SCALAR_BINARY_OP_GEN_TEMPLATE_LOOPx1(MAKE_BINARY_OP_TENSOR_SCALAR_FUNCTOR)
#undef MAKE_BINARY_OP_TENSOR_SCALAR_FUNCTOR

} // namespace nnops::cpu::ops

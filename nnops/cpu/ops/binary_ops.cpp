#include <nnops/cpu/ops/functional.h>
#include <nnops/tensor.h>
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

void do_binary_op_impl(const Tensor &self, const Tensor &other, const Tensor &out, int axis, ScalarBinaryOpParams &params) {
    if (axis < self.ndim() - 1) {
        const int loop = self.shape()[axis];
        for (int i = 0; i < loop; i++) {
            do_binary_op_impl(self, other, out, axis + 1, params);
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
Tensor binary_op_template(Tensor &self, Tensor &other) {
    NNOPS_CHECK(Tensor::is_broadcastable(self, other), "operands could not be broadcast together with shapes "
            + TensorMeta::shape_as_string(self.shape()) + " and " + TensorMeta::shape_as_string(other.shape()))

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

    do_binary_op_impl(self_br, other_br, ret, 0, params);

    return ret;
}

#define MAKE_BINARY_OP_FUNCTOR(op_type, op_name, op) \
template Tensor binary_op_template<op_type>(Tensor &self, Tensor &other);
SCALAR_BINARY_OP_GEN_TEMPLATE_LOOPx1(MAKE_BINARY_OP_FUNCTOR)
#undef MAKE_BINARY_OP_FUNCTOR

} // namespace nnops::cpu::ops

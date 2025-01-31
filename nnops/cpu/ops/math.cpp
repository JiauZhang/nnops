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
static Tensor binary_op_template(Tensor &self, Tensor &other) {
    NNOPS_CHECK(Tensor::is_broadcastable(self, other), "operands could not be broadcast together with shapes "
            + TensorMeta::shape_as_string(self.shape()) + " and " + TensorMeta::shape_as_string(other.shape()))

    TensorShape shape = Tensor::broadcast_shape(self, other);
    Tensor ret(get_promote_type(op_type, self.dtype(), other.dtype()), shape, self.device());
    Tensor self_br = self.broadcast_to(shape), other_br = other.broadcast_to(shape);
    auto scalar_binary_op = get_scalar_binary_op(op_type, self.dtype(), other.dtype());
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
Tensor op_name(Tensor &self, Tensor &other) {        \
    return binary_op_template<op_type>(self, other); \
}
SCALAR_BINARY_OP_GEN_TEMPLATE_LOOPx1(MAKE_BINARY_OP_FUNCTOR)

void matmul_2d_impl(float *lvalue, float *rvalue, float *out, const index_t *shape, const index_t *strides) {
    // shape(lvalue: (m, n), rvalue: (n, k), out: (m, k)): [m, n, k]
    // strides: [lvalue.strides, rvalue.strides, out.strides] in bytes
    for (int i = 0; i < shape[0]; i++) {
        for (int j = 0; j < shape[2]; j++) {
            auto idx = 0;
            for (int k = 0; k < shape[1]; k++) {

            }
        }
    }
}

Tensor matmul(const Tensor &lvalue, const Tensor &rvalue) {
    NNOPS_CHECK(lvalue.ndim() >= 2 && rvalue.ndim() >= 2, "matmul lvalue and rvalue ndim must be greater than 2.")
    NNOPS_CHECK(lvalue.shape(-1) == rvalue.shape(-2), "matmul lvalue and rvalue are incompatible.")
    NNOPS_CHECK(Tensor::is_broadcastable(lvalue.shape(), rvalue.shape(), 2), "matmul lvalue and rvalue are not broadcastable.")
    TensorShape shape = Tensor::broadcast_shape(lvalue.shape(), rvalue.shape(), 2);
    auto size = shape.size();
    shape.resize(size + 2);

    shape[size] = lvalue.shape(-2);
    shape[size + 1] = lvalue.shape(-1);
    Tensor lvalue_br = lvalue.broadcast_to(shape);

    shape[size] = rvalue.shape(-2);
    shape[size + 1] = rvalue.shape(-1);
    Tensor rvalue_br = rvalue.broadcast_to(shape);

    return Tensor();
}

} // namespace nnops::cpu::ops

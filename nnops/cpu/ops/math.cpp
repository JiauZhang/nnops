#include <nnops/cpu/ops/math.h>
#include <nnops/tensor.h>
#include <nnops/data_type.h>
#include <stdexcept>

using nnops::Tensor, nnops::TensorMeta;

namespace nnops::cpu::ops {

void do_binary_op_impl(Tensor &self, Tensor &other, Tensor &out, int axis, scalar_binary_op_t &op) {
    if (axis < self.ndim() - 1) {
        index_t &self_offset = self.mutable_offset();
        index_t &other_offset = other.mutable_offset();
        index_t &out_offset = out.mutable_offset();
        const int loop = self.shape()[axis];
        for (int i = 0; i < loop; i++) {
            do_binary_op_impl(self, other, out, axis + 1, op);
            self_offset += self.stride()[axis];
            other_offset += other.stride()[axis];
            out_offset += out.stride()[axis];
        }
        self_offset -= self.stride()[axis] * loop;
        other_offset -= other.stride()[axis] * loop;
        out_offset -= out.stride()[axis] * loop;
        return;
    }

    const index_t loop = self.shape()[axis];
    void *args[3] = {out.data_ptr(), self.data_ptr(), other.data_ptr()};
    const index_t strides[3] = {
        (index_t)(out.stride()[axis] * out.itemsize()),
        (index_t)(self.stride()[axis] * self.itemsize()),
        (index_t)(other.stride()[axis] * other.itemsize()),
    };
    op(args, strides, loop);
}

template<ScalarBinaryOpType op_type>
static Tensor binary_op_template(Tensor &self, Tensor &other) {
    if (!Tensor::is_broadcastable(self, other)) {
        std::string info = "operands could not be broadcast together with shapes "
            + TensorMeta::shape_as_string(self.shape())
            + " and " + TensorMeta::shape_as_string(other.shape());
        throw std::runtime_error(info);
    }

    TensorShape shape = Tensor::broadcast_shape(self, other);
    Tensor ret(get_promote_type(op_type, self.dtype(), other.dtype()), shape, self.device());
    Tensor self_br = self.broadcast_to(shape), other_br = other.broadcast_to(shape);
    auto scalar_binary_op = get_scalar_binary_op(op_type, self.dtype(), other.dtype());

    do_binary_op_impl(self_br, other_br, ret, 0, scalar_binary_op);

    return ret;
}

#define MAKE_BINARY_OP_FUNCTOR(op_type, op_name, op) \
Tensor op_name(Tensor &self, Tensor &other) {        \
    return binary_op_template<op_type>(self, other); \
}
SCALAR_BINARY_OP_GEN_TEMPLATE_LOOPx1(MAKE_BINARY_OP_FUNCTOR)

} // namespace nnops::cpu::ops

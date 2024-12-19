#include <nnops/cpu/ops/math.h>
#include <nnops/tensor.h>
#include <nnops/data_type.h>
#include <stdexcept>

using nnops::Tensor, nnops::TensorMeta;

namespace nnops::cpu::ops {

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

    auto self_iter = self_br.begin(), other_iter = other_br.begin(), ret_iter = ret.begin();
    for (; self_iter != self_br.end(); ++self_iter, ++other_iter, ++ret_iter)
        scalar_binary_op(*ret_iter, *self_iter, *other_iter);

    return ret;
}

#define MAKE_BINARY_OP_FUNCTOR(op_type, op_name, op) \
Tensor op_name(Tensor &self, Tensor &other) {        \
    return binary_op_template<op_type>(self, other); \
}
SCALAR_BINARY_OP_GEN_TEMPLATE_LOOPx1(MAKE_BINARY_OP_FUNCTOR)

} // namespace nnops::cpu::ops

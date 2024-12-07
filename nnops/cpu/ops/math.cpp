#include <nnops/cpu/ops/math.h>
#include <nnops/tensor.h>

using nnops::Tensor, nnops::TensorMeta;

namespace nnops::cpu::ops {

Tensor add(Tensor &self, Tensor &other) {
    if (!Tensor::is_broadcastable(self, other)) {
        std::string info = "operands could not be broadcast together with shapes "
            + TensorMeta::shape_as_string(self.shape())
            + " and " + TensorMeta::shape_as_string(other.shape());
        throw std::runtime_error(info);
    }

    TensorShape shape = Tensor::broadcast_shape(self, other);
    Tensor ret(self.dtype(), shape, self.device());
    Tensor self_br = self.broadcast_to(shape), other_br = other.broadcast_to(shape);

    auto self_iter = self_br.begin(), other_iter = other_br.begin(), ret_iter = ret.begin();
    for (; self_iter != self_br.end(); ++self_iter, ++other_iter, ++ret_iter) {
        *reinterpret_cast<float *>(*ret_iter) = *reinterpret_cast<float *>(*self_iter) + *reinterpret_cast<float *>(*other_iter);
    }

    return ret;
}

} // namespace nnops::cpu::ops

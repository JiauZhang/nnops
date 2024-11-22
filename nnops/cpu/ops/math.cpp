#include <nnops/cpu/ops/math.h>
#include <nnops/tensor.h>

using nnops::Tensor, nnops::TensorMeta;

namespace nnops::cpu::ops {

void add_impl(Tensor &self, int offset_self, Tensor &other, int offset_other, Tensor &ret, int offset_ret, int dim) {
    if (dim < self.ndim() - 1) {
        for (int i=0; i<self.shape()[dim]; i++)
            add_impl(
                self, offset_self + i * self.stride()[dim], other, offset_other + i * other.stride()[dim],
                ret, offset_ret + i * ret.stride()[dim], dim+1);
    }

    float *self_ptr = (float *)self.data_ptr() + offset_self;
    float *other_ptr = (float *)other.data_ptr() + offset_other;
    float *ret_ptr = (float *)ret.data_ptr() + offset_ret;
    for (int i=0; i<self.shape()[dim]; i++) {
        *ret_ptr = *self_ptr + *other_ptr;
        ret_ptr += ret.stride()[dim];
        self_ptr += self.stride()[dim];
        other_ptr += other.stride()[dim];
    }
}

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

    add_impl(self_br, self_br.offset(), other_br, other_br.offset(), ret, ret.offset(), 0);

    return ret;
}

} // namespace nnops::cpu::ops

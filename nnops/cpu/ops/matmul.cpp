#include <nnops/cpu/ops/functional.h>
#include <nnops/tensor.h>
#include <nnops/data_type.h>
#include <nnops/common.h>

using nnops::Tensor;

namespace nnops::cpu::ops {

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

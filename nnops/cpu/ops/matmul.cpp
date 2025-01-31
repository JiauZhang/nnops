#include <nnops/cpu/ops/functional.h>
#include <nnops/tensor.h>
#include <nnops/data_type.h>
#include <nnops/common.h>

using nnops::Tensor;

namespace nnops::cpu::ops {

struct MatMulParams{
    const index_t shape[3];
    const index_t strides[6];
};

void matmul_2d_impl(void *lvalue, void *rvalue, void *out, const index_t *shape, const index_t *strides) {
    // shape(lvalue: (m, n), rvalue: (n, k), out: (m, k)): [m, n, k]
    // strides: [lvalue.strides, rvalue.strides, out.strides] in bytes
    index_t out_ms = 0, lv_ms = 0;
    for (int m = 0; m < shape[0]; m++) {
        index_t out_ks = 0, rv_ks = 0;
        for (int k = 0; k < shape[2]; k++) {
            float *out_mk = (float *)((char *)out + out_ms + out_ks);
            index_t lv_ns = 0, rv_ns = 0;
            *out_mk = 0;
            for (int n = 0; n < shape[1]; n++) {
                const float *lv_mn = (float *)((char *)lvalue + lv_ms + lv_ns);
                const float *rv_nk = (float *)((char *)rvalue + rv_ns + rv_ks);
                *out_mk += (*lv_mn) * (*rv_nk);
                lv_ns += strides[1];
                rv_ns += strides[2];
            }
            out_ks += strides[5];
            rv_ks += strides[3];
        }
        out_ms += strides[4];
        lv_ms += strides[0];
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

    Tensor ret(DataType::TYPE_FLOAT32, {lvalue.shape(-2), rvalue.shape(-1)}, DeviceType::CPU);
    const MatMulParams params = {
        {lvalue_br.shape(-2), lvalue_br.shape(-1), rvalue_br.shape(-1)},
        {
            lvalue_br.stride(-2) * lvalue_br.itemsize(), lvalue_br.stride(-1) * lvalue_br.itemsize(),
            rvalue_br.stride(-2) * rvalue_br.itemsize(), rvalue_br.stride(-1) * rvalue_br.itemsize(),
            ret.stride(-2) * ret.itemsize(), ret.stride(-1) * ret.itemsize()
        }
    };
    matmul_2d_impl(lvalue_br.data_ptr(), rvalue_br.data_ptr(), ret.data_ptr(), params.shape, params.strides);

    return ret;
}

} // namespace nnops::cpu::ops

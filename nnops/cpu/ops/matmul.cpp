#include <nnops/cpu/ops/binary_ops.h>
#include <nnops/tensor.h>
#include <nnops/data_type.h>
#include <nnops/common.h>

using nnops::Tensor;

namespace nnops::cpu::ops {

struct MatMulParams{
    const index_t shape[3];
    const index_t strides[6];
    index_t offsets[3];
};

void matmul_2d_impl(void *lvalue, void *rvalue, void *out, const index_t *shape, const index_t *strides) {
    // shape(lvalue: (m, n), rvalue: (n, k), out: (m, k)): [m, n, k]
    // strides: [lvalue.strides, rvalue.strides, out.strides] in bytes

    index_t out_ms = 0, lv_ms = 0;
    for (int m = 0; m < shape[0]; m++) {
        const char *out_m = (char *)out + out_ms;
        index_t lv_ns = 0, rv_ns = 0;
        float *out_mk = (float *)out_m;

        for (int k = 0; k < shape[2]; k++) {
            *out_mk = 0;
            out_mk = (float *)((char *)out_mk + strides[5]);
        }

        for (int n = 0; n < shape[1]; n++) {
            index_t out_ks = 0, rv_ks = 0;
            const float *lv_mn = (float *)((char *)lvalue + lv_ms + lv_ns);
            float *rv_nk = (float *)((char *)rvalue + rv_ns);
            out_mk = (float *)out_m;
            for (int k = 0; k < shape[2]; k++) {
                *out_mk += (*lv_mn) * (*rv_nk);
                out_mk = (float *)((char *)out_mk + strides[5]);
                rv_nk = (float *)((char *)rv_nk + strides[3]);
            }
            lv_ns += strides[1];
            rv_ns += strides[2];
        }
        out_ms += strides[4];
        lv_ms += strides[0];
    }
}

void matmul_impl(const Tensor &lvalue, const Tensor &rvalue, const Tensor &out, int axis, MatMulParams &params) {
    if (axis < out.ndim() - 2) {
        const int loop = out.shape()[axis];
        for (int i = 0; i < loop; i++) {
            matmul_impl(lvalue, rvalue, out, axis + 1, params);
            params.offsets[0] += lvalue.stride()[axis];
            params.offsets[1] += rvalue.stride()[axis];
            params.offsets[2] += out.stride()[axis];
        }
        params.offsets[0] -= lvalue.stride()[axis] * loop;
        params.offsets[1] -= rvalue.stride()[axis] * loop;
        params.offsets[2] -= out.stride()[axis] * loop;
        return;
    }

    matmul_2d_impl(
        lvalue.data_ptr(params.offsets[0]), rvalue.data_ptr(params.offsets[1]),
        out.data_ptr(params.offsets[2]), params.shape, params.strides
    );
}

Tensor matmul(const Tensor &lvalue, const Tensor &rvalue) {
    NNOPS_CHECK(lvalue.ndim() >= 2 && rvalue.ndim() >= 2, "matmul lvalue and rvalue ndim must be greater than 2.");
    NNOPS_CHECK(lvalue.shape(-1) == rvalue.shape(-2), "matmul lvalue and rvalue are incompatible.");
    NNOPS_CHECK(Tensor::is_broadcastable(lvalue.shape(), rvalue.shape(), 2), "matmul lvalue and rvalue are not broadcastable.");
    TensorShape shape = Tensor::broadcast_shape(lvalue.shape(), rvalue.shape(), 2);
    auto size = shape.size();
    shape.resize(size + 2);

    shape[size] = lvalue.shape(-2);
    shape[size + 1] = lvalue.shape(-1);
    Tensor lvalue_br = lvalue.broadcast_to(shape);

    shape[size] = rvalue.shape(-2);
    shape[size + 1] = rvalue.shape(-1);
    Tensor rvalue_br = rvalue.broadcast_to(shape);

    shape[size] = lvalue.shape(-2);
    shape[size + 1] = rvalue.shape(-1);
    Tensor ret(DataType::TYPE_FLOAT32, shape, DeviceType::CPU);
    MatMulParams params = {
        {lvalue_br.shape(-2), lvalue_br.shape(-1), rvalue_br.shape(-1)},
        {
            lvalue_br.stride(-2) * lvalue_br.itemsize(), lvalue_br.stride(-1) * lvalue_br.itemsize(),
            rvalue_br.stride(-2) * rvalue_br.itemsize(), rvalue_br.stride(-1) * rvalue_br.itemsize(),
            ret.stride(-2) * ret.itemsize(), ret.stride(-1) * ret.itemsize()
        },
        {0, 0, 0}
    };
    matmul_impl(lvalue_br, rvalue_br, ret, 0, params);

    return ret;
}

} // namespace nnops::cpu::ops

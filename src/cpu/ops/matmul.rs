use crate::common::{Index, is_broadcastable, broadcast_shape};
use crate::data_type::DataType;
use crate::tensor::Tensor;

struct MatMulParams {
    shape: [Index; 3],
    strides: [Index; 6],
    offsets: [Index; 3],
}

fn matmul_2d_impl(lvalue: *const u8, rvalue: *const u8, out: *mut u8, shape: &[Index; 3], strides: &[Index; 6]) {
    let mut out_ms: Index = 0;
    let mut lv_ms: Index = 0;
    for _m in 0..shape[0] {
        let out_m = unsafe { out.offset(out_ms as isize) };
        let mut lv_ns: Index = 0;
        let mut rv_ns: Index = 0;

        let mut out_mk = out_m.cast::<f32>();
        for _ in 0..shape[2] {
            unsafe { *out_mk = 0.0; }
            out_mk = unsafe { out_mk.cast::<u8>().offset(strides[5] as isize).cast::<f32>() };
        }

        for _ in 0..shape[1] {
            let lv_mn = unsafe { lvalue.offset(lv_ms as isize).offset(lv_ns as isize).cast::<f32>() };
            let mut rv_nk = unsafe { rvalue.offset(rv_ns as isize).cast::<f32>() };
            out_mk = out_m.cast::<f32>();
            for _ in 0..shape[2] {
                unsafe {
                    *out_mk += (*lv_mn) * (*rv_nk);
                    out_mk = out_mk.cast::<u8>().offset(strides[5] as isize).cast::<f32>();
                    rv_nk = rv_nk.cast::<u8>().offset(strides[3] as isize).cast::<f32>();
                }
            }
            lv_ns += strides[1];
            rv_ns += strides[2];
        }
        out_ms += strides[4];
        lv_ms += strides[0];
    }
}

fn matmul_impl(lvalue: &Tensor, rvalue: &Tensor, out: &Tensor, axis: usize, params: &mut MatMulParams) {
    if axis < out.ndim() - 2 {
        let loop_size = out.shape()[axis];
        for _ in 0..loop_size {
            matmul_impl(lvalue, rvalue, out, axis + 1, params);
            params.offsets[0] += lvalue.stride_at(axis as Index);
            params.offsets[1] += rvalue.stride_at(axis as Index);
            params.offsets[2] += out.stride_at(axis as Index);
        }
        params.offsets[0] -= lvalue.stride_at(axis as Index) * loop_size;
        params.offsets[1] -= rvalue.stride_at(axis as Index) * loop_size;
        params.offsets[2] -= out.stride_at(axis as Index) * loop_size;
        return;
    }

    matmul_2d_impl(
        lvalue.data_ptr_with_offset(params.offsets[0]),
        rvalue.data_ptr_with_offset(params.offsets[1]),
        out.data_ptr_with_offset(params.offsets[2]) as *mut u8,
        &params.shape,
        &params.strides,
    );
}

pub fn matmul(lvalue: &Tensor, rvalue: &Tensor) -> Tensor {
    nnops_check!(lvalue.ndim() >= 2 && rvalue.ndim() >= 2, "matmul lvalue and rvalue ndim must be greater than 2.");
    nnops_check!(lvalue.shape_at(-1) == rvalue.shape_at(-2), "matmul lvalue and rvalue are incompatible.");
    nnops_check!(
        is_broadcastable(lvalue.shape(), rvalue.shape(), 2),
        "matmul lvalue and rvalue are not broadcastable."
    );

    let mut shape = broadcast_shape(lvalue.shape(), rvalue.shape(), 2);
    let size = shape.len();
    shape.resize(size + 2, 0);

    shape[size] = lvalue.shape_at(-2);
    shape[size + 1] = lvalue.shape_at(-1);
    let lvalue_br = lvalue.broadcast_to_shape(&shape, 0);

    shape[size] = rvalue.shape_at(-2);
    shape[size + 1] = rvalue.shape_at(-1);
    let rvalue_br = rvalue.broadcast_to_shape(&shape, 0);

    shape[size] = lvalue.shape_at(-2);
    shape[size + 1] = rvalue.shape_at(-1);
    let ret = Tensor::with_device_type(DataType::Float32, &shape, lvalue.device_type());

    let mut params = MatMulParams {
        shape: [lvalue_br.shape_at(-2), lvalue_br.shape_at(-1), rvalue_br.shape_at(-1)],
        strides: [
            lvalue_br.stride_at(-2) * lvalue_br.itemsize(),
            lvalue_br.stride_at(-1) * lvalue_br.itemsize(),
            rvalue_br.stride_at(-2) * rvalue_br.itemsize(),
            rvalue_br.stride_at(-1) * rvalue_br.itemsize(),
            ret.stride_at(-2) * ret.itemsize(),
            ret.stride_at(-1) * ret.itemsize(),
        ],
        offsets: [0, 0, 0],
    };

    matmul_impl(&lvalue_br, &rvalue_br, &ret, 0, &mut params);
    ret
}
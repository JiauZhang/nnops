use crate::common::{Index, is_broadcastable, is_broadcastable_to, broadcast_shape};
use crate::data_type::{ScalarBinaryOpType, DataType, get_promote_type, scalar_binary_op};
use crate::tensor::Tensor;
use crate::scalar::Scalar;

struct ScalarBinaryOpParams {
    op_type: ScalarBinaryOpType,
    dtype: DataType,
    loop_size: Index,
    offsets: [Index; 3],
    strides: [Index; 3],
}

fn calc_slice_span(itemsize: Index, stride: Index, size: Index) -> usize {
    if size <= 0 { return 0; }
    let s = stride as usize;
    s * (size as usize - 1) + itemsize as usize
}

fn do_binary_op_tensor_tensor_impl(
    self_t: &Tensor,
    other: &Tensor,
    out: &Tensor,
    axis: usize,
    params: &mut ScalarBinaryOpParams,
) {
    if axis < self_t.ndim() - 1 {
        let loop_size = self_t.shape()[axis];
        for _ in 0..loop_size {
            do_binary_op_tensor_tensor_impl(self_t, other, out, axis + 1, params);
            params.offsets[0] += out.stride_at(axis as Index);
            params.offsets[1] += self_t.stride_at(axis as Index);
            params.offsets[2] += other.stride_at(axis as Index);
        }
        params.offsets[0] -= out.stride_at(axis as Index) * loop_size;
        params.offsets[1] -= self_t.stride_at(axis as Index) * loop_size;
        params.offsets[2] -= other.stride_at(axis as Index) * loop_size;
        return;
    }

    let out_span = calc_slice_span(out.itemsize(), params.strides[0], params.loop_size);
    let self_span = calc_slice_span(self_t.itemsize(), params.strides[1], params.loop_size);
    let other_span = calc_slice_span(other.itemsize(), params.strides[2], params.loop_size);

    let out_slice = unsafe {
        std::slice::from_raw_parts_mut(
            out.data_ptr_with_offset(params.offsets[0]) as *mut u8,
            out_span,
        )
    };
    let self_slice = unsafe {
        std::slice::from_raw_parts(self_t.data_ptr_with_offset(params.offsets[1]), self_span)
    };
    let other_slice = unsafe {
        std::slice::from_raw_parts(other.data_ptr_with_offset(params.offsets[2]), other_span)
    };
    scalar_binary_op(
        params.op_type, params.dtype,
        out_slice, self_slice, other_slice,
        params.strides[0], params.strides[1], params.strides[2],
        params.loop_size,
    );
}

pub fn binary_op_tensor_tensor(op_type: ScalarBinaryOpType, self_t: &Tensor, other: &Tensor, ret: &mut Tensor) {
    let (self_ref, other_ref, dtype);
    let out_ref;

    if std::ptr::eq(self_t, ret) {
        nnops_check!(
            is_broadcastable_to(other.shape(), self_t.shape(), 0),
            "could not broadcast tensor from shape {} into shape {}",
            other.shape_as_string(),
            self_t.shape_as_string()
        );
        dtype = self_t.dtype();
        self_ref = self_t.clone();
        other_ref = other.astype(dtype).broadcast_to_shape(self_t.shape(), 0);
        out_ref = self_t.clone();
    } else {
        nnops_check!(
            is_broadcastable(self_t.shape(), other.shape(), 0),
            "operands could not be broadcast together with shape {} and shape {}",
            self_t.shape_as_string(),
            other.shape_as_string()
        );
        let shape = broadcast_shape(self_t.shape(), other.shape(), 0);
        dtype = get_promote_type(op_type, self_t.dtype(), other.dtype());
        self_ref = self_t.astype(dtype).broadcast_to_shape(&shape, 0);
        other_ref = other.astype(dtype).broadcast_to_shape(&shape, 0);
        out_ref = Tensor::with_device_type(dtype, &shape, self_t.device_type());
    }

    let mut params = ScalarBinaryOpParams {
        op_type,
        dtype,
        loop_size: self_ref.shape_at(-1),
        offsets: [0, 0, 0],
        strides: [
            out_ref.stride_at(-1) * out_ref.itemsize(),
            self_ref.stride_at(-1) * self_ref.itemsize(),
            other_ref.stride_at(-1) * other_ref.itemsize(),
        ],
    };

    do_binary_op_tensor_tensor_impl(&self_ref, &other_ref, &out_ref, 0, &mut params);
    *ret = out_ref;
}

pub fn binary_op_tensor_scalar(op_type: ScalarBinaryOpType, self_t: &Tensor, other: &Scalar, ret: &mut Tensor) {
    let other_tensor = other.to_tensor();
    binary_op_tensor_tensor(op_type, self_t, &other_tensor, ret);
}

pub fn binary_op_tensor_scalar_reverse(op_type: ScalarBinaryOpType, other: &Scalar, self_t: &Tensor, ret: &mut Tensor) {
    let other_tensor = other.to_tensor();
    binary_op_tensor_tensor(op_type, &other_tensor, self_t, ret);
}

pub fn add(self_t: &Tensor, other: &Tensor) -> Tensor {
    let mut ret = Tensor::new();
    binary_op_tensor_tensor(ScalarBinaryOpType::Add, self_t, other, &mut ret);
    ret
}

pub fn sub(self_t: &Tensor, other: &Tensor) -> Tensor {
    let mut ret = Tensor::new();
    binary_op_tensor_tensor(ScalarBinaryOpType::Sub, self_t, other, &mut ret);
    ret
}

pub fn mul(self_t: &Tensor, other: &Tensor) -> Tensor {
    let mut ret = Tensor::new();
    binary_op_tensor_tensor(ScalarBinaryOpType::Mul, self_t, other, &mut ret);
    ret
}

pub fn truediv(self_t: &Tensor, other: &Tensor) -> Tensor {
    let mut ret = Tensor::new();
    binary_op_tensor_tensor(ScalarBinaryOpType::Div, self_t, other, &mut ret);
    ret
}

pub fn add_scalar(self_t: &Tensor, other: &Scalar) -> Tensor {
    let mut ret = Tensor::new();
    binary_op_tensor_scalar(ScalarBinaryOpType::Add, self_t, other, &mut ret);
    ret
}

pub fn sub_scalar(self_t: &Tensor, other: &Scalar) -> Tensor {
    let mut ret = Tensor::new();
    binary_op_tensor_scalar(ScalarBinaryOpType::Sub, self_t, other, &mut ret);
    ret
}

pub fn mul_scalar(self_t: &Tensor, other: &Scalar) -> Tensor {
    let mut ret = Tensor::new();
    binary_op_tensor_scalar(ScalarBinaryOpType::Mul, self_t, other, &mut ret);
    ret
}

pub fn truediv_scalar(self_t: &Tensor, other: &Scalar) -> Tensor {
    let mut ret = Tensor::new();
    binary_op_tensor_scalar(ScalarBinaryOpType::Div, self_t, other, &mut ret);
    ret
}

pub fn add_scalar_reverse(other: &Scalar, self_t: &Tensor) -> Tensor {
    let mut ret = Tensor::new();
    binary_op_tensor_scalar_reverse(ScalarBinaryOpType::Add, other, self_t, &mut ret);
    ret
}

pub fn sub_scalar_reverse(other: &Scalar, self_t: &Tensor) -> Tensor {
    let mut ret = Tensor::new();
    binary_op_tensor_scalar_reverse(ScalarBinaryOpType::Sub, other, self_t, &mut ret);
    ret
}

pub fn mul_scalar_reverse(other: &Scalar, self_t: &Tensor) -> Tensor {
    let mut ret = Tensor::new();
    binary_op_tensor_scalar_reverse(ScalarBinaryOpType::Mul, other, self_t, &mut ret);
    ret
}

pub fn truediv_scalar_reverse(other: &Scalar, self_t: &Tensor) -> Tensor {
    let mut ret = Tensor::new();
    binary_op_tensor_scalar_reverse(ScalarBinaryOpType::Div, other, self_t, &mut ret);
    ret
}
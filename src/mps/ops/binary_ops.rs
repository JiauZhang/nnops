use crate::tensor::Tensor;
use crate::scalar::Scalar;
use crate::data_type::DataType;
use crate::device::DeviceType;
use crate::mps;

fn try_gpu_binary_op(
    kernel_name: &str,
    self_t: &Tensor,
    other: &Tensor,
) -> Option<Tensor> {
    if self_t.dtype() != DataType::Float32
        || other.dtype() != DataType::Float32
        || !self_t.is_contiguous()
        || !other.is_contiguous()
        || self_t.shape() != other.shape()
    {
        return None;
    }

    let a_buf = mps::tensor_metal_buffer(self_t)?;
    let b_buf = mps::tensor_metal_buffer(other)?;

    let out = Tensor::with_device_type(DataType::Float32, self_t.shape(), DeviceType::Mps);
    let out_buf = mps::tensor_metal_buffer(&out)?;

    let nelems = self_t.nelems() as u64;

    mps::with_context(|ctx| {
        if let Some(pso) = ctx.kernels.get(kernel_name) {
            crate::mps::kernel::dispatch_1d(
                &ctx.queue,
                pso,
                &[a_buf, b_buf, out_buf],
                nelems,
            );
        }
    });

    Some(out)
}

pub fn add(self_t: &Tensor, other: &Tensor) -> Tensor {
    if self_t.device_type() == DeviceType::Mps {
        if let Some(result) = try_gpu_binary_op("add_f32", self_t, other) {
            return result;
        }
    }
    crate::cpu::ops::binary_ops::add(self_t, other)
}

pub fn sub(self_t: &Tensor, other: &Tensor) -> Tensor {
    if self_t.device_type() == DeviceType::Mps {
        if let Some(result) = try_gpu_binary_op("sub_f32", self_t, other) {
            return result;
        }
    }
    crate::cpu::ops::binary_ops::sub(self_t, other)
}

pub fn mul(self_t: &Tensor, other: &Tensor) -> Tensor {
    if self_t.device_type() == DeviceType::Mps {
        if let Some(result) = try_gpu_binary_op("mul_f32", self_t, other) {
            return result;
        }
    }
    crate::cpu::ops::binary_ops::mul(self_t, other)
}

pub fn truediv(self_t: &Tensor, other: &Tensor) -> Tensor {
    if self_t.device_type() == DeviceType::Mps {
        if let Some(result) = try_gpu_binary_op("div_f32", self_t, other) {
            return result;
        }
    }
    crate::cpu::ops::binary_ops::truediv(self_t, other)
}

fn scalar_to_tensor(s: &Scalar) -> Tensor {
    let t = s.to_tensor();
    if t.dtype() != DataType::Float32 {
        t.astype(DataType::Float32)
    } else {
        t
    }
}

pub fn add_scalar(self_t: &Tensor, other: &Scalar) -> Tensor {
    if self_t.device_type() == DeviceType::Mps && self_t.dtype() == DataType::Float32 {
        let other_t = scalar_to_tensor(other);
        let other_br = other_t.broadcast_to_shape(self_t.shape(), 0);
        if let Some(result) = try_gpu_binary_op("add_f32", self_t, &other_br) {
            return result;
        }
    }
    crate::cpu::ops::binary_ops::add_scalar(self_t, other)
}

pub fn sub_scalar(self_t: &Tensor, other: &Scalar) -> Tensor {
    if self_t.device_type() == DeviceType::Mps && self_t.dtype() == DataType::Float32 {
        let other_t = scalar_to_tensor(other);
        let other_br = other_t.broadcast_to_shape(self_t.shape(), 0);
        if let Some(result) = try_gpu_binary_op("sub_f32", self_t, &other_br) {
            return result;
        }
    }
    crate::cpu::ops::binary_ops::sub_scalar(self_t, other)
}

pub fn mul_scalar(self_t: &Tensor, other: &Scalar) -> Tensor {
    if self_t.device_type() == DeviceType::Mps && self_t.dtype() == DataType::Float32 {
        let other_t = scalar_to_tensor(other);
        let other_br = other_t.broadcast_to_shape(self_t.shape(), 0);
        if let Some(result) = try_gpu_binary_op("mul_f32", self_t, &other_br) {
            return result;
        }
    }
    crate::cpu::ops::binary_ops::mul_scalar(self_t, other)
}

pub fn truediv_scalar(self_t: &Tensor, other: &Scalar) -> Tensor {
    if self_t.device_type() == DeviceType::Mps && self_t.dtype() == DataType::Float32 {
        let other_t = scalar_to_tensor(other);
        let other_br = other_t.broadcast_to_shape(self_t.shape(), 0);
        if let Some(result) = try_gpu_binary_op("div_f32", self_t, &other_br) {
            return result;
        }
    }
    crate::cpu::ops::binary_ops::truediv_scalar(self_t, other)
}

pub fn add_scalar_reverse(other: &Scalar, self_t: &Tensor) -> Tensor {
    if self_t.device_type() == DeviceType::Mps && self_t.dtype() == DataType::Float32 {
        let other_t = scalar_to_tensor(other);
        let other_br = other_t.broadcast_to_shape(self_t.shape(), 0);
        if let Some(result) = try_gpu_binary_op("add_f32", &other_br, self_t) {
            return result;
        }
    }
    crate::cpu::ops::binary_ops::add_scalar_reverse(other, self_t)
}

pub fn sub_scalar_reverse(other: &Scalar, self_t: &Tensor) -> Tensor {
    if self_t.device_type() == DeviceType::Mps && self_t.dtype() == DataType::Float32 {
        let other_t = scalar_to_tensor(other);
        let other_br = other_t.broadcast_to_shape(self_t.shape(), 0);
        if let Some(result) = try_gpu_binary_op("sub_f32", &other_br, self_t) {
            return result;
        }
    }
    crate::cpu::ops::binary_ops::sub_scalar_reverse(other, self_t)
}

pub fn mul_scalar_reverse(other: &Scalar, self_t: &Tensor) -> Tensor {
    if self_t.device_type() == DeviceType::Mps && self_t.dtype() == DataType::Float32 {
        let other_t = scalar_to_tensor(other);
        let other_br = other_t.broadcast_to_shape(self_t.shape(), 0);
        if let Some(result) = try_gpu_binary_op("mul_f32", &other_br, self_t) {
            return result;
        }
    }
    crate::cpu::ops::binary_ops::mul_scalar_reverse(other, self_t)
}

pub fn truediv_scalar_reverse(other: &Scalar, self_t: &Tensor) -> Tensor {
    if self_t.device_type() == DeviceType::Mps && self_t.dtype() == DataType::Float32 {
        let other_t = scalar_to_tensor(other);
        let other_br = other_t.broadcast_to_shape(self_t.shape(), 0);
        if let Some(result) = try_gpu_binary_op("div_f32", &other_br, self_t) {
            return result;
        }
    }
    crate::cpu::ops::binary_ops::truediv_scalar_reverse(other, self_t)
}
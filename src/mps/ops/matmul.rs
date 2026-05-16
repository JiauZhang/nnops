use crate::tensor::Tensor;
use crate::data_type::DataType;
use crate::device::DeviceType;
use crate::mps;

pub fn matmul(lvalue: &Tensor, rvalue: &Tensor) -> Tensor {
    if lvalue.device_type() != DeviceType::MPS
        || lvalue.dtype() != DataType::Float32
        || rvalue.dtype() != DataType::Float32
    {
        return crate::cpu::ops::matmul::matmul(lvalue, rvalue);
    }

    let a_buf = match mps::tensor_metal_buffer(lvalue) {
        Some(buf) => buf,
        None => return crate::cpu::ops::matmul::matmul(lvalue, rvalue),
    };
    let b_buf = match mps::tensor_metal_buffer(rvalue) {
        Some(buf) => buf,
        None => return crate::cpu::ops::matmul::matmul(lvalue, rvalue),
    };

    let m = lvalue.shape_at(-2) as u32;
    let k = lvalue.shape_at(-1) as u32;
    let n = rvalue.shape_at(-1) as u32;

    let out = Tensor::with_device_type(DataType::Float32, &vec![m as i64, n as i64], DeviceType::MPS);
    let out_buf = match mps::tensor_metal_buffer(&out) {
        Some(buf) => buf,
        None => return crate::cpu::ops::matmul::matmul(lvalue, rvalue),
    };

    mps::with_context(|ctx| {
        if let Some(pso) = ctx.kernels.get("matmul_f32") {
            crate::mps::kernel::dispatch_2d(
                &ctx.queue,
                pso,
                &[a_buf, b_buf, out_buf],
                &[(m, 3), (n, 4), (k, 5)],
                m as u64,
                n as u64,
            );
        }
    });

    out
}
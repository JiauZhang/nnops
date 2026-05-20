use crate::tensor::Tensor;
use crate::data_type::DataType;
use crate::device::DeviceType;
use crate::mps;

fn matmul_kernel_name(dtype: DataType) -> Option<&'static str> {
    match dtype {
        DataType::Float32 => Some("matmul_f32"),
        DataType::Int32 => Some("matmul_i32"),
        DataType::Uint32 => Some("matmul_u32"),
        DataType::Int16 => Some("matmul_i16"),
        DataType::Uint16 => Some("matmul_u16"),
        DataType::Int8 => Some("matmul_i8"),
        DataType::Uint8 => Some("matmul_u8"),
        _ => None,
    }
}

pub fn matmul(lvalue: &Tensor, rvalue: &Tensor) -> Tensor {
    if lvalue.device_type() != DeviceType::Mps {
        return crate::cpu::ops::matmul::matmul(lvalue, rvalue);
    }

    if lvalue.dtype() != rvalue.dtype() {
        return crate::cpu::ops::matmul::matmul(lvalue, rvalue);
    }

    let kernel_name = match matmul_kernel_name(lvalue.dtype()) {
        Some(name) => name,
        None => panic!(
            "MPS matmul does not support dtype {:?}",
            lvalue.dtype()
        ),
    };

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

    let out = Tensor::with_device_type(lvalue.dtype(), &vec![m as i64, n as i64], DeviceType::Mps);
    let out_buf = match mps::tensor_metal_buffer(&out) {
        Some(buf) => buf,
        None => return crate::cpu::ops::matmul::matmul(lvalue, rvalue),
    };

    mps::with_context(|ctx| {
        if let Some(pso) = ctx.kernels.get(kernel_name) {
            crate::mps::kernel::dispatch_2d(
                &ctx.queue,
                pso,
                &[a_buf, b_buf, out_buf],
                &[m, n, k],
                m as u64,
                n as u64,
            );
        }
    });

    out
}
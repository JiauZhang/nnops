use std::time::Duration;

use nnops::{DataType, DeviceType, Tensor};

pub fn tensor_f32(shape: &[i64], device: DeviceType) -> Tensor {
    let mut t = Tensor::with_device_type(DataType::Float32, &shape.to_vec(), device);
    t.fill_with(|i| (i % 997) as f64);
    t
}

pub const BINARY_SIZES: &[&[i64]] = &[
    &[16, 16],
    &[128, 128],
    &[512, 512],
    &[1024, 1024],
    &[4096, 4096],
];

pub const SCALAR_SIZES: &[&[i64]] = &[&[512, 512], &[1024, 1024]];

pub const MATMUL_SIZES: &[i64] = &[16, 32, 64, 128, 256, 512, 1024];

pub const MATMUL_RECT_SHAPES: &[(&[i64], &[i64])] = &[
    (&[128, 256], &[256, 512]),
    (&[512, 128], &[128, 256]),
    (&[1, 1024], &[1024, 512]),
    (&[32, 512], &[512, 1]),
];

pub const BINARY_MEASUREMENT: Duration = Duration::from_secs(5);
pub const BINARY_WARMUP: Duration = Duration::from_secs(1);

pub const MATMUL_MEASUREMENT: Duration = Duration::from_secs(10);
pub const MATMUL_WARMUP: Duration = Duration::from_secs(2);

pub const BINARY_SAMPLE_SIZE: usize = 50;
pub const MATMUL_SAMPLE_SIZE: usize = 30;
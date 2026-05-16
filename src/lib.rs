#[macro_use]
mod common;
pub mod data_type;
pub mod device;
pub mod tensor;
pub mod scalar;
pub mod random;
pub mod cpu;

#[cfg(feature = "mps")]
pub mod mps;

pub use common::{TensorShape, TensorStride};
pub use data_type::DataType;
pub use device::DeviceType;
pub use scalar::Scalar;
pub use tensor::Tensor;
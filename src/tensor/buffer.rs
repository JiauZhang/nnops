use crate::device::{Device, DeviceType};
use std::cell::UnsafeCell;

pub enum TensorBuffer {
    Cpu(UnsafeCell<Vec<u8>>),
    #[cfg(feature = "mps")]
    Mps(crate::mps::MpsBuffer),
}

impl TensorBuffer {
    pub fn new(device: &Device, size: usize) -> Self {
        match device {
            Device::Cpu(_) => TensorBuffer::Cpu(UnsafeCell::new(vec![0u8; size])),
            Device::Mps(_) => {
                #[cfg(feature = "mps")]
                if crate::mps::is_available() {
                    if let Some(buf) = crate::mps::MpsBuffer::new(size) {
                        return TensorBuffer::Mps(buf);
                    }
                }
                panic!("MPS device is not available on this system. Build with `--features mps` and ensure Metal is supported.");
            }
            Device::Cuda(_) => panic!("CUDA device is not available. Build with `--features cuda`."),
        }
    }

    pub fn device_type(&self) -> DeviceType {
        match self {
            TensorBuffer::Cpu(_) => DeviceType::Cpu,
            #[cfg(feature = "mps")]
            TensorBuffer::Mps(_) => DeviceType::Mps,
        }
    }

    pub fn data_ptr(&self) -> *const u8 {
        match self {
            TensorBuffer::Cpu(data) => unsafe { (*data.get()).as_ptr() },
            #[cfg(feature = "mps")]
            TensorBuffer::Mps(buf) => buf.data_ptr(),
        }
    }

    pub fn data_mut_ptr(&mut self) -> *mut u8 {
        match self {
            TensorBuffer::Cpu(data) => unsafe { (*data.get()).as_mut_ptr() },
            #[cfg(feature = "mps")]
            TensorBuffer::Mps(buf) => buf.data_mut_ptr(),
        }
    }

    pub fn size(&self) -> usize {
        match self {
            TensorBuffer::Cpu(data) => unsafe { (*data.get()).len() },
            #[cfg(feature = "mps")]
            TensorBuffer::Mps(buf) => buf.size,
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        match self {
            TensorBuffer::Cpu(data) => unsafe { &*data.get() },
            #[cfg(feature = "mps")]
            TensorBuffer::Mps(buf) => unsafe { std::slice::from_raw_parts(buf.data_ptr(), buf.size) },
        }
    }

    pub fn copy_from_cpu(&self, src: &[u8]) {
        match self {
            TensorBuffer::Cpu(data) => {
                let data = unsafe { &mut *data.get() };
                let len = src.len().min(data.len());
                data[..len].copy_from_slice(&src[..len]);
            }
            #[cfg(feature = "mps")]
            TensorBuffer::Mps(buf) => {
                buf.copy_from_cpu(src);
            }
        }
    }

    pub fn copy_to_cpu(&self, dst: &mut [u8]) {
        match self {
            TensorBuffer::Cpu(data) => {
                let data = unsafe { &*data.get() };
                let len = dst.len().min(data.len());
                dst[..len].copy_from_slice(&data[..len]);
            }
            #[cfg(feature = "mps")]
            TensorBuffer::Mps(buf) => {
                buf.copy_to_cpu(dst);
            }
        }
    }
}

unsafe impl Send for TensorBuffer {}
unsafe impl Sync for TensorBuffer {}
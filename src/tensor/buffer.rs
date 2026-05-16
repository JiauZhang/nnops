use crate::device::{Device, DeviceType};

pub enum TensorBuffer {
    Cpu(Vec<u8>),
    #[cfg(feature = "mps")]
    Mps(crate::mps::MpsBuffer),
}

impl TensorBuffer {
    pub fn new(device: &Device, size: usize) -> Self {
        match device {
            Device::Cpu(_) => TensorBuffer::Cpu(vec![0u8; size]),
            #[cfg(feature = "mps")]
            Device::MPS(_) => {
                crate::mps::MpsBuffer::new(size)
                    .map(TensorBuffer::Mps)
                    .unwrap_or_else(|| TensorBuffer::Cpu(vec![0u8; size]))
            }
            _ => TensorBuffer::Cpu(vec![0u8; size]),
        }
    }

    pub fn device_type(&self) -> DeviceType {
        match self {
            TensorBuffer::Cpu(_) => DeviceType::Cpu,
            #[cfg(feature = "mps")]
            TensorBuffer::Mps(_) => DeviceType::MPS,
        }
    }

    pub fn data_ptr(&self) -> *const u8 {
        match self {
            TensorBuffer::Cpu(data) => data.as_ptr(),
            #[cfg(feature = "mps")]
            TensorBuffer::Mps(buf) => buf.data_ptr(),
        }
    }

    pub fn data_mut_ptr(&mut self) -> *mut u8 {
        match self {
            TensorBuffer::Cpu(data) => data.as_mut_ptr(),
            #[cfg(feature = "mps")]
            TensorBuffer::Mps(buf) => buf.data_mut_ptr(),
        }
    }

    pub fn size(&self) -> usize {
        match self {
            TensorBuffer::Cpu(data) => data.len(),
            #[cfg(feature = "mps")]
            TensorBuffer::Mps(buf) => buf.size,
        }
    }

    pub fn copy_from_cpu(&self, src: *const u8, size: usize) {
        match self {
            TensorBuffer::Cpu(data) => {
                let dst = data.as_ptr() as *mut u8;
                unsafe {
                    std::ptr::copy_nonoverlapping(src, dst, size.min(data.len()));
                }
            }
            #[cfg(feature = "mps")]
            TensorBuffer::Mps(buf) => {
                buf.copy_from_cpu(src, size);
            }
        }
    }

    pub fn copy_to_cpu(&self, dst: *mut u8, size: usize) {
        match self {
            TensorBuffer::Cpu(data) => {
                let src = data.as_ptr();
                unsafe {
                    std::ptr::copy_nonoverlapping(src, dst, size.min(data.len()));
                }
            }
            #[cfg(feature = "mps")]
            TensorBuffer::Mps(buf) => {
                buf.copy_to_cpu(dst, size);
            }
        }
    }
}

unsafe impl Send for TensorBuffer {}
unsafe impl Sync for TensorBuffer {}
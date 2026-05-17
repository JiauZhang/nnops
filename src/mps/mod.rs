pub mod ops;
pub mod shaders;
pub mod kernel;

use crate::tensor::Tensor;
use crate::tensor::TensorBuffer;
use std::sync::LazyLock;

pub fn tensor_metal_buffer(tensor: &Tensor) -> Option<&metal::Buffer> {
    match &*tensor.buffer {
        TensorBuffer::Mps(buf) => Some(&buf.metal_buffer),
        _ => None,
    }
}

pub struct MpsContext {
    pub device: metal::Device,
    pub queue: metal::CommandQueue,
    pub kernels: kernel::KernelCache,
}

impl MpsContext {
    pub fn new() -> Option<Self> {
        let device = metal::Device::system_default()?;
        let queue = device.new_command_queue();
        let library = kernel::compile_library(&device)?;
        let kernels = kernel::KernelCache::new(&device, &library);
        Some(MpsContext { device, queue, kernels })
    }
}

static MPS_CONTEXT: LazyLock<Option<MpsContext>> = LazyLock::new(MpsContext::new);

pub fn is_available() -> bool {
    MPS_CONTEXT.is_some()
}

pub fn with_context<F, R>(f: F) -> Option<R>
where
    F: FnOnce(&MpsContext) -> R,
{
    MPS_CONTEXT.as_ref().map(f)
}

pub struct MpsBuffer {
    pub(crate) metal_buffer: metal::Buffer,
    pub(crate) size: usize,
}

impl MpsBuffer {
    pub fn new(size: usize) -> Option<Self> {
        with_context(|ctx| {
            let metal_buffer = ctx.device.new_buffer(
                size as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );
            MpsBuffer { metal_buffer, size }
        })
    }

    pub fn from_data(data: &[u8]) -> Option<Self> {
        with_context(|ctx| {
            let metal_buffer = ctx.device.new_buffer_with_data(
                data.as_ptr() as *const std::ffi::c_void,
                data.len() as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );
            MpsBuffer { metal_buffer, size: data.len() }
        })
    }

    pub fn data_ptr(&self) -> *const u8 {
        self.metal_buffer.contents() as *const u8
    }

    pub fn data_mut_ptr(&self) -> *mut u8 {
        self.metal_buffer.contents() as *mut u8
    }

    pub fn copy_from_cpu(&self, src: &[u8]) {
        let dst = self.data_mut_ptr();
        let len = src.len().min(self.size);
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), dst, len);
        }
        self.metal_buffer.did_modify_range(metal::NSRange::new(0, len as u64));
    }

    pub fn copy_to_cpu(&self, dst: &mut [u8]) {
        let src = self.data_ptr();
        let len = dst.len().min(self.size);
        unsafe {
            std::ptr::copy_nonoverlapping(src, dst.as_mut_ptr(), len);
        }
    }
}

unsafe impl Send for MpsBuffer {}
unsafe impl Sync for MpsBuffer {}
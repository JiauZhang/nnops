#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceType {
    Cpu = 0,
    Cuda,
    Npu,
    MPS,
}

#[derive(Debug, Clone)]
pub enum Device {
    Cpu(CpuDevice),
    Cuda(CudaDevice),
    Npu(NpuDevice),
    MPS(MpsDevice),
}

impl Device {
    pub fn from_type(device_type: DeviceType) -> Self {
        match device_type {
            DeviceType::Cpu => Device::Cpu(CpuDevice),
            DeviceType::Cuda => Device::Cuda(CudaDevice),
            DeviceType::Npu => Device::Npu(NpuDevice),
            DeviceType::MPS => Device::MPS(MpsDevice),
        }
    }

    pub fn malloc(&self, size: usize) -> *mut u8 {
        match self {
            Device::Cpu(d) => d.malloc(size),
            Device::Cuda(d) => d.malloc(size),
            Device::Npu(d) => d.malloc(size),
            Device::MPS(d) => d.malloc(size),
        }
    }

    pub fn free(&self, ptr: *mut u8) {
        match self {
            Device::Cpu(d) => d.free(ptr),
            Device::Cuda(d) => d.free(ptr),
            Device::Npu(d) => d.free(ptr),
            Device::MPS(d) => d.free(ptr),
        }
    }

    pub fn copy_to_cpu(&self, src: *const u8, dst: *mut u8, size: usize) {
        match self {
            Device::Cpu(d) => d.copy_to_cpu(src, dst, size),
            Device::Cuda(d) => d.copy_to_cpu(src, dst, size),
            Device::Npu(d) => d.copy_to_cpu(src, dst, size),
            Device::MPS(d) => d.copy_to_cpu(src, dst, size),
        }
    }

    pub fn copy_from_cpu(&self, src: *const u8, dst: *mut u8, size: usize) {
        match self {
            Device::Cpu(d) => d.copy_from_cpu(src, dst, size),
            Device::Cuda(d) => d.copy_from_cpu(src, dst, size),
            Device::Npu(d) => d.copy_from_cpu(src, dst, size),
            Device::MPS(d) => d.copy_from_cpu(src, dst, size),
        }
    }

    pub fn device_type(&self) -> DeviceType {
        match self {
            Device::Cpu(_) => DeviceType::Cpu,
            Device::Cuda(_) => DeviceType::Cuda,
            Device::Npu(_) => DeviceType::Npu,
            Device::MPS(_) => DeviceType::MPS,
        }
    }

    pub fn device_name(&self) -> &str {
        match self {
            Device::Cpu(_) => "cpu",
            Device::Cuda(_) => "cuda",
            Device::Npu(_) => "npu",
            Device::MPS(_) => "mps",
        }
    }
}

#[derive(Debug, Clone)]
pub struct CpuDevice;

impl CpuDevice {
    pub fn malloc(&self, size: usize) -> *mut u8 {
        let mut v = vec![0u8; size];
        let ptr = v.as_mut_ptr();
        std::mem::forget(v);
        ptr
    }

    pub fn free(&self, ptr: *mut u8) {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, 0);
        }
    }

    pub fn copy_to_cpu(&self, src: *const u8, dst: *mut u8, size: usize) {
        unsafe {
            std::ptr::copy_nonoverlapping(src, dst, size);
        }
    }

    pub fn copy_from_cpu(&self, src: *const u8, dst: *mut u8, size: usize) {
        unsafe {
            std::ptr::copy_nonoverlapping(src, dst, size);
        }
    }
}

#[derive(Debug, Clone)]
pub struct CudaDevice;

impl CudaDevice {
    pub fn malloc(&self, _size: usize) -> *mut u8 {
        unimplemented!("CudaDevice::malloc")
    }

    pub fn free(&self, _ptr: *mut u8) {
        unimplemented!("CudaDevice::free")
    }

    pub fn copy_to_cpu(&self, _src: *const u8, _dst: *mut u8, _size: usize) {
        unimplemented!("CudaDevice::copy_to_cpu")
    }

    pub fn copy_from_cpu(&self, _src: *const u8, _dst: *mut u8, _size: usize) {
        unimplemented!("CudaDevice::copy_from_cpu")
    }
}

#[derive(Debug, Clone)]
pub struct NpuDevice;

impl NpuDevice {
    pub fn malloc(&self, _size: usize) -> *mut u8 {
        unimplemented!("NpuDevice::malloc")
    }

    pub fn free(&self, _ptr: *mut u8) {
        unimplemented!("NpuDevice::free")
    }

    pub fn copy_to_cpu(&self, _src: *const u8, _dst: *mut u8, _size: usize) {
        unimplemented!("NpuDevice::copy_to_cpu")
    }

    pub fn copy_from_cpu(&self, _src: *const u8, _dst: *mut u8, _size: usize) {
        unimplemented!("NpuDevice::copy_from_cpu")
    }
}

#[derive(Debug, Clone)]
pub struct MpsDevice;

impl MpsDevice {
    pub fn malloc(&self, size: usize) -> *mut u8 {
        #[cfg(feature = "mps")]
        if crate::mps::is_available() {
            return crate::mps::with_context(|ctx| {
                let buffer = ctx.device.new_buffer(
                    size as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                let ptr = buffer.contents() as *mut u8;
                std::mem::forget(buffer);
                ptr
            }).unwrap_or_else(|| {
                let mut v = vec![0u8; size];
                let ptr = v.as_mut_ptr();
                std::mem::forget(v);
                ptr
            });
        }
        let mut v = vec![0u8; size];
        let ptr = v.as_mut_ptr();
        std::mem::forget(v);
        ptr
    }

    pub fn free(&self, _ptr: *mut u8) {
    }

    pub fn copy_to_cpu(&self, src: *const u8, dst: *mut u8, size: usize) {
        unsafe {
            std::ptr::copy_nonoverlapping(src, dst, size);
        }
    }

    pub fn copy_from_cpu(&self, src: *const u8, dst: *mut u8, size: usize) {
        unsafe {
            std::ptr::copy_nonoverlapping(src, dst, size);
        }
    }
}
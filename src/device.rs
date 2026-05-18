#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceType {
    Cpu = 0,
    Cuda,
    Npu,
    MPS,
}

impl DeviceType {
    pub fn is_available(&self) -> bool {
        match self {
            DeviceType::Cpu => true,
            DeviceType::Cuda => false,
            DeviceType::Npu => false,
            DeviceType::MPS => {
                #[cfg(feature = "mps")]
                {
                    crate::mps::is_available()
                }
                #[cfg(not(feature = "mps"))]
                {
                    false
                }
            }
        }
    }
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
}

#[derive(Debug, Clone)]
pub struct CudaDevice;
#[derive(Debug, Clone)]
pub struct NpuDevice;

#[derive(Debug, Clone)]
pub struct MpsDevice;
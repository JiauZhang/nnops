#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceType {
    Cpu = 0,
    Cuda,
    Mps,
}

impl DeviceType {
    pub fn is_available(&self) -> bool {
        match self {
            DeviceType::Cpu => true,
            DeviceType::Cuda => false,
            DeviceType::Mps => {
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
    Mps(MpsDevice),
}

impl Device {
    pub fn from_type(device_type: DeviceType) -> Self {
        match device_type {
            DeviceType::Cpu => Device::Cpu(CpuDevice),
            DeviceType::Cuda => Device::Cuda(CudaDevice),
            DeviceType::Mps => Device::Mps(MpsDevice),
        }
    }

    pub fn device_type(&self) -> DeviceType {
        match self {
            Device::Cpu(_) => DeviceType::Cpu,
            Device::Cuda(_) => DeviceType::Cuda,
            Device::Mps(_) => DeviceType::Mps,
        }
    }

    pub fn device_name(&self) -> &str {
        match self {
            Device::Cpu(_) => "cpu",
            Device::Cuda(_) => "cuda",
            Device::Mps(_) => "mps",
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
pub struct MpsDevice;
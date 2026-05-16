use pyo3::prelude::*;

#[pyclass(name = "DeviceType", eq, eq_int)]
#[derive(Clone, PartialEq)]
pub enum PyDeviceType {
    CPU = 0,
    CUDA = 1,
    NPU = 2,
    MPS = 3,
}

impl From<nnops::DeviceType> for PyDeviceType {
    fn from(dt: nnops::DeviceType) -> Self {
        match dt {
            nnops::DeviceType::Cpu => PyDeviceType::CPU,
            nnops::DeviceType::Cuda => PyDeviceType::CUDA,
            nnops::DeviceType::Npu => PyDeviceType::NPU,
            nnops::DeviceType::MPS => PyDeviceType::MPS,
        }
    }
}

impl From<PyDeviceType> for nnops::DeviceType {
    fn from(dt: PyDeviceType) -> Self {
        match dt {
            PyDeviceType::CPU => nnops::DeviceType::Cpu,
            PyDeviceType::CUDA => nnops::DeviceType::Cuda,
            PyDeviceType::NPU => nnops::DeviceType::Npu,
            PyDeviceType::MPS => nnops::DeviceType::MPS,
        }
    }
}

#[pymethods]
impl PyDeviceType {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        match name.to_lowercase().as_str() {
            "cpu" => Ok(PyDeviceType::CPU),
            "cuda" => Ok(PyDeviceType::CUDA),
            "npu" => Ok(PyDeviceType::NPU),
            "mps" => Ok(PyDeviceType::MPS),
            _ => Err(pyo3::exceptions::PyRuntimeError::new_err(
                format!("unknown device type: {}", name)
            )),
        }
    }

    fn __str__(&self) -> String {
        match self {
            PyDeviceType::CPU => "cpu".to_string(),
            PyDeviceType::CUDA => "cuda".to_string(),
            PyDeviceType::NPU => "npu".to_string(),
            PyDeviceType::MPS => "mps".to_string(),
        }
    }

    fn __repr__(&self) -> String {
        format!("<DeviceType.{}>", match self {
            PyDeviceType::CPU => "CPU",
            PyDeviceType::CUDA => "CUDA",
            PyDeviceType::NPU => "NPU",
            PyDeviceType::MPS => "MPS",
        })
    }
}

#[pyfunction]
fn is_device_available(device: &Bound<'_, PyAny>) -> PyResult<bool> {
    let py_dt = device.extract::<PyDeviceType>()?;
    match py_dt {
        PyDeviceType::CPU => Ok(true),
        PyDeviceType::CUDA => Ok(false),
        PyDeviceType::NPU => Ok(false),
        PyDeviceType::MPS => {
            #[cfg(feature = "mps")]
            {
                return Ok(nnops::mps::is_available());
            }
            #[cfg(not(feature = "mps"))]
            {
                Ok(false)
            }
        }
    }
}

pub fn register_device_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDeviceType>()?;
    m.add("CPU", PyDeviceType::CPU)?;
    m.add("CUDA", PyDeviceType::CUDA)?;
    m.add("NPU", PyDeviceType::NPU)?;
    m.add("MPS", PyDeviceType::MPS)?;
    m.add_function(wrap_pyfunction!(is_device_available, m)?)?;
    Ok(())
}
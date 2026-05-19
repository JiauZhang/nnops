use pyo3::prelude::*;

#[pyclass(name = "DeviceType", eq, eq_int)]
#[derive(Clone, PartialEq)]
pub enum PyDeviceType {
    Cpu = 0,
    Cuda = 1,
    Mps = 2,
}

impl From<nnops::DeviceType> for PyDeviceType {
    fn from(dt: nnops::DeviceType) -> Self {
        match dt {
            nnops::DeviceType::Cpu => PyDeviceType::Cpu,
            nnops::DeviceType::Cuda => PyDeviceType::Cuda,
            nnops::DeviceType::Mps => PyDeviceType::Mps,
        }
    }
}

impl From<PyDeviceType> for nnops::DeviceType {
    fn from(dt: PyDeviceType) -> Self {
        match dt {
            PyDeviceType::Cpu => nnops::DeviceType::Cpu,
            PyDeviceType::Cuda => nnops::DeviceType::Cuda,
            PyDeviceType::Mps => nnops::DeviceType::Mps,
        }
    }
}

#[pymethods]
impl PyDeviceType {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        match name.to_lowercase().as_str() {
            "cpu" => Ok(PyDeviceType::Cpu),
            "cuda" => Ok(PyDeviceType::Cuda),
            "mps" => Ok(PyDeviceType::Mps),
            _ => Err(pyo3::exceptions::PyRuntimeError::new_err(
                format!("unknown device type: {}", name)
            )),
        }
    }

    fn __str__(&self) -> String {
        match self {
            PyDeviceType::Cpu => "cpu".to_string(),
            PyDeviceType::Cuda => "cuda".to_string(),
            PyDeviceType::Mps => "mps".to_string(),
        }
    }

    fn __repr__(&self) -> String {
        format!("<DeviceType.{}>", match self {
            PyDeviceType::Cpu => "CPU",
            PyDeviceType::Cuda => "CUDA",
            PyDeviceType::Mps => "MPS",
        })
    }

    fn is_available(&self) -> bool {
        match self {
            PyDeviceType::Cpu => true,
            PyDeviceType::Cuda => false,
            PyDeviceType::Mps => {
                #[cfg(feature = "mps")]
                {
                    nnops::mps::is_available()
                }
                #[cfg(not(feature = "mps"))]
                {
                    false
                }
            }
        }
    }
}

#[pyfunction]
fn is_device_available(device: &Bound<'_, PyAny>) -> PyResult<bool> {
    let py_dt = device.extract::<PyDeviceType>()?;
    match py_dt {
        PyDeviceType::Cpu => Ok(true),
        PyDeviceType::Cuda => Ok(false),
        PyDeviceType::Mps => {
            #[cfg(feature = "mps")]
            {
                Ok(nnops::mps::is_available())
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
    m.add("CPU", PyDeviceType::Cpu)?;
    m.add("CUDA", PyDeviceType::Cuda)?;
    m.add("MPS", PyDeviceType::Mps)?;
    m.add_function(wrap_pyfunction!(is_device_available, m)?)?;
    Ok(())
}
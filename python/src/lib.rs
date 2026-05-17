use pyo3::prelude::*;
use pyo3::types::PyTuple;
use pyo3::exceptions::PyTypeError;

mod dtype;
mod device;
mod tensor;

use dtype::register_dtype_module;
use device::{PyDeviceType, register_device_module};
use tensor::{PyTensor, register_tensor_module};

#[pyfunction]
#[pyo3(signature = (*shape))]
fn randn(_py: Python<'_>, shape: &Bound<'_, PyTuple>) -> PyResult<PyTensor> {
    if shape.is_empty() {
        return Err(PyTypeError::new_err("shape is required"));
    }

    let mut dims = Vec::with_capacity(shape.len());
    for item in shape.iter() {
        dims.push(item.extract::<i64>()?);
    }

    let mut tensor = nnops::Tensor::with_device_type(nnops::DataType::Float32, &dims, nnops::DeviceType::Cpu);
    let nelems = tensor.nelems();
    let mut data = vec![0.0f32; nelems];
    let rng = nnops::random::RandN::new(0.0, 1.0);
    rng.sample(&mut data);

    if let Some(slice) = tensor.as_slice_mut::<f32>() {
        let copy_len = slice.len().min(data.len());
        slice[..copy_len].copy_from_slice(&data[..copy_len]);
    }

    Ok(PyTensor { inner: tensor })
}

#[pyfunction]
#[pyo3(signature = (input, weight, bias=None))]
fn linear(input: &PyTensor, weight: &PyTensor, bias: Option<&PyTensor>) -> PyTensor {
    let result = match bias {
        Some(b) => nnops::cpu::ops::unary_ops::linear(&input.inner, &weight.inner, Some(&b.inner)),
        None => nnops::cpu::ops::unary_ops::linear(&input.inner, &weight.inner, None),
    };
    PyTensor { inner: result }
}

#[pyfunction]
fn show_device_info(device: &Bound<'_, PyAny>) -> PyResult<String> {
    let name = if let Ok(py_dt) = device.extract::<PyDeviceType>() {
        match py_dt {
            PyDeviceType::CPU => "CPU".to_string(),
            PyDeviceType::CUDA => "CUDA".to_string(),
            PyDeviceType::NPU => "NPU".to_string(),
            PyDeviceType::MPS => "MPS".to_string(),
        }
    } else if let Ok(s) = device.extract::<String>() {
        s
    } else {
        return Err(PyTypeError::new_err("unsupported device type"));
    };

    Ok(format!("Device: {} (not implemented)", name))
}

#[pymodule]
fn _rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    register_dtype_module(m)?;
    register_device_module(m)?;
    register_tensor_module(m)?;

    m.add_function(wrap_pyfunction!(randn, m)?)?;
    m.add_function(wrap_pyfunction!(linear, m)?)?;
    m.add_function(wrap_pyfunction!(show_device_info, m)?)?;

    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}
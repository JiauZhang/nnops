use pyo3::prelude::*;
use pyo3::types::{PyTuple, PySlice, PyBytes};
use pyo3::exceptions::{PyTypeError, PyIndexError, PyValueError, PyRuntimeError};

use nnops::{Tensor, Scalar, DataType, DeviceType, TensorShape};

use crate::dtype::PyDataType;
use crate::device::PyDeviceType;

fn py_to_scalar(val: &Bound<'_, PyAny>) -> PyResult<Scalar> {
    if let Ok(v) = val.extract::<bool>() {
        Ok(Scalar::Bool(v))
    } else if let Ok(v) = val.extract::<i64>() {
        Ok(Scalar::Int64(v))
    } else if let Ok(v) = val.extract::<f64>() {
        Ok(Scalar::Float64(v))
    } else {
        Err(PyTypeError::new_err(format!(
            "unsupported scalar type: {}", val.get_type().name()?
        )))
    }
}

fn extract_slice(py_slice: &Bound<'_, PySlice>, dim_size: i64) -> PyResult<(i64, i64, i64)> {
    let start_attr = py_slice.getattr("start")?;
    let stop_attr = py_slice.getattr("stop")?;
    let step_attr = py_slice.getattr("step")?;

    let step: i64 = if step_attr.is_none() {
        1
    } else {
        let s = step_attr.extract::<i64>()?;
        if s == 0 {
            return Err(PyValueError::new_err("slice step cannot be zero"));
        }
        s
    };

    let start: i64 = if start_attr.is_none() {
        if step > 0 { 0 } else { dim_size - 1 }
    } else {
        let mut s = start_attr.extract::<i64>()?;
        if s < 0 { s += dim_size; }
        if step > 0 {
            s.clamp(0, dim_size - 1)
        } else {
            if s < 0 { 0 } else if s >= dim_size { dim_size - 1 } else { s }
        }
    };

    let stop: i64 = if stop_attr.is_none() {
        if step > 0 { dim_size } else { -1 }
    } else {
        let mut s = stop_attr.extract::<i64>()?;
        if s < 0 { s += dim_size; }
        if step > 0 {
            s.clamp(0, dim_size)
        } else if step < 0 {
            s.clamp(-1, dim_size - 1)
        } else {
            s
        }
    };

    Ok((start, stop, step))
}

fn is_ellipsis(obj: &Bound<'_, PyAny>) -> bool {
    obj.get_type().name().map_or(false, |n| n == "ellipsis")
}

fn indexing(meta: &mut nnops::tensor::TensorMeta, indices: &Bound<'_, PyAny>, axis: usize) -> PyResult<()> {
    if let Ok(tuple) = indices.downcast::<PyTuple>() {
        let len = tuple.len();
        let mut ellipsis_idx: Option<usize> = None;
        for i in 0..len {
            if is_ellipsis(&tuple.get_item(i)?) {
                if ellipsis_idx.is_some() {
                    return Err(PyIndexError::new_err("an index can only have a single ellipsis ('...')"));
                }
                ellipsis_idx = Some(i);
            }
        }

        let mut axis = axis;
        for i in 0..len {
            if let Some(ei) = ellipsis_idx {
                if i == ei {
                    if meta.ndim() + i >= len - 1 {
                        axis = meta.ndim() + i - len + 1;
                    }
                    continue;
                }
            }
            let item = tuple.get_item(i)?;
            indexing(meta, &item, axis)?;
            if item.downcast::<PySlice>().is_ok() {
                axis += 1;
            }
        }
    } else if let Ok(py_slice) = indices.downcast::<PySlice>() {
        let dim_size = meta.dims[axis];
        let (start, stop, step) = extract_slice(py_slice, dim_size)?;
        meta.slice_inplace(&nnops::tensor::Slice::new_full(start, stop, step), axis);
    } else if let Ok(idx) = indices.extract::<i64>() {
        meta.index_inplace(idx, axis);
    } else if is_ellipsis(indices) {
        // do nothing
    } else {
        return Err(PyTypeError::new_err(format!(
            "not supported indexing type: {}", indices.get_type().name()?
        )));
    }
    Ok(())
}

fn parse_tensor_shape(shape: &Bound<'_, PyAny>) -> PyResult<TensorShape> {
    if let Ok(tuple) = shape.downcast::<PyTuple>() {
        let mut result = Vec::with_capacity(tuple.len());
        for item in tuple.iter() {
            result.push(item.extract::<i64>()?);
        }
        Ok(result)
    } else if let Ok(list) = shape.downcast::<pyo3::types::PyList>() {
        let mut result = Vec::with_capacity(list.len());
        for item in list.iter() {
            result.push(item.extract::<i64>()?);
        }
        Ok(result)
    } else {
        Err(PyTypeError::new_err("Only list or tuple is supported for Tensor shape"))
    }
}

fn parse_int_args(args: &Bound<'_, PyTuple>) -> PyResult<TensorShape> {
    let mut indices = Vec::with_capacity(args.len());
    for item in args.iter() {
        indices.push(item.extract::<i64>()?);
    }
    Ok(indices)
}

#[pyclass(name = "Tensor")]
pub struct PyTensor {
    pub inner: Tensor,
}

#[pymethods]
impl PyTensor {
    #[new]
    #[pyo3(signature = (shape=None, dtype=None, device=None))]
    fn new(
        shape: Option<Bound<'_, PyAny>>,
        dtype: Option<Bound<'_, PyAny>>,
        device: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let dtype: DataType = match dtype {
            Some(ref val) => {
                if let Ok(py_dt) = val.extract::<PyDataType>() {
                    py_dt.into()
                } else {
                    return Err(PyTypeError::new_err("unsupported dtype"));
                }
            }
            None => DataType::Float32,
        };

        let device_type: DeviceType = match device {
            Some(ref val) => {
                if let Ok(py_dt) = val.extract::<PyDeviceType>() {
                    py_dt.into()
                } else if let Ok(name) = val.extract::<String>() {
                    match name.to_lowercase().as_str() {
                        "cpu" => DeviceType::Cpu,
                        "cuda" => DeviceType::Cuda,
                        "npu" => DeviceType::Npu,
                        "mps" => DeviceType::MPS,
                        _ => return Err(PyRuntimeError::new_err(
                            format!("unknown device type: {}", name)
                        )),
                    }
                } else {
                    return Err(PyRuntimeError::new_err("unsupported device type"));
                }
            }
            None => DeviceType::Cpu,
        };

        let shape = match shape {
            Some(ref s) => parse_tensor_shape(s)?,
            None => return Err(PyTypeError::new_err("shape is required")),
        };

        let tensor = Tensor::with_device_type(dtype, &shape, device_type);
        Ok(PyTensor { inner: tensor })
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn __repr__(&self) -> String {
        self.inner.to_repr()
    }

    fn __getitem__(&self, indices: Bound<'_, PyAny>) -> PyResult<PyTensor> {
        let mut meta = self.inner.meta().clone();
        indexing(&mut meta, &indices, 0)?;
        let tensor = Tensor::with_meta_buffer(meta, self.inner.buffer());
        Ok(PyTensor { inner: tensor })
    }

    fn __setitem__(&self, indices: Bound<'_, PyAny>, value: Bound<'_, PyAny>) -> PyResult<()> {
        let mut meta = self.inner.meta().clone();
        indexing(&mut meta, &indices, 0)?;
        let mut sub_tensor = Tensor::with_meta_buffer(meta, self.inner.buffer());

        if let Ok(py_tensor) = value.extract::<PyRef<'_, PyTensor>>() {
            sub_tensor.fill_from(&py_tensor.inner);
        } else {
            let scalar = py_to_scalar(&value)?;
            let scalar_tensor = scalar.to_tensor();
            sub_tensor.fill_from(&scalar_tensor);
        }
        Ok(())
    }

    fn is_contiguous(&self) -> bool {
        self.inner.is_contiguous()
    }

    fn contiguous(&self) -> PyTensor {
        PyTensor { inner: self.inner.contiguous() }
    }

    fn clone(&self) -> PyTensor {
        PyTensor { inner: self.inner.clone_tensor() }
    }

    fn astype(&self, dtype: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        let dt: DataType = if let Ok(py_dt) = dtype.extract::<PyDataType>() {
            py_dt.into()
        } else {
            return Err(PyTypeError::new_err("unsupported dtype"));
        };
        Ok(PyTensor { inner: self.inner.astype(dt) })
    }

    fn to(&self, device: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        let device_type: DeviceType = if let Ok(py_dt) = device.extract::<PyDeviceType>() {
            py_dt.into()
        } else if let Ok(name) = device.extract::<String>() {
            match name.to_lowercase().as_str() {
                "cpu" => DeviceType::Cpu,
                "cuda" => DeviceType::Cuda,
                "npu" => DeviceType::Npu,
                "mps" => DeviceType::MPS,
                _ => return Err(PyRuntimeError::new_err(
                    format!("unknown device type: {}", name)
                )),
            }
        } else {
            return Err(PyRuntimeError::new_err("unsupported device type"));
        };
        Ok(PyTensor { inner: self.inner.to_device(device_type) })
    }

    fn numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let np = py.import_bound("numpy")?;

        let dtype_map: &[(DataType, &str)] = &[
            (DataType::Bool, "bool"),
            (DataType::Uint8, "uint8"),
            (DataType::Int8, "int8"),
            (DataType::Uint16, "uint16"),
            (DataType::Int16, "int16"),
            (DataType::Uint32, "uint32"),
            (DataType::Int32, "int32"),
            (DataType::Uint64, "uint64"),
            (DataType::Int64, "int64"),
            (DataType::Float32, "float32"),
            (DataType::Float64, "float64"),
        ];

        let tensor = self.inner.contiguous();
        let nbytes = tensor.nbytes();
        let data = unsafe {
            std::slice::from_raw_parts(tensor.data_ptr(), nbytes)
        };
        let py_bytes = PyBytes::new_bound(py, data);

        let np_dtype_str = dtype_map.iter()
            .find(|(dt, _)| *dt == tensor.dtype())
            .map(|(_, name)| *name)
            .unwrap_or("float32");

        let np_dtype = np.getattr(np_dtype_str)?;
        let arr = np.call_method1("frombuffer", (py_bytes, np_dtype))?;

        let shape: Vec<usize> = tensor.shape().iter().map(|&s| s as usize).collect();
        let arr = arr.call_method1("reshape", (shape,))?;
        let arr = arr.call_method0("copy")?;

        Ok(arr)
    }

    #[staticmethod]
    fn from_numpy<'py>(array: Bound<'py, PyAny>) -> PyResult<PyTensor> {
        let _np = array.py().import_bound("numpy")?;
        let ndim = array.getattr("ndim")?.extract::<usize>()?;
        let np_shape = array.getattr("shape")?;
        let shape: TensorShape = (0..ndim).map(|i| {
            np_shape.get_item(i).and_then(|v| v.extract::<i64>())
        }).collect::<PyResult<Vec<i64>>>()?;

        let np_dtype = array.getattr("dtype")?;
        let dtype_name = np_dtype.getattr("name")?.extract::<String>()?;

        let dtype = match dtype_name.as_str() {
            "bool" => DataType::Bool,
            "uint8" => DataType::Uint8,
            "int8" => DataType::Int8,
            "uint16" => DataType::Uint16,
            "int16" => DataType::Int16,
            "uint32" => DataType::Uint32,
            "int32" => DataType::Int32,
            "uint64" => DataType::Uint64,
            "int64" => DataType::Int64,
            "float32" | "float" => DataType::Float32,
            "float64" | "double" => DataType::Float64,
            _ => return Err(PyTypeError::new_err(format!("unsupported numpy dtype: {}", dtype_name))),
        };

        let tensor = Tensor::with_device_type(dtype, &shape, DeviceType::Cpu);
        let nbytes = tensor.nbytes();
        let flat_bytes = array.call_method1("tobytes", ())?;
        let flat_bytes = flat_bytes.extract::<Vec<u8>>()?;

        unsafe {
            std::ptr::copy_nonoverlapping(
                flat_bytes.as_ptr(),
                tensor.data_ptr() as *mut u8,
                nbytes.min(flat_bytes.len()),
            );
        }

        Ok(PyTensor { inner: tensor })
    }

    #[pyo3(signature = (*args))]
    fn reshape(&self, args: &Bound<'_, PyTuple>) -> PyResult<PyTensor> {
        let indices = parse_int_args(args)?;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.inner.reshape(&mut indices.clone())
        }));
        match result {
            Ok(tensor) => Ok(PyTensor { inner: tensor }),
            Err(_) => Err(PyRuntimeError::new_err(
                format!("cannot reshape tensor of shape {:?} into shape {:?}", self.inner.shape(), indices)
            )),
        }
    }

    #[pyo3(signature = (*args))]
    fn permute(&self, args: &Bound<'_, PyTuple>) -> PyResult<PyTensor> {
        let indices = parse_int_args(args)?;
        Ok(PyTensor { inner: self.inner.permute(&indices) })
    }

    fn transpose(&self, dim0: i64, dim1: i64) -> PyTensor {
        PyTensor { inner: self.inner.transpose(dim0, dim1) }
    }

    fn broadcast_to(&self, shape: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        let shape = parse_tensor_shape(shape)?;
        Ok(PyTensor { inner: self.inner.broadcast_to_shape(&shape, 0) })
    }

    #[getter]
    fn dtype(&self) -> PyDataType {
        self.inner.dtype().into()
    }

    #[getter]
    fn device(&self) -> PyDeviceType {
        self.inner.device_type().into()
    }

    #[getter]
    fn data_ptr(&self) -> u64 {
        self.inner.data_ptr() as u64
    }

    #[getter]
    fn ref_count(&self) -> usize {
        self.inner.ref_count()
    }

    #[getter]
    fn ndim(&self) -> usize {
        self.inner.ndim()
    }

    #[getter]
    fn nbytes(&self) -> usize {
        self.inner.nbytes()
    }

    #[getter]
    fn nelems(&self) -> usize {
        self.inner.nelems()
    }

    #[getter]
    fn stride(&self) -> Vec<i64> {
        self.inner.stride().clone()
    }

    #[getter]
    fn shape(&self) -> Vec<i64> {
        self.inner.shape().clone()
    }

    fn __matmul__(&self, other: &PyTensor) -> PyTensor {
        PyTensor { inner: nnops::cpu::ops::matmul::matmul(&self.inner, &other.inner) }
    }

    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        if let Ok(py_tensor) = other.extract::<PyRef<'_, PyTensor>>() {
            Ok(PyTensor { inner: nnops::cpu::ops::binary_ops::add(&self.inner, &py_tensor.inner) })
        } else {
            let scalar = py_to_scalar(other)?;
            Ok(PyTensor { inner: nnops::cpu::ops::binary_ops::add_scalar(&self.inner, &scalar) })
        }
    }

    fn __iadd__(&mut self, other: &Bound<'_, PyAny>) -> PyResult<()> {
        let result = if let Ok(py_tensor) = other.extract::<PyRef<'_, PyTensor>>() {
            nnops::cpu::ops::binary_ops::add(&self.inner, &py_tensor.inner)
        } else {
            let scalar = py_to_scalar(other)?;
            let scalar = scalar.astype(self.inner.dtype());
            nnops::cpu::ops::binary_ops::add_scalar(&self.inner, &scalar)
        };
        self.inner = result;
        Ok(())
    }

    fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        self.__add__(other)
    }

    fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        if let Ok(py_tensor) = other.extract::<PyRef<'_, PyTensor>>() {
            Ok(PyTensor { inner: nnops::cpu::ops::binary_ops::sub(&self.inner, &py_tensor.inner) })
        } else {
            let scalar = py_to_scalar(other)?;
            Ok(PyTensor { inner: nnops::cpu::ops::binary_ops::sub_scalar(&self.inner, &scalar) })
        }
    }

    fn __isub__(&mut self, other: &Bound<'_, PyAny>) -> PyResult<()> {
        let result = if let Ok(py_tensor) = other.extract::<PyRef<'_, PyTensor>>() {
            nnops::cpu::ops::binary_ops::sub(&self.inner, &py_tensor.inner)
        } else {
            let scalar = py_to_scalar(other)?;
            let scalar = scalar.astype(self.inner.dtype());
            nnops::cpu::ops::binary_ops::sub_scalar(&self.inner, &scalar)
        };
        self.inner = result;
        Ok(())
    }

    fn __rsub__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        let scalar = py_to_scalar(other)?;
        Ok(PyTensor { inner: nnops::cpu::ops::binary_ops::sub_scalar_reverse(&scalar, &self.inner) })
    }

    fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        if let Ok(py_tensor) = other.extract::<PyRef<'_, PyTensor>>() {
            Ok(PyTensor { inner: nnops::cpu::ops::binary_ops::mul(&self.inner, &py_tensor.inner) })
        } else {
            let scalar = py_to_scalar(other)?;
            Ok(PyTensor { inner: nnops::cpu::ops::binary_ops::mul_scalar(&self.inner, &scalar) })
        }
    }

    fn __imul__(&mut self, other: &Bound<'_, PyAny>) -> PyResult<()> {
        let result = if let Ok(py_tensor) = other.extract::<PyRef<'_, PyTensor>>() {
            nnops::cpu::ops::binary_ops::mul(&self.inner, &py_tensor.inner)
        } else {
            let scalar = py_to_scalar(other)?;
            let scalar = scalar.astype(self.inner.dtype());
            nnops::cpu::ops::binary_ops::mul_scalar(&self.inner, &scalar)
        };
        self.inner = result;
        Ok(())
    }

    fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        self.__mul__(other)
    }

    fn __truediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        if let Ok(py_tensor) = other.extract::<PyRef<'_, PyTensor>>() {
            Ok(PyTensor { inner: nnops::cpu::ops::binary_ops::truediv(&self.inner, &py_tensor.inner) })
        } else {
            let scalar = py_to_scalar(other)?;
            Ok(PyTensor { inner: nnops::cpu::ops::binary_ops::truediv_scalar(&self.inner, &scalar) })
        }
    }

    fn __itruediv__(&mut self, other: &Bound<'_, PyAny>) -> PyResult<()> {
        let result = if let Ok(py_tensor) = other.extract::<PyRef<'_, PyTensor>>() {
            nnops::cpu::ops::binary_ops::truediv(&self.inner, &py_tensor.inner)
        } else {
            let scalar = py_to_scalar(other)?;
            let scalar = scalar.astype(self.inner.dtype());
            nnops::cpu::ops::binary_ops::truediv_scalar(&self.inner, &scalar)
        };
        self.inner = result;
        Ok(())
    }

    fn __rtruediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        let scalar = py_to_scalar(other)?;
        Ok(PyTensor { inner: nnops::cpu::ops::binary_ops::truediv_scalar_reverse(&scalar, &self.inner) })
    }
}

pub fn register_tensor_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTensor>()?;

    m.add_function(wrap_pyfunction!(tensor_from_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_is_broadcastable, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_broadcast_shape, m)?)?;

    m.add_function(wrap_pyfunction!(ops_add, m)?)?;
    m.add_function(wrap_pyfunction!(ops_sub, m)?)?;
    m.add_function(wrap_pyfunction!(ops_mul, m)?)?;
    m.add_function(wrap_pyfunction!(ops_truediv, m)?)?;
    m.add_function(wrap_pyfunction!(ops_iadd, m)?)?;
    m.add_function(wrap_pyfunction!(ops_isub, m)?)?;
    m.add_function(wrap_pyfunction!(ops_imul, m)?)?;
    m.add_function(wrap_pyfunction!(ops_itruediv, m)?)?;
    m.add_function(wrap_pyfunction!(ops_matmul, m)?)?;

    Ok(())
}

#[pyfunction]
#[pyo3(name = "from_numpy")]
fn tensor_from_numpy<'py>(array: Bound<'py, PyAny>) -> PyResult<PyTensor> {
    PyTensor::from_numpy(array)
}

#[pyfunction]
#[pyo3(name = "is_broadcastable")]
fn tensor_is_broadcastable(t1: &PyTensor, t2: &PyTensor) -> bool {
    Tensor::is_broadcastable_tensors(&t1.inner, &t2.inner)
}

#[pyfunction]
#[pyo3(name = "broadcast_shape")]
fn tensor_broadcast_shape(t1: &PyTensor, t2: &PyTensor) -> Vec<i64> {
    Tensor::broadcast_shape_tensors(&t1.inner, &t2.inner)
}

#[pyfunction]
#[pyo3(name = "add")]
fn ops_add(a: &PyTensor, b: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
    a.__add__(b)
}

#[pyfunction]
#[pyo3(name = "sub")]
fn ops_sub(a: &PyTensor, b: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
    a.__sub__(b)
}

#[pyfunction]
#[pyo3(name = "mul")]
fn ops_mul(a: &PyTensor, b: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
    a.__mul__(b)
}

#[pyfunction]
#[pyo3(name = "truediv")]
fn ops_truediv(a: &PyTensor, b: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
    a.__truediv__(b)
}

#[pyfunction]
#[pyo3(name = "iadd")]
fn ops_iadd(a: &mut PyTensor, b: &Bound<'_, PyAny>) -> PyResult<()> {
    a.__iadd__(b)
}

#[pyfunction]
#[pyo3(name = "isub")]
fn ops_isub(a: &mut PyTensor, b: &Bound<'_, PyAny>) -> PyResult<()> {
    a.__isub__(b)
}

#[pyfunction]
#[pyo3(name = "imul")]
fn ops_imul(a: &mut PyTensor, b: &Bound<'_, PyAny>) -> PyResult<()> {
    a.__imul__(b)
}

#[pyfunction]
#[pyo3(name = "itruediv")]
fn ops_itruediv(a: &mut PyTensor, b: &Bound<'_, PyAny>) -> PyResult<()> {
    a.__itruediv__(b)
}

#[pyfunction]
#[pyo3(name = "matmul")]
fn ops_matmul(a: &PyTensor, b: &PyTensor) -> PyTensor {
    PyTensor { inner: nnops::cpu::ops::matmul::matmul(&a.inner, &b.inner) }
}
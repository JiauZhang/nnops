use pyo3::prelude::*;

#[pyclass(name = "DataType", eq, eq_int)]
#[derive(Debug, Clone, PartialEq)]
pub enum PyDataType {
    Bool = 0,
    Uint8,
    Int8,
    Uint16,
    Int16,
    Uint32,
    Int32,
    Uint64,
    Int64,
    Float32,
    Float64,
}

impl From<nnops::DataType> for PyDataType {
    fn from(dt: nnops::DataType) -> Self {
        match dt {
            nnops::DataType::Bool => PyDataType::Bool,
            nnops::DataType::Uint8 => PyDataType::Uint8,
            nnops::DataType::Int8 => PyDataType::Int8,
            nnops::DataType::Uint16 => PyDataType::Uint16,
            nnops::DataType::Int16 => PyDataType::Int16,
            nnops::DataType::Uint32 => PyDataType::Uint32,
            nnops::DataType::Int32 => PyDataType::Int32,
            nnops::DataType::Uint64 => PyDataType::Uint64,
            nnops::DataType::Int64 => PyDataType::Int64,
            nnops::DataType::Float32 => PyDataType::Float32,
            nnops::DataType::Float64 => PyDataType::Float64,
        }
    }
}

impl From<PyDataType> for nnops::DataType {
    fn from(dt: PyDataType) -> Self {
        match dt {
            PyDataType::Bool => nnops::DataType::Bool,
            PyDataType::Uint8 => nnops::DataType::Uint8,
            PyDataType::Int8 => nnops::DataType::Int8,
            PyDataType::Uint16 => nnops::DataType::Uint16,
            PyDataType::Int16 => nnops::DataType::Int16,
            PyDataType::Uint32 => nnops::DataType::Uint32,
            PyDataType::Int32 => nnops::DataType::Int32,
            PyDataType::Uint64 => nnops::DataType::Uint64,
            PyDataType::Int64 => nnops::DataType::Int64,
            PyDataType::Float32 => nnops::DataType::Float32,
            PyDataType::Float64 => nnops::DataType::Float64,
        }
    }
}

#[pymethods]
impl PyDataType {
    fn __str__(&self) -> String {
        format!("{:?}", self)
    }

    fn __repr__(&self) -> String {
        format!("<DataType.{}>", match self {
            PyDataType::Bool => "BOOL",
            PyDataType::Uint8 => "UINT8",
            PyDataType::Int8 => "INT8",
            PyDataType::Uint16 => "UINT16",
            PyDataType::Int16 => "INT16",
            PyDataType::Uint32 => "UINT32",
            PyDataType::Int32 => "INT32",
            PyDataType::Uint64 => "UINT64",
            PyDataType::Int64 => "INT64",
            PyDataType::Float32 => "FLOAT32",
            PyDataType::Float64 => "FLOAT64",
        })
    }
}

pub fn register_dtype_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDataType>()?;
    m.add("bool", PyDataType::Bool)?;
    m.add("uint8", PyDataType::Uint8)?;
    m.add("int8", PyDataType::Int8)?;
    m.add("uint16", PyDataType::Uint16)?;
    m.add("int16", PyDataType::Int16)?;
    m.add("uint32", PyDataType::Uint32)?;
    m.add("int32", PyDataType::Int32)?;
    m.add("uint64", PyDataType::Uint64)?;
    m.add("int64", PyDataType::Int64)?;
    m.add("float32", PyDataType::Float32)?;
    m.add("float64", PyDataType::Float64)?;
    Ok(())
}
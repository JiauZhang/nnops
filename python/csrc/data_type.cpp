#include <nanobind/nanobind.h>
#include <nnops/data_type.h>

namespace nb = nanobind;

namespace pynnops {

void DEFINE_DATA_TYPE_MODULE(nb::module_ & (m)) {
    nb::enum_<nnops::DataType>(m, "DataType")
        .value("float64", nnops::DataType::TYPE_FLOAT64)
        .value("float32", nnops::DataType::TYPE_FLOAT32)
        .value("int64", nnops::DataType::TYPE_INT64)
        .value("uint64", nnops::DataType::TYPE_UINT64)
        .value("int32", nnops::DataType::TYPE_INT32)
        .value("uint32", nnops::DataType::TYPE_UINT32)
        .value("int16", nnops::DataType::TYPE_INT16)
        .value("uint16", nnops::DataType::TYPE_UINT16)
        .value("int8", nnops::DataType::TYPE_INT8)
        .value("uint8", nnops::DataType::TYPE_UINT8)
        .value("bool", nnops::DataType::TYPE_BOOL)
        .export_values();
}

} // namespace pynnops
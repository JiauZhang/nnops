#include <nanobind/nanobind.h>
#include <nnops/data_type.h>

namespace nb = nanobind;

void DEFINE_DATA_TYPE_MODULE(nb::module_ & (m)) {
    nb::enum_<DataType>(m, "DataType")
        .value("float32", DataType::TYPE_FLOAT32)
        .value("int32", DataType::TYPE_INT32)
        .value("uint32", DataType::TYPE_UINT32)
        .value("int16", DataType::TYPE_INT16)
        .value("uint16", DataType::TYPE_UINT16)
        .value("int8", DataType::TYPE_INT8)
        .value("uint8", DataType::TYPE_UINT8)
        .export_values();
}
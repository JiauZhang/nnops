#include <nanobind/nanobind.h>
#include <data_type.h>
#include <map>

using namespace std;
namespace nb = nanobind;

static map<DataType, size_t> dtype_size = {
    {DataType::TYPE_FLOAT32, 4},
    {DataType::TYPE_INT32, 4},
    {DataType::TYPE_UINT32, 4},
    {DataType::TYPE_FLOAT16, 2},
    {DataType::TYPE_INT16, 2},
    {DataType::TYPE_UINT16, 2},
    {DataType::TYPE_INT8, 1},
    {DataType::TYPE_UINT8, 1},
};

size_t sizeof_dtype(DataType dtype) {
    return dtype_size[dtype];
}

void DEFINE_DATA_TYPE_MODULE(nb::module_ & (m)) {
    nb::enum_<DataType>(m, "DataType")
        .value("float32", DataType::TYPE_FLOAT32)
        .value("float16", DataType::TYPE_FLOAT16)
        .value("int32", DataType::TYPE_INT32)
        .value("uint32", DataType::TYPE_UINT32)
        .value("int16", DataType::TYPE_INT16)
        .value("uint16", DataType::TYPE_UINT16)
        .value("int8", DataType::TYPE_INT8)
        .value("uint8", DataType::TYPE_UINT8)
        .export_values();
}
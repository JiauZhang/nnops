#include <nanobind/nanobind.h>
#include "data_type.h"

using namespace std;
namespace nb = nanobind;

void DEFINE_DATA_TYPE_MODULE(nb::module_ & (m)) {
    nb::class_<DataType> dtype(m, "DataType");

    dtype.def(nb::init<>())
        .def(nb::init<DataType::Type>())
        .def(nb::init<DataType &>())
        .def_ro("type", &DataType::type_);

    nb::enum_<DataType::Type>(dtype, "Type")
        .value("float32", DataType::Type::TYPE_FLOAT32)
        .value("float16", DataType::Type::TYPE_FLOAT16)
        .value("int32", DataType::Type::TYPE_INT32)
        .value("int16", DataType::Type::TYPE_INT16)
        .value("uint16", DataType::Type::TYPE_UINT16)
        .value("int8", DataType::Type::TYPE_INT8)
        .value("uint8", DataType::Type::TYPE_UINT8)
        .export_values();
}
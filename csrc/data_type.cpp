#include <nanobind/nanobind.h>
#include "data_type.h"

using namespace std;
namespace nb = nanobind;

void DEFINE_DATA_TYPE_MODULE(nb::module_ & (m)) {
    nb::class_<DataType>(m, "DataType")
        .def(nb::init<>())
        .def(nb::init<DataType &>());
}
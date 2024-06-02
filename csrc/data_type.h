#ifndef __DATA_TYPE_H__
#define __DATA_TYPE_H__

#include <nanobind/nanobind.h>

using namespace std;
namespace nb = nanobind;

enum DataType {
    TYPE_FLOAT32,
    TYPE_FLOAT16,
    TYPE_INT32,
    TYPE_UINT32,
    TYPE_INT16,
    TYPE_UINT16,
    TYPE_INT8,
    TYPE_UINT8,
};

void DEFINE_DATA_TYPE_MODULE(nb::module_ & (m));

#endif // __DATA_TYPE_H__
#ifndef __DATA_TYPE_H__
#define __DATA_TYPE_H__

#include <nanobind/nanobind.h>

using namespace std;
namespace nb = nanobind;

class DataType {
public:
    enum Type {
        TYPE_FLOAT32,
        TYPE_FLOAT16,
        TYPE_INT32,
        TYPE_INT16,
        TYPE_UINT16,
        TYPE_INT8,
        TYPE_UINT8,
    };

    DataType() {}
    DataType(DataType &dtype) { type_ = dtype.type_; }
    DataType(Type type) { type_ = type; }

    enum Type type_;
};

void DEFINE_DATA_TYPE_MODULE(nb::module_ & (m));

#endif // __DATA_TYPE_H__
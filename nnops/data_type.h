#ifndef __DATA_TYPE_H__
#define __DATA_TYPE_H__

#include <cstddef>

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

size_t sizeof_dtype(DataType dtype);

#endif // __DATA_TYPE_H__
#include <data_type.h>
#include <map>

using namespace std;

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

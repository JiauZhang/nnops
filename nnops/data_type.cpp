#include <nnops/data_type.h>
#include <unordered_map>

using namespace std;

#define DTYPE_SIZE_ITEM(dtype) { dtype, sizeof_dtype<dtype>() }

static unordered_map<DataType, size_t> dtype_size = {
    DTYPE_SIZE_ITEM(DataType::TYPE_FLOAT32),
    DTYPE_SIZE_ITEM(DataType::TYPE_INT32),
    DTYPE_SIZE_ITEM(DataType::TYPE_UINT32),
    DTYPE_SIZE_ITEM(DataType::TYPE_INT16),
    DTYPE_SIZE_ITEM(DataType::TYPE_UINT16),
    DTYPE_SIZE_ITEM(DataType::TYPE_INT8),
    DTYPE_SIZE_ITEM(DataType::TYPE_UINT8),
};

size_t sizeof_dtype(DataType dtype) {
    return dtype_size[dtype];
}

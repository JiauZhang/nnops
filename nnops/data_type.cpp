#include <nnops/data_type.h>
#include <unordered_map>

using namespace std;

#define DATATYPE_TO_TYPE_ITEM(dtype, type) template<> \
    struct datatype_to_type<dtype> { using Type = type; };

DATATYPE_TO_TYPE_ITEM(DataType::TYPE_FLOAT32, float)
DATATYPE_TO_TYPE_ITEM(DataType::TYPE_INT32, int)
DATATYPE_TO_TYPE_ITEM(DataType::TYPE_UINT32, unsigned int)
DATATYPE_TO_TYPE_ITEM(DataType::TYPE_INT16, short)
DATATYPE_TO_TYPE_ITEM(DataType::TYPE_UINT16, unsigned short)
DATATYPE_TO_TYPE_ITEM(DataType::TYPE_INT8, char)
DATATYPE_TO_TYPE_ITEM(DataType::TYPE_UINT8, unsigned char)

template<DataType dtype>
auto *reinterpret(void *ptr) {
    using T = typename datatype_to_type<dtype>::Type;
    return (T *)ptr;
}

template<DataType dtype>
size_t sizeof_dtype() {
    using T = typename datatype_to_type<dtype>::Type;
    return sizeof(T);
}

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

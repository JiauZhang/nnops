#include <nanobind/nanobind.h>
#include "tensor_shape.h"
#include "data_type.h"
#include "tensor.h"

NB_MODULE(_C, m) {
    DEFINE_TENSOR_SHAPE_MODULE(m);
    DEFINE_DATA_TYPE_MODULE(m);
    DEFINE_TENSOR_MODULE(m);
}

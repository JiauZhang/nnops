#include <nanobind/nanobind.h>
#include "tensor_shape.h"

NB_MODULE(_C, m) {
    DEFINE_TENSOR_SHAPE_MODULE(m);
}

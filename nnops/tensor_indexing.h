#ifndef __TENSOR_INDEXING_H__
#define __TENSOR_INDEXING_H__

#include <optional>
#include <nnops/tensor.h>
#include <nnops/tensor_meta.h>

namespace nnops {

class Slice {
public:
    Slice() {}
    Slice(int start): start_(start) {}
    Slice(int start, int stop): start_(start), stop_(stop) {}
    Slice(int start, int stop, int step): start_(start), stop_(stop), step_(step) {}

    std::optional<int> start_, stop_, step_;
};

void slice_inplace(TensorMeta &meta, Slice &slice, int axis);
void index_inplace(TensorMeta &meta, int dim, int axis);
void slice_inplace(Tensor &tensor, Slice &slice, int axis);
void index_inplace(Tensor &tensor, int dim, int axis);

} // namespace nnops

#endif // __TENSOR_INDEXING_H__
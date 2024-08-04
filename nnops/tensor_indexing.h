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
    Slice(int start, int end): start_(start), end_(end) {}
    Slice(int start, int end, int step): start_(start), end_(end), step_(step) {}

    std::optional<int> start_, end_, step_;
};

void slice_inplace(TensorMeta &meta, Slice &slice, int axis);
void index_inplace(TensorMeta &meta, int dim, int axis);
void slice_inplace(Tensor &tensor, Slice &slice, int axis);
void index_inplace(Tensor &tensor, int dim, int axis);

} // namespace nnops

#endif // __TENSOR_INDEXING_H__
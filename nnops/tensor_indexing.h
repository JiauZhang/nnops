#ifndef __TENSOR_INDEXING_H__
#define __TENSOR_INDEXING_H__

#include <optional>
#include <nnops/tensor_meta.h>

namespace nnops {

class Slice {
public:
    std::optional<int> start, end, step;
};

void slice_inplace(TensorMeta &meta, Slice &slice);
void index_inplace(TensorMeta &meta, int dim);

} // namespace nnops

#endif // __TENSOR_INDEXING_H__
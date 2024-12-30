#include <nnops/tensor.h>

namespace nnops {

class Tensor;

namespace cuda {

__global__ void do_clone_impl() {

}

void clone(Tensor &src, Tensor &dst) {

}

} // namespace nnops::cuda

} // namespace nnops
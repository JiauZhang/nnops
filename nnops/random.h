#ifndef __NNOPS_RANDOM_H__
#define __NNOPS_RANDOM_H__

#include <random>
#include <nnops/common.h>

namespace nnops {

class RandN {
public:
    RandN(float mean, float stddev);
    void sample(float *ptr, index_t nelems);

private:
    float mean_, stddev_;
    std::normal_distribution<float> dist_;
};

} // namespace nnops

#endif // __NNOPS_RANDOM_H__
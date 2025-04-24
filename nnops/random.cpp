#include <random>
#include <nnops/random.h>

namespace nnops {

static std::random_device rd{};
static std::mt19937 gen{rd()};

RandN::RandN(float mean, float stddev) : mean_(mean), stddev_(stddev) {
    dist_ = std::normal_distribution<float>(mean, stddev);
}

void RandN::sample(float *ptr, index_t nelems) {
    for (int i = 0; i < nelems; i++)
        ptr[i] = dist_(gen);
}

} // namespace nnops
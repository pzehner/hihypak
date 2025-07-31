#include <omp.h>

#include "omp_extra.hpp"

namespace omp_extra {

std::size_t getNumThreads() {
    std::size_t numThreads = 1;

    #pragma omp parallel
    {
        #pragma omp single
        numThreads = omp_get_num_threads();
    }

    return numThreads;
}

}

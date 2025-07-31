#include <algorithm>
#include <vector>

#include <omp.h>

#include <Kokkos_Core.hpp>

#include "command_line.hpp"
#include "omp_extra.hpp"

/**
 * Here, we want a single multicore CPU to run one parallel loop per OpenMP
 * thread on a single GPU.
 *
 * We compute `numElements` distinct sums, each being the sum of integers from
 * 0 to `numSubElements` times the element ID.
 */
int main(int argc, char * argv[]) {
    Kokkos::ScopeGuard kokkos;

    Kokkos::Profiling::pushRegion("preparation");

    // parameters
    std::size_t const numThreads = omp_extra::getNumThreads();
    std::size_t const numElements = command_line::getInt(argc, argv, 1, numThreads);
    std::size_t const numSubElements = command_line::getInt(argc, argv, 2, 1'000'000);
    std::size_t const numPasses = command_line::getInt(argc, argv, 3, 100);

    Kokkos::printf("numThreads = %d\n", numThreads);
    Kokkos::printf("numElements = %d\n", numElements);
    Kokkos::printf("numSubElements = %d\n", numSubElements);
    Kokkos::printf("numPasses = %d\n", numPasses);

    // create dataset
    Kokkos::View<int *> data("data", numSubElements);
    Kokkos::parallel_for(
        "Fill data", numSubElements,
        KOKKOS_LAMBDA(std::size_t const i) { data(i) = i; });
    Kokkos::fence("Wait for fill data");

    // create vector of sums
    std::vector<long> sums(numElements);

    // create the GPU partition
    std::vector<std::size_t> weights(numThreads);
    std::fill(weights.begin(), weights.end(), 1);
    auto instances = Kokkos::Experimental::partition_space(
        Kokkos::DefaultExecutionSpace{}, weights);

    Kokkos::Profiling::popRegion();

    Kokkos::Profiling::pushRegion("work");

    // start host parallel region
    #pragma omp parallel for
    for (std::size_t element = 0; element < numElements; element++) {
        std::size_t const threadId = omp_get_thread_num();

        // parallel region on GPU
        long sum = 0;
        Kokkos::parallel_reduce(
            "Work on an element",
            Kokkos::RangePolicy(instances[threadId], 0, numSubElements),
            KOKKOS_LAMBDA(std::size_t const i, long &sumLocal) {
                // artificially increase kernel complexity
                long sumCurrent = 0;
                for (std::size_t pass = 0; pass < numPasses; pass++) {
                    sumCurrent += data(i) * element;
                }
                sumCurrent /= numPasses;
                sumLocal += sumCurrent;
            },
            sum);

        // // immediately fence the parallel region on GPU
        // instances[threadId].fence("Wait for work on an element");

        // store sum
        sums[element] = sum;
    }

    Kokkos::Profiling::popRegion();

    Kokkos::Profiling::pushRegion("check");

    bool outcome = true;
    for (std::size_t element = 0; element < numElements; element++) {
        long sumExpected = numSubElements * (numSubElements - 1) * element / 2;
#ifndef NDEBUG
        // debug print
        Kokkos::printf("Element %02d: %d (current: %ld, expected: %ld)\n",
                       element, sums[element] == sumExpected, sums[element],
                       sumExpected);
#else
        // normal print
        Kokkos::printf("Element %02d: %d\n", element,
                       sums[element] == sumExpected);
#endif

        outcome &= sums[element] == sumExpected;
    }

    Kokkos::Profiling::popRegion();

    Kokkos::Profiling::pushRegion("cleanup");

    // fence on all partitions (extraneous)
    for (auto instance : instances) {
        instance.fence();
    }

    Kokkos::Profiling::popRegion();

    return !outcome;
}

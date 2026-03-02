#include <cmath>

#include <vector>

#include <omp.h>

#include <Kokkos_Core.hpp>

#include "command_line.hpp"
#include "devices.hpp"
#include "omp_extra.hpp"

using ViewType = Kokkos::View<int *>;

std::size_t getThreadsPerDevice(std::size_t const numThreads,
                                std::size_t const numDevices) {
  return std::ceil(static_cast<float>(numThreads) /
                   static_cast<float>(numDevices));
}

/**
 * Here, we want a single multicore CPU to run one parallel loop per OpenMP
 * thread on several GPUs.
 *
 * We compute `numElements` distinct sums, each being the sum of integers from
 * 0 to `numSubElements` times the element ID.
 */
int main(int argc, char *argv[]) {
  Kokkos::ScopeGuard kokkos;

  Kokkos::Profiling::pushRegion("preparation");

  // get all GPUs
  devices::Devices devices;
  auto spaces = devices.getSpaces();

  // parameters
  std::size_t const numDevices = devices::Devices::getNumDevices();
  std::size_t const numThreads = omp_extra::getNumThreads();
  std::size_t const threadsPerDevice =
      getThreadsPerDevice(numThreads, numDevices);
  std::size_t const numElements =
      command_line::getInt(argc, argv, 1, numThreads);
  std::size_t const numSubElements =
      command_line::getInt(argc, argv, 2, 1'000'000);
  std::size_t const numPasses = command_line::getInt(argc, argv, 3, 100);

  Kokkos::printf("numDevices = %d\n", numDevices);
  Kokkos::printf("numThreads = %d\n", numThreads);
  Kokkos::printf("threadsPerDevice = %d\n", threadsPerDevice);
  Kokkos::printf("numElements = %d\n", numElements);
  Kokkos::printf("numSubElements = %d\n", numSubElements);
  Kokkos::printf("numPasses = %d\n", numPasses);

  // create views
  std::vector<ViewType> views;
  views.reserve(numDevices);
  for (std::size_t device = 0; device < numDevices; device++) {
    views.emplace_back(
        ViewType(Kokkos::view_alloc("data", spaces[device]), numSubElements));
  }
  // create one view on the first device, then duplicate it on the other devices
  auto viewMain = views[0];
  Kokkos::parallel_for(
      "Fill data", Kokkos::RangePolicy(spaces[0], 0, numSubElements),
      KOKKOS_LAMBDA(std::size_t const i) { viewMain(i) = i; });
  Kokkos::fence("Wait for fill data");
  for (std::size_t device = 1; device < numDevices; device++) {
    Kokkos::deep_copy(views[device], viewMain);
  }

  // create vector of sums
  std::vector<long> sums(numElements);

  // create the GPU partition
  std::vector<Kokkos::DefaultExecutionSpace> instances;
  for (auto const space : spaces) { // i.e. loop on devices
    std::vector<Kokkos::DefaultExecutionSpace> spaceInstances;

    std::vector<std::size_t> weights(threadsPerDevice, 1);
    spaceInstances = Kokkos::Experimental::partition_space(space, weights);

    instances.insert(instances.cend(), spaceInstances.cbegin(),
                     spaceInstances.cend());
  }

  // create the data partition
  std::vector<ViewType> dataSet;
  for (auto const view : views) { // i.e. loop on devices
    std::vector<ViewType> viewInstances(threadsPerDevice, view);
    dataSet.insert(dataSet.cend(), viewInstances.cbegin(),
                   viewInstances.cend());
  }

  Kokkos::Profiling::popRegion();

  Kokkos::Profiling::pushRegion("work");

// start host parallel region
#pragma omp parallel for
  for (std::size_t element = 0; element < numElements; element++) {
    std::size_t const threadId = ::omp_get_thread_num();

    auto instance = instances[threadId];
    auto data = dataSet[threadId];

    // parallel region on GPU
    long sum = 0;
    Kokkos::parallel_reduce(
        "Work on an element", Kokkos::RangePolicy(instance, 0, numSubElements),
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

    // store sum
    sums[element] = sum;
  }

  Kokkos::Profiling::popRegion();

  Kokkos::Profiling::pushRegion("check");

  bool outcome = true;
  for (std::size_t element = 0; element < numElements; element++) {
    long sumExpected = numSubElements * (numSubElements - 1) * element / 2;
    bool isSame = sums[element] == sumExpected;

#ifndef NDEBUG
    // debug print
    Kokkos::printf("Element %02d: %s (current: %ld, expected: %ld)\n", element,
                   isSame ? "ok" : "ng", sums[element], sumExpected);
#else
    // normal print
    Kokkos::printf("Element %02d: %s\n", element, isSame ? "ok" : "ng");
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

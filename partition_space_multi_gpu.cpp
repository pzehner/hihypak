#include <Kokkos_Core.hpp>

int main() {
  Kokkos::ScopeGuard kokkos;

  cudaStream_t stream0;
  cudaSetDevice(0);
  cudaStreamCreate(&stream0);

  cudaStream_t stream1;
  cudaSetDevice(1);
  cudaStreamCreate(&stream1);

  Kokkos::DefaultExecutionSpace space0(stream0);
  Kokkos::DefaultExecutionSpace space1(stream1);

  auto instances0 = Kokkos::Experimental::partition_space(space0, 1, 1);
  auto instances1 = Kokkos::Experimental::partition_space(space1, 1, 1);

  Kokkos::View<int *> view0(Kokkos::view_alloc("view 0", space0), 1);
  Kokkos::View<int *> view1(Kokkos::view_alloc("view 1", space1), 1);

  for (int instance = 0; instance < 2; instance++) {
    Kokkos::parallel_for(
        Kokkos::RangePolicy(instances0[instance], 0, 1),
        KOKKOS_LAMBDA(int const i) { view0(i) = i; });

    Kokkos::parallel_for(
        Kokkos::RangePolicy(instances1[instance], 0, 1),
        KOKKOS_LAMBDA(int const i) { view1(i) = i; });

    Kokkos::fence();
  }

  cudaSetDevice(0);
  cudaStreamDestroy(stream0);

  cudaSetDevice(1);
  cudaStreamDestroy(stream1);
}

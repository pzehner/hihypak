cmake_minimum_required(VERSION 3.28)

project(
    HiHyPaK
    LANGUAGES CXX
)

include(FetchContent)

find_package(OpenMP REQUIRED)

find_package(Kokkos 5.0.2 QUIET)
if(NOT Kokkos_FOUND)
    message(STATUS "Getting Kokkos from the repository")
    FetchContent_Declare(
        kokkos
        GIT_REPOSITORY https://github.com/kokkos/kokkos
        GIT_TAG 5.0.2
    )
    FetchContent_MakeAvailable(kokkos)
endif()

# prevent to use the OpenMP backend
if(Kokkos_ENABLE_OPENMP OR Kokkos_ENABLE_THREADS)
    message(FATAL_ERROR "Cannot enable any Kokkos host backend but the serial one")
endif()

add_library(
    utils
    command_line.cpp
    omp_extra.cpp
)

target_link_libraries(
    utils
    PRIVATE
        OpenMP::OpenMP_CXX
)

add_executable(
    single-gpu
    single_gpu.cpp
)

target_link_libraries(
    single-gpu
    PRIVATE
        Kokkos::kokkos
        OpenMP::OpenMP_CXX
        utils
)

add_executable(
    multi-gpu
    multi_gpu.cpp
)

target_link_libraries(
    multi-gpu
    PRIVATE
        Kokkos::kokkos
        OpenMP::OpenMP_CXX
        utils
)

option(
    HIHYPAK_TEST_PARTITION_SPACE
    "Build a library to test partition space with Kokkos"
)

if(HIHYPAK_TEST_PARTITION_SPACE)
    add_executable(
        partition-space-multi-gpu
        partition_space_multi_gpu.cpp
    )

    target_link_libraries(
        partition-space-multi-gpu
        PRIVATE
            Kokkos::kokkos
    )
endif()

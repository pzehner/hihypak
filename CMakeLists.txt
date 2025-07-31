cmake_minimum_required(VERSION 3.28)

project(
    HiHyP
    LANGUAGES CXX
)

include(FetchContent)

find_package(OpenMP REQUIRED)

find_package(Kokkos 4.6.2 QUIET)
if(NOT Kokkos_FOUND)
    FetchContent_Declare(
        kokkos
        URL https://github.com/kokkos/kokkos/releases/download/4.6.02/kokkos-4.6.02.tar.gz
        URL_HASH SHA256=baf1ebbe67abe2bbb8bb6aed81b4247d53ae98ab8475e516d9c87e87fa2422ce
        DOWNLOAD_EXTRACT_TIMESTAMP ON
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

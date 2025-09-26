#pragma once

#include <cstdlib>

#include <sstream>
#include <vector>

#include <Kokkos_Core.hpp>

#define checkOutcome(error) \
    cuda_extra::_checkOutcome(error, __FILE__, __LINE__)

namespace cuda_extra {

void _checkOutcome(cudaError_t const error, char const* file,
                        int const line) {
    if (error != cudaSuccess) {
        std::stringstream message;
        message << "Cuda error (" << file << ":" << line
                << "): " << ::cudaGetErrorString(error);
        Kokkos::abort(message.str().c_str());
    }
}

}  // namespace cuda_extra

namespace devices {

namespace impl {

class DevicesCuda {
    std::vector<::cudaStream_t> streams;

   public:
    DevicesCuda() {
        std::size_t const numDevices = this->getNumDevices();
        this->streams.reserve(numDevices);
        for (std::size_t device = 0; device < numDevices; device++) {
            ::cudaStream_t stream;
            checkOutcome(::cudaSetDevice(device));
            checkOutcome(::cudaStreamCreate(&stream));

            this->streams.emplace_back(stream);
        }
    }

    DevicesCuda(const DevicesCuda& other) = delete;
    DevicesCuda& operator=(const DevicesCuda&) = delete;

    ~DevicesCuda() {
        std::size_t const numDevices = this->getNumDevices();
        for (std::size_t device = 0; device < numDevices; device++) {
            ::cudaStream_t stream = this->streams[device];
            checkOutcome(::cudaSetDevice(device));
            checkOutcome(::cudaStreamDestroy(stream));
        }
    }

    std::vector<Kokkos::Cuda> getSpaces() const {
        std::vector<Kokkos::Cuda> spaces;
        spaces.reserve(this->streams.size());
        for (::cudaStream_t const stream : this->streams) {
            spaces.emplace_back(Kokkos::Cuda(stream));
        }

        return spaces;
    }

    static std::size_t getNumDevices() {
        int numDevices;
        checkOutcome(::cudaGetDeviceCount(&numDevices));

        return numDevices;
    }
};

}  // namespace impl

}  // namespace devices

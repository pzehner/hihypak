#pragma once

#include <cstdlib>

#include <algorithm>
#include <stdexcept>
#include <vector>

#include <Kokkos_Core.hpp>

namespace devices {

namespace impl {

class DevicesCuda {
    std::vector<cudaStream_t> m_streams;

   public:
    DevicesCuda() {
        std::size_t const numDevices = getNumDevices();
        for (std::size_t device = 0; device < numDevices; device++) {
            cudaStream_t stream;
            cudaSetDevice(device);
            cudaStreamCreate(&stream);
            m_streams.push_back(stream);
        }
    }

    DevicesCuda(const DevicesCuda& other) = delete;
    DevicesCuda& operator=(const DevicesCuda&) = delete;

    ~DevicesCuda() {
        std::size_t const numDevices = getNumDevices();
        for (std::size_t device = 0; device < numDevices; device++) {
            cudaStream_t stream = m_streams[device];
            cudaSetDevice(device);
            cudaStreamDestroy(stream);
        }
    }

    std::vector<Kokkos::Cuda> getSpaces() const {
        std::vector<Kokkos::Cuda> spaces(getNumDevices());
        std::transform(
            m_streams.cbegin(), m_streams.cend(), spaces.begin(),
            [](cudaStream_t const stream) { return Kokkos::Cuda(stream); });

        return spaces;
    }

    static std::size_t getNumDevices() {
        int numDevices;
        cudaError_t outcome = cudaGetDeviceCount(&numDevices);

        if (outcome != cudaSuccess) {
            throw std::runtime_error("No usable device found");
        }

        return numDevices;
    }
};

}  // namespace impl

}  // namespace devices

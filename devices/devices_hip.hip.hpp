#pragma once

#include <cstdlib>

#include <algorithm>
#include <stdexcept>
#include <vector>

#include <Kokkos_Core.hpp>

namespace devices {

namespace impl {

class DevicesHip {
    std::vector<::hipStream_t> streams;

   public:
    DevicesHip() {
        std::size_t const numDevices = this->getNumDevices();
        for (std::size_t device = 0; device < numDevices; device++) {
            ::hipStream_t stream;
            ::hipSetDevice(device);
            ::hipStreamCreate(&stream);
            this->streams.push_back(stream);
        }
    }

    DevicesHip(const DevicesHip& other) = delete;
    DevicesHip& operator=(const DevicesHip&) = delete;

    ~DevicesHip() {
        std::size_t const numDevices = this->getNumDevices();
        for (std::size_t device = 0; device < numDevices; device++) {
            ::hipStream_t stream = streams[device];
            ::hipSetDevice(device);
            ::hipStreamDestroy(stream);
        }
    }

    std::vector<Kokkos::HIP> getSpaces() const {
        std::vector<Kokkos::HIP> spaces(this->getNumDevices());
        std::transform(
            streams.cbegin(), streams.cend(), spaces.begin(),
            [](::hipStream_t const stream) { return Kokkos::HIP(stream); });

        return spaces;
    }

    static std::size_t getNumDevices() {
        int numDevices;
        ::hipError_t outcome = ::hipGetDeviceCount(&numDevices);

        if (outcome != ::hipSuccess) {
            throw std::runtime_error("No usable device found");
        }

        return numDevices;
    }
};

}  // namespace impl

}  // namespace devices

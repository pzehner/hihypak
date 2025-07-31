#pragma once

#include <cstdlib>

#include <vector>

#include <Kokkos_Core.hpp>

namespace devices {

namespace impl {

class DevicesSerial {
   public:
    DevicesSerial() = default;
    DevicesSerial(const DevicesSerial& other) = delete;
    DevicesSerial& operator=(const DevicesSerial&) = delete;

    std::vector<Kokkos::Serial> getSpaces() const {
        return {Kokkos::DefaultExecutionSpace{}};
    }

    static std::size_t getNumDevices() {
        return 1;
    }
};

}  // namespace impl

}  // namespace devices

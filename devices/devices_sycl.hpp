#pragma once

#include <cstdlib>

#include <algorithm>
#include <vector>

#include <sycl/sycl.hpp>

#include <Kokkos_Core.hpp>

namespace devices {

namespace impl {

class DevicesSycl {
    std::vector<sycl::queue> m_queues;

   public:
    DevicesSycl() {
        std::size_t const numDevices = getNumDevices();
        auto const devices = getDevices();
        for (std::size_t device = 0; device < numDevices; device++) {
            sycl::queue queue(devices[device], sycl::property::queue::in_order());
            m_queues.push_back(queue);
        }
    }

    DevicesSycl(const DevicesSycl& other) = delete;
    DevicesSycl& operator=(const DevicesSycl&) = delete;

    std::vector<Kokkos::SYCL> getSpaces() const {
        std::vector<Kokkos::SYCL> spaces(getNumDevices());
        std::transform(
            m_streams.cbegin(), m_streams.cend(), spaces.begin(),
            [](sycl::queue const queue) { return Kokkos::SYCL(queue); });

        return spaces;
    }

    static std::size_t getNumDevices() {
        return getDevices().size();
    }

   private:
    static std::vector<sycl::device> getDevices() {
         return sycl::device::gett_devices(sycl::info::device_type::gpu);
    }
};

}  // namespace impl

}  // namespace devices

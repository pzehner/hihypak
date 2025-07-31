#pragma once

#if defined(KOKKOS_ENABLE_CUDA)

#include "devices/devices_cuda.cuh"

namespace devices {

using Devices = impl::DevicesCuda;

}

#elif defined(KOKKOS_ENABLE_HIP)

#include "devices/devices_hip.hip.hpp"

namespace devices {

using Devices = impl::DevicesHip;

}

#elif defined(KOKKOS_ENABLE_SYCL)

#include "devices/devices_sycl.hpp"

namespace devices {

using Devices = impl::DevicesSycl;

}

#elif defined(KOKKOS_ENABLE_SERIAL)

#include "devices/devices_serial.hpp"

namespace devices {

using Devices = impl::DevicesSerial;

}

#else

#error "Unknown backend"

#endif

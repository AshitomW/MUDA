#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace muda::runtime {

struct DeviceInfo {
  std::string name;
  std::string backend;
  std::uint64_t recommended_working_set_bytes = 0;
  std::uint32_t max_threads_per_threadgroup = 0;
};

[[nodiscard]] std::vector<DeviceInfo> EnumerateDevices();

}

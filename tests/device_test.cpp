#include "muda/runtime/device.hpp"

#include <iostream>
#include <vector>

int main() {
  const std::vector<muda::runtime::DeviceInfo> devices = muda::runtime::EnumerateDevices();
  if (devices.empty()) {
    std::cout << "no Metal devices available\n";
    return 0;
  }

  for (const muda::runtime::DeviceInfo& device : devices) {
    if (device.backend != "metal") {
      std::cerr << "unexpected backend: " << device.backend << "\n";
      return 1;
    }
    if (device.name.empty()) {
      std::cerr << "empty device name\n";
      return 1;
    }
    if (device.max_threads_per_threadgroup == 0) {
      std::cerr << "invalid threadgroup size\n";
      return 1;
    }
  }

  std::cout << "devices: " << devices.size() << "\n";
  return 0;
}

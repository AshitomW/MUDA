/**
 * @file device.hpp
 * @brief Wraps a Metal device and allocation entry points.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace muda::runtime {

struct DeviceInfo {
  std::string name;
  std::string backend;
  std::uint64_t recommended_working_set_bytes = 0;
  std::uint32_t max_threads_per_threadgroup = 0;
};

struct DeviceStorage;
class Buffer;

class Device {
 public:
  Device();
  ~Device();

  Device(Device&&) noexcept;
  Device& operator=(Device&&) noexcept;
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;

  [[nodiscard]] static Device Default();
  [[nodiscard]] bool valid() const;
  [[nodiscard]] DeviceInfo info() const;
  [[nodiscard]] Buffer Allocate(std::size_t bytes) const;

 private:
  explicit Device(std::shared_ptr<DeviceStorage> storage);

  std::shared_ptr<DeviceStorage> storage_;
  friend class Kernel;
};

[[nodiscard]] std::vector<DeviceInfo> EnumerateDevices();

}

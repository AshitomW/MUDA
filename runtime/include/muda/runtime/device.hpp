/**
 * @file device.hpp
 * @brief Wraps a Metal device and allocation entry points.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace muda::runtime {

enum class BackendKind : std::uint8_t {
  kMetal,
};

[[nodiscard]] constexpr std::string_view BackendName(BackendKind kind) noexcept {
  switch (kind) {
    case BackendKind::kMetal:
      return "metal";
  }
  return "unknown";
}

struct DeviceInfo {
  std::string name;
  BackendKind backend = BackendKind::kMetal;
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
  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] DeviceInfo info() const;
  [[nodiscard]] Buffer Allocate(std::size_t bytes) const;

 private:
  explicit Device(std::unique_ptr<DeviceStorage> storage);

  std::unique_ptr<DeviceStorage> storage_;
  friend class Kernel;
};

[[nodiscard]] std::vector<DeviceInfo> EnumerateDevices();

}  // namespace muda::runtime

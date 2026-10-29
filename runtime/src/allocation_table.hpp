/**
 * @file allocation_table.hpp
 * @brief Tracks Metal buffer allocations and validates device pointer ranges.
 */

#pragma once

#import <Metal/Metal.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>

#include "cuda_runtime.h"

namespace muda::runtime::cudart {

enum class AttachMode : std::uint8_t {
  kGlobal,
  kHost,
  kSingle,
};

[[nodiscard]] constexpr std::optional<AttachMode> ParseAttachMode(unsigned int flags) noexcept {
  switch (flags) {
    case cudaMemAttachGlobal:
      return AttachMode::kGlobal;
    case cudaMemAttachHost:
      return AttachMode::kHost;
    case cudaMemAttachSingle:
      return AttachMode::kSingle;
    default:
      return std::nullopt;
  }
}

struct DeviceAddress {
  std::uintptr_t value = 0;

  [[nodiscard]] static DeviceAddress FromPointer(const void* ptr) noexcept {
    return {reinterpret_cast<std::uintptr_t>(ptr)};
  }

  constexpr auto operator<=>(const DeviceAddress&) const noexcept = default;
};

struct AllocationRecord {
  DeviceAddress base_address;
  std::size_t size = 0;
  std::size_t capacity = 0;
  AttachMode attach_mode = AttachMode::kGlobal;
  id<MTLBuffer> buffer = nil;
};

class AllocationTable {
 public:
  void Insert(AllocationRecord record);
  [[nodiscard]] std::optional<AllocationRecord> Remove(DeviceAddress base_address);
  [[nodiscard]] const AllocationRecord* FindContaining(DeviceAddress address) const noexcept;
  [[nodiscard]] static bool IsRangeValid(const AllocationRecord* record, DeviceAddress address,
                                         std::size_t count) noexcept;
  [[nodiscard]] static bool IsOffsetValid(const AllocationRecord* record,
                                          DeviceAddress address) noexcept;
  void Clear();

 private:
  std::map<DeviceAddress, AllocationRecord> records_;
};

}  // namespace muda::runtime::cudart


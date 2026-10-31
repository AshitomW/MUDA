/**
 * @file memory_manager.hpp
 * @brief Manages Metal buffer allocations and device pointer tables.
 */

#pragma once

#import <Metal/Metal.h>

#include <cstddef>
#include <optional>
#include <shared_mutex>

#include "allocation_table.hpp"
#include "cuda_runtime.h"
#include "stream_manager.hpp"

namespace muda::runtime::cudart {

inline constexpr std::size_t kMinAllocationAlignment = 256;

[[nodiscard]] constexpr std::size_t AlignUp(std::size_t value, std::size_t alignment) noexcept {
  const std::size_t remainder = value % alignment;
  if (remainder == 0) {
    return value;
  }
  if (value > static_cast<std::size_t>(-1) - (alignment - remainder)) {
    return value;
  }
  return value + (alignment - remainder);
}

struct ResolvedEndpoint {
  id<MTLBuffer> buffer = nil;
  std::size_t offset = 0;
  std::size_t size = 0;
  bool is_device = false;
};

class MemoryManager {
 public:
  MemoryManager() = default;
  ~MemoryManager() = default;
  MemoryManager(const MemoryManager&) = delete;
  MemoryManager& operator=(const MemoryManager&) = delete;

  cudaError_t Allocate(void** dev_ptr, std::size_t size, unsigned int flags,
                       cudaMemoryType memory_type, id<MTLDevice> device) noexcept;
  cudaError_t AllocateHost(void** host_ptr, std::size_t size, unsigned int flags,
                           id<MTLDevice> device) noexcept;
  cudaError_t AllocatePitch(void** dev_ptr, std::size_t* pitch, std::size_t width,
                            std::size_t height, id<MTLDevice> device) noexcept;
  cudaError_t Free(void* dev_ptr, StreamManager& streams,
                   id<MTLCommandQueue> default_queue) noexcept;
  cudaError_t FreeHost(void* ptr, StreamManager& streams,
                       id<MTLCommandQueue> default_queue) noexcept;

  cudaError_t MemGetInfo(size_t* free_bytes, size_t* total_bytes,
                         id<MTLDevice> device) const noexcept;
  cudaError_t PointerGetAttributes(cudaPointerAttributes* attributes,
                                   const void* ptr) const noexcept;

  [[nodiscard]] bool ResolveBuffer(DeviceAddress addr, id<MTLBuffer> __strong* out_buffer,
                                   std::size_t* out_offset) const noexcept;
  [[nodiscard]] bool ResolveEndpoint(DeviceAddress addr,
                                     ResolvedEndpoint* out_endpoint) const noexcept;

  [[nodiscard]] std::optional<AllocationRecord> SnapshotRecord(DeviceAddress addr) const noexcept;
  [[nodiscard]] bool ValidateRange(DeviceAddress addr, std::size_t count) const noexcept;
  [[nodiscard]] std::size_t TotalAllocatedBytes() const noexcept;

  void Reset() noexcept;

 private:
  mutable std::shared_mutex mutex_;
  AllocationTable allocations_;
};

}  // namespace muda::runtime::cudart

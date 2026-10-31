/**
 * @file memory_manager.mm
 * @brief Manages Metal buffer allocations and device pointer tables.
 */

#include "memory_manager.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <mutex>
#include <utility>
#include <vector>

#include "error_state.hpp"

namespace muda::runtime::cudart {
namespace {

[[nodiscard]] constexpr bool IsValidHostAllocFlags(unsigned int flags) noexcept {
  constexpr unsigned int kAllowed = cudaHostAllocDefault | cudaHostAllocPortable |
                                    cudaHostAllocMapped | cudaHostAllocWriteCombined;
  return (flags & ~kAllowed) == 0u;
}

}  // namespace

cudaError_t MemoryManager::Allocate(void** dev_ptr, std::size_t size, unsigned int flags,
                                    cudaMemoryType memory_type, id<MTLDevice> device) noexcept {
  if (dev_ptr == nullptr || size == 0 || device == nil) {
    return RecordError(cudaErrorInvalidValue);
  }
  const std::optional<AttachMode> attach_mode = ParseAttachMode(flags);
  if (!attach_mode.has_value()) {
    return RecordError(cudaErrorInvalidValue);
  }

  @autoreleasepool {
    const std::size_t aligned_size = AlignUp(size, kMinAllocationAlignment);
    if (aligned_size > device.maxBufferLength) {
      return RecordError(cudaErrorMemoryAllocation);
    }
    id<MTLBuffer> buffer = [device newBufferWithLength:aligned_size
                                               options:MTLResourceStorageModeShared];
    if (buffer == nil || buffer.contents == nullptr) {
      return RecordError(cudaErrorMemoryAllocation);
    }

    void* contents = buffer.contents;
    {
      std::unique_lock<std::shared_mutex> lock(mutex_);
      allocations_.Insert(AllocationRecord{
          .base_address = DeviceAddress::FromPointer(contents),
          .size = size,
          .capacity = aligned_size,
          .attach_mode = *attach_mode,
          .memory_type = memory_type,
          .buffer = buffer,
      });
    }

    *dev_ptr = contents;
    return cudaSuccess;
  }
}

cudaError_t MemoryManager::AllocateHost(void** host_ptr, std::size_t size, unsigned int flags,
                                        id<MTLDevice> device) noexcept {
  if (!IsValidHostAllocFlags(flags)) {
    return RecordError(cudaErrorInvalidValue);
  }
  return Allocate(host_ptr, size, cudaMemAttachGlobal, cudaMemoryTypeHost, device);
}

cudaError_t MemoryManager::AllocatePitch(void** dev_ptr, std::size_t* pitch, std::size_t width,
                                         std::size_t height, id<MTLDevice> device) noexcept {
  if (dev_ptr == nullptr || pitch == nullptr || width == 0 || height == 0) {
    return RecordError(cudaErrorInvalidValue);
  }
  const std::size_t row_pitch = AlignUp(width, kMinAllocationAlignment);
  if (row_pitch < width || row_pitch > static_cast<std::size_t>(-1) / height) {
    return RecordError(cudaErrorMemoryAllocation);
  }
  const std::size_t total_bytes = row_pitch * height;
  const cudaError_t status =
      Allocate(dev_ptr, total_bytes, cudaMemAttachGlobal, cudaMemoryTypeDevice, device);
  if (status != cudaSuccess) {
    return status;
  }
  *pitch = row_pitch;
  return cudaSuccess;
}

cudaError_t MemoryManager::Free(void* dev_ptr, StreamManager& streams,
                                id<MTLCommandQueue> default_queue) noexcept {
  if (dev_ptr == nullptr) {
    return cudaSuccess;
  }
  const DeviceAddress address = DeviceAddress::FromPointer(dev_ptr);
  @autoreleasepool {
    std::optional<AllocationRecord> removed;
    {
      std::unique_lock<std::shared_mutex> lock(mutex_);
      removed = allocations_.Remove(address);
    }
    if (!removed.has_value()) {
      return RecordError(cudaErrorInvalidValue);
    }
    (void)streams.SynchronizeDevice(default_queue);
    removed.reset();
  }
  return cudaSuccess;
}

cudaError_t MemoryManager::FreeHost(void* ptr, StreamManager& streams,
                                    id<MTLCommandQueue> default_queue) noexcept {
  return Free(ptr, streams, default_queue);
}

cudaError_t MemoryManager::MemGetInfo(size_t* free_bytes, size_t* total_bytes,
                                      id<MTLDevice> device) const noexcept {
  if (free_bytes == nullptr || total_bytes == nullptr) {
    return RecordError(cudaErrorInvalidValue);
  }
  if (device == nil) {
    return RecordError(cudaErrorInitializationError);
  }
  std::shared_lock<std::shared_mutex> lock(mutex_);
  const std::size_t total = static_cast<std::size_t>(device.recommendedMaxWorkingSetSize);
  const std::size_t allocated = allocations_.TotalAllocatedBytes();
  *total_bytes = total;
  *free_bytes = (allocated <= total) ? (total - allocated) : 0;
  return cudaSuccess;
}

cudaError_t MemoryManager::PointerGetAttributes(cudaPointerAttributes* attributes,
                                                const void* ptr) const noexcept {
  if (attributes == nullptr || ptr == nullptr) {
    return RecordError(cudaErrorInvalidValue);
  }
  std::shared_lock<std::shared_mutex> lock(mutex_);
  const DeviceAddress addr = DeviceAddress::FromPointer(ptr);
  const AllocationRecord* record = allocations_.FindContaining(addr);
  if (record == nullptr) {
    attributes->type = cudaMemoryTypeUnregistered;
    attributes->device = 0;
    attributes->devicePointer = nullptr;
    attributes->hostPointer = nullptr;
    return cudaSuccess;
  }
  attributes->type = record->memory_type;
  attributes->device = 0;
  attributes->devicePointer = reinterpret_cast<void*>(record->base_address.value);
  attributes->hostPointer =
      (record->memory_type == cudaMemoryTypeHost) ? attributes->devicePointer : nullptr;
  return cudaSuccess;
}

bool MemoryManager::ResolveBuffer(DeviceAddress addr, id<MTLBuffer> __strong* out_buffer,
                                  std::size_t* out_offset) const noexcept {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  const AllocationRecord* record = allocations_.FindContaining(addr);
  if (record == nullptr || (addr.value - record->base_address.value) >= record->size) {
    return false;
  }
  *out_buffer = record->buffer;
  *out_offset = static_cast<std::size_t>(addr.value - record->base_address.value);
  return true;
}

bool MemoryManager::ResolveEndpoint(DeviceAddress addr,
                                    ResolvedEndpoint* out_endpoint) const noexcept {
  if (out_endpoint == nullptr) {
    return false;
  }
  std::shared_lock<std::shared_mutex> lock(mutex_);
  const AllocationRecord* record = allocations_.FindContaining(addr);
  if (record == nullptr) {
    out_endpoint->buffer = nil;
    out_endpoint->offset = 0;
    out_endpoint->size = 0;
    out_endpoint->is_device = false;
    return false;
  }
  out_endpoint->buffer = record->buffer;
  out_endpoint->offset = static_cast<std::size_t>(addr.value - record->base_address.value);
  out_endpoint->size = record->size;
  out_endpoint->is_device = true;
  return true;
}

std::optional<AllocationRecord> MemoryManager::SnapshotRecord(DeviceAddress addr) const noexcept {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  const AllocationRecord* record = allocations_.FindContaining(addr);
  if (record == nullptr) {
    return std::nullopt;
  }
  return *record;
}

bool MemoryManager::ValidateRange(DeviceAddress addr, std::size_t count) const noexcept {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  const AllocationRecord* record = allocations_.FindContaining(addr);
  return AllocationTable::IsRangeValid(record, addr, count);
}

std::size_t MemoryManager::TotalAllocatedBytes() const noexcept {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  return allocations_.TotalAllocatedBytes();
}

void MemoryManager::Reset() noexcept {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  allocations_.Clear();
}

}  // namespace muda::runtime::cudart

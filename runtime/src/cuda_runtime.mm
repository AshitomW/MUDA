/**
 * @file cuda_runtime.mm
 * @brief Implements the Phase 0 CUDA Runtime C API over Metal shared buffers.
 */

#include "cuda_runtime.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdint>
#include <cstring>
#include <map>
#include <mutex>
#include <shared_mutex>

namespace {

constexpr std::size_t kMinAlignment = 256;

thread_local cudaError_t g_last_error = cudaSuccess;

cudaError_t RecordError(cudaError_t error) noexcept {
  if (error != cudaSuccess) {
    g_last_error = error;
  }
  return error;
}

std::size_t AlignUp(std::size_t value, std::size_t alignment) noexcept {
  const std::size_t remainder = value % alignment;
  if (remainder == 0) {
    return value;
  }
  if (value > static_cast<std::size_t>(-1) - (alignment - remainder)) {
    return value;
  }
  return value + (alignment - remainder);
}

struct AllocationRecord {
  std::uintptr_t base_address = 0;
  std::size_t size = 0;
  std::size_t capacity = 0;
  unsigned int flags = 0;
  id<MTLBuffer> buffer = nil;
};

class RuntimeState {
 public:
  static RuntimeState& Instance() {
    static RuntimeState* state = new RuntimeState();
    return *state;
  }

  cudaError_t EnsureInitialized() {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    return EnsureInitializedLocked();
  }

  cudaError_t GetDeviceCount(int* count) {
    if (count == nullptr) {
      return RecordError(cudaErrorInvalidValue);
    }
    @autoreleasepool {
      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      *count = (device != nil) ? 1 : 0;
      return cudaSuccess;
    }
  }

  cudaError_t SetDevice(int device) {
    if (device != 0) {
      return RecordError(cudaErrorInvalidDevice);
    }
    return EnsureInitialized();
  }

  cudaError_t GetDevice(int* device) {
    if (device == nullptr) {
      return RecordError(cudaErrorInvalidValue);
    }
    const cudaError_t status = EnsureInitialized();
    if (status != cudaSuccess) {
      return status;
    }
    *device = 0;
    return cudaSuccess;
  }

  cudaError_t Synchronize() {
    id<MTLCommandQueue> queue = nil;
    {
      std::unique_lock<std::shared_mutex> lock(mutex_);
      const cudaError_t status = EnsureInitializedLocked();
      if (status != cudaSuccess) {
        return status;
      }
      queue = queue_;
    }

    @autoreleasepool {
      id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
      if (command_buffer == nil) {
        return RecordError(cudaErrorInternalError);
      }
      [command_buffer commit];
      [command_buffer waitUntilCompleted];
      if (command_buffer.error != nil) {
        return RecordError(cudaErrorUnknown);
      }
    }
    return cudaSuccess;
  }

  cudaError_t Reset() {
    id<MTLCommandQueue> queue = nil;
    {
      std::shared_lock<std::shared_mutex> lock(mutex_);
      queue = queue_;
    }
    if (queue != nil) {
      @autoreleasepool {
        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
        if (command_buffer != nil) {
          [command_buffer commit];
          [command_buffer waitUntilCompleted];
        }
      }
    }

    std::unique_lock<std::shared_mutex> lock(mutex_);
    @autoreleasepool {
      allocations_.clear();
      queue_ = nil;
      device_ = nil;
    }
    return cudaSuccess;
  }

  cudaError_t Allocate(void** dev_ptr, std::size_t size, unsigned int flags) {
    if (dev_ptr == nullptr || size == 0) {
      return RecordError(cudaErrorInvalidValue);
    }
    if (flags != cudaMemAttachGlobal && flags != cudaMemAttachHost &&
        flags != cudaMemAttachSingle) {
      return RecordError(cudaErrorInvalidValue);
    }

    id<MTLDevice> device = nil;
    {
      std::unique_lock<std::shared_mutex> lock(mutex_);
      const cudaError_t status = EnsureInitializedLocked();
      if (status != cudaSuccess) {
        return status;
      }
      device = device_;
    }

    @autoreleasepool {
      const std::size_t aligned_size = AlignUp(size, kMinAlignment);
      if (aligned_size > device.maxBufferLength) {
        return RecordError(cudaErrorMemoryAllocation);
      }

      id<MTLBuffer> buffer = [device newBufferWithLength:aligned_size
                                                 options:MTLResourceStorageModeShared];
      if (buffer == nil || buffer.contents == nullptr) {
        return RecordError(cudaErrorMemoryAllocation);
      }

      void* contents = buffer.contents;
      const auto base = reinterpret_cast<std::uintptr_t>(contents);

      std::unique_lock<std::shared_mutex> lock(mutex_);
      AllocationRecord record;
      record.base_address = base;
      record.size = size;
      record.capacity = aligned_size;
      record.flags = flags;
      record.buffer = buffer;
      allocations_[base] = record;

      *dev_ptr = contents;
      return cudaSuccess;
    }
  }

  cudaError_t Free(void* dev_ptr) {
    if (dev_ptr == nullptr) {
      return cudaSuccess;
    }

    const auto address = reinterpret_cast<std::uintptr_t>(dev_ptr);
    @autoreleasepool {
      std::unique_lock<std::shared_mutex> lock(mutex_);
      auto it = allocations_.find(address);
      if (it == allocations_.end()) {
        return RecordError(cudaErrorInvalidValue);
      }
      allocations_.erase(it);
    }
    return cudaSuccess;
  }

  cudaError_t Memcpy(void* dst, const void* src, std::size_t count, cudaMemcpyKind kind) {
    if (dst == nullptr || src == nullptr) {
      return RecordError(cudaErrorInvalidValue);
    }
    if (kind != cudaMemcpyHostToHost && kind != cudaMemcpyHostToDevice &&
        kind != cudaMemcpyDeviceToHost && kind != cudaMemcpyDeviceToDevice &&
        kind != cudaMemcpyDefault) {
      return RecordError(cudaErrorInvalidMemcpyDirection);
    }

    const auto dst_addr = reinterpret_cast<std::uintptr_t>(dst);
    const auto src_addr = reinterpret_cast<std::uintptr_t>(src);

    const auto validate_copy_locked = [&]() noexcept -> cudaError_t {
      const AllocationRecord* dst_record = FindContainingRecordLocked(dst_addr);
      const AllocationRecord* src_record = FindContainingRecordLocked(src_addr);

      const bool dst_valid_device = IsRangeValidInRecord(dst_record, dst_addr, count);
      const bool src_valid_device = IsRangeValidInRecord(src_record, src_addr, count);

      switch (kind) {
        case cudaMemcpyHostToHost:
          if (dst_record != nullptr || src_record != nullptr) {
            return RecordError(cudaErrorInvalidValue);
          }
          break;
        case cudaMemcpyHostToDevice:
          if (!dst_valid_device || src_record != nullptr) {
            return RecordError(cudaErrorInvalidValue);
          }
          break;
        case cudaMemcpyDeviceToHost:
          if (!src_valid_device || dst_record != nullptr) {
            return RecordError(cudaErrorInvalidValue);
          }
          break;
        case cudaMemcpyDeviceToDevice:
          if (!dst_valid_device || !src_valid_device) {
            return RecordError(cudaErrorInvalidValue);
          }
          break;
        case cudaMemcpyDefault:
          if (dst_record != nullptr && !dst_valid_device) {
            return RecordError(cudaErrorInvalidValue);
          }
          if (src_record != nullptr && !src_valid_device) {
            return RecordError(cudaErrorInvalidValue);
          }
          break;
      }
      return cudaSuccess;
    };

    bool touches_device = false;
    {
      std::shared_lock<std::shared_mutex> lock(mutex_);
      const AllocationRecord* dst_record = FindContainingRecordLocked(dst_addr);
      const AllocationRecord* src_record = FindContainingRecordLocked(src_addr);
      const cudaError_t validation_status = validate_copy_locked();
      if (validation_status != cudaSuccess) {
        return validation_status;
      }
      touches_device = (dst_record != nullptr || src_record != nullptr);
    }

    if (touches_device) {
      const cudaError_t sync_status = Synchronize();
      if (sync_status != cudaSuccess) {
        return sync_status;
      }
    }

    std::shared_lock<std::shared_mutex> lock(mutex_);
    const cudaError_t validation_status = validate_copy_locked();
    if (validation_status != cudaSuccess) {
      return validation_status;
    }

    if (count > 0 && dst != src) {
      std::memmove(dst, src, count);
    }
    return cudaSuccess;
  }

  cudaError_t Memset(void* dev_ptr, int value, std::size_t count) {
    if (dev_ptr == nullptr) {
      return RecordError(cudaErrorInvalidValue);
    }

    const auto addr = reinterpret_cast<std::uintptr_t>(dev_ptr);

    const cudaError_t sync_status = Synchronize();
    if (sync_status != cudaSuccess) {
      return sync_status;
    }

    std::shared_lock<std::shared_mutex> lock(mutex_);
    const AllocationRecord* record = FindContainingRecordLocked(addr);
    if (!IsRangeValidInRecord(record, addr, count)) {
      return RecordError(cudaErrorInvalidValue);
    }

    if (count > 0) {
      std::memset(dev_ptr, value, count);
    }
    return cudaSuccess;
  }

 private:
  RuntimeState() = default;

  cudaError_t EnsureInitializedLocked() {
    if (device_ != nil && queue_ != nil) {
      return cudaSuccess;
    }
    @autoreleasepool {
      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      if (device == nil) {
        return RecordError(cudaErrorNoDevice);
      }
      id<MTLCommandQueue> queue = [device newCommandQueue];
      if (queue == nil) {
        return RecordError(cudaErrorInitializationError);
      }
      device_ = device;
      queue_ = queue;
      return cudaSuccess;
    }
  }

  const AllocationRecord* FindContainingRecordLocked(std::uintptr_t address) const noexcept {
    if (allocations_.empty()) {
      return nullptr;
    }
    auto it = allocations_.upper_bound(address);
    if (it == allocations_.begin()) {
      return nullptr;
    }
    --it;
    const AllocationRecord& record = it->second;
    if (address >= record.base_address && (address - record.base_address) < record.capacity) {
      return &record;
    }
    return nullptr;
  }

  static bool IsRangeValidInRecord(const AllocationRecord* record, std::uintptr_t address,
                                   std::size_t count) noexcept {
    if (record == nullptr || address < record->base_address) {
      return false;
    }
    const std::size_t offset = static_cast<std::size_t>(address - record->base_address);
    if (offset > record->size) {
      return false;
    }
    if (offset == record->size && count == 0) {
      return true;
    }
    return count <= (record->size - offset);
  }

  mutable std::shared_mutex mutex_;
  id<MTLDevice> device_ = nil;
  id<MTLCommandQueue> queue_ = nil;
  std::map<std::uintptr_t, AllocationRecord> allocations_;
};

}  // namespace

extern "C" {

cudaError_t cudaGetDeviceCount(int* count) {
  @try {
    return RuntimeState::Instance().GetDeviceCount(count);
  } @catch (...) {
    return RecordError(cudaErrorInternalError);
  }
}

cudaError_t cudaSetDevice(int device) {
  @try {
    return RuntimeState::Instance().SetDevice(device);
  } @catch (...) {
    return RecordError(cudaErrorInternalError);
  }
}

cudaError_t cudaGetDevice(int* device) {
  @try {
    return RuntimeState::Instance().GetDevice(device);
  } @catch (...) {
    return RecordError(cudaErrorInternalError);
  }
}

cudaError_t cudaDeviceSynchronize(void) {
  @try {
    return RuntimeState::Instance().Synchronize();
  } @catch (...) {
    return RecordError(cudaErrorInternalError);
  }
}

cudaError_t cudaDeviceReset(void) {
  @try {
    return RuntimeState::Instance().Reset();
  } @catch (...) {
    return RecordError(cudaErrorInternalError);
  }
}

cudaError_t cudaMalloc(void** devPtr, size_t size) {
  @try {
    return RuntimeState::Instance().Allocate(devPtr, size, cudaMemAttachGlobal);
  } @catch (...) {
    return RecordError(cudaErrorInternalError);
  }
}

cudaError_t cudaMallocManaged(void** devPtr, size_t size, unsigned int flags) {
  @try {
    return RuntimeState::Instance().Allocate(devPtr, size, flags);
  } @catch (...) {
    return RecordError(cudaErrorInternalError);
  }
}

cudaError_t cudaFree(void* devPtr) {
  @try {
    return RuntimeState::Instance().Free(devPtr);
  } @catch (...) {
    return RecordError(cudaErrorInternalError);
  }
}

cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) {
  @try {
    return RuntimeState::Instance().Memcpy(dst, src, count, kind);
  } @catch (...) {
    return RecordError(cudaErrorInternalError);
  }
}

cudaError_t cudaMemset(void* devPtr, int value, size_t count) {
  @try {
    return RuntimeState::Instance().Memset(devPtr, value, count);
  } @catch (...) {
    return RecordError(cudaErrorInternalError);
  }
}

cudaError_t cudaGetLastError(void) {
  const cudaError_t error = g_last_error;
  g_last_error = cudaSuccess;
  return error;
}

cudaError_t cudaPeekAtLastError(void) { return g_last_error; }

const char* cudaGetErrorName(cudaError_t error) {
  switch (error) {
    case cudaSuccess:
      return "cudaSuccess";
    case cudaErrorInvalidValue:
      return "cudaErrorInvalidValue";
    case cudaErrorMemoryAllocation:
      return "cudaErrorMemoryAllocation";
    case cudaErrorInitializationError:
      return "cudaErrorInitializationError";
    case cudaErrorInvalidDevice:
      return "cudaErrorInvalidDevice";
    case cudaErrorInvalidDevicePointer:
      return "cudaErrorInvalidDevicePointer";
    case cudaErrorInvalidMemcpyDirection:
      return "cudaErrorInvalidMemcpyDirection";
    case cudaErrorNoDevice:
      return "cudaErrorNoDevice";
    case cudaErrorNotSupported:
      return "cudaErrorNotSupported";
    case cudaErrorInternalError:
      return "cudaErrorInternalError";
    case cudaErrorUnknown:
      return "cudaErrorUnknown";
  }
  return "cudaErrorUnknown";
}

const char* cudaGetErrorString(cudaError_t error) {
  switch (error) {
    case cudaSuccess:
      return "no error";
    case cudaErrorInvalidValue:
      return "invalid argument";
    case cudaErrorMemoryAllocation:
      return "out of memory";
    case cudaErrorInitializationError:
      return "initialization error";
    case cudaErrorInvalidDevice:
      return "invalid device ordinal";
    case cudaErrorInvalidDevicePointer:
      return "invalid device pointer";
    case cudaErrorInvalidMemcpyDirection:
      return "invalid memcpy direction";
    case cudaErrorNoDevice:
      return "no CUDA-capable Metal device is detected";
    case cudaErrorNotSupported:
      return "operation not supported";
    case cudaErrorInternalError:
      return "internal error";
    case cudaErrorUnknown:
      return "unknown error";
  }
  return "unknown error";
}

}  // extern "C"

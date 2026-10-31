/**
 * @file cuda_runtime.mm
 * @brief Implements the CUDA Runtime C API over Metal buffers, streams, events, and compute
 *        pipelines.
 */

#include "cuda_runtime.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <dispatch/dispatch.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <shared_mutex>
#include <span>
#include <utility>
#include <vector>

#include "allocation_table.hpp"
#include "error_state.hpp"
#include "handle_table.hpp"
#include "kernel_registry.hpp"
#include "metal_bridge.hpp"

namespace muda::runtime::cudart {
namespace {

constexpr std::size_t kMinAlignment = 256;
constexpr int kEmulatedCudaRuntimeVersion = 12040;
constexpr int kEmulatedCudaDriverVersion = 12040;
constexpr int kDefaultComputeMajor = 8;
constexpr int kDefaultComputeMinor = 0;
constexpr int kDefaultWarpSize = 32;
constexpr int kDefaultMaxGridDimX = 2147483647;
constexpr int kDefaultMaxGridDimYZ = 65535;
constexpr int kDefaultClockRateKhz = 1000000;
constexpr std::size_t kDefaultTotalConstMem = 65536;

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

[[nodiscard]] constexpr bool IsValidMemcpyKind(cudaMemcpyKind kind) noexcept {
  return kind == cudaMemcpyHostToHost || kind == cudaMemcpyHostToDevice ||
         kind == cudaMemcpyDeviceToHost || kind == cudaMemcpyDeviceToDevice ||
         kind == cudaMemcpyDefault;
}

[[nodiscard]] constexpr bool IsValidStreamFlags(unsigned int flags) noexcept {
  return flags == cudaStreamDefault || flags == cudaStreamNonBlocking;
}

[[nodiscard]] constexpr bool IsValidEventFlags(unsigned int flags) noexcept {
  constexpr unsigned int kAllowed =
      cudaEventDefault | cudaEventBlockingSync | cudaEventDisableTiming;
  return (flags & ~kAllowed) == 0u && (flags & cudaEventInterprocess) == 0u;
}

[[nodiscard]] constexpr bool IsValidHostAllocFlags(unsigned int flags) noexcept {
  constexpr unsigned int kAllowed = cudaHostAllocDefault | cudaHostAllocPortable |
                                    cudaHostAllocMapped | cudaHostAllocWriteCombined;
  return (flags & ~kAllowed) == 0u;
}

[[nodiscard]] double NowSeconds() noexcept {
  using Clock = std::chrono::steady_clock;
  return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}

void WaitForQueue(id<MTLCommandQueue> queue, id<MTLCommandBuffer> last_cmd = nil,
                  std::atomic<std::uint64_t>* pending = nullptr) noexcept {
  if (queue == nil) {
    return;
  }
  if (pending != nullptr && pending->load(std::memory_order_acquire) == 0) {
    return;
  }
  if (last_cmd != nil) {
    [last_cmd waitUntilCompleted];
    return;
  }
  @autoreleasepool {
    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    if (command_buffer != nil) {
      [command_buffer commit];
      [command_buffer waitUntilCompleted];
    }
  }
}

struct BoundArgument {
  mudaKernelParamKind kind = mudaKernelParamBuffer;
  id<MTLBuffer> buffer = nil;
  std::size_t offset = 0;
  const void* bytes = nullptr;
  std::size_t size = 0;
};

struct StreamState {
  id<MTLCommandQueue> queue = nil;
  unsigned int flags = cudaStreamDefault;
  mutable std::mutex mutex;
  id<MTLCommandBuffer> last_command_buffer = nil;
  id<MTLCommandBuffer> active_command_buffer = nil;
  id<MTLComputeCommandEncoder> active_encoder = nil;
  id<MTLComputePipelineState> active_pipeline = nil;
  std::size_t active_dispatches = 0;
  std::atomic<std::uint64_t> pending_commands{0};
};

struct EventState {
  id<MTLSharedEvent> shared_event = nil;
  unsigned int flags = cudaEventDefault;
  mutable std::mutex mutex;
  std::uint64_t recorded_ticket = 0;
  id<MTLCommandBuffer> recorded_command_buffer = nil;
  std::atomic<bool> recorded{false};
  std::atomic<double> completion_timestamp_seconds{0.0};
};

class RuntimeContext {
 public:
  // Intentionally placement-constructed without static destruction so Objective-C
  // Metal objects are never released during C++ static teardown (DESIGN-AND-CODE-STANDARDS §10).
  static RuntimeContext& Instance() noexcept {
    static std::once_flag init_flag;
    alignas(RuntimeContext) static std::byte storage[sizeof(RuntimeContext)];
    static RuntimeContext* instance = nullptr;
    std::call_once(init_flag, [&]() { instance = new (storage) RuntimeContext(); });
    return *instance;
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

  cudaError_t GetRuntimeVersion(int* version) {
    if (version == nullptr) {
      return RecordError(cudaErrorInvalidValue);
    }
    *version = kEmulatedCudaRuntimeVersion;
    return cudaSuccess;
  }

  cudaError_t GetDriverVersion(int* version) {
    if (version == nullptr) {
      return RecordError(cudaErrorInvalidValue);
    }
    *version = kEmulatedCudaDriverVersion;
    return cudaSuccess;
  }

  cudaError_t GetDeviceProperties(cudaDeviceProp* prop, int device_ordinal) {
    if (prop == nullptr) {
      return RecordError(cudaErrorInvalidValue);
    }
    if (device_ordinal != 0) {
      return RecordError(cudaErrorInvalidDevice);
    }

    id<MTLDevice> device = nil;
    const cudaError_t status = AcquireInitializedDevice(&device);
    if (status != cudaSuccess) {
      return status;
    }

    @autoreleasepool {
      std::memset(prop, 0, sizeof(cudaDeviceProp));
      const std::string device_name = bridge::ToStdString(device.name);
      const std::size_t copy_len = std::min(device_name.size(), sizeof(prop->name) - 1);
      std::memcpy(prop->name, device_name.data(), copy_len);
      prop->name[copy_len] = '\0';

      const MTLSize max_threads = device.maxThreadsPerThreadgroup;
      prop->totalGlobalMem = static_cast<std::size_t>(device.recommendedMaxWorkingSetSize);
      prop->sharedMemPerBlock = static_cast<std::size_t>(device.maxThreadgroupMemoryLength);
      prop->sharedMemPerMultiprocessor =
          static_cast<std::size_t>(device.maxThreadgroupMemoryLength);
      prop->sharedMemPerBlockOptin = static_cast<std::size_t>(device.maxThreadgroupMemoryLength);
      prop->regsPerBlock = 65536;
      prop->warpSize = kDefaultWarpSize;
      prop->memPitch = static_cast<std::size_t>(device.maxBufferLength);
      prop->maxThreadsPerBlock = static_cast<int>(max_threads.width);
      prop->maxThreadsDim[0] = static_cast<int>(max_threads.width);
      prop->maxThreadsDim[1] = static_cast<int>(max_threads.height);
      prop->maxThreadsDim[2] = static_cast<int>(max_threads.depth);
      prop->maxGridSize[0] = kDefaultMaxGridDimX;
      prop->maxGridSize[1] = kDefaultMaxGridDimYZ;
      prop->maxGridSize[2] = kDefaultMaxGridDimYZ;
      prop->clockRate = kDefaultClockRateKhz;
      prop->totalConstMem = kDefaultTotalConstMem;
      prop->major = kDefaultComputeMajor;
      prop->minor = kDefaultComputeMinor;
      prop->multiProcessorCount = 8;
      prop->integrated = 1;
      prop->canMapHostMemory = 1;
      prop->computeMode = cudaComputeModeDefault;
      prop->concurrentKernels = 1;
      prop->unifiedAddressing = 1;
      prop->memoryClockRate = kDefaultClockRateKhz;
      prop->memoryBusWidth = 128;
      prop->l2CacheSize = 4 * 1024 * 1024;
      prop->maxThreadsPerMultiProcessor = static_cast<int>(max_threads.width) * 2;
      prop->managedMemory = 1;
      prop->concurrentManagedAccess = 1;
      prop->pageableMemoryAccess = 1;
      return cudaSuccess;
    }
  }

  cudaError_t GetDeviceAttribute(int* value, cudaDeviceAttr attr, int device_ordinal) {
    if (value == nullptr) {
      return RecordError(cudaErrorInvalidValue);
    }
    if (device_ordinal != 0) {
      return RecordError(cudaErrorInvalidDevice);
    }

    cudaDeviceProp prop{};
    const cudaError_t status = GetDeviceProperties(&prop, device_ordinal);
    if (status != cudaSuccess) {
      return status;
    }

    switch (attr) {
      case cudaDevAttrMaxThreadsPerBlock:
        *value = prop.maxThreadsPerBlock;
        return cudaSuccess;
      case cudaDevAttrMaxBlockDimX:
        *value = prop.maxThreadsDim[0];
        return cudaSuccess;
      case cudaDevAttrMaxBlockDimY:
        *value = prop.maxThreadsDim[1];
        return cudaSuccess;
      case cudaDevAttrMaxBlockDimZ:
        *value = prop.maxThreadsDim[2];
        return cudaSuccess;
      case cudaDevAttrMaxGridDimX:
        *value = prop.maxGridSize[0];
        return cudaSuccess;
      case cudaDevAttrMaxGridDimY:
        *value = prop.maxGridSize[1];
        return cudaSuccess;
      case cudaDevAttrMaxGridDimZ:
        *value = prop.maxGridSize[2];
        return cudaSuccess;
      case cudaDevAttrMaxSharedMemoryPerBlock:
      case cudaDevAttrMaxSharedMemoryPerBlockOptin:
        *value = static_cast<int>(prop.sharedMemPerBlock);
        return cudaSuccess;
      case cudaDevAttrTotalConstantMemory:
        *value = static_cast<int>(prop.totalConstMem);
        return cudaSuccess;
      case cudaDevAttrWarpSize:
        *value = prop.warpSize;
        return cudaSuccess;
      case cudaDevAttrMaxPitch:
        *value = static_cast<int>(
            std::min<std::size_t>(prop.memPitch, static_cast<std::size_t>(2147483647)));
        return cudaSuccess;
      case cudaDevAttrClockRate:
        *value = prop.clockRate;
        return cudaSuccess;
      case cudaDevAttrMultiProcessorCount:
        *value = prop.multiProcessorCount;
        return cudaSuccess;
      case cudaDevAttrComputeMode:
        *value = prop.computeMode;
        return cudaSuccess;
      case cudaDevAttrConcurrentKernels:
        *value = prop.concurrentKernels;
        return cudaSuccess;
      case cudaDevAttrUnifiedAddressing:
        *value = prop.unifiedAddressing;
        return cudaSuccess;
      case cudaDevAttrComputeCapabilityMajor:
        *value = prop.major;
        return cudaSuccess;
      case cudaDevAttrComputeCapabilityMinor:
        *value = prop.minor;
        return cudaSuccess;
      case cudaDevAttrManagedMemory:
        *value = prop.managedMemory;
        return cudaSuccess;
      case cudaDevAttrHostNativeAtomicSupported:
      case cudaDevAttrPageableMemoryAccess:
      case cudaDevAttrConcurrentManagedAccess:
        *value = 1;
        return cudaSuccess;
    }
    return RecordError(cudaErrorInvalidValue);
  }

  cudaError_t MemGetInfo(std::size_t* free_bytes, std::size_t* total_bytes) {
    if (free_bytes == nullptr || total_bytes == nullptr) {
      return RecordError(cudaErrorInvalidValue);
    }

    id<MTLDevice> device = nil;
    const cudaError_t status = AcquireInitializedDevice(&device);
    if (status != cudaSuccess) {
      return status;
    }

    std::size_t allocated = 0;
    {
      std::shared_lock<std::shared_mutex> lock(mutex_);
      allocated = allocations_.TotalAllocatedBytes();
    }

    @autoreleasepool {
      const auto working_set = static_cast<std::size_t>(device.recommendedMaxWorkingSetSize);
      *total_bytes = working_set;
      *free_bytes = (allocated < working_set) ? (working_set - allocated) : 0u;
      return cudaSuccess;
    }
  }

  cudaError_t PointerGetAttributes(cudaPointerAttributes* attributes, const void* ptr) {
    if (attributes == nullptr) {
      return RecordError(cudaErrorInvalidValue);
    }

    const cudaError_t init_status = EnsureInitialized();
    if (init_status != cudaSuccess) {
      return init_status;
    }

    std::memset(attributes, 0, sizeof(cudaPointerAttributes));
    if (ptr == nullptr) {
      attributes->type = cudaMemoryTypeUnregistered;
      attributes->device = -1;
      return cudaSuccess;
    }

    const DeviceAddress addr = DeviceAddress::FromPointer(ptr);
    std::shared_lock<std::shared_mutex> lock(mutex_);
    const AllocationRecord* record = allocations_.FindContaining(addr);
    if (!AllocationTable::IsOffsetValid(record, addr)) {
      attributes->type = cudaMemoryTypeUnregistered;
      attributes->device = -1;
      attributes->devicePointer = nullptr;
      attributes->hostPointer = const_cast<void*>(ptr);
      return cudaSuccess;
    }

    attributes->type = record->memory_type;
    attributes->device = 0;
    attributes->devicePointer = const_cast<void*>(ptr);
    attributes->hostPointer = const_cast<void*>(ptr);
    return cudaSuccess;
  }

  cudaError_t Synchronize() {
    id<MTLCommandQueue> default_queue = nil;
    id<MTLCommandBuffer> default_last_cmd = nil;
    std::vector<std::shared_ptr<StreamState>> active_streams;
    {
      std::unique_lock<std::shared_mutex> lock(mutex_);
      const cudaError_t status = EnsureInitializedLocked();
      if (status != cudaSuccess) {
        return status;
      }
      default_queue = queue_;
      active_streams = streams_.SnapshotAll();

      {
        std::lock_guard<std::mutex> def_lock(default_stream_mutex_);
        FlushDefaultStreamLocked();
        default_last_cmd = default_last_command_buffer_;
      }
      for (const std::shared_ptr<StreamState>& stream : active_streams) {
        if (stream != nullptr) {
          std::lock_guard<std::mutex> stream_lock(stream->mutex);
          FlushStreamLocked(stream.get());
        }
      }
    }

    @autoreleasepool {
      WaitForQueue(default_queue, default_last_cmd, &default_pending_commands_);
      for (const std::shared_ptr<StreamState>& stream : active_streams) {
        if (stream != nullptr) {
          id<MTLCommandBuffer> s_last = nil;
          {
            std::lock_guard<std::mutex> s_lock(stream->mutex);
            s_last = stream->last_command_buffer;
          }
          WaitForQueue(stream->queue, s_last, &stream->pending_commands);
        }
      }
      const cudaError_t prior_error = async_error_.exchange(cudaSuccess, std::memory_order_acq_rel);
      if (prior_error != cudaSuccess) {
        return RecordError(prior_error);
      }
    }
    return cudaSuccess;
  }

  cudaError_t Reset() {
    @autoreleasepool {
      AllocationTable old_allocations;
      KernelRegistry old_kernels;
      id<MTLCommandQueue> old_queue = nil;
      std::vector<std::shared_ptr<StreamState>> old_streams = streams_.DrainAll();
      std::vector<std::shared_ptr<EventState>> old_events = events_.DrainAll();

      {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        old_allocations = std::move(allocations_);
        old_kernels = std::move(kernels_);
        old_queue = queue_;
        queue_ = nil;
        device_ = nil;
        async_error_.store(cudaSuccess, std::memory_order_release);

        {
          std::lock_guard<std::mutex> def_lock(default_stream_mutex_);
          FlushDefaultStreamLocked();
        }
        for (const std::shared_ptr<StreamState>& stream : old_streams) {
          if (stream != nullptr) {
            std::lock_guard<std::mutex> stream_lock(stream->mutex);
            FlushStreamLocked(stream.get());
          }
        }
      }

      WaitForQueue(old_queue);
      for (const std::shared_ptr<StreamState>& stream : old_streams) {
        if (stream != nullptr) {
          WaitForQueue(stream->queue);
        }
      }
      old_streams.clear();
      old_events.clear();
      old_allocations.Clear();
      old_kernels.Clear();
      old_queue = nil;
    }
    return cudaSuccess;
  }

  cudaError_t Allocate(void** dev_ptr, std::size_t size, unsigned int flags,
                       cudaMemoryType memory_type = cudaMemoryTypeDevice) {
    if (dev_ptr == nullptr || size == 0) {
      return RecordError(cudaErrorInvalidValue);
    }
    const std::optional<AttachMode> attach_mode = ParseAttachMode(flags);
    if (!attach_mode.has_value()) {
      return RecordError(cudaErrorInvalidValue);
    }

    id<MTLDevice> device = nil;
    const cudaError_t status = AcquireInitializedDevice(&device);
    if (status != cudaSuccess) {
      return status;
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

      std::unique_lock<std::shared_mutex> lock(mutex_);
      allocations_.Insert(AllocationRecord{
          .base_address = DeviceAddress::FromPointer(contents),
          .size = size,
          .capacity = aligned_size,
          .attach_mode = *attach_mode,
          .memory_type = memory_type,
          .buffer = buffer,
      });

      *dev_ptr = contents;
      return cudaSuccess;
    }
  }

  cudaError_t AllocateHost(void** host_ptr, std::size_t size, unsigned int flags) {
    if (!IsValidHostAllocFlags(flags)) {
      return RecordError(cudaErrorInvalidValue);
    }
    return Allocate(host_ptr, size, cudaMemAttachGlobal, cudaMemoryTypeHost);
  }

  cudaError_t AllocatePitch(void** dev_ptr, std::size_t* pitch, std::size_t width,
                            std::size_t height) {
    if (dev_ptr == nullptr || pitch == nullptr || width == 0 || height == 0) {
      return RecordError(cudaErrorInvalidValue);
    }
    const std::size_t row_pitch = AlignUp(width, kMinAlignment);
    if (row_pitch < width || row_pitch > static_cast<std::size_t>(-1) / height) {
      return RecordError(cudaErrorMemoryAllocation);
    }
    const std::size_t total_bytes = row_pitch * height;
    const cudaError_t status = Allocate(dev_ptr, total_bytes, cudaMemAttachGlobal);
    if (status != cudaSuccess) {
      return status;
    }
    *pitch = row_pitch;
    return cudaSuccess;
  }

  cudaError_t Free(void* dev_ptr) {
    if (dev_ptr == nullptr) {
      return cudaSuccess;
    }

    const DeviceAddress address = DeviceAddress::FromPointer(dev_ptr);
    @autoreleasepool {
      std::optional<AllocationRecord> removed;
      id<MTLCommandQueue> default_queue = nil;
      std::vector<std::shared_ptr<StreamState>> active_streams;
      {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        removed = allocations_.Remove(address);
        if (!removed.has_value()) {
          return RecordError(cudaErrorInvalidValue);
        }
        default_queue = queue_;
        active_streams = streams_.SnapshotAll();

        {
          std::lock_guard<std::mutex> def_lock(default_stream_mutex_);
          FlushDefaultStreamLocked();
        }
        for (const std::shared_ptr<StreamState>& stream : active_streams) {
          if (stream != nullptr) {
            std::lock_guard<std::mutex> stream_lock(stream->mutex);
            FlushStreamLocked(stream.get());
          }
        }
      }

      WaitForQueue(default_queue);
      for (const std::shared_ptr<StreamState>& stream : active_streams) {
        if (stream != nullptr) {
          WaitForQueue(stream->queue);
        }
      }
      removed.reset();
    }
    return cudaSuccess;
  }

  cudaError_t Memcpy(void* dst, const void* src, std::size_t count, cudaMemcpyKind kind) {
    if (dst == nullptr || src == nullptr) {
      return RecordError(cudaErrorInvalidValue);
    }
    if (!IsValidMemcpyKind(kind)) {
      return RecordError(cudaErrorInvalidMemcpyDirection);
    }

    const DeviceAddress dst_addr = DeviceAddress::FromPointer(dst);
    const DeviceAddress src_addr = DeviceAddress::FromPointer(src);

    bool touches_device = false;
    {
      std::shared_lock<std::shared_mutex> lock(mutex_);
      const cudaError_t validation_status =
          ValidateCopyLocked(dst_addr, src_addr, count, kind, &touches_device);
      if (validation_status != cudaSuccess) {
        return validation_status;
      }
    }

    if (touches_device) {
      const cudaError_t sync_status = Synchronize();
      if (sync_status != cudaSuccess) {
        return sync_status;
      }
    }

    std::shared_lock<std::shared_mutex> lock(mutex_);
    const cudaError_t validation_status =
        ValidateCopyLocked(dst_addr, src_addr, count, kind, nullptr);
    if (validation_status != cudaSuccess) {
      return validation_status;
    }

    if (count > 0 && dst != src) {
      const AllocationRecord* dst_rec = allocations_.FindContaining(dst_addr);
      const AllocationRecord* src_rec = allocations_.FindContaining(src_addr);
      if (dst_rec != nullptr && src_rec != nullptr &&
          (dst_rec->buffer != src_rec->buffer || dst_addr.value != src_addr.value)) {
        const std::size_t dst_offset =
            static_cast<std::size_t>(dst_addr.value - dst_rec->base_address.value);
        const std::size_t src_offset =
            static_cast<std::size_t>(src_addr.value - src_rec->base_address.value);
        @autoreleasepool {
          id<MTLCommandBuffer> cmd = [queue_ commandBuffer];
          id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
          if (cmd == nil || blit == nil) {
            return RecordError(cudaErrorLaunchFailure);
          }
          [blit copyFromBuffer:src_rec->buffer
                   sourceOffset:src_offset
                       toBuffer:dst_rec->buffer
              destinationOffset:dst_offset
                           size:count];
          [blit endEncoding];
          [cmd commit];
          [cmd waitUntilCompleted];
          return cudaSuccess;
        }
      }
      std::memmove(dst, src, count);
    }
    return cudaSuccess;
  }

  cudaError_t MemcpyAsync(void* dst, const void* src, std::size_t count, cudaMemcpyKind kind,
                          cudaStream_t stream_handle) {
    if (dst == nullptr || src == nullptr) {
      return RecordError(cudaErrorInvalidValue);
    }
    if (!IsValidMemcpyKind(kind)) {
      return RecordError(cudaErrorInvalidMemcpyDirection);
    }

    std::shared_ptr<StreamState> stream_state;
    id<MTLCommandQueue> target_queue = nil;
    const cudaError_t stream_status =
        ResolveStreamQueue(stream_handle, &stream_state, &target_queue);
    if (stream_status != cudaSuccess) {
      return stream_status;
    }

    const DeviceAddress dst_addr = DeviceAddress::FromPointer(dst);
    const DeviceAddress src_addr = DeviceAddress::FromPointer(src);

    id<MTLBuffer> dst_buffer = nil;
    std::size_t dst_offset = 0;
    id<MTLBuffer> src_buffer = nil;
    std::size_t src_offset = 0;
    bool touches_device = false;

    {
      std::shared_lock<std::shared_mutex> lock(mutex_);
      const cudaError_t validation_status =
          ValidateCopyLocked(dst_addr, src_addr, count, kind, &touches_device);
      if (validation_status != cudaSuccess) {
        return validation_status;
      }
      const AllocationRecord* dst_record = allocations_.FindContaining(dst_addr);
      const AllocationRecord* src_record = allocations_.FindContaining(src_addr);
      if (dst_record != nullptr) {
        dst_buffer = dst_record->buffer;
        dst_offset = static_cast<std::size_t>(dst_addr.value - dst_record->base_address.value);
      }
      if (src_record != nullptr) {
        src_buffer = src_record->buffer;
        src_offset = static_cast<std::size_t>(src_addr.value - src_record->base_address.value);
      }
    }

    if (count == 0 || dst == src) {
      return cudaSuccess;
    }

    if (stream_state != nullptr) {
      std::lock_guard<std::mutex> stream_lock(stream_state->mutex);
      FlushStreamLocked(stream_state.get());
    } else {
      std::lock_guard<std::mutex> def_lock(default_stream_mutex_);
      FlushDefaultStreamLocked();
    }

    // If both endpoints are GPU buffers and do not overlap at identical offsets, encode a
    // stream-ordered Metal blit copy so the host does not block.
    if (dst_buffer != nil && src_buffer != nil &&
        (dst_buffer != src_buffer || dst_offset != src_offset)) {
      @autoreleasepool {
        id<MTLCommandBuffer> cmd = [target_queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
        if (cmd == nil || blit == nil) {
          return RecordError(cudaErrorLaunchFailure);
        }
        [blit copyFromBuffer:src_buffer
                 sourceOffset:src_offset
                     toBuffer:dst_buffer
            destinationOffset:dst_offset
                         size:count];
        [blit endEncoding];
        CommitStreamCommandBuffer(stream_state.get(), cmd);
        return cudaSuccess;
      }
    }

    // Host-involved copy: synchronize the target stream so prior GPU work on that stream is visible
    // before reading/writing unified shared memory.
    if (touches_device) {
      WaitForQueue(target_queue);
      const cudaError_t prior_error = async_error_.exchange(cudaSuccess, std::memory_order_acq_rel);
      if (prior_error != cudaSuccess) {
        return RecordError(prior_error);
      }
    }

    std::shared_lock<std::shared_mutex> lock(mutex_);
    const cudaError_t recheck = ValidateCopyLocked(dst_addr, src_addr, count, kind, nullptr);
    if (recheck != cudaSuccess) {
      return recheck;
    }
    std::memmove(dst, src, count);
    return cudaSuccess;
  }

  cudaError_t Memcpy2D(void* dst, std::size_t dpitch, const void* src, std::size_t spitch,
                       std::size_t width, std::size_t height, cudaMemcpyKind kind,
                       cudaStream_t stream_handle, bool is_async) {
    if (dst == nullptr || src == nullptr) {
      return RecordError(cudaErrorInvalidValue);
    }
    if (!IsValidMemcpyKind(kind)) {
      return RecordError(cudaErrorInvalidMemcpyDirection);
    }
    if (width > dpitch || width > spitch) {
      return RecordError(cudaErrorInvalidPitchValue);
    }

    std::shared_ptr<StreamState> stream_state;
    id<MTLCommandQueue> target_queue = nil;
    if (is_async) {
      const cudaError_t s_status = ResolveStreamQueue(stream_handle, &stream_state, &target_queue);
      if (s_status != cudaSuccess) {
        return s_status;
      }
    } else if (stream_handle != nullptr) {
      return RecordError(cudaErrorInvalidResourceHandle);
    }

    if (width == 0 || height == 0) {
      return cudaSuccess;
    }

    const std::size_t dst_span = (height - 1) * dpitch + width;
    const std::size_t src_span = (height - 1) * spitch + width;
    const DeviceAddress dst_addr = DeviceAddress::FromPointer(dst);
    const DeviceAddress src_addr = DeviceAddress::FromPointer(src);

    bool touches_device = false;
    {
      std::shared_lock<std::shared_mutex> lock(mutex_);
      const cudaError_t status =
          ValidateCopy2DLocked(dst_addr, dst_span, src_addr, src_span, kind, &touches_device);
      if (status != cudaSuccess) {
        return status;
      }
    }

    if (touches_device) {
      if (is_async) {
        WaitForQueue(target_queue);
      } else {
        const cudaError_t sync_status = Synchronize();
        if (sync_status != cudaSuccess) {
          return sync_status;
        }
      }
    }

    std::shared_lock<std::shared_mutex> lock(mutex_);
    const cudaError_t status =
        ValidateCopy2DLocked(dst_addr, dst_span, src_addr, src_span, kind, nullptr);
    if (status != cudaSuccess) {
      return status;
    }

    auto* dst_bytes = static_cast<std::byte*>(dst);
    const auto* src_bytes = static_cast<const std::byte*>(src);
    if (dpitch == width && spitch == width) {
      std::memmove(dst_bytes, src_bytes, width * height);
    } else {
      for (std::size_t row = 0; row < height; ++row) {
        std::memmove(dst_bytes + row * dpitch, src_bytes + row * spitch, width);
      }
    }
    return cudaSuccess;
  }

  cudaError_t Memset(void* dev_ptr, int value, std::size_t count) {
    if (dev_ptr == nullptr) {
      return RecordError(cudaErrorInvalidValue);
    }

    const DeviceAddress addr = DeviceAddress::FromPointer(dev_ptr);

    const cudaError_t sync_status = Synchronize();
    if (sync_status != cudaSuccess) {
      return sync_status;
    }

    std::shared_lock<std::shared_mutex> lock(mutex_);
    const AllocationRecord* record = allocations_.FindContaining(addr);
    if (!AllocationTable::IsRangeValid(record, addr, count)) {
      return RecordError(cudaErrorInvalidValue);
    }

    if (count > 0) {
      std::memset(dev_ptr, value, count);
    }
    return cudaSuccess;
  }

  cudaError_t MemsetAsync(void* dev_ptr, int value, std::size_t count, cudaStream_t stream_handle) {
    if (dev_ptr == nullptr) {
      return RecordError(cudaErrorInvalidValue);
    }

    std::shared_ptr<StreamState> stream_state;
    id<MTLCommandQueue> target_queue = nil;
    const cudaError_t stream_status =
        ResolveStreamQueue(stream_handle, &stream_state, &target_queue);
    if (stream_status != cudaSuccess) {
      return stream_status;
    }

    const DeviceAddress addr = DeviceAddress::FromPointer(dev_ptr);
    id<MTLBuffer> buffer = nil;
    std::size_t offset = 0;

    {
      std::shared_lock<std::shared_mutex> lock(mutex_);
      const AllocationRecord* record = allocations_.FindContaining(addr);
      if (!AllocationTable::IsRangeValid(record, addr, count)) {
        return RecordError(cudaErrorInvalidValue);
      }
      buffer = record->buffer;
      offset = static_cast<std::size_t>(addr.value - record->base_address.value);
    }

    if (count == 0) {
      return cudaSuccess;
    }

    if (stream_state != nullptr) {
      std::lock_guard<std::mutex> stream_lock(stream_state->mutex);
      FlushStreamLocked(stream_state.get());
    } else {
      std::lock_guard<std::mutex> def_lock(default_stream_mutex_);
      FlushDefaultStreamLocked();
    }

    // Metal fillBuffer requires 4-byte aligned offset and length on some OS versions; use
    // fillBuffer when aligned, otherwise synchronize the target stream and memset directly.
    if ((offset % 4u) == 0u && (count % 4u) == 0u) {
      @autoreleasepool {
        id<MTLCommandBuffer> cmd = [target_queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
        if (cmd == nil || blit == nil) {
          return RecordError(cudaErrorLaunchFailure);
        }
        [blit fillBuffer:buffer
                   range:NSMakeRange(offset, count)
                   value:static_cast<std::uint8_t>(value & 0xFF)];
        [blit endEncoding];
        CommitStreamCommandBuffer(stream_state.get(), cmd);
        return cudaSuccess;
      }
    }

    WaitForQueue(target_queue);
    std::shared_lock<std::shared_mutex> lock(mutex_);
    const AllocationRecord* record = allocations_.FindContaining(addr);
    if (!AllocationTable::IsRangeValid(record, addr, count)) {
      return RecordError(cudaErrorInvalidValue);
    }
    std::memset(dev_ptr, value, count);
    return cudaSuccess;
  }

  cudaError_t Memset2D(void* dev_ptr, std::size_t pitch, int value, std::size_t width,
                       std::size_t height) {
    if (dev_ptr == nullptr || pitch == 0 || width > pitch) {
      return RecordError(cudaErrorInvalidValue);
    }
    if (width == 0 || height == 0) {
      return cudaSuccess;
    }

    const std::size_t total_span = (height - 1) * pitch + width;
    const DeviceAddress addr = DeviceAddress::FromPointer(dev_ptr);

    const cudaError_t sync_status = Synchronize();
    if (sync_status != cudaSuccess) {
      return sync_status;
    }

    std::shared_lock<std::shared_mutex> lock(mutex_);
    const AllocationRecord* record = allocations_.FindContaining(addr);
    if (!AllocationTable::IsRangeValid(record, addr, total_span)) {
      return RecordError(cudaErrorInvalidValue);
    }

    auto* dst_bytes = static_cast<std::byte*>(dev_ptr);
    for (std::size_t row = 0; row < height; ++row) {
      std::memset(dst_bytes + row * pitch, value, width);
    }
    return cudaSuccess;
  }

  cudaError_t StreamCreate(cudaStream_t* out_stream, unsigned int flags) {
    if (out_stream == nullptr || !IsValidStreamFlags(flags)) {
      return RecordError(cudaErrorInvalidValue);
    }

    id<MTLDevice> device = nil;
    const cudaError_t status = AcquireInitializedDevice(&device);
    if (status != cudaSuccess) {
      return status;
    }

    @autoreleasepool {
      id<MTLCommandQueue> queue = [device newCommandQueue];
      if (queue == nil) {
        return RecordError(cudaErrorInitializationError);
      }
      auto state = std::make_shared<StreamState>();
      state->queue = queue;
      state->flags = flags;
      *out_stream = streams_.Insert<cudaStream_t>(std::move(state));
      return cudaSuccess;
    }
  }

  cudaError_t StreamDestroy(cudaStream_t stream_handle) {
    if (stream_handle == nullptr) {
      return RecordError(cudaErrorInvalidResourceHandle);
    }
    std::shared_ptr<StreamState> removed = streams_.Remove(stream_handle);
    if (removed == nullptr) {
      return RecordError(cudaErrorInvalidResourceHandle);
    }
    {
      std::lock_guard<std::mutex> stream_lock(removed->mutex);
      FlushStreamLocked(removed.get());
    }
    WaitForQueue(removed->queue);
    return cudaSuccess;
  }

  cudaError_t StreamSynchronize(cudaStream_t stream_handle) {
    if (stream_handle == nullptr) {
      return Synchronize();
    }
    std::shared_ptr<StreamState> stream = streams_.Lookup(stream_handle);
    if (stream == nullptr) {
      return RecordError(cudaErrorInvalidResourceHandle);
    }
    @autoreleasepool {
      id<MTLCommandBuffer> s_last = nil;
      {
        std::lock_guard<std::mutex> stream_lock(stream->mutex);
        FlushStreamLocked(stream.get());
        s_last = stream->last_command_buffer;
      }
      WaitForQueue(stream->queue, s_last, &stream->pending_commands);
      const cudaError_t prior_error = async_error_.exchange(cudaSuccess, std::memory_order_acq_rel);
      if (prior_error != cudaSuccess) {
        return RecordError(prior_error);
      }
      return cudaSuccess;
    }
  }

  cudaError_t StreamQuery(cudaStream_t stream_handle) {
    if (stream_handle == nullptr) {
      const cudaError_t status = EnsureInitialized();
      if (status != cudaSuccess) {
        return status;
      }
      {
        std::lock_guard<std::mutex> def_lock(default_stream_mutex_);
        FlushDefaultStreamLocked();
      }
      if (default_pending_commands_.load(std::memory_order_acquire) > 0) {
        return cudaErrorNotReady;
      }
      return cudaSuccess;
    }
    std::shared_ptr<StreamState> stream = streams_.Lookup(stream_handle);
    if (stream == nullptr) {
      return RecordError(cudaErrorInvalidResourceHandle);
    }
    {
      std::lock_guard<std::mutex> stream_lock(stream->mutex);
      FlushStreamLocked(stream.get());
    }
    if (stream->pending_commands.load(std::memory_order_acquire) > 0) {
      return cudaErrorNotReady;
    }
    return cudaSuccess;
  }

  cudaError_t EventCreate(cudaEvent_t* out_event, unsigned int flags) {
    if (out_event == nullptr || !IsValidEventFlags(flags)) {
      return RecordError(cudaErrorInvalidValue);
    }

    id<MTLDevice> device = nil;
    const cudaError_t status = AcquireInitializedDevice(&device);
    if (status != cudaSuccess) {
      return status;
    }

    @autoreleasepool {
      id<MTLSharedEvent> shared_event = [device newSharedEvent];
      if (shared_event == nil) {
        return RecordError(cudaErrorInitializationError);
      }
      auto state = std::make_shared<EventState>();
      state->shared_event = shared_event;
      state->flags = flags;
      *out_event = events_.Insert<cudaEvent_t>(std::move(state));
      return cudaSuccess;
    }
  }

  cudaError_t EventDestroy(cudaEvent_t event_handle) {
    if (event_handle == nullptr) {
      return RecordError(cudaErrorInvalidResourceHandle);
    }
    std::shared_ptr<EventState> removed = events_.Remove(event_handle);
    if (removed == nullptr) {
      return RecordError(cudaErrorInvalidResourceHandle);
    }
    return cudaSuccess;
  }

  cudaError_t EventRecord(cudaEvent_t event_handle, cudaStream_t stream_handle) {
    std::shared_ptr<EventState> event = events_.Lookup(event_handle);
    if (event == nullptr) {
      return RecordError(cudaErrorInvalidResourceHandle);
    }

    std::shared_ptr<StreamState> stream_state;
    id<MTLCommandQueue> target_queue = nil;
    const cudaError_t stream_status =
        ResolveStreamQueue(stream_handle, &stream_state, &target_queue);
    if (stream_status != cudaSuccess) {
      return stream_status;
    }

    @autoreleasepool {
      if (stream_state != nullptr) {
        std::lock_guard<std::mutex> stream_lock(stream_state->mutex);
        FlushStreamLocked(stream_state.get());
      } else {
        std::lock_guard<std::mutex> def_lock(default_stream_mutex_);
        FlushDefaultStreamLocked();
      }

      id<MTLCommandBuffer> command_buffer = [target_queue commandBuffer];
      if (command_buffer == nil) {
        return RecordError(cudaErrorLaunchFailure);
      }

      std::uint64_t next_ticket = 0;
      {
        std::lock_guard<std::mutex> event_lock(event->mutex);
        next_ticket = ++event->recorded_ticket;
        event->recorded_command_buffer = command_buffer;
        event->recorded.store(true, std::memory_order_release);
      }

      [command_buffer encodeSignalEvent:event->shared_event value:next_ticket];
      std::weak_ptr<EventState> weak_event = event;
      [command_buffer addCompletedHandler:^(id<MTLCommandBuffer> completed) {
        if (auto locked_event = weak_event.lock()) {
          const double gpu_end = completed.GPUEndTime;
          const double stamp = (gpu_end > 0.0) ? gpu_end : NowSeconds();
          locked_event->completion_timestamp_seconds.store(stamp, std::memory_order_release);
        }
      }];
      CommitStreamCommandBuffer(stream_state.get(), command_buffer);
      return cudaSuccess;
    }
  }

  cudaError_t EventSynchronize(cudaEvent_t event_handle) {
    std::shared_ptr<EventState> event = events_.Lookup(event_handle);
    if (event == nullptr) {
      return RecordError(cudaErrorInvalidResourceHandle);
    }

    id<MTLCommandBuffer> cmd = nil;
    {
      std::lock_guard<std::mutex> event_lock(event->mutex);
      if (!event->recorded.load(std::memory_order_acquire)) {
        return cudaSuccess;
      }
      cmd = event->recorded_command_buffer;
    }

    @autoreleasepool {
      if (cmd != nil) {
        [cmd waitUntilCompleted];
        if (event->completion_timestamp_seconds.load(std::memory_order_acquire) <= 0.0) {
          const double gpu_end = cmd.GPUEndTime;
          event->completion_timestamp_seconds.store((gpu_end > 0.0) ? gpu_end : NowSeconds(),
                                                    std::memory_order_release);
        }
      }
    }
    const cudaError_t prior_error = async_error_.exchange(cudaSuccess, std::memory_order_acq_rel);
    if (prior_error != cudaSuccess) {
      return RecordError(prior_error);
    }
    return cudaSuccess;
  }

  cudaError_t EventQuery(cudaEvent_t event_handle) {
    std::shared_ptr<EventState> event = events_.Lookup(event_handle);
    if (event == nullptr) {
      return RecordError(cudaErrorInvalidResourceHandle);
    }

    std::uint64_t ticket = 0;
    {
      std::lock_guard<std::mutex> event_lock(event->mutex);
      if (!event->recorded.load(std::memory_order_acquire)) {
        return cudaSuccess;
      }
      ticket = event->recorded_ticket;
    }

    @autoreleasepool {
      if (event->shared_event.signaledValue < ticket) {
        return cudaErrorNotReady;
      }
      return cudaSuccess;
    }
  }

  cudaError_t EventElapsedTime(float* ms, cudaEvent_t start_handle, cudaEvent_t end_handle) {
    if (ms == nullptr) {
      return RecordError(cudaErrorInvalidValue);
    }
    std::shared_ptr<EventState> start_event = events_.Lookup(start_handle);
    std::shared_ptr<EventState> end_event = events_.Lookup(end_handle);
    if (start_event == nullptr || end_event == nullptr) {
      return RecordError(cudaErrorInvalidResourceHandle);
    }
    if ((start_event->flags & cudaEventDisableTiming) != 0u ||
        (end_event->flags & cudaEventDisableTiming) != 0u) {
      return RecordError(cudaErrorInvalidResourceHandle);
    }
    if (!start_event->recorded.load(std::memory_order_acquire) ||
        !end_event->recorded.load(std::memory_order_acquire)) {
      return RecordError(cudaErrorInvalidResourceHandle);
    }

    if (EventQuery(start_handle) != cudaSuccess || EventQuery(end_handle) != cudaSuccess) {
      return RecordError(cudaErrorNotReady);
    }

    // Ensure command buffers have finalized their completion handlers.
    (void)EventSynchronize(start_handle);
    (void)EventSynchronize(end_handle);

    const double start_sec =
        start_event->completion_timestamp_seconds.load(std::memory_order_acquire);
    const double end_sec = end_event->completion_timestamp_seconds.load(std::memory_order_acquire);
    const double delta_ms = (end_sec >= start_sec) ? ((end_sec - start_sec) * 1000.0) : 0.0;
    *ms = static_cast<float>(delta_ms);
    return cudaSuccess;
  }

  cudaError_t StreamWaitEvent(cudaStream_t stream_handle, cudaEvent_t event_handle,
                              unsigned int flags) {
    if (flags != cudaEventWaitDefault && flags != cudaEventWaitExternal) {
      return RecordError(cudaErrorInvalidValue);
    }
    std::shared_ptr<EventState> event = events_.Lookup(event_handle);
    if (event == nullptr) {
      return RecordError(cudaErrorInvalidResourceHandle);
    }

    std::shared_ptr<StreamState> stream_state;
    id<MTLCommandQueue> target_queue = nil;
    const cudaError_t stream_status =
        ResolveStreamQueue(stream_handle, &stream_state, &target_queue);
    if (stream_status != cudaSuccess) {
      return stream_status;
    }

    std::uint64_t ticket = 0;
    {
      std::lock_guard<std::mutex> event_lock(event->mutex);
      if (!event->recorded.load(std::memory_order_acquire)) {
        return cudaSuccess;
      }
      ticket = event->recorded_ticket;
    }

    @autoreleasepool {
      if (stream_state != nullptr) {
        std::lock_guard<std::mutex> stream_lock(stream_state->mutex);
        FlushStreamLocked(stream_state.get());
      } else {
        std::lock_guard<std::mutex> def_lock(default_stream_mutex_);
        FlushDefaultStreamLocked();
      }

      id<MTLCommandBuffer> command_buffer = [target_queue commandBuffer];
      if (command_buffer == nil) {
        return RecordError(cudaErrorLaunchFailure);
      }
      [command_buffer encodeWaitForEvent:event->shared_event value:ticket];
      CommitStreamCommandBuffer(stream_state.get(), command_buffer);
      return cudaSuccess;
    }
  }

  cudaError_t RegisterKernelFromFile(const void* host_func, const char* metallib_path,
                                     const char* kernel_name, const mudaKernelParamDesc* params,
                                     std::size_t param_count) {
    if (host_func == nullptr || metallib_path == nullptr || metallib_path[0] == '\0' ||
        kernel_name == nullptr || kernel_name[0] == '\0' ||
        !KernelRegistry::ValidateParams(params, param_count)) {
      return RecordError(cudaErrorInvalidValue);
    }

    id<MTLDevice> device = nil;
    const cudaError_t status = AcquireInitializedDevice(&device);
    if (status != cudaSuccess) {
      return status;
    }

    @autoreleasepool {
      NSString* path_str = bridge::ToNSString(metallib_path);
      if (path_str == nil) {
        return RecordError(cudaErrorInvalidValue);
      }
      NSURL* url = [NSURL fileURLWithPath:path_str];
      NSError* error = nil;
      id<MTLLibrary> library = [device newLibraryWithURL:url error:&error];
      if (library == nil) {
        return RecordError(cudaErrorInvalidDeviceFunction);
      }
      return RegisterLibraryKernel(device, library, host_func, kernel_name, params, param_count);
    }
  }

  cudaError_t RegisterKernelFromData(const void* host_func, const void* metallib_data,
                                     std::size_t metallib_size, const char* kernel_name,
                                     const mudaKernelParamDesc* params, std::size_t param_count) {
    if (host_func == nullptr || metallib_data == nullptr || metallib_size == 0 ||
        kernel_name == nullptr || kernel_name[0] == '\0' ||
        !KernelRegistry::ValidateParams(params, param_count)) {
      return RecordError(cudaErrorInvalidValue);
    }

    id<MTLDevice> device = nil;
    const cudaError_t status = AcquireInitializedDevice(&device);
    if (status != cudaSuccess) {
      return status;
    }

    @autoreleasepool {
      dispatch_data_t data = dispatch_data_create(metallib_data, metallib_size, nullptr,
                                                  DISPATCH_DATA_DESTRUCTOR_DEFAULT);
      if (data == nil) {
        return RecordError(cudaErrorMemoryAllocation);
      }
      NSError* error = nil;
      id<MTLLibrary> library = [device newLibraryWithData:data error:&error];
      if (library == nil) {
        return RecordError(cudaErrorInvalidDeviceFunction);
      }
      return RegisterLibraryKernel(device, library, host_func, kernel_name, params, param_count);
    }
  }

  cudaError_t LaunchKernel(const void* func, dim3 grid_dim, dim3 block_dim, void** args,
                           std::size_t shared_mem, cudaStream_t stream_handle) {
    std::shared_ptr<StreamState> stream_state;
    id<MTLCommandQueue> target_queue = nil;
    if (stream_handle != nullptr) {
      stream_state = streams_.Lookup(stream_handle);
      if (stream_state == nullptr) {
        return RecordError(cudaErrorInvalidResourceHandle);
      }
    }
    if (func == nullptr) {
      return RecordError(cudaErrorInvalidDeviceFunction);
    }
    if (grid_dim.x == 0 || grid_dim.y == 0 || grid_dim.z == 0 || block_dim.x == 0 ||
        block_dim.y == 0 || block_dim.z == 0) {
      return RecordError(cudaErrorInvalidConfiguration);
    }

    const cudaError_t init_status = EnsureInitialized();
    if (init_status != cudaSuccess) {
      return init_status;
    }

    @autoreleasepool {
      std::shared_lock<std::shared_mutex> lock(mutex_);
      if (device_ == nil || queue_ == nil) {
        return RecordError(cudaErrorInitializationError);
      }
      target_queue = (stream_state != nullptr) ? stream_state->queue : queue_;

      const RegisteredKernel* kernel = kernels_.Find(func);
      if (kernel == nullptr) {
        return RecordError(cudaErrorInvalidDeviceFunction);
      }

      const cudaError_t config_status = ValidateLaunchConfig(*kernel, block_dim, shared_mem);
      if (config_status != cudaSuccess) {
        return config_status;
      }

      constexpr std::size_t kMaxInlineArgs = 32;
      std::array<BoundArgument, kMaxInlineArgs> inline_args;
      std::vector<BoundArgument> heap_args;
      const std::size_t param_count = kernel->params.size();
      std::span<BoundArgument> bound_span;
      if (param_count <= kMaxInlineArgs) {
        bound_span = std::span<BoundArgument>(inline_args.data(), param_count);
      } else {
        heap_args.resize(param_count);
        bound_span = std::span<BoundArgument>(heap_args.data(), param_count);
      }

      const cudaError_t bind_status = BindLaunchArguments(*kernel, args, bound_span);
      if (bind_status != cudaSuccess) {
        return bind_status;
      }

      return EncodeAndCommitLaunch(*kernel, bound_span, grid_dim, block_dim, shared_mem,
                                   target_queue, stream_state.get());
    }
  }

 private:
  RuntimeContext() = default;

  cudaError_t AcquireInitializedDevice(id<MTLDevice>* out_device) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    const cudaError_t status = EnsureInitializedLocked();
    if (status != cudaSuccess) {
      return status;
    }
    *out_device = device_;
    return cudaSuccess;
  }

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

  cudaError_t ResolveStreamQueue(cudaStream_t stream_handle,
                                 std::shared_ptr<StreamState>* out_stream,
                                 id<MTLCommandQueue>* out_queue) {
    if (stream_handle != nullptr) {
      std::shared_ptr<StreamState> found = streams_.Lookup(stream_handle);
      if (found == nullptr) {
        return RecordError(cudaErrorInvalidResourceHandle);
      }
      *out_stream = found;
      *out_queue = found->queue;
      return cudaSuccess;
    }

    std::unique_lock<std::shared_mutex> lock(mutex_);
    const cudaError_t status = EnsureInitializedLocked();
    if (status != cudaSuccess) {
      return status;
    }
    *out_stream = nullptr;
    *out_queue = queue_;
    return cudaSuccess;
  }

  void CommitStreamCommandBufferLocked(StreamState* stream_state,
                                       id<MTLCommandBuffer> command_buffer) {
    std::atomic<cudaError_t>* async_error_ptr = &async_error_;
    std::atomic<std::uint64_t>* pending_counter =
        (stream_state != nullptr) ? &stream_state->pending_commands : &default_pending_commands_;
    pending_counter->fetch_add(1, std::memory_order_acq_rel);

    if (stream_state != nullptr) {
      stream_state->last_command_buffer = command_buffer;
    } else {
      default_last_command_buffer_ = command_buffer;
    }

    [command_buffer addCompletedHandler:^(id<MTLCommandBuffer> completed_buffer) {
      if (completed_buffer.error != nil) {
        cudaError_t expected = cudaSuccess;
        async_error_ptr->compare_exchange_strong(expected, cudaErrorLaunchFailure,
                                                 std::memory_order_acq_rel);
      }
      pending_counter->fetch_sub(1, std::memory_order_acq_rel);
    }];
    [command_buffer commit];
  }

  void CommitStreamCommandBuffer(StreamState* stream_state, id<MTLCommandBuffer> command_buffer) {
    if (stream_state != nullptr) {
      std::lock_guard<std::mutex> stream_lock(stream_state->mutex);
      CommitStreamCommandBufferLocked(stream_state, command_buffer);
    } else {
      std::lock_guard<std::mutex> def_lock(default_stream_mutex_);
      CommitStreamCommandBufferLocked(nullptr, command_buffer);
    }
  }

  void FlushStreamLocked(StreamState* stream) {
    if (stream == nullptr || stream->active_encoder == nil) {
      return;
    }
    [stream->active_encoder endEncoding];
    stream->active_encoder = nil;
    stream->active_pipeline = nil;
    stream->active_dispatches = 0;
    CommitStreamCommandBufferLocked(stream, stream->active_command_buffer);
    stream->active_command_buffer = nil;
  }

  void FlushDefaultStreamLocked() {
    if (default_active_encoder_ == nil) {
      return;
    }
    [default_active_encoder_ endEncoding];
    default_active_encoder_ = nil;
    default_active_pipeline_ = nil;
    default_active_dispatches_ = 0;
    CommitStreamCommandBufferLocked(nullptr, default_active_command_buffer_);
    default_active_command_buffer_ = nil;
  }

  cudaError_t ValidateCopyLocked(DeviceAddress dst_addr, DeviceAddress src_addr, std::size_t count,
                                 cudaMemcpyKind kind, bool* out_touches_device) const noexcept {
    const AllocationRecord* dst_record = allocations_.FindContaining(dst_addr);
    const AllocationRecord* src_record = allocations_.FindContaining(src_addr);

    if (out_touches_device != nullptr) {
      *out_touches_device = (dst_record != nullptr || src_record != nullptr);
    }

    const bool dst_valid_device = AllocationTable::IsRangeValid(dst_record, dst_addr, count);
    const bool src_valid_device = AllocationTable::IsRangeValid(src_record, src_addr, count);

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
        if ((dst_record != nullptr && !dst_valid_device) ||
            (src_record != nullptr && !src_valid_device)) {
          return RecordError(cudaErrorInvalidValue);
        }
        break;
    }
    return cudaSuccess;
  }

  cudaError_t ValidateCopy2DLocked(DeviceAddress dst_addr, std::size_t dst_span,
                                   DeviceAddress src_addr, std::size_t src_span,
                                   cudaMemcpyKind kind, bool* out_touches_device) const noexcept {
    const AllocationRecord* dst_record = allocations_.FindContaining(dst_addr);
    const AllocationRecord* src_record = allocations_.FindContaining(src_addr);

    if (out_touches_device != nullptr) {
      *out_touches_device = (dst_record != nullptr || src_record != nullptr);
    }

    const bool dst_valid_device = AllocationTable::IsRangeValid(dst_record, dst_addr, dst_span);
    const bool src_valid_device = AllocationTable::IsRangeValid(src_record, src_addr, src_span);

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
        if ((dst_record != nullptr && !dst_valid_device) ||
            (src_record != nullptr && !src_valid_device)) {
          return RecordError(cudaErrorInvalidValue);
        }
        break;
    }
    return cudaSuccess;
  }

  cudaError_t ValidateLaunchConfig(const RegisteredKernel& kernel, dim3 block_dim,
                                   std::size_t shared_mem) const noexcept {
    const MTLSize max_threads = device_.maxThreadsPerThreadgroup;
    if (block_dim.x > max_threads.width || block_dim.y > max_threads.height ||
        block_dim.z > max_threads.depth) {
      return RecordError(cudaErrorInvalidConfiguration);
    }

    const std::uint64_t total_block_threads = static_cast<std::uint64_t>(block_dim.x) *
                                              static_cast<std::uint64_t>(block_dim.y) *
                                              static_cast<std::uint64_t>(block_dim.z);
    if (total_block_threads > kernel.pipeline.maxTotalThreadsPerThreadgroup) {
      return RecordError(cudaErrorInvalidConfiguration);
    }

    if (shared_mem > device_.maxThreadgroupMemoryLength) {
      return RecordError(cudaErrorInvalidConfiguration);
    }
    return cudaSuccess;
  }

  cudaError_t BindLaunchArguments(const RegisteredKernel& kernel, void** args,
                                  std::span<BoundArgument> out_bound) const {
    const std::size_t param_count = kernel.params.size();
    if (param_count > 0 && args == nullptr) {
      return RecordError(cudaErrorInvalidValue);
    }
    if (out_bound.size() != param_count) {
      return RecordError(cudaErrorInvalidValue);
    }

    for (std::size_t index = 0; index < param_count; ++index) {
      if (args[index] == nullptr) {
        return RecordError(cudaErrorInvalidValue);
      }
      const mudaKernelParamDesc& desc = kernel.params[index];
      BoundArgument& bound = out_bound[index];
      bound.kind = desc.kind;

      if (desc.kind == mudaKernelParamBuffer) {
        const void* dev_ptr = nullptr;
        std::memcpy(&dev_ptr, args[index], sizeof(void*));
        if (dev_ptr == nullptr) {
          return RecordError(cudaErrorInvalidValue);
        }
        const DeviceAddress addr = DeviceAddress::FromPointer(dev_ptr);
        const AllocationRecord* record = allocations_.FindContaining(addr);
        if (!AllocationTable::IsOffsetValid(record, addr)) {
          return RecordError(cudaErrorInvalidValue);
        }
        bound.buffer = record->buffer;
        bound.offset = static_cast<std::size_t>(addr.value - record->base_address.value);
      } else {
        bound.bytes = args[index];
        bound.size = desc.size;
      }
    }
    return cudaSuccess;
  }

  cudaError_t EncodeAndCommitLaunch(const RegisteredKernel& kernel,
                                    std::span<const BoundArgument> bound_args, dim3 grid_dim,
                                    dim3 block_dim, std::size_t shared_mem,
                                    id<MTLCommandQueue> target_queue, StreamState* stream_state) {
    constexpr std::size_t kMaxBatchDispatches = 64;

    std::unique_lock<std::mutex> lock;
    if (stream_state != nullptr) {
      lock = std::unique_lock<std::mutex>(stream_state->mutex);
    } else {
      lock = std::unique_lock<std::mutex>(default_stream_mutex_);
    }

    __strong id<MTLCommandBuffer>& active_cmd = (stream_state != nullptr)
                                                    ? stream_state->active_command_buffer
                                                    : default_active_command_buffer_;
    __strong id<MTLComputeCommandEncoder>& active_enc =
        (stream_state != nullptr) ? stream_state->active_encoder : default_active_encoder_;
    __strong id<MTLComputePipelineState>& active_pipe =
        (stream_state != nullptr) ? stream_state->active_pipeline : default_active_pipeline_;
    std::size_t& active_disp =
        (stream_state != nullptr) ? stream_state->active_dispatches : default_active_dispatches_;

    if (active_cmd == nil || active_enc == nil) {
      active_cmd = [target_queue commandBuffer];
      if (active_cmd == nil) {
        return RecordError(cudaErrorLaunchFailure);
      }
      active_enc = [active_cmd computeCommandEncoder];
      if (active_enc == nil) {
        active_cmd = nil;
        return RecordError(cudaErrorLaunchFailure);
      }
      active_pipe = nil;
      active_disp = 0;
    }

    if (active_pipe != kernel.pipeline) {
      [active_enc setComputePipelineState:kernel.pipeline];
      active_pipe = kernel.pipeline;
    }

    if (active_disp > 0) {
      [active_enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }

    for (std::size_t index = 0; index < bound_args.size(); ++index) {
      const BoundArgument& bound = bound_args[index];
      if (bound.kind == mudaKernelParamBuffer) {
        [active_enc setBuffer:bound.buffer offset:bound.offset atIndex:index];
      } else {
        [active_enc setBytes:bound.bytes length:bound.size atIndex:index];
      }
    }

    if (shared_mem > 0) {
      const std::size_t aligned_shared = AlignUp(shared_mem, 16u);
      [active_enc setThreadgroupMemoryLength:aligned_shared atIndex:0];
    }

    [active_enc dispatchThreadgroups:MTLSizeMake(grid_dim.x, grid_dim.y, grid_dim.z)
               threadsPerThreadgroup:MTLSizeMake(block_dim.x, block_dim.y, block_dim.z)];
    active_disp++;

    if (active_disp >= kMaxBatchDispatches) {
      if (stream_state != nullptr) {
        FlushStreamLocked(stream_state);
      } else {
        FlushDefaultStreamLocked();
      }
    }

    return cudaSuccess;
  }

  cudaError_t RegisterLibraryKernel(id<MTLDevice> device, id<MTLLibrary> library,
                                    const void* host_func, const char* kernel_name,
                                    const mudaKernelParamDesc* params, std::size_t param_count) {
    NSString* fn_str = bridge::ToNSString(kernel_name);
    if (fn_str == nil) {
      return RecordError(cudaErrorInvalidValue);
    }
    id<MTLFunction> function = [library newFunctionWithName:fn_str];
    if (function == nil) {
      return RecordError(cudaErrorInvalidDeviceFunction);
    }
    NSError* error = nil;
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function
                                                                                 error:&error];
    if (pipeline == nil) {
      return RecordError(cudaErrorInvalidDeviceFunction);
    }

    const std::span<const mudaKernelParamDesc> param_span =
        param_count > 0 ? std::span<const mudaKernelParamDesc>(params, param_count)
                        : std::span<const mudaKernelParamDesc>();

    std::unique_lock<std::shared_mutex> lock(mutex_);
    kernels_.Insert(host_func, param_span, pipeline);
    return cudaSuccess;
  }

  mutable std::shared_mutex mutex_;
  id<MTLDevice> device_ = nil;
  id<MTLCommandQueue> queue_ = nil;
  AllocationTable allocations_;
  KernelRegistry kernels_;
  HandleTable<StreamState, HandleKind::kStream> streams_;
  HandleTable<EventState, HandleKind::kEvent> events_;
  std::atomic<std::uint64_t> default_pending_commands_{0};
  std::atomic<cudaError_t> async_error_{cudaSuccess};
  id<MTLCommandBuffer> default_last_command_buffer_ = nil;
  id<MTLCommandBuffer> default_active_command_buffer_ = nil;
  id<MTLComputeCommandEncoder> default_active_encoder_ = nil;
  id<MTLComputePipelineState> default_active_pipeline_ = nil;
  std::size_t default_active_dispatches_ = 0;
  std::mutex default_stream_mutex_;
};

template <typename Fn>
cudaError_t InvokeCAbi(Fn&& fn) noexcept {
  @try {
    return fn();
  } @catch (...) {
    return RecordError(cudaErrorInternalError);
  }
}

}  // namespace
}  // namespace muda::runtime::cudart

extern "C" {

cudaError_t cudaGetDeviceCount(int* count) {
  return muda::runtime::cudart::InvokeCAbi(
      [&] { return muda::runtime::cudart::RuntimeContext::Instance().GetDeviceCount(count); });
}

cudaError_t cudaSetDevice(int device) {
  return muda::runtime::cudart::InvokeCAbi(
      [&] { return muda::runtime::cudart::RuntimeContext::Instance().SetDevice(device); });
}

cudaError_t cudaGetDevice(int* device) {
  return muda::runtime::cudart::InvokeCAbi(
      [&] { return muda::runtime::cudart::RuntimeContext::Instance().GetDevice(device); });
}

cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int device) {
  return muda::runtime::cudart::InvokeCAbi([&] {
    return muda::runtime::cudart::RuntimeContext::Instance().GetDeviceProperties(prop, device);
  });
}

cudaError_t cudaDeviceGetAttribute(int* value, cudaDeviceAttr attr, int device) {
  return muda::runtime::cudart::InvokeCAbi([&] {
    return muda::runtime::cudart::RuntimeContext::Instance().GetDeviceAttribute(value, attr,
                                                                                device);
  });
}

cudaError_t cudaDeviceSynchronize(void) {
  return muda::runtime::cudart::InvokeCAbi(
      [&] { return muda::runtime::cudart::RuntimeContext::Instance().Synchronize(); });
}

cudaError_t cudaDeviceReset(void) {
  return muda::runtime::cudart::InvokeCAbi(
      [&] { return muda::runtime::cudart::RuntimeContext::Instance().Reset(); });
}

cudaError_t cudaRuntimeGetVersion(int* runtimeVersion) {
  return muda::runtime::cudart::InvokeCAbi([&] {
    return muda::runtime::cudart::RuntimeContext::Instance().GetRuntimeVersion(runtimeVersion);
  });
}

cudaError_t cudaDriverGetVersion(int* driverVersion) {
  return muda::runtime::cudart::InvokeCAbi([&] {
    return muda::runtime::cudart::RuntimeContext::Instance().GetDriverVersion(driverVersion);
  });
}

cudaError_t cudaMalloc(void** devPtr, size_t size) {
  return muda::runtime::cudart::InvokeCAbi([&] {
    return muda::runtime::cudart::RuntimeContext::Instance().Allocate(
        devPtr, size, cudaMemAttachGlobal, cudaMemoryTypeDevice);
  });
}

cudaError_t cudaMallocManaged(void** devPtr, size_t size, unsigned int flags) {
  return muda::runtime::cudart::InvokeCAbi([&] {
    return muda::runtime::cudart::RuntimeContext::Instance().Allocate(devPtr, size, flags,
                                                                      cudaMemoryTypeManaged);
  });
}

cudaError_t cudaMallocHost(void** ptr, size_t size) {
  return muda::runtime::cudart::InvokeCAbi([&] {
    return muda::runtime::cudart::RuntimeContext::Instance().AllocateHost(ptr, size,
                                                                          cudaHostAllocDefault);
  });
}

cudaError_t cudaHostAlloc(void** pHost, size_t size, unsigned int flags) {
  return muda::runtime::cudart::InvokeCAbi([&] {
    return muda::runtime::cudart::RuntimeContext::Instance().AllocateHost(pHost, size, flags);
  });
}

cudaError_t cudaMallocPitch(void** devPtr, size_t* pitch, size_t width, size_t height) {
  return muda::runtime::cudart::InvokeCAbi([&] {
    return muda::runtime::cudart::RuntimeContext::Instance().AllocatePitch(devPtr, pitch, width,
                                                                           height);
  });
}

cudaError_t cudaFree(void* devPtr) {
  return muda::runtime::cudart::InvokeCAbi(
      [&] { return muda::runtime::cudart::RuntimeContext::Instance().Free(devPtr); });
}

cudaError_t cudaFreeHost(void* ptr) {
  return muda::runtime::cudart::InvokeCAbi(
      [&] { return muda::runtime::cudart::RuntimeContext::Instance().Free(ptr); });
}

cudaError_t cudaMemGetInfo(size_t* free, size_t* total) {
  return muda::runtime::cudart::InvokeCAbi(
      [&] { return muda::runtime::cudart::RuntimeContext::Instance().MemGetInfo(free, total); });
}

cudaError_t cudaPointerGetAttributes(cudaPointerAttributes* attributes, const void* ptr) {
  return muda::runtime::cudart::InvokeCAbi([&] {
    return muda::runtime::cudart::RuntimeContext::Instance().PointerGetAttributes(attributes, ptr);
  });
}

cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) {
  return muda::runtime::cudart::InvokeCAbi([&] {
    return muda::runtime::cudart::RuntimeContext::Instance().Memcpy(dst, src, count, kind);
  });
}

cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind,
                            cudaStream_t stream) {
  return muda::runtime::cudart::InvokeCAbi([&] {
    return muda::runtime::cudart::RuntimeContext::Instance().MemcpyAsync(dst, src, count, kind,
                                                                         stream);
  });
}

cudaError_t cudaMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                         size_t height, cudaMemcpyKind kind) {
  return muda::runtime::cudart::InvokeCAbi([&] {
    return muda::runtime::cudart::RuntimeContext::Instance().Memcpy2D(
        dst, dpitch, src, spitch, width, height, kind, nullptr, false);
  });
}

cudaError_t cudaMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch,
                              size_t width, size_t height, cudaMemcpyKind kind,
                              cudaStream_t stream) {
  return muda::runtime::cudart::InvokeCAbi([&] {
    return muda::runtime::cudart::RuntimeContext::Instance().Memcpy2D(
        dst, dpitch, src, spitch, width, height, kind, stream, true);
  });
}

cudaError_t cudaMemset(void* devPtr, int value, size_t count) {
  return muda::runtime::cudart::InvokeCAbi([&] {
    return muda::runtime::cudart::RuntimeContext::Instance().Memset(devPtr, value, count);
  });
}

cudaError_t cudaMemsetAsync(void* devPtr, int value, size_t count, cudaStream_t stream) {
  return muda::runtime::cudart::InvokeCAbi([&] {
    return muda::runtime::cudart::RuntimeContext::Instance().MemsetAsync(devPtr, value, count,
                                                                         stream);
  });
}

cudaError_t cudaMemset2D(void* devPtr, size_t pitch, int value, size_t width, size_t height) {
  return muda::runtime::cudart::InvokeCAbi([&] {
    return muda::runtime::cudart::RuntimeContext::Instance().Memset2D(devPtr, pitch, value, width,
                                                                      height);
  });
}

cudaError_t cudaStreamCreate(cudaStream_t* pStream) {
  return muda::runtime::cudart::InvokeCAbi([&] {
    return muda::runtime::cudart::RuntimeContext::Instance().StreamCreate(pStream,
                                                                          cudaStreamDefault);
  });
}

cudaError_t cudaStreamCreateWithFlags(cudaStream_t* pStream, unsigned int flags) {
  return muda::runtime::cudart::InvokeCAbi([&] {
    return muda::runtime::cudart::RuntimeContext::Instance().StreamCreate(pStream, flags);
  });
}

cudaError_t cudaStreamDestroy(cudaStream_t stream) {
  return muda::runtime::cudart::InvokeCAbi(
      [&] { return muda::runtime::cudart::RuntimeContext::Instance().StreamDestroy(stream); });
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
  return muda::runtime::cudart::InvokeCAbi(
      [&] { return muda::runtime::cudart::RuntimeContext::Instance().StreamSynchronize(stream); });
}

cudaError_t cudaStreamQuery(cudaStream_t stream) {
  return muda::runtime::cudart::InvokeCAbi(
      [&] { return muda::runtime::cudart::RuntimeContext::Instance().StreamQuery(stream); });
}

cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags) {
  return muda::runtime::cudart::InvokeCAbi([&] {
    return muda::runtime::cudart::RuntimeContext::Instance().StreamWaitEvent(stream, event, flags);
  });
}

cudaError_t cudaEventCreate(cudaEvent_t* event) {
  return muda::runtime::cudart::InvokeCAbi([&] {
    return muda::runtime::cudart::RuntimeContext::Instance().EventCreate(event, cudaEventDefault);
  });
}

cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags) {
  return muda::runtime::cudart::InvokeCAbi(
      [&] { return muda::runtime::cudart::RuntimeContext::Instance().EventCreate(event, flags); });
}

cudaError_t cudaEventDestroy(cudaEvent_t event) {
  return muda::runtime::cudart::InvokeCAbi(
      [&] { return muda::runtime::cudart::RuntimeContext::Instance().EventDestroy(event); });
}

cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
  return muda::runtime::cudart::InvokeCAbi(
      [&] { return muda::runtime::cudart::RuntimeContext::Instance().EventRecord(event, stream); });
}

cudaError_t cudaEventSynchronize(cudaEvent_t event) {
  return muda::runtime::cudart::InvokeCAbi(
      [&] { return muda::runtime::cudart::RuntimeContext::Instance().EventSynchronize(event); });
}

cudaError_t cudaEventQuery(cudaEvent_t event) {
  return muda::runtime::cudart::InvokeCAbi(
      [&] { return muda::runtime::cudart::RuntimeContext::Instance().EventQuery(event); });
}

cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end) {
  return muda::runtime::cudart::InvokeCAbi([&] {
    return muda::runtime::cudart::RuntimeContext::Instance().EventElapsedTime(ms, start, end);
  });
}

cudaError_t mudaRegisterKernelFromFile(const void* hostFunc, const char* metallibPath,
                                       const char* kernelName, const mudaKernelParamDesc* params,
                                       size_t paramCount) {
  return muda::runtime::cudart::InvokeCAbi([&] {
    return muda::runtime::cudart::RuntimeContext::Instance().RegisterKernelFromFile(
        hostFunc, metallibPath, kernelName, params, paramCount);
  });
}

cudaError_t mudaRegisterKernelFromData(const void* hostFunc, const void* metallibData,
                                       size_t metallibSize, const char* kernelName,
                                       const mudaKernelParamDesc* params, size_t paramCount) {
  return muda::runtime::cudart::InvokeCAbi([&] {
    return muda::runtime::cudart::RuntimeContext::Instance().RegisterKernelFromData(
        hostFunc, metallibData, metallibSize, kernelName, params, paramCount);
  });
}

cudaError_t cudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim, void** args,
                             size_t sharedMem, cudaStream_t stream) {
  return muda::runtime::cudart::InvokeCAbi([&] {
    return muda::runtime::cudart::RuntimeContext::Instance().LaunchKernel(func, gridDim, blockDim,
                                                                          args, sharedMem, stream);
  });
}

cudaError_t cudaGetLastError(void) { return muda::runtime::cudart::ConsumeLastError(); }

cudaError_t cudaPeekAtLastError(void) { return muda::runtime::cudart::PeekLastError(); }

const char* cudaGetErrorName(cudaError_t error) { return muda::runtime::cudart::ErrorName(error); }

const char* cudaGetErrorString(cudaError_t error) {
  return muda::runtime::cudart::ErrorString(error);
}

}  // extern "C"

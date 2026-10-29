/**
 * @file cuda_runtime.mm
 * @brief Implements the CUDA Runtime C API over Metal shared buffers and compute pipelines.
 */

#include "cuda_runtime.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <dispatch/dispatch.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr std::size_t kMinAlignment = 256;
constexpr std::size_t kMaxSetBytesSize = 4096;
constexpr std::size_t kMaxBufferBindings = 31;

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

struct RegisteredKernel {
  std::string name;
  std::vector<mudaKernelParamDesc> params;
  id<MTLComputePipelineState> pipeline = nil;
};

struct BoundArgument {
  mudaKernelParamKind kind = mudaKernelParamBuffer;
  id<MTLBuffer> buffer = nil;
  std::size_t offset = 0;
  const void* bytes = nullptr;
  std::size_t size = 0;
};

bool ValidateKernelParams(const mudaKernelParamDesc* params, std::size_t param_count) noexcept {
  if (param_count > kMaxBufferBindings) {
    return false;
  }
  if (param_count > 0 && params == nullptr) {
    return false;
  }
  for (std::size_t index = 0; index < param_count; ++index) {
    const mudaKernelParamDesc& desc = params[index];
    if (desc.kind == mudaKernelParamBuffer) {
      continue;
    }
    if (desc.kind == mudaKernelParamBytes) {
      if (desc.size == 0 || desc.size > kMaxSetBytesSize) {
        return false;
      }
      continue;
    }
    return false;
  }
  return true;
}

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
      const cudaError_t prior_error = async_error_.exchange(cudaSuccess, std::memory_order_acq_rel);
      if (prior_error != cudaSuccess) {
        return RecordError(prior_error);
      }
      if (command_buffer.error != nil) {
        return RecordError(cudaErrorLaunchFailure);
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
      kernels_.clear();
      async_error_.store(cudaSuccess, std::memory_order_release);
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

  cudaError_t RegisterKernelFromFile(const void* host_func, const char* metallib_path,
                                     const char* kernel_name, const mudaKernelParamDesc* params,
                                     std::size_t param_count) {
    if (host_func == nullptr || metallib_path == nullptr || metallib_path[0] == '\0' ||
        kernel_name == nullptr || kernel_name[0] == '\0' ||
        !ValidateKernelParams(params, param_count)) {
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
      NSString* path_str = [NSString stringWithUTF8String:metallib_path];
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
        !ValidateKernelParams(params, param_count)) {
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
                           std::size_t shared_mem, cudaStream_t stream) {
    if (stream != nullptr) {
      return RecordError(cudaErrorInvalidResourceHandle);
    }
    if (func == nullptr) {
      return RecordError(cudaErrorInvalidDeviceFunction);
    }
    if (grid_dim.x == 0 || grid_dim.y == 0 || grid_dim.z == 0 || block_dim.x == 0 ||
        block_dim.y == 0 || block_dim.z == 0) {
      return RecordError(cudaErrorInvalidConfiguration);
    }

    {
      const cudaError_t init_status = EnsureInitialized();
      if (init_status != cudaSuccess) {
        return init_status;
      }
    }

    @autoreleasepool {
      std::shared_lock<std::shared_mutex> lock(mutex_);
      auto kernel_it = kernels_.find(func);
      if (kernel_it == kernels_.end()) {
        return RecordError(cudaErrorInvalidDeviceFunction);
      }
      const RegisteredKernel& kernel = kernel_it->second;

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

      const std::size_t param_count = kernel.params.size();
      if (param_count > 0 && args == nullptr) {
        return RecordError(cudaErrorInvalidValue);
      }

      std::vector<BoundArgument> bound_args(param_count);
      for (std::size_t index = 0; index < param_count; ++index) {
        if (args[index] == nullptr) {
          return RecordError(cudaErrorInvalidValue);
        }
        const mudaKernelParamDesc& desc = kernel.params[index];
        BoundArgument& bound = bound_args[index];
        bound.kind = desc.kind;

        if (desc.kind == mudaKernelParamBuffer) {
          const void* dev_ptr = *static_cast<const void* const*>(args[index]);
          if (dev_ptr == nullptr) {
            return RecordError(cudaErrorInvalidValue);
          }
          const auto addr = reinterpret_cast<std::uintptr_t>(dev_ptr);
          const AllocationRecord* record = FindContainingRecordLocked(addr);
          if (record == nullptr || addr < record->base_address ||
              (addr - record->base_address) >= record->size) {
            return RecordError(cudaErrorInvalidValue);
          }
          bound.buffer = record->buffer;
          bound.offset = static_cast<std::size_t>(addr - record->base_address);
        } else {
          bound.bytes = args[index];
          bound.size = desc.size;
        }
      }

      id<MTLCommandBuffer> command_buffer = [queue_ commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
      if (command_buffer == nil || encoder == nil) {
        return RecordError(cudaErrorLaunchFailure);
      }

      [encoder setComputePipelineState:kernel.pipeline];
      for (std::size_t index = 0; index < param_count; ++index) {
        const BoundArgument& bound = bound_args[index];
        if (bound.kind == mudaKernelParamBuffer) {
          [encoder setBuffer:bound.buffer offset:bound.offset atIndex:index];
        } else {
          [encoder setBytes:bound.bytes length:bound.size atIndex:index];
        }
      }

      if (shared_mem > 0) {
        [encoder setThreadgroupMemoryLength:shared_mem atIndex:0];
      }

      [encoder dispatchThreadgroups:MTLSizeMake(grid_dim.x, grid_dim.y, grid_dim.z)
              threadsPerThreadgroup:MTLSizeMake(block_dim.x, block_dim.y, block_dim.z)];
      [encoder endEncoding];

      std::atomic<cudaError_t>* async_error_ptr = &async_error_;
      [command_buffer addCompletedHandler:^(id<MTLCommandBuffer> completed_buffer) {
        if (completed_buffer.error != nil) {
          cudaError_t expected = cudaSuccess;
          async_error_ptr->compare_exchange_strong(expected, cudaErrorLaunchFailure,
                                                   std::memory_order_acq_rel);
        }
      }];
      [command_buffer commit];
      return cudaSuccess;
    }
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

  cudaError_t RegisterLibraryKernel(id<MTLDevice> device, id<MTLLibrary> library,
                                    const void* host_func, const char* kernel_name,
                                    const mudaKernelParamDesc* params, std::size_t param_count) {
    NSString* fn_str = [NSString stringWithUTF8String:kernel_name];
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

    RegisteredKernel entry;
    entry.name = kernel_name;
    if (param_count > 0) {
      entry.params.assign(params, params + param_count);
    }
    entry.pipeline = pipeline;

    std::unique_lock<std::shared_mutex> lock(mutex_);
    kernels_[host_func] = std::move(entry);
    return cudaSuccess;
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
  std::unordered_map<const void*, RegisteredKernel> kernels_;
  std::atomic<cudaError_t> async_error_{cudaSuccess};
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

cudaError_t mudaRegisterKernelFromFile(const void* hostFunc, const char* metallibPath,
                                       const char* kernelName, const mudaKernelParamDesc* params,
                                       size_t paramCount) {
  @try {
    return RuntimeState::Instance().RegisterKernelFromFile(hostFunc, metallibPath, kernelName,
                                                           params, paramCount);
  } @catch (...) {
    return RecordError(cudaErrorInternalError);
  }
}

cudaError_t mudaRegisterKernelFromData(const void* hostFunc, const void* metallibData,
                                       size_t metallibSize, const char* kernelName,
                                       const mudaKernelParamDesc* params, size_t paramCount) {
  @try {
    return RuntimeState::Instance().RegisterKernelFromData(hostFunc, metallibData, metallibSize,
                                                           kernelName, params, paramCount);
  } @catch (...) {
    return RecordError(cudaErrorInternalError);
  }
}

cudaError_t cudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim, void** args,
                             size_t sharedMem, cudaStream_t stream) {
  @try {
    return RuntimeState::Instance().LaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
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
    case cudaErrorInvalidConfiguration:
      return "cudaErrorInvalidConfiguration";
    case cudaErrorInvalidDevice:
      return "cudaErrorInvalidDevice";
    case cudaErrorInvalidDevicePointer:
      return "cudaErrorInvalidDevicePointer";
    case cudaErrorInvalidMemcpyDirection:
      return "cudaErrorInvalidMemcpyDirection";
    case cudaErrorInvalidDeviceFunction:
      return "cudaErrorInvalidDeviceFunction";
    case cudaErrorNoDevice:
      return "cudaErrorNoDevice";
    case cudaErrorInvalidResourceHandle:
      return "cudaErrorInvalidResourceHandle";
    case cudaErrorLaunchOutOfResources:
      return "cudaErrorLaunchOutOfResources";
    case cudaErrorLaunchFailure:
      return "cudaErrorLaunchFailure";
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
    case cudaErrorInvalidConfiguration:
      return "invalid configuration argument";
    case cudaErrorInvalidDevice:
      return "invalid device ordinal";
    case cudaErrorInvalidDevicePointer:
      return "invalid device pointer";
    case cudaErrorInvalidMemcpyDirection:
      return "invalid memcpy direction";
    case cudaErrorInvalidDeviceFunction:
      return "invalid device function";
    case cudaErrorNoDevice:
      return "no CUDA-capable Metal device is detected";
    case cudaErrorInvalidResourceHandle:
      return "invalid resource handle";
    case cudaErrorLaunchOutOfResources:
      return "too many resources requested for launch";
    case cudaErrorLaunchFailure:
      return "unspecified launch failure";
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

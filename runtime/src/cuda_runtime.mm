/**
 * @file cuda_runtime.mm
 * @brief Implements the CUDA Runtime C API over Metal buffers and compute pipelines.
 */

#include "cuda_runtime.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <dispatch/dispatch.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <new>
#include <optional>
#include <shared_mutex>
#include <span>
#include <utility>
#include <vector>

#include "allocation_table.hpp"
#include "error_state.hpp"
#include "kernel_registry.hpp"
#include "metal_bridge.hpp"

namespace muda::runtime::cudart {
namespace {

constexpr std::size_t kMinAlignment = 256;

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

void WaitForQueue(id<MTLCommandQueue> queue) noexcept {
  if (queue == nil) {
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
    @autoreleasepool {
      AllocationTable old_allocations;
      KernelRegistry old_kernels;
      id<MTLCommandQueue> old_queue = nil;

      {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        old_allocations = std::move(allocations_);
        old_kernels = std::move(kernels_);
        old_queue = queue_;
        queue_ = nil;
        device_ = nil;
        async_error_.store(cudaSuccess, std::memory_order_release);
      }

      WaitForQueue(old_queue);
      old_allocations.Clear();
      old_kernels.Clear();
      old_queue = nil;
    }
    return cudaSuccess;
  }

  cudaError_t Allocate(void** dev_ptr, std::size_t size, unsigned int flags) {
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
          .buffer = buffer,
      });

      *dev_ptr = contents;
      return cudaSuccess;
    }
  }

  cudaError_t Free(void* dev_ptr) {
    if (dev_ptr == nullptr) {
      return cudaSuccess;
    }

    const DeviceAddress address = DeviceAddress::FromPointer(dev_ptr);
    @autoreleasepool {
      std::optional<AllocationRecord> removed;
      id<MTLCommandQueue> queue = nil;
      {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        removed = allocations_.Remove(address);
        if (!removed.has_value()) {
          return RecordError(cudaErrorInvalidValue);
        }
        queue = queue_;
      }

      WaitForQueue(queue);
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
      std::memmove(dst, src, count);
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

    const cudaError_t init_status = EnsureInitialized();
    if (init_status != cudaSuccess) {
      return init_status;
    }

    @autoreleasepool {
      std::shared_lock<std::shared_mutex> lock(mutex_);
      if (device_ == nil || queue_ == nil) {
        return RecordError(cudaErrorInitializationError);
      }

      const RegisteredKernel* kernel = kernels_.Find(func);
      if (kernel == nullptr) {
        return RecordError(cudaErrorInvalidDeviceFunction);
      }

      const cudaError_t config_status = ValidateLaunchConfig(*kernel, block_dim, shared_mem);
      if (config_status != cudaSuccess) {
        return config_status;
      }

      std::vector<BoundArgument> bound_args;
      const cudaError_t bind_status = BindLaunchArguments(*kernel, args, &bound_args);
      if (bind_status != cudaSuccess) {
        return bind_status;
      }

      return EncodeAndCommitLaunch(*kernel, bound_args, grid_dim, block_dim, shared_mem);
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
                                  std::vector<BoundArgument>* out_bound) const {
    const std::size_t param_count = kernel.params.size();
    if (param_count > 0 && args == nullptr) {
      return RecordError(cudaErrorInvalidValue);
    }

    out_bound->resize(param_count);
    for (std::size_t index = 0; index < param_count; ++index) {
      if (args[index] == nullptr) {
        return RecordError(cudaErrorInvalidValue);
      }
      const mudaKernelParamDesc& desc = kernel.params[index];
      BoundArgument& bound = (*out_bound)[index];
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
                                    dim3 block_dim, std::size_t shared_mem) {
    id<MTLCommandBuffer> command_buffer = [queue_ commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (command_buffer == nil || encoder == nil) {
      return RecordError(cudaErrorLaunchFailure);
    }

    [encoder setComputePipelineState:kernel.pipeline];
    for (std::size_t index = 0; index < bound_args.size(); ++index) {
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
  std::atomic<cudaError_t> async_error_{cudaSuccess};
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

cudaError_t cudaDeviceSynchronize(void) {
  return muda::runtime::cudart::InvokeCAbi(
      [&] { return muda::runtime::cudart::RuntimeContext::Instance().Synchronize(); });
}

cudaError_t cudaDeviceReset(void) {
  return muda::runtime::cudart::InvokeCAbi(
      [&] { return muda::runtime::cudart::RuntimeContext::Instance().Reset(); });
}

cudaError_t cudaMalloc(void** devPtr, size_t size) {
  return muda::runtime::cudart::InvokeCAbi([&] {
    return muda::runtime::cudart::RuntimeContext::Instance().Allocate(devPtr, size,
                                                                      cudaMemAttachGlobal);
  });
}

cudaError_t cudaMallocManaged(void** devPtr, size_t size, unsigned int flags) {
  return muda::runtime::cudart::InvokeCAbi([&] {
    return muda::runtime::cudart::RuntimeContext::Instance().Allocate(devPtr, size, flags);
  });
}

cudaError_t cudaFree(void* devPtr) {
  return muda::runtime::cudart::InvokeCAbi(
      [&] { return muda::runtime::cudart::RuntimeContext::Instance().Free(devPtr); });
}

cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) {
  return muda::runtime::cudart::InvokeCAbi([&] {
    return muda::runtime::cudart::RuntimeContext::Instance().Memcpy(dst, src, count, kind);
  });
}

cudaError_t cudaMemset(void* devPtr, int value, size_t count) {
  return muda::runtime::cudart::InvokeCAbi([&] {
    return muda::runtime::cudart::RuntimeContext::Instance().Memset(devPtr, value, count);
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

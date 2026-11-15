/**
 * @file launch_engine.mm
 * @brief Manages kernel registration, argument binding, and compute pipeline dispatches.
 */

#include "launch_engine.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <array>
#include <cstring>
#include <mutex>
#include <span>
#include <vector>

#include "error_state.hpp"
#include "metal_bridge.hpp"

namespace muda::runtime::cudart {
namespace {

constexpr std::size_t kMaxInlineArgs = 32;

struct KernelMruEntry {
  std::uint64_t epoch = 0;
  const void* host_func = nullptr;
  const RegisteredKernel* kernel = nullptr;
};

constexpr std::size_t kKernelMruSize = 4;
thread_local std::array<KernelMruEntry, kKernelMruSize> tl_kernel_mru{};
thread_local std::size_t tl_kernel_mru_cursor = 0;

}  // namespace

cudaError_t LaunchEngine::ValidateLaunchConfig(const RegisteredKernel& kernel, dim3 block_dim,
                                               std::size_t shared_mem) noexcept {
  const std::size_t threads_per_block =
      static_cast<std::size_t>(block_dim.x) * block_dim.y * block_dim.z;
  if (threads_per_block > kernel.max_threads_per_threadgroup) {
    return RecordError(cudaErrorInvalidConfiguration);
  }
  if (shared_mem > kernel.max_threadgroup_memory_bytes) {
    return RecordError(cudaErrorInvalidConfiguration);
  }
  return cudaSuccess;
}

cudaError_t LaunchEngine::BindLaunchArguments(const RegisteredKernel& kernel, void** args,
                                              const MemoryManager& memory,
                                              std::span<ComputeDispatchArg> out_bound) const noexcept {
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
    ComputeDispatchArg& bound = out_bound[index];

    if (desc.kind == mudaKernelParamBuffer) {
      bound.is_buffer = true;
      const void* dev_ptr = nullptr;
      std::memcpy(&dev_ptr, args[index], sizeof(void*));
      if (dev_ptr == nullptr) {
        return RecordError(cudaErrorInvalidValue);
      }
      const DeviceAddress addr = DeviceAddress::FromPointer(dev_ptr);
      if (!memory.ResolveBufferFast(addr, &bound.buffer, &bound.offset)) {
        return RecordError(cudaErrorInvalidValue);
      }
    } else {
      bound.is_buffer = false;
      bound.bytes = args[index];
      bound.size = desc.size;
    }
  }
  return cudaSuccess;
}

cudaError_t LaunchEngine::LaunchKernel(const void* func, dim3 grid_dim, dim3 block_dim, void** args,
                                       std::size_t shared_mem, cudaStream_t stream_handle,
                                       const MemoryManager& memory, StreamManager& streams,
                                       id<MTLCommandQueue> default_queue) noexcept {
  if (func == nullptr) {
    return RecordError(cudaErrorInvalidDeviceFunction);
  }
  if (grid_dim.x == 0 || grid_dim.y == 0 || grid_dim.z == 0 || block_dim.x == 0 ||
      block_dim.y == 0 || block_dim.z == 0) {
    return RecordError(cudaErrorInvalidConfiguration);
  }

  std::shared_ptr<StreamState> stream_state;
  id<MTLCommandQueue> target_queue = default_queue;
  if (stream_handle != nullptr) {
    const cudaError_t s_status =
        streams.ResolveStreamQueue(stream_handle, default_queue, &stream_state, &target_queue);
    if (s_status != cudaSuccess) {
      return s_status;
    }
  } else if (default_queue == nil) {
    return RecordError(cudaErrorInitializationError);
  }

  const RegisteredKernel* kernel = nullptr;
  const std::uint64_t cur_epoch = kernels_.Epoch();
  for (const KernelMruEntry& entry : tl_kernel_mru) {
    if (entry.epoch == cur_epoch && entry.host_func == func && entry.kernel != nullptr) {
      kernel = entry.kernel;
      break;
    }
  }

  std::shared_lock<std::shared_mutex> fallback_lock;
  if (kernel == nullptr) {
    fallback_lock = std::shared_lock<std::shared_mutex>(mutex_);
    kernel = kernels_.Find(func);
    if (kernel == nullptr) {
      return RecordError(cudaErrorInvalidDeviceFunction);
    }
    tl_kernel_mru[tl_kernel_mru_cursor] = {
        .epoch = kernels_.Epoch(),
        .host_func = func,
        .kernel = kernel,
    };
    tl_kernel_mru_cursor = (tl_kernel_mru_cursor + 1) % kKernelMruSize;
  }

  const cudaError_t config_status = ValidateLaunchConfig(*kernel, block_dim, shared_mem);
  if (config_status != cudaSuccess) {
    return config_status;
  }

  std::array<ComputeDispatchArg, kMaxInlineArgs> inline_args;
  const std::size_t param_count = kernel->params.size();
  if (param_count > kMaxInlineArgs) {
    return RecordError(cudaErrorInvalidValue);
  }
  const std::span<ComputeDispatchArg> bound_span(inline_args.data(), param_count);

  const cudaError_t bind_status = BindLaunchArguments(*kernel, args, memory, bound_span);
  if (bind_status != cudaSuccess) {
    return bind_status;
  }

  return streams.EncodeComputeDispatch(
      kernel->pipeline, MTLSizeMake(grid_dim.x, grid_dim.y, grid_dim.z),
      MTLSizeMake(block_dim.x, block_dim.y, block_dim.z), bound_span, shared_mem, target_queue,
      stream_state.get());
}

cudaError_t LaunchEngine::RegisterLibraryKernel(id<MTLDevice> device, id<MTLLibrary> library,
                                                const void* host_func, const char* kernel_name,
                                                const mudaKernelParamDesc* params,
                                                std::size_t param_count) noexcept {
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

cudaError_t LaunchEngine::RegisterKernelFromFile(const void* host_func, const char* metallib_path,
                                                 const char* kernel_name,
                                                 const mudaKernelParamDesc* params,
                                                 std::size_t param_count,
                                                 id<MTLDevice> device) noexcept {
  if (host_func == nullptr || metallib_path == nullptr || metallib_path[0] == '\0' ||
      kernel_name == nullptr || kernel_name[0] == '\0' ||
      !KernelRegistry::ValidateParams(params, param_count)) {
    return RecordError(cudaErrorInvalidValue);
  }
  if (device == nil) {
    return RecordError(cudaErrorInitializationError);
  }
  @autoreleasepool {
    NSString* path_str = bridge::ToNSString(metallib_path);
    if (path_str == nil) {
      return RecordError(cudaErrorInvalidValue);
    }
    NSError* error = nil;
    id<MTLLibrary> library = [device newLibraryWithURL:[NSURL fileURLWithPath:path_str]
                                                 error:&error];
    if (library == nil) {
      return RecordError(cudaErrorInvalidDeviceFunction);
    }
    return RegisterLibraryKernel(device, library, host_func, kernel_name, params, param_count);
  }
}

cudaError_t LaunchEngine::RegisterKernelFromData(const void* host_func, const void* metallib_data,
                                                 std::size_t size, const char* kernel_name,
                                                 const mudaKernelParamDesc* params,
                                                 std::size_t param_count,
                                                 id<MTLDevice> device) noexcept {
  if (host_func == nullptr || metallib_data == nullptr || size == 0 || kernel_name == nullptr ||
      kernel_name[0] == '\0' || !KernelRegistry::ValidateParams(params, param_count)) {
    return RecordError(cudaErrorInvalidValue);
  }
  if (device == nil) {
    return RecordError(cudaErrorInitializationError);
  }
  @autoreleasepool {
    dispatch_data_t data = dispatch_data_create(metallib_data, size, dispatch_get_main_queue(),
                                                DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    if (data == nil) {
      return RecordError(cudaErrorInvalidValue);
    }
    NSError* error = nil;
    id<MTLLibrary> library = [device newLibraryWithData:data error:&error];
    if (library == nil) {
      return RecordError(cudaErrorInvalidDeviceFunction);
    }
    return RegisterLibraryKernel(device, library, host_func, kernel_name, params, param_count);
  }
}

cudaError_t LaunchEngine::RegisterKernelFromSource(const void* host_func, const char* msl_source,
                                                   std::size_t size, const char* kernel_name,
                                                   const mudaKernelParamDesc* params,
                                                   std::size_t param_count,
                                                   id<MTLDevice> device) noexcept {
  if (host_func == nullptr || msl_source == nullptr || size == 0 || kernel_name == nullptr ||
      kernel_name[0] == '\0' || !KernelRegistry::ValidateParams(params, param_count)) {
    return RecordError(cudaErrorInvalidValue);
  }
  if (device == nil) {
    return RecordError(cudaErrorInitializationError);
  }
  @autoreleasepool {
    NSString* source_str = [[NSString alloc] initWithBytes:msl_source
                                                    length:size
                                                  encoding:NSUTF8StringEncoding];
    if (source_str == nil) {
      return RecordError(cudaErrorInvalidValue);
    }
    MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
    options.mathMode = MTLMathModeFast;
    NSError* error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:source_str options:options error:&error];
    if (library == nil) {
      return RecordError(cudaErrorInvalidDeviceFunction);
    }
    return RegisterLibraryKernel(device, library, host_func, kernel_name, params, param_count);
  }
}

cudaError_t LaunchEngine::RegisterKernelPipeline(const void* host_func,
                                                 std::span<const mudaKernelParamDesc> params,
                                                 id<MTLComputePipelineState> pipeline) noexcept {
  if (host_func == nullptr || pipeline == nil) {
    return RecordError(cudaErrorInvalidValue);
  }
  std::unique_lock<std::shared_mutex> lock(mutex_);
  kernels_.Insert(host_func, params, pipeline);
  return cudaSuccess;
}

void LaunchEngine::Clear() noexcept {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  kernels_.Clear();
}

}  // namespace muda::runtime::cudart

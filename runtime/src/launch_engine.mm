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

constexpr std::size_t kMaxBatchDispatches = 64;
constexpr std::size_t kMaxInlineArgs = 32;

}  // namespace

cudaError_t LaunchEngine::ValidateLaunchConfig(const RegisteredKernel& kernel, dim3 block_dim,
                                               std::size_t shared_mem) noexcept {
  const std::size_t threads_per_block =
      static_cast<std::size_t>(block_dim.x) * block_dim.y * block_dim.z;
  const std::size_t max_threads = kernel.pipeline.maxTotalThreadsPerThreadgroup;
  if (threads_per_block > max_threads) {
    return RecordError(cudaErrorInvalidConfiguration);
  }
  if (shared_mem > 0) {
    const std::size_t max_shared = kernel.pipeline.device.maxThreadgroupMemoryLength;
    if (shared_mem > max_shared) {
      return RecordError(cudaErrorInvalidConfiguration);
    }
  }
  return cudaSuccess;
}

cudaError_t LaunchEngine::BindLaunchArguments(const RegisteredKernel& kernel, void** args,
                                              const MemoryManager& memory,
                                              std::span<BoundArgument> out_bound) const noexcept {
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
      if (!memory.ResolveBuffer(addr, &bound.buffer, &bound.offset)) {
        return RecordError(cudaErrorInvalidValue);
      }
    } else {
      bound.bytes = args[index];
      bound.size = desc.size;
    }
  }
  return cudaSuccess;
}

cudaError_t LaunchEngine::EncodeAndCommitLaunch(
    const RegisteredKernel& kernel, std::span<const BoundArgument> bound_args, dim3 grid_dim,
    dim3 block_dim, std::size_t shared_mem, id<MTLCommandQueue> target_queue,
    StreamState* stream_state, StreamManager& streams) noexcept {
  std::unique_lock<std::mutex> lock;
  if (stream_state != nullptr) {
    lock = std::unique_lock<std::mutex>(stream_state->mutex);
  } else {
    lock = std::unique_lock<std::mutex>(streams.DefaultStreamMutex());
  }

  __strong id<MTLCommandBuffer>& active_cmd = (stream_state != nullptr)
                                                  ? stream_state->active_command_buffer
                                                  : streams.DefaultActiveCommandBuffer();
  __strong id<MTLComputeCommandEncoder>& active_enc =
      (stream_state != nullptr) ? stream_state->active_encoder : streams.DefaultActiveEncoder();
  __strong id<MTLComputePipelineState>& active_pipe =
      (stream_state != nullptr) ? stream_state->active_pipeline : streams.DefaultActivePipeline();
  std::size_t& active_disp = (stream_state != nullptr) ? stream_state->active_dispatches
                                                       : streams.DefaultActiveDispatches();

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
      streams.FlushStreamLocked(stream_state);
    } else {
      streams.FlushDefaultStreamLocked();
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
  id<MTLCommandQueue> target_queue = nil;
  const cudaError_t s_status =
      streams.ResolveStreamQueue(stream_handle, default_queue, &stream_state, &target_queue);
  if (s_status != cudaSuccess) {
    return s_status;
  }

  @autoreleasepool {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    const RegisteredKernel* kernel = kernels_.Find(func);
    if (kernel == nullptr) {
      return RecordError(cudaErrorInvalidDeviceFunction);
    }

    const cudaError_t config_status = ValidateLaunchConfig(*kernel, block_dim, shared_mem);
    if (config_status != cudaSuccess) {
      return config_status;
    }

    std::array<BoundArgument, kMaxInlineArgs> inline_args;
    std::vector<BoundArgument> heap_args;
    const std::size_t param_count = kernel->params.size();
    std::span<BoundArgument> bound_span =
        (param_count <= kMaxInlineArgs)
            ? std::span<BoundArgument>(inline_args.data(), param_count)
            : std::span<BoundArgument>((heap_args.resize(param_count), heap_args.data()),
                                       param_count);

    const cudaError_t bind_status = BindLaunchArguments(*kernel, args, memory, bound_span);
    if (bind_status != cudaSuccess) {
      return bind_status;
    }

    return EncodeAndCommitLaunch(*kernel, bound_span, grid_dim, block_dim, shared_mem, target_queue,
                                 stream_state.get(), streams);
  }
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

void LaunchEngine::Clear() noexcept {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  kernels_.Clear();
}

}  // namespace muda::runtime::cudart

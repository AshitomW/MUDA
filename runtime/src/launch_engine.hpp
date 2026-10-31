/**
 * @file launch_engine.hpp
 * @brief Manages kernel registration, argument binding, and compute pipeline dispatches.
 */

#pragma once

#import <Metal/Metal.h>

#include <cstddef>
#include <shared_mutex>
#include <span>

#include "cuda_runtime.h"
#include "kernel_registry.hpp"
#include "memory_manager.hpp"
#include "stream_manager.hpp"

namespace muda::runtime::cudart {

struct BoundArgument {
  mudaKernelParamKind kind = mudaKernelParamBuffer;
  id<MTLBuffer> buffer = nil;
  std::size_t offset = 0;
  const void* bytes = nullptr;
  std::size_t size = 0;
};

class LaunchEngine {
 public:
  LaunchEngine() = default;
  ~LaunchEngine() = default;
  LaunchEngine(const LaunchEngine&) = delete;
  LaunchEngine& operator=(const LaunchEngine&) = delete;

  cudaError_t RegisterKernelFromFile(const void* host_func, const char* metallib_path,
                                     const char* kernel_name, const mudaKernelParamDesc* params,
                                     std::size_t param_count, id<MTLDevice> device) noexcept;

  cudaError_t RegisterKernelFromData(const void* host_func, const void* metallib_data,
                                     std::size_t size, const char* kernel_name,
                                     const mudaKernelParamDesc* params, std::size_t param_count,
                                     id<MTLDevice> device) noexcept;

  cudaError_t LaunchKernel(const void* func, dim3 grid_dim, dim3 block_dim, void** args,
                           std::size_t shared_mem, cudaStream_t stream_handle,
                           const MemoryManager& memory, StreamManager& streams,
                           id<MTLCommandQueue> default_queue) noexcept;

  void Clear() noexcept;

 private:
  cudaError_t RegisterLibraryKernel(id<MTLDevice> device, id<MTLLibrary> library,
                                    const void* host_func, const char* kernel_name,
                                    const mudaKernelParamDesc* params,
                                    std::size_t param_count) noexcept;

  [[nodiscard]] static cudaError_t ValidateLaunchConfig(const RegisteredKernel& kernel,
                                                        dim3 block_dim,
                                                        std::size_t shared_mem) noexcept;

  cudaError_t BindLaunchArguments(const RegisteredKernel& kernel, void** args,
                                  const MemoryManager& memory,
                                  std::span<BoundArgument> out_bound) const noexcept;

  cudaError_t EncodeAndCommitLaunch(const RegisteredKernel& kernel,
                                    std::span<const BoundArgument> bound_args, dim3 grid_dim,
                                    dim3 block_dim, std::size_t shared_mem,
                                    id<MTLCommandQueue> target_queue, StreamState* stream_state,
                                    StreamManager& streams) noexcept;

  mutable std::shared_mutex mutex_;
  KernelRegistry kernels_;
};

}  // namespace muda::runtime::cudart

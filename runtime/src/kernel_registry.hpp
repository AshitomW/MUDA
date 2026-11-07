/**
 * @file kernel_registry.hpp
 * @brief Caches Metal compute pipelines and parameter descriptors for registered kernels.
 */

#pragma once

#import <Metal/Metal.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <span>
#include <unordered_map>
#include <vector>

#include "cuda_runtime.h"

namespace muda::runtime::cudart {

inline constexpr std::size_t kMaxSetBytesSize = 4096;
inline constexpr std::size_t kMaxBufferBindings = 31;

struct RegisteredKernel {
  std::vector<mudaKernelParamDesc> params;
  id<MTLComputePipelineState> pipeline = nil;
  std::size_t max_threads_per_threadgroup = 1024;
  std::size_t max_threadgroup_memory_bytes = 32768;
};

class KernelRegistry {
 public:
  [[nodiscard]] static bool ValidateParams(const mudaKernelParamDesc* params,
                                           std::size_t param_count) noexcept;

  void Insert(const void* host_func, std::span<const mudaKernelParamDesc> params,
              id<MTLComputePipelineState> pipeline);
  [[nodiscard]] const RegisteredKernel* Find(const void* host_func) const noexcept;
  [[nodiscard]] std::uint64_t Epoch() const noexcept {
    return epoch_.load(std::memory_order_acquire);
  }
  void Clear();

 private:
  std::unordered_map<const void*, RegisteredKernel> kernels_;
  std::atomic<std::uint64_t> epoch_{1};
};

}  // namespace muda::runtime::cudart

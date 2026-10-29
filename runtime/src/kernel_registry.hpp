/**
 * @file kernel_registry.hpp
 * @brief Caches Metal compute pipelines and parameter descriptors for registered kernels.
 */

#pragma once

#include "cuda_runtime.h"

#include <cstddef>
#include <span>
#include <unordered_map>
#include <vector>

#import <Metal/Metal.h>

namespace muda::runtime::cudart {

inline constexpr std::size_t kMaxSetBytesSize = 4096;
inline constexpr std::size_t kMaxBufferBindings = 31;

struct RegisteredKernel {
  std::vector<mudaKernelParamDesc> params;
  id<MTLComputePipelineState> pipeline = nil;
};

class KernelRegistry {
 public:
  [[nodiscard]] static bool ValidateParams(const mudaKernelParamDesc* params,
                                           std::size_t param_count) noexcept;

  void Insert(const void* host_func, std::span<const mudaKernelParamDesc> params,
              id<MTLComputePipelineState> pipeline);
  [[nodiscard]] const RegisteredKernel* Find(const void* host_func) const noexcept;
  void Clear();

 private:
  std::unordered_map<const void*, RegisteredKernel> kernels_;
};

}  // namespace muda::runtime::cudart


/**
 * @file kernel_registry.mm
 * @brief Caches Metal compute pipelines and parameter descriptors for registered kernels.
 */

#include "kernel_registry.hpp"

#include <utility>

namespace muda::runtime::cudart {

bool KernelRegistry::ValidateParams(const mudaKernelParamDesc* params,
                                    std::size_t param_count) noexcept {
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

void KernelRegistry::Insert(const void* host_func, std::span<const mudaKernelParamDesc> params,
                            id<MTLComputePipelineState> pipeline) {
  RegisteredKernel entry;
  entry.params.assign(params.begin(), params.end());
  entry.pipeline = pipeline;
  if (pipeline != nil) {
    entry.max_threads_per_threadgroup = pipeline.maxTotalThreadsPerThreadgroup;
    if (pipeline.device != nil) {
      entry.max_threadgroup_memory_bytes = pipeline.device.maxThreadgroupMemoryLength;
    }
  }
  kernels_[host_func] = std::move(entry);
  epoch_.fetch_add(1, std::memory_order_release);
}

const RegisteredKernel* KernelRegistry::Find(const void* host_func) const noexcept {
  auto it = kernels_.find(host_func);
  if (it == kernels_.end()) {
    return nullptr;
  }
  return &it->second;
}

void KernelRegistry::Clear() {
  kernels_.clear();
  epoch_.fetch_add(1, std::memory_order_release);
}

}  // namespace muda::runtime::cudart

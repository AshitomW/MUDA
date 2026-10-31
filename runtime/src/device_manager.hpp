/**
 * @file device_manager.hpp
 * @brief Manages Metal device discovery and caches device properties.
 */

#pragma once

#import <Metal/Metal.h>

#include <shared_mutex>

#include "cuda_runtime.h"

namespace muda::runtime::cudart {

class DeviceManager {
 public:
  DeviceManager() = default;
  ~DeviceManager() = default;
  DeviceManager(const DeviceManager&) = delete;
  DeviceManager& operator=(const DeviceManager&) = delete;

  cudaError_t EnsureInitialized() noexcept {
    if (initialized_.load(std::memory_order_acquire)) {
      return cudaSuccess;
    }
    return EnsureInitializedSlow();
  }

  [[nodiscard]] id<MTLDevice> Device() const noexcept { return device_; }
  [[nodiscard]] id<MTLCommandQueue> DefaultQueue() const noexcept { return default_queue_; }

  [[nodiscard]] cudaError_t GetDeviceCount(int* count) const noexcept;
  cudaError_t SetDevice(int device) noexcept;
  [[nodiscard]] cudaError_t GetDevice(int* device) const noexcept;
  [[nodiscard]] cudaError_t GetDeviceProperties(cudaDeviceProp* prop,
                                                int device_ordinal) const noexcept;
  [[nodiscard]] cudaError_t GetDeviceAttribute(int* value, cudaDeviceAttr attr,
                                               int device_ordinal) const noexcept;
  [[nodiscard]] cudaError_t GetRuntimeVersion(int* version) const noexcept;
  [[nodiscard]] cudaError_t GetDriverVersion(int* version) const noexcept;

  void Reset() noexcept;

 private:
  cudaError_t EnsureInitializedSlow() noexcept;
  void PopulateProperties(id<MTLDevice> device) noexcept;

  mutable std::shared_mutex mutex_;
  id<MTLDevice> device_ = nil;
  id<MTLCommandQueue> default_queue_ = nil;
  cudaDeviceProp cached_prop_{};
  std::atomic<bool> initialized_{false};
};

}  // namespace muda::runtime::cudart

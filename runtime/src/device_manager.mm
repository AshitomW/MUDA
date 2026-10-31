/**
 * @file device_manager.mm
 * @brief Manages Metal device discovery and caches device properties.
 */

#include "device_manager.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cstring>
#include <mutex>
#include <string>

#include "error_state.hpp"
#include "metal_bridge.hpp"

namespace muda::runtime::cudart {
namespace {

constexpr int kEmulatedCudaRuntimeVersion = 12040;
constexpr int kEmulatedCudaDriverVersion = 12040;
constexpr int kDefaultComputeMajor = 8;
constexpr int kDefaultComputeMinor = 0;
constexpr int kDefaultWarpSize = 32;
constexpr int kDefaultMaxGridDimX = 2147483647;
constexpr int kDefaultMaxGridDimYZ = 65535;
constexpr int kDefaultClockRateKhz = 1000000;
constexpr std::size_t kDefaultTotalConstMem = 65536;

void PopulateIdentityAndLimits(id<MTLDevice> device, cudaDeviceProp& prop) noexcept {
  const std::string name = bridge::ToStdString(device.name);
  const std::size_t copy_len = std::min(name.size(), sizeof(prop.name) - 1);
  std::memcpy(prop.name, name.data(), copy_len);
  prop.name[copy_len] = '\0';

  const MTLSize max_threads = device.maxThreadsPerThreadgroup;
  prop.maxThreadsPerBlock = static_cast<int>(max_threads.width);
  prop.maxThreadsDim[0] = static_cast<int>(max_threads.width);
  prop.maxThreadsDim[1] = static_cast<int>(max_threads.height);
  prop.maxThreadsDim[2] = static_cast<int>(max_threads.depth);
  prop.maxGridSize[0] = kDefaultMaxGridDimX;
  prop.maxGridSize[1] = kDefaultMaxGridDimYZ;
  prop.maxGridSize[2] = kDefaultMaxGridDimYZ;
  prop.maxThreadsPerMultiProcessor = static_cast<int>(max_threads.width) * 2;
  prop.warpSize = kDefaultWarpSize;
}

void PopulateMemoryAndCompute(id<MTLDevice> device, cudaDeviceProp& prop) noexcept {
  prop.totalGlobalMem = static_cast<std::size_t>(device.recommendedMaxWorkingSetSize);
  prop.sharedMemPerBlock = static_cast<std::size_t>(device.maxThreadgroupMemoryLength);
  prop.sharedMemPerMultiprocessor = static_cast<std::size_t>(device.maxThreadgroupMemoryLength);
  prop.sharedMemPerBlockOptin = static_cast<std::size_t>(device.maxThreadgroupMemoryLength);
  prop.regsPerBlock = 65536;
  prop.memPitch = static_cast<std::size_t>(device.maxBufferLength);
  prop.clockRate = kDefaultClockRateKhz;
  prop.totalConstMem = kDefaultTotalConstMem;
  prop.major = kDefaultComputeMajor;
  prop.minor = kDefaultComputeMinor;
  prop.multiProcessorCount = 8;
  prop.integrated = 1;
  prop.canMapHostMemory = 1;
  prop.computeMode = cudaComputeModeDefault;
  prop.concurrentKernels = 1;
  prop.unifiedAddressing = 1;
  prop.memoryClockRate = kDefaultClockRateKhz;
  prop.memoryBusWidth = 128;
  prop.l2CacheSize = 4 * 1024 * 1024;
  prop.managedMemory = 1;
  prop.concurrentManagedAccess = 1;
  prop.pageableMemoryAccess = 1;
}

bool QueryDimensionAttribute(cudaDeviceAttr attr, const cudaDeviceProp& prop, int* value) noexcept {
  switch (attr) {
    case cudaDevAttrMaxThreadsPerBlock:
      *value = prop.maxThreadsPerBlock;
      return true;
    case cudaDevAttrMaxBlockDimX:
      *value = prop.maxThreadsDim[0];
      return true;
    case cudaDevAttrMaxBlockDimY:
      *value = prop.maxThreadsDim[1];
      return true;
    case cudaDevAttrMaxBlockDimZ:
      *value = prop.maxThreadsDim[2];
      return true;
    case cudaDevAttrMaxGridDimX:
      *value = prop.maxGridSize[0];
      return true;
    case cudaDevAttrMaxGridDimY:
      *value = prop.maxGridSize[1];
      return true;
    case cudaDevAttrMaxGridDimZ:
      *value = prop.maxGridSize[2];
      return true;
    case cudaDevAttrWarpSize:
      *value = prop.warpSize;
      return true;
    default:
      return false;
  }
}

bool QueryMemoryAttribute(cudaDeviceAttr attr, const cudaDeviceProp& prop, int* value) noexcept {
  switch (attr) {
    case cudaDevAttrMaxSharedMemoryPerBlock:
    case cudaDevAttrMaxSharedMemoryPerBlockOptin:
      *value = static_cast<int>(prop.sharedMemPerBlock);
      return true;
    case cudaDevAttrTotalConstantMemory:
      *value = static_cast<int>(prop.totalConstMem);
      return true;
    case cudaDevAttrMaxPitch:
      *value = static_cast<int>(
          std::min<std::size_t>(prop.memPitch, static_cast<std::size_t>(2147483647)));
      return true;
    case cudaDevAttrUnifiedAddressing:
      *value = prop.unifiedAddressing;
      return true;
    case cudaDevAttrManagedMemory:
      *value = prop.managedMemory;
      return true;
    case cudaDevAttrHostNativeAtomicSupported:
    case cudaDevAttrPageableMemoryAccess:
    case cudaDevAttrConcurrentManagedAccess:
      *value = 1;
      return true;
    default:
      return false;
  }
}

bool QueryComputeAttribute(cudaDeviceAttr attr, const cudaDeviceProp& prop, int* value) noexcept {
  switch (attr) {
    case cudaDevAttrClockRate:
      *value = prop.clockRate;
      return true;
    case cudaDevAttrMultiProcessorCount:
      *value = prop.multiProcessorCount;
      return true;
    case cudaDevAttrComputeMode:
      *value = prop.computeMode;
      return true;
    case cudaDevAttrConcurrentKernels:
      *value = prop.concurrentKernels;
      return true;
    case cudaDevAttrComputeCapabilityMajor:
      *value = prop.major;
      return true;
    case cudaDevAttrComputeCapabilityMinor:
      *value = prop.minor;
      return true;
    default:
      return false;
  }
}

}  // namespace

void DeviceManager::PopulateProperties(id<MTLDevice> device) noexcept {
  std::memset(&cached_prop_, 0, sizeof(cudaDeviceProp));
  PopulateIdentityAndLimits(device, cached_prop_);
  PopulateMemoryAndCompute(device, cached_prop_);
}

cudaError_t DeviceManager::EnsureInitializedSlow() noexcept {
  std::unique_lock<std::shared_mutex> write_lock(mutex_);
  if (initialized_.load(std::memory_order_relaxed)) {
    return cudaSuccess;
  }
  @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (dev == nil) {
      return RecordError(cudaErrorNoDevice);
    }
    id<MTLCommandQueue> queue = [dev newCommandQueue];
    if (queue == nil) {
      return RecordError(cudaErrorInitializationError);
    }
    device_ = dev;
    default_queue_ = queue;
    PopulateProperties(dev);
    initialized_.store(true, std::memory_order_release);
    return cudaSuccess;
  }
}

cudaError_t DeviceManager::GetDeviceCount(int* count) const noexcept {
  if (count == nullptr) {
    return RecordError(cudaErrorInvalidValue);
  }
  @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    *count = (dev != nil) ? 1 : 0;
    return cudaSuccess;
  }
}

cudaError_t DeviceManager::SetDevice(int device) noexcept {
  if (device != 0) {
    return RecordError(cudaErrorInvalidDevice);
  }
  return EnsureInitialized();
}

cudaError_t DeviceManager::GetDevice(int* device) const noexcept {
  if (device == nullptr) {
    return RecordError(cudaErrorInvalidValue);
  }
  if (!initialized_) {
    return RecordError(cudaErrorInitializationError);
  }
  *device = 0;
  return cudaSuccess;
}

cudaError_t DeviceManager::GetDeviceProperties(cudaDeviceProp* prop,
                                               int device_ordinal) const noexcept {
  if (prop == nullptr) {
    return RecordError(cudaErrorInvalidValue);
  }
  if (device_ordinal != 0) {
    return RecordError(cudaErrorInvalidDevice);
  }
  std::shared_lock<std::shared_mutex> lock(mutex_);
  if (!initialized_) {
    return RecordError(cudaErrorInitializationError);
  }
  std::memcpy(prop, &cached_prop_, sizeof(cudaDeviceProp));
  return cudaSuccess;
}

cudaError_t DeviceManager::GetDeviceAttribute(int* value, cudaDeviceAttr attr,
                                              int device_ordinal) const noexcept {
  if (value == nullptr) {
    return RecordError(cudaErrorInvalidValue);
  }
  if (device_ordinal != 0) {
    return RecordError(cudaErrorInvalidDevice);
  }
  std::shared_lock<std::shared_mutex> lock(mutex_);
  if (!initialized_) {
    return RecordError(cudaErrorInitializationError);
  }
  if (QueryDimensionAttribute(attr, cached_prop_, value) ||
      QueryMemoryAttribute(attr, cached_prop_, value) ||
      QueryComputeAttribute(attr, cached_prop_, value)) {
    return cudaSuccess;
  }
  return RecordError(cudaErrorInvalidValue);
}

cudaError_t DeviceManager::GetRuntimeVersion(int* version) const noexcept {
  if (version == nullptr) {
    return RecordError(cudaErrorInvalidValue);
  }
  *version = kEmulatedCudaRuntimeVersion;
  return cudaSuccess;
}

cudaError_t DeviceManager::GetDriverVersion(int* version) const noexcept {
  if (version == nullptr) {
    return RecordError(cudaErrorInvalidValue);
  }
  *version = kEmulatedCudaDriverVersion;
  return cudaSuccess;
}

void DeviceManager::Reset() noexcept {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  device_ = nil;
  default_queue_ = nil;
  std::memset(&cached_prop_, 0, sizeof(cudaDeviceProp));
  initialized_.store(false, std::memory_order_release);
}

}  // namespace muda::runtime::cudart

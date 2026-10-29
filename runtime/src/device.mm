/**
 * @file device.mm
 * @brief Implements Metal device discovery and buffer allocation.
 */

#include "muda/runtime/device.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdexcept>
#include <utility>

#include "muda/runtime/buffer.hpp"
#include "storage.hpp"

namespace muda::runtime {
namespace {

DeviceInfo MakeInfo(id<MTLDevice> device) {
  DeviceInfo info;
  const char* name = device.name.UTF8String;
  info.name = name != nullptr ? name : "";
  info.backend = BackendKind::kMetal;
  info.recommended_working_set_bytes =
      static_cast<std::uint64_t>(device.recommendedMaxWorkingSetSize);
  info.max_threads_per_threadgroup =
      static_cast<std::uint32_t>(device.maxThreadsPerThreadgroup.width);
  return info;
}

}  // namespace

Device::Device() = default;
Device::~Device() = default;
Device::Device(Device&&) noexcept = default;
Device& Device::operator=(Device&&) noexcept = default;

Device::Device(std::shared_ptr<DeviceStorage> storage) : storage_(std::move(storage)) {}

Device Device::Default() {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
      throw std::runtime_error("no Metal device available");
    }
    auto storage = std::make_shared<DeviceStorage>();
    storage->device = device;
    return Device(std::move(storage));
  }
}

bool Device::valid() const { return storage_ != nullptr && storage_->device != nil; }

DeviceInfo Device::info() const {
  if (!valid()) {
    throw std::runtime_error("invalid Metal device");
  }
  @autoreleasepool {
    return MakeInfo(storage_->device);
  }
}

Buffer Device::Allocate(std::size_t bytes) const {
  if (!valid()) {
    throw std::runtime_error("invalid Metal device");
  }
  if (bytes == 0) {
    throw std::runtime_error("cannot allocate an empty buffer");
  }
  @autoreleasepool {
    id<MTLBuffer> buffer = [storage_->device newBufferWithLength:bytes
                                                         options:MTLResourceStorageModeShared];
    if (buffer == nil) {
      throw std::runtime_error("Metal buffer allocation failed");
    }
    auto storage = std::make_shared<BufferStorage>();
    storage->buffer = buffer;
    storage->size = bytes;
    return Buffer(std::move(storage));
  }
}

std::vector<DeviceInfo> EnumerateDevices() {
  std::vector<DeviceInfo> result;
  @autoreleasepool {
    NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
    for (id<MTLDevice> device in devices) {
      result.push_back(MakeInfo(device));
    }
  }
  return result;
}

}  // namespace muda::runtime

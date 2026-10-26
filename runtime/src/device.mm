#include "muda/runtime/device.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <utility>

namespace muda::runtime {

std::vector<DeviceInfo> EnumerateDevices() {
  std::vector<DeviceInfo> result;

  @autoreleasepool {
    NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
    for (id<MTLDevice> device in devices) {
      DeviceInfo info;
      const char* name = device.name.UTF8String;
      info.name = name != nullptr ? name : "";
      info.backend = "metal";
      info.recommended_working_set_bytes =
          static_cast<std::uint64_t>(device.recommendedMaxWorkingSetSize);
      info.max_threads_per_threadgroup =
          static_cast<std::uint32_t>(device.maxThreadsPerThreadgroup.width);
      result.push_back(std::move(info));
    }
  }

  return result;
}

}

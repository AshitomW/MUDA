/**
 * @file storage.hpp
 * @brief Internal Metal storage types shared by runtime translation units.
 */

#pragma once

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstddef>

namespace muda::runtime {

struct DeviceStorage {
  id<MTLDevice> device = nil;
};

struct BufferStorage {
  id<MTLBuffer> buffer = nil;
  std::size_t size = 0;
};

struct KernelStorage {
  id<MTLDevice> device = nil;
  id<MTLComputePipelineState> pipeline = nil;
  id<MTLCommandQueue> queue = nil;
};

}

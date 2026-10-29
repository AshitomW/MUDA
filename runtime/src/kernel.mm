/**
 * @file kernel.mm
 * @brief Implements Metal library loading and compute dispatch.
 */

#include "muda/runtime/kernel.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdexcept>
#include <string>
#include <utility>

#include "metal_bridge.hpp"
#include "muda/runtime/device.hpp"
#include "storage.hpp"

namespace muda::runtime {
namespace {

void ValidateDispatchArguments(const KernelStorage* storage,
                               std::span<const KernelArgument> arguments, DispatchSize size) {
  if (storage == nullptr || storage->pipeline == nil || storage->queue == nil) {
    throw std::runtime_error("invalid Metal kernel");
  }
  if (size.threads == 0) {
    throw std::runtime_error("dispatch requires at least one thread");
  }
  if (size.threads_per_threadgroup == 0) {
    throw std::runtime_error("dispatch requires at least one thread per threadgroup");
  }
  if (size.threads_per_threadgroup > storage->pipeline.maxTotalThreadsPerThreadgroup) {
    throw std::runtime_error("threadgroup size exceeds Metal limits");
  }

  for (const KernelArgument& argument : arguments) {
    if (argument.buffer != nullptr) {
      if (!argument.buffer->valid()) {
        throw std::runtime_error("invalid Metal buffer argument");
      }
      continue;
    }
    if (argument.bytes == nullptr || argument.size == 0) {
      throw std::runtime_error("invalid Metal scalar argument");
    }
  }
}

}  // namespace

Kernel::Kernel() = default;
Kernel::~Kernel() = default;
Kernel::Kernel(Kernel&&) noexcept = default;
Kernel& Kernel::operator=(Kernel&&) noexcept = default;

Kernel::Kernel(std::unique_ptr<KernelStorage> storage) : storage_(std::move(storage)) {}

Kernel Kernel::Load(const Device& device, const std::filesystem::path& metallib, std::string name) {
  if (!device.valid()) {
    throw std::runtime_error("invalid Metal device");
  }
  if (metallib.empty()) {
    throw std::runtime_error("empty Metal library path");
  }
  if (name.empty()) {
    throw std::runtime_error("empty Metal kernel name");
  }

  @autoreleasepool {
    NSError* error = nil;
    NSString* path = bridge::ToNSString(metallib.string().c_str());
    if (path == nil) {
      throw std::runtime_error("invalid Metal library path");
    }

    NSURL* url = [NSURL fileURLWithPath:path];
    id<MTLLibrary> library = [device.storage_->device newLibraryWithURL:url error:&error];
    if (library == nil) {
      throw std::runtime_error("cannot load Metal library: " + bridge::FormatNSError(error));
    }

    NSString* function_name = bridge::ToNSString(name.c_str());
    if (function_name == nil) {
      throw std::runtime_error("invalid Metal kernel name");
    }

    id<MTLFunction> function = [library newFunctionWithName:function_name];
    if (function == nil) {
      throw std::runtime_error("Metal kernel not found: " + name);
    }

    id<MTLComputePipelineState> pipeline =
        [device.storage_->device newComputePipelineStateWithFunction:function error:&error];
    if (pipeline == nil) {
      throw std::runtime_error("cannot create Metal pipeline: " + bridge::FormatNSError(error));
    }

    id<MTLCommandQueue> queue = [device.storage_->device newCommandQueue];
    if (queue == nil) {
      throw std::runtime_error("cannot create Metal command queue");
    }

    auto storage = std::make_unique<KernelStorage>();
    storage->device = device.storage_->device;
    storage->pipeline = pipeline;
    storage->queue = queue;
    return Kernel(std::move(storage));
  }
}

void Kernel::Dispatch(std::span<const KernelArgument> arguments, DispatchSize size) const {
  ValidateDispatchArguments(storage_.get(), arguments, size);

  @autoreleasepool {
    id<MTLCommandBuffer> command_buffer = [storage_->queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (command_buffer == nil || encoder == nil) {
      throw std::runtime_error("cannot create Metal command encoder");
    }

    [encoder setComputePipelineState:storage_->pipeline];

    for (const KernelArgument& argument : arguments) {
      if (argument.buffer != nullptr) {
        [encoder setBuffer:argument.buffer->storage_->buffer offset:0 atIndex:argument.index];
      } else {
        [encoder setBytes:argument.bytes length:argument.size atIndex:argument.index];
      }
    }

    [encoder dispatchThreads:MTLSizeMake(size.threads, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(size.threads_per_threadgroup, 1, 1)];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];

    if (command_buffer.error != nil) {
      throw std::runtime_error("Metal dispatch failed: " +
                               bridge::FormatNSError(command_buffer.error));
    }
  }
}

}  // namespace muda::runtime

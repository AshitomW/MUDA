/**
 * @file stream_types.hpp
 * @brief State structures for Metal streams and shared events.
 */

#pragma once

#import <Metal/Metal.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>

#include "cuda_runtime.h"

namespace muda::runtime::cudart {

struct StreamState {
  id<MTLCommandQueue> queue = nil;
  unsigned int flags = cudaStreamDefault;
  mutable std::mutex mutex;
  id<MTLCommandBuffer> last_command_buffer = nil;
  id<MTLCommandBuffer> active_command_buffer = nil;
  id<MTLComputeCommandEncoder> active_encoder = nil;
  id<MTLComputePipelineState> active_pipeline = nil;
  std::size_t active_dispatches = 0;
  std::atomic<std::uint64_t> pending_commands{0};
};

struct EventState {
  id<MTLSharedEvent> shared_event = nil;
  unsigned int flags = cudaEventDefault;
  mutable std::mutex mutex;
  std::uint64_t recorded_ticket = 0;
  id<MTLCommandBuffer> recorded_command_buffer = nil;
  std::atomic<bool> recorded{false};
  std::atomic<double> completion_timestamp_seconds{0.0};
};

void WaitForQueue(id<MTLCommandQueue> queue, id<MTLCommandBuffer> last_cmd = nil,
                  std::atomic<std::uint64_t>* pending = nullptr) noexcept;

}  // namespace muda::runtime::cudart


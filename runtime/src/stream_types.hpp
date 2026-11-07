/**
 * @file stream_types.hpp
 * @brief State structures for Metal streams and shared events.
 */

#pragma once

#import <Metal/Metal.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>

#include "cuda_runtime.h"

namespace muda::runtime::cudart {

inline constexpr std::size_t kMaxEncoderBufferSlots = 31;
inline constexpr std::size_t kMaxDirtyBuffers = 16;

struct BoundBufferSlot {
  const void* buffer_ptr = nullptr;
  std::size_t offset = 0;
  bool valid = false;
};

struct DirtyBufferEntry {
  const void* buffer_ptr = nullptr;
  std::size_t offset = 0;
  std::size_t slot = 0;
};

struct EncoderHazardState {
  std::array<BoundBufferSlot, kMaxEncoderBufferSlots> slots{};
  std::array<DirtyBufferEntry, kMaxDirtyBuffers> dirty_buffers{};
  std::size_t dirty_count = 0;

  void Reset() noexcept {
    for (auto& slot : slots) {
      slot.valid = false;
    }
    dirty_count = 0;
  }

  void ClearHazards() noexcept { dirty_count = 0; }

  [[nodiscard]] bool HasHazard(const void* buf_ptr, std::size_t offset, std::size_t slot,
                               bool pipeline_changed) const noexcept {
    for (std::size_t i = 0; i < dirty_count; ++i) {
      if (dirty_buffers[i].buffer_ptr == buf_ptr) {
        if (pipeline_changed || dirty_buffers[i].slot != slot ||
            dirty_buffers[i].offset != offset) {
          return true;
        }
      }
    }
    return false;
  }

  void MarkDirty(const void* buf_ptr, std::size_t offset, std::size_t slot) noexcept {
    for (std::size_t i = 0; i < dirty_count; ++i) {
      if (dirty_buffers[i].buffer_ptr == buf_ptr && dirty_buffers[i].slot == slot &&
          dirty_buffers[i].offset == offset) {
        return;
      }
    }
    if (dirty_count < kMaxDirtyBuffers) {
      dirty_buffers[dirty_count++] = {
          .buffer_ptr = buf_ptr,
          .offset = offset,
          .slot = slot,
      };
    }
  }
};

struct StreamState {
  id<MTLCommandQueue> queue = nil;
  unsigned int flags = cudaStreamDefault;
  mutable std::mutex mutex;
  id<MTLCommandBuffer> last_command_buffer = nil;
  id<MTLCommandBuffer> active_command_buffer = nil;
  id<MTLComputeCommandEncoder> active_encoder = nil;
  id<MTLBlitCommandEncoder> active_blit_encoder = nil;
  id<MTLComputePipelineState> active_pipeline = nil;
  std::size_t active_dispatches = 0;
  EncoderHazardState encoder_state{};
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

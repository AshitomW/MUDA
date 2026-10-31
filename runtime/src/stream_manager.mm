/**
 * @file stream_manager.mm
 * @brief Manages Metal command queues, execution streams, and batching.
 */

#include "stream_manager.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <utility>

#include "error_state.hpp"

namespace muda::runtime::cudart {
namespace {

[[nodiscard]] constexpr bool IsValidStreamFlags(unsigned int flags) noexcept {
  return flags == cudaStreamDefault || flags == cudaStreamNonBlocking;
}

}  // namespace

void WaitForQueue(id<MTLCommandQueue> queue, id<MTLCommandBuffer> last_cmd,
                  std::atomic<std::uint64_t>* pending) noexcept {
  if (queue == nil) {
    return;
  }
  if (pending != nullptr && pending->load(std::memory_order_acquire) == 0) {
    return;
  }
  if (last_cmd != nil) {
    [last_cmd waitUntilCompleted];
    return;
  }
  @autoreleasepool {
    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    if (command_buffer != nil) {
      [command_buffer commit];
      [command_buffer waitUntilCompleted];
    }
  }
}

cudaError_t StreamManager::StreamCreate(cudaStream_t* out_stream, unsigned int flags,
                                        id<MTLDevice> device) noexcept {
  if (out_stream == nullptr || !IsValidStreamFlags(flags)) {
    return RecordError(cudaErrorInvalidValue);
  }
  if (device == nil) {
    return RecordError(cudaErrorInitializationError);
  }
  @autoreleasepool {
    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (queue == nil) {
      return RecordError(cudaErrorInitializationError);
    }
    auto state = std::make_shared<StreamState>();
    state->queue = queue;
    state->flags = flags;
    *out_stream = streams_.Insert<cudaStream_t>(std::move(state));
    return cudaSuccess;
  }
}

cudaError_t StreamManager::StreamDestroy(cudaStream_t stream_handle) noexcept {
  if (stream_handle == nullptr) {
    return RecordError(cudaErrorInvalidResourceHandle);
  }
  std::shared_ptr<StreamState> state = streams_.Remove(stream_handle);
  if (state == nullptr) {
    return RecordError(cudaErrorInvalidResourceHandle);
  }
  @autoreleasepool {
    id<MTLCommandBuffer> s_last = nil;
    {
      std::lock_guard<std::mutex> stream_lock(state->mutex);
      FlushStreamLocked(state.get());
      s_last = state->last_command_buffer;
    }
    WaitForQueue(state->queue, s_last, &state->pending_commands);
  }
  return cudaSuccess;
}

cudaError_t StreamManager::StreamSynchronize(cudaStream_t stream_handle,
                                             id<MTLCommandQueue> default_queue) noexcept {
  if (stream_handle == nullptr) {
    return SynchronizeDevice(default_queue);
  }
  std::shared_ptr<StreamState> stream = streams_.Lookup(stream_handle);
  if (stream == nullptr) {
    return RecordError(cudaErrorInvalidResourceHandle);
  }
  @autoreleasepool {
    id<MTLCommandBuffer> s_last = nil;
    {
      std::lock_guard<std::mutex> stream_lock(stream->mutex);
      FlushStreamLocked(stream.get());
      s_last = stream->last_command_buffer;
    }
    WaitForQueue(stream->queue, s_last, &stream->pending_commands);
    return ConsumeAsyncError();
  }
}

cudaError_t StreamManager::StreamQuery(cudaStream_t stream_handle) noexcept {
  if (stream_handle == nullptr) {
    {
      std::lock_guard<std::mutex> def_lock(default_stream_mutex_);
      FlushDefaultStreamLocked();
    }
    return (default_pending_commands_.load(std::memory_order_acquire) > 0) ? cudaErrorNotReady
                                                                           : cudaSuccess;
  }
  std::shared_ptr<StreamState> stream = streams_.Lookup(stream_handle);
  if (stream == nullptr) {
    return RecordError(cudaErrorInvalidResourceHandle);
  }
  {
    std::lock_guard<std::mutex> stream_lock(stream->mutex);
    FlushStreamLocked(stream.get());
  }
  return (stream->pending_commands.load(std::memory_order_acquire) > 0) ? cudaErrorNotReady
                                                                        : cudaSuccess;
}

cudaError_t StreamManager::SynchronizeDevice(id<MTLCommandQueue> default_queue) noexcept {
  if (default_queue == nil) {
    return RecordError(cudaErrorInitializationError);
  }
  id<MTLCommandBuffer> default_last_cmd = nil;
  std::vector<std::shared_ptr<StreamState>> active_streams = streams_.SnapshotAll();

  {
    std::lock_guard<std::mutex> def_lock(default_stream_mutex_);
    FlushDefaultStreamLocked();
    default_last_cmd = default_last_command_buffer_;
  }
  for (const std::shared_ptr<StreamState>& stream : active_streams) {
    if (stream != nullptr) {
      std::lock_guard<std::mutex> stream_lock(stream->mutex);
      FlushStreamLocked(stream.get());
    }
  }

  @autoreleasepool {
    WaitForQueue(default_queue, default_last_cmd, &default_pending_commands_);
    for (const std::shared_ptr<StreamState>& stream : active_streams) {
      if (stream != nullptr) {
        id<MTLCommandBuffer> s_last = nil;
        {
          std::lock_guard<std::mutex> s_lock(stream->mutex);
          s_last = stream->last_command_buffer;
        }
        WaitForQueue(stream->queue, s_last, &stream->pending_commands);
      }
    }
    return ConsumeAsyncError();
  }
}

cudaError_t StreamManager::ResolveStreamQueue(cudaStream_t stream_handle,
                                              id<MTLCommandQueue> default_queue,
                                              std::shared_ptr<StreamState>* out_stream,
                                              id<MTLCommandQueue>* out_queue) noexcept {
  if (out_stream == nullptr || out_queue == nullptr) {
    return RecordError(cudaErrorInvalidValue);
  }
  if (stream_handle != nullptr) {
    std::shared_ptr<StreamState> state = streams_.Lookup(stream_handle);
    if (state == nullptr) {
      return RecordError(cudaErrorInvalidResourceHandle);
    }
    *out_stream = state;
    *out_queue = state->queue;
    return cudaSuccess;
  }
  if (default_queue == nil) {
    return RecordError(cudaErrorInitializationError);
  }
  *out_stream = nullptr;
  *out_queue = default_queue;
  return cudaSuccess;
}

std::shared_ptr<StreamState> StreamManager::Lookup(cudaStream_t stream_handle) const noexcept {
  return streams_.Lookup(stream_handle);
}

std::vector<std::shared_ptr<StreamState>> StreamManager::SnapshotAll() const {
  return streams_.SnapshotAll();
}

std::vector<std::shared_ptr<StreamState>> StreamManager::DrainAll() { return streams_.DrainAll(); }

void StreamManager::CommitStreamCommandBufferLocked(StreamState* stream_state,
                                                    id<MTLCommandBuffer> command_buffer) noexcept {
  std::atomic<cudaError_t>* async_error_ptr = &async_error_;
  std::atomic<std::uint64_t>* pending_counter =
      (stream_state != nullptr) ? &stream_state->pending_commands : &default_pending_commands_;
  pending_counter->fetch_add(1, std::memory_order_acq_rel);

  if (stream_state != nullptr) {
    stream_state->last_command_buffer = command_buffer;
  } else {
    default_last_command_buffer_ = command_buffer;
  }

  [command_buffer addCompletedHandler:^(id<MTLCommandBuffer> completed_buffer) {
    if (completed_buffer.error != nil) {
      cudaError_t expected = cudaSuccess;
      async_error_ptr->compare_exchange_strong(expected, cudaErrorLaunchFailure,
                                               std::memory_order_acq_rel);
    }
    pending_counter->fetch_sub(1, std::memory_order_acq_rel);
  }];
  [command_buffer commit];
}

void StreamManager::CommitStreamCommandBuffer(StreamState* stream,
                                              id<MTLCommandBuffer> command_buffer) noexcept {
  if (stream != nullptr) {
    std::lock_guard<std::mutex> stream_lock(stream->mutex);
    CommitStreamCommandBufferLocked(stream, command_buffer);
  } else {
    std::lock_guard<std::mutex> def_lock(default_stream_mutex_);
    CommitStreamCommandBufferLocked(nullptr, command_buffer);
  }
}

void StreamManager::FlushStreamLocked(StreamState* stream) noexcept {
  if (stream == nullptr || stream->active_encoder == nil) {
    return;
  }
  [stream->active_encoder endEncoding];
  stream->active_encoder = nil;
  stream->active_pipeline = nil;
  stream->active_dispatches = 0;
  CommitStreamCommandBufferLocked(stream, stream->active_command_buffer);
  stream->active_command_buffer = nil;
}

void StreamManager::FlushDefaultStreamLocked() noexcept {
  if (default_active_encoder_ == nil) {
    return;
  }
  [default_active_encoder_ endEncoding];
  default_active_encoder_ = nil;
  default_active_pipeline_ = nil;
  default_active_dispatches_ = 0;
  CommitStreamCommandBufferLocked(nullptr, default_active_command_buffer_);
  default_active_command_buffer_ = nil;
}

void StreamManager::RecordAsyncError(cudaError_t error) noexcept {
  cudaError_t expected = cudaSuccess;
  async_error_.compare_exchange_strong(expected, error, std::memory_order_acq_rel);
}

cudaError_t StreamManager::ConsumeAsyncError() noexcept {
  const cudaError_t prior = async_error_.exchange(cudaSuccess, std::memory_order_acq_rel);
  if (prior != cudaSuccess) {
    return RecordError(prior);
  }
  return cudaSuccess;
}

void StreamManager::Reset() noexcept {
  std::vector<std::shared_ptr<StreamState>> old_streams = streams_.DrainAll();
  {
    std::lock_guard<std::mutex> def_lock(default_stream_mutex_);
    FlushDefaultStreamLocked();
  }
  for (const std::shared_ptr<StreamState>& stream : old_streams) {
    if (stream != nullptr) {
      std::lock_guard<std::mutex> stream_lock(stream->mutex);
      FlushStreamLocked(stream.get());
    }
  }
  for (const std::shared_ptr<StreamState>& stream : old_streams) {
    if (stream != nullptr) {
      WaitForQueue(stream->queue);
    }
  }
  async_error_.store(cudaSuccess, std::memory_order_release);
}

}  // namespace muda::runtime::cudart

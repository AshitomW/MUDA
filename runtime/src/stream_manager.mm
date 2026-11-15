/**
 * @file stream_manager.mm
 * @brief Manages Metal command queues, execution streams, and batching.
 */

#include "stream_manager.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/runtime.h>

#include <utility>

#include "error_state.hpp"

namespace muda::runtime::cudart {
namespace {

constexpr std::size_t kMaxBatchDispatches = 256;

[[nodiscard]] constexpr std::size_t AlignUp16(std::size_t value) noexcept {
  const std::size_t rem = value % 16u;
  return rem == 0 ? value : (value + (16u - rem));
}

struct EncoderImpCache {
  Class cls = Nil;
  void (*set_buffer)(id, SEL, id<MTLBuffer>, NSUInteger, NSUInteger) = nullptr;
  void (*set_buffer_offset)(id, SEL, NSUInteger, NSUInteger) = nullptr;
  void (*set_bytes)(id, SEL, const void*, NSUInteger, NSUInteger) = nullptr;
  void (*dispatch)(id, SEL, MTLSize, MTLSize) = nullptr;
};

EncoderImpCache g_encoder_imps{};
std::mutex g_encoder_imp_mutex;

void EnsureEncoderImps(id<MTLComputeCommandEncoder> enc, EncoderImpCache* out_imps) noexcept {
  Class cls = object_getClass(enc);
  if (g_encoder_imps.cls == cls && g_encoder_imps.dispatch != nullptr) {
    *out_imps = g_encoder_imps;
    return;
  }
  std::lock_guard<std::mutex> lock(g_encoder_imp_mutex);
  if (g_encoder_imps.cls != cls) {
    g_encoder_imps.set_buffer =
        reinterpret_cast<void (*)(id, SEL, id<MTLBuffer>, NSUInteger, NSUInteger)>(
            class_getMethodImplementation(cls, @selector(setBuffer:offset:atIndex:)));
    g_encoder_imps.set_buffer_offset =
        reinterpret_cast<void (*)(id, SEL, NSUInteger, NSUInteger)>(
            class_getMethodImplementation(cls, @selector(setBufferOffset:atIndex:)));
    g_encoder_imps.set_bytes =
        reinterpret_cast<void (*)(id, SEL, const void*, NSUInteger, NSUInteger)>(
            class_getMethodImplementation(cls, @selector(setBytes:length:atIndex:)));
    g_encoder_imps.dispatch =
        reinterpret_cast<void (*)(id, SEL, MTLSize, MTLSize)>(
            class_getMethodImplementation(cls, @selector(dispatchThreadgroups:threadsPerThreadgroup:)));
    g_encoder_imps.cls = cls;
  }
  *out_imps = g_encoder_imps;
}

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
    active_stream_count_.fetch_add(1, std::memory_order_release);
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
  active_stream_count_.fetch_sub(1, std::memory_order_release);
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
    if (default_queue == nil) {
      return RecordError(cudaErrorInitializationError);
    }
    if (active_stream_count_.load(std::memory_order_acquire) == 0) {
      if (default_active_dispatches_.load(std::memory_order_relaxed) == 0 &&
          default_pending_commands_.load(std::memory_order_acquire) == 0) {
        return ConsumeAsyncError();
      }
      @autoreleasepool {
        id<MTLCommandBuffer> def_last = nil;
        {
          std::lock_guard<std::mutex> def_lock(default_stream_mutex_);
          FlushDefaultStreamLocked();
          def_last = default_last_command_buffer_;
        }
        WaitForQueue(default_queue, def_last, &default_pending_commands_);
        return ConsumeAsyncError();
      }
    }
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
    if (default_active_dispatches_.load(std::memory_order_relaxed) > 0) {
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
  if (!HasPendingWork()) {
    return ConsumeAsyncError();
  }

  id<MTLCommandBuffer> default_last_cmd = nil;
  {
    std::lock_guard<std::mutex> def_lock(default_stream_mutex_);
    FlushDefaultStreamLocked();
    default_last_cmd = default_last_command_buffer_;
  }

  if (active_stream_count_.load(std::memory_order_acquire) == 0) {
    @autoreleasepool {
      WaitForQueue(default_queue, default_last_cmd, &default_pending_commands_);
      return ConsumeAsyncError();
    }
  }

  std::vector<std::shared_ptr<StreamState>> active_streams = streams_.SnapshotAll();
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

void StreamManager::EndActiveEncodersLocked(StreamState* stream) noexcept {
  if (stream != nullptr) {
    if (stream->active_encoder != nil) {
      [stream->active_encoder endEncoding];
      stream->active_encoder = nil;
      stream->active_pipeline = nil;
      stream->encoder_state.Reset();
    }
    if (stream->active_blit_encoder != nil) {
      [stream->active_blit_encoder endEncoding];
      stream->active_blit_encoder = nil;
    }
  } else {
    if (default_active_encoder_ != nil) {
      [default_active_encoder_ endEncoding];
      default_active_encoder_ = nil;
      default_active_pipeline_ = nil;
      default_encoder_state_.Reset();
    }
    if (default_active_blit_encoder_ != nil) {
      [default_active_blit_encoder_ endEncoding];
      default_active_blit_encoder_ = nil;
    }
  }
}

void StreamManager::FlushStreamLocked(StreamState* stream) noexcept {
  if (stream == nullptr || stream->active_command_buffer == nil) {
    return;
  }
  EndActiveEncodersLocked(stream);
  stream->active_dispatches = 0;
  CommitStreamCommandBufferLocked(stream, stream->active_command_buffer);
  stream->active_command_buffer = nil;
}

void StreamManager::FlushDefaultStreamLocked() noexcept {
  if (default_active_command_buffer_ == nil) {
    return;
  }
  EndActiveEncodersLocked(nullptr);
  default_active_dispatches_count_ = 0;
  default_active_dispatches_.store(0, std::memory_order_relaxed);
  CommitStreamCommandBufferLocked(nullptr, default_active_command_buffer_);
  default_active_command_buffer_ = nil;
}

cudaError_t StreamManager::EncodeComputeDispatch(
    id<MTLComputePipelineState> pipeline, MTLSize grid_size, MTLSize block_size,
    std::span<const ComputeDispatchArg> args, std::size_t shared_mem_bytes,
    id<MTLCommandQueue> target_queue, StreamState* stream_state) noexcept {
  if (pipeline == nil || target_queue == nil) {
    return RecordError(cudaErrorInvalidConfiguration);
  }

  std::unique_lock<std::mutex> lock =
      (stream_state != nullptr) ? std::unique_lock<std::mutex>(stream_state->mutex)
                                : std::unique_lock<std::mutex>(default_stream_mutex_);

  __strong id<MTLCommandBuffer>& active_cmd =
      (stream_state != nullptr) ? stream_state->active_command_buffer : default_active_command_buffer_;
  __strong id<MTLComputeCommandEncoder>& active_enc =
      (stream_state != nullptr) ? stream_state->active_encoder : default_active_encoder_;
  __strong id<MTLBlitCommandEncoder>& active_blit =
      (stream_state != nullptr) ? stream_state->active_blit_encoder : default_active_blit_encoder_;
  __strong id<MTLComputePipelineState>& active_pipe =
      (stream_state != nullptr) ? stream_state->active_pipeline : default_active_pipeline_;
  std::size_t& active_disp =
      (stream_state != nullptr) ? stream_state->active_dispatches : default_active_dispatches_count_;
  EncoderHazardState& enc_state =
      (stream_state != nullptr) ? stream_state->encoder_state : default_encoder_state_;

  if (active_blit != nil) {
    [active_blit endEncoding];
    active_blit = nil;
  }

  if (active_cmd == nil) {
    active_cmd = [target_queue commandBufferWithUnretainedReferences];
    if (active_cmd == nil) {
      return RecordError(cudaErrorLaunchFailure);
    }
    active_disp = 0;
  }

  if (active_enc == nil) {
    active_enc = [active_cmd computeCommandEncoder];
    if (active_enc == nil) {
      active_cmd = nil;
      return RecordError(cudaErrorLaunchFailure);
    }
    active_pipe = nil;
    enc_state.Reset();
  }

  const bool pipeline_changed = (active_pipe != nil && active_pipe != pipeline);
  if (active_pipe != pipeline) {
    [active_enc setComputePipelineState:pipeline];
    active_pipe = pipeline;
  }

  if (active_disp > 0 && enc_state.dirty_count > 0) {
    bool needs_barrier = false;
    for (std::size_t idx = 0; idx < args.size(); ++idx) {
      const ComputeDispatchArg& arg = args[idx];
      if (arg.is_buffer) {
        const void* buf_ptr = (__bridge const void*)arg.buffer;
        if (enc_state.HasHazard(buf_ptr, arg.offset, idx, pipeline_changed)) {
          needs_barrier = true;
          break;
        }
      }
    }
    if (needs_barrier) {
      [active_enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
      enc_state.ClearHazards();
    }
  }

  EncoderImpCache imps;
  EnsureEncoderImps(active_enc, &imps);

  for (std::size_t idx = 0; idx < args.size(); ++idx) {
    const ComputeDispatchArg& arg = args[idx];
    if (arg.is_buffer) {
      const void* buf_ptr = (__bridge const void*)arg.buffer;
      if (idx < kMaxEncoderBufferSlots) {
        BoundBufferSlot& slot = enc_state.slots[idx];
        if (slot.valid && slot.buffer_ptr == buf_ptr) {
          if (slot.offset != arg.offset) {
            imps.set_buffer_offset(active_enc, @selector(setBufferOffset:atIndex:), arg.offset, idx);
            slot.offset = arg.offset;
          }
        } else {
          imps.set_buffer(active_enc, @selector(setBuffer:offset:atIndex:), arg.buffer, arg.offset,
                          idx);
          slot.buffer_ptr = buf_ptr;
          slot.offset = arg.offset;
          slot.valid = true;
        }
      } else {
        imps.set_buffer(active_enc, @selector(setBuffer:offset:atIndex:), arg.buffer, arg.offset,
                        idx);
      }
      enc_state.MarkDirty(buf_ptr, arg.offset, idx);
    } else {
      imps.set_bytes(active_enc, @selector(setBytes:length:atIndex:), arg.bytes, arg.size, idx);
      if (idx < kMaxEncoderBufferSlots) {
        enc_state.slots[idx].valid = false;
      }
    }
  }

  if (shared_mem_bytes > 0) {
    [active_enc setThreadgroupMemoryLength:AlignUp16(shared_mem_bytes) atIndex:0];
  }

  imps.dispatch(active_enc, @selector(dispatchThreadgroups:threadsPerThreadgroup:), grid_size,
                block_size);
  active_disp++;
  if (stream_state == nullptr) {
    default_active_dispatches_.store(active_disp, std::memory_order_relaxed);
  }

  if (active_disp >= kMaxBatchDispatches) {
    if (stream_state != nullptr) {
      FlushStreamLocked(stream_state);
    } else {
      FlushDefaultStreamLocked();
    }
  }
  return cudaSuccess;
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
  active_stream_count_.store(0, std::memory_order_release);
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

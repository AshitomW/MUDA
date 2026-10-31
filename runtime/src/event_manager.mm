/**
 * @file event_manager.mm
 * @brief Manages Metal shared events and timing.
 */

#include "event_manager.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <chrono>
#include <mutex>
#include <utility>

#include "error_state.hpp"

namespace muda::runtime::cudart {
namespace {

[[nodiscard]] constexpr bool IsValidEventFlags(unsigned int flags) noexcept {
  constexpr unsigned int kAllowed =
      cudaEventDefault | cudaEventBlockingSync | cudaEventDisableTiming;
  return (flags & ~kAllowed) == 0u && (flags & cudaEventInterprocess) == 0u;
}

[[nodiscard]] double NowSeconds() noexcept {
  using Clock = std::chrono::steady_clock;
  return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}

}  // namespace

cudaError_t EventManager::EventCreate(cudaEvent_t* out_event, unsigned int flags,
                                      id<MTLDevice> device) noexcept {
  if (out_event == nullptr || !IsValidEventFlags(flags)) {
    return RecordError(cudaErrorInvalidValue);
  }
  if (device == nil) {
    return RecordError(cudaErrorInitializationError);
  }
  @autoreleasepool {
    id<MTLSharedEvent> shared_event = [device newSharedEvent];
    if (shared_event == nil) {
      return RecordError(cudaErrorInitializationError);
    }
    auto state = std::make_shared<EventState>();
    state->shared_event = shared_event;
    state->flags = flags;
    *out_event = events_.Insert<cudaEvent_t>(std::move(state));
    return cudaSuccess;
  }
}

cudaError_t EventManager::EventDestroy(cudaEvent_t event_handle) noexcept {
  if (event_handle == nullptr) {
    return RecordError(cudaErrorInvalidResourceHandle);
  }
  std::shared_ptr<EventState> removed = events_.Remove(event_handle);
  if (removed == nullptr) {
    return RecordError(cudaErrorInvalidResourceHandle);
  }
  return cudaSuccess;
}

cudaError_t EventManager::EventRecord(cudaEvent_t event_handle, cudaStream_t stream_handle,
                                      StreamManager& streams,
                                      id<MTLCommandQueue> default_queue) noexcept {
  std::shared_ptr<EventState> event = events_.Lookup(event_handle);
  if (event == nullptr) {
    return RecordError(cudaErrorInvalidResourceHandle);
  }
  std::shared_ptr<StreamState> stream_state;
  id<MTLCommandQueue> target_queue = nil;
  const cudaError_t stream_status =
      streams.ResolveStreamQueue(stream_handle, default_queue, &stream_state, &target_queue);
  if (stream_status != cudaSuccess) {
    return stream_status;
  }

  @autoreleasepool {
    if (stream_state != nullptr) {
      std::lock_guard<std::mutex> stream_lock(stream_state->mutex);
      streams.FlushStreamLocked(stream_state.get());
    } else {
      std::lock_guard<std::mutex> def_lock(streams.DefaultStreamMutex());
      streams.FlushDefaultStreamLocked();
    }

    id<MTLCommandBuffer> command_buffer = [target_queue commandBuffer];
    if (command_buffer == nil) {
      return RecordError(cudaErrorLaunchFailure);
    }

    std::uint64_t next_ticket = 0;
    {
      std::lock_guard<std::mutex> event_lock(event->mutex);
      next_ticket = ++event->recorded_ticket;
      event->recorded_command_buffer = command_buffer;
      event->recorded.store(true, std::memory_order_release);
    }

    [command_buffer encodeSignalEvent:event->shared_event value:next_ticket];
    std::weak_ptr<EventState> weak_event = event;
    [command_buffer addCompletedHandler:^(id<MTLCommandBuffer> completed) {
      if (auto locked_event = weak_event.lock()) {
        const double gpu_end = completed.GPUEndTime;
        const double stamp = (gpu_end > 0.0) ? gpu_end : NowSeconds();
        locked_event->completion_timestamp_seconds.store(stamp, std::memory_order_release);
      }
    }];
    streams.CommitStreamCommandBuffer(stream_state.get(), command_buffer);
    return cudaSuccess;
  }
}

cudaError_t EventManager::EventSynchronize(cudaEvent_t event_handle,
                                           StreamManager& streams) noexcept {
  std::shared_ptr<EventState> event = events_.Lookup(event_handle);
  if (event == nullptr) {
    return RecordError(cudaErrorInvalidResourceHandle);
  }
  id<MTLCommandBuffer> cmd = nil;
  {
    std::lock_guard<std::mutex> event_lock(event->mutex);
    if (!event->recorded.load(std::memory_order_acquire)) {
      return cudaSuccess;
    }
    cmd = event->recorded_command_buffer;
  }

  @autoreleasepool {
    if (cmd != nil) {
      [cmd waitUntilCompleted];
      if (event->completion_timestamp_seconds.load(std::memory_order_acquire) <= 0.0) {
        const double gpu_end = cmd.GPUEndTime;
        event->completion_timestamp_seconds.store((gpu_end > 0.0) ? gpu_end : NowSeconds(),
                                                  std::memory_order_release);
      }
    }
  }
  return streams.ConsumeAsyncError();
}

cudaError_t EventManager::EventQuery(cudaEvent_t event_handle) noexcept {
  std::shared_ptr<EventState> event = events_.Lookup(event_handle);
  if (event == nullptr) {
    return RecordError(cudaErrorInvalidResourceHandle);
  }
  std::uint64_t ticket = 0;
  {
    std::lock_guard<std::mutex> event_lock(event->mutex);
    if (!event->recorded.load(std::memory_order_acquire)) {
      return cudaSuccess;
    }
    ticket = event->recorded_ticket;
  }
  @autoreleasepool {
    if (event->shared_event.signaledValue < ticket) {
      return cudaErrorNotReady;
    }
    return cudaSuccess;
  }
}

cudaError_t EventManager::EventElapsedTime(float* ms, cudaEvent_t start_handle,
                                           cudaEvent_t end_handle,
                                           StreamManager& streams) noexcept {
  if (ms == nullptr) {
    return RecordError(cudaErrorInvalidValue);
  }
  std::shared_ptr<EventState> start_event = events_.Lookup(start_handle);
  std::shared_ptr<EventState> end_event = events_.Lookup(end_handle);
  if (start_event == nullptr || end_event == nullptr) {
    return RecordError(cudaErrorInvalidResourceHandle);
  }
  if ((start_event->flags & cudaEventDisableTiming) != 0u ||
      (end_event->flags & cudaEventDisableTiming) != 0u) {
    return RecordError(cudaErrorInvalidResourceHandle);
  }
  if (!start_event->recorded.load(std::memory_order_acquire) ||
      !end_event->recorded.load(std::memory_order_acquire)) {
    return RecordError(cudaErrorInvalidResourceHandle);
  }

  if (EventQuery(start_handle) != cudaSuccess || EventQuery(end_handle) != cudaSuccess) {
    return cudaErrorNotReady;
  }

  (void)EventSynchronize(start_handle, streams);
  (void)EventSynchronize(end_handle, streams);

  const double start_sec =
      start_event->completion_timestamp_seconds.load(std::memory_order_acquire);
  const double end_sec = end_event->completion_timestamp_seconds.load(std::memory_order_acquire);
  const double delta_ms = (end_sec >= start_sec) ? ((end_sec - start_sec) * 1000.0) : 0.0;
  *ms = static_cast<float>(delta_ms);
  return cudaSuccess;
}

cudaError_t EventManager::StreamWaitEvent(cudaStream_t stream_handle, cudaEvent_t event_handle,
                                          unsigned int flags, StreamManager& streams,
                                          id<MTLCommandQueue> default_queue) noexcept {
  if (flags != cudaEventWaitDefault && flags != cudaEventWaitExternal) {
    return RecordError(cudaErrorInvalidValue);
  }
  std::shared_ptr<EventState> event = events_.Lookup(event_handle);
  if (event == nullptr) {
    return RecordError(cudaErrorInvalidResourceHandle);
  }
  std::shared_ptr<StreamState> stream_state;
  id<MTLCommandQueue> target_queue = nil;
  const cudaError_t stream_status =
      streams.ResolveStreamQueue(stream_handle, default_queue, &stream_state, &target_queue);
  if (stream_status != cudaSuccess) {
    return stream_status;
  }

  std::uint64_t ticket = 0;
  {
    std::lock_guard<std::mutex> event_lock(event->mutex);
    if (!event->recorded.load(std::memory_order_acquire)) {
      return cudaSuccess;
    }
    ticket = event->recorded_ticket;
  }

  @autoreleasepool {
    if (stream_state != nullptr) {
      std::lock_guard<std::mutex> stream_lock(stream_state->mutex);
      streams.FlushStreamLocked(stream_state.get());
    } else {
      std::lock_guard<std::mutex> def_lock(streams.DefaultStreamMutex());
      streams.FlushDefaultStreamLocked();
    }

    id<MTLCommandBuffer> command_buffer = [target_queue commandBuffer];
    if (command_buffer == nil) {
      return RecordError(cudaErrorLaunchFailure);
    }
    [command_buffer encodeWaitForEvent:event->shared_event value:ticket];
    streams.CommitStreamCommandBuffer(stream_state.get(), command_buffer);
    return cudaSuccess;
  }
}

std::shared_ptr<EventState> EventManager::Lookup(cudaEvent_t event_handle) const noexcept {
  return events_.Lookup(event_handle);
}

std::vector<std::shared_ptr<EventState>> EventManager::DrainAll() { return events_.DrainAll(); }

void EventManager::Reset() noexcept { (void)events_.DrainAll(); }

}  // namespace muda::runtime::cudart

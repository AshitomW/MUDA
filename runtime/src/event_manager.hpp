/**
 * @file event_manager.hpp
 * @brief Manages Metal shared events and timing.
 */

#pragma once

#import <Metal/Metal.h>

#include <memory>
#include <vector>

#include "cuda_runtime.h"
#include "handle_table.hpp"
#include "stream_manager.hpp"
#include "stream_types.hpp"

namespace muda::runtime::cudart {

class EventManager {
 public:
  EventManager() = default;
  ~EventManager() = default;
  EventManager(const EventManager&) = delete;
  EventManager& operator=(const EventManager&) = delete;

  cudaError_t EventCreate(cudaEvent_t* out_event, unsigned int flags,
                          id<MTLDevice> device) noexcept;
  cudaError_t EventDestroy(cudaEvent_t event_handle) noexcept;
  cudaError_t EventRecord(cudaEvent_t event_handle, cudaStream_t stream_handle,
                          StreamManager& streams, id<MTLCommandQueue> default_queue) noexcept;
  cudaError_t EventSynchronize(cudaEvent_t event_handle, StreamManager& streams) noexcept;
  cudaError_t EventQuery(cudaEvent_t event_handle) noexcept;
  cudaError_t EventElapsedTime(float* ms, cudaEvent_t start_handle, cudaEvent_t end_handle,
                               StreamManager& streams) noexcept;
  cudaError_t StreamWaitEvent(cudaStream_t stream_handle, cudaEvent_t event_handle,
                              unsigned int flags, StreamManager& streams,
                              id<MTLCommandQueue> default_queue) noexcept;

  [[nodiscard]] std::shared_ptr<EventState> Lookup(cudaEvent_t event_handle) const noexcept;
  std::vector<std::shared_ptr<EventState>> DrainAll();
  void Reset() noexcept;

 private:
  HandleTable<EventState, HandleKind::kEvent> events_;
};

}  // namespace muda::runtime::cudart


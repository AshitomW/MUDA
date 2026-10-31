/**
 * @file stream_manager.hpp
 * @brief Manages Metal command queues, execution streams, and batching.
 */

#pragma once

#import <Metal/Metal.h>

#include <atomic>
#include <memory>
#include <mutex>
#include <vector>

#include "cuda_runtime.h"
#include "handle_table.hpp"
#include "stream_types.hpp"

namespace muda::runtime::cudart {

class StreamManager {
 public:
  StreamManager() = default;
  ~StreamManager() = default;
  StreamManager(const StreamManager&) = delete;
  StreamManager& operator=(const StreamManager&) = delete;

  cudaError_t StreamCreate(cudaStream_t* out_stream, unsigned int flags,
                           id<MTLDevice> device) noexcept;
  cudaError_t StreamDestroy(cudaStream_t stream_handle) noexcept;
  cudaError_t StreamSynchronize(cudaStream_t stream_handle,
                                id<MTLCommandQueue> default_queue) noexcept;
  cudaError_t StreamQuery(cudaStream_t stream_handle) noexcept;
  cudaError_t SynchronizeDevice(id<MTLCommandQueue> default_queue) noexcept;

  cudaError_t ResolveStreamQueue(cudaStream_t stream_handle, id<MTLCommandQueue> default_queue,
                                 std::shared_ptr<StreamState>* out_stream,
                                 id<MTLCommandQueue>* out_queue) noexcept;

  [[nodiscard]] std::shared_ptr<StreamState> Lookup(cudaStream_t stream_handle) const noexcept;
  [[nodiscard]] std::vector<std::shared_ptr<StreamState>> SnapshotAll() const;
  std::vector<std::shared_ptr<StreamState>> DrainAll();

  void FlushStreamLocked(StreamState* stream) noexcept;
  void FlushDefaultStreamLocked() noexcept;
  void CommitStreamCommandBuffer(StreamState* stream, id<MTLCommandBuffer> command_buffer) noexcept;

  void RecordAsyncError(cudaError_t error) noexcept;
  [[nodiscard]] cudaError_t ConsumeAsyncError() noexcept;

  std::mutex& DefaultStreamMutex() noexcept { return default_stream_mutex_; }
  __strong id<MTLCommandBuffer>& DefaultActiveCommandBuffer() noexcept {
    return default_active_command_buffer_;
  }
  __strong id<MTLComputeCommandEncoder>& DefaultActiveEncoder() noexcept {
    return default_active_encoder_;
  }
  __strong id<MTLComputePipelineState>& DefaultActivePipeline() noexcept {
    return default_active_pipeline_;
  }
  std::size_t& DefaultActiveDispatches() noexcept { return default_active_dispatches_; }

  void Reset() noexcept;

 private:
  void CommitStreamCommandBufferLocked(StreamState* stream_state,
                                       id<MTLCommandBuffer> command_buffer) noexcept;

  HandleTable<StreamState, HandleKind::kStream> streams_;
  std::atomic<std::uint64_t> default_pending_commands_{0};
  std::atomic<cudaError_t> async_error_{cudaSuccess};
  id<MTLCommandBuffer> default_last_command_buffer_ = nil;
  id<MTLCommandBuffer> default_active_command_buffer_ = nil;
  id<MTLComputeCommandEncoder> default_active_encoder_ = nil;
  id<MTLComputePipelineState> default_active_pipeline_ = nil;
  std::size_t default_active_dispatches_ = 0;
  std::mutex default_stream_mutex_;
};

}  // namespace muda::runtime::cudart

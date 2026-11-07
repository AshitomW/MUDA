/**
 * @file transfer_manager.mm
 * @brief Manages 1D and 2D memory transfers, blits, and memsets.
 */

#include "transfer_manager.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstring>
#include <mutex>
#include <optional>

#include "error_state.hpp"

namespace muda::runtime::cudart {
namespace {

constexpr std::size_t kMaxBatchDispatches = 256;

[[nodiscard]] bool IsEndpointRangeValid(const LightweightEndpoint& ep, DeviceAddress addr,
                                        std::size_t count) noexcept {
  if (!ep.found || addr.value < ep.base_address.value) {
    return false;
  }
  if (ep.offset > ep.size) {
    return false;
  }
  if (ep.offset == ep.size && count == 0) {
    return true;
  }
  return count <= (ep.size - ep.offset);
}

cudaError_t ValidateLightweightEndpoints(const LightweightEndpoint& dst_ep,
                                         const LightweightEndpoint& src_ep, DeviceAddress dst_addr,
                                         std::size_t dst_span, DeviceAddress src_addr,
                                         std::size_t src_span, cudaMemcpyKind kind,
                                         bool* out_touches_device) noexcept {
  if (out_touches_device != nullptr) {
    *out_touches_device = (dst_ep.found || src_ep.found);
  }

  const bool dst_valid = IsEndpointRangeValid(dst_ep, dst_addr, dst_span);
  const bool src_valid = IsEndpointRangeValid(src_ep, src_addr, src_span);

  const bool dst_is_host =
      (!dst_ep.found || (dst_ep.memory_type == cudaMemoryTypeHost && dst_valid));
  const bool src_is_host =
      (!src_ep.found || (src_ep.memory_type == cudaMemoryTypeHost && src_valid));
  const bool dst_is_device =
      (dst_ep.found && dst_ep.memory_type != cudaMemoryTypeHost && dst_valid);
  const bool src_is_device =
      (src_ep.found && src_ep.memory_type != cudaMemoryTypeHost && src_valid);

  switch (kind) {
    case cudaMemcpyHostToHost:
      return (dst_is_host && src_is_host) ? cudaSuccess : RecordError(cudaErrorInvalidValue);
    case cudaMemcpyHostToDevice:
      return (dst_is_device && src_is_host) ? cudaSuccess : RecordError(cudaErrorInvalidValue);
    case cudaMemcpyDeviceToHost:
      return (src_is_device && dst_is_host) ? cudaSuccess : RecordError(cudaErrorInvalidValue);
    case cudaMemcpyDeviceToDevice:
      return (dst_is_device && src_is_device) ? cudaSuccess : RecordError(cudaErrorInvalidValue);
    case cudaMemcpyDefault:
      if ((dst_ep.found && !dst_valid) || (src_ep.found && !src_valid)) {
        return RecordError(cudaErrorInvalidValue);
      }
      return cudaSuccess;
  }
  return RecordError(cudaErrorInvalidMemcpyDirection);
}

cudaError_t EncodeBlitCopy(id<MTLCommandQueue> queue, id<MTLBuffer> src_buf, std::size_t src_off,
                           id<MTLBuffer> dst_buf, std::size_t dst_off, std::size_t count,
                           StreamState* stream_state, StreamManager* streams) noexcept {
  @autoreleasepool {
    if (streams != nullptr) {
      std::unique_lock<std::mutex> lock =
          (stream_state != nullptr) ? std::unique_lock<std::mutex>(stream_state->mutex)
                                    : std::unique_lock<std::mutex>(streams->DefaultStreamMutex());

      __strong id<MTLCommandBuffer>& active_cmd = (stream_state != nullptr)
                                                      ? stream_state->active_command_buffer
                                                      : streams->DefaultActiveCommandBuffer();
      __strong id<MTLComputeCommandEncoder>& active_enc = (stream_state != nullptr)
                                                              ? stream_state->active_encoder
                                                              : streams->DefaultActiveEncoder();
      __strong id<MTLBlitCommandEncoder>& active_blit = (stream_state != nullptr)
                                                            ? stream_state->active_blit_encoder
                                                            : streams->DefaultActiveBlitEncoder();
      __strong id<MTLComputePipelineState>& active_pipe = (stream_state != nullptr)
                                                              ? stream_state->active_pipeline
                                                              : streams->DefaultActivePipeline();
      std::size_t& active_disp = (stream_state != nullptr) ? stream_state->active_dispatches
                                                           : streams->DefaultActiveDispatches();
      EncoderHazardState& enc_state =
          (stream_state != nullptr) ? stream_state->encoder_state : streams->DefaultEncoderState();

      if (active_enc != nil) {
        [active_enc endEncoding];
        active_enc = nil;
        active_pipe = nil;
        enc_state.Reset();
      }

      if (active_cmd == nil) {
        active_cmd = [queue commandBufferWithUnretainedReferences];
        if (active_cmd == nil) {
          return RecordError(cudaErrorLaunchFailure);
        }
        active_disp = 0;
      }

      if (active_blit == nil) {
        active_blit = [active_cmd blitCommandEncoder];
        if (active_blit == nil) {
          active_cmd = nil;
          return RecordError(cudaErrorLaunchFailure);
        }
      }

      [active_blit copyFromBuffer:src_buf
                     sourceOffset:src_off
                         toBuffer:dst_buf
                destinationOffset:dst_off
                             size:count];
      active_disp++;
      if (stream_state == nullptr) {
        streams->SetDefaultActiveDispatchesAtomic(active_disp);
      }
      if (active_disp >= kMaxBatchDispatches) {
        if (stream_state != nullptr) {
          streams->FlushStreamLocked(stream_state);
        } else {
          streams->FlushDefaultStreamLocked();
        }
      }
      return cudaSuccess;
    }

    id<MTLCommandBuffer> cmd = [queue commandBufferWithUnretainedReferences];
    id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
    if (cmd == nil || blit == nil) {
      return RecordError(cudaErrorLaunchFailure);
    }
    [blit copyFromBuffer:src_buf
             sourceOffset:src_off
                 toBuffer:dst_buf
        destinationOffset:dst_off
                     size:count];
    [blit endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    return cudaSuccess;
  }
}

}  // namespace

cudaError_t TransferManager::Memcpy(void* dst, const void* src, std::size_t count,
                                    cudaMemcpyKind kind, const MemoryManager& memory,
                                    StreamManager& streams,
                                    id<MTLCommandQueue> default_queue) noexcept {
  if (dst == nullptr || src == nullptr) {
    return RecordError(cudaErrorInvalidValue);
  }
  if (!IsValidMemcpyKind(kind)) {
    return RecordError(cudaErrorInvalidMemcpyDirection);
  }

  const DeviceAddress dst_addr = DeviceAddress::FromPointer(dst);
  const DeviceAddress src_addr = DeviceAddress::FromPointer(src);
  LightweightEndpoint dst_ep;
  LightweightEndpoint src_ep;
  memory.ResolveEndpointsPair(dst_addr, src_addr, &dst_ep, &src_ep);

  bool touches_device = false;
  const cudaError_t val_status = ValidateLightweightEndpoints(
      dst_ep, src_ep, dst_addr, count, src_addr, count, kind, &touches_device);
  if (val_status != cudaSuccess) {
    return val_status;
  }

  if (touches_device && streams.HasPendingWork()) {
    const cudaError_t sync_status = streams.SynchronizeDevice(default_queue);
    if (sync_status != cudaSuccess) {
      return sync_status;
    }
  }

  if (count > 0 && dst != src) {
    std::memmove(dst, src, count);
  }
  return cudaSuccess;
}

cudaError_t TransferManager::MemcpyAsync(void* dst, const void* src, std::size_t count,
                                         cudaMemcpyKind kind, cudaStream_t stream_handle,
                                         const MemoryManager& memory, StreamManager& streams,
                                         id<MTLCommandQueue> default_queue) noexcept {
  if (dst == nullptr || src == nullptr) {
    return RecordError(cudaErrorInvalidValue);
  }
  if (!IsValidMemcpyKind(kind)) {
    return RecordError(cudaErrorInvalidMemcpyDirection);
  }

  std::shared_ptr<StreamState> stream_state;
  id<MTLCommandQueue> target_queue = default_queue;
  if (stream_handle != nullptr) {
    const cudaError_t s_status =
        streams.ResolveStreamQueue(stream_handle, default_queue, &stream_state, &target_queue);
    if (s_status != cudaSuccess) {
      return s_status;
    }
  } else if (default_queue == nil) {
    return RecordError(cudaErrorInitializationError);
  }

  const DeviceAddress dst_addr = DeviceAddress::FromPointer(dst);
  const DeviceAddress src_addr = DeviceAddress::FromPointer(src);
  LightweightEndpoint dst_ep;
  LightweightEndpoint src_ep;
  memory.ResolveEndpointsPair(dst_addr, src_addr, &dst_ep, &src_ep);

  bool touches_device = false;
  const cudaError_t val_status = ValidateLightweightEndpoints(
      dst_ep, src_ep, dst_addr, count, src_addr, count, kind, &touches_device);
  if (val_status != cudaSuccess || count == 0 || dst == src) {
    return val_status;
  }

  if (dst_ep.found && src_ep.found && (dst_ep.buffer != src_ep.buffer || dst_addr != src_addr)) {
    return EncodeBlitCopy(target_queue, src_ep.buffer, src_ep.offset, dst_ep.buffer, dst_ep.offset,
                          count, stream_state.get(), &streams);
  }

  if (touches_device && streams.HasPendingWork()) {
    (void)streams.StreamSynchronize(stream_handle, default_queue);
  }
  std::memmove(dst, src, count);
  return cudaSuccess;
}

cudaError_t TransferManager::Memcpy2D(void* dst, std::size_t dpitch, const void* src,
                                      std::size_t spitch, std::size_t width, std::size_t height,
                                      cudaMemcpyKind kind, const MemoryManager& memory,
                                      StreamManager& streams,
                                      id<MTLCommandQueue> default_queue) noexcept {
  return Memcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, nullptr, memory, streams,
                       default_queue);
}

cudaError_t TransferManager::Memcpy2DAsync(void* dst, std::size_t dpitch, const void* src,
                                           std::size_t spitch, std::size_t width,
                                           std::size_t height, cudaMemcpyKind kind,
                                           cudaStream_t stream_handle, const MemoryManager& memory,
                                           StreamManager& streams,
                                           id<MTLCommandQueue> default_queue) noexcept {
  if (width == 0 || height == 0) {
    return cudaSuccess;
  }
  if (dst == nullptr || src == nullptr) {
    return RecordError(cudaErrorInvalidValue);
  }
  if (dpitch < width || spitch < width) {
    return RecordError(cudaErrorInvalidPitchValue);
  }
  if (!IsValidMemcpyKind(kind)) {
    return RecordError(cudaErrorInvalidMemcpyDirection);
  }

  const std::size_t dst_span = dpitch * (height - 1) + width;
  const std::size_t src_span = spitch * (height - 1) + width;
  const DeviceAddress dst_addr = DeviceAddress::FromPointer(dst);
  const DeviceAddress src_addr = DeviceAddress::FromPointer(src);
  LightweightEndpoint dst_ep;
  LightweightEndpoint src_ep;
  memory.ResolveEndpointsPair(dst_addr, src_addr, &dst_ep, &src_ep);

  bool touches_device = false;
  const cudaError_t val_status = ValidateLightweightEndpoints(
      dst_ep, src_ep, dst_addr, dst_span, src_addr, src_span, kind, &touches_device);
  if (val_status != cudaSuccess) {
    return val_status;
  }

  if (touches_device && streams.HasPendingWork()) {
    const cudaError_t sync_status = streams.StreamSynchronize(stream_handle, default_queue);
    if (sync_status != cudaSuccess) {
      return sync_status;
    }
  }

  auto* d_bytes = static_cast<std::uint8_t*>(dst);
  const auto* s_bytes = static_cast<const std::uint8_t*>(src);
  for (std::size_t row = 0; row < height; ++row) {
    std::memmove(d_bytes + row * dpitch, s_bytes + row * spitch, width);
  }
  return cudaSuccess;
}

cudaError_t TransferManager::Memset(void* dev_ptr, int value, std::size_t count,
                                    const MemoryManager& memory, StreamManager& streams,
                                    id<MTLCommandQueue> default_queue) noexcept {
  if (dev_ptr == nullptr) {
    return RecordError(cudaErrorInvalidValue);
  }
  if (count == 0) {
    return cudaSuccess;
  }
  const DeviceAddress addr = DeviceAddress::FromPointer(dev_ptr);
  if (!memory.ValidateRange(addr, count)) {
    return RecordError(cudaErrorInvalidValue);
  }
  if (streams.HasPendingWork()) {
    const cudaError_t sync_status = streams.SynchronizeDevice(default_queue);
    if (sync_status != cudaSuccess) {
      return sync_status;
    }
  }
  std::memset(dev_ptr, value, count);
  return cudaSuccess;
}

cudaError_t TransferManager::MemsetAsync(void* dev_ptr, int value, std::size_t count,
                                         cudaStream_t stream_handle, const MemoryManager& memory,
                                         StreamManager& streams,
                                         id<MTLCommandQueue> default_queue) noexcept {
  if (dev_ptr == nullptr) {
    return RecordError(cudaErrorInvalidValue);
  }
  if (count == 0) {
    return cudaSuccess;
  }

  std::shared_ptr<StreamState> stream_state;
  id<MTLCommandQueue> target_queue = nil;
  const cudaError_t s_status =
      streams.ResolveStreamQueue(stream_handle, default_queue, &stream_state, &target_queue);
  if (s_status != cudaSuccess) {
    return s_status;
  }

  const DeviceAddress addr = DeviceAddress::FromPointer(dev_ptr);
  const auto rec = memory.SnapshotRecord(addr);
  if (!rec.has_value() || !AllocationTable::IsRangeValid(&*rec, addr, count)) {
    return RecordError(cudaErrorInvalidValue);
  }

  if (stream_state != nullptr) {
    std::lock_guard<std::mutex> stream_lock(stream_state->mutex);
    streams.FlushStreamLocked(stream_state.get());
  } else {
    std::lock_guard<std::mutex> def_lock(streams.DefaultStreamMutex());
    streams.FlushDefaultStreamLocked();
  }

  @autoreleasepool {
    id<MTLCommandBuffer> cmd = [target_queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
    if (cmd == nil || blit == nil) {
      return RecordError(cudaErrorLaunchFailure);
    }
    const std::size_t offset = static_cast<std::size_t>(addr.value - rec->base_address.value);
    [blit fillBuffer:rec->buffer
               range:NSMakeRange(offset, count)
               value:static_cast<std::uint8_t>(value & 0xFF)];
    [blit endEncoding];
    streams.CommitStreamCommandBuffer(stream_state.get(), cmd);
    return cudaSuccess;
  }
}

cudaError_t TransferManager::Memset2D(void* dev_ptr, std::size_t pitch, int value,
                                      std::size_t width, std::size_t height,
                                      const MemoryManager& memory, StreamManager& streams,
                                      id<MTLCommandQueue> default_queue) noexcept {
  if (dev_ptr == nullptr) {
    return RecordError(cudaErrorInvalidValue);
  }
  if (pitch < width) {
    return RecordError(cudaErrorInvalidPitchValue);
  }
  if (width == 0 || height == 0) {
    return cudaSuccess;
  }
  const std::size_t total_span = pitch * (height - 1) + width;
  const DeviceAddress addr = DeviceAddress::FromPointer(dev_ptr);
  if (!memory.ValidateRange(addr, total_span)) {
    return RecordError(cudaErrorInvalidValue);
  }
  const cudaError_t sync_status = streams.SynchronizeDevice(default_queue);
  if (sync_status != cudaSuccess) {
    return sync_status;
  }
  auto* bytes = static_cast<std::uint8_t*>(dev_ptr);
  for (std::size_t row = 0; row < height; ++row) {
    std::memset(bytes + row * pitch, value, width);
  }
  return cudaSuccess;
}

}  // namespace muda::runtime::cudart

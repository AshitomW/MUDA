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

cudaError_t ValidateEndpoints(const std::optional<AllocationRecord>& dst_record,
                              const std::optional<AllocationRecord>& src_record,
                              DeviceAddress dst_addr, std::size_t dst_span, DeviceAddress src_addr,
                              std::size_t src_span, cudaMemcpyKind kind,
                              bool* out_touches_device) noexcept {
  const AllocationRecord* dst_ptr = dst_record ? &*dst_record : nullptr;
  const AllocationRecord* src_ptr = src_record ? &*src_record : nullptr;

  if (out_touches_device != nullptr) {
    *out_touches_device = (dst_ptr != nullptr || src_ptr != nullptr);
  }

  const bool dst_valid = AllocationTable::IsRangeValid(dst_ptr, dst_addr, dst_span);
  const bool src_valid = AllocationTable::IsRangeValid(src_ptr, src_addr, src_span);

  switch (kind) {
    case cudaMemcpyHostToHost:
      return (dst_ptr == nullptr && src_ptr == nullptr) ? cudaSuccess
                                                        : RecordError(cudaErrorInvalidValue);
    case cudaMemcpyHostToDevice:
      return (dst_valid && src_ptr == nullptr) ? cudaSuccess : RecordError(cudaErrorInvalidValue);
    case cudaMemcpyDeviceToHost:
      return (src_valid && dst_ptr == nullptr) ? cudaSuccess : RecordError(cudaErrorInvalidValue);
    case cudaMemcpyDeviceToDevice:
      return (dst_valid && src_valid) ? cudaSuccess : RecordError(cudaErrorInvalidValue);
    case cudaMemcpyDefault:
      if ((dst_ptr != nullptr && !dst_valid) || (src_ptr != nullptr && !src_valid)) {
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
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
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

    if (streams != nullptr) {
      streams->CommitStreamCommandBuffer(stream_state, cmd);
      return cudaSuccess;
    }
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
  const auto dst_rec = memory.SnapshotRecord(dst_addr);
  const auto src_rec = memory.SnapshotRecord(src_addr);

  bool touches_device = false;
  const cudaError_t val_status =
      ValidateEndpoints(dst_rec, src_rec, dst_addr, count, src_addr, count, kind, &touches_device);
  if (val_status != cudaSuccess) {
    return val_status;
  }

  if (touches_device) {
    const cudaError_t sync_status = streams.SynchronizeDevice(default_queue);
    if (sync_status != cudaSuccess) {
      return sync_status;
    }
  }

  if (count > 0 && dst != src) {
    if (dst_rec && src_rec && (dst_rec->buffer != src_rec->buffer || dst_addr != src_addr)) {
      const std::size_t dst_off =
          static_cast<std::size_t>(dst_addr.value - dst_rec->base_address.value);
      const std::size_t src_off =
          static_cast<std::size_t>(src_addr.value - src_rec->base_address.value);
      return EncodeBlitCopy(default_queue, src_rec->buffer, src_off, dst_rec->buffer, dst_off,
                            count, nullptr, nullptr);
    }
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
  id<MTLCommandQueue> target_queue = nil;
  const cudaError_t s_status =
      streams.ResolveStreamQueue(stream_handle, default_queue, &stream_state, &target_queue);
  if (s_status != cudaSuccess) {
    return s_status;
  }

  const DeviceAddress dst_addr = DeviceAddress::FromPointer(dst);
  const DeviceAddress src_addr = DeviceAddress::FromPointer(src);
  const auto dst_rec = memory.SnapshotRecord(dst_addr);
  const auto src_rec = memory.SnapshotRecord(src_addr);

  bool touches_device = false;
  const cudaError_t val_status =
      ValidateEndpoints(dst_rec, src_rec, dst_addr, count, src_addr, count, kind, &touches_device);
  if (val_status != cudaSuccess || count == 0 || dst == src) {
    return val_status;
  }

  if (stream_state != nullptr) {
    std::lock_guard<std::mutex> stream_lock(stream_state->mutex);
    streams.FlushStreamLocked(stream_state.get());
  } else {
    std::lock_guard<std::mutex> def_lock(streams.DefaultStreamMutex());
    streams.FlushDefaultStreamLocked();
  }

  if (dst_rec && src_rec && (dst_rec->buffer != src_rec->buffer || dst_addr != src_addr)) {
    const std::size_t dst_off =
        static_cast<std::size_t>(dst_addr.value - dst_rec->base_address.value);
    const std::size_t src_off =
        static_cast<std::size_t>(src_addr.value - src_rec->base_address.value);
    return EncodeBlitCopy(target_queue, src_rec->buffer, src_off, dst_rec->buffer, dst_off, count,
                          stream_state.get(), &streams);
  }

  if (touches_device) {
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
  const auto dst_rec = memory.SnapshotRecord(dst_addr);
  const auto src_rec = memory.SnapshotRecord(src_addr);

  bool touches_device = false;
  const cudaError_t val_status = ValidateEndpoints(dst_rec, src_rec, dst_addr, dst_span, src_addr,
                                                   src_span, kind, &touches_device);
  if (val_status != cudaSuccess) {
    return val_status;
  }

  if (touches_device) {
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
  const cudaError_t sync_status = streams.SynchronizeDevice(default_queue);
  if (sync_status != cudaSuccess) {
    return sync_status;
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

/**
 * @file transfer_manager.hpp
 * @brief Manages 1D and 2D memory transfers, blits, and memsets.
 */

#pragma once

#import <Metal/Metal.h>

#include <cstddef>

#include "cuda_runtime.h"
#include "memory_manager.hpp"
#include "stream_manager.hpp"

namespace muda::runtime::cudart {

[[nodiscard]] constexpr bool IsValidMemcpyKind(cudaMemcpyKind kind) noexcept {
  return kind == cudaMemcpyHostToHost || kind == cudaMemcpyHostToDevice ||
         kind == cudaMemcpyDeviceToHost || kind == cudaMemcpyDeviceToDevice ||
         kind == cudaMemcpyDefault;
}

class TransferManager {
 public:
  TransferManager() = default;
  ~TransferManager() = default;
  TransferManager(const TransferManager&) = delete;
  TransferManager& operator=(const TransferManager&) = delete;

  cudaError_t Memcpy(void* dst, const void* src, std::size_t count, cudaMemcpyKind kind,
                     const MemoryManager& memory, StreamManager& streams,
                     id<MTLCommandQueue> default_queue) noexcept;

  cudaError_t MemcpyAsync(void* dst, const void* src, std::size_t count, cudaMemcpyKind kind,
                          cudaStream_t stream_handle, const MemoryManager& memory,
                          StreamManager& streams, id<MTLCommandQueue> default_queue) noexcept;

  cudaError_t Memcpy2D(void* dst, std::size_t dpitch, const void* src, std::size_t spitch,
                       std::size_t width, std::size_t height, cudaMemcpyKind kind,
                       const MemoryManager& memory, StreamManager& streams,
                       id<MTLCommandQueue> default_queue) noexcept;

  cudaError_t Memcpy2DAsync(void* dst, std::size_t dpitch, const void* src, std::size_t spitch,
                            std::size_t width, std::size_t height, cudaMemcpyKind kind,
                            cudaStream_t stream_handle, const MemoryManager& memory,
                            StreamManager& streams, id<MTLCommandQueue> default_queue) noexcept;

  cudaError_t Memset(void* dev_ptr, int value, std::size_t count, const MemoryManager& memory,
                     StreamManager& streams, id<MTLCommandQueue> default_queue) noexcept;

  cudaError_t MemsetAsync(void* dev_ptr, int value, std::size_t count, cudaStream_t stream_handle,
                          const MemoryManager& memory, StreamManager& streams,
                          id<MTLCommandQueue> default_queue) noexcept;

  cudaError_t Memset2D(void* dev_ptr, std::size_t pitch, int value, std::size_t width,
                       std::size_t height, const MemoryManager& memory, StreamManager& streams,
                       id<MTLCommandQueue> default_queue) noexcept;
};

}  // namespace muda::runtime::cudart

/**
 * @file buffer_pool.hpp
 * @brief High-performance segregated-freelist size-classed buffer pool for Metal memory allocations.
 */

#pragma once

#import <Metal/Metal.h>

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

namespace muda::runtime::cudart {

inline constexpr std::size_t kBufferPoolMinBinSize = 256;
inline constexpr std::size_t kBufferPoolMaxBinSize = 64 * 1024 * 1024;  // 64 MiB
inline constexpr std::size_t kBufferPoolMinPower = 8;                   // 2^8 = 256
inline constexpr std::size_t kBufferPoolMaxPower = 26;                  // 2^26 = 67,108,864
inline constexpr std::size_t kBufferPoolNumBins =
    (kBufferPoolMaxPower - kBufferPoolMinPower) + 1;                    // 19 bins
inline constexpr std::size_t kBufferPoolMaxBuffersPerBin = 64;
inline constexpr std::size_t kBufferPoolDefaultMaxTotalBytes =
    1024ULL * 1024ULL * 1024ULL;                                       // 1 GiB

class BufferPool {
 public:
  BufferPool() noexcept;
  ~BufferPool() noexcept;

  BufferPool(const BufferPool&) = delete;
  BufferPool& operator=(const BufferPool&) = delete;
  BufferPool(BufferPool&&) = delete;
  BufferPool& operator=(BufferPool&&) = delete;

  [[nodiscard]] id<MTLBuffer> Acquire(std::size_t size, id<MTLDevice> device,
                                      std::size_t* out_capacity) noexcept;
  void Release(id<MTLBuffer> buffer, std::size_t capacity) noexcept;
  void Purge() noexcept;

  [[nodiscard]] std::size_t TotalCachedBytes() const noexcept;
  [[nodiscard]] bool IsEnabled() const noexcept;
  void SetEnabled(bool enabled) noexcept;

  [[nodiscard]] static std::size_t ClassifyBinCapacity(std::size_t size) noexcept;
  [[nodiscard]] static std::size_t BinIndexForCapacity(std::size_t capacity) noexcept;

 private:
  struct Bin {
    std::size_t capacity = 0;
    std::vector<id<MTLBuffer>> free_buffers;
  };

  mutable std::mutex mutex_;
  std::array<Bin, kBufferPoolNumBins> bins_{};
  std::size_t total_cached_bytes_ = 0;
  std::size_t max_total_bytes_ = kBufferPoolDefaultMaxTotalBytes;
  bool enabled_ = true;
};

}  // namespace muda::runtime::cudart

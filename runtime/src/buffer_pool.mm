/**
 * @file buffer_pool.mm
 * @brief High-performance segregated-freelist size-classed buffer pool for Metal memory allocations.
 */

#include "buffer_pool.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <bit>
#include <cstdlib>
#include <cstring>
#include <string_view>

#include "memory_manager.hpp"

namespace muda::runtime::cudart {
namespace {

[[nodiscard]] bool CheckPoolDisabledEnv() noexcept {
  const char* const env = std::getenv("MUDA_DISABLE_BUFFER_POOL");
  if (env == nullptr) {
    return false;
  }
  const std::string_view val{env};
  return val == "1" || val == "true" || val == "TRUE" || val == "yes" || val == "YES";
}

[[nodiscard]] std::size_t CheckPoolMaxBytesEnv() noexcept {
  const char* const env = std::getenv("MUDA_BUFFER_POOL_MAX_BYTES");
  if (env == nullptr || *env == '\0') {
    return kBufferPoolDefaultMaxTotalBytes;
  }
  char* end_ptr = nullptr;
  const unsigned long long parsed = std::strtoull(env, &end_ptr, 10);
  if (end_ptr == env || parsed == 0) {
    return kBufferPoolDefaultMaxTotalBytes;
  }
  return static_cast<std::size_t>(parsed);
}

}  // namespace

BufferPool::BufferPool() noexcept
    : max_total_bytes_(CheckPoolMaxBytesEnv()),
      enabled_(!CheckPoolDisabledEnv()) {
  for (std::size_t i = 0; i < kBufferPoolNumBins; ++i) {
    bins_[i].capacity = std::size_t{1} << (kBufferPoolMinPower + i);
  }
}

BufferPool::~BufferPool() noexcept {
  Purge();
}

std::size_t BufferPool::ClassifyBinCapacity(std::size_t size) noexcept {
  if (size <= kBufferPoolMinBinSize) {
    return kBufferPoolMinBinSize;
  }
  if (size > kBufferPoolMaxBinSize) {
    return AlignUp(size, kMinAllocationAlignment);
  }
  return std::bit_ceil(size);
}

std::size_t BufferPool::BinIndexForCapacity(std::size_t capacity) noexcept {
  const auto power = static_cast<std::size_t>(std::countr_zero(capacity));
  if (power < kBufferPoolMinPower) {
    return 0;
  }
  const std::size_t idx = power - kBufferPoolMinPower;
  if (idx >= kBufferPoolNumBins) {
    return kBufferPoolNumBins - 1;
  }
  return idx;
}

id<MTLBuffer> BufferPool::Acquire(std::size_t size, id<MTLDevice> device,
                                  std::size_t* out_capacity) noexcept {
  if (device == nil || out_capacity == nullptr || size == 0) {
    return nil;
  }

  if (!enabled_ || size > kBufferPoolMaxBinSize) {
    const std::size_t capacity = AlignUp(size, kMinAllocationAlignment);
    *out_capacity = capacity;
    return [device newBufferWithLength:capacity options:MTLResourceStorageModeShared];
  }

  const std::size_t target_capacity = ClassifyBinCapacity(size);
  const std::size_t bin_idx = BinIndexForCapacity(target_capacity);

  id<MTLBuffer> pooled_buffer = nil;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& free_list = bins_[bin_idx].free_buffers;
    if (!free_list.empty()) {
      pooled_buffer = free_list.back();
      free_list.pop_back();
      total_cached_bytes_ -= target_capacity;
    }
  }

  if (pooled_buffer != nil) {
    *out_capacity = target_capacity;
    return pooled_buffer;
  }

  id<MTLBuffer> new_buffer = [device newBufferWithLength:target_capacity
                                                 options:MTLResourceStorageModeShared];
  if (new_buffer == nil) {
    // Under memory pressure: purge cached buffers and attempt allocation once more.
    Purge();
    new_buffer = [device newBufferWithLength:target_capacity
                                     options:MTLResourceStorageModeShared];
  }

  *out_capacity = target_capacity;
  return new_buffer;
}

void BufferPool::Release(id<MTLBuffer> buffer, std::size_t capacity) noexcept {
  if (buffer == nil || !enabled_ || capacity < kBufferPoolMinBinSize ||
      capacity > kBufferPoolMaxBinSize) {
    return;
  }
  if (!std::has_single_bit(capacity)) {
    return;
  }

  const std::size_t bin_idx = BinIndexForCapacity(capacity);

  std::lock_guard<std::mutex> lock(mutex_);
  if (bins_[bin_idx].free_buffers.size() >= kBufferPoolMaxBuffersPerBin) {
    return;
  }
  if (total_cached_bytes_ + capacity > max_total_bytes_) {
    return;
  }

  bins_[bin_idx].free_buffers.push_back(buffer);
  total_cached_bytes_ += capacity;
}

void BufferPool::Purge() noexcept {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto& bin : bins_) {
    bin.free_buffers.clear();
  }
  total_cached_bytes_ = 0;
}

std::size_t BufferPool::TotalCachedBytes() const noexcept {
  std::lock_guard<std::mutex> lock(mutex_);
  return total_cached_bytes_;
}

bool BufferPool::IsEnabled() const noexcept {
  return enabled_;
}

void BufferPool::SetEnabled(bool enabled) noexcept {
  if (!enabled) {
    Purge();
  }
  enabled_ = enabled;
}

}  // namespace muda::runtime::cudart

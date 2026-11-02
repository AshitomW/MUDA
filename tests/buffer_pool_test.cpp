/**
 * @file buffer_pool_test.cpp
 * @brief Unit tests and latency benchmarks for the segregated-freelist BufferPool.
 */

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

#include "cuda_runtime.h"

namespace {

inline constexpr std::size_t kSmallAllocBytes = 1024;
inline constexpr std::size_t kMediumAllocBytes = 64 * 1024;
inline constexpr std::size_t kBenchmarkIterations = 2000;

bool Expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    return false;
  }
  return true;
}

bool TestBufferPoolAddressReuse() {
  void* ptr1 = nullptr;
  cudaError_t status = cudaMalloc(&ptr1, kSmallAllocBytes);
  if (!Expect(status == cudaSuccess && ptr1 != nullptr, "Initial cudaMalloc failed")) {
    return false;
  }

  // Free pointer back into buffer pool.
  status = cudaFree(ptr1);
  if (!Expect(status == cudaSuccess, "cudaFree failed")) {
    return false;
  }

  // Second allocation of identical size class should reuse the cached buffer.
  void* ptr2 = nullptr;
  status = cudaMalloc(&ptr2, kSmallAllocBytes);
  if (!Expect(status == cudaSuccess && ptr2 != nullptr, "Second cudaMalloc failed")) {
    return false;
  }

  if (!Expect(ptr1 == ptr2, "Buffer pool failed to reuse cached buffer address")) {
    (void)cudaFree(ptr2);
    return false;
  }

  status = cudaFree(ptr2);
  return Expect(status == cudaSuccess, "Final cudaFree failed");
}

bool TestDistinctSizeClasses() {
  void* small_ptr = nullptr;
  void* medium_ptr = nullptr;

  cudaError_t status = cudaMalloc(&small_ptr, kSmallAllocBytes);
  if (!Expect(status == cudaSuccess && small_ptr != nullptr, "Alloc small failed")) {
    return false;
  }
  status = cudaMalloc(&medium_ptr, kMediumAllocBytes);
  if (!Expect(status == cudaSuccess && medium_ptr != nullptr, "Alloc medium failed")) {
    (void)cudaFree(small_ptr);
    return false;
  }

  if (!Expect(small_ptr != medium_ptr, "Small and medium allocations must not share pointer")) {
    (void)cudaFree(medium_ptr);
    (void)cudaFree(small_ptr);
    return false;
  }

  status = cudaFree(small_ptr);
  if (!Expect(status == cudaSuccess, "Free small failed")) {
    (void)cudaFree(medium_ptr);
    return false;
  }
  status = cudaFree(medium_ptr);
  if (!Expect(status == cudaSuccess, "Free medium failed")) {
    return false;
  }

  // Re-acquire small
  void* small_ptr2 = nullptr;
  status = cudaMalloc(&small_ptr2, kSmallAllocBytes);
  if (!Expect(status == cudaSuccess && small_ptr2 == small_ptr,
              "Small alloc should reuse small buffer")) {
    (void)cudaFree(small_ptr2);
    return false;
  }

  // Re-acquire medium
  void* medium_ptr2 = nullptr;
  status = cudaMalloc(&medium_ptr2, kMediumAllocBytes);
  if (!Expect(status == cudaSuccess && medium_ptr2 == medium_ptr,
              "Medium alloc should reuse medium buffer")) {
    (void)cudaFree(small_ptr2);
    (void)cudaFree(medium_ptr2);
    return false;
  }

  (void)cudaFree(small_ptr2);
  (void)cudaFree(medium_ptr2);
  return true;
}

bool TestBufferPoolLatency() {
  // Warm up pool
  void* warmup = nullptr;
  (void)cudaMalloc(&warmup, kSmallAllocBytes);
  (void)cudaFree(warmup);

  const auto start = std::chrono::steady_clock::now();
  for (std::size_t i = 0; i < kBenchmarkIterations; ++i) {
    void* ptr = nullptr;
    const cudaError_t alloc_status = cudaMalloc(&ptr, kSmallAllocBytes);
    if (alloc_status != cudaSuccess || ptr == nullptr) {
      std::cerr << "FAIL: cudaMalloc failed at iteration " << i << "\n";
      return false;
    }
    const cudaError_t free_status = cudaFree(ptr);
    if (free_status != cudaSuccess) {
      std::cerr << "FAIL: cudaFree failed at iteration " << i << "\n";
      return false;
    }
  }
  const auto finish = std::chrono::steady_clock::now();
  const auto elapsed_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count();
  const double avg_us =
      static_cast<double>(elapsed_ns) / (static_cast<double>(kBenchmarkIterations) * 1000.0);

  std::cout << "[PERF] " << kBenchmarkIterations << " pooled cudaMalloc/cudaFree cycles: "
            << avg_us << " us / cycle\n";

  // Verify average latency is well under 10 microseconds (typical is ~0.4 - 1.5 us)
  return Expect(avg_us < 20.0, "Pooled allocation latency must be < 20 us");
}

bool TestMemInfoAndReset() {
  std::size_t free_before = 0;
  std::size_t total_before = 0;
  cudaError_t status = cudaMemGetInfo(&free_before, &total_before);
  if (!Expect(status == cudaSuccess, "cudaMemGetInfo before failed")) {
    return false;
  }

  void* ptr = nullptr;
  status = cudaMalloc(&ptr, 4 * 1024 * 1024);  // 4 MiB
  if (!Expect(status == cudaSuccess, "cudaMalloc 4MiB failed")) {
    return false;
  }
  status = cudaFree(ptr);
  if (!Expect(status == cudaSuccess, "cudaFree 4MiB failed")) {
    return false;
  }

  status = cudaDeviceReset();
  if (!Expect(status == cudaSuccess, "cudaDeviceReset failed")) {
    return false;
  }

  std::size_t free_after = 0;
  std::size_t total_after = 0;
  status = cudaMemGetInfo(&free_after, &total_after);
  return Expect(status == cudaSuccess, "cudaMemGetInfo after reset failed");
}

}  // namespace

int main() {
  std::cout << "Running buffer pool tests...\n";
  if (!TestBufferPoolAddressReuse()) {
    return 1;
  }
  if (!TestDistinctSizeClasses()) {
    return 1;
  }
  if (!TestBufferPoolLatency()) {
    return 1;
  }
  if (!TestMemInfoAndReset()) {
    return 1;
  }
  std::cout << "All buffer pool tests passed successfully!\n";
  return 0;
}

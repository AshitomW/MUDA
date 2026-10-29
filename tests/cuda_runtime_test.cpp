/**
 * @file cuda_runtime_test.cpp
 * @brief Tests the CUDA Runtime C API subset for memory, device, and error state.
 */

#include "cuda_runtime.h"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

namespace {

bool Expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    return false;
  }
  return true;
}

bool TestErrorReporting() {
  (void)cudaGetLastError();
  if (!Expect(cudaPeekAtLastError() == cudaSuccess, "initial error state should be cudaSuccess")) {
    return false;
  }
  if (!Expect(std::strcmp(cudaGetErrorName(cudaSuccess), "cudaSuccess") == 0,
              "unexpected cudaSuccess name")) {
    return false;
  }
  if (!Expect(std::strcmp(cudaGetErrorName(cudaErrorInvalidValue), "cudaErrorInvalidValue") == 0,
              "unexpected cudaErrorInvalidValue name")) {
    return false;
  }
  if (!Expect(std::strlen(cudaGetErrorString(cudaErrorMemoryAllocation)) > 0,
              "error string should not be empty")) {
    return false;
  }

  if (!Expect(cudaMalloc(nullptr, 64) == cudaErrorInvalidValue,
              "cudaMalloc(nullptr, 64) should fail with cudaErrorInvalidValue")) {
    return false;
  }
  if (!Expect(cudaPeekAtLastError() == cudaErrorInvalidValue,
              "cudaPeekAtLastError should preserve last error")) {
    return false;
  }
  if (!Expect(cudaGetLastError() == cudaErrorInvalidValue,
              "cudaGetLastError should return last error")) {
    return false;
  }
  if (!Expect(cudaGetLastError() == cudaSuccess,
              "cudaGetLastError should clear last error to cudaSuccess")) {
    return false;
  }
  return true;
}

bool TestDeviceManagement(int* out_count) {
  if (!Expect(cudaGetDeviceCount(nullptr) == cudaErrorInvalidValue,
              "cudaGetDeviceCount(nullptr) should fail")) {
    return false;
  }
  (void)cudaGetLastError();

  int count = -1;
  if (!Expect(cudaGetDeviceCount(&count) == cudaSuccess, "cudaGetDeviceCount should succeed")) {
    return false;
  }
  if (!Expect(count == 0 || count == 1, "device count must be 0 or 1")) {
    return false;
  }
  *out_count = count;

  if (!Expect(cudaSetDevice(1) == cudaErrorInvalidDevice,
              "cudaSetDevice(1) should return cudaErrorInvalidDevice")) {
    return false;
  }
  if (!Expect(cudaGetLastError() == cudaErrorInvalidDevice,
              "last error should reflect cudaErrorInvalidDevice")) {
    return false;
  }
  if (!Expect(cudaGetDevice(nullptr) == cudaErrorInvalidValue,
              "cudaGetDevice(nullptr) should fail")) {
    return false;
  }
  (void)cudaGetLastError();

  if (count == 0) {
    return true;
  }

  if (!Expect(cudaSetDevice(0) == cudaSuccess, "cudaSetDevice(0) should succeed")) {
    return false;
  }
  int active_device = -1;
  if (!Expect(cudaGetDevice(&active_device) == cudaSuccess && active_device == 0,
              "cudaGetDevice should report device 0")) {
    return false;
  }
  return true;
}

bool TestMemoryAndSync() {
  void* raw_ptr = nullptr;
  if (!Expect(cudaMalloc(&raw_ptr, 0) == cudaErrorInvalidValue,
              "cudaMalloc with size 0 should fail")) {
    return false;
  }
  (void)cudaGetLastError();

  if (!Expect(cudaMallocManaged(&raw_ptr, 64, 0x99u) == cudaErrorInvalidValue,
              "cudaMallocManaged with invalid flags should fail")) {
    return false;
  }
  (void)cudaGetLastError();

  if (!Expect(cudaFree(nullptr) == cudaSuccess, "cudaFree(nullptr) should be a no-op")) {
    return false;
  }

  constexpr std::size_t kElements = 256;
  constexpr std::size_t kBytes = kElements * sizeof(int);

  int* dev_a = nullptr;
  int* dev_b = nullptr;
  if (!Expect(
          cudaMalloc(reinterpret_cast<void**>(&dev_a), kBytes) == cudaSuccess && dev_a != nullptr,
          "cudaMalloc(dev_a) should succeed")) {
    return false;
  }
  if (!Expect((reinterpret_cast<std::uintptr_t>(dev_a) % 256u) == 0u,
              "cudaMalloc pointer should be at least 256-byte aligned")) {
    return false;
  }

  if (!Expect(cudaMallocManaged(reinterpret_cast<void**>(&dev_b), kBytes) == cudaSuccess &&
                  dev_b != nullptr,
              "cudaMallocManaged(dev_b) should succeed")) {
    return false;
  }

  if (!Expect(cudaMemset(dev_a, 0, kBytes) == cudaSuccess, "cudaMemset should succeed")) {
    return false;
  }

  std::vector<int> host_in(kElements);
  std::vector<int> host_out(kElements, -1);
  for (std::size_t index = 0; index < kElements; ++index) {
    host_in[index] = static_cast<int>(index * 3 + 7);
  }

  if (!Expect(cudaMemcpy(dev_a, host_in.data(), kBytes, cudaMemcpyHostToDevice) == cudaSuccess,
              "cudaMemcpy HostToDevice should succeed")) {
    return false;
  }

  // Test sub-range and pointer-offset arithmetic within an allocation.
  constexpr std::size_t kHalfElements = kElements / 2;
  constexpr std::size_t kHalfBytes = kHalfElements * sizeof(int);
  if (!Expect(cudaMemset(dev_b, 0, kBytes) == cudaSuccess, "cudaMemset(dev_b) should succeed")) {
    return false;
  }
  if (!Expect(cudaMemcpy(dev_b + kHalfElements, dev_a + kHalfElements, kHalfBytes,
                         cudaMemcpyDeviceToDevice) == cudaSuccess,
              "cudaMemcpy DeviceToDevice with pointer offsets should succeed")) {
    return false;
  }
  if (!Expect(cudaMemcpy(dev_b, dev_a, kHalfBytes, cudaMemcpyDefault) == cudaSuccess,
              "cudaMemcpy Default should succeed")) {
    return false;
  }
  if (!Expect(cudaDeviceSynchronize() == cudaSuccess, "cudaDeviceSynchronize should succeed")) {
    return false;
  }
  if (!Expect(cudaMemcpy(host_out.data(), dev_b, kBytes, cudaMemcpyDeviceToHost) == cudaSuccess,
              "cudaMemcpy DeviceToHost should succeed")) {
    return false;
  }

  for (std::size_t index = 0; index < kElements; ++index) {
    if (!Expect(host_out[index] == host_in[index], "copied value mismatch")) {
      return false;
    }
  }

  // Negative/rejection cases for bounds and direction validation.
  if (!Expect(cudaMemcpy(dev_a, host_in.data(), kBytes + 4, cudaMemcpyHostToDevice) ==
                  cudaErrorInvalidValue,
              "out-of-bounds cudaMemcpy should fail")) {
    return false;
  }
  (void)cudaGetLastError();

  if (!Expect(cudaMemcpy(host_out.data(), host_in.data(), kBytes, cudaMemcpyHostToDevice) ==
                  cudaErrorInvalidValue,
              "HostToDevice with host destination pointer should fail")) {
    return false;
  }
  (void)cudaGetLastError();

  if (!Expect(cudaMemcpy(dev_a, host_in.data(), kBytes, cudaMemcpyDeviceToHost) ==
                  cudaErrorInvalidValue,
              "DeviceToHost with host source pointer should fail")) {
    return false;
  }
  (void)cudaGetLastError();

  if (!Expect(cudaFree(dev_a + 1) == cudaErrorInvalidValue,
              "cudaFree on interior pointer should fail")) {
    return false;
  }
  (void)cudaGetLastError();

  if (!Expect(cudaFree(dev_a) == cudaSuccess, "cudaFree(dev_a) should succeed")) {
    return false;
  }
  if (!Expect(cudaFree(dev_a) == cudaErrorInvalidValue, "double cudaFree(dev_a) should fail")) {
    return false;
  }
  (void)cudaGetLastError();

  if (!Expect(cudaFree(dev_b) == cudaSuccess, "cudaFree(dev_b) should succeed")) {
    return false;
  }
  if (!Expect(cudaDeviceReset() == cudaSuccess, "cudaDeviceReset should succeed")) {
    return false;
  }
  if (!Expect(cudaGetLastError() == cudaSuccess, "no residual error should remain")) {
    return false;
  }

  return true;
}

}  // namespace

int main() {
  if (!TestErrorReporting()) {
    return 1;
  }

  int device_count = 0;
  if (!TestDeviceManagement(&device_count)) {
    return 1;
  }

  if (device_count == 0) {
    std::cout << "no Metal devices available; host error checks passed\n";
    return 0;
  }

  if (!TestMemoryAndSync()) {
    return 1;
  }

  std::cout << "cuda_runtime_test passed\n";
  return 0;
}

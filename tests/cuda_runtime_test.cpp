/**
 * @file cuda_runtime_test.cpp
 * @brief Tests the CUDA Runtime C API subset for memory, device, kernel launch, and error state.
 */

#include "cuda_runtime.h"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "muda/compiler/driver.hpp"
#include "muda/compiler/toolchain.hpp"
#include "test_utils.hpp"

namespace {

void VectorAddHostStub() {}
void VectorAddDataHostStub() {}
void UnregisteredHostStub() {}

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
  if (!Expect(std::strcmp(cudaGetErrorName(cudaErrorInvalidConfiguration),
                          "cudaErrorInvalidConfiguration") == 0,
              "unexpected cudaErrorInvalidConfiguration name")) {
    return false;
  }
  if (!Expect(std::strcmp(cudaGetErrorName(cudaErrorInvalidDeviceFunction),
                          "cudaErrorInvalidDeviceFunction") == 0,
              "unexpected cudaErrorInvalidDeviceFunction name")) {
    return false;
  }
  if (!Expect(std::strlen(cudaGetErrorString(cudaErrorMemoryAllocation)) > 0,
              "error string should not be empty")) {
    return false;
  }
  if (!Expect(std::strlen(cudaGetErrorString(cudaErrorInvalidConfiguration)) > 0,
              "configuration error string should not be empty")) {
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
  if (!Expect(
          cudaMemcpy(dev_a + kElements, host_in.data(), 0, cudaMemcpyHostToDevice) == cudaSuccess,
          "zero-length HostToDevice at end pointer should succeed")) {
    return false;
  }
  if (!Expect(
          cudaMemcpy(host_out.data(), dev_a + kElements, 0, cudaMemcpyDeviceToHost) == cudaSuccess,
          "zero-length DeviceToHost at end pointer should succeed")) {
    return false;
  }
  if (!Expect(cudaMemset(dev_a + kElements, 0, 0) == cudaSuccess,
              "zero-length cudaMemset at end pointer should succeed")) {
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

bool TestKernelLaunch() {
  const auto host_fn = reinterpret_cast<const void*>(&VectorAddHostStub);
  const auto data_host_fn = reinterpret_cast<const void*>(&VectorAddDataHostStub);
  const auto unregistered_fn = reinterpret_cast<const void*>(&UnregisteredHostStub);

  const mudaKernelParamDesc params[4] = {
      {mudaKernelParamBuffer, sizeof(void*)},
      {mudaKernelParamBuffer, sizeof(void*)},
      {mudaKernelParamBuffer, sizeof(void*)},
      {mudaKernelParamBytes, sizeof(int)},
  };

  if (!Expect(mudaRegisterKernelFromFile(nullptr, "nonexistent.metallib", "vector_add", params,
                                         4) == cudaErrorInvalidValue,
              "null hostFunc should fail")) {
    return false;
  }
  (void)cudaGetLastError();

  if (!Expect(
          mudaRegisterKernelFromFile(host_fn, "", "vector_add", params, 4) == cudaErrorInvalidValue,
          "empty metallibPath should fail")) {
    return false;
  }
  (void)cudaGetLastError();

  if (!Expect(mudaRegisterKernelFromFile(host_fn, "nonexistent.metallib", "vector_add", nullptr,
                                         4) == cudaErrorInvalidValue,
              "null params with non-zero paramCount should fail")) {
    return false;
  }
  (void)cudaGetLastError();

  const mudaKernelParamDesc bad_bytes_param[1] = {{mudaKernelParamBytes, 0}};
  if (!Expect(mudaRegisterKernelFromFile(host_fn, "nonexistent.metallib", "vector_add",
                                         bad_bytes_param, 1) == cudaErrorInvalidValue,
              "zero-sized mudaKernelParamBytes should fail")) {
    return false;
  }
  (void)cudaGetLastError();

  if (!Expect(cudaLaunchKernel(unregistered_fn, dim3(1), dim3(1), nullptr, 0, nullptr) ==
                  cudaErrorInvalidDeviceFunction,
              "launching unregistered function should return cudaErrorInvalidDeviceFunction")) {
    return false;
  }
  (void)cudaGetLastError();

  if (!Expect(cudaLaunchKernel(unregistered_fn, dim3(0, 1, 1), dim3(1), nullptr, 0, nullptr) ==
                  cudaErrorInvalidConfiguration,
              "zero gridDim should return cudaErrorInvalidConfiguration")) {
    return false;
  }
  (void)cudaGetLastError();

  if (!Expect(
          cudaLaunchKernel(unregistered_fn, dim3(1), dim3(1), nullptr, 0,
                           reinterpret_cast<cudaStream_t>(0x1)) == cudaErrorInvalidResourceHandle,
          "non-null invalid stream should return cudaErrorInvalidResourceHandle")) {
    return false;
  }
  (void)cudaGetLastError();

  if (!muda::compiler::MetalToolchainAvailable()) {
    std::cout << "Metal toolchain unavailable; skipping metallib GPU launch checks\n";
    return true;
  }

  const muda::testing::ScopedTempDir temp_dir("muda_cudart_launch_test");
  if (!Expect(temp_dir.valid(), "failed to create temp directory for kernel launch test")) {
    return false;
  }

  muda::compiler::CompileOptions options;
  options.input = std::filesystem::path(MUDA_SOURCE_DIR) / "samples/vector_add.cu";
  options.output = temp_dir.path() / "vector_add.metallib";

  const muda::compiler::CompileResult compiled = muda::compiler::Compile(options);
  if (!Expect(compiled.status == muda::compiler::Status::kOk, "failed to compile vector_add.cu")) {
    return false;
  }

  if (!Expect(
          mudaRegisterKernelFromFile(host_fn, compiled.metallib_path.string().c_str(),
                                     "missing_kernel", params, 4) == cudaErrorInvalidDeviceFunction,
          "registering missing kernel name should return cudaErrorInvalidDeviceFunction")) {
    return false;
  }
  (void)cudaGetLastError();

  if (!Expect(mudaRegisterKernelFromFile(host_fn, compiled.metallib_path.string().c_str(),
                                         compiled.kernel_name.c_str(), params, 4) == cudaSuccess,
              "mudaRegisterKernelFromFile should succeed")) {
    return false;
  }

  std::ifstream lib_stream(compiled.metallib_path, std::ios::binary | std::ios::ate);
  const std::streamoff raw_lib_size = lib_stream.tellg();
  if (!Expect(raw_lib_size > 0, "compiled metallib should not be empty")) {
    return false;
  }
  const auto lib_size = static_cast<std::size_t>(raw_lib_size);
  lib_stream.seekg(0, std::ios::beg);
  std::vector<char> lib_bytes(lib_size);
  lib_stream.read(lib_bytes.data(), static_cast<std::streamsize>(lib_size));

  if (!Expect(mudaRegisterKernelFromData(data_host_fn, lib_bytes.data(), lib_bytes.size(),
                                         compiled.kernel_name.c_str(), params, 4) == cudaSuccess,
              "mudaRegisterKernelFromData should succeed")) {
    return false;
  }

  if (!Expect(cudaLaunchKernel(host_fn, dim3(1), dim3(2048, 1, 1), nullptr, 0, nullptr) ==
                  cudaErrorInvalidConfiguration,
              "oversized blockDim should return cudaErrorInvalidConfiguration")) {
    return false;
  }
  (void)cudaGetLastError();

  if (!Expect(cudaLaunchKernel(host_fn, dim3(1), dim3(64), nullptr, 1u << 24, nullptr) ==
                  cudaErrorInvalidConfiguration,
              "oversized sharedMem should return cudaErrorInvalidConfiguration")) {
    return false;
  }
  (void)cudaGetLastError();

  if (!Expect(cudaLaunchKernel(host_fn, dim3(1), dim3(64), nullptr, 0, nullptr) ==
                  cudaErrorInvalidValue,
              "null args array should return cudaErrorInvalidValue")) {
    return false;
  }
  (void)cudaGetLastError();

  constexpr std::size_t kCount = 1024;
  constexpr unsigned int kBlockSize = 256;
  constexpr unsigned int kGridSize =
      static_cast<unsigned int>((kCount + kBlockSize - 1) / kBlockSize);
  constexpr std::size_t kBytes = kCount * sizeof(float);

  float* dev_a = nullptr;
  float* dev_b = nullptr;
  float* dev_out = nullptr;
  if (!Expect(cudaMallocManaged(reinterpret_cast<void**>(&dev_a), kBytes) == cudaSuccess &&
                  cudaMallocManaged(reinterpret_cast<void**>(&dev_b), kBytes) == cudaSuccess &&
                  cudaMallocManaged(reinterpret_cast<void**>(&dev_out), kBytes) == cudaSuccess,
              "cudaMallocManaged for launch buffers should succeed")) {
    return false;
  }

  for (std::size_t index = 0; index < kCount; ++index) {
    dev_a[index] = 1.0f;
    dev_b[index] = 2.0f;
    dev_out[index] = 0.0f;
  }

  float host_dummy = 0.0f;
  float* bad_ptr = &host_dummy;
  int n = static_cast<int>(kCount);
  void* bad_args[4] = {&bad_ptr, &dev_b, &dev_out, &n};
  if (!Expect(cudaLaunchKernel(host_fn, dim3(kGridSize), dim3(kBlockSize), bad_args, 0, nullptr) ==
                  cudaErrorInvalidValue,
              "unallocated host pointer argument should return cudaErrorInvalidValue")) {
    return false;
  }
  (void)cudaGetLastError();

  void* launch_args[4] = {&dev_a, &dev_b, &dev_out, &n};
  if (!Expect(cudaLaunchKernel(host_fn, dim3(kGridSize), dim3(kBlockSize), launch_args, 0,
                               nullptr) == cudaSuccess,
              "cudaLaunchKernel(vector_add) should succeed")) {
    return false;
  }
  if (!Expect(cudaDeviceSynchronize() == cudaSuccess,
              "cudaDeviceSynchronize after launch should succeed")) {
    return false;
  }

  for (std::size_t index = 0; index < kCount; ++index) {
    if (!Expect(muda::testing::AreClose(dev_out[index], 3.0f), "vector_add output mismatch")) {
      return false;
    }
  }

  constexpr std::size_t kOffset = 512;
  constexpr int kSubCount = 512;
  for (std::size_t index = 0; index < kCount; ++index) {
    dev_a[index] = 10.0f;
    dev_b[index] = 20.0f;
    dev_out[index] = -1.0f;
  }

  float* sub_a = dev_a + kOffset;
  float* sub_b = dev_b + kOffset;
  float* sub_out = dev_out + kOffset;
  int sub_n = kSubCount;
  void* sub_args[4] = {&sub_a, &sub_b, &sub_out, &sub_n};
  if (!Expect(cudaLaunchKernel(data_host_fn, dim3(2), dim3(kBlockSize), sub_args, 0, nullptr) ==
                  cudaSuccess,
              "cudaLaunchKernel with pointer offsets should succeed")) {
    return false;
  }
  if (!Expect(cudaDeviceSynchronize() == cudaSuccess,
              "cudaDeviceSynchronize after offset launch should succeed")) {
    return false;
  }

  for (std::size_t index = 0; index < kOffset; ++index) {
    if (!Expect(muda::testing::AreClose(dev_out[index], -1.0f),
                "elements before offset should be untouched")) {
      return false;
    }
  }
  for (std::size_t index = kOffset; index < kCount; ++index) {
    if (!Expect(muda::testing::AreClose(dev_out[index], 30.0f),
                "elements after offset should equal 30.0f")) {
      return false;
    }
  }

  if (!Expect(cudaFree(dev_a) == cudaSuccess && cudaFree(dev_b) == cudaSuccess &&
                  cudaFree(dev_out) == cudaSuccess,
              "cudaFree of launch buffers should succeed")) {
    return false;
  }
  if (!Expect(cudaDeviceReset() == cudaSuccess, "cudaDeviceReset after launch should succeed")) {
    return false;
  }
  return true;
}

bool TestDevicePropertiesAndPointerAttributes() {
  int runtime_ver = 0;
  int driver_ver = 0;
  if (!Expect(cudaRuntimeGetVersion(nullptr) == cudaErrorInvalidValue,
              "cudaRuntimeGetVersion(nullptr) should fail")) {
    return false;
  }
  (void)cudaGetLastError();
  if (!Expect(cudaRuntimeGetVersion(&runtime_ver) == cudaSuccess && runtime_ver >= 11000,
              "cudaRuntimeGetVersion should succeed")) {
    return false;
  }
  if (!Expect(cudaDriverGetVersion(&driver_ver) == cudaSuccess && driver_ver >= 11000,
              "cudaDriverGetVersion should succeed")) {
    return false;
  }

  cudaDeviceProp prop{};
  if (!Expect(cudaGetDeviceProperties(nullptr, 0) == cudaErrorInvalidValue,
              "cudaGetDeviceProperties(nullptr, 0) should fail")) {
    return false;
  }
  (void)cudaGetLastError();
  if (!Expect(cudaGetDeviceProperties(&prop, 1) == cudaErrorInvalidDevice,
              "cudaGetDeviceProperties(&prop, 1) should fail")) {
    return false;
  }
  (void)cudaGetLastError();
  if (!Expect(cudaGetDeviceProperties(&prop, 0) == cudaSuccess,
              "cudaGetDeviceProperties(&prop, 0) should succeed")) {
    return false;
  }
  if (!Expect(std::strlen(prop.name) > 0 && prop.warpSize == 32 && prop.maxThreadsPerBlock > 0 &&
                  prop.sharedMemPerBlock > 0 && prop.totalGlobalMem > 0 &&
                  prop.unifiedAddressing == 1 && prop.managedMemory == 1,
              "synthesized cudaDeviceProp fields should be populated")) {
    return false;
  }

  int warp_size = 0;
  int max_threads = 0;
  if (!Expect(cudaDeviceGetAttribute(&warp_size, cudaDevAttrWarpSize, 0) == cudaSuccess &&
                  warp_size == 32,
              "cudaDevAttrWarpSize should be 32")) {
    return false;
  }
  if (!Expect(
          cudaDeviceGetAttribute(&max_threads, cudaDevAttrMaxThreadsPerBlock, 0) == cudaSuccess &&
              max_threads == prop.maxThreadsPerBlock,
          "cudaDevAttrMaxThreadsPerBlock should match prop.maxThreadsPerBlock")) {
    return false;
  }

  std::size_t free_mem = 0;
  std::size_t total_mem = 0;
  if (!Expect(cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess && total_mem > 0 &&
                  free_mem <= total_mem,
              "cudaMemGetInfo should report valid memory bounds")) {
    return false;
  }

  void* dev_ptr = nullptr;
  void* managed_ptr = nullptr;
  void* host_ptr = nullptr;
  if (!Expect(cudaMalloc(&dev_ptr, 512) == cudaSuccess &&
                  cudaMallocManaged(&managed_ptr, 512) == cudaSuccess &&
                  cudaMallocHost(&host_ptr, 512) == cudaSuccess,
              "allocations for pointer attribute checks should succeed")) {
    return false;
  }

  cudaPointerAttributes attrs{};
  if (!Expect(cudaPointerGetAttributes(&attrs, static_cast<char*>(dev_ptr) + 64) == cudaSuccess &&
                  attrs.type == cudaMemoryTypeDevice && attrs.device == 0,
              "cudaPointerGetAttributes on device interior pointer should report Device")) {
    return false;
  }
  if (!Expect(cudaPointerGetAttributes(&attrs, managed_ptr) == cudaSuccess &&
                  attrs.type == cudaMemoryTypeManaged && attrs.device == 0,
              "cudaPointerGetAttributes on managed pointer should report Managed")) {
    return false;
  }
  if (!Expect(cudaPointerGetAttributes(&attrs, host_ptr) == cudaSuccess &&
                  attrs.type == cudaMemoryTypeHost && attrs.device == 0,
              "cudaPointerGetAttributes on pinned host pointer should report Host")) {
    return false;
  }

  int stack_var = 42;
  if (!Expect(cudaPointerGetAttributes(&attrs, &stack_var) == cudaSuccess &&
                  attrs.type == cudaMemoryTypeUnregistered,
              "cudaPointerGetAttributes on stack pointer should report Unregistered")) {
    return false;
  }

  if (!Expect(cudaFree(dev_ptr) == cudaSuccess && cudaFree(managed_ptr) == cudaSuccess &&
                  cudaFreeHost(host_ptr) == cudaSuccess,
              "freeing attribute test pointers should succeed")) {
    return false;
  }
  return true;
}

bool TestPitchAnd2DMemory() {
  constexpr std::size_t kWidthCols = 17;
  constexpr std::size_t kHeightRows = 8;
  constexpr std::size_t kRowBytes = kWidthCols * sizeof(int);

  int* pitched_dev = nullptr;
  std::size_t pitch = 0;
  if (!Expect(cudaMallocPitch(reinterpret_cast<void**>(&pitched_dev), &pitch, kRowBytes,
                              kHeightRows) == cudaSuccess &&
                  pitched_dev != nullptr && pitch >= kRowBytes && (pitch % 256u) == 0u,
              "cudaMallocPitch should succeed with 256-byte aligned pitch")) {
    return false;
  }

  if (!Expect(cudaMemset2D(pitched_dev, pitch, 0, kRowBytes, kHeightRows) == cudaSuccess,
              "cudaMemset2D should succeed")) {
    return false;
  }

  std::vector<int> host_src(kWidthCols * kHeightRows);
  std::vector<int> host_dst(kWidthCols * kHeightRows, -1);
  for (std::size_t r = 0; r < kHeightRows; ++r) {
    for (std::size_t c = 0; c < kWidthCols; ++c) {
      host_src[r * kWidthCols + c] = static_cast<int>(r * 100 + c + 1);
    }
  }

  if (!Expect(cudaMemcpy2D(pitched_dev, pitch, host_src.data(), kRowBytes, kRowBytes, kHeightRows,
                           cudaMemcpyHostToDevice) == cudaSuccess,
              "cudaMemcpy2D HostToDevice should succeed")) {
    return false;
  }
  if (!Expect(cudaMemcpy2D(host_dst.data(), kRowBytes, pitched_dev, pitch, kRowBytes, kHeightRows,
                           cudaMemcpyDeviceToHost) == cudaSuccess,
              "cudaMemcpy2D DeviceToHost should succeed")) {
    return false;
  }

  for (std::size_t idx = 0; idx < host_src.size(); ++idx) {
    if (!Expect(host_dst[idx] == host_src[idx], "cudaMemcpy2D round-trip mismatch")) {
      return false;
    }
  }

  if (!Expect(cudaMemcpy2D(host_dst.data(), kRowBytes - 4, pitched_dev, pitch, kRowBytes,
                           kHeightRows, cudaMemcpyDeviceToHost) == cudaErrorInvalidPitchValue,
              "cudaMemcpy2D with width > dpitch should return cudaErrorInvalidPitchValue")) {
    return false;
  }
  (void)cudaGetLastError();

  if (!Expect(cudaFree(pitched_dev) == cudaSuccess, "cudaFree(pitched_dev) should succeed")) {
    return false;
  }
  return true;
}

bool TestStreamsAndEvents() {
  if (!Expect(cudaStreamCreate(nullptr) == cudaErrorInvalidValue,
              "cudaStreamCreate(nullptr) should fail")) {
    return false;
  }
  (void)cudaGetLastError();

  if (!Expect(cudaStreamCreateWithFlags(nullptr, 0x99u) == cudaErrorInvalidValue,
              "cudaStreamCreateWithFlags invalid flags should fail")) {
    return false;
  }
  (void)cudaGetLastError();

  cudaStream_t stream1 = nullptr;
  cudaStream_t stream2 = nullptr;
  if (!Expect(cudaStreamCreate(&stream1) == cudaSuccess && stream1 != nullptr,
              "cudaStreamCreate(stream1) should succeed")) {
    return false;
  }
  if (!Expect(cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking) == cudaSuccess &&
                  stream2 != nullptr && stream2 != stream1,
              "cudaStreamCreateWithFlags(stream2) should succeed")) {
    return false;
  }

  cudaEvent_t ev_start = nullptr;
  cudaEvent_t ev_end = nullptr;
  cudaEvent_t ev_notiming = nullptr;
  if (!Expect(cudaEventCreate(&ev_start) == cudaSuccess &&
                  cudaEventCreate(&ev_end) == cudaSuccess &&
                  cudaEventCreateWithFlags(&ev_notiming, cudaEventDisableTiming) == cudaSuccess,
              "cudaEventCreate should succeed")) {
    return false;
  }

  // Type-safety validation: passing an event handle cast to cudaStream_t must be rejected
  const auto bogus_stream = reinterpret_cast<cudaStream_t>(ev_start);
  if (!Expect(
          cudaStreamSynchronize(bogus_stream) == cudaErrorInvalidResourceHandle,
          "passing event handle to cudaStreamSynchronize must fail with InvalidResourceHandle")) {
    return false;
  }
  (void)cudaGetLastError();

  const auto bogus_event = reinterpret_cast<cudaEvent_t>(stream1);
  if (!Expect(
          cudaEventSynchronize(bogus_event) == cudaErrorInvalidResourceHandle,
          "passing stream handle to cudaEventSynchronize must fail with InvalidResourceHandle")) {
    return false;
  }
  (void)cudaGetLastError();

  constexpr std::size_t kElements = 256;
  constexpr std::size_t kBytes = kElements * sizeof(int);
  int* dev_src = nullptr;
  int* dev_dst = nullptr;
  if (!Expect(cudaMalloc(reinterpret_cast<void**>(&dev_src), kBytes) == cudaSuccess &&
                  cudaMalloc(reinterpret_cast<void**>(&dev_dst), kBytes) == cudaSuccess,
              "cudaMalloc for async stream test should succeed")) {
    return false;
  }

  if (!Expect(cudaEventRecord(ev_start, stream1) == cudaSuccess,
              "cudaEventRecord(ev_start) should succeed")) {
    return false;
  }
  if (!Expect(cudaMemsetAsync(dev_src, 0x11, kBytes, stream1) == cudaSuccess,
              "cudaMemsetAsync on stream1 should succeed")) {
    return false;
  }
  if (!Expect(cudaMemcpyAsync(dev_dst, dev_src, kBytes, cudaMemcpyDeviceToDevice, stream1) ==
                  cudaSuccess,
              "cudaMemcpyAsync D2D on stream1 should succeed")) {
    return false;
  }
  if (!Expect(cudaEventRecord(ev_end, stream1) == cudaSuccess,
              "cudaEventRecord(ev_end) should succeed")) {
    return false;
  }

  if (!Expect(cudaStreamWaitEvent(stream2, ev_end, cudaEventWaitDefault) == cudaSuccess,
              "cudaStreamWaitEvent on stream2 should succeed")) {
    return false;
  }

  std::vector<int> host_out(kElements, 0);
  if (!Expect(cudaMemcpyAsync(host_out.data(), dev_dst, kBytes, cudaMemcpyDeviceToHost, stream2) ==
                  cudaSuccess,
              "cudaMemcpyAsync D2H on stream2 should succeed")) {
    return false;
  }
  if (!Expect(cudaStreamSynchronize(stream2) == cudaSuccess &&
                  cudaEventSynchronize(ev_end) == cudaSuccess,
              "synchronizing stream2 and ev_end should succeed")) {
    return false;
  }
  if (!Expect(cudaStreamQuery(stream1) == cudaSuccess && cudaEventQuery(ev_end) == cudaSuccess,
              "stream1 and ev_end should report cudaSuccess after completion")) {
    return false;
  }

  for (std::size_t idx = 0; idx < kElements; ++idx) {
    if (!Expect(host_out[idx] == 0x11111111, "async memset/memcpy output mismatch")) {
      return false;
    }
  }

  float elapsed_ms = -1.0f;
  if (!Expect(
          cudaEventElapsedTime(&elapsed_ms, ev_start, ev_end) == cudaSuccess && elapsed_ms >= 0.0f,
          "cudaEventElapsedTime should return non-negative duration")) {
    return false;
  }

  if (!Expect(cudaEventRecord(ev_notiming, stream1) == cudaSuccess &&
                  cudaEventSynchronize(ev_notiming) == cudaSuccess,
              "recording and syncing disable-timing event should succeed")) {
    return false;
  }
  if (!Expect(cudaEventElapsedTime(&elapsed_ms, ev_start, ev_notiming) ==
                  cudaErrorInvalidResourceHandle,
              "cudaEventElapsedTime with cudaEventDisableTiming should fail")) {
    return false;
  }
  (void)cudaGetLastError();

  // Stale handle detection after destroy
  if (!Expect(cudaEventDestroy(ev_start) == cudaSuccess &&
                  cudaEventDestroy(ev_end) == cudaSuccess &&
                  cudaEventDestroy(ev_notiming) == cudaSuccess,
              "cudaEventDestroy should succeed")) {
    return false;
  }
  if (!Expect(cudaEventDestroy(ev_start) == cudaErrorInvalidResourceHandle,
              "double cudaEventDestroy must fail with cudaErrorInvalidResourceHandle")) {
    return false;
  }
  (void)cudaGetLastError();

  if (!Expect(
          cudaStreamDestroy(stream1) == cudaSuccess && cudaStreamDestroy(stream2) == cudaSuccess,
          "cudaStreamDestroy should succeed")) {
    return false;
  }
  if (!Expect(cudaStreamDestroy(stream1) == cudaErrorInvalidResourceHandle,
              "double cudaStreamDestroy must fail with cudaErrorInvalidResourceHandle")) {
    return false;
  }
  (void)cudaGetLastError();

  if (!Expect(cudaFree(dev_src) == cudaSuccess && cudaFree(dev_dst) == cudaSuccess &&
                  cudaDeviceReset() == cudaSuccess,
              "cleanup after stream/event test should succeed")) {
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

  if (!TestDevicePropertiesAndPointerAttributes()) {
    return 1;
  }

  if (!TestMemoryAndSync()) {
    return 1;
  }

  if (!TestPitchAnd2DMemory()) {
    return 1;
  }

  if (!TestStreamsAndEvents()) {
    return 1;
  }

  if (!TestKernelLaunch()) {
    return 1;
  }

  std::cout << "cuda_runtime_test passed\n";
  return 0;
}

/**
 * @file atomics_test.cpp
 * @brief Tests integral atomics, float CAS atomics, and shared memory atomics on Apple Silicon GPU.
 */

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "cuda_runtime.h"
#include "muda/compiler/driver.hpp"
#include "muda/compiler/toolchain.hpp"
#include "muda/compiler/translator.hpp"
#include "test_utils.hpp"

namespace {

bool Expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    return false;
  }
  return true;
}

constexpr const char* kAtomicsCudaSource = R"CUDA(
// 1. Parallel integer histogram and min/max
__global__ void test_int_atomics(const int* in_vals, int* out_hist, int* out_min, int* out_max, int n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    const int val = in_vals[idx];
    atomicAdd(&out_hist[val % 16], 1);
    atomicMin(out_min, val);
    atomicMax(out_max, val);
  }
}

// 2. Parallel floating-point CAS atomic add
__global__ void test_float_atomics(const float* in_vals, float* out_sum, int n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    atomicAdd(out_sum, in_vals[idx]);
  }
}

// 3. atomicCAS and atomicInc / atomicDec
__global__ void test_cas_and_inc(int* flag_cell, unsigned int* inc_cell, unsigned int* dec_cell, int* out_winner, int n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    // Exactly one thread succeeds in changing flag_cell from 0 to idx + 1
    const int old = atomicCAS(flag_cell, 0, idx + 1);
    if (old == 0) {
      out_winner[0] = idx + 1;
    }
    atomicInc(inc_cell, 15u);
    atomicDec(dec_cell, 15u);
  }
}

// 4. Threadgroup shared memory atomics
__global__ void test_shared_atomics(const int* in_vals, int* out_global, int n) {
  __shared__ int s_bins[16];
  const int tid = threadIdx.x;
  if (tid < 16) {
    s_bins[tid] = 0;
  }
  __syncthreads();

  const int idx = blockIdx.x * blockDim.x + tid;
  if (idx < n) {
    atomicAdd(&s_bins[in_vals[idx] % 16], 1);
  }
  __syncthreads();

  if (tid < 16) {
    atomicAdd(&out_global[tid], s_bins[tid]);
  }
}
)CUDA";

void HostStubIntAtomics() {}
void HostStubFloatAtomics() {}
void HostStubCasInc() {}
void HostStubSharedAtomics() {}

bool RegisterKernel(const void* host_func, const std::filesystem::path& metallib,
                    const muda::compiler::TranslatedKernel& kernel) {
  std::vector<mudaKernelParamDesc> descs;
  descs.reserve(kernel.parameters.size());
  for (const auto& param : kernel.parameters) {
    descs.push_back({
        .kind = param.is_pointer() ? mudaKernelParamBuffer : mudaKernelParamBytes,
        .size = param.byte_size(),
    });
  }
  return mudaRegisterKernelFromFile(host_func, metallib.c_str(), kernel.name.c_str(), descs.data(),
                                    descs.size()) == cudaSuccess;
}

}  // namespace

int main() {
  if (!muda::compiler::MetalToolchainAvailable()) {
    std::cout << "skip: Metal toolchain unavailable\n";
    return 77;
  }

  const muda::testing::ScopedTempDir temp_dir("muda_atomics_test");
  if (!temp_dir.valid()) {
    std::cerr << "cannot create temp directory\n";
    return 1;
  }

  const std::filesystem::path cu_path = temp_dir.path() / "atomics.cu";
  const std::filesystem::path metallib_path = temp_dir.path() / "atomics.metallib";

  {
    std::ofstream cu_file(cu_path, std::ios::binary);
    cu_file.write(kAtomicsCudaSource, static_cast<std::streamsize>(std::string_view(kAtomicsCudaSource).size()));
  }

  const muda::compiler::TranslationResult tr = muda::compiler::TranslateCudaToMsl(kAtomicsCudaSource);
  if (!Expect(tr.ok(), "atomics translation should succeed") ||
      !Expect(tr.kernels.size() == 4u, "expected 4 translated kernels")) {
    return 1;
  }

  muda::compiler::CompileOptions options;
  options.input = cu_path;
  options.output = metallib_path;
  options.output_kind = muda::compiler::OutputKind::kMetallib;

  const muda::compiler::CompileResult compiled = muda::compiler::Compile(options);
  if (!Expect(compiled.status == muda::compiler::Status::kOk, "atomics compilation should succeed")) {
    return 1;
  }

  if (!Expect(RegisterKernel(reinterpret_cast<const void*>(&HostStubIntAtomics), metallib_path, tr.kernels[0]),
              "register int_atomics kernel failed") ||
      !Expect(RegisterKernel(reinterpret_cast<const void*>(&HostStubFloatAtomics), metallib_path, tr.kernels[1]),
              "register float_atomics kernel failed") ||
      !Expect(RegisterKernel(reinterpret_cast<const void*>(&HostStubCasInc), metallib_path, tr.kernels[2]),
              "register cas_inc kernel failed") ||
      !Expect(RegisterKernel(reinterpret_cast<const void*>(&HostStubSharedAtomics), metallib_path, tr.kernels[3]),
              "register shared_atomics kernel failed")) {
    return 1;
  }

  // -------------------------------------------------------------------------
  // Test 1: Integer Atomics (Histogram & Min/Max)
  // -------------------------------------------------------------------------
  {
    constexpr std::size_t kN = 1024;
    std::vector<int> h_in(kN);
    for (std::size_t i = 0; i < kN; ++i) {
      h_in[i] = static_cast<int>(i);
    }

    int *d_in = nullptr, *d_hist = nullptr, *d_min = nullptr, *d_max = nullptr;
    (void)cudaMalloc(reinterpret_cast<void**>(&d_in), kN * sizeof(int));
    (void)cudaMalloc(reinterpret_cast<void**>(&d_hist), 16 * sizeof(int));
    (void)cudaMalloc(reinterpret_cast<void**>(&d_min), sizeof(int));
    (void)cudaMalloc(reinterpret_cast<void**>(&d_max), sizeof(int));

    (void)cudaMemset(d_hist, 0, 16 * sizeof(int));
    int init_min = 999999;
    int init_max = -999999;
    (void)cudaMemcpy(d_min, &init_min, sizeof(int), cudaMemcpyHostToDevice);
    (void)cudaMemcpy(d_max, &init_max, sizeof(int), cudaMemcpyHostToDevice);
    (void)cudaMemcpy(d_in, h_in.data(), kN * sizeof(int), cudaMemcpyHostToDevice);

    int n_arg = static_cast<int>(kN);
    void* args[] = {&d_in, &d_hist, &d_min, &d_max, &n_arg};
    const auto err = cudaLaunchKernel(reinterpret_cast<const void*>(&HostStubIntAtomics), dim3(4),
                                      dim3(256), args, 0, nullptr);
    if (!Expect(err == cudaSuccess, "launch int_atomics kernel failed")) {
      return 1;
    }
    (void)cudaDeviceSynchronize();

    std::vector<int> h_hist(16, 0);
    int h_min = 0, h_max = 0;
    (void)cudaMemcpy(h_hist.data(), d_hist, 16 * sizeof(int), cudaMemcpyDeviceToHost);
    (void)cudaMemcpy(&h_min, d_min, sizeof(int), cudaMemcpyDeviceToHost);
    (void)cudaMemcpy(&h_max, d_max, sizeof(int), cudaMemcpyDeviceToHost);

    // 1024 elements across 16 bins -> 64 per bin
    for (std::size_t b = 0; b < 16; ++b) {
      if (!Expect(h_hist[b] == 64, "histogram bin count mismatch (expected 64)")) {
        std::cerr << "bin[" << b << "] got " << h_hist[b] << "\n";
        return 1;
      }
    }

    if (!Expect(h_min == 0, "atomicMin mismatch (expected 0)")) {
      std::cerr << "got min: " << h_min << "\n";
      return 1;
    }
    if (!Expect(h_max == 1023, "atomicMax mismatch (expected 1023)")) {
      std::cerr << "got max: " << h_max << "\n";
      return 1;
    }

    (void)cudaFree(d_in);
    (void)cudaFree(d_hist);
    (void)cudaFree(d_min);
    (void)cudaFree(d_max);
  }

  // -------------------------------------------------------------------------
  // Test 2: Floating-point CAS atomic add
  // -------------------------------------------------------------------------
  {
    constexpr std::size_t kN = 256;
    std::vector<float> h_in(kN, 1.25f);

    float *d_in = nullptr, *d_sum = nullptr;
    (void)cudaMalloc(reinterpret_cast<void**>(&d_in), kN * sizeof(float));
    (void)cudaMalloc(reinterpret_cast<void**>(&d_sum), sizeof(float));

    (void)cudaMemset(d_sum, 0, sizeof(float));
    (void)cudaMemcpy(d_in, h_in.data(), kN * sizeof(float), cudaMemcpyHostToDevice);

    int n_arg = static_cast<int>(kN);
    void* args[] = {&d_in, &d_sum, &n_arg};
    const auto err = cudaLaunchKernel(reinterpret_cast<const void*>(&HostStubFloatAtomics), dim3(1),
                                      dim3(static_cast<unsigned int>(kN)), args, 0, nullptr);
    if (!Expect(err == cudaSuccess, "launch float_atomics kernel failed")) {
      return 1;
    }
    (void)cudaDeviceSynchronize();

    float h_sum = 0.0f;
    (void)cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);

    // 256 * 1.25f = 320.0f
    if (!Expect(std::fabs(h_sum - 320.0f) < 1e-3f, "float atomicAdd sum mismatch (expected 320.0f)")) {
      std::cerr << "got float sum: " << h_sum << "\n";
      return 1;
    }

    (void)cudaFree(d_in);
    (void)cudaFree(d_sum);
  }

  // -------------------------------------------------------------------------
  // Test 3: atomicCAS & atomicInc / atomicDec
  // -------------------------------------------------------------------------
  {
    constexpr std::size_t kN = 128;
    int *d_flag = nullptr, *d_winner = nullptr;
    unsigned int *d_inc = nullptr, *d_dec = nullptr;

    (void)cudaMalloc(reinterpret_cast<void**>(&d_flag), sizeof(int));
    (void)cudaMalloc(reinterpret_cast<void**>(&d_inc), sizeof(unsigned int));
    (void)cudaMalloc(reinterpret_cast<void**>(&d_dec), sizeof(unsigned int));
    (void)cudaMalloc(reinterpret_cast<void**>(&d_winner), sizeof(int));

    (void)cudaMemset(d_flag, 0, sizeof(int));
    (void)cudaMemset(d_inc, 0, sizeof(unsigned int));
    unsigned int init_dec = 0u;
    (void)cudaMemcpy(d_dec, &init_dec, sizeof(unsigned int), cudaMemcpyHostToDevice);
    (void)cudaMemset(d_winner, 0, sizeof(int));

    int n_arg = static_cast<int>(kN);
    void* args[] = {&d_flag, &d_inc, &d_dec, &d_winner, &n_arg};
    const auto err = cudaLaunchKernel(reinterpret_cast<const void*>(&HostStubCasInc), dim3(1),
                                      dim3(static_cast<unsigned int>(kN)), args, 0, nullptr);
    if (!Expect(err == cudaSuccess, "launch cas_inc kernel failed")) {
      return 1;
    }
    (void)cudaDeviceSynchronize();

    int h_flag = 0, h_winner = 0;
    unsigned int h_inc = 0, h_dec = 0;
    (void)cudaMemcpy(&h_flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost);
    (void)cudaMemcpy(&h_winner, d_winner, sizeof(int), cudaMemcpyDeviceToHost);
    (void)cudaMemcpy(&h_inc, d_inc, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    (void)cudaMemcpy(&h_dec, d_dec, sizeof(unsigned int), cudaMemcpyDeviceToHost);

    if (!Expect(h_winner >= 1 && h_winner <= static_cast<int>(kN), "valid winner ID expected")) {
      std::cerr << "got winner: " << h_winner << "\n";
      return 1;
    }
    if (!Expect(h_flag == h_winner, "flag must match winner ID")) {
      std::cerr << "got flag: " << h_flag << " winner: " << h_winner << "\n";
      return 1;
    }
    // atomicInc modulo 15: 128 % 16 = 0
    if (!Expect(h_inc == 0u, "atomicInc 128 % 16 expected 0")) {
      std::cerr << "got inc: " << h_inc << "\n";
      return 1;
    }

    (void)cudaFree(d_flag);
    (void)cudaFree(d_inc);
    (void)cudaFree(d_dec);
    (void)cudaFree(d_winner);
  }

  // -------------------------------------------------------------------------
  // Test 4: Shared Memory Atomics
  // -------------------------------------------------------------------------
  {
    constexpr std::size_t kN = 512;
    std::vector<int> h_in(kN);
    for (std::size_t i = 0; i < kN; ++i) {
      h_in[i] = static_cast<int>(i);
    }

    int *d_in = nullptr, *d_global = nullptr;
    (void)cudaMalloc(reinterpret_cast<void**>(&d_in), kN * sizeof(int));
    (void)cudaMalloc(reinterpret_cast<void**>(&d_global), 16 * sizeof(int));
    (void)cudaMemset(d_global, 0, 16 * sizeof(int));
    (void)cudaMemcpy(d_in, h_in.data(), kN * sizeof(int), cudaMemcpyHostToDevice);

    int n_arg = static_cast<int>(kN);
    void* args[] = {&d_in, &d_global, &n_arg};
    // 2 blocks of 256 threads = 512 total threads
    const auto err = cudaLaunchKernel(reinterpret_cast<const void*>(&HostStubSharedAtomics), dim3(2),
                                      dim3(256), args, 0, nullptr);
    if (!Expect(err == cudaSuccess, "launch shared_atomics kernel failed")) {
      return 1;
    }
    (void)cudaDeviceSynchronize();

    std::vector<int> h_global(16, 0);
    (void)cudaMemcpy(h_global.data(), d_global, 16 * sizeof(int), cudaMemcpyDeviceToHost);

    // 512 elements across 16 bins -> 32 per bin
    for (std::size_t b = 0; b < 16; ++b) {
      if (!Expect(h_global[b] == 32, "shared atomic bin count mismatch (expected 32)")) {
        std::cerr << "shared bin[" << b << "] got " << h_global[b] << "\n";
        return 1;
      }
    }

    (void)cudaFree(d_in);
    (void)cudaFree(d_global);
  }

  std::cout << "atomics_test: ALL PASSED\n";
  return 0;
}

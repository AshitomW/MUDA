/**
 * @file warp_intrinsics_test.cpp
 * @brief Tests CUDA warp shuffle, vote, ballot, and bit-manipulation intrinsics on Apple Silicon GPU.
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

constexpr const char* kWarpCudaSource = R"CUDA(
__global__ void test_warp_shuffle(const float* in, float* out_reduce, float* out_scan,
                                  float* out_bcast, float* out_xor) {
  const int tid = threadIdx.x;
  float val = in[tid];

  // 1. Warp sum reduction using __shfl_down_sync
  float sum = val;
  for (int offset = 16; offset > 0; offset /= 2) {
    sum += __shfl_down_sync(0xFFFFFFFFu, sum, offset);
  }
  if (tid == 0) {
    out_reduce[0] = sum;
  }

  // 2. Warp inclusive scan using __shfl_up_sync
  float scan = val;
  for (int offset = 1; offset < 32; offset *= 2) {
    float n = __shfl_up_sync(0xFFFFFFFFu, scan, offset);
    if (tid >= offset) {
      scan += n;
    }
  }
  out_scan[tid] = scan;

  // 3. Broadcast from lane 7 using __shfl_sync
  out_bcast[tid] = __shfl_sync(0xFFFFFFFFu, val, 7);

  // 4. Butterfly neighbor exchange using __shfl_xor_sync
  out_xor[tid] = __shfl_xor_sync(0xFFFFFFFFu, val, 1);
}

__global__ void test_warp_vote(const int* in_flags, unsigned int* out_ballot, int* out_any,
                               int* out_all, unsigned int* out_active) {
  const int tid = threadIdx.x;
  const int pred = in_flags[tid];

  const unsigned int ballot_val = __ballot_sync(0xFFFFFFFFu, pred);
  const int any_val = __any_sync(0xFFFFFFFFu, pred);
  const int all_val = __all_sync(0xFFFFFFFFu, pred);
  const unsigned int active_val = __activemask();

  if (tid == 0) {
    out_ballot[0] = ballot_val;
    out_any[0] = any_val;
    out_all[0] = all_val;
    out_active[0] = active_val;
  }
}

__global__ void test_bit_ops(const unsigned int* in_vals, int* out_popc, int* out_clz,
                             int* out_ffs, unsigned int* out_brev, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    const unsigned int val = in_vals[i];
    out_popc[i] = __popc(val);
    out_clz[i] = __clz(val);
    out_ffs[i] = __ffs(static_cast<int>(val));
    out_brev[i] = __brev(val);
  }
}
)CUDA";

void HostStubShuffle() {}
void HostStubVote() {}
void HostStubBitOps() {}

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

  const muda::testing::ScopedTempDir temp_dir("muda_warp_test");
  if (!temp_dir.valid()) {
    std::cerr << "cannot create temp directory\n";
    return 1;
  }

  const std::filesystem::path cu_path = temp_dir.path() / "warp.cu";
  const std::filesystem::path metallib_path = temp_dir.path() / "warp.metallib";

  {
    std::ofstream cu_file(cu_path, std::ios::binary);
    cu_file.write(kWarpCudaSource, static_cast<std::streamsize>(std::string_view(kWarpCudaSource).size()));
  }

  const muda::compiler::TranslationResult tr = muda::compiler::TranslateCudaToMsl(kWarpCudaSource);
  if (!Expect(tr.ok(), "warp translation should succeed") ||
      !Expect(tr.kernels.size() == 3u, "expected 3 translated kernels")) {
    return 1;
  }

  muda::compiler::CompileOptions options;
  options.input = cu_path;
  options.output = metallib_path;
  options.output_kind = muda::compiler::OutputKind::kMetallib;

  const muda::compiler::CompileResult compiled = muda::compiler::Compile(options);
  if (!Expect(compiled.status == muda::compiler::Status::kOk, "warp compilation should succeed")) {
    return 1;
  }

  if (!Expect(RegisterKernel(reinterpret_cast<const void*>(&HostStubShuffle), metallib_path, tr.kernels[0]),
              "register shuffle kernel failed") ||
      !Expect(RegisterKernel(reinterpret_cast<const void*>(&HostStubVote), metallib_path, tr.kernels[1]),
              "register vote kernel failed") ||
      !Expect(RegisterKernel(reinterpret_cast<const void*>(&HostStubBitOps), metallib_path, tr.kernels[2]),
              "register bit_ops kernel failed")) {
    return 1;
  }

  // -------------------------------------------------------------------------
  // Test 1: Warp Shuffle Operations
  // -------------------------------------------------------------------------
  {
    constexpr std::size_t kWarpSize = 32;
    std::vector<float> h_in(kWarpSize);
    for (std::size_t i = 0; i < kWarpSize; ++i) {
      h_in[i] = static_cast<float>(i + 1);  // 1.0f .. 32.0f
    }

    float *d_in = nullptr, *d_reduce = nullptr, *d_scan = nullptr, *d_bcast = nullptr, *d_xor = nullptr;
    (void)cudaMalloc(reinterpret_cast<void**>(&d_in), kWarpSize * sizeof(float));
    (void)cudaMalloc(reinterpret_cast<void**>(&d_reduce), sizeof(float));
    (void)cudaMalloc(reinterpret_cast<void**>(&d_scan), kWarpSize * sizeof(float));
    (void)cudaMalloc(reinterpret_cast<void**>(&d_bcast), kWarpSize * sizeof(float));
    (void)cudaMalloc(reinterpret_cast<void**>(&d_xor), kWarpSize * sizeof(float));

    (void)cudaMemcpy(d_in, h_in.data(), kWarpSize * sizeof(float), cudaMemcpyHostToDevice);

    void* args[] = {&d_in, &d_reduce, &d_scan, &d_bcast, &d_xor};
    const auto err = cudaLaunchKernel(reinterpret_cast<const void*>(&HostStubShuffle), dim3(1),
                                      dim3(static_cast<unsigned int>(kWarpSize)), args, 0, nullptr);
    if (!Expect(err == cudaSuccess, "launch shuffle kernel failed")) {
      return 1;
    }
    (void)cudaDeviceSynchronize();

    float h_reduce = 0.0f;
    std::vector<float> h_scan(kWarpSize, 0.0f);
    std::vector<float> h_bcast(kWarpSize, 0.0f);
    std::vector<float> h_xor(kWarpSize, 0.0f);

    (void)cudaMemcpy(&h_reduce, d_reduce, sizeof(float), cudaMemcpyDeviceToHost);
    (void)cudaMemcpy(h_scan.data(), d_scan, kWarpSize * sizeof(float), cudaMemcpyDeviceToHost);
    (void)cudaMemcpy(h_bcast.data(), d_bcast, kWarpSize * sizeof(float), cudaMemcpyDeviceToHost);
    (void)cudaMemcpy(h_xor.data(), d_xor, kWarpSize * sizeof(float), cudaMemcpyDeviceToHost);

    // Sum of 1 .. 32 = 32 * 33 / 2 = 528.0f
    if (!Expect(std::fabs(h_reduce - 528.0f) < 1e-4f, "warp reduce sum mismatch (expected 528.0f)")) {
      std::cerr << "got reduce sum: " << h_reduce << "\n";
      return 1;
    }

    // Inclusive scan: prefix sum of 1..i+1
    float running_sum = 0.0f;
    for (std::size_t i = 0; i < kWarpSize; ++i) {
      running_sum += static_cast<float>(i + 1);
      if (!Expect(std::fabs(h_scan[i] - running_sum) < 1e-4f, "warp scan mismatch")) {
        std::cerr << "scan[" << i << "] got " << h_scan[i] << " expected " << running_sum << "\n";
        return 1;
      }
    }

    // Broadcast from lane 7 should be 8.0f for all lanes
    for (std::size_t i = 0; i < kWarpSize; ++i) {
      if (!Expect(std::fabs(h_bcast[i] - 8.0f) < 1e-4f, "warp broadcast mismatch")) {
        std::cerr << "bcast[" << i << "] got " << h_bcast[i] << " expected 8.0f\n";
        return 1;
      }
    }

    // Butterfly exchange: lane i exchanged with i ^ 1
    for (std::size_t i = 0; i < kWarpSize; ++i) {
      const float expected = static_cast<float>((i ^ 1u) + 1u);
      if (!Expect(std::fabs(h_xor[i] - expected) < 1e-4f, "warp xor exchange mismatch")) {
        std::cerr << "xor[" << i << "] got " << h_xor[i] << " expected " << expected << "\n";
        return 1;
      }
    }

    (void)cudaFree(d_in);
    (void)cudaFree(d_reduce);
    (void)cudaFree(d_scan);
    (void)cudaFree(d_bcast);
    (void)cudaFree(d_xor);
  }

  // -------------------------------------------------------------------------
  // Test 2: Warp Voting & Ballot
  // -------------------------------------------------------------------------
  {
    constexpr std::size_t kWarpSize = 32;
    std::vector<int> h_flags(kWarpSize);
    for (std::size_t i = 0; i < kWarpSize; ++i) {
      h_flags[i] = (i % 2 == 0) ? 1 : 0;  // Even lanes active: 0, 2, 4, ...
    }

    int* d_flags = nullptr;
    unsigned int* d_ballot = nullptr;
    int* d_any = nullptr;
    int* d_all = nullptr;
    unsigned int* d_active = nullptr;

    (void)cudaMalloc(reinterpret_cast<void**>(&d_flags), kWarpSize * sizeof(int));
    (void)cudaMalloc(reinterpret_cast<void**>(&d_ballot), sizeof(unsigned int));
    (void)cudaMalloc(reinterpret_cast<void**>(&d_any), sizeof(int));
    (void)cudaMalloc(reinterpret_cast<void**>(&d_all), sizeof(int));
    (void)cudaMalloc(reinterpret_cast<void**>(&d_active), sizeof(unsigned int));

    (void)cudaMemcpy(d_flags, h_flags.data(), kWarpSize * sizeof(int), cudaMemcpyHostToDevice);

    void* args[] = {&d_flags, &d_ballot, &d_any, &d_all, &d_active};
    const auto err = cudaLaunchKernel(reinterpret_cast<const void*>(&HostStubVote), dim3(1),
                                      dim3(static_cast<unsigned int>(kWarpSize)), args, 0, nullptr);
    if (!Expect(err == cudaSuccess, "launch vote kernel failed")) {
      return 1;
    }
    (void)cudaDeviceSynchronize();

    unsigned int h_ballot = 0;
    int h_any = 0;
    int h_all = 0;
    unsigned int h_active = 0;

    (void)cudaMemcpy(&h_ballot, d_ballot, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    (void)cudaMemcpy(&h_any, d_any, sizeof(int), cudaMemcpyDeviceToHost);
    (void)cudaMemcpy(&h_all, d_all, sizeof(int), cudaMemcpyDeviceToHost);
    (void)cudaMemcpy(&h_active, d_active, sizeof(unsigned int), cudaMemcpyDeviceToHost);

    // Even lanes 0, 2, 4, ... -> bits 0, 2, 4, ... set: 0x55555555
    if (!Expect(h_ballot == 0x55555555u, "ballot expected 0x55555555")) {
      std::cerr << "got ballot: 0x" << std::hex << h_ballot << std::dec << "\n";
      return 1;
    }

    if (!Expect(h_any == 1, "any_sync expected 1")) {
      return 1;
    }

    if (!Expect(h_all == 0, "all_sync expected 0 (not all lanes active)")) {
      return 1;
    }

    if (!Expect(h_active == 0xFFFFFFFFu, "activemask expected 0xFFFFFFFF")) {
      std::cerr << "got activemask: 0x" << std::hex << h_active << std::dec << "\n";
      return 1;
    }

    (void)cudaFree(d_flags);
    (void)cudaFree(d_ballot);
    (void)cudaFree(d_any);
    (void)cudaFree(d_all);
    (void)cudaFree(d_active);
  }

  // -------------------------------------------------------------------------
  // Test 3: Bit-Manipulation Intrinsics (__popc, __clz, __ffs, __brev)
  // -------------------------------------------------------------------------
  {
    const std::vector<unsigned int> h_vals = {
        0x00000000u,  // popc=0,  clz=32, ffs=0,  brev=0x00000000
        0x00000001u,  // popc=1,  clz=31, ffs=1,  brev=0x80000000
        0x00000008u,  // popc=1,  clz=28, ffs=4,  brev=0x10000000
        0x55555555u,  // popc=16, clz=1,  ffs=1,  brev=0xAAAAAAAA
        0x80000000u,  // popc=1,  clz=0,  ffs=32, brev=0x00000001
        0xFFFFFFFFu,  // popc=32, clz=0,  ffs=1,  brev=0xFFFFFFFF
    };
    const std::size_t count = h_vals.size();
    int count_arg = static_cast<int>(count);

    unsigned int* d_vals = nullptr;
    int* d_popc = nullptr;
    int* d_clz = nullptr;
    int* d_ffs = nullptr;
    unsigned int* d_brev = nullptr;

    (void)cudaMalloc(reinterpret_cast<void**>(&d_vals), count * sizeof(unsigned int));
    (void)cudaMalloc(reinterpret_cast<void**>(&d_popc), count * sizeof(int));
    (void)cudaMalloc(reinterpret_cast<void**>(&d_clz), count * sizeof(int));
    (void)cudaMalloc(reinterpret_cast<void**>(&d_ffs), count * sizeof(int));
    (void)cudaMalloc(reinterpret_cast<void**>(&d_brev), count * sizeof(unsigned int));

    (void)cudaMemcpy(d_vals, h_vals.data(), count * sizeof(unsigned int), cudaMemcpyHostToDevice);

    void* args[] = {&d_vals, &d_popc, &d_clz, &d_ffs, &d_brev, &count_arg};
    const auto err = cudaLaunchKernel(reinterpret_cast<const void*>(&HostStubBitOps), dim3(1),
                                      dim3(static_cast<unsigned int>(count)), args, 0, nullptr);
    if (!Expect(err == cudaSuccess, "launch bit_ops kernel failed")) {
      return 1;
    }
    (void)cudaDeviceSynchronize();

    std::vector<int> h_popc(count);
    std::vector<int> h_clz(count);
    std::vector<int> h_ffs(count);
    std::vector<unsigned int> h_brev(count);

    (void)cudaMemcpy(h_popc.data(), d_popc, count * sizeof(int), cudaMemcpyDeviceToHost);
    (void)cudaMemcpy(h_clz.data(), d_clz, count * sizeof(int), cudaMemcpyDeviceToHost);
    (void)cudaMemcpy(h_ffs.data(), d_ffs, count * sizeof(int), cudaMemcpyDeviceToHost);
    (void)cudaMemcpy(h_brev.data(), d_brev, count * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    const std::vector<int> exp_popc = {0, 1, 1, 16, 1, 32};
    const std::vector<int> exp_clz = {32, 31, 28, 1, 0, 0};
    const std::vector<int> exp_ffs = {0, 1, 4, 1, 32, 1};
    const std::vector<unsigned int> exp_brev = {
        0x00000000u, 0x80000000u, 0x10000000u, 0xAAAAAAAAu, 0x00000001u, 0xFFFFFFFFu};

    for (std::size_t i = 0; i < count; ++i) {
      if (!Expect(h_popc[i] == exp_popc[i], "popc mismatch") ||
          !Expect(h_clz[i] == exp_clz[i], "clz mismatch") ||
          !Expect(h_ffs[i] == exp_ffs[i], "ffs mismatch") ||
          !Expect(h_brev[i] == exp_brev[i], "brev mismatch")) {
        std::cerr << "bit ops failed at index " << i << ": val=0x" << std::hex << h_vals[i]
                  << " popc=" << h_popc[i] << " clz=" << h_clz[i] << " ffs=" << h_ffs[i]
                  << " brev=0x" << h_brev[i] << std::dec << "\n";
        return 1;
      }
    }

    (void)cudaFree(d_vals);
    (void)cudaFree(d_popc);
    (void)cudaFree(d_clz);
    (void)cudaFree(d_ffs);
    (void)cudaFree(d_brev);
  }

  std::cout << "warp_intrinsics_test: ALL PASSED\n";
  return 0;
}

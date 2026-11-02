/**
 * @file corpus_test.cpp
 * @brief 30-kernel numerical parity suite against CPU reference implementations.
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
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

// 30 distinct host stubs for registering translated kernels
void Stub1() {}  void Stub2() {}  void Stub3() {}  void Stub4() {}  void Stub5() {}
void Stub6() {}  void Stub7() {}  void Stub8() {}  void Stub9() {}  void Stub10() {}
void Stub11() {} void Stub12() {} void Stub13() {} void Stub14() {} void Stub15() {}
void Stub16() {} void Stub17() {} void Stub18() {} void Stub19() {} void Stub20() {}
void Stub21() {} void Stub22() {} void Stub23() {} void Stub24() {} void Stub25() {}
void Stub26() {} void Stub27() {} void Stub28() {} void Stub29() {} void Stub30() {}

bool RegisterTranslatedKernel(const void* stub, const std::filesystem::path& metallib_path,
                              const muda::compiler::TranslatedKernel& kernel) {
  std::vector<mudaKernelParamDesc> descs;
  descs.reserve(kernel.parameters.size());
  for (const muda::compiler::KernelParameter& param : kernel.parameters) {
    descs.push_back(mudaKernelParamDesc{
        .kind = param.is_pointer() ? mudaKernelParamBuffer : mudaKernelParamBytes,
        .size = param.byte_size(),
    });
  }
  return mudaRegisterKernelFromFile(stub, metallib_path.string().c_str(), kernel.name.c_str(),
                                    descs.data(), descs.size()) == cudaSuccess;
}

[[nodiscard]] std::uint32_t ReverseBits32(std::uint32_t x) noexcept {
  std::uint32_t rev = 0;
  for (std::size_t i = 0; i < 32; ++i) {
    if ((x & (1u << i)) != 0u) {
      rev |= (1u << (31u - i));
    }
  }
  return rev;
}

}  // namespace

int main() {
  if (!muda::compiler::MetalToolchainAvailable()) {
    std::cout << "skip: Metal toolchain unavailable\n";
    return 77;
  }

  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    std::cout << "skip: no Metal device available\n";
    return 77;
  }

  const std::filesystem::path cu_source_path =
      std::filesystem::path(MUDA_SOURCE_DIR) / "tests" / "corpus_kernels.cu";
  std::ifstream cu_file(cu_source_path);
  if (!Expect(cu_file.is_open(), "failed to open corpus_kernels.cu")) {
    return 1;
  }
  std::stringstream buffer;
  buffer << cu_file.rdbuf();
  const std::string cu_source = buffer.str();

  const muda::testing::ScopedTempDir temp_dir("muda_corpus30");
  if (!Expect(temp_dir.valid(), "failed to create temp directory")) {
    return 1;
  }

  const std::filesystem::path metallib_path = temp_dir.path() / "corpus30.metallib";

  const muda::compiler::TranslationResult tr =
      muda::compiler::TranslateCudaToMsl(cu_source);
  if (!Expect(tr.ok() && tr.kernels.size() == 30, "corpus should translate all 30 kernels")) {
    std::cerr << "translation error [" << tr.diagnostic_id() << "]: " << tr.error() << "\n";
    return 1;
  }

  muda::compiler::CompileOptions options;
  options.input = cu_source_path;
  options.output = metallib_path;
  options.output_kind = muda::compiler::OutputKind::kMetallib;
  options.include_dirs.push_back(std::filesystem::path(MUDA_SOURCE_DIR) / "runtime" / "include");

  const muda::compiler::CompileResult compiled = muda::compiler::Compile(options);
  if (!Expect(compiled.status == muda::compiler::Status::kOk,
              "corpus_kernels.cu should compile to corpus30.metallib")) {
    std::cerr << "compile error [" << compiled.diagnostic_id() << "]: " << compiled.message()
              << "\n";
    return 1;
  }

  const void* stubs[30] = {
      reinterpret_cast<const void*>(&Stub1),  reinterpret_cast<const void*>(&Stub2),
      reinterpret_cast<const void*>(&Stub3),  reinterpret_cast<const void*>(&Stub4),
      reinterpret_cast<const void*>(&Stub5),  reinterpret_cast<const void*>(&Stub6),
      reinterpret_cast<const void*>(&Stub7),  reinterpret_cast<const void*>(&Stub8),
      reinterpret_cast<const void*>(&Stub9),  reinterpret_cast<const void*>(&Stub10),
      reinterpret_cast<const void*>(&Stub11), reinterpret_cast<const void*>(&Stub12),
      reinterpret_cast<const void*>(&Stub13), reinterpret_cast<const void*>(&Stub14),
      reinterpret_cast<const void*>(&Stub15), reinterpret_cast<const void*>(&Stub16),
      reinterpret_cast<const void*>(&Stub17), reinterpret_cast<const void*>(&Stub18),
      reinterpret_cast<const void*>(&Stub19), reinterpret_cast<const void*>(&Stub20),
      reinterpret_cast<const void*>(&Stub21), reinterpret_cast<const void*>(&Stub22),
      reinterpret_cast<const void*>(&Stub23), reinterpret_cast<const void*>(&Stub24),
      reinterpret_cast<const void*>(&Stub25), reinterpret_cast<const void*>(&Stub26),
      reinterpret_cast<const void*>(&Stub27), reinterpret_cast<const void*>(&Stub28),
      reinterpret_cast<const void*>(&Stub29), reinterpret_cast<const void*>(&Stub30),
  };

  for (std::size_t k = 0; k < 30; ++k) {
    if (!Expect(RegisterTranslatedKernel(stubs[k], metallib_path, tr.kernels[k]),
                "failed to register corpus kernel")) {
      return 1;
    }
  }

  constexpr int kN = 256;
  constexpr std::size_t kBytes = static_cast<std::size_t>(kN) * sizeof(float);
  float* d_a = nullptr;
  float* d_b = nullptr;
  float* d_c = nullptr;
  if (!Expect(cudaMallocManaged(reinterpret_cast<void**>(&d_a), kBytes) == cudaSuccess &&
                  cudaMallocManaged(reinterpret_cast<void**>(&d_b), kBytes) == cudaSuccess &&
                  cudaMallocManaged(reinterpret_cast<void**>(&d_c), kBytes) == cudaSuccess,
              "cudaMallocManaged for float buffers failed")) {
    return 1;
  }

  for (int i = 0; i < kN; ++i) {
    d_a[i] = static_cast<float>(i) * 0.25f - 10.0f;
    d_b[i] = static_cast<float>(kN - i) * 0.5f;
    d_c[i] = 0.0f;
  }

  // 1. vector_add
  {
    int n = kN;
    void* args[4] = {&d_a, &d_b, &d_c, &n};
    if (!Expect(cudaLaunchKernel(stubs[0], dim3(4), dim3(64), args, 0, nullptr) == cudaSuccess &&
                    cudaDeviceSynchronize() == cudaSuccess,
                "k1_vector_add launch failed")) {
      return 1;
    }
    for (int i = 0; i < kN; ++i) {
      if (!Expect(muda::testing::AreClose(d_c[i], d_a[i] + d_b[i]), "k1_vector_add mismatch")) {
        return 1;
      }
    }
  }

  // 2. saxpy
  {
    float alpha = 2.5f;
    int n = kN;
    void* args[5] = {&alpha, &d_a, &d_b, &d_c, &n};
    if (!Expect(cudaLaunchKernel(stubs[1], dim3(4), dim3(64), args, 0, nullptr) == cudaSuccess &&
                    cudaDeviceSynchronize() == cudaSuccess,
                "k2_saxpy launch failed")) {
      return 1;
    }
    for (int i = 0; i < kN; ++i) {
      if (!Expect(muda::testing::AreClose(d_c[i], alpha * d_a[i] + d_b[i]), "k2_saxpy mismatch")) {
        return 1;
      }
    }
  }

  // 3. scale_clamp
  {
    float scale = 1.5f;
    float bias = -2.0f;
    float lo = -5.0f;
    float hi = 20.0f;
    int n = kN;
    void* args[7] = {&d_a, &d_c, &scale, &bias, &lo, &hi, &n};
    if (!Expect(cudaLaunchKernel(stubs[2], dim3(4), dim3(64), args, 0, nullptr) == cudaSuccess &&
                    cudaDeviceSynchronize() == cudaSuccess,
                "k3_scale_clamp launch failed")) {
      return 1;
    }
    for (int i = 0; i < kN; ++i) {
      const float expected = std::clamp(d_a[i] * scale + bias, lo, hi);
      if (!Expect(muda::testing::AreClose(d_c[i], expected), "k3_scale_clamp mismatch")) {
        return 1;
      }
    }
  }

  // 4. dot_reduce
  {
    int n = kN;
    void* args[4] = {&d_a, &d_b, &d_c, &n};
    if (!Expect(cudaLaunchKernel(stubs[3], dim3(4), dim3(64), args, 0, nullptr) == cudaSuccess &&
                    cudaDeviceSynchronize() == cudaSuccess,
                "k4_dot_reduce launch failed")) {
      return 1;
    }
    float gpu_total = d_c[0] + d_c[1] + d_c[2] + d_c[3];
    float cpu_total = 0.0f;
    for (int i = 0; i < kN; ++i) {
      cpu_total += d_a[i] * d_b[i];
    }
    if (!Expect(muda::testing::AreClose(gpu_total, cpu_total, 1e-2f), "k4_dot_reduce mismatch")) {
      return 1;
    }
  }

  // 5. transpose_naive (16 x 16)
  {
    int w = 16;
    int h = 16;
    void* args[4] = {&d_a, &d_c, &w, &h};
    if (!Expect(
            cudaLaunchKernel(stubs[4], dim3(2, 2), dim3(8, 8), args, 0, nullptr) == cudaSuccess &&
                cudaDeviceSynchronize() == cudaSuccess,
            "k5_transpose_naive launch failed")) {
      return 1;
    }
    for (int r = 0; r < h; ++r) {
      for (int c = 0; c < w; ++c) {
        if (!Expect(muda::testing::AreClose(d_c[c * h + r], d_a[r * w + c]),
                    "k5_transpose_naive mismatch")) {
          return 1;
        }
      }
    }
  }

  // 6. transpose_tiled (16 x 16)
  {
    int w = 16;
    int h = 16;
    (void)cudaMemset(d_c, 0, kBytes);
    void* args[4] = {&d_a, &d_c, &w, &h};
    if (!Expect(
            cudaLaunchKernel(stubs[5], dim3(2, 2), dim3(8, 8), args, 0, nullptr) == cudaSuccess &&
                cudaDeviceSynchronize() == cudaSuccess,
            "k6_transpose_tiled launch failed")) {
      return 1;
    }
    for (int r = 0; r < h; ++r) {
      for (int c = 0; c < w; ++c) {
        if (!Expect(muda::testing::AreClose(d_c[c * h + r], d_a[r * w + c]),
                    "k6_transpose_tiled mismatch")) {
          return 1;
        }
      }
    }
  }

  // 7. gemm_naive (16 x 16)
  std::vector<float> cpu_gemm(kN, 0.0f);
  {
    int m = 16;
    int n = 16;
    int k = 16;
    for (int r = 0; r < m; ++r) {
      for (int c = 0; c < n; ++c) {
        float sum = 0.0f;
        for (int p = 0; p < k; ++p) {
          sum += d_a[r * k + p] * d_b[p * n + c];
        }
        cpu_gemm[static_cast<std::size_t>(r * n + c)] = sum;
      }
    }
    (void)cudaMemset(d_c, 0, kBytes);
    void* args[6] = {&d_a, &d_b, &d_c, &m, &n, &k};
    if (!Expect(
            cudaLaunchKernel(stubs[6], dim3(2, 2), dim3(8, 8), args, 0, nullptr) == cudaSuccess &&
                cudaDeviceSynchronize() == cudaSuccess,
            "k7_gemm_naive launch failed")) {
      return 1;
    }
    for (int i = 0; i < kN; ++i) {
      if (!Expect(muda::testing::AreClose(d_c[i], cpu_gemm[static_cast<std::size_t>(i)], 1e-2f),
                  "k7_gemm_naive mismatch")) {
        return 1;
      }
    }
  }

  // 8. gemm_tiled (16 x 16)
  {
    int n = 16;
    (void)cudaMemset(d_c, 0, kBytes);
    void* args[4] = {&d_a, &d_b, &d_c, &n};
    if (!Expect(
            cudaLaunchKernel(stubs[7], dim3(2, 2), dim3(8, 8), args, 0, nullptr) == cudaSuccess &&
                cudaDeviceSynchronize() == cudaSuccess,
            "k8_gemm_tiled launch failed")) {
      return 1;
    }
    for (int i = 0; i < kN; ++i) {
      if (!Expect(muda::testing::AreClose(d_c[i], cpu_gemm[static_cast<std::size_t>(i)], 1e-2f),
                  "k8_gemm_tiled mismatch")) {
        return 1;
      }
    }
  }

  // 9. stencil_1d (dynamic shared memory)
  {
    int n = kN;
    (void)cudaMemset(d_c, 0, kBytes);
    void* args[3] = {&d_a, &d_c, &n};
    const std::size_t dyn_shared_bytes = (64 + 2) * sizeof(float);
    if (!Expect(cudaLaunchKernel(stubs[8], dim3(4), dim3(64), args, dyn_shared_bytes, nullptr) ==
                        cudaSuccess &&
                    cudaDeviceSynchronize() == cudaSuccess,
                "k9_stencil_1d launch failed")) {
      return 1;
    }
    for (int i = 0; i < kN; ++i) {
      const float left = (i > 0) ? d_a[i - 1] : d_a[0];
      const float center = d_a[i];
      const float right = (i + 1 < kN) ? d_a[i + 1] : d_a[kN - 1];
      const float expected = 0.25f * left + 0.5f * center + 0.25f * right;
      if (!Expect(muda::testing::AreClose(d_c[i], expected), "k9_stencil_1d mismatch")) {
        return 1;
      }
    }
  }

  // 10. row_sum_2d (16 x 16)
  {
    int rows = 16;
    int cols = 16;
    (void)cudaMemset(d_c, 0, kBytes);
    void* args[4] = {&d_a, &d_c, &rows, &cols};
    if (!Expect(cudaLaunchKernel(stubs[9], dim3(1), dim3(16), args, 0, nullptr) == cudaSuccess &&
                    cudaDeviceSynchronize() == cudaSuccess,
                "k10_row_sum launch failed")) {
      return 1;
    }
    for (int r = 0; r < rows; ++r) {
      float expected = 0.0f;
      for (int c = 0; c < cols; ++c) {
        expected += d_a[r * cols + c];
      }
      if (!Expect(muda::testing::AreClose(d_c[r], expected, 1e-3f), "k10_row_sum mismatch")) {
        return 1;
      }
    }
  }

  // 11. warp_shuffle_reduce (4 warps of 32 = 128 items)
  {
    int n = 128;
    (void)cudaMemset(d_c, 0, kBytes);
    void* args[3] = {&d_a, &d_c, &n};
    if (!Expect(cudaLaunchKernel(stubs[10], dim3(4), dim3(32), args, 0, nullptr) == cudaSuccess &&
                    cudaDeviceSynchronize() == cudaSuccess,
                "k11_warp_shuffle_reduce launch failed")) {
      return 1;
    }
    for (int w = 0; w < 4; ++w) {
      float expected = 0.0f;
      for (int i = 0; i < 32; ++i) {
        expected += d_a[w * 32 + i];
      }
      if (!Expect(muda::testing::AreClose(d_c[w], expected, 1e-3f),
                  "k11_warp_shuffle_reduce mismatch")) {
        return 1;
      }
    }
  }

  // 12. warp_prefix_scan (4 warps of 32 = 128 items)
  {
    int n = 128;
    (void)cudaMemset(d_c, 0, kBytes);
    void* args[3] = {&d_a, &d_c, &n};
    if (!Expect(cudaLaunchKernel(stubs[11], dim3(4), dim3(32), args, 0, nullptr) == cudaSuccess &&
                    cudaDeviceSynchronize() == cudaSuccess,
                "k12_warp_prefix_scan launch failed")) {
      return 1;
    }
    for (int w = 0; w < 4; ++w) {
      float running = 0.0f;
      for (int i = 0; i < 32; ++i) {
        running += d_a[w * 32 + i];
        if (!Expect(muda::testing::AreClose(d_c[w * 32 + i], running, 1e-3f),
                    "k12_warp_prefix_scan mismatch")) {
          return 1;
        }
      }
    }
  }

  // 13. warp_broadcast
  {
    int n = 128;
    (void)cudaMemset(d_c, 0, kBytes);
    void* args[3] = {&d_a, &d_c, &n};
    if (!Expect(cudaLaunchKernel(stubs[12], dim3(4), dim3(32), args, 0, nullptr) == cudaSuccess &&
                    cudaDeviceSynchronize() == cudaSuccess,
                "k13_warp_broadcast launch failed")) {
      return 1;
    }
    for (int w = 0; w < 4; ++w) {
      const float expected = d_a[w * 32];
      for (int i = 0; i < 32; ++i) {
        if (!Expect(muda::testing::AreClose(d_c[w * 32 + i], expected),
                    "k13_warp_broadcast mismatch")) {
          return 1;
        }
      }
    }
  }

  // Allocate integer buffers
  int* d_int_in = nullptr;
  int* d_int_out1 = nullptr;
  int* d_int_out2 = nullptr;
  const std::size_t int_bytes = static_cast<std::size_t>(kN) * sizeof(int);
  if (!Expect(cudaMallocManaged(reinterpret_cast<void**>(&d_int_in), int_bytes) == cudaSuccess &&
                  cudaMallocManaged(reinterpret_cast<void**>(&d_int_out1), int_bytes) ==
                      cudaSuccess &&
                  cudaMallocManaged(reinterpret_cast<void**>(&d_int_out2), int_bytes) ==
                      cudaSuccess,
              "cudaMallocManaged for integer buffers failed")) {
    return 1;
  }

  // 14. warp_vote_all
  {
    int n = 128;
    for (int i = 0; i < n; ++i) {
      if (i < 32) d_a[i] = 1.0f;
      else if (i < 64) d_a[i] = -1.0f;
      else if (i < 96) d_a[i] = (i % 2 == 0) ? 1.0f : -1.0f;
      else d_a[i] = 2.0f;
    }
    void* args[4] = {&d_a, &d_int_out1, &d_int_out2, &n};
    if (!Expect(cudaLaunchKernel(stubs[13], dim3(4), dim3(32), args, 0, nullptr) == cudaSuccess &&
                    cudaDeviceSynchronize() == cudaSuccess,
                "k14_warp_vote_all launch failed")) {
      return 1;
    }
    if (!Expect(d_int_out1[0] == 1 && d_int_out2[0] == 1, "k14 warp 0 vote mismatch")) return 1;
    if (!Expect(d_int_out1[1] == 0 && d_int_out2[1] == 0, "k14 warp 1 vote mismatch")) return 1;
    if (!Expect(d_int_out1[2] == 0 && d_int_out2[2] == 1, "k14 warp 2 vote mismatch")) return 1;
    if (!Expect(d_int_out1[3] == 1 && d_int_out2[3] == 1, "k14 warp 3 vote mismatch")) return 1;
  }

  // 15. warp_activemask_ballot
  {
    int n = 128;
    for (int i = 0; i < n; ++i) {
      d_int_in[i] = (i % 3 == 0) ? 1 : 0;
    }
    void* args[3] = {&d_int_in, &d_int_out1, &n};
    if (!Expect(cudaLaunchKernel(stubs[14], dim3(4), dim3(32), args, 0, nullptr) == cudaSuccess &&
                    cudaDeviceSynchronize() == cudaSuccess,
                "k15_warp_activemask_ballot launch failed")) {
      return 1;
    }
    for (int w = 0; w < 4; ++w) {
      int expected_count = 0;
      for (int i = 0; i < 32; ++i) {
        if (d_int_in[w * 32 + i] != 0) {
          ++expected_count;
        }
      }
      if (!Expect(d_int_out1[w] == expected_count, "k15_warp_activemask_ballot mismatch")) {
        return 1;
      }
    }
  }

  // 16. bit_reverse_brev
  {
    int n = kN;
    for (int i = 0; i < n; ++i) {
      d_int_in[i] = (1 << (i % 31)) | (i * 17);
    }
    void* args[3] = {&d_int_in, &d_int_out1, &n};
    if (!Expect(cudaLaunchKernel(stubs[15], dim3(4), dim3(64), args, 0, nullptr) == cudaSuccess &&
                    cudaDeviceSynchronize() == cudaSuccess,
                "k16_bit_reverse_brev launch failed")) {
      return 1;
    }
    for (int i = 0; i < n; ++i) {
      const auto expected =
          static_cast<int>(ReverseBits32(static_cast<std::uint32_t>(d_int_in[i])));
      if (!Expect(d_int_out1[i] == expected, "k16_bit_reverse_brev mismatch")) {
        return 1;
      }
    }
  }

  // 17. leading_zeros_clz
  {
    int n = kN;
    for (int i = 0; i < n; ++i) {
      d_int_in[i] = 1 << (i % 31);
    }
    void* args[3] = {&d_int_in, &d_int_out1, &n};
    if (!Expect(cudaLaunchKernel(stubs[16], dim3(4), dim3(64), args, 0, nullptr) == cudaSuccess &&
                    cudaDeviceSynchronize() == cudaSuccess,
                "k17_leading_zeros_clz launch failed")) {
      return 1;
    }
    for (int i = 0; i < n; ++i) {
      const auto val = static_cast<std::uint32_t>(d_int_in[i]);
      const int expected = (val == 0u) ? 32 : __builtin_clz(val);
      if (!Expect(d_int_out1[i] == expected, "k17_leading_zeros_clz mismatch")) {
        return 1;
      }
    }
  }

  // 18. atomic_histogram
  {
    int n = kN;
    std::vector<int> cpu_hist(16, 0);
    for (int i = 0; i < n; ++i) {
      d_int_in[i] = i % 16;
      cpu_hist[static_cast<std::size_t>(i % 16)]++;
    }
    (void)cudaMemset(d_int_out1, 0, 16 * sizeof(int));
    void* args[3] = {&d_int_in, &d_int_out1, &n};
    if (!Expect(cudaLaunchKernel(stubs[17], dim3(4), dim3(64), args, 0, nullptr) == cudaSuccess &&
                    cudaDeviceSynchronize() == cudaSuccess,
                "k18_atomic_histogram launch failed")) {
      return 1;
    }
    for (int bin = 0; bin < 16; ++bin) {
      if (!Expect(d_int_out1[bin] == cpu_hist[static_cast<std::size_t>(bin)],
                  "k18_atomic_histogram mismatch")) {
        return 1;
      }
    }
  }

  // 19. atomic_min_max
  {
    int n = kN;
    int cpu_min = 1000000;
    int cpu_max = -1000000;
    for (int i = 0; i < n; ++i) {
      const int val = (i * 73) % 500 - 250;
      d_int_in[i] = val;
      cpu_min = std::min(cpu_min, val);
      cpu_max = std::max(cpu_max, val);
    }
    d_int_out1[0] = 1000000;
    d_int_out2[0] = -1000000;
    void* args[4] = {&d_int_in, &d_int_out1, &d_int_out2, &n};
    if (!Expect(cudaLaunchKernel(stubs[18], dim3(4), dim3(64), args, 0, nullptr) == cudaSuccess &&
                    cudaDeviceSynchronize() == cudaSuccess,
                "k19_atomic_min_max launch failed")) {
      return 1;
    }
    if (!Expect(d_int_out1[0] == cpu_min && d_int_out2[0] == cpu_max,
                "k19_atomic_min_max mismatch")) {
      return 1;
    }
  }

  // 20. atomic_float_add
  {
    int n = kN;
    float cpu_sum = 0.0f;
    for (int i = 0; i < n; ++i) {
      d_a[i] = static_cast<float>(i % 10) * 0.1f;
      cpu_sum += d_a[i];
    }
    d_c[0] = 0.0f;
    void* args[3] = {&d_a, &d_c, &n};
    if (!Expect(cudaLaunchKernel(stubs[19], dim3(4), dim3(64), args, 0, nullptr) == cudaSuccess &&
                    cudaDeviceSynchronize() == cudaSuccess,
                "k20_atomic_float_add launch failed")) {
      return 1;
    }
    if (!Expect(muda::testing::AreClose(d_c[0], cpu_sum, 1e-2f),
                "k20_atomic_float_add mismatch")) {
      return 1;
    }
  }

  // 21. atomic_cas_counter
  {
    int n = kN;
    d_int_out1[0] = 0;
    (void)cudaMemset(d_int_out2, 0xff, int_bytes);
    void* args[3] = {&d_int_out1, &d_int_out2, &n};
    if (!Expect(cudaLaunchKernel(stubs[20], dim3(4), dim3(64), args, 0, nullptr) == cudaSuccess &&
                    cudaDeviceSynchronize() == cudaSuccess,
                "k21_atomic_cas_counter launch failed")) {
      return 1;
    }
    if (!Expect(d_int_out1[0] == n, "k21 final counter mismatch")) {
      return 1;
    }
    std::vector<int> claimed_slots(static_cast<std::size_t>(n), 0);
    for (int i = 0; i < n; ++i) {
      const int slot = d_int_out2[i];
      if (!Expect(slot >= 0 && slot < n, "k21 invalid slot value")) {
        return 1;
      }
      claimed_slots[static_cast<std::size_t>(slot)]++;
    }
    for (int i = 0; i < n; ++i) {
      if (!Expect(claimed_slots[static_cast<std::size_t>(i)] == 1,
                  "k21 duplicate slot detected")) {
        return 1;
      }
    }
  }

  // 22. half_vector_add
  {
    int n = kN;
    for (int i = 0; i < n; ++i) {
      d_a[i] = static_cast<float>(i) * 0.125f;
      d_b[i] = 1.5f;
    }
    (void)cudaMemset(d_c, 0, kBytes);
    void* args[4] = {&d_a, &d_b, &d_c, &n};
    if (!Expect(cudaLaunchKernel(stubs[21], dim3(4), dim3(64), args, 0, nullptr) == cudaSuccess &&
                    cudaDeviceSynchronize() == cudaSuccess,
                "k22_half_vector_add launch failed")) {
      return 1;
    }
    for (int i = 0; i < n; ++i) {
      const float expected = d_a[i] + d_b[i];
      if (!Expect(muda::testing::AreClose(d_c[i], expected, 1e-2f),
                  "k22_half_vector_add mismatch")) {
        return 1;
      }
    }
  }

  // 23. half2_simd_saxpy
  {
    int n = kN;
    float alpha = 2.0f;
    for (int i = 0; i < n; ++i) {
      d_a[i] = static_cast<float>(i) * 0.25f;
      d_b[i] = 0.5f;
    }
    (void)cudaMemset(d_c, 0, kBytes);
    void* args[5] = {&d_a, &d_b, &d_c, &alpha, &n};
    if (!Expect(cudaLaunchKernel(stubs[22], dim3(2), dim3(64), args, 0, nullptr) == cudaSuccess &&
                    cudaDeviceSynchronize() == cudaSuccess,
                "k23_half2_simd_saxpy launch failed")) {
      return 1;
    }
    for (int i = 0; i < n; ++i) {
      const float expected = alpha * d_a[i] + d_b[i];
      if (!Expect(muda::testing::AreClose(d_c[i], expected, 1e-2f),
                  "k23_half2_simd_saxpy mismatch")) {
        return 1;
      }
    }
  }

  // 24. vector4_unpack_math
  {
    int n = kN;
    for (int i = 0; i < n; ++i) {
      d_a[i] = static_cast<float>(i) + 1.0f;
    }
    (void)cudaMemset(d_c, 0, kBytes);
    void* args[3] = {&d_a, &d_c, &n};
    if (!Expect(cudaLaunchKernel(stubs[23], dim3(1), dim3(64), args, 0, nullptr) == cudaSuccess &&
                    cudaDeviceSynchronize() == cudaSuccess,
                "k24_vector4_unpack_math launch failed")) {
      return 1;
    }
    for (int i = 0; i < n; i += 4) {
      if (!Expect(muda::testing::AreClose(d_c[i], d_a[i] * 2.0f), "k24 x mismatch")) return 1;
      if (!Expect(muda::testing::AreClose(d_c[i + 1], d_a[i + 1] + 1.0f), "k24 y mismatch")) return 1;
      if (!Expect(muda::testing::AreClose(d_c[i + 2], d_a[i + 2] - 0.5f), "k24 z mismatch")) return 1;
      if (!Expect(muda::testing::AreClose(d_c[i + 3], d_a[i + 3] * 0.25f), "k24 w mismatch")) return 1;
    }
  }

  // 25. int4_vector_transform
  {
    int n = kN;
    for (int i = 0; i < n; ++i) {
      d_int_in[i] = i * 2;
    }
    void* args[3] = {&d_int_in, &d_int_out1, &n};
    if (!Expect(cudaLaunchKernel(stubs[24], dim3(1), dim3(64), args, 0, nullptr) == cudaSuccess &&
                    cudaDeviceSynchronize() == cudaSuccess,
                "k25_int4_vector_transform launch failed")) {
      return 1;
    }
    for (int i = 0; i < n; i += 4) {
      if (!Expect(d_int_out1[i] == d_int_in[i] + 10, "k25 x mismatch")) return 1;
      if (!Expect(d_int_out1[i + 1] == d_int_in[i + 1] * 2, "k25 y mismatch")) return 1;
      if (!Expect(d_int_out1[i + 2] == (d_int_in[i + 2] ^ 0xff), "k25 z mismatch")) return 1;
      if (!Expect(d_int_out1[i + 3] == d_int_in[i + 3] - 5, "k25 w mismatch")) return 1;
    }
  }

  // 26. relu_activation
  {
    int n = kN;
    for (int i = 0; i < n; ++i) {
      d_a[i] = static_cast<float>(i) - 128.0f;
    }
    (void)cudaMemset(d_c, 0, kBytes);
    void* args[3] = {&d_a, &d_c, &n};
    if (!Expect(cudaLaunchKernel(stubs[25], dim3(4), dim3(64), args, 0, nullptr) == cudaSuccess &&
                    cudaDeviceSynchronize() == cudaSuccess,
                "k26_relu_activation launch failed")) {
      return 1;
    }
    for (int i = 0; i < n; ++i) {
      const float expected = std::max(0.0f, d_a[i]);
      if (!Expect(muda::testing::AreClose(d_c[i], expected), "k26_relu_activation mismatch")) {
        return 1;
      }
    }
  }

  // 27. leaky_relu
  {
    int n = kN;
    float alpha = 0.01f;
    for (int i = 0; i < n; ++i) {
      d_a[i] = static_cast<float>(i) - 128.0f;
    }
    (void)cudaMemset(d_c, 0, kBytes);
    void* args[4] = {&d_a, &d_c, &alpha, &n};
    if (!Expect(cudaLaunchKernel(stubs[26], dim3(4), dim3(64), args, 0, nullptr) == cudaSuccess &&
                    cudaDeviceSynchronize() == cudaSuccess,
                "k27_leaky_relu launch failed")) {
      return 1;
    }
    for (int i = 0; i < n; ++i) {
      const float expected = (d_a[i] > 0.0f) ? d_a[i] : (d_a[i] * alpha);
      if (!Expect(muda::testing::AreClose(d_c[i], expected), "k27_leaky_relu mismatch")) {
        return 1;
      }
    }
  }

  // 28. softmax_warp (4 rows of 32 = 128 items)
  {
    int n = 128;
    for (int i = 0; i < n; ++i) {
      d_a[i] = static_cast<float>(i % 32) * 0.1f;
    }
    (void)cudaMemset(d_c, 0, kBytes);
    void* args[3] = {&d_a, &d_c, &n};
    if (!Expect(cudaLaunchKernel(stubs[27], dim3(4), dim3(32), args, 0, nullptr) == cudaSuccess &&
                    cudaDeviceSynchronize() == cudaSuccess,
                "k28_softmax_warp launch failed")) {
      return 1;
    }
    for (int r = 0; r < 4; ++r) {
      float max_val = -1e20f;
      for (int i = 0; i < 32; ++i) {
        max_val = std::max(max_val, d_a[r * 32 + i]);
      }
      float sum_exp = 0.0f;
      for (int i = 0; i < 32; ++i) {
        sum_exp += std::exp(d_a[r * 32 + i] - max_val);
      }
      for (int i = 0; i < 32; ++i) {
        const float expected = std::exp(d_a[r * 32 + i] - max_val) / sum_exp;
        if (!Expect(muda::testing::AreClose(d_c[r * 32 + i], expected, 1e-4f),
                    "k28_softmax_warp mismatch")) {
          return 1;
        }
      }
    }
  }

  // 29. layer_norm (4 rows of 32 = 128 items)
  {
    int n = 128;
    float gamma = 1.2f;
    float beta = 0.1f;
    float eps = 1e-5f;
    for (int i = 0; i < n; ++i) {
      d_a[i] = static_cast<float>(i % 32) * 0.2f - 3.0f;
    }
    (void)cudaMemset(d_c, 0, kBytes);
    void* args[6] = {&d_a, &d_c, &gamma, &beta, &eps, &n};
    if (!Expect(cudaLaunchKernel(stubs[28], dim3(4), dim3(32), args, 0, nullptr) == cudaSuccess &&
                    cudaDeviceSynchronize() == cudaSuccess,
                "k29_layer_norm launch failed")) {
      return 1;
    }
    for (int r = 0; r < 4; ++r) {
      float sum = 0.0f;
      for (int i = 0; i < 32; ++i) {
        sum += d_a[r * 32 + i];
      }
      const float mean = sum / 32.0f;
      float sq_sum = 0.0f;
      for (int i = 0; i < 32; ++i) {
        const float diff = d_a[r * 32 + i] - mean;
        sq_sum += diff * diff;
      }
      const float variance = sq_sum / 32.0f;
      const float inv_std = 1.0f / std::sqrt(variance + eps);

      for (int i = 0; i < 32; ++i) {
        const float expected = (d_a[r * 32 + i] - mean) * inv_std * gamma + beta;
        if (!Expect(muda::testing::AreClose(d_c[r * 32 + i], expected, 1e-3f),
                    "k29_layer_norm mismatch")) {
          return 1;
        }
      }
    }
  }

  // 30. spmv_csr (16 rows, 16 cols, ~4 nonzeros per row = 64 nonzeros)
  {
    const int num_rows = 16;
    int* d_offsets = nullptr;
    int* d_indices = nullptr;
    float* d_values = nullptr;
    float* d_x = nullptr;
    float* d_y = nullptr;

    const std::size_t offset_bytes = static_cast<std::size_t>(num_rows + 1) * sizeof(int);
    const std::size_t nz_bytes = 64 * sizeof(int);
    const std::size_t nz_val_bytes = 64 * sizeof(float);
    const std::size_t vec_bytes = 16 * sizeof(float);

    if (!Expect(cudaMallocManaged(reinterpret_cast<void**>(&d_offsets), offset_bytes) ==
                        cudaSuccess &&
                    cudaMallocManaged(reinterpret_cast<void**>(&d_indices), nz_bytes) ==
                        cudaSuccess &&
                    cudaMallocManaged(reinterpret_cast<void**>(&d_values), nz_val_bytes) ==
                        cudaSuccess &&
                    cudaMallocManaged(reinterpret_cast<void**>(&d_x), vec_bytes) ==
                        cudaSuccess &&
                    cudaMallocManaged(reinterpret_cast<void**>(&d_y), vec_bytes) ==
                        cudaSuccess,
                "cudaMallocManaged for SpMV CSR buffers failed")) {
      return 1;
    }

    std::vector<float> cpu_y(num_rows, 0.0f);
    for (int i = 0; i < 16; ++i) {
      d_x[i] = 1.0f + static_cast<float>(i) * 0.1f;
    }

    int current_nz = 0;
    for (int r = 0; r < num_rows; ++r) {
      d_offsets[r] = current_nz;
      for (int k = 0; k < 4; ++k) {
        const int col = (r + k * 3) % 16;
        const float val = 0.5f + static_cast<float>(k);
        d_indices[current_nz] = col;
        d_values[current_nz] = val;
        cpu_y[static_cast<std::size_t>(r)] += val * d_x[col];
        ++current_nz;
      }
    }
    d_offsets[num_rows] = current_nz;

    (void)cudaMemset(d_y, 0, vec_bytes);
    int num_rows_param = num_rows;
    void* args[6] = {&d_offsets, &d_indices, &d_values, &d_x, &d_y, &num_rows_param};
    if (!Expect(cudaLaunchKernel(stubs[29], dim3(1), dim3(16), args, 0, nullptr) == cudaSuccess &&
                    cudaDeviceSynchronize() == cudaSuccess,
                "k30_spmv_csr launch failed")) {
      return 1;
    }

    for (int r = 0; r < num_rows; ++r) {
      if (!Expect(muda::testing::AreClose(d_y[r], cpu_y[static_cast<std::size_t>(r)], 1e-3f),
                  "k30_spmv_csr mismatch")) {
        return 1;
      }
    }

    (void)cudaFree(d_offsets);
    (void)cudaFree(d_indices);
    (void)cudaFree(d_values);
    (void)cudaFree(d_x);
    (void)cudaFree(d_y);
  }

  (void)cudaFree(d_a);
  (void)cudaFree(d_b);
  (void)cudaFree(d_c);
  (void)cudaFree(d_int_in);
  (void)cudaFree(d_int_out1);
  (void)cudaFree(d_int_out2);
  (void)cudaDeviceReset();

  std::cout << "corpus_test: all 30 kernels passed numerical parity successfully!\n";
  return 0;
}

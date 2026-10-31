/**
 * @file corpus_test.cpp
 * @brief 10-kernel numerical parity suite against CPU reference implementations.
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
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

constexpr const char* kCorpusCudaSource = R"CUDA(
__device__ float affine_clamp(float x, float scale, float bias, float lo, float hi) {
  return fminf(fmaxf(fmaf(x, scale, bias), lo), hi);
}

__device__ float accumulate_pair(float a, float b) {
  return a + b;
}

// 1. vector_add
__global__ void k1_vector_add(const float* a, const float* b, float* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = a[i] + b[i];
  }
}

// 2. saxpy
__global__ void k2_saxpy(float alpha, const float* x, const float* y, float* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = fmaf(alpha, x[i], y[i]);
  }
}

// 3. vector_scale_clamp (uses __device__ helper + math intrinsics)
__global__ void k3_scale_clamp(const float* x, float* out, float scale, float bias, float lo,
                               float hi, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = affine_clamp(x[i], scale, bias, lo, hi);
  }
}

// 4. dot_product_block_reduce (static __shared__ + __syncthreads)
__global__ void k4_dot_reduce(const float* a, const float* b, float* block_sums, int n) {
  __shared__ float sdata[64];
  const int tid = threadIdx.x;
  const int idx = blockIdx.x * blockDim.x + tid;
  sdata[tid] = (idx < n) ? (a[idx] * b[idx]) : 0.0f;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sdata[tid] = accumulate_pair(sdata[tid], sdata[tid + stride]);
    }
    __syncthreads();
  }

  if (tid == 0) {
    block_sums[blockIdx.x] = sdata[0];
  }
}

// 5. matrix_transpose_naive (2D grid/block)
__global__ void k5_transpose_naive(const float* in, float* out, int width, int height) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (col < width && row < height) {
    out[col * height + row] = in[row * width + col];
  }
}

// 6. matrix_transpose_tiled (static __shared__ tile + __syncthreads)
__global__ void k6_transpose_tiled(const float* in, float* out, int width, int height) {
  __shared__ float tile[64];
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int col = blockIdx.x * 8 + tx;
  const int row = blockIdx.y * 8 + ty;
  if (col < width && row < height) {
    tile[ty * 8 + tx] = in[row * width + col];
  }
  __syncthreads();

  const int out_col = blockIdx.y * 8 + tx;
  const int out_row = blockIdx.x * 8 + ty;
  if (out_col < height && out_row < width) {
    out[out_row * height + out_col] = tile[tx * 8 + ty];
  }
}

// 7. gemm_naive (2D matrix multiplication C = A * B)
__global__ void k7_gemm_naive(const float* a, const float* b, float* c, int m, int n, int k) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < m && col < n) {
    float acc = 0.0f;
    for (int p = 0; p < k; ++p) {
      acc = fmaf(a[row * k + p], b[p * n + col], acc);
    }
    c[row * n + col] = acc;
  }
}

// 8. gemm_tiled (shared-memory tiled 8x8 matrix multiplication)
__global__ void k8_gemm_tiled(const float* a, const float* b, float* c, int n) {
  __shared__ float tile_a[64];
  __shared__ float tile_b[64];
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int row = blockIdx.y * 8 + ty;
  const int col = blockIdx.x * 8 + tx;

  float acc = 0.0f;
  const int num_tiles = (n + 7) / 8;
  for (int t = 0; t < num_tiles; ++t) {
    const int a_col = t * 8 + tx;
    const int b_row = t * 8 + ty;
    tile_a[ty * 8 + tx] = (row < n && a_col < n) ? a[row * n + a_col] : 0.0f;
    tile_b[ty * 8 + tx] = (b_row < n && col < n) ? b[b_row * n + col] : 0.0f;
    __syncthreads();

    for (int p = 0; p < 8; ++p) {
      acc = fmaf(tile_a[ty * 8 + p], tile_b[p * 8 + tx], acc);
    }
    __syncthreads();
  }

  if (row < n && col < n) {
    c[row * n + col] = acc;
  }
}

// 9. stencil_1d_dynamic_shared (extern __shared__ dynamic threadgroup memory)
__global__ void k9_stencil_1d(const float* in, float* out, int n) {
  extern __shared__ float halo[];
  const int tid = threadIdx.x;
  const int gid = blockIdx.x * blockDim.x + tid;
  const int bdim = blockDim.x;

  if (gid < n) {
    halo[tid + 1] = in[gid];
    if (tid == 0) {
      halo[0] = (gid > 0) ? in[gid - 1] : in[0];
    }
    if (tid == bdim - 1 || gid == n - 1) {
      const int right_idx = (gid + 1 < n) ? (gid + 1) : (n - 1);
      halo[tid + 2] = in[right_idx];
    }
  }
  __syncthreads();

  if (gid < n) {
    out[gid] = 0.25f * halo[tid] + 0.5f * halo[tid + 1] + 0.25f * halo[tid + 2];
  }
}

// 10. row_sum_2d (2D matrix row reduction with __device__ helper)
__global__ void k10_row_sum(const float* mat, float* row_sums, int rows, int cols) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r < rows) {
    float sum = 0.0f;
    for (int c = 0; c < cols; ++c) {
      sum = accumulate_pair(sum, mat[r * cols + c]);
    }
    row_sums[r] = sum;
  }
}
)CUDA";

void HostStub1() {}
void HostStub2() {}
void HostStub3() {}
void HostStub4() {}
void HostStub5() {}
void HostStub6() {}
void HostStub7() {}
void HostStub8() {}
void HostStub9() {}
void HostStub10() {}

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

  const muda::testing::ScopedTempDir temp_dir("muda_corpus");
  if (!Expect(temp_dir.valid(), "failed to create temp directory")) {
    return 1;
  }

  const std::filesystem::path cu_path = temp_dir.path() / "corpus10.cu";
  const std::filesystem::path metallib_path = temp_dir.path() / "corpus10.metallib";
  {
    std::ofstream out(cu_path, std::ios::binary);
    out << kCorpusCudaSource;
  }

  const muda::compiler::TranslationResult tr =
      muda::compiler::TranslateCudaToMsl(kCorpusCudaSource);
  if (!Expect(tr.ok() && tr.kernels.size() == 10, "corpus should translate all 10 kernels")) {
    std::cerr << "translation error [" << tr.diagnostic_id() << "]: " << tr.error() << "\n";
    return 1;
  }

  muda::compiler::CompileOptions options;
  options.input = cu_path;
  options.output = metallib_path;
  options.output_kind = muda::compiler::OutputKind::kMetallib;
  const muda::compiler::CompileResult compiled = muda::compiler::Compile(options);
  if (!Expect(compiled.status == muda::compiler::Status::kOk,
              "corpus10.cu should compile to corpus10.metallib")) {
    std::cerr << "compile error [" << compiled.diagnostic_id() << "]: " << compiled.message()
              << "\n";
    return 1;
  }

  const void* stubs[10] = {
      reinterpret_cast<const void*>(&HostStub1), reinterpret_cast<const void*>(&HostStub2),
      reinterpret_cast<const void*>(&HostStub3), reinterpret_cast<const void*>(&HostStub4),
      reinterpret_cast<const void*>(&HostStub5), reinterpret_cast<const void*>(&HostStub6),
      reinterpret_cast<const void*>(&HostStub7), reinterpret_cast<const void*>(&HostStub8),
      reinterpret_cast<const void*>(&HostStub9), reinterpret_cast<const void*>(&HostStub10),
  };

  for (std::size_t k = 0; k < 10; ++k) {
    if (!Expect(RegisterTranslatedKernel(stubs[k], metallib_path, tr.kernels[k]),
                "failed to register corpus kernel")) {
      return 1;
    }
  }

  // Allocate shared buffers for testing all 10 kernels
  constexpr int kN = 256;
  constexpr std::size_t kBytes = static_cast<std::size_t>(kN) * sizeof(float);
  float* d_a = nullptr;
  float* d_b = nullptr;
  float* d_c = nullptr;
  if (!Expect(cudaMallocManaged(reinterpret_cast<void**>(&d_a), kBytes) == cudaSuccess &&
                  cudaMallocManaged(reinterpret_cast<void**>(&d_b), kBytes) == cudaSuccess &&
                  cudaMallocManaged(reinterpret_cast<void**>(&d_c), kBytes) == cudaSuccess,
              "cudaMallocManaged for corpus buffers failed")) {
    return 1;
  }

  for (int i = 0; i < kN; ++i) {
    d_a[i] = static_cast<float>(i) * 0.25f - 10.0f;
    d_b[i] = static_cast<float>(kN - i) * 0.5f;
    d_c[i] = 0.0f;
  }

  // Kernel 1: vector_add
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

  // Kernel 2: saxpy
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

  // Kernel 3: scale_clamp
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

  // Kernel 4: dot_reduce
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

  // Kernel 5: transpose_naive (16 x 16)
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

  // Kernel 6: transpose_tiled (16 x 16)
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

  // Kernel 7: gemm_naive (16 x 16)
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

  // Kernel 8: gemm_tiled (16 x 16)
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

  // Kernel 9: stencil_1d (dynamic shared memory)
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

  // Kernel 10: row_sum_2d (16 x 16)
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

  (void)cudaFree(d_a);
  (void)cudaFree(d_b);
  (void)cudaFree(d_c);
  (void)cudaDeviceReset();

  std::cout << "corpus_test: all 10 kernels passed numerical parity\n";
  return 0;
}

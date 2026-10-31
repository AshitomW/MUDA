/**
 * @file runtime_bench.cpp
 * @brief Benchmark harness measuring launch latency, copy bandwidth, vector_add, and GEMM.
 */

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string_view>
#include <vector>

#include "cuda_runtime.h"
#include "muda/compiler/driver.hpp"
#include "muda/compiler/toolchain.hpp"
#include "muda/compiler/translator.hpp"

namespace {

constexpr std::string_view kBenchSource = R"CUDA(
__global__ void bench_noop(float* out) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    out[0] = 1.0f;
  }
}

__global__ void bench_vector_add(const float* a, const float* b, float* c, int n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}

__global__ void bench_vector_add_float4(const float4* a, const float4* b, float4* c, int n4) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n4) {
    c[idx] = a[idx] + b[idx];
  }
}

__global__ void bench_gemm_tiled(const float* a, const float* b, float* c, int n) {
  __shared__ float tile_a[256];
  __shared__ float tile_b[256];
  const int row = blockIdx.y * 16 + threadIdx.y;
  const int col = blockIdx.x * 16 + threadIdx.x;
  float acc = 0.0f;
  const int num_tiles = (n + 15) / 16;
  for (int t = 0; t < num_tiles; ++t) {
    const int a_col = t * 16 + threadIdx.x;
    const int b_row = t * 16 + threadIdx.y;
    if (row < n && a_col < n) {
      tile_a[threadIdx.y * 16 + threadIdx.x] = a[row * n + a_col];
    } else {
      tile_a[threadIdx.y * 16 + threadIdx.x] = 0.0f;
    }
    if (b_row < n && col < n) {
      tile_b[threadIdx.y * 16 + threadIdx.x] = b[b_row * n + col];
    } else {
      tile_b[threadIdx.y * 16 + threadIdx.x] = 0.0f;
    }
    __syncthreads();
    for (int k = 0; k < 16; ++k) {
      acc = fmaf(tile_a[threadIdx.y * 16 + k], tile_b[k * 16 + threadIdx.x], acc);
    }
    __syncthreads();
  }
  if (row < n && col < n) {
    c[row * n + col] = acc;
  }
}

__global__ void bench_gemm_register_tiled(const float* a, const float* b, float* c, int n) {
  __shared__ float s_a[1024];
  __shared__ float s_b[1024];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * 16 + tx;

  const int bx = blockIdx.x * 64;
  const int by = blockIdx.y * 64;

  float acc[16];
  for (int i = 0; i < 16; ++i) {
    acc[i] = 0.0f;
  }

  const int num_tiles = (n + 15) / 16;
  for (int t = 0; t < num_tiles; ++t) {
    for (int step = 0; step < 4; ++step) {
      const int a_r = (tid + step * 256) / 16;
      const int a_c = ((tid + step * 256) % 16) + t * 16;
      if (by + a_r < n && a_c < n) {
        s_a[a_r * 16 + ((tid + step * 256) % 16)] = a[(by + a_r) * n + a_c];
      } else {
        s_a[a_r * 16 + ((tid + step * 256) % 16)] = 0.0f;
      }

      const int b_r = ((tid + step * 256) / 64) + t * 16;
      const int b_c = (tid + step * 256) % 64;
      if (b_r < n && bx + b_c < n) {
        s_b[((tid + step * 256) / 64) * 64 + b_c] = b[b_r * n + bx + b_c];
      } else {
        s_b[((tid + step * 256) / 64) * 64 + b_c] = 0.0f;
      }
    }
    __syncthreads();

    for (int k = 0; k < 16; ++k) {
      for (int i = 0; i < 4; ++i) {
        const float val_a = s_a[(ty * 4 + i) * 16 + k];
        for (int j = 0; j < 4; ++j) {
          const float val_b = s_b[k * 64 + (tx * 4 + j)];
          acc[i * 4 + j] = fmaf(val_a, val_b, acc[i * 4 + j]);
        }
      }
    }
    __syncthreads();
  }

  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      const int out_r = by + ty * 4 + i;
      const int out_c = bx + tx * 4 + j;
      if (out_r < n && out_c < n) {
        c[out_r * n + out_c] = acc[i * 4 + j];
      }
    }
  }
}
)CUDA";

void StubNoop() {}
void StubVectorAdd() {}
void StubVectorAddFloat4() {}
void StubGemmTiled() {}
void StubGemmRegisterTiled() {}

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
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0 ||
      !muda::compiler::MetalToolchainAvailable()) {
    std::fprintf(stderr, "Metal GPU or Metal toolchain unavailable; skipping benchmark.\n");
    return 0;
  }

  cudaDeviceProp prop{};
  (void)cudaGetDeviceProperties(&prop, 0);
  std::printf("=================================================================\n");
  std::printf("                 MUDA GPU BENCHMARK SUITE (macOS)                \n");
  std::printf("=================================================================\n");
  std::printf("Device Name             : %s\n", prop.name);
  std::printf("Unified Memory (Zero-Cp): %s\n", prop.unifiedAddressing != 0 ? "Yes" : "No");
  std::printf("Theoretical DRAM Peak   : 68.25 GB/s (Apple M1 128-bit LPDDR4X)\n");
  std::printf("Theoretical Compute Peak: 2617 GFLOP/s (8 GPU Cores, FP32)\n");
  std::printf("=================================================================\n\n");

  const std::filesystem::path temp_dir =
      std::filesystem::temp_directory_path() / "muda_runtime_bench";
  std::filesystem::create_directories(temp_dir);
  const std::filesystem::path cu_path = temp_dir / "bench.cu";
  const std::filesystem::path metallib_path = temp_dir / "bench.metallib";

  {
    std::ofstream out(cu_path, std::ios::binary | std::ios::trunc);
    out.write(kBenchSource.data(), static_cast<std::streamsize>(kBenchSource.size()));
  }

  const muda::compiler::TranslationResult tr = muda::compiler::TranslateCudaToMsl(kBenchSource);
  if (!tr.ok() || tr.kernels.size() != 5u) {
    std::fprintf(stderr, "Benchmark kernel translation failed (got %zu kernels, expected 5).\n",
                 tr.kernels.size());
    return EXIT_FAILURE;
  }

  muda::compiler::CompileOptions options{};
  options.input = cu_path;
  options.output = metallib_path;
  options.output_kind = muda::compiler::OutputKind::kMetallib;

  const muda::compiler::CompileResult compiled = muda::compiler::Compile(options);
  if (compiled.status != muda::compiler::Status::kOk) {
    std::fprintf(stderr, "Benchmark kernel compilation failed.\n");
    return EXIT_FAILURE;
  }

  if (!RegisterTranslatedKernel(reinterpret_cast<const void*>(&StubNoop), metallib_path,
                                tr.kernels[0]) ||
      !RegisterTranslatedKernel(reinterpret_cast<const void*>(&StubVectorAdd), metallib_path,
                                tr.kernels[1]) ||
      !RegisterTranslatedKernel(reinterpret_cast<const void*>(&StubVectorAddFloat4), metallib_path,
                                tr.kernels[2]) ||
      !RegisterTranslatedKernel(reinterpret_cast<const void*>(&StubGemmTiled), metallib_path,
                                tr.kernels[3]) ||
      !RegisterTranslatedKernel(reinterpret_cast<const void*>(&StubGemmRegisterTiled),
                                metallib_path, tr.kernels[4])) {
    std::fprintf(stderr, "Benchmark kernel registration failed.\n");
    return EXIT_FAILURE;
  }

  // 1. Steady-state launch latency
  {
    float* d_out = nullptr;
    (void)cudaMalloc(reinterpret_cast<void**>(&d_out), sizeof(float));
    void* args[] = {&d_out};
    for (int i = 0; i < 20; ++i) {
      (void)cudaLaunchKernel(reinterpret_cast<const void*>(&StubNoop), dim3(1u), dim3(1u), args, 0u,
                             nullptr);
    }
    (void)cudaDeviceSynchronize();

    constexpr int kLaunchCount = 500;
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < kLaunchCount; ++i) {
      (void)cudaLaunchKernel(reinterpret_cast<const void*>(&StubNoop), dim3(1u), dim3(1u), args, 0u,
                             nullptr);
    }
    const auto t_enqueue = std::chrono::steady_clock::now();
    (void)cudaDeviceSynchronize();
    const auto t_sync = std::chrono::steady_clock::now();

    const double enqueue_us =
        std::chrono::duration<double, std::micro>(t_enqueue - t0).count() / kLaunchCount;
    const double total_us =
        std::chrono::duration<double, std::micro>(t_sync - t0).count() / kLaunchCount;
    std::printf("[1] Empty Kernel Launch Latency (%d iterations):\n", kLaunchCount);
    std::printf("    Pipelined enqueue latency : %.2f us / launch (amortized over stream)\n",
                enqueue_us);
    std::printf("    Synchronized round-trip   : %.2f us / launch\n\n", total_us);
    (void)cudaFree(d_out);
  }

  // 2. Copy Bandwidth and Zero-Copy Access (64 MiB)
  {
    constexpr std::size_t kBytes = 64ull * 1024ull * 1024ull;
    constexpr int kIters = 25;
    std::vector<std::uint8_t> host_buf(kBytes, 0x5Au);
    void* d_a = nullptr;
    void* d_b = nullptr;
    (void)cudaMalloc(&d_a, kBytes);
    (void)cudaMalloc(&d_b, kBytes);

    const auto h2d_start = std::chrono::steady_clock::now();
    for (int i = 0; i < kIters; ++i) {
      (void)cudaMemcpy(d_a, host_buf.data(), kBytes, cudaMemcpyHostToDevice);
    }
    const auto h2d_end = std::chrono::steady_clock::now();

    const auto d2d_start = std::chrono::steady_clock::now();
    for (int i = 0; i < kIters; ++i) {
      (void)cudaMemcpy(d_b, d_a, kBytes, cudaMemcpyDeviceToDevice);
    }
    const auto d2d_end = std::chrono::steady_clock::now();

    const auto d2h_start = std::chrono::steady_clock::now();
    for (int i = 0; i < kIters; ++i) {
      (void)cudaMemcpy(host_buf.data(), d_b, kBytes, cudaMemcpyDeviceToHost);
    }
    const auto d2h_end = std::chrono::steady_clock::now();

    const double total_gb = (static_cast<double>(kBytes) * static_cast<double>(kIters)) / 1.0e9;
    const double h2d_gbps = total_gb / std::chrono::duration<double>(h2d_end - h2d_start).count();
    const double d2d_gbps = total_gb / std::chrono::duration<double>(d2d_end - d2d_start).count();
    const double d2h_gbps = total_gb / std::chrono::duration<double>(d2h_end - d2h_start).count();

    std::printf("[2] Copy Bandwidth & Access Latency (64 MiB, %d iterations):\n", kIters);
    std::printf("    Host-to-Device (H2D)      : %.2f GB/s\n", h2d_gbps);
    std::printf("    Device-to-Device (D2D)    : %.2f GB/s\n", d2d_gbps);
    std::printf("    Device-to-Host (D2H)      : %.2f GB/s\n", d2h_gbps);

    // Measure Zero-Copy Unified Memory direct host-pointer access
    const auto zc_start = std::chrono::steady_clock::now();
    std::uint64_t sum = 0;
    const auto* d_ptr = static_cast<const std::uint64_t*>(d_a);
    const std::size_t u64_count = kBytes / sizeof(std::uint64_t);
    for (int it = 0; it < kIters; ++it) {
      for (std::size_t i = 0; i < u64_count; i += 8) {
        sum += d_ptr[i];
      }
    }
    const auto zc_end = std::chrono::steady_clock::now();
    const double zc_gbps = total_gb / std::chrono::duration<double>(zc_end - zc_start).count();
    std::printf("    Zero-Copy Host Direct Rd  : %.2f GB/s (checksum: %llu)\n\n", zc_gbps,
                static_cast<unsigned long long>(sum));

    (void)cudaFree(d_b);
    (void)cudaFree(d_a);
  }

  // 3. Memory Bandwidth: Scalar vs Vectorized float4 (16M floats = 192 MiB traffic)
  {
    constexpr int kN = 16 * 1024 * 1024;
    constexpr std::size_t kBytes = static_cast<std::size_t>(kN) * sizeof(float);
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    (void)cudaMalloc(reinterpret_cast<void**>(&d_a), kBytes);
    (void)cudaMalloc(reinterpret_cast<void**>(&d_b), kBytes);
    (void)cudaMalloc(reinterpret_cast<void**>(&d_c), kBytes);
    (void)cudaMemset(d_a, 0, kBytes);
    (void)cudaMemset(d_b, 0, kBytes);

    int n_arg = kN;
    void* scalar_args[] = {&d_a, &d_b, &d_c, &n_arg};
    const dim3 block(256u);
    const dim3 grid(static_cast<unsigned int>((kN + 255) / 256));

    (void)cudaLaunchKernel(reinterpret_cast<const void*>(&StubVectorAdd), grid, block, scalar_args,
                           0u, nullptr);
    (void)cudaDeviceSynchronize();

    cudaEvent_t ev_start = nullptr;
    cudaEvent_t ev_stop = nullptr;
    (void)cudaEventCreate(&ev_start);
    (void)cudaEventCreate(&ev_stop);

    constexpr int kIters = 25;
    (void)cudaEventRecord(ev_start, nullptr);
    for (int i = 0; i < kIters; ++i) {
      (void)cudaLaunchKernel(reinterpret_cast<const void*>(&StubVectorAdd), grid, block,
                             scalar_args, 0u, nullptr);
    }
    (void)cudaEventRecord(ev_stop, nullptr);
    (void)cudaEventSynchronize(ev_stop);

    float elapsed_ms = 0.0f;
    (void)cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop);
    const double total_bytes = 3.0 * static_cast<double>(kBytes) * static_cast<double>(kIters);
    const double scalar_gbps = (total_bytes / 1.0e9) / (static_cast<double>(elapsed_ms) / 1000.0);

    // Vectorized float4 (128-bit memory transactions)
    constexpr int kN4 = kN / 4;
    int n4_arg = kN4;
    void* vec4_args[] = {&d_a, &d_b, &d_c, &n4_arg};
    const dim3 block4(256u);
    const dim3 grid4(static_cast<unsigned int>((kN4 + 255) / 256));

    (void)cudaLaunchKernel(reinterpret_cast<const void*>(&StubVectorAddFloat4), grid4, block4,
                           vec4_args, 0u, nullptr);
    (void)cudaDeviceSynchronize();

    (void)cudaEventRecord(ev_start, nullptr);
    for (int i = 0; i < kIters; ++i) {
      (void)cudaLaunchKernel(reinterpret_cast<const void*>(&StubVectorAddFloat4), grid4, block4,
                             vec4_args, 0u, nullptr);
    }
    (void)cudaEventRecord(ev_stop, nullptr);
    (void)cudaEventSynchronize(ev_stop);

    float elapsed4_ms = 0.0f;
    (void)cudaEventElapsedTime(&elapsed4_ms, ev_start, ev_stop);
    const double vec4_gbps = (total_bytes / 1.0e9) / (static_cast<double>(elapsed4_ms) / 1000.0);

    constexpr double kM1DramPeakGBps = 68.25;
    std::printf("[3] Memory Bandwidth (N = %d floats, 192 MiB traffic, %d iterations):\n", kN,
                kIters);
    std::printf("    Scalar (float) bandwidth   : %.2f GB/s (%.1f%% of DRAM peak)\n", scalar_gbps,
                (scalar_gbps / kM1DramPeakGBps) * 100.0);
    std::printf("    Vectorized (float4) bandw. : %.2f GB/s (%.1f%% of DRAM peak)\n\n", vec4_gbps,
                (vec4_gbps / kM1DramPeakGBps) * 100.0);

    (void)cudaEventDestroy(ev_stop);
    (void)cudaEventDestroy(ev_start);
    (void)cudaFree(d_c);
    (void)cudaFree(d_b);
    (void)cudaFree(d_a);
  }

  // 4. Compute Throughput: 16x16 Shared-Tiled vs 64x64 Register-Tiled GEMM (512x512 FP32)
  {
    constexpr int kN = 512;
    constexpr std::size_t kBytes = static_cast<std::size_t>(kN * kN) * sizeof(float);
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    (void)cudaMalloc(reinterpret_cast<void**>(&d_a), kBytes);
    (void)cudaMalloc(reinterpret_cast<void**>(&d_b), kBytes);
    (void)cudaMalloc(reinterpret_cast<void**>(&d_c), kBytes);
    (void)cudaMemset(d_a, 0, kBytes);
    (void)cudaMemset(d_b, 0, kBytes);

    int n_arg = kN;
    void* args[] = {&d_a, &d_b, &d_c, &n_arg};

    // 16x16 Shared-Memory Tiled
    const dim3 block_tiled(16u, 16u, 1u);
    const dim3 grid_tiled(static_cast<unsigned int>(kN / 16), static_cast<unsigned int>(kN / 16),
                          1u);

    (void)cudaLaunchKernel(reinterpret_cast<const void*>(&StubGemmTiled), grid_tiled, block_tiled,
                           args, 0u, nullptr);
    (void)cudaDeviceSynchronize();

    cudaEvent_t ev_start = nullptr;
    cudaEvent_t ev_stop = nullptr;
    (void)cudaEventCreate(&ev_start);
    (void)cudaEventCreate(&ev_stop);

    constexpr int kIters = 25;
    (void)cudaEventRecord(ev_start, nullptr);
    for (int i = 0; i < kIters; ++i) {
      (void)cudaLaunchKernel(reinterpret_cast<const void*>(&StubGemmTiled), grid_tiled, block_tiled,
                             args, 0u, nullptr);
    }
    (void)cudaEventRecord(ev_stop, nullptr);
    (void)cudaEventSynchronize(ev_stop);

    float elapsed_tiled_ms = 0.0f;
    (void)cudaEventElapsedTime(&elapsed_tiled_ms, ev_start, ev_stop);
    const double flops = 2.0 * static_cast<double>(kN) * static_cast<double>(kN) *
                         static_cast<double>(kN) * static_cast<double>(kIters);
    const double gflops_tiled = (flops / 1.0e9) / (static_cast<double>(elapsed_tiled_ms) / 1000.0);

    // 64x64 Block, 4x4 Register-Tiled
    const dim3 block_reg(16u, 16u, 1u);
    const dim3 grid_reg(static_cast<unsigned int>(kN / 64), static_cast<unsigned int>(kN / 64), 1u);

    (void)cudaLaunchKernel(reinterpret_cast<const void*>(&StubGemmRegisterTiled), grid_reg,
                           block_reg, args, 0u, nullptr);
    (void)cudaDeviceSynchronize();

    (void)cudaEventRecord(ev_start, nullptr);
    for (int i = 0; i < kIters; ++i) {
      (void)cudaLaunchKernel(reinterpret_cast<const void*>(&StubGemmRegisterTiled), grid_reg,
                             block_reg, args, 0u, nullptr);
    }
    (void)cudaEventRecord(ev_stop, nullptr);
    (void)cudaEventSynchronize(ev_stop);

    float elapsed_reg_ms = 0.0f;
    (void)cudaEventElapsedTime(&elapsed_reg_ms, ev_start, ev_stop);
    const double gflops_reg = (flops / 1.0e9) / (static_cast<double>(elapsed_reg_ms) / 1000.0);

    constexpr double kM1Fp32PeakGFlops = 2617.0;
    std::printf("[4] Compute Throughput (512x512 FP32 GEMM, %d iterations):\n", kIters);
    std::printf("    16x16 Shared-Tiled Throughput  : %.2f GFLOP/s (%.1f%% of FP32 peak)\n",
                gflops_tiled, (gflops_tiled / kM1Fp32PeakGFlops) * 100.0);
    std::printf("    64x64 Register-Tiled Throughp. : %.2f GFLOP/s (%.1f%% of FP32 peak)\n",
                gflops_reg, (gflops_reg / kM1Fp32PeakGFlops) * 100.0);
    std::printf("    Register-Tiling Acceleration   : %.2fx speedup\n\n",
                gflops_reg / gflops_tiled);

    (void)cudaEventDestroy(ev_stop);
    (void)cudaEventDestroy(ev_start);
    (void)cudaFree(d_c);
    (void)cudaFree(d_b);
    (void)cudaFree(d_a);
  }

  std::error_code ec{};
  std::filesystem::remove_all(temp_dir, ec);
  return EXIT_SUCCESS;
}

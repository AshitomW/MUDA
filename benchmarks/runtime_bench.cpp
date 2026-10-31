/**
 * @file runtime_bench.cpp
 * @brief Benchmark harness measuring launch latency, copy bandwidth, vector_add, and GEMM.
 */

#include "cuda_runtime.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string_view>
#include <vector>

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
)CUDA";

void StubNoop() {}
void StubVectorAdd() {}
void StubGemmTiled() {}

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
  std::printf("=== MUDA Benchmark Suite ===\n");
  std::printf("Device: %s (Unified Memory: %s)\n\n",
              prop.name,
              prop.unifiedAddressing != 0 ? "Yes" : "No");

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
  if (!tr.ok() || tr.kernels.size() != 3u) {
    std::fprintf(stderr, "Benchmark kernel translation failed.\n");
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
      !RegisterTranslatedKernel(reinterpret_cast<const void*>(&StubGemmTiled), metallib_path,
                                tr.kernels[2])) {
    std::fprintf(stderr, "Benchmark kernel registration failed.\n");
    return EXIT_FAILURE;
  }

  // 1. Steady-state launch latency (async enqueue + sync)
  {
    float* d_out = nullptr;
    (void)cudaMalloc(reinterpret_cast<void**>(&d_out), sizeof(float));
    void* args[] = {&d_out};
    for (int i = 0; i < 20; ++i) {
      (void)cudaLaunchKernel(
          reinterpret_cast<const void*>(&StubNoop), dim3(1u), dim3(1u), args, 0u, nullptr);
    }
    (void)cudaDeviceSynchronize();

    constexpr int kLaunchCount = 500;
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < kLaunchCount; ++i) {
      (void)cudaLaunchKernel(
          reinterpret_cast<const void*>(&StubNoop), dim3(1u), dim3(1u), args, 0u, nullptr);
    }
    const auto t_enqueue = std::chrono::steady_clock::now();
    (void)cudaDeviceSynchronize();
    const auto t_sync = std::chrono::steady_clock::now();

    const double enqueue_us =
        std::chrono::duration<double, std::micro>(t_enqueue - t0).count() / kLaunchCount;
    const double total_us =
        std::chrono::duration<double, std::micro>(t_sync - t0).count() / kLaunchCount;
    std::printf("[1] Empty Kernel Launch Latency (%d iterations):\n", kLaunchCount);
    std::printf("    Async enqueue latency : %.2f us / launch\n", enqueue_us);
    std::printf("    Synchronized latency  : %.2f us / launch\n\n", total_us);
    (void)cudaFree(d_out);
  }

  // 2. Copy bandwidth (H2D, D2H, D2D — 64 MiB)
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

    const double total_gb =
        (static_cast<double>(kBytes) * static_cast<double>(kIters)) / 1.0e9;
    const double h2d_gbps =
        total_gb / std::chrono::duration<double>(h2d_end - h2d_start).count();
    const double d2d_gbps =
        total_gb / std::chrono::duration<double>(d2d_end - d2d_start).count();
    const double d2h_gbps =
        total_gb / std::chrono::duration<double>(d2h_end - d2h_start).count();

    std::printf("[2] Copy Bandwidth (64 MiB, %d iterations):\n", kIters);
    std::printf("    H2D Bandwidth : %.2f GB/s\n", h2d_gbps);
    std::printf("    D2D Bandwidth : %.2f GB/s\n", d2d_gbps);
    std::printf("    D2H Bandwidth : %.2f GB/s\n\n", d2h_gbps);

    (void)cudaFree(d_b);
    (void)cudaFree(d_a);
  }

  // 3. Vector Add throughput (16M floats = 64 MiB per vector, 192 MiB total traffic)
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
    void* args[] = {&d_a, &d_b, &d_c, &n_arg};
    const dim3 block(256u);
    const dim3 grid(static_cast<unsigned int>((kN + 255) / 256));

    (void)cudaLaunchKernel(
        reinterpret_cast<const void*>(&StubVectorAdd), grid, block, args, 0u, nullptr);
    (void)cudaDeviceSynchronize();

    cudaEvent_t ev_start = nullptr;
    cudaEvent_t ev_stop = nullptr;
    (void)cudaEventCreate(&ev_start);
    (void)cudaEventCreate(&ev_stop);

    constexpr int kIters = 20;
    (void)cudaEventRecord(ev_start, nullptr);
    for (int i = 0; i < kIters; ++i) {
      (void)cudaLaunchKernel(
          reinterpret_cast<const void*>(&StubVectorAdd), grid, block, args, 0u, nullptr);
    }
    (void)cudaEventRecord(ev_stop, nullptr);
    (void)cudaEventSynchronize(ev_stop);

    float elapsed_ms = 0.0f;
    (void)cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop);
    const double total_bytes = 3.0 * static_cast<double>(kBytes) * static_cast<double>(kIters);
    const double gbps = (total_bytes / 1.0e9) / (static_cast<double>(elapsed_ms) / 1000.0);

    std::printf("[3] vector_add Throughput (N = %d, %d iterations):\n", kN, kIters);
    std::printf("    Avg kernel time     : %.3f ms\n",
                static_cast<double>(elapsed_ms) / kIters);
    std::printf("    Effective bandwidth : %.2f GB/s\n\n", gbps);

    (void)cudaEventDestroy(ev_stop);
    (void)cudaEventDestroy(ev_start);
    (void)cudaFree(d_c);
    (void)cudaFree(d_b);
    (void)cudaFree(d_a);
  }

  // 4. Tiled GEMM baseline (512x512 FP32)
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
    const dim3 block(16u, 16u, 1u);
    const dim3 grid(static_cast<unsigned int>(kN / 16), static_cast<unsigned int>(kN / 16), 1u);

    (void)cudaLaunchKernel(
        reinterpret_cast<const void*>(&StubGemmTiled), grid, block, args, 0u, nullptr);
    (void)cudaDeviceSynchronize();

    cudaEvent_t ev_start = nullptr;
    cudaEvent_t ev_stop = nullptr;
    (void)cudaEventCreate(&ev_start);
    (void)cudaEventCreate(&ev_stop);

    constexpr int kIters = 20;
    (void)cudaEventRecord(ev_start, nullptr);
    for (int i = 0; i < kIters; ++i) {
      (void)cudaLaunchKernel(
          reinterpret_cast<const void*>(&StubGemmTiled), grid, block, args, 0u, nullptr);
    }
    (void)cudaEventRecord(ev_stop, nullptr);
    (void)cudaEventSynchronize(ev_stop);

    float elapsed_ms = 0.0f;
    (void)cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop);
    const double flops = 2.0 * static_cast<double>(kN) * static_cast<double>(kN) *
                         static_cast<double>(kN) * static_cast<double>(kIters);
    const double gflops = (flops / 1.0e9) / (static_cast<double>(elapsed_ms) / 1000.0);

    std::printf("[4] Tiled GEMM Baseline (%dx%d FP32, %d iterations):\n", kN, kN, kIters);
    std::printf("    Avg kernel time : %.3f ms\n", static_cast<double>(elapsed_ms) / kIters);
    std::printf("    Throughput      : %.2f GFLOP/s\n", gflops);

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

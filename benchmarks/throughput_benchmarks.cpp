/**
 * @file throughput_benchmarks.cpp
 * @brief Throughput benchmark suite implementation measuring memory bandwidth and compute
 * scalability.
 */

#include "throughput_benchmarks.hpp"

#include <cstdio>
#include <string>
#include <vector>

#include "cuda_runtime.h"

namespace muda::bench {

ThroughputBenchmarkSuiteResults RunThroughputBenchmarks(
    const ThroughputKernelStubs& stubs, [[maybe_unused]] const BenchmarkConfig& base_config) {
  ThroughputBenchmarkSuiteResults suite_results;

  std::printf("=================================================================\n");
  std::printf("               MUDA THROUGHPUT BENCHMARK SUITE                   \n");
  std::printf("=================================================================\n\n");

  constexpr double kM1DramPeakGBps = 68.25;
  constexpr double kM1EstimatedPhysicalCeilingGBps = 64.50;  // ~94.5% LPDDR4X protocol ceiling
  constexpr double kM1Fp32PeakGFlops = 2617.0;

  // -------------------------------------------------------------------------
  // 1. Memory Bandwidth Throughput (Scalar vs Float4 vs Float4x2)
  // -------------------------------------------------------------------------
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

    const BenchmarkConfig mem_config{
        .warmup_iterations = 20,
        .samples = 100,
        .iterations_per_sample = 10,
    };
    const double traffic_gb = (3.0 * static_cast<double>(kBytes)) / 1.0e9;

    // 1A. Scalar (float) - 1 element / thread
    int n_arg = kN;
    void* scalar_args[] = {&d_a, &d_b, &d_c, &n_arg};
    const dim3 block_scalar(256u);
    const dim3 grid_scalar(static_cast<unsigned int>((kN + 255) / 256));

    const auto scalar_stats = MeasureGpuDuration(mem_config, [&]() {
      (void)cudaLaunchKernel(stubs.vector_add, grid_scalar, block_scalar, scalar_args, 0u, nullptr);
    });
    const auto scalar_bw = scalar_stats.Transform([=](double s) { return traffic_gb / s; });
    PrintBenchmarkReport("[1] Streaming Memory Bandwidth (N = 16M floats, 192 MiB Traffic)",
                         "Scalar float (1 elem/thread)", mem_config, "GB/s", scalar_bw,
                         "GPU Kernel Execution (Coalesced 32-lane SIMD)");
    suite_results.records.push_back({
        .suite = "Throughput - Streaming Bandwidth",
        .implementation = "Scalar float (1 elem/thread)",
        .timing_domain = "GPU Kernel Execution (Coalesced 32-lane SIMD)",
        .metric_unit = "GB/s",
        .config = mem_config,
        .stats = scalar_bw,
    });

    // 1B. Vectorized (float4) - 4 elements / thread
    constexpr int kN4 = kN / 4;
    int n4_arg = kN4;
    void* vec4_args[] = {&d_a, &d_b, &d_c, &n4_arg};
    const dim3 block_vec4(256u);
    const dim3 grid_vec4(static_cast<unsigned int>((kN4 + 255) / 256));

    const auto vec4_stats = MeasureGpuDuration(mem_config, [&]() {
      (void)cudaLaunchKernel(stubs.vector_add_float4, grid_vec4, block_vec4, vec4_args, 0u,
                             nullptr);
    });
    const auto vec4_bw = vec4_stats.Transform([=](double s) { return traffic_gb / s; });
    PrintBenchmarkReport("[1] Streaming Memory Bandwidth (N = 16M floats, 192 MiB Traffic)",
                         "Vectorized float4 (4 elem/thread)", mem_config, "GB/s", vec4_bw,
                         "GPU Kernel Execution (128-bit Vector Load/Store)");
    suite_results.records.push_back({
        .suite = "Throughput - Streaming Bandwidth",
        .implementation = "Vectorized float4 (4 elem/thread)",
        .timing_domain = "GPU Kernel Execution (128-bit Vector Load/Store)",
        .metric_unit = "GB/s",
        .config = mem_config,
        .stats = vec4_bw,
    });

    // 1C. Multi-Vector Unrolled (float4x2) - 8 elements / thread
    const dim3 block_vec8(256u);
    const dim3 grid_vec8(static_cast<unsigned int>((kN4 / 2 + 255) / 256));

    const auto vec8_stats = MeasureGpuDuration(mem_config, [&]() {
      (void)cudaLaunchKernel(stubs.vector_add_float4x2, grid_vec8, block_vec8, vec4_args, 0u,
                             nullptr);
    });
    const auto vec8_bw = vec8_stats.Transform([=](double s) { return traffic_gb / s; });
    PrintBenchmarkReport("[1] Streaming Memory Bandwidth (N = 16M floats, 192 MiB Traffic)",
                         "Multi-Vector float4x2 (8 elem/thread)", mem_config, "GB/s", vec8_bw,
                         "GPU Kernel Execution (2x 128-bit Unrolled)");
    suite_results.records.push_back({
        .suite = "Throughput - Streaming Bandwidth",
        .implementation = "Multi-Vector float4x2 (8 elem/thread)",
        .timing_domain = "GPU Kernel Execution (2x 128-bit Unrolled)",
        .metric_unit = "GB/s",
        .config = mem_config,
        .stats = vec8_bw,
    });

    std::printf(">> Bandwidth Summary:\n");
    std::printf("   Theoretical Hardware Peak : %6.2f GB/s (Apple M1 128-bit LPDDR4X-4266)\n",
                kM1DramPeakGBps);
    std::printf("   Estimated Physical Ceiling: %6.2f GB/s (~94.5%% bus turnaround limit)\n",
                kM1EstimatedPhysicalCeilingGBps);
    std::printf(
        "   Scalar float              : %6.2f GB/s (%4.1f%% theoretical | %4.1f%% physical "
        "ceiling)\n",
        scalar_bw.median_val, (scalar_bw.median_val / kM1DramPeakGBps) * 100.0,
        (scalar_bw.median_val / kM1EstimatedPhysicalCeilingGBps) * 100.0);
    std::printf(
        "   Vector float4             : %6.2f GB/s (%4.1f%% theoretical | %4.1f%% physical "
        "ceiling)\n",
        vec4_bw.median_val, (vec4_bw.median_val / kM1DramPeakGBps) * 100.0,
        (vec4_bw.median_val / kM1EstimatedPhysicalCeilingGBps) * 100.0);
    std::printf(
        "   Vector float4x2           : %6.2f GB/s (%4.1f%% theoretical | %4.1f%% physical "
        "ceiling)\n",
        vec8_bw.median_val, (vec8_bw.median_val / kM1DramPeakGBps) * 100.0,
        (vec8_bw.median_val / kM1EstimatedPhysicalCeilingGBps) * 100.0);
    std::printf(
        "   Insight: On Apple Silicon, 32-lane SIMD hardware coalesces contiguous 4-byte\n"
        "   scalar accesses into 128-byte DRAM cacheline transactions. Vector float4 achieves\n"
        "   %.1f%% of theoretical DRAM peak and %.1f%% of achievable physical saturation.\n\n",
        (vec4_bw.median_val / kM1DramPeakGBps) * 100.0,
        (vec4_bw.median_val / kM1EstimatedPhysicalCeilingGBps) * 100.0);
    std::printf("-----------------------------------------------------------------\n\n");

    (void)cudaFree(d_c);
    (void)cudaFree(d_b);
    (void)cudaFree(d_a);
  }

  // -------------------------------------------------------------------------
  // 2. Compute Throughput: GEMM Scaling (N = 256, 512, 1024, 2048, 4096)
  // -------------------------------------------------------------------------
  const std::vector<int> gemm_dimensions = {256, 512, 1024, 2048, 4096};

  for (int n : gemm_dimensions) {
    const std::size_t kBytes = static_cast<std::size_t>(n * n) * sizeof(float);
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    (void)cudaMalloc(reinterpret_cast<void**>(&d_a), kBytes);
    (void)cudaMalloc(reinterpret_cast<void**>(&d_b), kBytes);
    (void)cudaMalloc(reinterpret_cast<void**>(&d_c), kBytes);
    (void)cudaMemset(d_a, 0, kBytes);
    (void)cudaMemset(d_b, 0, kBytes);

    int n_arg = n;
    void* args[] = {&d_a, &d_b, &d_c, &n_arg};
    const double gflops_total = (2.0 * static_cast<double>(n) * n * n) / 1.0e9;

    const BenchmarkConfig gemm_config{
        .warmup_iterations = (n >= 4096) ? 3u : ((n >= 2048) ? 5u : 20u),
        .samples = (n >= 4096) ? 20u : ((n >= 2048) ? 50u : 100u),
        .iterations_per_sample = (n >= 4096) ? 1u : ((n >= 2048) ? 3u : 10u),
    };

    // 2A. 16x16 Shared-Memory Tiled
    const dim3 block_tiled(16u, 16u, 1u);
    const dim3 grid_tiled(static_cast<unsigned int>((n + 15) / 16),
                          static_cast<unsigned int>((n + 15) / 16), 1u);

    const auto tiled_stats = MeasureGpuDuration(gemm_config, [&]() {
      (void)cudaLaunchKernel(stubs.gemm_tiled, grid_tiled, block_tiled, args, 0u, nullptr);
    });
    const auto tiled_gflops = tiled_stats.Transform([=](double s) { return gflops_total / s; });

    const std::string title_tiled =
        "[2] Compute Throughput (FP32 GEMM " + std::to_string(n) + "x" + std::to_string(n) + ")";
    PrintBenchmarkReport(title_tiled, "16x16 Shared-Memory Tiled", gemm_config, "GFLOP/s",
                         tiled_gflops, "GPU Hardware Execution");
    suite_results.records.push_back({
        .suite = "Throughput - Compute GEMM",
        .implementation = "16x16 Shared-Tiled (N=" + std::to_string(n) + ")",
        .timing_domain = "GPU Hardware Execution",
        .metric_unit = "GFLOP/s",
        .config = gemm_config,
        .stats = tiled_gflops,
    });

    // 2B. 64x64 Register-Tiled (Single-Buffered)
    const dim3 block_reg(16u, 16u, 1u);
    const dim3 grid_reg(static_cast<unsigned int>((n + 63) / 64),
                        static_cast<unsigned int>((n + 63) / 64), 1u);

    const auto reg_stats = MeasureGpuDuration(gemm_config, [&]() {
      (void)cudaLaunchKernel(stubs.gemm_register_tiled, grid_reg, block_reg, args, 0u, nullptr);
    });
    const auto reg_gflops = reg_stats.Transform([=](double s) { return gflops_total / s; });

    const std::string title_reg =
        "[2] Compute Throughput (FP32 GEMM " + std::to_string(n) + "x" + std::to_string(n) + ")";
    PrintBenchmarkReport(title_reg, "64x64 Register-Tiled (Single-Buffered)", gemm_config,
                         "GFLOP/s", reg_gflops, "GPU Hardware Execution");
    suite_results.records.push_back({
        .suite = "Throughput - Compute GEMM",
        .implementation = "64x64 Register-Tiled (N=" + std::to_string(n) + ")",
        .timing_domain = "GPU Hardware Execution",
        .metric_unit = "GFLOP/s",
        .config = gemm_config,
        .stats = reg_gflops,
    });

    // 2C. 64x64 Multistage Double-Buffered Register-Tiled
    BenchmarkStatistics multi_gflops{};
    if (stubs.gemm_multistage != nullptr) {
      const auto multi_stats = MeasureGpuDuration(gemm_config, [&]() {
        (void)cudaLaunchKernel(stubs.gemm_multistage, grid_reg, block_reg, args, 0u, nullptr);
      });
      multi_gflops = multi_stats.Transform([=](double s) { return gflops_total / s; });

      PrintBenchmarkReport(title_reg, "64x64 Multistage (Double-Buffered)", gemm_config, "GFLOP/s",
                           multi_gflops, "GPU Hardware Execution");
      suite_results.records.push_back({
          .suite = "Throughput - Compute GEMM",
          .implementation = "64x64 Multistage Double-Buffered (N=" + std::to_string(n) + ")",
          .timing_domain = "GPU Hardware Execution",
          .metric_unit = "GFLOP/s",
          .config = gemm_config,
          .stats = multi_gflops,
      });
    }

    const double reg_speedup = reg_gflops.median_val / tiled_gflops.median_val;
    std::printf(">> GEMM N = %4d Scaling Summary (Apple M1 FP32 Peak = %.1f GFLOP/s):\n", n,
                kM1Fp32PeakGFlops);
    std::printf("   16x16 Shared-Tiled     : %6.2f GFLOP/s (%4.1f%% FP32 peak)\n",
                tiled_gflops.median_val, (tiled_gflops.median_val / kM1Fp32PeakGFlops) * 100.0);
    std::printf(
        "   64x64 Register-Tiled   : %6.2f GFLOP/s (%4.1f%% FP32 peak) [%.2fx over shared]\n",
        reg_gflops.median_val, (reg_gflops.median_val / kM1Fp32PeakGFlops) * 100.0, reg_speedup);
    if (stubs.gemm_multistage != nullptr) {
      const double multi_speedup_shared = multi_gflops.median_val / tiled_gflops.median_val;
      const double multi_speedup_reg = multi_gflops.median_val / reg_gflops.median_val;
      std::printf(
          "   64x64 Double-Buffered  : %6.2f GFLOP/s (%4.1f%% FP32 peak) [%.2fx over shared, %.2fx "
          "over reg]\n",
          multi_gflops.median_val, (multi_gflops.median_val / kM1Fp32PeakGFlops) * 100.0,
          multi_speedup_shared, multi_speedup_reg);
    }
    std::printf("-----------------------------------------------------------------\n\n");

    (void)cudaFree(d_c);
    (void)cudaFree(d_b);
    (void)cudaFree(d_a);
  }

  return suite_results;
}

}  // namespace muda::bench

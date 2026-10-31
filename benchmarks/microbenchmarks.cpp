/**
 * @file microbenchmarks.cpp
 * @brief Microbenchmark suite implementation measuring launch latency, transfer mechanics, and
 * synchronization.
 */

#include "microbenchmarks.hpp"

#include <cstdint>
#include <cstdio>
#include <vector>

#include "cuda_runtime.h"

namespace muda::bench {

MicrobenchmarkSuiteResults RunMicrobenchmarks(const void* stub_noop,
                                              [[maybe_unused]] const BenchmarkConfig& base_config) {
  MicrobenchmarkSuiteResults suite_results;

  std::printf("=================================================================\n");
  std::printf("               MUDA MICROBENCHMARK SUITE                         \n");
  std::printf("=================================================================\n\n");

  // -------------------------------------------------------------------------
  // 1. Launch Latency Microbenchmarks
  // -------------------------------------------------------------------------
  {
    float* d_out = nullptr;
    (void)cudaMalloc(reinterpret_cast<void**>(&d_out), sizeof(float));
    void* args[] = {&d_out};

    const BenchmarkConfig launch_config{
        .warmup_iterations = 50,
        .samples = 100,
        .iterations_per_sample = 25,
    };

    // 1A. Host CPU Enqueue / Encode Latency (Amortized inside batch, zero flush)
    // Synchronizing in setup and teardown ensures each sample begins with active_disp = 0.
    // Since 25 < kMaxBatchDispatches (64), exactly 0 command buffer flushes occur in the timed
    // loop.
    const auto enqueue_stats = MeasureCpuDurationWithSetup(
        launch_config,
        [&]() { (void)cudaLaunchKernel(stub_noop, dim3(1u), dim3(1u), args, 0u, nullptr); },
        [&]() { (void)cudaDeviceSynchronize(); }, [&]() { (void)cudaDeviceSynchronize(); });

    const auto enqueue_us = enqueue_stats.Transform([](double s) { return s * 1.0e6; });
    PrintBenchmarkReport("[1] Launch Latency (Empty Kernel)", "CPU Enqueue / Encode Latency",
                         launch_config, "us / launch", enqueue_us,
                         "Host Argument Binding & Command Encoder Recording");
    suite_results.records.push_back({
        .suite = "Microbenchmark - Launch Latency",
        .implementation = "CPU Enqueue / Encode Latency",
        .timing_domain = "Host Argument Binding & Command Encoder Recording",
        .metric_unit = "us / launch",
        .config = launch_config,
        .stats = enqueue_us,
    });

    // 1B. Command Buffer Commit Latency (Ending encoding & committing to queue)
    // Uses per-iteration setup to ensure each commit is preceded by an active dispatch,
    // and teardown synchronizes the stream to eliminate command queue backpressure and lock stalls.
    cudaStream_t stream = nullptr;
    (void)cudaStreamCreate(&stream);
    cudaEvent_t ev_commit = nullptr;
    (void)cudaEventCreate(&ev_commit);

    const auto commit_stats = MeasureCpuDurationWithPerIterationSetup(
        launch_config,
        [&]() {
          // Recording an event forces FlushStreamLocked, ending encoding & committing command
          // buffer
          (void)cudaEventRecord(ev_commit, stream);
        },
        [&]() {
          // Pre-encode kernel into stream so active command buffer exists
          (void)cudaLaunchKernel(stub_noop, dim3(1u), dim3(1u), args, 0u, stream);
        },
        [&]() {
          // Drain stream so queue depth stays at 1 and prevents driver worker thread contention
          (void)cudaStreamSynchronize(stream);
        });

    const auto commit_us = commit_stats.Transform([](double s) { return s * 1.0e6; });
    PrintBenchmarkReport("[1] Launch Latency (Empty Kernel)", "Command Buffer Commit Latency",
                         launch_config, "us / commit", commit_us,
                         "Host Command Buffer Submission to Queue");
    suite_results.records.push_back({
        .suite = "Microbenchmark - Launch Latency",
        .implementation = "Command Buffer Commit Latency",
        .timing_domain = "Host Command Buffer Submission to Queue",
        .metric_unit = "us / commit",
        .config = launch_config,
        .stats = commit_us,
    });

    (void)cudaEventDestroy(ev_commit);

    // 1C1. GPU Hardware Execution Duration (Recorded on same stream)
    const auto gpu_exec_stats = MeasureGpuDuration(
        launch_config,
        [&]() { (void)cudaLaunchKernel(stub_noop, dim3(1u), dim3(1u), args, 0u, stream); }, stream);

    const auto gpu_exec_us = gpu_exec_stats.Transform([](double s) { return s * 1.0e6; });
    PrintBenchmarkReport("[1] Launch Latency (Empty Kernel)", "GPU Execution Duration",
                         launch_config, "us / launch", gpu_exec_us,
                         "GPU Hardware Pipeline & Kernel Execution (Same-Stream Timed)");
    suite_results.records.push_back({
        .suite = "Microbenchmark - Launch Latency",
        .implementation = "GPU Execution Duration",
        .timing_domain = "GPU Hardware Pipeline & Kernel Execution (Same-Stream Timed)",
        .metric_unit = "us / launch",
        .config = launch_config,
        .stats = gpu_exec_us,
    });

    // 1C2. GPU Dispatch Overhead (Marginal Delta: 60 dispatches vs 10 dispatches in single CB)
    // Isolates marginal per-dispatch cost by subtracting command buffer setup and completion
    // overhead.
    cudaEvent_t ev_d1 = nullptr;
    cudaEvent_t ev_d2 = nullptr;
    (void)cudaEventCreate(&ev_d1);
    (void)cudaEventCreate(&ev_d2);

    std::vector<double> dispatch_delta_samples;
    dispatch_delta_samples.reserve(launch_config.samples);

    for (std::size_t s = 0; s < launch_config.samples; ++s) {
      // 10 launches
      (void)cudaEventRecord(ev_d1, stream);
      for (int i = 0; i < 10; ++i) {
        (void)cudaLaunchKernel(stub_noop, dim3(1u), dim3(1u), args, 0u, stream);
      }
      (void)cudaEventRecord(ev_d2, stream);
      (void)cudaEventSynchronize(ev_d2);
      float ms_10 = 0.0f;
      (void)cudaEventElapsedTime(&ms_10, ev_d1, ev_d2);

      // 60 launches
      (void)cudaEventRecord(ev_d1, stream);
      for (int i = 0; i < 60; ++i) {
        (void)cudaLaunchKernel(stub_noop, dim3(1u), dim3(1u), args, 0u, stream);
      }
      (void)cudaEventRecord(ev_d2, stream);
      (void)cudaEventSynchronize(ev_d2);
      float ms_60 = 0.0f;
      (void)cudaEventElapsedTime(&ms_60, ev_d1, ev_d2);

      const double delta_ms = static_cast<double>(ms_60 - ms_10);
      const double per_dispatch_us = (delta_ms > 0.0) ? (delta_ms * 1000.0 / 50.0) : 0.0;
      dispatch_delta_samples.push_back(per_dispatch_us);
    }

    (void)cudaEventDestroy(ev_d2);
    (void)cudaEventDestroy(ev_d1);

    const auto dispatch_stats = ComputeStatistics(std::move(dispatch_delta_samples));
    PrintBenchmarkReport("[1] Launch Latency (Empty Kernel)", "GPU Marginal Dispatch Overhead",
                         launch_config, "us / launch", dispatch_stats,
                         "GPU Hardware Dispatch Slope (50-Kernel Differential)");
    suite_results.records.push_back({
        .suite = "Microbenchmark - Launch Latency",
        .implementation = "GPU Marginal Dispatch Overhead",
        .timing_domain = "GPU Hardware Dispatch Slope (50-Kernel Differential)",
        .metric_unit = "us / launch",
        .config = launch_config,
        .stats = dispatch_stats,
    });

    // 1D. Synchronous Host Round-Trip Latency (Single launch + host sync per iteration)
    const BenchmarkConfig sync_config{
        .warmup_iterations = 20,
        .samples = 100,
        .iterations_per_sample = 20,
    };
    const auto sync_stats = MeasureCpuDuration(
        sync_config,
        [&]() {
          (void)cudaLaunchKernel(stub_noop, dim3(1u), dim3(1u), args, 0u, stream);
          (void)cudaStreamSynchronize(stream);
        },
        [&]() {});

    const auto sync_us = sync_stats.Transform([](double s) { return s * 1.0e6; });
    PrintBenchmarkReport("[1] Launch Latency (Empty Kernel)", "Synchronous Host Round-Trip",
                         sync_config, "us / launch", sync_us,
                         "Host-Blocking End-to-End Round-Trip");
    suite_results.records.push_back({
        .suite = "Microbenchmark - Launch Latency",
        .implementation = "Synchronous Host Round-Trip",
        .timing_domain = "Host-Blocking End-to-End Round-Trip",
        .metric_unit = "us / launch",
        .config = sync_config,
        .stats = sync_us,
    });

    (void)cudaStreamDestroy(stream);
    (void)cudaFree(d_out);
  }

  // -------------------------------------------------------------------------
  // 2. Memory Transfer Microbenchmarks
  // -------------------------------------------------------------------------
  {
    constexpr std::size_t kBytes = 64ull * 1024ull * 1024ull;
    std::vector<std::uint8_t> host_buf(kBytes, 0x5Au);
    void* d_a = nullptr;
    void* d_b = nullptr;
    (void)cudaMalloc(&d_a, kBytes);
    (void)cudaMalloc(&d_b, kBytes);

    const BenchmarkConfig copy_config{
        .warmup_iterations = 10,
        .samples = 100,
        .iterations_per_sample = 5,
    };
    const double total_gb = static_cast<double>(kBytes) / 1.0e9;

    // 2A. CPU-to-CPU unified copy (Host-to-Device via memmove)
    const auto h2d = MeasureCpuDuration(
        copy_config,
        [&]() { (void)cudaMemcpy(d_a, host_buf.data(), kBytes, cudaMemcpyHostToDevice); },
        [&]() {});
    const auto h2d_bw = h2d.Transform([=](double s) { return total_gb / s; });
    PrintBenchmarkReport("[2] Memory Transfer Bandwidth (64 MiB)",
                         "CPU-to-CPU unified copy (Host-to-Device)", copy_config, "GB/s", h2d_bw,
                         "Host CPU memmove to Unified Device Buffer");
    suite_results.records.push_back({
        .suite = "Microbenchmark - Memory Transfer",
        .implementation = "CPU-to-CPU unified copy (Host-to-Device)",
        .timing_domain = "Host CPU memmove to Unified Device Buffer",
        .metric_unit = "GB/s",
        .config = copy_config,
        .stats = h2d_bw,
    });

    // 2B. CPU-to-CPU unified copy (Device-to-Host via memmove)
    const auto d2h = MeasureCpuDuration(
        copy_config,
        [&]() { (void)cudaMemcpy(host_buf.data(), d_b, kBytes, cudaMemcpyDeviceToHost); },
        [&]() {});
    const auto d2h_bw = d2h.Transform([=](double s) { return total_gb / s; });
    PrintBenchmarkReport("[2] Memory Transfer Bandwidth (64 MiB)",
                         "CPU-to-CPU unified copy (Device-to-Host)", copy_config, "GB/s", d2h_bw,
                         "Host CPU memmove from Unified Device Buffer");
    suite_results.records.push_back({
        .suite = "Microbenchmark - Memory Transfer",
        .implementation = "CPU-to-CPU unified copy (Device-to-Host)",
        .timing_domain = "Host CPU memmove from Unified Device Buffer",
        .metric_unit = "GB/s",
        .config = copy_config,
        .stats = d2h_bw,
    });

    // 2C. GPU blit-engine copy (Device-to-Device via MTLBlitCommandEncoder)
    const auto d2d_gpu = MeasureGpuDuration(copy_config, [&]() {
      (void)cudaMemcpyAsync(d_b, d_a, kBytes, cudaMemcpyDeviceToDevice, nullptr);
    });
    const auto d2d_gpu_bw = d2d_gpu.Transform([=](double s) { return total_gb / s; });
    PrintBenchmarkReport("[2] Memory Transfer Bandwidth (64 MiB)",
                         "GPU blit-engine copy (Device-to-Device)", copy_config, "GB/s", d2d_gpu_bw,
                         "Metal MTLBlitCommandEncoder DMA Transfer (Hardware Timed)");
    suite_results.records.push_back({
        .suite = "Microbenchmark - Memory Transfer",
        .implementation = "GPU blit-engine copy (Device-to-Device)",
        .timing_domain = "Metal MTLBlitCommandEncoder DMA Transfer (Hardware Timed)",
        .metric_unit = "GB/s",
        .config = copy_config,
        .stats = d2d_gpu_bw,
    });

    // 2D. Host-blocking copy (GPU blit + Host Sync)
    const auto d2d_sync = MeasureCpuDuration(
        copy_config, [&]() { (void)cudaMemcpy(d_b, d_a, kBytes, cudaMemcpyDeviceToDevice); },
        [&]() {});
    const auto d2d_sync_bw = d2d_sync.Transform([=](double s) { return total_gb / s; });
    PrintBenchmarkReport("[2] Memory Transfer Bandwidth (64 MiB)",
                         "Host-blocking copy (GPU blit + Host Sync)", copy_config, "GB/s",
                         d2d_sync_bw, "Synchronous Host-Blocking Blit (Includes Commit & Wait)");
    suite_results.records.push_back({
        .suite = "Microbenchmark - Memory Transfer",
        .implementation = "Host-blocking copy (GPU blit + Host Sync)",
        .timing_domain = "Synchronous Host-Blocking Blit (Includes Commit & Wait)",
        .metric_unit = "GB/s",
        .config = copy_config,
        .stats = d2d_sync_bw,
    });

    // 2E. CPU sequential read (Cold DRAM, 64 MiB)
    const auto zc_dram = MeasureCpuDuration(
        copy_config,
        [&]() {
          const auto* ptr = static_cast<const std::uint64_t*>(d_a);
          const std::size_t count = kBytes / sizeof(std::uint64_t);
          std::uint64_t sum0 = 0;
          std::uint64_t sum1 = 0;
          std::uint64_t sum2 = 0;
          std::uint64_t sum3 = 0;
          for (std::size_t i = 0; i < count; i += 4) {
            sum0 += ptr[i];
            sum1 += ptr[i + 1];
            sum2 += ptr[i + 2];
            sum3 += ptr[i + 3];
          }
          const volatile std::uint64_t sum = sum0 + sum1 + sum2 + sum3;
          asm volatile("" : : "r"(sum) : "memory");
        },
        [&]() {});
    const auto zc_dram_bw = zc_dram.Transform([=](double s) { return total_gb / s; });
    PrintBenchmarkReport("[2] Memory Transfer Bandwidth (64 MiB)",
                         "CPU sequential read (Cold DRAM, 64 MiB)", copy_config, "GB/s", zc_dram_bw,
                         "Host CPU 64-bit Sequential Read of Unified Memory");
    suite_results.records.push_back({
        .suite = "Microbenchmark - Memory Transfer",
        .implementation = "CPU sequential read (Cold DRAM, 64 MiB)",
        .timing_domain = "Host CPU 64-bit Sequential Read of Unified Memory",
        .metric_unit = "GB/s",
        .config = copy_config,
        .stats = zc_dram_bw,
    });

    // 2F. CPU sequential read (Warm L2 Cache, 4 MiB)
    constexpr std::size_t kCacheBytes = 4ull * 1024ull * 1024ull;
    const double cache_gb = static_cast<double>(kCacheBytes) / 1.0e9;
    const BenchmarkConfig cache_config{
        .warmup_iterations = 20,
        .samples = 100,
        .iterations_per_sample = 25,
    };
    const auto zc_cache = MeasureCpuDuration(
        cache_config,
        [&]() {
          const auto* ptr = static_cast<const std::uint64_t*>(d_a);
          const std::size_t count = kCacheBytes / sizeof(std::uint64_t);
          std::uint64_t sum0 = 0;
          std::uint64_t sum1 = 0;
          std::uint64_t sum2 = 0;
          std::uint64_t sum3 = 0;
          for (std::size_t i = 0; i < count; i += 4) {
            sum0 += ptr[i];
            sum1 += ptr[i + 1];
            sum2 += ptr[i + 2];
            sum3 += ptr[i + 3];
          }
          const volatile std::uint64_t sum = sum0 + sum1 + sum2 + sum3;
          asm volatile("" : : "r"(sum) : "memory");
        },
        [&]() {});
    const auto zc_cache_bw = zc_cache.Transform([=](double s) { return cache_gb / s; });
    PrintBenchmarkReport("[2] Memory Transfer Bandwidth (4 MiB)",
                         "CPU sequential read (Warm L2 Cache, 4 MiB)", cache_config, "GB/s",
                         zc_cache_bw, "Host CPU 64-bit Sequential Read of Cache-Resident Buffer");
    suite_results.records.push_back({
        .suite = "Microbenchmark - Memory Transfer",
        .implementation = "CPU sequential read (Warm L2 Cache, 4 MiB)",
        .timing_domain = "Host CPU 64-bit Sequential Read of Cache-Resident Buffer",
        .metric_unit = "GB/s",
        .config = cache_config,
        .stats = zc_cache_bw,
    });

    (void)cudaFree(d_b);
    (void)cudaFree(d_a);
  }

  // -------------------------------------------------------------------------
  // 3. Synchronization Microbenchmarks
  // -------------------------------------------------------------------------
  {
    cudaEvent_t ev = nullptr;
    (void)cudaEventCreate(&ev);

    const BenchmarkConfig sync_bench_config{
        .warmup_iterations = 20,
        .samples = 100,
        .iterations_per_sample = 25,
    };

    const auto event_sync_stats = MeasureCpuDuration(
        sync_bench_config,
        [&]() {
          (void)cudaEventRecord(ev, nullptr);
          (void)cudaEventSynchronize(ev);
        },
        [&]() {});
    const auto event_sync_us = event_sync_stats.Transform([](double s) { return s * 1.0e6; });
    PrintBenchmarkReport("[3] Synchronization Latency", "Event Record + Synchronize",
                         sync_bench_config, "us / call", event_sync_us,
                         "Host Event Signal & Completion Sync");
    suite_results.records.push_back({
        .suite = "Microbenchmark - Synchronization",
        .implementation = "Event Record + Synchronize",
        .timing_domain = "Host Event Signal & Completion Sync",
        .metric_unit = "us / call",
        .config = sync_bench_config,
        .stats = event_sync_us,
    });

    (void)cudaEventDestroy(ev);

    // 3B. Idle Stream Synchronize (No-Op Fast Path)
    const auto idle_sync_stats =
        MeasureCpuDuration(sync_bench_config, [&]() { (void)cudaDeviceSynchronize(); }, [&]() {});
    const auto idle_sync_us = idle_sync_stats.Transform([](double s) { return s * 1.0e6; });
    PrintBenchmarkReport("[3] Synchronization Latency", "Idle Device Synchronize (Fast Path)",
                         sync_bench_config, "us / call", idle_sync_us,
                         "Host Device Synchronize (Zero Pending Work Fast Path)");
    suite_results.records.push_back({
        .suite = "Microbenchmark - Synchronization",
        .implementation = "Idle Device Synchronize (Fast Path)",
        .timing_domain = "Host Device Synchronize (Zero Pending Work Fast Path)",
        .metric_unit = "us / call",
        .config = sync_bench_config,
        .stats = idle_sync_us,
    });

    // 3C. Active Workload Device Synchronize (Measures host wait for pending GPU work)
    float* d_sync = nullptr;
    (void)cudaMalloc(reinterpret_cast<void**>(&d_sync), sizeof(float));
    void* sync_args[] = {&d_sync};

    const auto active_sync_stats = MeasureCpuDurationWithPerIterationSetup(
        sync_bench_config,
        [&]() {
          // Timed operation: wait for pending kernel to finish
          (void)cudaDeviceSynchronize();
        },
        [&]() {
          // Setup: dispatch active GPU kernel
          (void)cudaLaunchKernel(stub_noop, dim3(1u), dim3(1u), sync_args, 0u, nullptr);
        },
        [&]() {});

    const auto active_sync_us = active_sync_stats.Transform([](double s) { return s * 1.0e6; });
    PrintBenchmarkReport("[3] Synchronization Latency", "Active Workload Device Synchronize",
                         sync_bench_config, "us / call", active_sync_us,
                         "Host Wait for Active GPU Work Completion & Thread Wakeup");
    suite_results.records.push_back({
        .suite = "Microbenchmark - Synchronization",
        .implementation = "Active Workload Device Synchronize",
        .timing_domain = "Host Wait for Active GPU Work Completion & Thread Wakeup",
        .metric_unit = "us / call",
        .config = sync_bench_config,
        .stats = active_sync_us,
    });

    (void)cudaFree(d_sync);
  }

  return suite_results;
}

}  // namespace muda::bench

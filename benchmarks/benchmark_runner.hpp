/**
 * @file benchmark_runner.hpp
 * @brief Statistical benchmark runner for MUDA measuring latency, bandwidth, and throughput.
 */

#pragma once

#include <chrono>
#include <cstddef>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include "cuda_runtime.h"

namespace muda::bench {

using Clock = std::chrono::steady_clock;

struct BenchmarkConfig {
  std::size_t warmup_iterations = 20;
  std::size_t samples = 100;
  std::size_t iterations_per_sample = 25;
};

struct BenchmarkStatistics {
  std::vector<double> samples;
  double min_val = 0.0;
  double max_val = 0.0;
  double median_val = 0.0;
  double mad_val = 0.0;   // Median Absolute Deviation
  double nmad_val = 0.0;  // Normalized MAD (1.4826 * MAD)
  double mean_val = 0.0;
  double stddev_val = 0.0;
  double cv_percent = 0.0;
  double p90_val = 0.0;
  double p95_val = 0.0;
  double p99_val = 0.0;
  double ci95_low = 0.0;          // 95% Confidence Interval (bootstrap lower)
  double ci95_high = 0.0;         // 95% Confidence Interval (bootstrap upper)
  std::size_t outlier_count = 0;  // Count of samples outside median +/- 3 * NMAD

  [[nodiscard]] const char* Status() const noexcept;
  [[nodiscard]] const char* DistributionClass() const noexcept;
  [[nodiscard]] BenchmarkStatistics Transform(const std::function<double(double)>& func) const;
};

struct BenchmarkResultRecord {
  std::string suite;
  std::string implementation;
  std::string timing_domain;
  std::string metric_unit;
  BenchmarkConfig config;
  BenchmarkStatistics stats;
};

[[nodiscard]] BenchmarkStatistics ComputeStatistics(std::vector<double> raw_samples);

void PrintBenchmarkReport(std::string_view section_title, std::string_view implementation,
                          const BenchmarkConfig& config, std::string_view metric_unit,
                          const BenchmarkStatistics& stats,
                          std::string_view timing_domain = "GPU Execution");

void ExportStatisticsCsv(std::string_view file_path,
                         const std::vector<BenchmarkResultRecord>& results);
void ExportStatisticsJson(std::string_view file_path,
                          const std::vector<BenchmarkResultRecord>& results);

template <typename KernelFn, typename SetupFn, typename TeardownFn>
BenchmarkStatistics MeasureCpuDurationWithSetup(const BenchmarkConfig& config, KernelFn&& kernel_fn,
                                                SetupFn&& setup_fn, TeardownFn&& teardown_fn) {
  for (std::size_t i = 0; i < config.warmup_iterations; ++i) {
    setup_fn();
    kernel_fn();
    teardown_fn();
  }

  std::vector<double> sample_durations_sec;
  sample_durations_sec.reserve(config.samples);

  for (std::size_t s = 0; s < config.samples; ++s) {
    setup_fn();
    const auto t0 = Clock::now();
    for (std::size_t iter = 0; iter < config.iterations_per_sample; ++iter) {
      kernel_fn();
    }
    const auto t1 = Clock::now();
    teardown_fn();

    const double total_sec = std::chrono::duration<double>(t1 - t0).count();
    sample_durations_sec.push_back(total_sec / static_cast<double>(config.iterations_per_sample));
  }

  return ComputeStatistics(std::move(sample_durations_sec));
}

template <typename KernelFn, typename SetupFn, typename TeardownFn>
BenchmarkStatistics MeasureCpuDurationWithPerIterationSetup(const BenchmarkConfig& config,
                                                            KernelFn&& kernel_fn,
                                                            SetupFn&& setup_fn,
                                                            TeardownFn&& teardown_fn) {
  for (std::size_t i = 0; i < config.warmup_iterations; ++i) {
    setup_fn();
    kernel_fn();
    teardown_fn();
  }

  std::vector<double> sample_durations_sec;
  sample_durations_sec.reserve(config.samples);

  for (std::size_t s = 0; s < config.samples; ++s) {
    double sample_total_sec = 0.0;
    for (std::size_t iter = 0; iter < config.iterations_per_sample; ++iter) {
      setup_fn();
      const auto t0 = Clock::now();
      kernel_fn();
      const auto t1 = Clock::now();
      teardown_fn();
      sample_total_sec += std::chrono::duration<double>(t1 - t0).count();
    }
    sample_durations_sec.push_back(sample_total_sec /
                                   static_cast<double>(config.iterations_per_sample));
  }

  return ComputeStatistics(std::move(sample_durations_sec));
}

template <typename KernelFn, typename SyncFn>
BenchmarkStatistics MeasureCpuDuration(const BenchmarkConfig& config, KernelFn&& kernel_fn,
                                       SyncFn&& sync_fn) {
  return MeasureCpuDurationWithSetup(
      config, std::forward<KernelFn>(kernel_fn), [&]() {}, std::forward<SyncFn>(sync_fn));
}

template <typename KernelFn>
BenchmarkStatistics MeasureGpuDuration(const BenchmarkConfig& config, KernelFn&& kernel_fn,
                                       cudaStream_t stream = nullptr) {
  for (std::size_t i = 0; i < config.warmup_iterations; ++i) {
    kernel_fn();
  }
  if (stream != nullptr) {
    (void)cudaStreamSynchronize(stream);
  } else {
    (void)cudaDeviceSynchronize();
  }

  cudaEvent_t ev_start = nullptr;
  cudaEvent_t ev_stop = nullptr;
  (void)cudaEventCreate(&ev_start);
  (void)cudaEventCreate(&ev_stop);

  std::vector<double> sample_durations_sec;
  sample_durations_sec.reserve(config.samples);

  for (std::size_t s = 0; s < config.samples; ++s) {
    (void)cudaEventRecord(ev_start, stream);
    for (std::size_t iter = 0; iter < config.iterations_per_sample; ++iter) {
      kernel_fn();
    }
    (void)cudaEventRecord(ev_stop, stream);
    (void)cudaEventSynchronize(ev_stop);

    float elapsed_ms = 0.0f;
    (void)cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop);
    const double total_sec = static_cast<double>(elapsed_ms) / 1000.0;
    sample_durations_sec.push_back(total_sec / static_cast<double>(config.iterations_per_sample));
  }

  (void)cudaEventDestroy(ev_stop);
  (void)cudaEventDestroy(ev_start);

  return ComputeStatistics(std::move(sample_durations_sec));
}

struct InterleavedTask {
  std::string name;
  std::function<void()> run;
  std::function<void()> setup = []() {};
  std::function<void()> teardown = []() {};
};

std::vector<BenchmarkStatistics> MeasureInterleavedTasks(const BenchmarkConfig& config,
                                                         std::vector<InterleavedTask>& tasks);

}  // namespace muda::bench

/**
 * @file benchmark_runner.cpp
 * @brief Statistical calculation and reporting implementation for MUDA benchmarks.
 */

#include "benchmark_runner.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <random>

namespace muda::bench {
namespace {

[[nodiscard]] double Percentile(const std::vector<double>& sorted, double p) noexcept {
  if (sorted.empty()) {
    return 0.0;
  }
  if (sorted.size() == 1) {
    return sorted.front();
  }
  const double index = p * static_cast<double>(sorted.size() - 1);
  const auto lower = static_cast<std::size_t>(std::floor(index));
  const auto upper = static_cast<std::size_t>(std::ceil(index));
  const double fraction = index - static_cast<double>(lower);
  return sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction;
}

[[nodiscard]] std::pair<double, double> ComputeBootstrapMedianCi95(
    const std::vector<double>& sorted) {
  if (sorted.empty()) {
    return {0.0, 0.0};
  }
  if (sorted.size() == 1) {
    return {sorted.front(), sorted.front()};
  }

  constexpr std::size_t kBootstrapRounds = 2000;
  std::mt19937_64 rng(1337);
  std::uniform_int_distribution<std::size_t> dist(0, sorted.size() - 1);

  std::vector<double> boot_medians;
  boot_medians.reserve(kBootstrapRounds);
  std::vector<double> resample(sorted.size());

  for (std::size_t b = 0; b < kBootstrapRounds; ++b) {
    for (std::size_t i = 0; i < sorted.size(); ++i) {
      resample[i] = sorted[dist(rng)];
    }
    std::sort(resample.begin(), resample.end());
    boot_medians.push_back(Percentile(resample, 0.50));
  }

  std::sort(boot_medians.begin(), boot_medians.end());
  return {Percentile(boot_medians, 0.025), Percentile(boot_medians, 0.975)};
}

}  // namespace

const char* BenchmarkStatistics::Status() const noexcept {
  if (cv_percent < 2.0) {
    return "STABLE";
  }
  if (cv_percent <= 5.0) {
    return "REASONABLE";
  }
  return "HIGH VARIABILITY";
}

const char* BenchmarkStatistics::DistributionClass() const noexcept {
  if (nmad_val > 0.0 && (stddev_val / nmad_val > 2.0)) {
    return "LONG-TAILED";
  }
  if (cv_percent > 20.0) {
    return "DISPERSED";
  }
  return "SYMMETRIC";
}

BenchmarkStatistics BenchmarkStatistics::Transform(
    const std::function<double(double)>& func) const {
  std::vector<double> transformed;
  transformed.reserve(samples.size());
  for (double val : samples) {
    transformed.push_back(func(val));
  }
  return ComputeStatistics(std::move(transformed));
}

BenchmarkStatistics ComputeStatistics(std::vector<double> raw_samples) {
  BenchmarkStatistics stats;
  if (raw_samples.empty()) {
    return stats;
  }

  stats.samples = std::move(raw_samples);
  std::vector<double> sorted = stats.samples;
  std::sort(sorted.begin(), sorted.end());

  const double sum = std::accumulate(sorted.begin(), sorted.end(), 0.0);
  const auto count = static_cast<double>(sorted.size());
  stats.mean_val = sum / count;
  stats.min_val = sorted.front();
  stats.max_val = sorted.back();
  stats.median_val = Percentile(sorted, 0.50);
  stats.p90_val = Percentile(sorted, 0.90);
  stats.p95_val = Percentile(sorted, 0.95);
  stats.p99_val = Percentile(sorted, 0.99);

  if (sorted.size() > 1) {
    double var_sum = 0.0;
    for (double val : sorted) {
      const double diff = val - stats.mean_val;
      var_sum += diff * diff;
    }
    stats.stddev_val = std::sqrt(var_sum / (count - 1.0));
  } else {
    stats.stddev_val = 0.0;
  }

  stats.cv_percent = (stats.mean_val != 0.0) ? (stats.stddev_val / stats.mean_val) * 100.0 : 0.0;

  // Compute Median Absolute Deviation (MAD)
  std::vector<double> abs_devs;
  abs_devs.reserve(sorted.size());
  for (double val : sorted) {
    abs_devs.push_back(std::abs(val - stats.median_val));
  }
  std::sort(abs_devs.begin(), abs_devs.end());
  stats.mad_val = Percentile(abs_devs, 0.50);
  stats.nmad_val = 1.4826 * stats.mad_val;

  // Count Hampel outliers: |val - median| > 3 * NMAD
  if (stats.nmad_val > 0.0) {
    const double threshold = 3.0 * stats.nmad_val;
    for (double dev : abs_devs) {
      if (dev > threshold) {
        stats.outlier_count++;
      }
    }
  }

  // 95% Confidence Interval for median
  const auto [ci_low, ci_high] = ComputeBootstrapMedianCi95(sorted);
  stats.ci95_low = ci_low;
  stats.ci95_high = ci_high;

  return stats;
}

void PrintBenchmarkReport(std::string_view section_title, std::string_view implementation,
                          const BenchmarkConfig& config, std::string_view metric_unit,
                          const BenchmarkStatistics& stats, std::string_view timing_domain) {
  std::printf("%.*s\n\n", static_cast<int>(section_title.size()), section_title.data());
  std::printf("Implementation: %.*s\n", static_cast<int>(implementation.size()),
              implementation.data());
  std::printf("Warmup:         %zu iterations\n", config.warmup_iterations);
  std::printf("Samples:        %zu\n", config.samples);
  std::printf("Iterations:     %zu per sample\n\n", config.iterations_per_sample);

  std::printf("%.*s:\n", static_cast<int>(timing_domain.size()), timing_domain.data());
  std::printf("    Min:        %8.2f %.*s\n", stats.min_val, static_cast<int>(metric_unit.size()),
              metric_unit.data());
  std::printf("    Median:     %8.2f %.*s\n", stats.median_val,
              static_cast<int>(metric_unit.size()), metric_unit.data());
  std::printf("    MAD:        %8.2f %.*s (Normalized: %8.2f)\n", stats.mad_val,
              static_cast<int>(metric_unit.size()), metric_unit.data(), stats.nmad_val);
  std::printf("    95%% CI:     [%8.2f, %8.2f] %.*s\n", stats.ci95_low, stats.ci95_high,
              static_cast<int>(metric_unit.size()), metric_unit.data());
  std::printf("    Mean:       %8.2f %.*s\n", stats.mean_val, static_cast<int>(metric_unit.size()),
              metric_unit.data());
  std::printf("    P90:        %8.2f %.*s\n", stats.p90_val, static_cast<int>(metric_unit.size()),
              metric_unit.data());
  std::printf("    P95:        %8.2f %.*s\n", stats.p95_val, static_cast<int>(metric_unit.size()),
              metric_unit.data());
  std::printf("    P99:        %8.2f %.*s\n", stats.p99_val, static_cast<int>(metric_unit.size()),
              metric_unit.data());
  std::printf("    StdDev:     %8.2f %.*s\n", stats.stddev_val,
              static_cast<int>(metric_unit.size()), metric_unit.data());
  std::printf("    CV:         %8.2f%%\n", stats.cv_percent);
  std::printf("    Outliers:   %zu (outside 3x NMAD)\n", stats.outlier_count);
  std::printf("    Profile:    %s\n\n", stats.DistributionClass());
  std::printf("Status: %s\n", stats.Status());
  std::printf("-----------------------------------------------------------------\n\n");
}

void ExportStatisticsCsv(std::string_view file_path,
                         const std::vector<BenchmarkResultRecord>& results) {
  std::ofstream out(std::string(file_path), std::ios::out | std::ios::trunc);
  if (!out.is_open()) {
    std::fprintf(stderr, "Failed to open CSV output path: %.*s\n",
                 static_cast<int>(file_path.size()), file_path.data());
    return;
  }

  out << "Suite,Implementation,Domain,Unit,Min,Median,MAD,NMAD,CI95_Low,CI95_High,Mean,P90,P95,P99,"
         "StdDev,CV_Percent,Outliers,Profile,Status\n";

  for (const auto& rec : results) {
    out << '"' << rec.suite << "\"," << '"' << rec.implementation << "\"," << '"'
        << rec.timing_domain << "\"," << '"' << rec.metric_unit << "\"," << rec.stats.min_val << ','
        << rec.stats.median_val << ',' << rec.stats.mad_val << ',' << rec.stats.nmad_val << ','
        << rec.stats.ci95_low << ',' << rec.stats.ci95_high << ',' << rec.stats.mean_val << ','
        << rec.stats.p90_val << ',' << rec.stats.p95_val << ',' << rec.stats.p99_val << ','
        << rec.stats.stddev_val << ',' << rec.stats.cv_percent << ',' << rec.stats.outlier_count
        << ',' << '"' << rec.stats.DistributionClass() << "\"," << '"' << rec.stats.Status()
        << "\"\n";
  }
}

void ExportStatisticsJson(std::string_view file_path,
                          const std::vector<BenchmarkResultRecord>& results) {
  std::ofstream out(std::string(file_path), std::ios::out | std::ios::trunc);
  if (!out.is_open()) {
    std::fprintf(stderr, "Failed to open JSON output path: %.*s\n",
                 static_cast<int>(file_path.size()), file_path.data());
    return;
  }

  out << "[\n";
  for (std::size_t i = 0; i < results.size(); ++i) {
    const auto& rec = results[i];
    out << "  {\n"
        << "    \"suite\": \"" << rec.suite << "\",\n"
        << "    \"implementation\": \"" << rec.implementation << "\",\n"
        << "    \"domain\": \"" << rec.timing_domain << "\",\n"
        << "    \"unit\": \"" << rec.metric_unit << "\",\n"
        << "    \"min\": " << rec.stats.min_val << ",\n"
        << "    \"median\": " << rec.stats.median_val << ",\n"
        << "    \"mad\": " << rec.stats.mad_val << ",\n"
        << "    \"nmad\": " << rec.stats.nmad_val << ",\n"
        << "    \"ci95_low\": " << rec.stats.ci95_low << ",\n"
        << "    \"ci95_high\": " << rec.stats.ci95_high << ",\n"
        << "    \"mean\": " << rec.stats.mean_val << ",\n"
        << "    \"p90\": " << rec.stats.p90_val << ",\n"
        << "    \"p95\": " << rec.stats.p95_val << ",\n"
        << "    \"p99\": " << rec.stats.p99_val << ",\n"
        << "    \"stddev\": " << rec.stats.stddev_val << ",\n"
        << "    \"cv_percent\": " << rec.stats.cv_percent << ",\n"
        << "    \"outliers\": " << rec.stats.outlier_count << ",\n"
        << "    \"profile\": \"" << rec.stats.DistributionClass() << "\",\n"
        << "    \"status\": \"" << rec.stats.Status() << "\",\n"
        << "    \"samples\": [";

    for (std::size_t s = 0; s < rec.stats.samples.size(); ++s) {
      out << rec.stats.samples[s] << (s + 1 < rec.stats.samples.size() ? ", " : "");
    }
    out << "]\n  }" << (i + 1 < results.size() ? ",\n" : "\n");
  }
  out << "]\n";
}

std::vector<BenchmarkStatistics> MeasureInterleavedTasks(const BenchmarkConfig& config,
                                                         std::vector<InterleavedTask>& tasks) {
  const std::size_t task_count = tasks.size();
  std::vector<std::vector<double>> task_samples(task_count);
  for (auto& s : task_samples) {
    s.reserve(config.samples);
  }

  // Warmup for each task
  for (auto& task : tasks) {
    task.setup();
    for (std::size_t w = 0; w < config.warmup_iterations; ++w) {
      task.run();
    }
    task.teardown();
  }

  std::vector<std::size_t> task_order(task_count);
  std::iota(task_order.begin(), task_order.end(), 0);
  std::mt19937_64 shuffle_rng(1337);

  // Interleaved randomized sample collection
  for (std::size_t sample_idx = 0; sample_idx < config.samples; ++sample_idx) {
    std::shuffle(task_order.begin(), task_order.end(), shuffle_rng);
    for (std::size_t t_idx : task_order) {
      auto& task = tasks[t_idx];
      task.setup();
      const auto t0 = Clock::now();
      for (std::size_t iter = 0; iter < config.iterations_per_sample; ++iter) {
        task.run();
      }
      const auto t1 = Clock::now();
      task.teardown();

      const double total_sec = std::chrono::duration<double>(t1 - t0).count();
      task_samples[t_idx].push_back(total_sec / static_cast<double>(config.iterations_per_sample));
    }
  }

  std::vector<BenchmarkStatistics> result_stats;
  result_stats.reserve(task_count);
  for (auto& samples : task_samples) {
    result_stats.push_back(ComputeStatistics(std::move(samples)));
  }
  return result_stats;
}

}  // namespace muda::bench

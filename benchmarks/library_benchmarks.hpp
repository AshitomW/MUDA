/**
 * @file library_benchmarks.hpp
 * @brief Benchmark suite covering MUDA libraries (cuBLAS, cuFFT, cuDNN, cuSPARSE, NVRTC)
 *        and runtime BufferPool / memset subsystems.
 */

#pragma once

#include <vector>

#include "benchmark_runner.hpp"

namespace muda::bench {

struct LibraryBenchmarkSuiteResults {
  std::vector<BenchmarkResultRecord> records;
};

LibraryBenchmarkSuiteResults RunLibraryBenchmarks(const BenchmarkConfig& base_config);

}  // namespace muda::bench

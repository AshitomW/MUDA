/**
 * @file throughput_benchmarks.hpp
 * @brief Throughput benchmark suite for MUDA measuring memory bandwidth and compute scalability.
 */

#pragma once

#include <vector>

#include "benchmark_runner.hpp"

namespace muda::bench {

struct ThroughputBenchmarkSuiteResults {
  std::vector<BenchmarkResultRecord> records;
};

struct ThroughputKernelStubs {
  const void* vector_add = nullptr;
  const void* vector_add_float4 = nullptr;
  const void* vector_add_float4x2 = nullptr;
  const void* gemm_tiled = nullptr;
  const void* gemm_register_tiled = nullptr;
  const void* gemm_multistage = nullptr;
};

ThroughputBenchmarkSuiteResults RunThroughputBenchmarks(const ThroughputKernelStubs& stubs,
                                                        const BenchmarkConfig& base_config);

}  // namespace muda::bench

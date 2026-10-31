/**
 * @file microbenchmarks.hpp
 * @brief Microbenchmark suite for MUDA measuring launch latency, memory transfer mechanics, and
 * synchronization.
 */

#pragma once

#include <vector>

#include "benchmark_runner.hpp"

namespace muda::bench {

struct MicrobenchmarkSuiteResults {
  std::vector<BenchmarkResultRecord> records;
};

MicrobenchmarkSuiteResults RunMicrobenchmarks(const void* stub_noop,
                                              const BenchmarkConfig& base_config);

}  // namespace muda::bench

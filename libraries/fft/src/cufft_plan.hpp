/**
 * @file cufft_plan.hpp
 * @brief Internal cuFFT plan state, LruPlanCache key definitions, and handle table helpers.
 */

#pragma once

#import <Metal/Metal.h>

#include <array>
#include <cstddef>
#include <memory>
#include <mutex>

#include "cufft.h"
#include "handle_table.hpp"
#include "muda/libraries/plan_cache.hpp"

namespace muda::libraries::fft {

struct FftPlanConfig {
  int rank = 1;
  std::array<int, 3> dims{1, 1, 1};
  int batch = 1;
  cufftType type = CUFFT_C2C;
  int istride = 1;
  int idist = 0;
  int ostride = 1;
  int odist = 0;

  bool operator==(const FftPlanConfig& other) const noexcept {
    return rank == other.rank && dims == other.dims && batch == other.batch && type == other.type &&
           istride == other.istride && idist == other.idist && ostride == other.ostride &&
           odist == other.odist;
  }
};

struct FftPlanConfigHash {
  std::size_t operator()(const FftPlanConfig& cfg) const noexcept {
    std::size_t seed = 0;
    HashCombine(seed, cfg.rank);
    HashCombine(seed, cfg.dims[0]);
    HashCombine(seed, cfg.dims[1]);
    HashCombine(seed, cfg.dims[2]);
    HashCombine(seed, cfg.batch);
    HashCombine(seed, static_cast<int>(cfg.type));
    HashCombine(seed, cfg.istride);
    HashCombine(seed, cfg.idist);
    HashCombine(seed, cfg.ostride);
    HashCombine(seed, cfg.odist);
    return seed;
  }
};

struct CufftPlanState {
  mutable std::mutex mutex;
  bool configured = false;
  FftPlanConfig config{};
  cudaStream_t stream = nullptr;
  std::size_t work_size_bytes = 0;
  id<MTLBuffer> workspace_a = nil;
  id<MTLBuffer> workspace_b = nil;
};

using CufftHandleTable =
    runtime::cudart::HandleTable<CufftPlanState, runtime::cudart::HandleKind::kCufftPlan>;

[[nodiscard]] CufftHandleTable& GetCufftHandleTable() noexcept;
[[nodiscard]] std::shared_ptr<CufftPlanState> LookupPlan(cufftHandle handle) noexcept;

}  // namespace muda::libraries::fft


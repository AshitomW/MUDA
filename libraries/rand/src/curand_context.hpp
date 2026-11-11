/**
 * @file curand_context.hpp
 * @brief Internal cuRAND state management, handle table, and pipeline cache helpers.
 */

#pragma once

#import <Metal/Metal.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string_view>

#include "curand.h"
#include "handle_table.hpp"

namespace muda::libraries::rand {

enum class GeneratorTarget : std::uint8_t {
  kGpu = 0,
  kHost = 1,
};

struct CurandContext {
  mutable std::mutex mutex;
  curandRngType_t rng_type = CURAND_RNG_PSEUDO_DEFAULT;
  GeneratorTarget target = GeneratorTarget::kGpu;
  cudaStream_t stream = nullptr;
  std::uint64_t seed = 0;
  std::uint64_t offset = 0;
  curandOrdering_t ordering = CURAND_ORDERING_PSEUDO_DEFAULT;
  std::uint32_t dimensions = 1;
};

using CurandHandleTable =
    runtime::cudart::HandleTable<CurandContext, runtime::cudart::HandleKind::kCurandGenerator>;

[[nodiscard]] CurandHandleTable& GetCurandHandleTable() noexcept;

[[nodiscard]] std::shared_ptr<CurandContext> LookupContext(curandGenerator_t handle) noexcept;

[[nodiscard]] id<MTLComputePipelineState> GetCurandPipeline(
    std::string_view function_name) noexcept;

void PhiloxHostStep(std::uint64_t ctr_idx, std::uint64_t seed, std::uint32_t out[4]) noexcept;

curandStatus_t GenerateHostUint(CurandContext& ctx, unsigned int* out, std::size_t num) noexcept;
curandStatus_t GenerateHostUniform(CurandContext& ctx, float* out, std::size_t num) noexcept;
curandStatus_t GenerateHostUniformDouble(CurandContext& ctx, double* out, std::size_t num) noexcept;
curandStatus_t GenerateHostNormal(CurandContext& ctx, float* out, std::size_t num, float mean,
                                  float stddev) noexcept;
curandStatus_t GenerateHostNormalDouble(CurandContext& ctx, double* out, std::size_t num,
                                        double mean, double stddev) noexcept;
curandStatus_t GenerateHostLogNormal(CurandContext& ctx, float* out, std::size_t num, float mean,
                                     float stddev) noexcept;
curandStatus_t GenerateHostLogNormalDouble(CurandContext& ctx, double* out, std::size_t num,
                                           double mean, double stddev) noexcept;
curandStatus_t GenerateHostPoisson(CurandContext& ctx, unsigned int* out, std::size_t num,
                                   double lambda) noexcept;

}  // namespace muda::libraries::rand

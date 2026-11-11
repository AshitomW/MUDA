/**
 * @file curand_context.mm
 * @brief Implementation of cuRAND context lookup, pipeline loading, and host generator routines.
 */

#include "curand_context.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cmath>
#include <cstdint>
#include <cstring>

#include "muda/libraries/metal_executor.hpp"

namespace muda::libraries::rand {
namespace {

constexpr float kTwoPowMinus32 = 2.3283064365386963e-10f;
constexpr double kTwoPowMinus32D = 2.3283064365386963e-10;
constexpr double kTwoPowMinus53 = 1.1102230246251565e-16;
constexpr float kTwoPi = 6.28318530717958647692f;
constexpr double kTwoPiD = 6.28318530717958647692;

}  // namespace

CurandHandleTable& GetCurandHandleTable() noexcept {
  static CurandHandleTable table;
  return table;
}

std::shared_ptr<CurandContext> LookupContext(curandGenerator_t handle) noexcept {
  return GetCurandHandleTable().Lookup(handle);
}

id<MTLComputePipelineState> GetCurandPipeline(std::string_view function_name) noexcept {
  return LibraryExecutor::Instance().GetOrLoadPipeline(
      MUDA_CURAND_METALLIB_PATH, MUDA_CURAND_METAL_SOURCE_PATH, function_name);
}

void PhiloxHostStep(std::uint64_t ctr_idx, std::uint64_t seed, std::uint32_t out[4]) noexcept {
  std::uint32_t ctr[4] = {static_cast<std::uint32_t>(ctr_idx),
                          static_cast<std::uint32_t>(ctr_idx >> 32), 0u, 0u};
  std::uint32_t key[2] = {static_cast<std::uint32_t>(seed), static_cast<std::uint32_t>(seed >> 32)};

  for (int r = 0; r < 10; ++r) {
    const std::uint64_t prod0 = 0xD2511F53ull * static_cast<std::uint64_t>(ctr[0]);
    const std::uint64_t prod1 = 0xCD9E8D57ull * static_cast<std::uint64_t>(ctr[2]);
    const std::uint32_t hi0 = static_cast<std::uint32_t>(prod0 >> 32);
    const std::uint32_t lo0 = static_cast<std::uint32_t>(prod0);
    const std::uint32_t hi1 = static_cast<std::uint32_t>(prod1 >> 32);
    const std::uint32_t lo1 = static_cast<std::uint32_t>(prod1);

    const std::uint32_t new0 = hi1 ^ ctr[1] ^ key[0];
    const std::uint32_t new1 = lo1;
    const std::uint32_t new2 = hi0 ^ ctr[3] ^ key[1];
    const std::uint32_t new3 = lo0;

    ctr[0] = new0;
    ctr[1] = new1;
    ctr[2] = new2;
    ctr[3] = new3;

    key[0] += 0x9E3779B9u;
    key[1] += 0xBB67AE85u;
  }

  out[0] = ctr[0];
  out[1] = ctr[1];
  out[2] = ctr[2];
  out[3] = ctr[3];
}

curandStatus_t GenerateHostUint(CurandContext& ctx, unsigned int* out, std::size_t num) noexcept {
  if (out == nullptr) {
    return CURAND_STATUS_INVALID_VALUE;
  }
  std::lock_guard<std::mutex> lock(ctx.mutex);
  const std::size_t num_rounds = (num + 3) / 4;
  std::uint32_t r[4];

  for (std::size_t i = 0; i < num_rounds; ++i) {
    const std::uint64_t ctr_idx = i + (ctx.offset >> 2);
    PhiloxHostStep(ctr_idx, ctx.seed, r);
    const std::size_t base = i * 4;
    for (std::size_t j = 0; j < 4 && base + j < num; ++j) {
      out[base + j] = r[j];
    }
  }
  ctx.offset += num_rounds * 4;
  return CURAND_STATUS_SUCCESS;
}

curandStatus_t GenerateHostUniform(CurandContext& ctx, float* out, std::size_t num) noexcept {
  if (out == nullptr) {
    return CURAND_STATUS_INVALID_VALUE;
  }
  std::lock_guard<std::mutex> lock(ctx.mutex);
  const std::size_t num_rounds = (num + 3) / 4;
  std::uint32_t r[4];

  for (std::size_t i = 0; i < num_rounds; ++i) {
    const std::uint64_t ctr_idx = i + (ctx.offset >> 2);
    PhiloxHostStep(ctr_idx, ctx.seed, r);
    const std::size_t base = i * 4;
    for (std::size_t j = 0; j < 4 && base + j < num; ++j) {
      out[base + j] = (static_cast<float>(r[j]) + 1.0f) * kTwoPowMinus32;
    }
  }
  ctx.offset += num_rounds * 4;
  return CURAND_STATUS_SUCCESS;
}

curandStatus_t GenerateHostUniformDouble(CurandContext& ctx, double* out,
                                         std::size_t num) noexcept {
  if (out == nullptr) {
    return CURAND_STATUS_INVALID_VALUE;
  }
  std::lock_guard<std::mutex> lock(ctx.mutex);
  const std::size_t num_rounds = (num + 1) / 2;
  std::uint32_t r[4];

  for (std::size_t i = 0; i < num_rounds; ++i) {
    const std::uint64_t ctr_idx = i + (ctx.offset >> 1);
    PhiloxHostStep(ctr_idx, ctx.seed, r);
    const std::size_t base = i * 2;

    const std::uint64_t m0 =
        ((static_cast<std::uint64_t>(r[0]) << 21) ^ static_cast<std::uint64_t>(r[1])) &
        0x1FFFFFFFFFFFFFull;
    out[base] = (static_cast<double>(m0) + 1.0) * kTwoPowMinus53;

    if (base + 1 < num) {
      const std::uint64_t m1 =
          ((static_cast<std::uint64_t>(r[2]) << 21) ^ static_cast<std::uint64_t>(r[3])) &
          0x1FFFFFFFFFFFFFull;
      out[base + 1] = (static_cast<double>(m1) + 1.0) * kTwoPowMinus53;
    }
  }
  ctx.offset += num_rounds * 2;
  return CURAND_STATUS_SUCCESS;
}

curandStatus_t GenerateHostNormal(CurandContext& ctx, float* out, std::size_t num, float mean,
                                  float stddev) noexcept {
  if (out == nullptr) {
    return CURAND_STATUS_INVALID_VALUE;
  }
  std::lock_guard<std::mutex> lock(ctx.mutex);
  const std::size_t num_rounds = (num + 3) / 4;
  std::uint32_t r[4];

  for (std::size_t i = 0; i < num_rounds; ++i) {
    const std::uint64_t ctr_idx = i + (ctx.offset >> 2);
    PhiloxHostStep(ctr_idx, ctx.seed, r);

    const float u0 = (static_cast<float>(r[0]) + 1.0f) * kTwoPowMinus32;
    const float u1 = (static_cast<float>(r[1]) + 1.0f) * kTwoPowMinus32;
    const float u2 = (static_cast<float>(r[2]) + 1.0f) * kTwoPowMinus32;
    const float u3 = (static_cast<float>(r[3]) + 1.0f) * kTwoPowMinus32;

    const float r0 = std::sqrt(-2.0f * std::log(u0));
    const float theta0 = kTwoPi * u1;
    const float z0 = r0 * std::cos(theta0) * stddev + mean;
    const float z1 = r0 * std::sin(theta0) * stddev + mean;

    const float r1 = std::sqrt(-2.0f * std::log(u2));
    const float theta1 = kTwoPi * u3;
    const float z2 = r1 * std::cos(theta1) * stddev + mean;
    const float z3 = r1 * std::sin(theta1) * stddev + mean;

    const std::size_t base = i * 4;
    if (base < num)
      out[base] = z0;
    if (base + 1 < num)
      out[base + 1] = z1;
    if (base + 2 < num)
      out[base + 2] = z2;
    if (base + 3 < num)
      out[base + 3] = z3;
  }
  ctx.offset += num_rounds * 4;
  return CURAND_STATUS_SUCCESS;
}

curandStatus_t GenerateHostNormalDouble(CurandContext& ctx, double* out, std::size_t num,
                                        double mean, double stddev) noexcept {
  if (out == nullptr) {
    return CURAND_STATUS_INVALID_VALUE;
  }
  std::lock_guard<std::mutex> lock(ctx.mutex);
  const std::size_t num_rounds = (num + 1) / 2;
  std::uint32_t r[4];

  for (std::size_t i = 0; i < num_rounds; ++i) {
    const std::uint64_t ctr_idx = i + (ctx.offset >> 1);
    PhiloxHostStep(ctr_idx, ctx.seed, r);

    const std::uint64_t m0 =
        ((static_cast<std::uint64_t>(r[0]) << 21) ^ static_cast<std::uint64_t>(r[1])) &
        0x1FFFFFFFFFFFFFull;
    const std::uint64_t m1 =
        ((static_cast<std::uint64_t>(r[2]) << 21) ^ static_cast<std::uint64_t>(r[3])) &
        0x1FFFFFFFFFFFFFull;

    const double u0 = (static_cast<double>(m0) + 1.0) * kTwoPowMinus53;
    const double u1 = (static_cast<double>(m1) + 1.0) * kTwoPowMinus53;

    const double r_val = std::sqrt(-2.0 * std::log(u0));
    const double theta = kTwoPiD * u1;
    const double z0 = r_val * std::cos(theta) * stddev + mean;
    const double z1 = r_val * std::sin(theta) * stddev + mean;

    const std::size_t base = i * 2;
    out[base] = z0;
    if (base + 1 < num) {
      out[base + 1] = z1;
    }
  }
  ctx.offset += num_rounds * 2;
  return CURAND_STATUS_SUCCESS;
}

curandStatus_t GenerateHostLogNormal(CurandContext& ctx, float* out, std::size_t num, float mean,
                                     float stddev) noexcept {
  const curandStatus_t st = GenerateHostNormal(ctx, out, num, mean, stddev);
  if (st != CURAND_STATUS_SUCCESS) {
    return st;
  }
  for (std::size_t i = 0; i < num; ++i) {
    out[i] = std::exp(out[i]);
  }
  return CURAND_STATUS_SUCCESS;
}

curandStatus_t GenerateHostLogNormalDouble(CurandContext& ctx, double* out, std::size_t num,
                                           double mean, double stddev) noexcept {
  const curandStatus_t st = GenerateHostNormalDouble(ctx, out, num, mean, stddev);
  if (st != CURAND_STATUS_SUCCESS) {
    return st;
  }
  for (std::size_t i = 0; i < num; ++i) {
    out[i] = std::exp(out[i]);
  }
  return CURAND_STATUS_SUCCESS;
}

curandStatus_t GenerateHostPoisson(CurandContext& ctx, unsigned int* out, std::size_t num,
                                   double lambda) noexcept {
  if (out == nullptr) {
    return CURAND_STATUS_INVALID_VALUE;
  }
  std::lock_guard<std::mutex> lock(ctx.mutex);

  for (std::size_t i = 0; i < num; ++i) {
    const std::uint64_t ctr_idx = i + ctx.offset;
    std::uint32_t r[4];
    PhiloxHostStep(ctr_idx, ctx.seed, r);

    if (lambda < 30.0) {
      const double L = std::exp(-lambda);
      double p = 1.0;
      unsigned int k = 0;
      unsigned int sub = 0;
      std::uint64_t sub_ctr = ctr_idx;

      while (p > L && k < 1000u) {
        ++k;
        const unsigned int val = r[sub % 4];
        const double u = (static_cast<double>(val) + 1.0) * kTwoPowMinus32D;
        p *= u;
        ++sub;
        if (sub % 4 == 0) {
          sub_ctr += 0x100000000ull;
          PhiloxHostStep(sub_ctr, ctx.seed, r);
        }
      }
      out[i] = (k > 0) ? (k - 1) : 0u;
    } else {
      const double u0 = (static_cast<double>(r[0]) + 1.0) * kTwoPowMinus32D;
      const double u1 = (static_cast<double>(r[1]) + 1.0) * kTwoPowMinus32D;
      const double z = std::sqrt(-2.0 * std::log(u0)) * std::cos(kTwoPiD * u1);
      const double val = lambda + std::sqrt(lambda) * z + 0.5;
      out[i] = (val > 0.0) ? static_cast<unsigned int>(val) : 0u;
    }
  }
  ctx.offset += num;
  return CURAND_STATUS_SUCCESS;
}

}  // namespace muda::libraries::rand

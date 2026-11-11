/**
 * @file curand_kernels.metal
 * @brief High-performance Metal compute shaders for cuRAND (Philox4x32-10, distributions).
 */

#include <metal_stdlib>
using namespace metal;

namespace {

// Philox4x32-10 Constants
constant ulong kPhiloxW32 = 0xD2511F53ul;
constant ulong kPhiloxM32 = 0xCD9E8D57ul;
constant uint kPhiloxKeyW0 = 0x9E3779B9u;
constant uint kPhiloxKeyW1 = 0xBB67AE85u;
constant float kTwoPowMinus32 = 2.3283064365386963e-10f;
constant float kTwoPi = 6.28318530717958647692f;

inline uint4 PhiloxRound(uint4 ctr, uint2 key) {
  const ulong prod0 = kPhiloxW32 * static_cast<ulong>(ctr.x);
  const ulong prod1 = kPhiloxM32 * static_cast<ulong>(ctr.z);
  const uint hi0 = static_cast<uint>(prod0 >> 32);
  const uint lo0 = static_cast<uint>(prod0);
  const uint hi1 = static_cast<uint>(prod1 >> 32);
  const uint lo1 = static_cast<uint>(prod1);
  return uint4(hi1 ^ ctr.y ^ key.x, lo1, hi0 ^ ctr.w ^ key.y, lo0);
}

inline uint4 Philox4x32_10(uint4 ctr, uint2 key) {
  for (int r = 0; r < 10; ++r) {
    ctr = PhiloxRound(ctr, key);
    key.x += kPhiloxKeyW0;
    key.y += kPhiloxKeyW1;
  }
  return ctr;
}

inline float4 PhiloxToUniformFloat4(uint4 u) {
  return float4(
      (static_cast<float>(u.x) + 1.0f) * kTwoPowMinus32,
      (static_cast<float>(u.y) + 1.0f) * kTwoPowMinus32,
      (static_cast<float>(u.z) + 1.0f) * kTwoPowMinus32,
      (static_cast<float>(u.w) + 1.0f) * kTwoPowMinus32);
}

inline float4 BoxMullerTransform(float4 u, float mean, float stddev) {
  const float r0 = sqrt(-2.0f * log(u.x));
  const float theta0 = kTwoPi * u.y;
  const float z0 = r0 * cos(theta0);
  const float z1 = r0 * sin(theta0);

  const float r1 = sqrt(-2.0f * log(u.z));
  const float theta1 = kTwoPi * u.w;
  const float z2 = r1 * cos(theta1);
  const float z3 = r1 * sin(theta1);

  return float4(z0, z1, z2, z3) * stddev + mean;
}

}  // namespace

// =============================================================================
// Philox4x32-10 Kernels
// =============================================================================

kernel void muda_curand_philox_uint(
    device uint* out [[buffer(0)]],
    constant ulong& num [[buffer(1)]],
    constant ulong& seed [[buffer(2)]],
    constant ulong& offset [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  const ulong thread_idx = static_cast<ulong>(gid);
  const ulong base_idx = thread_idx * 4ul;
  if (base_idx >= num) {
    return;
  }

  const ulong ctr_idx = thread_idx + (offset >> 2);
  const uint4 ctr = uint4(
      static_cast<uint>(ctr_idx),
      static_cast<uint>(ctr_idx >> 32),
      0u,
      0u);
  const uint2 key = uint2(
      static_cast<uint>(seed),
      static_cast<uint>(seed >> 32));

  const uint4 r = Philox4x32_10(ctr, key);

  if (base_idx + 4ul <= num) {
    *reinterpret_cast<device uint4*>(out + base_idx) = r;
  } else {
    out[base_idx] = r.x;
    if (base_idx + 1ul < num) out[base_idx + 1ul] = r.y;
    if (base_idx + 2ul < num) out[base_idx + 2ul] = r.z;
    if (base_idx + 3ul < num) out[base_idx + 3ul] = r.w;
  }
}

kernel void muda_curand_philox_uniform_f32(
    device float* out [[buffer(0)]],
    constant ulong& num [[buffer(1)]],
    constant ulong& seed [[buffer(2)]],
    constant ulong& offset [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  const ulong thread_idx = static_cast<ulong>(gid);
  const ulong base_idx = thread_idx * 4ul;
  if (base_idx >= num) {
    return;
  }

  const ulong ctr_idx = thread_idx + (offset >> 2);
  const uint4 ctr = uint4(
      static_cast<uint>(ctr_idx),
      static_cast<uint>(ctr_idx >> 32),
      0u,
      0u);
  const uint2 key = uint2(
      static_cast<uint>(seed),
      static_cast<uint>(seed >> 32));

  const uint4 r = Philox4x32_10(ctr, key);
  const float4 u = PhiloxToUniformFloat4(r);

  if (base_idx + 4ul <= num) {
    *reinterpret_cast<device float4*>(out + base_idx) = u;
  } else {
    out[base_idx] = u.x;
    if (base_idx + 1ul < num) out[base_idx + 1ul] = u.y;
    if (base_idx + 2ul < num) out[base_idx + 2ul] = u.z;
    if (base_idx + 3ul < num) out[base_idx + 3ul] = u.w;
  }
}

kernel void muda_curand_philox_normal_f32(
    device float* out [[buffer(0)]],
    constant ulong& num [[buffer(1)]],
    constant ulong& seed [[buffer(2)]],
    constant ulong& offset [[buffer(3)]],
    constant float& mean [[buffer(4)]],
    constant float& stddev [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
  const ulong thread_idx = static_cast<ulong>(gid);
  const ulong base_idx = thread_idx * 4ul;
  if (base_idx >= num) {
    return;
  }

  const ulong ctr_idx = thread_idx + (offset >> 2);
  const uint4 ctr = uint4(
      static_cast<uint>(ctr_idx),
      static_cast<uint>(ctr_idx >> 32),
      0u,
      0u);
  const uint2 key = uint2(
      static_cast<uint>(seed),
      static_cast<uint>(seed >> 32));

  const uint4 r = Philox4x32_10(ctr, key);
  const float4 u = PhiloxToUniformFloat4(r);
  const float4 norm = BoxMullerTransform(u, mean, stddev);

  if (base_idx + 4ul <= num) {
    *reinterpret_cast<device float4*>(out + base_idx) = norm;
  } else {
    out[base_idx] = norm.x;
    if (base_idx + 1ul < num) out[base_idx + 1ul] = norm.y;
    if (base_idx + 2ul < num) out[base_idx + 2ul] = norm.z;
    if (base_idx + 3ul < num) out[base_idx + 3ul] = norm.w;
  }
}

kernel void muda_curand_philox_log_normal_f32(
    device float* out [[buffer(0)]],
    constant ulong& num [[buffer(1)]],
    constant ulong& seed [[buffer(2)]],
    constant ulong& offset [[buffer(3)]],
    constant float& mean [[buffer(4)]],
    constant float& stddev [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
  const ulong thread_idx = static_cast<ulong>(gid);
  const ulong base_idx = thread_idx * 4ul;
  if (base_idx >= num) {
    return;
  }

  const ulong ctr_idx = thread_idx + (offset >> 2);
  const uint4 ctr = uint4(
      static_cast<uint>(ctr_idx),
      static_cast<uint>(ctr_idx >> 32),
      0u,
      0u);
  const uint2 key = uint2(
      static_cast<uint>(seed),
      static_cast<uint>(seed >> 32));

  const uint4 r = Philox4x32_10(ctr, key);
  const float4 u = PhiloxToUniformFloat4(r);
  const float4 norm = BoxMullerTransform(u, mean, stddev);
  const float4 log_norm = exp(norm);

  if (base_idx + 4ul <= num) {
    *reinterpret_cast<device float4*>(out + base_idx) = log_norm;
  } else {
    out[base_idx] = log_norm.x;
    if (base_idx + 1ul < num) out[base_idx + 1ul] = log_norm.y;
    if (base_idx + 2ul < num) out[base_idx + 2ul] = log_norm.z;
    if (base_idx + 3ul < num) out[base_idx + 3ul] = log_norm.w;
  }
}

kernel void muda_curand_philox_poisson(
    device uint* out [[buffer(0)]],
    constant ulong& num [[buffer(1)]],
    constant ulong& seed [[buffer(2)]],
    constant ulong& offset [[buffer(3)]],
    constant float& lambda [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  const ulong thread_idx = static_cast<ulong>(gid);
  if (thread_idx >= num) {
    return;
  }

  const ulong ctr_idx = thread_idx + offset;
  uint4 ctr = uint4(
      static_cast<uint>(ctr_idx),
      static_cast<uint>(ctr_idx >> 32),
      0u,
      0u);
  const uint2 key = uint2(
      static_cast<uint>(seed),
      static_cast<uint>(seed >> 32));

  if (lambda < 30.0f) {
    // Knuth's method for small lambda
    const float L = exp(-lambda);
    float p = 1.0f;
    uint k = 0;
    uint sub_round = 0;
    uint4 r = Philox4x32_10(ctr, key);

    while (p > L && k < 1000u) {
      ++k;
      const uint val = (sub_round == 0) ? r.x : ((sub_round == 1) ? r.y : ((sub_round == 2) ? r.z : r.w));
      const float u = (static_cast<float>(val) + 1.0f) * kTwoPowMinus32;
      p *= u;
      ++sub_round;
      if (sub_round == 4) {
        sub_round = 0;
        ctr.z += 1u;
        r = Philox4x32_10(ctr, key);
      }
    }
    out[thread_idx] = (k > 0) ? (k - 1) : 0u;
  } else {
    // Gaussian approximation for large lambda
    const uint4 r = Philox4x32_10(ctr, key);
    const float u0 = (static_cast<float>(r.x) + 1.0f) * kTwoPowMinus32;
    const float u1 = (static_cast<float>(r.y) + 1.0f) * kTwoPowMinus32;
    const float z = sqrt(-2.0f * log(u0)) * cos(kTwoPi * u1);
    const float val = lambda + sqrt(lambda) * z + 0.5f;
    out[thread_idx] = (val > 0.0f) ? static_cast<uint>(val) : 0u;
  }
}

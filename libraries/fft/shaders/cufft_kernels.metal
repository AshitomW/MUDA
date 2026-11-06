/**
 * @file cufft_kernels.metal
 * @brief Metal compute kernels for Stockham Radix-2 and general-radix FFT axes.
 */

#include <metal_stdlib>
using namespace metal;

inline float2 cmul(float2 a, float2 b) {
  return float2(fma(a.x, b.x, -a.y * b.y), fma(a.x, b.y, a.y * b.x));
}

kernel void muda_cufft_gather_c2c(
    device const float2* src [[buffer(0)]],
    device float2* dst [[buffer(1)]],
    constant uint& elem_per_batch [[buffer(2)]],
    constant uint& batch_count [[buffer(3)]],
    constant uint& istride [[buffer(4)]],
    constant uint& idist [[buffer(5)]],
    uint tid [[thread_position_in_grid]]) {
  const uint total = elem_per_batch * batch_count;
  if (tid >= total) return;
  const uint b = tid / elem_per_batch;
  const uint idx = tid % elem_per_batch;
  dst[tid] = src[b * idist + idx * istride];
}

kernel void muda_cufft_scatter_c2c(
    device const float2* src [[buffer(0)]],
    device float2* dst [[buffer(1)]],
    constant uint& elem_per_batch [[buffer(2)]],
    constant uint& batch_count [[buffer(3)]],
    constant uint& ostride [[buffer(4)]],
    constant uint& odist [[buffer(5)]],
    uint tid [[thread_position_in_grid]]) {
  const uint total = elem_per_batch * batch_count;
  if (tid >= total) return;
  const uint b = tid / elem_per_batch;
  const uint idx = tid % elem_per_batch;
  dst[b * odist + idx * ostride] = src[tid];
}

kernel void muda_cufft_gather_r2c(
    device const float* src [[buffer(0)]],
    device float2* dst [[buffer(1)]],
    constant uint& elem_per_batch [[buffer(2)]],
    constant uint& batch_count [[buffer(3)]],
    constant uint& istride [[buffer(4)]],
    constant uint& idist [[buffer(5)]],
    uint tid [[thread_position_in_grid]]) {
  const uint total = elem_per_batch * batch_count;
  if (tid >= total) return;
  const uint b = tid / elem_per_batch;
  const uint idx = tid % elem_per_batch;
  dst[tid] = float2(src[b * idist + idx * istride], 0.0f);
}

kernel void muda_cufft_scatter_r2c_hermitian(
    device const float2* src [[buffer(0)]],
    device float2* dst [[buffer(1)]],
    constant uint& prefix_count [[buffer(2)]],
    constant uint& last_dim [[buffer(3)]],
    constant uint& herm_dim [[buffer(4)]],
    constant uint& batch_count [[buffer(5)]],
    constant uint& ostride [[buffer(6)]],
    constant uint& odist [[buffer(7)]],
    uint tid [[thread_position_in_grid]]) {
  const uint herm_per_batch = prefix_count * herm_dim;
  const uint total = herm_per_batch * batch_count;
  if (tid >= total) return;
  const uint b = tid / herm_per_batch;
  const uint rem = tid % herm_per_batch;
  const uint p = rem / herm_dim;
  const uint k = rem % herm_dim;
  const uint src_idx = (b * prefix_count + p) * last_dim + k;
  dst[b * odist + rem * ostride] = src[src_idx];
}

kernel void muda_cufft_gather_c2r_hermitian(
    device const float2* src [[buffer(0)]],
    device float2* dst [[buffer(1)]],
    constant uint& herm_per_batch [[buffer(2)]],
    constant uint& batch_count [[buffer(3)]],
    constant uint& istride [[buffer(4)]],
    constant uint& idist [[buffer(5)]],
    uint tid [[thread_position_in_grid]]) {
  const uint total = herm_per_batch * batch_count;
  if (tid >= total) return;
  const uint b = tid / herm_per_batch;
  const uint idx = tid % herm_per_batch;
  dst[tid] = src[b * idist + idx * istride];
}

kernel void muda_cufft_expand_hermitian_last_axis(
    device const float2* src [[buffer(0)]],
    device float2* dst [[buffer(1)]],
    constant uint& prefix_count [[buffer(2)]],
    constant uint& last_dim [[buffer(3)]],
    constant uint& herm_dim [[buffer(4)]],
    constant uint& batch_count [[buffer(5)]],
    uint tid [[thread_position_in_grid]]) {
  const uint full_per_batch = prefix_count * last_dim;
  const uint total = full_per_batch * batch_count;
  if (tid >= total) return;
  const uint b = tid / full_per_batch;
  const uint rem = tid % full_per_batch;
  const uint p = rem / last_dim;
  const uint k = rem % last_dim;
  const uint herm_base = (b * prefix_count + p) * herm_dim;
  if (k < herm_dim) {
    dst[tid] = src[herm_base + k];
  } else {
    const uint mirror_k = last_dim - k;
    const float2 v = src[herm_base + mirror_k];
    dst[tid] = float2(v.x, -v.y);
  }
}

kernel void muda_cufft_scatter_c2r_real(
    device const float2* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant uint& elem_per_batch [[buffer(2)]],
    constant uint& batch_count [[buffer(3)]],
    constant uint& ostride [[buffer(4)]],
    constant uint& odist [[buffer(5)]],
    uint tid [[thread_position_in_grid]]) {
  const uint total = elem_per_batch * batch_count;
  if (tid >= total) return;
  const uint b = tid / elem_per_batch;
  const uint idx = tid % elem_per_batch;
  dst[b * odist + idx * ostride] = src[tid].x;
}

kernel void muda_cufft_stockham_stage(
    device const float2* src [[buffer(0)]],
    device float2* dst [[buffer(1)]],
    constant uint& total_elements [[buffer(2)]],
    constant uint& axis_len [[buffer(3)]],
    constant uint& inner_stride [[buffer(4)]],
    constant uint& half_m [[buffer(5)]],
    constant float& sign_dir [[buffer(6)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid >= total_elements) return;
  const uint inner_idx = tid % inner_stride;
  const uint temp = tid / inner_stride;
  const uint k = temp % axis_len;
  const uint outer_idx = temp / axis_len;

  const uint m = half_m << 1u;
  const uint j = k / m;
  const uint r = k % m;
  const uint r_mod = r % half_m;

  const uint base_offset = outer_idx * (axis_len * inner_stride) + inner_idx;
  const uint idx0 = base_offset + (j * half_m + r_mod) * inner_stride;
  const uint idx1 = base_offset + (j * half_m + r_mod + (axis_len >> 1u)) * inner_stride;

  const float angle = sign_dir * (3.14159265358979323846f * float(r_mod)) / float(half_m);
  float s = 0.0f;
  float c = 1.0f;
  s = sincos(angle, c);
  const float2 w = float2(c, s);

  const float2 u = src[idx0];
  const float2 v = cmul(w, src[idx1]);
  dst[tid] = (r < half_m) ? (u + v) : (u - v);
}

kernel void muda_cufft_axis_dft(
    device const float2* src [[buffer(0)]],
    device float2* dst [[buffer(1)]],
    constant uint& total_elements [[buffer(2)]],
    constant uint& axis_len [[buffer(3)]],
    constant uint& inner_stride [[buffer(4)]],
    constant float& sign_dir [[buffer(5)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid >= total_elements) return;
  const uint inner_idx = tid % inner_stride;
  const uint temp = tid / inner_stride;
  const uint k = temp % axis_len;
  const uint outer_idx = temp / axis_len;

  const uint base_offset = outer_idx * (axis_len * inner_stride) + inner_idx;
  const float base_angle = sign_dir * (6.28318530717958647692f * float(k)) / float(axis_len);

  float2 acc = float2(0.0f, 0.0f);
  for (uint n = 0; n < axis_len; ++n) {
    const float angle = base_angle * float(n);
    float s = 0.0f;
    float c = 1.0f;
    s = sincos(angle, c);
    const float2 w = float2(c, s);
    acc += cmul(src[base_offset + n * inner_stride], w);
  }
  dst[tid] = acc;
}


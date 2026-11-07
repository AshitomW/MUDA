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

kernel void muda_cufft_c2c_1d_shared(
    device const float2* src [[buffer(0)]],
    device float2* dst [[buffer(1)]],
    constant uint& axis_len [[buffer(2)]],
    constant float& sign_dir [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]]) {
  threadgroup float2 s_a[1024];
  threadgroup float2 s_b[1024];

  const uint base = tgid * axis_len;
  for (uint k = lid; k < axis_len; k += lsize) {
    s_a[k] = src[base + k];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint half_axis = axis_len >> 1u;
  bool ping = true;
  for (uint half_m = 1u; half_m < axis_len; half_m <<= 1u) {
    const uint half_m_mask = half_m - 1u;
    const float inv_half_m = 3.14159265358979323846f * sign_dir / float(half_m);
    if (ping) {
      for (uint p = lid; p < half_axis; p += lsize) {
        const uint r_mod = p & half_m_mask;
        const uint out0 = ((p & ~half_m_mask) << 1u) + r_mod;
        float c = 1.0f;
        const float s = sincos(inv_half_m * float(r_mod), c);
        const float2 u = s_a[p];
        const float2 v = cmul(float2(c, s), s_a[p + half_axis]);
        s_b[out0] = u + v;
        s_b[out0 + half_m] = u - v;
      }
    } else {
      for (uint p = lid; p < half_axis; p += lsize) {
        const uint r_mod = p & half_m_mask;
        const uint out0 = ((p & ~half_m_mask) << 1u) + r_mod;
        float c = 1.0f;
        const float s = sincos(inv_half_m * float(r_mod), c);
        const float2 u = s_b[p];
        const float2 v = cmul(float2(c, s), s_b[p + half_axis]);
        s_a[out0] = u + v;
        s_a[out0 + half_m] = u - v;
      }
    }
    ping = !ping;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (ping) {
    for (uint k = lid; k < axis_len; k += lsize) {
      dst[base + k] = s_a[k];
    }
  } else {
    for (uint k = lid; k < axis_len; k += lsize) {
      dst[base + k] = s_b[k];
    }
  }
}

kernel void muda_cufft_r2c_1d_shared(
    device const float* src [[buffer(0)]],
    device float2* dst [[buffer(1)]],
    constant uint& axis_len [[buffer(2)]],
    constant uint& herm_dim [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]]) {
  threadgroup float2 s_a[1024];
  threadgroup float2 s_b[1024];

  const uint in_base = tgid * axis_len;
  for (uint k = lid; k < axis_len; k += lsize) {
    s_a[k] = float2(src[in_base + k], 0.0f);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint half_axis = axis_len >> 1u;
  bool ping = true;
  for (uint half_m = 1u; half_m < axis_len; half_m <<= 1u) {
    const uint half_m_mask = half_m - 1u;
    const float inv_half_m = -3.14159265358979323846f / float(half_m);
    if (ping) {
      for (uint p = lid; p < half_axis; p += lsize) {
        const uint r_mod = p & half_m_mask;
        const uint out0 = ((p & ~half_m_mask) << 1u) + r_mod;
        float c = 1.0f;
        const float s = sincos(inv_half_m * float(r_mod), c);
        const float2 u = s_a[p];
        const float2 v = cmul(float2(c, s), s_a[p + half_axis]);
        s_b[out0] = u + v;
        s_b[out0 + half_m] = u - v;
      }
    } else {
      for (uint p = lid; p < half_axis; p += lsize) {
        const uint r_mod = p & half_m_mask;
        const uint out0 = ((p & ~half_m_mask) << 1u) + r_mod;
        float c = 1.0f;
        const float s = sincos(inv_half_m * float(r_mod), c);
        const float2 u = s_b[p];
        const float2 v = cmul(float2(c, s), s_b[p + half_axis]);
        s_a[out0] = u + v;
        s_a[out0 + half_m] = u - v;
      }
    }
    ping = !ping;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const uint out_base = tgid * herm_dim;
  if (ping) {
    for (uint k = lid; k < herm_dim; k += lsize) {
      dst[out_base + k] = s_a[k];
    }
  } else {
    for (uint k = lid; k < herm_dim; k += lsize) {
      dst[out_base + k] = s_b[k];
    }
  }
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
  const uint axis_mask = axis_len - 1u;
  uint inner_idx = 0u;
  uint k = 0u;
  uint outer_idx = 0u;
  if (inner_stride == 1u) {
    k = tid & axis_mask;
    outer_idx = tid >> ctz(axis_len);
  } else {
    inner_idx = tid % inner_stride;
    const uint temp = tid / inner_stride;
    k = temp & axis_mask;
    outer_idx = temp >> ctz(axis_len);
  }

  const uint m = half_m << 1u;
  const uint r = k & (m - 1u);
  const uint r_mod = k & (half_m - 1u);
  const uint j_half_m = (k & ~(m - 1u)) >> 1u;

  const uint base_offset = outer_idx * (axis_len * inner_stride) + inner_idx;
  const uint idx0 = base_offset + (j_half_m + r_mod) * inner_stride;
  const uint idx1 = idx0 + (axis_len >> 1u) * inner_stride;

  const float angle = sign_dir * (3.14159265358979323846f * float(r_mod)) / float(half_m);
  float c = 1.0f;
  const float s = sincos(angle, c);
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


/**
 * @file cudnn_kernels.metal
 * @brief Metal compute kernels for cuDNN Conv2D, Activations, Pooling, Softmax,
 *        BatchNorm, and AddTensor.
 */

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
#include "../src/cudnn_kernel_params.h"

using namespace metal;

inline int filter_offset(int k, int c_g, int r, int s, int c_per_group, int kh, int kw, int fmt) {
  if (fmt == 0) {
    return ((k * c_per_group + c_g) * kh + r) * kw + s;
  }
  return ((k * kh + r) * kw + s) * c_per_group + c_g;
}

kernel void muda_cudnn_conv2d_fwd(
    device const float* x [[buffer(0)]],
    device const float* w [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant CudnnConvParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  const uint total = uint(p.n) * uint(p.c_out) * uint(p.h_out) * uint(p.w_out);
  if (tid >= total) return;

  const int ox = int(tid % uint(p.w_out));
  uint rem = tid / uint(p.w_out);
  const int oy = int(rem % uint(p.h_out));
  rem /= uint(p.h_out);
  const int oc = int(rem % uint(p.c_out));
  const int b = int(rem / uint(p.c_out));

  const int c_in_per_g = p.c_in / p.groups;
  const int c_out_per_g = p.c_out / p.groups;
  const int g = oc / c_out_per_g;
  const int ic_start = g * c_in_per_g;

  float acc = 0.0f;
  for (int c_g = 0; c_g < c_in_per_g; ++c_g) {
    const int ic = ic_start + c_g;
    for (int r = 0; r < p.kh; ++r) {
      const int fr = (p.conv_mode == 0) ? (p.kh - 1 - r) : r;
      const int iy = oy * p.stride_h - p.pad_h + r * p.dil_h;
      if (iy < 0 || iy >= p.h_in) continue;
      for (int s = 0; s < p.kw; ++s) {
        const int fs = (p.conv_mode == 0) ? (p.kw - 1 - s) : s;
        const int ix = ox * p.stride_w - p.pad_w + s * p.dil_w;
        if (ix < 0 || ix >= p.w_in) continue;
        const int x_idx = b * p.x_ns + ic * p.x_cs + iy * p.x_hs + ix * p.x_ws;
        const int w_idx = filter_offset(oc, c_g, fr, fs, c_in_per_g, p.kh, p.kw, p.w_format);
        acc = fma(x[x_idx], w[w_idx], acc);
      }
    }
  }

  const int y_idx = b * p.y_ns + oc * p.y_cs + oy * p.y_hs + ox * p.y_ws;
  const float prev_y = (p.beta == 0.0f) ? 0.0f : (p.beta * y[y_idx]);
  y[y_idx] = fma(p.alpha, acc, prev_y);
}

kernel void muda_cudnn_conv2d_fwd_implicit_gemm(
    device const float* x [[buffer(0)]],
    device const float* w [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant CudnnConvParams& p [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]) {
  const uint M = uint(p.c_out);
  const uint hw_out = uint(p.h_out) * uint(p.w_out);
  const uint N = uint(p.n) * hw_out;
  const uint kw_u = uint(p.kw);
  const uint kh_kw = uint(p.kh) * kw_u;
  const uint K = uint(p.c_in) * kh_kw;

  const uint base_m = tgid.y * 32u;
  const uint base_n = tgid.x * 32u;
  if (base_m >= M || base_n >= N) return;

  const uint sg_rm = (simd_id >> 1u) * 16u;
  const uint sg_cn = (simd_id & 1u) * 16u;
  const uint tid = simd_id * 32u + simd_lane;

  threadgroup float s_w[512];  // 32 (M = c_out) x 16 (K)
  threadgroup float s_x[512];  // 16 (K) x 32 (N = batch * h_out * w_out)
  threadgroup int4 s_k_info[16];

  const uint cn = tid & 31u;
  const uint gn = base_n + cn;
  const bool valid_gn = (gn < N);
  int b_offset = 0;
  int oy_base = 0;
  int ox_base = 0;
  if (valid_gn) {
    const uint b = gn / hw_out;
    const uint rem_n = gn % hw_out;
    const int oy = int(rem_n / uint(p.w_out));
    const int ox = int(rem_n % uint(p.w_out));
    b_offset = int(b) * p.x_ns;
    oy_base = oy * p.stride_h - p.pad_h;
    ox_base = ox * p.stride_w - p.pad_w;
  }

  simdgroup_float8x8 acc00 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  simdgroup_float8x8 acc01 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  simdgroup_float8x8 acc10 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  simdgroup_float8x8 acc11 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

  for (uint k_base = 0; k_base < K; k_base += 16u) {
    if (tid < 16u) {
      const uint x_gk = k_base + tid;
      if (x_gk < K) {
        const uint ic = x_gk / kh_kw;
        const uint rem_k = x_gk % kh_kw;
        const int fr = int(rem_k / kw_u);
        const int fs = int(rem_k % kw_u);
        const int r = (p.conv_mode == 0) ? (p.kh - 1 - fr) : fr;
        const int s = (p.conv_mode == 0) ? (p.kw - 1 - fs) : fs;
        s_k_info[tid] = int4(int(ic) * p.x_cs, r * p.dil_h, s * p.dil_w, 1);
      } else {
        s_k_info[tid] = int4(0, 0, 0, 0);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = tid; idx < 512u; idx += 128u) {
      const uint rm = idx >> 4u;
      const uint ck = idx & 15u;
      const uint gm = base_m + rm;
      const uint gk = k_base + ck;
      s_w[rm * 16u + ck] = (gm < M && gk < K) ? w[gm * K + gk] : 0.0f;

      const uint rk = idx >> 5u;
      const int4 ki = s_k_info[rk];
      float x_val = 0.0f;
      if (valid_gn && ki.w != 0) {
        const int iy = oy_base + ki.y;
        const int ix = ox_base + ki.z;
        if (uint(iy) < uint(p.h_in) && uint(ix) < uint(p.w_in)) {
          x_val = x[b_offset + ki.x + iy * p.x_hs + ix * p.x_ws];
        }
      }
      s_x[rk * 32u + cn] = x_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint ks = 0; ks < 16u; ks += 8u) {
      simdgroup_float8x8 fw0;
      simdgroup_float8x8 fw1;
      simdgroup_float8x8 fx0;
      simdgroup_float8x8 fx1;
      simdgroup_load(fw0, s_w + (sg_rm + 0u) * 16u + ks, 16u);
      simdgroup_load(fw1, s_w + (sg_rm + 8u) * 16u + ks, 16u);
      simdgroup_load(fx0, s_x + ks * 32u + sg_cn + 0u, 32u);
      simdgroup_load(fx1, s_x + ks * 32u + sg_cn + 8u, 32u);

      simdgroup_multiply_accumulate(acc00, fw0, fx0, acc00);
      simdgroup_multiply_accumulate(acc01, fw0, fx1, acc01);
      simdgroup_multiply_accumulate(acc10, fw1, fx0, acc10);
      simdgroup_multiply_accumulate(acc11, fw1, fx1, acc11);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  threadgroup float* sg_scratch = (simd_id < 2u) ? (s_w + simd_id * 256u)
                                                 : (s_x + (simd_id - 2u) * 256u);
  simdgroup_store(acc00, sg_scratch + 0u, 16u);
  simdgroup_store(acc01, sg_scratch + 8u, 16u);
  simdgroup_store(acc10, sg_scratch + 128u, 16u);
  simdgroup_store(acc11, sg_scratch + 136u, 16u);
  simdgroup_barrier(mem_flags::mem_threadgroup);

  for (uint idx = simd_lane; idx < 256u; idx += 32u) {
    const uint rm = idx >> 4u;
    const uint cn = idx & 15u;
    const uint gm = base_m + sg_rm + rm;
    const uint gn = base_n + sg_cn + cn;
    if (gm < M && gn < N) {
      const uint b = gn / hw_out;
      const uint rem_n = gn % hw_out;
      const int oy = int(rem_n / uint(p.w_out));
      const int ox = int(rem_n % uint(p.w_out));
      const int y_idx = int(b) * p.y_ns + int(gm) * p.y_cs + oy * p.y_hs + ox * p.y_ws;
      const float prev_y = (p.beta == 0.0f) ? 0.0f : (p.beta * y[y_idx]);
      y[y_idx] = fma(p.alpha, sg_scratch[rm * 16u + cn], prev_y);
    }
  }
}

kernel void muda_cudnn_conv2d_bwd_data(
    device const float* w [[buffer(0)]],
    device const float* dy [[buffer(1)]],
    device float* dx [[buffer(2)]],
    constant CudnnConvParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  const uint total = uint(p.n) * uint(p.c_in) * uint(p.h_in) * uint(p.w_in);
  if (tid >= total) return;

  const int ix = int(tid % uint(p.w_in));
  uint rem = tid / uint(p.w_in);
  const int iy = int(rem % uint(p.h_in));
  rem /= uint(p.h_in);
  const int ic = int(rem % uint(p.c_in));
  const int b = int(rem / uint(p.c_in));

  const int c_in_per_g = p.c_in / p.groups;
  const int c_out_per_g = p.c_out / p.groups;
  const int g = ic / c_in_per_g;
  const int c_g = ic - g * c_in_per_g;
  const int oc_start = g * c_out_per_g;

  float acc = 0.0f;
  for (int oc_local = 0; oc_local < c_out_per_g; ++oc_local) {
    const int oc = oc_start + oc_local;
    for (int r = 0; r < p.kh; ++r) {
      const int num_y = iy + p.pad_h - r * p.dil_h;
      if (num_y < 0 || (num_y % p.stride_h) != 0) continue;
      const int oy = num_y / p.stride_h;
      if (oy >= p.h_out) continue;
      const int fr = (p.conv_mode == 0) ? (p.kh - 1 - r) : r;

      for (int s = 0; s < p.kw; ++s) {
        const int num_x = ix + p.pad_w - s * p.dil_w;
        if (num_x < 0 || (num_x % p.stride_w) != 0) continue;
        const int ox = num_x / p.stride_w;
        if (ox >= p.w_out) continue;
        const int fs = (p.conv_mode == 0) ? (p.kw - 1 - s) : s;

        const int dy_idx = b * p.y_ns + oc * p.y_cs + oy * p.y_hs + ox * p.y_ws;
        const int w_idx = filter_offset(oc, c_g, fr, fs, c_in_per_g, p.kh, p.kw, p.w_format);
        acc = fma(dy[dy_idx], w[w_idx], acc);
      }
    }
  }

  const int dx_idx = b * p.x_ns + ic * p.x_cs + iy * p.x_hs + ix * p.x_ws;
  const float prev_dx = (p.beta == 0.0f) ? 0.0f : (p.beta * dx[dx_idx]);
  dx[dx_idx] = fma(p.alpha, acc, prev_dx);
}

kernel void muda_cudnn_conv2d_bwd_filter(
    device const float* x [[buffer(0)]],
    device const float* dy [[buffer(1)]],
    device float* dw [[buffer(2)]],
    constant CudnnConvParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  const int c_in_per_g = p.c_in / p.groups;
  const uint total = uint(p.c_out) * uint(c_in_per_g) * uint(p.kh) * uint(p.kw);
  if (tid >= total) return;

  const int fs = int(tid % uint(p.kw));
  uint rem = tid / uint(p.kw);
  const int fr = int(rem % uint(p.kh));
  rem /= uint(p.kh);
  const int c_g = int(rem % uint(c_in_per_g));
  const int oc = int(rem / uint(c_in_per_g));

  const int c_out_per_g = p.c_out / p.groups;
  const int g = oc / c_out_per_g;
  const int ic = g * c_in_per_g + c_g;
  const int r = (p.conv_mode == 0) ? (p.kh - 1 - fr) : fr;
  const int s = (p.conv_mode == 0) ? (p.kw - 1 - fs) : fs;

  float acc = 0.0f;
  for (int b = 0; b < p.n; ++b) {
    for (int oy = 0; oy < p.h_out; ++oy) {
      const int iy = oy * p.stride_h - p.pad_h + r * p.dil_h;
      if (iy < 0 || iy >= p.h_in) continue;
      for (int ox = 0; ox < p.w_out; ++ox) {
        const int ix = ox * p.stride_w - p.pad_w + s * p.dil_w;
        if (ix < 0 || ix >= p.w_in) continue;
        const int x_idx = b * p.x_ns + ic * p.x_cs + iy * p.x_hs + ix * p.x_ws;
        const int dy_idx = b * p.y_ns + oc * p.y_cs + oy * p.y_hs + ox * p.y_ws;
        acc = fma(x[x_idx], dy[dy_idx], acc);
      }
    }
  }

  const int w_idx = filter_offset(oc, c_g, fr, fs, c_in_per_g, p.kh, p.kw, p.w_format);
  const float prev_dw = (p.beta == 0.0f) ? 0.0f : (p.beta * dw[w_idx]);
  dw[w_idx] = fma(p.alpha, acc, prev_dw);
}

kernel void muda_cudnn_conv2d_bwd_bias(
    device const float* dy [[buffer(0)]],
    device float* db [[buffer(1)]],
    constant CudnnConvParams& p [[buffer(2)]],
    uint oc [[thread_position_in_grid]]) {
  if (int(oc) >= p.c_out) return;
  float acc = 0.0f;
  for (int b = 0; b < p.n; ++b) {
    for (int oy = 0; oy < p.h_out; ++oy) {
      for (int ox = 0; ox < p.w_out; ++ox) {
        const int dy_idx = b * p.y_ns + int(oc) * p.y_cs + oy * p.y_hs + ox * p.y_ws;
        acc += dy[dy_idx];
      }
    }
  }
  const float prev_db = (p.beta == 0.0f) ? 0.0f : (p.beta * db[oc]);
  db[oc] = fma(p.alpha, acc, prev_db);
}

kernel void muda_cudnn_add_tensor(
    device const float* a [[buffer(0)]],
    device float* c [[buffer(1)]],
    constant CudnnTensorOpParams& p [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
  const uint total = uint(p.n) * uint(p.c) * uint(p.h) * uint(p.w);
  if (tid >= total) return;

  const int iw = int(tid % uint(p.w));
  uint rem = tid / uint(p.w);
  const int ih = int(rem % uint(p.h));
  rem /= uint(p.h);
  const int ic = int(rem % uint(p.c));
  const int in_b = int(rem / uint(p.c));

  const int an = (p.a_n == 1) ? 0 : in_b;
  const int ac = (p.a_c == 1) ? 0 : ic;
  const int ah = (p.a_h == 1) ? 0 : ih;
  const int aw = (p.a_w == 1) ? 0 : iw;

  const int a_idx = an * p.x_ns + ac * p.x_cs + ah * p.x_hs + aw * p.x_ws;
  const int c_idx = in_b * p.y_ns + ic * p.y_cs + ih * p.y_hs + iw * p.y_ws;
  const float prev_c = (p.beta == 0.0f) ? 0.0f : (p.beta * c[c_idx]);
  c[c_idx] = fma(p.alpha, a[a_idx], prev_c);
}

kernel void muda_cudnn_activation_fwd(
    device const float* x [[buffer(0)]],
    device float* y [[buffer(1)]],
    constant CudnnTensorOpParams& p [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
  const uint total = uint(p.n) * uint(p.c) * uint(p.h) * uint(p.w);
  if (tid >= total) return;

  const int iw = int(tid % uint(p.w));
  uint rem = tid / uint(p.w);
  const int ih = int(rem % uint(p.h));
  rem /= uint(p.h);
  const int ic = int(rem % uint(p.c));
  const int in_b = int(rem / uint(p.c));

  const int x_idx = in_b * p.x_ns + ic * p.x_cs + ih * p.x_hs + iw * p.x_ws;
  const int y_idx = in_b * p.y_ns + ic * p.y_cs + ih * p.y_hs + iw * p.y_ws;
  const float val = x[x_idx];
  float act = val;
  if (p.mode == 0) {
    act = 1.0f / (1.0f + exp(-val));
  } else if (p.mode == 1) {
    act = max(val, 0.0f);
  } else if (p.mode == 2) {
    act = tanh(val);
  } else if (p.mode == 3) {
    act = clamp(val, 0.0f, p.coef);
  } else if (p.mode == 4) {
    act = (val >= 0.0f) ? val : (p.coef * (exp(val) - 1.0f));
  } else if (p.mode == 6) {
    act = val / (1.0f + exp(-val));
  }
  const float prev_y = (p.beta == 0.0f) ? 0.0f : (p.beta * y[y_idx]);
  y[y_idx] = fma(p.alpha, act, prev_y);
}

inline float4 apply_activation_vec4(float4 val, int mode, float coef) {
  if (mode == 0) {
    return 1.0f / (1.0f + exp(-val));
  } else if (mode == 1) {
    return max(val, float4(0.0f));
  } else if (mode == 2) {
    return tanh(val);
  } else if (mode == 3) {
    return clamp(val, float4(0.0f), float4(coef));
  } else if (mode == 4) {
    return select(coef * (exp(val) - 1.0f), val, val >= float4(0.0f));
  } else if (mode == 6) {
    return val / (1.0f + exp(-val));
  }
  return val;
}

kernel void muda_cudnn_activation_fwd_vec4(
    device const float4* x [[buffer(0)]],
    device float4* y [[buffer(1)]],
    constant uint& n_vec4 [[buffer(2)]],
    constant int& mode [[buffer(3)]],
    constant float& alpha [[buffer(4)]],
    constant float& beta [[buffer(5)]],
    constant float& coef [[buffer(6)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid >= n_vec4) return;
  const float4 act = apply_activation_vec4(x[tid], mode, coef);
  const float4 prev_y = (beta == 0.0f) ? float4(0.0f) : (beta * y[tid]);
  y[tid] = fma(float4(alpha), act, prev_y);
}

kernel void muda_cudnn_pooling2d_fwd(
    device const float* x [[buffer(0)]],
    device float* y [[buffer(1)]],
    constant CudnnPoolParams& p [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
  const uint total = uint(p.n) * uint(p.c) * uint(p.h_out) * uint(p.w_out);
  if (tid >= total) return;

  const int ox = int(tid % uint(p.w_out));
  uint rem = tid / uint(p.w_out);
  const int oy = int(rem % uint(p.h_out));
  rem /= uint(p.h_out);
  const int ic = int(rem % uint(p.c));
  const int b = int(rem / uint(p.c));

  const bool is_max = (p.mode == 0 || p.mode == 3);
  float acc = is_max ? -INFINITY : 0.0f;
  int valid_count = 0;

  for (int r = 0; r < p.win_h; ++r) {
    const int iy = oy * p.stride_h - p.pad_h + r;
    if (iy < 0 || iy >= p.h_in) continue;
    for (int s = 0; s < p.win_w; ++s) {
      const int ix = ox * p.stride_w - p.pad_w + s;
      if (ix < 0 || ix >= p.w_in) continue;
      const int x_idx = b * p.x_ns + ic * p.x_cs + iy * p.x_hs + ix * p.x_ws;
      const float val = x[x_idx];
      if (is_max) {
        acc = max(acc, val);
      } else {
        acc += val;
      }
      ++valid_count;
    }
  }

  float out_val = acc;
  if (!is_max) {
    const int denom = (p.mode == 1) ? (p.win_h * p.win_w) : max(valid_count, 1);
    out_val = acc / float(denom);
  }
  const int y_idx = b * p.y_ns + ic * p.y_cs + oy * p.y_hs + ox * p.y_ws;
  const float prev_y = (p.beta == 0.0f) ? 0.0f : (p.beta * y[y_idx]);
  y[y_idx] = fma(p.alpha, out_val, prev_y);
}

kernel void muda_cudnn_softmax_fwd(
    device const float* x [[buffer(0)]],
    device float* y [[buffer(1)]],
    constant CudnnTensorOpParams& p [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
  const int sm_mode = p.mode & 1;
  const int algo = p.mode >> 1;

  if (sm_mode == 1) {
    const uint total = uint(p.n) * uint(p.h) * uint(p.w);
    if (tid >= total) return;
    const int iw = int(tid % uint(p.w));
    uint rem = tid / uint(p.w);
    const int ih = int(rem % uint(p.h));
    const int b = int(rem / uint(p.h));

    float max_v = -INFINITY;
    if (algo != 0) {
      for (int ic = 0; ic < p.c; ++ic) {
        const int x_idx = b * p.x_ns + ic * p.x_cs + ih * p.x_hs + iw * p.x_ws;
        max_v = max(max_v, x[x_idx]);
      }
    } else {
      max_v = 0.0f;
    }

    float sum_exp = 0.0f;
    for (int ic = 0; ic < p.c; ++ic) {
      const int x_idx = b * p.x_ns + ic * p.x_cs + ih * p.x_hs + iw * p.x_ws;
      sum_exp += exp(x[x_idx] - max_v);
    }
    const float log_sum = log(sum_exp);

    for (int ic = 0; ic < p.c; ++ic) {
      const int x_idx = b * p.x_ns + ic * p.x_cs + ih * p.x_hs + iw * p.x_ws;
      const int y_idx = b * p.y_ns + ic * p.y_cs + ih * p.y_hs + iw * p.y_ws;
      const float shifted = x[x_idx] - max_v;
      const float val = (algo == 2) ? (shifted - log_sum) : (exp(shifted) / sum_exp);
      const float prev_y = (p.beta == 0.0f) ? 0.0f : (p.beta * y[y_idx]);
      y[y_idx] = fma(p.alpha, val, prev_y);
    }
  } else {
    if (int(tid) >= p.n) return;
    const int b = int(tid);
    float max_v = -INFINITY;
    if (algo != 0) {
      for (int ic = 0; ic < p.c; ++ic) {
        for (int ih = 0; ih < p.h; ++ih) {
          for (int iw = 0; iw < p.w; ++iw) {
            max_v = max(max_v, x[b * p.x_ns + ic * p.x_cs + ih * p.x_hs + iw * p.x_ws]);
          }
        }
      }
    } else {
      max_v = 0.0f;
    }

    float sum_exp = 0.0f;
    for (int ic = 0; ic < p.c; ++ic) {
      for (int ih = 0; ih < p.h; ++ih) {
        for (int iw = 0; iw < p.w; ++iw) {
          sum_exp += exp(x[b * p.x_ns + ic * p.x_cs + ih * p.x_hs + iw * p.x_ws] - max_v);
        }
      }
    }
    const float log_sum = log(sum_exp);

    for (int ic = 0; ic < p.c; ++ic) {
      for (int ih = 0; ih < p.h; ++ih) {
        for (int iw = 0; iw < p.w; ++iw) {
          const int x_idx = b * p.x_ns + ic * p.x_cs + ih * p.x_hs + iw * p.x_ws;
          const int y_idx = b * p.y_ns + ic * p.y_cs + ih * p.y_hs + iw * p.y_ws;
          const float shifted = x[x_idx] - max_v;
          const float val = (algo == 2) ? (shifted - log_sum) : (exp(shifted) / sum_exp);
          const float prev_y = (p.beta == 0.0f) ? 0.0f : (p.beta * y[y_idx]);
          y[y_idx] = fma(p.alpha, val, prev_y);
        }
      }
    }
  }
}

kernel void muda_cudnn_softmax_warp_contiguous(
    device const float* x [[buffer(0)]],
    device float* y [[buffer(1)]],
    constant uint& num_slices [[buffer(2)]],
    constant uint& slice_len [[buffer(3)]],
    constant int& algo [[buffer(4)]],
    constant float& alpha [[buffer(5)]],
    constant float& beta [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]) {
  const uint slice = tgid * 8u + simd_id;
  if (slice >= num_slices) return;

  const uint base = slice * slice_len;
  float max_v = 0.0f;
  if (algo != 0) {
    float local_max = -INFINITY;
    for (uint i = simd_lane; i < slice_len; i += 32u) {
      local_max = max(local_max, x[base + i]);
    }
    max_v = simd_max(local_max);
  }

  float local_sum = 0.0f;
  for (uint i = simd_lane; i < slice_len; i += 32u) {
    local_sum += exp(x[base + i] - max_v);
  }
  const float sum_exp = simd_sum(local_sum);
  const float log_sum = log(sum_exp);
  const float inv_sum = 1.0f / sum_exp;

  for (uint i = simd_lane; i < slice_len; i += 32u) {
    const uint idx = base + i;
    const float shifted = x[idx] - max_v;
    const float val = (algo == 2) ? (shifted - log_sum) : (exp(shifted) * inv_sum);
    const float prev_y = (beta == 0.0f) ? 0.0f : (beta * y[idx]);
    y[idx] = fma(alpha, val, prev_y);
  }
}

kernel void muda_cudnn_batchnorm_fwd_inference(
    device const float* x [[buffer(0)]],
    device float* y [[buffer(1)]],
    device const float* scale [[buffer(2)]],
    device const float* bias [[buffer(3)]],
    device const float* mean [[buffer(4)]],
    device const float* var [[buffer(5)]],
    constant CudnnTensorOpParams& p [[buffer(6)]],
    uint tid [[thread_position_in_grid]]) {
  const uint total = uint(p.n) * uint(p.c) * uint(p.h) * uint(p.w);
  if (tid >= total) return;

  const uint hw = uint(p.h) * uint(p.w);
  if (p.x_ws == 1 && p.y_ws == 1 && uint(p.x_hs) == uint(p.w) && uint(p.y_hs) == uint(p.w) &&
      uint(p.x_cs) == hw && uint(p.y_cs) == hw && uint(p.x_ns) == uint(p.c) * hw &&
      uint(p.y_ns) == uint(p.c) * hw) {
    const uint chw = uint(p.c) * hw;
    const uint rem_n = tid % chw;
    const uint param_idx = (p.mode == 0) ? rem_n : (rem_n / hw);
    const float inv_std = rsqrt(var[param_idx] + p.coef);
    const float norm_val = fma((x[tid] - mean[param_idx]) * inv_std, scale[param_idx], bias[param_idx]);
    const float prev_y = (p.beta == 0.0f) ? 0.0f : (p.beta * y[tid]);
    y[tid] = fma(p.alpha, norm_val, prev_y);
    return;
  }

  const int iw = int(tid % uint(p.w));
  uint rem = tid / uint(p.w);
  const int ih = int(rem % uint(p.h));
  rem /= uint(p.h);
  const int ic = int(rem % uint(p.c));
  const int b = int(rem / uint(p.c));

  const int param_idx = (p.mode == 0) ? ((ic * p.h + ih) * p.w + iw) : ic;
  const int x_idx = b * p.x_ns + ic * p.x_cs + ih * p.x_hs + iw * p.x_ws;
  const int y_idx = b * p.y_ns + ic * p.y_cs + ih * p.y_hs + iw * p.y_ws;

  const float inv_std = rsqrt(var[param_idx] + p.coef);
  const float norm_val = fma((x[x_idx] - mean[param_idx]) * inv_std, scale[param_idx], bias[param_idx]);
  const float prev_y = (p.beta == 0.0f) ? 0.0f : (p.beta * y[y_idx]);
  y[y_idx] = fma(p.alpha, norm_val, prev_y);
}

kernel void muda_cudnn_batchnorm_fwd_training(
    device const float* x [[buffer(0)]],
    device float* y [[buffer(1)]],
    device const float* scale [[buffer(2)]],
    device const float* bias [[buffer(3)]],
    device float* running_mean [[buffer(4)]],
    device float* running_var [[buffer(5)]],
    device float* save_mean [[buffer(6)]],
    device float* save_inv_var [[buffer(7)]],
    constant CudnnTensorOpParams& p [[buffer(8)]],
    constant float& exp_avg_factor [[buffer(9)]],
    constant uint& has_running [[buffer(10)]],
    constant uint& has_save [[buffer(11)]],
    uint ic [[thread_position_in_grid]]) {
  if (int(ic) >= p.c) return;
  const int m_count = p.n * p.h * p.w;
  const float inv_m = 1.0f / float(m_count);

  float sum = 0.0f;
  for (int b = 0; b < p.n; ++b) {
    for (int ih = 0; ih < p.h; ++ih) {
      for (int iw = 0; iw < p.w; ++iw) {
        sum += x[b * p.x_ns + int(ic) * p.x_cs + ih * p.x_hs + iw * p.x_ws];
      }
    }
  }
  const float batch_mean = sum * inv_m;

  float sq_sum = 0.0f;
  for (int b = 0; b < p.n; ++b) {
    for (int ih = 0; ih < p.h; ++ih) {
      for (int iw = 0; iw < p.w; ++iw) {
        const float d = x[b * p.x_ns + int(ic) * p.x_cs + ih * p.x_hs + iw * p.x_ws] - batch_mean;
        sq_sum = fma(d, d, sq_sum);
      }
    }
  }
  const float batch_var = sq_sum * inv_m;
  const float unbiased_var = (m_count > 1) ? (sq_sum / float(m_count - 1)) : batch_var;
  const float inv_std = rsqrt(batch_var + p.coef);

  if (has_save != 0u) {
    save_mean[ic] = batch_mean;
    save_inv_var[ic] = inv_std;
  }
  if (has_running != 0u) {
    running_mean[ic] = (1.0f - exp_avg_factor) * running_mean[ic] + exp_avg_factor * batch_mean;
    running_var[ic] = (1.0f - exp_avg_factor) * running_var[ic] + exp_avg_factor * unbiased_var;
  }

  const float sc = scale[ic];
  const float bi = bias[ic];
  for (int b = 0; b < p.n; ++b) {
    for (int ih = 0; ih < p.h; ++ih) {
      for (int iw = 0; iw < p.w; ++iw) {
        const int x_idx = b * p.x_ns + int(ic) * p.x_cs + ih * p.x_hs + iw * p.x_ws;
        const int y_idx = b * p.y_ns + int(ic) * p.y_cs + ih * p.y_hs + iw * p.y_ws;
        const float norm_val = fma((x[x_idx] - batch_mean) * inv_std, sc, bi);
        const float prev_y = (p.beta == 0.0f) ? 0.0f : (p.beta * y[y_idx]);
        y[y_idx] = fma(p.alpha, norm_val, prev_y);
      }
    }
  }
}


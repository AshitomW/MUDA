/**
 * @file cublas_kernels.metal
 * @brief Metal compute kernels for cuBLAS Level-1, Level-2, and Level-3 operations,
 *        including Apple Silicon simdgroup_matrix 8x8 tiled microkernels.
 */

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

kernel void muda_cublas_saxpy(
    device const float* x [[buffer(0)]],
    device float* y [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant float& alpha [[buffer(3)]],
    constant int& incx [[buffer(4)]],
    constant int& incy [[buffer(5)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid >= n) return;
  const int ix = (incx >= 0) ? (int(tid) * incx) : ((int(tid) - int(n) + 1) * incx);
  const int iy = (incy >= 0) ? (int(tid) * incy) : ((int(tid) - int(n) + 1) * incy);
  y[iy] = fma(alpha, x[ix], y[iy]);
}

kernel void muda_cublas_saxpy_vec4(
    device const float4* x [[buffer(0)]],
    device float4* y [[buffer(1)]],
    constant uint& n_vec4 [[buffer(2)]],
    constant float& alpha [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid >= n_vec4) return;
  y[tid] = fma(float4(alpha), x[tid], y[tid]);
}

kernel void muda_cublas_scopy(
    device const float* x [[buffer(0)]],
    device float* y [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant int& incx [[buffer(3)]],
    constant int& incy [[buffer(4)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid >= n) return;
  const int ix = (incx >= 0) ? (int(tid) * incx) : ((int(tid) - int(n) + 1) * incx);
  const int iy = (incy >= 0) ? (int(tid) * incy) : ((int(tid) - int(n) + 1) * incy);
  y[iy] = x[ix];
}

kernel void muda_cublas_scopy_vec4(
    device const float4* x [[buffer(0)]],
    device float4* y [[buffer(1)]],
    constant uint& n_vec4 [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid >= n_vec4) return;
  y[tid] = x[tid];
}

kernel void muda_cublas_sscal(
    device float* x [[buffer(0)]],
    constant uint& n [[buffer(1)]],
    constant float& alpha [[buffer(2)]],
    constant int& incx [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid >= n) return;
  if (incx <= 0) return;
  const uint ix = tid * uint(incx);
  x[ix] = (alpha == 0.0f) ? 0.0f : (alpha * x[ix]);
}

kernel void muda_cublas_sscal_vec4(
    device float4* x [[buffer(0)]],
    constant uint& n_vec4 [[buffer(1)]],
    constant float& alpha [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid >= n_vec4) return;
  x[tid] = (alpha == 0.0f) ? float4(0.0f) : (alpha * x[tid]);
}

kernel void muda_cublas_sswap(
    device float* x [[buffer(0)]],
    device float* y [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant int& incx [[buffer(3)]],
    constant int& incy [[buffer(4)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid >= n) return;
  const int ix = (incx >= 0) ? (int(tid) * incx) : ((int(tid) - int(n) + 1) * incx);
  const int iy = (incy >= 0) ? (int(tid) * incy) : ((int(tid) - int(n) + 1) * incy);
  const float tmp = x[ix];
  x[ix] = y[iy];
  y[iy] = tmp;
}

kernel void muda_cublas_sdot(
    device const float* x [[buffer(0)]],
    device const float* y [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant int& incx [[buffer(4)]],
    constant int& incy [[buffer(5)]],
    uint lid [[thread_position_in_threadgroup]],
    uint tsize [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]) {
  threadgroup float warp_sums[8];
  float acc = 0.0f;
  if (incx == 1 && incy == 1) {
    for (uint i = lid; i < n; i += tsize) {
      acc = fma(x[i], y[i], acc);
    }
  } else {
    for (uint i = lid; i < n; i += tsize) {
      const int ix = (incx >= 0) ? (int(i) * incx) : ((int(i) - int(n) + 1) * incx);
      const int iy = (incy >= 0) ? (int(i) * incy) : ((int(i) - int(n) + 1) * incy);
      acc = fma(x[ix], y[iy], acc);
    }
  }
  acc = simd_sum(acc);
  if (simd_lane == 0u) {
    warp_sums[simd_id] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_id == 0u) {
    const uint num_warps = (tsize + 31u) >> 5u;
    float block_sum = (simd_lane < num_warps) ? warp_sums[simd_lane] : 0.0f;
    block_sum = simd_sum(block_sum);
    if (simd_lane == 0u) {
      out[0] = block_sum;
    }
  }
}

kernel void muda_cublas_sdot_stage1(
    device const float* x [[buffer(0)]],
    device const float* y [[buffer(1)]],
    device float* partial_sums [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant int& incx [[buffer(4)]],
    constant int& incy [[buffer(5)]],
    constant uint& use_vec4 [[buffer(6)]],
    uint tid [[thread_position_in_grid]],
    uint gsize [[threads_per_grid]],
    uint tsize [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]) {
  threadgroup float warp_sums[8];
  float acc = 0.0f;
  if (use_vec4 != 0u) {
    const uint n_vec4 = n >> 2u;
    device const float4* x4 = reinterpret_cast<device const float4*>(x);
    device const float4* y4 = reinterpret_cast<device const float4*>(y);
    for (uint i = tid; i < n_vec4; i += gsize) {
      acc += dot(x4[i], y4[i]);
    }
    const uint tail_idx = (n_vec4 << 2u) + tid;
    if (tail_idx < n) {
      acc = fma(x[tail_idx], y[tail_idx], acc);
    }
  } else if (incx == 1 && incy == 1) {
    for (uint i = tid; i < n; i += gsize) {
      acc = fma(x[i], y[i], acc);
    }
  } else {
    for (uint i = tid; i < n; i += gsize) {
      const int ix = (incx >= 0) ? (int(i) * incx) : ((int(i) - int(n) + 1) * incx);
      const int iy = (incy >= 0) ? (int(i) * incy) : ((int(i) - int(n) + 1) * incy);
      acc = fma(x[ix], y[iy], acc);
    }
  }
  acc = simd_sum(acc);
  if (simd_lane == 0u) {
    warp_sums[simd_id] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_id == 0u) {
    const uint num_warps = (tsize + 31u) >> 5u;
    float block_sum = (simd_lane < num_warps) ? warp_sums[simd_lane] : 0.0f;
    block_sum = simd_sum(block_sum);
    if (simd_lane == 0u) {
      partial_sums[tgid] = block_sum;
    }
  }
}

kernel void muda_cublas_snrm2(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant int& incx [[buffer(3)]],
    uint lid [[thread_position_in_threadgroup]],
    uint tsize [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]) {
  threadgroup float warp_sums[8];
  float acc = 0.0f;
  if (incx > 0) {
    for (uint i = lid; i < n; i += tsize) {
      const float val = x[i * uint(incx)];
      acc = fma(val, val, acc);
    }
  }
  acc = simd_sum(acc);
  if (simd_lane == 0u) {
    warp_sums[simd_id] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_id == 0u) {
    const uint num_warps = (tsize + 31u) >> 5u;
    float block_sum = (simd_lane < num_warps) ? warp_sums[simd_lane] : 0.0f;
    block_sum = simd_sum(block_sum);
    if (simd_lane == 0u) {
      out[0] = sqrt(block_sum);
    }
  }
}

kernel void muda_cublas_snrm2_stage1(
    device const float* x [[buffer(0)]],
    device float* partial_sums [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant int& incx [[buffer(3)]],
    constant uint& use_vec4 [[buffer(4)]],
    uint tid [[thread_position_in_grid]],
    uint gsize [[threads_per_grid]],
    uint tsize [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]) {
  threadgroup float warp_sums[8];
  float acc = 0.0f;
  if (use_vec4 != 0u) {
    const uint n_vec4 = n >> 2u;
    device const float4* x4 = reinterpret_cast<device const float4*>(x);
    for (uint i = tid; i < n_vec4; i += gsize) {
      const float4 v = x4[i];
      acc += dot(v, v);
    }
    const uint tail_idx = (n_vec4 << 2u) + tid;
    if (tail_idx < n) {
      const float v = x[tail_idx];
      acc = fma(v, v, acc);
    }
  } else if (incx > 0) {
    for (uint i = tid; i < n; i += gsize) {
      const float val = x[i * uint(incx)];
      acc = fma(val, val, acc);
    }
  }
  acc = simd_sum(acc);
  if (simd_lane == 0u) {
    warp_sums[simd_id] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_id == 0u) {
    const uint num_warps = (tsize + 31u) >> 5u;
    float block_sum = (simd_lane < num_warps) ? warp_sums[simd_lane] : 0.0f;
    block_sum = simd_sum(block_sum);
    if (simd_lane == 0u) {
      partial_sums[tgid] = block_sum;
    }
  }
}

kernel void muda_cublas_sasum(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant int& incx [[buffer(3)]],
    uint lid [[thread_position_in_threadgroup]],
    uint tsize [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]) {
  threadgroup float warp_sums[8];
  float acc = 0.0f;
  if (incx > 0) {
    for (uint i = lid; i < n; i += tsize) {
      acc += fabs(x[i * uint(incx)]);
    }
  }
  acc = simd_sum(acc);
  if (simd_lane == 0u) {
    warp_sums[simd_id] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_id == 0u) {
    const uint num_warps = (tsize + 31u) >> 5u;
    float block_sum = (simd_lane < num_warps) ? warp_sums[simd_lane] : 0.0f;
    block_sum = simd_sum(block_sum);
    if (simd_lane == 0u) {
      out[0] = block_sum;
    }
  }
}

kernel void muda_cublas_sasum_stage1(
    device const float* x [[buffer(0)]],
    device float* partial_sums [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant int& incx [[buffer(3)]],
    constant uint& use_vec4 [[buffer(4)]],
    uint tid [[thread_position_in_grid]],
    uint gsize [[threads_per_grid]],
    uint tsize [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]) {
  threadgroup float warp_sums[8];
  float acc = 0.0f;
  if (use_vec4 != 0u) {
    const uint n_vec4 = n >> 2u;
    device const float4* x4 = reinterpret_cast<device const float4*>(x);
    for (uint i = tid; i < n_vec4; i += gsize) {
      const float4 v = fabs(x4[i]);
      acc += (v.x + v.y) + (v.z + v.w);
    }
    const uint tail_idx = (n_vec4 << 2u) + tid;
    if (tail_idx < n) {
      acc += fabs(x[tail_idx]);
    }
  } else if (incx > 0) {
    for (uint i = tid; i < n; i += gsize) {
      acc += fabs(x[i * uint(incx)]);
    }
  }
  acc = simd_sum(acc);
  if (simd_lane == 0u) {
    warp_sums[simd_id] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_id == 0u) {
    const uint num_warps = (tsize + 31u) >> 5u;
    float block_sum = (simd_lane < num_warps) ? warp_sums[simd_lane] : 0.0f;
    block_sum = simd_sum(block_sum);
    if (simd_lane == 0u) {
      partial_sums[tgid] = block_sum;
    }
  }
}

kernel void muda_cublas_reduce_stage2(
    device const float* partial_sums [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& num_partials [[buffer(2)]],
    constant uint& apply_sqrt [[buffer(3)]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]) {
  threadgroup float warp_sums[2];
  float val = (lid < num_partials) ? partial_sums[lid] : 0.0f;
  val = simd_sum(val);
  if (simd_lane == 0u) {
    warp_sums[simd_id] = val;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_id == 0u) {
    float total = (simd_lane < 2u) ? warp_sums[simd_lane] : 0.0f;
    total = simd_sum(total);
    if (simd_lane == 0u) {
      out[0] = (apply_sqrt != 0u) ? sqrt(total) : total;
    }
  }
}

kernel void muda_cublas_isamax(
    device const float* x [[buffer(0)]],
    device int* out [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant int& incx [[buffer(3)]],
    uint lid [[thread_position_in_threadgroup]],
    uint tsize [[threads_per_threadgroup]]) {
  threadgroup float shared_val[256];
  threadgroup int shared_idx[256];

  float best_val = -1.0f;
  int best_idx = 0;
  if (incx > 0 && n > 0) {
    for (uint i = lid; i < n; i += tsize) {
      const float v = fabs(x[i * uint(incx)]);
      const int idx1 = int(i) + 1;
      if (v > best_val || (v == best_val && (best_idx == 0 || idx1 < best_idx))) {
        best_val = v;
        best_idx = idx1;
      }
    }
  }
  shared_val[lid] = best_val;
  shared_idx[lid] = best_idx;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint s = tsize >> 1; s > 0; s >>= 1) {
    if (lid < s) {
      const float v_other = shared_val[lid + s];
      const int idx_other = shared_idx[lid + s];
      if (idx_other > 0) {
        if (shared_idx[lid] == 0 || v_other > shared_val[lid] ||
            (v_other == shared_val[lid] && idx_other < shared_idx[lid])) {
          shared_val[lid] = v_other;
          shared_idx[lid] = idx_other;
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lid == 0) {
    out[0] = shared_idx[0];
  }
}

kernel void muda_cublas_sgemv(
    device const float* A [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& m [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& lda [[buffer(5)]],
    constant int& incx [[buffer(6)]],
    constant int& incy [[buffer(7)]],
    constant float& alpha [[buffer(8)]],
    constant float& beta [[buffer(9)]],
    constant uint& trans [[buffer(10)]],
    uint row [[thread_position_in_grid]]) {
  const uint out_len = (trans == 0u) ? m : n;
  const uint inner_len = (trans == 0u) ? n : m;
  if (row >= out_len) return;

  float sum = 0.0f;
  for (uint j = 0; j < inner_len; ++j) {
    const int ix = (incx >= 0) ? (int(j) * incx) : ((int(j) - int(inner_len) + 1) * incx);
    const float a_val = (trans == 0u) ? A[row + j * lda] : A[j + row * lda];
    sum = fma(a_val, x[ix], sum);
  }

  const int iy = (incy >= 0) ? (int(row) * incy) : ((int(row) - int(out_len) + 1) * incy);
  const float prev_y = (beta == 0.0f) ? 0.0f : (beta * y[iy]);
  y[iy] = fma(alpha, sum, prev_y);
}

kernel void muda_cublas_sgemv_n_fast(
    device const float* A [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& m [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& lda [[buffer(5)]],
    constant float& alpha [[buffer(6)]],
    constant float& beta [[buffer(7)]],
    uint row [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]]) {
  threadgroup float shared_x[256];
  float sum = 0.0f;
  for (uint j_base = 0; j_base < n; j_base += 256u) {
    const uint j_load = j_base + lid;
    shared_x[lid] = (j_load < n) ? x[j_load] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (row < m) {
      const uint tile_end = min(256u, n - j_base);
      device const float* A_col = A + row + j_base * lda;
      uint k = 0u;
      for (; k + 4u <= tile_end; k += 4u) {
        sum = fma(A_col[0], shared_x[k + 0u], sum);
        sum = fma(A_col[lda], shared_x[k + 1u], sum);
        sum = fma(A_col[2u * lda], shared_x[k + 2u], sum);
        sum = fma(A_col[3u * lda], shared_x[k + 3u], sum);
        A_col += 4u * lda;
      }
      for (; k < tile_end; ++k) {
        sum = fma(A_col[0], shared_x[k], sum);
        A_col += lda;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (row < m) {
    const float prev_y = (beta == 0.0f) ? 0.0f : (beta * y[row]);
    y[row] = fma(alpha, sum, prev_y);
  }
}

kernel void muda_cublas_sgemv_t_fast(
    device const float* A [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& m [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& lda [[buffer(5)]],
    constant float& alpha [[buffer(6)]],
    constant float& beta [[buffer(7)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]) {
  const uint col = tgid * 8u + simd_id;
  if (col >= n) return;

  device const float* A_col = A + col * lda;
  float sum = 0.0f;
  for (uint i = simd_lane; i < m; i += 32u) {
    sum = fma(A_col[i], x[i], sum);
  }
  sum = simd_sum(sum);
  if (simd_lane == 0u) {
    const float prev_y = (beta == 0.0f) ? 0.0f : (beta * y[col]);
    y[col] = fma(alpha, sum, prev_y);
  }
}

kernel void muda_cublas_sgemm_simdgroup(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    constant uint& lda [[buffer(6)]],
    constant uint& ldb [[buffer(7)]],
    constant uint& ldc [[buffer(8)]],
    constant float& alpha [[buffer(9)]],
    constant float& beta [[buffer(10)]],
    constant uint& transa [[buffer(11)]],
    constant uint& transb [[buffer(12)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]) {
  const uint base_m = tgid.x * 32u;
  const uint base_n = tgid.y * 32u;
  if (base_m >= M || base_n >= N) return;

  // Compute C^T (N x M, row-stride ldc) = B^T (N x K) * A^T (K x M)
  const uint sg_rn = (simd_id & 1u) * 16u;
  const uint sg_cm = (simd_id >> 1u) * 16u;
  const uint tid = simd_id * 32u + simd_lane;
  const bool full_tile = (base_m + 32u <= M) && (base_n + 32u <= N);

  threadgroup float s_left[512];   // 32 (N) x 16 (K)
  threadgroup float s_right[512];  // 16 (K) x 32 (M)

  simdgroup_float8x8 acc00 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  simdgroup_float8x8 acc01 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  simdgroup_float8x8 acc10 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  simdgroup_float8x8 acc11 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

  for (uint k_base = 0; k_base < K; k_base += 16u) {
    if (full_tile && (k_base + 16u <= K)) {
      for (uint idx = tid; idx < 512u; idx += 128u) {
        const uint rn = idx >> 4u;
        const uint ck = idx & 15u;
        const uint gn = base_n + rn;
        const uint gk = k_base + ck;
        s_left[rn * 16u + ck] = (transb == 0u) ? B[gk + gn * ldb] : B[gn + gk * ldb];

        const uint rk = idx >> 5u;
        const uint cm = idx & 31u;
        const uint agk = k_base + rk;
        const uint agm = base_m + cm;
        s_right[rk * 32u + cm] = (transa == 0u) ? A[agm + agk * lda] : A[agk + agm * lda];
      }
    } else {
      for (uint idx = tid; idx < 512u; idx += 128u) {
        const uint rn = idx >> 4u;
        const uint ck = idx & 15u;
        const uint gn = base_n + rn;
        const uint gk = k_base + ck;
        float val_b = 0.0f;
        if (gk < K && gn < N) {
          val_b = (transb == 0u) ? B[gk + gn * ldb] : B[gn + gk * ldb];
        }
        s_left[rn * 16u + ck] = val_b;

        const uint rk = idx >> 5u;
        const uint cm = idx & 31u;
        const uint agk = k_base + rk;
        const uint agm = base_m + cm;
        float val_a = 0.0f;
        if (agm < M && agk < K) {
          val_a = (transa == 0u) ? A[agm + agk * lda] : A[agk + agm * lda];
        }
        s_right[rk * 32u + cm] = val_a;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint ks = 0; ks < 16u; ks += 8u) {
      simdgroup_float8x8 fl0;
      simdgroup_float8x8 fl1;
      simdgroup_float8x8 fr0;
      simdgroup_float8x8 fr1;
      simdgroup_load(fl0, s_left + (sg_rn + 0u) * 16u + ks, 16u);
      simdgroup_load(fl1, s_left + (sg_rn + 8u) * 16u + ks, 16u);
      simdgroup_load(fr0, s_right + ks * 32u + sg_cm + 0u, 32u);
      simdgroup_load(fr1, s_right + ks * 32u + sg_cm + 8u, 32u);

      simdgroup_multiply_accumulate(acc00, fl0, fr0, acc00);
      simdgroup_multiply_accumulate(acc01, fl0, fr1, acc01);
      simdgroup_multiply_accumulate(acc10, fl1, fr0, acc10);
      simdgroup_multiply_accumulate(acc11, fl1, fr1, acc11);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (full_tile && alpha == 1.0f && beta == 0.0f) {
    simdgroup_store(acc00, C + (base_n + sg_rn + 0u) * ldc + (base_m + sg_cm + 0u), ldc);
    simdgroup_store(acc01, C + (base_n + sg_rn + 0u) * ldc + (base_m + sg_cm + 8u), ldc);
    simdgroup_store(acc10, C + (base_n + sg_rn + 8u) * ldc + (base_m + sg_cm + 0u), ldc);
    simdgroup_store(acc11, C + (base_n + sg_rn + 8u) * ldc + (base_m + sg_cm + 8u), ldc);
    return;
  }

  threadgroup float* sg_scratch = (simd_id < 2u) ? (s_left + simd_id * 256u)
                                                 : (s_right + (simd_id - 2u) * 256u);
  simdgroup_store(acc00, sg_scratch + 0u, 16u);
  simdgroup_store(acc01, sg_scratch + 8u, 16u);
  simdgroup_store(acc10, sg_scratch + 128u, 16u);
  simdgroup_store(acc11, sg_scratch + 136u, 16u);
  simdgroup_barrier(mem_flags::mem_threadgroup);

  for (uint idx = simd_lane; idx < 256u; idx += 32u) {
    const uint rn = idx >> 4u;
    const uint cm = idx & 15u;
    const uint gm = base_m + sg_cm + cm;
    const uint gn = base_n + sg_rn + rn;
    if (gm < M && gn < N) {
      const uint c_idx = gm + gn * ldc;
      const float prev_c = (beta == 0.0f) ? 0.0f : (beta * C[c_idx]);
      C[c_idx] = fma(alpha, sg_scratch[rn * 16u + cm], prev_c);
    }
  }
}

kernel void muda_cublas_sgemm_strided_batched_simdgroup(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    constant uint& lda [[buffer(6)]],
    constant uint& ldb [[buffer(7)]],
    constant uint& ldc [[buffer(8)]],
    constant long& strideA [[buffer(9)]],
    constant long& strideB [[buffer(10)]],
    constant long& strideC [[buffer(11)]],
    constant float& alpha [[buffer(12)]],
    constant float& beta [[buffer(13)]],
    constant uint& transa [[buffer(14)]],
    constant uint& transb [[buffer(15)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]) {
  const uint base_m = tgid.x * 32u;
  const uint base_n = tgid.y * 32u;
  const uint batch = tgid.z;
  if (base_m >= M || base_n >= N) return;

  device const float* Ab = A + batch * strideA;
  device const float* Bb = B + batch * strideB;
  device float* Cb = C + batch * strideC;

  const uint sg_rn = (simd_id & 1u) * 16u;
  const uint sg_cm = (simd_id >> 1u) * 16u;
  const uint tid = simd_id * 32u + simd_lane;
  const bool full_tile = (base_m + 32u <= M) && (base_n + 32u <= N);

  threadgroup float s_left[512];
  threadgroup float s_right[512];

  simdgroup_float8x8 acc00 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  simdgroup_float8x8 acc01 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  simdgroup_float8x8 acc10 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  simdgroup_float8x8 acc11 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

  for (uint k_base = 0; k_base < K; k_base += 16u) {
    if (full_tile && (k_base + 16u <= K)) {
      for (uint idx = tid; idx < 512u; idx += 128u) {
        const uint rn = idx >> 4u;
        const uint ck = idx & 15u;
        const uint gn = base_n + rn;
        const uint gk = k_base + ck;
        s_left[rn * 16u + ck] = (transb == 0u) ? Bb[gk + gn * ldb] : Bb[gn + gk * ldb];

        const uint rk = idx >> 5u;
        const uint cm = idx & 31u;
        const uint agk = k_base + rk;
        const uint agm = base_m + cm;
        s_right[rk * 32u + cm] = (transa == 0u) ? Ab[agm + agk * lda] : Ab[agk + agm * lda];
      }
    } else {
      for (uint idx = tid; idx < 512u; idx += 128u) {
        const uint rn = idx >> 4u;
        const uint ck = idx & 15u;
        const uint gn = base_n + rn;
        const uint gk = k_base + ck;
        float val_b = 0.0f;
        if (gk < K && gn < N) {
          val_b = (transb == 0u) ? Bb[gk + gn * ldb] : Bb[gn + gk * ldb];
        }
        s_left[rn * 16u + ck] = val_b;

        const uint rk = idx >> 5u;
        const uint cm = idx & 31u;
        const uint agk = k_base + rk;
        const uint agm = base_m + cm;
        float val_a = 0.0f;
        if (agm < M && agk < K) {
          val_a = (transa == 0u) ? Ab[agm + agk * lda] : Ab[agk + agm * lda];
        }
        s_right[rk * 32u + cm] = val_a;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint ks = 0; ks < 16u; ks += 8u) {
      simdgroup_float8x8 fl0;
      simdgroup_float8x8 fl1;
      simdgroup_float8x8 fr0;
      simdgroup_float8x8 fr1;
      simdgroup_load(fl0, s_left + (sg_rn + 0u) * 16u + ks, 16u);
      simdgroup_load(fl1, s_left + (sg_rn + 8u) * 16u + ks, 16u);
      simdgroup_load(fr0, s_right + ks * 32u + sg_cm + 0u, 32u);
      simdgroup_load(fr1, s_right + ks * 32u + sg_cm + 8u, 32u);

      simdgroup_multiply_accumulate(acc00, fl0, fr0, acc00);
      simdgroup_multiply_accumulate(acc01, fl0, fr1, acc01);
      simdgroup_multiply_accumulate(acc10, fl1, fr0, acc10);
      simdgroup_multiply_accumulate(acc11, fl1, fr1, acc11);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (full_tile && alpha == 1.0f && beta == 0.0f) {
    simdgroup_store(acc00, Cb + (base_n + sg_rn + 0u) * ldc + (base_m + sg_cm + 0u), ldc);
    simdgroup_store(acc01, Cb + (base_n + sg_rn + 0u) * ldc + (base_m + sg_cm + 8u), ldc);
    simdgroup_store(acc10, Cb + (base_n + sg_rn + 8u) * ldc + (base_m + sg_cm + 0u), ldc);
    simdgroup_store(acc11, Cb + (base_n + sg_rn + 8u) * ldc + (base_m + sg_cm + 8u), ldc);
    return;
  }

  threadgroup float* sg_scratch = (simd_id < 2u) ? (s_left + simd_id * 256u)
                                                 : (s_right + (simd_id - 2u) * 256u);
  simdgroup_store(acc00, sg_scratch + 0u, 16u);
  simdgroup_store(acc01, sg_scratch + 8u, 16u);
  simdgroup_store(acc10, sg_scratch + 128u, 16u);
  simdgroup_store(acc11, sg_scratch + 136u, 16u);
  simdgroup_barrier(mem_flags::mem_threadgroup);

  for (uint idx = simd_lane; idx < 256u; idx += 32u) {
    const uint rn = idx >> 4u;
    const uint cm = idx & 15u;
    const uint gm = base_m + sg_cm + cm;
    const uint gn = base_n + sg_rn + rn;
    if (gm < M && gn < N) {
      const uint c_idx = gm + gn * ldc;
      const float prev_c = (beta == 0.0f) ? 0.0f : (beta * Cb[c_idx]);
      Cb[c_idx] = fma(alpha, sg_scratch[rn * 16u + cm], prev_c);
    }
  }
}

kernel void muda_cublas_hgemm_strided_batched_simdgroup(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    constant uint& lda [[buffer(6)]],
    constant uint& ldb [[buffer(7)]],
    constant uint& ldc [[buffer(8)]],
    constant long& strideA [[buffer(9)]],
    constant long& strideB [[buffer(10)]],
    constant long& strideC [[buffer(11)]],
    constant float& alpha [[buffer(12)]],
    constant float& beta [[buffer(13)]],
    constant uint& transa [[buffer(14)]],
    constant uint& transb [[buffer(15)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]) {
  const uint base_m = tgid.x * 32u;
  const uint base_n = tgid.y * 32u;
  const uint batch = tgid.z;
  if (base_m >= M || base_n >= N) return;

  device const half* Ab = A + batch * strideA;
  device const half* Bb = B + batch * strideB;
  device half* Cb = C + batch * strideC;

  const uint sg_rn = (simd_id & 1u) * 16u;
  const uint sg_cm = (simd_id >> 1u) * 16u;
  const uint tid = simd_id * 32u + simd_lane;
  const bool full_tile = (base_m + 32u <= M) && (base_n + 32u <= N);

  threadgroup float s_left[512];
  threadgroup float s_right[512];

  simdgroup_float8x8 acc00 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  simdgroup_float8x8 acc01 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  simdgroup_float8x8 acc10 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  simdgroup_float8x8 acc11 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

  for (uint k_base = 0; k_base < K; k_base += 16u) {
    if (full_tile && (k_base + 16u <= K)) {
      for (uint idx = tid; idx < 512u; idx += 128u) {
        const uint rn = idx >> 4u;
        const uint ck = idx & 15u;
        const uint gn = base_n + rn;
        const uint gk = k_base + ck;
        s_left[rn * 16u + ck] = float((transb == 0u) ? Bb[gk + gn * ldb] : Bb[gn + gk * ldb]);

        const uint rk = idx >> 5u;
        const uint cm = idx & 31u;
        const uint agk = k_base + rk;
        const uint agm = base_m + cm;
        s_right[rk * 32u + cm] = float((transa == 0u) ? Ab[agm + agk * lda] : Ab[agk + agm * lda]);
      }
    } else {
      for (uint idx = tid; idx < 512u; idx += 128u) {
        const uint rn = idx >> 4u;
        const uint ck = idx & 15u;
        const uint gn = base_n + rn;
        const uint gk = k_base + ck;
        float val_b = 0.0f;
        if (gk < K && gn < N) {
          val_b = float((transb == 0u) ? Bb[gk + gn * ldb] : Bb[gn + gk * ldb]);
        }
        s_left[rn * 16u + ck] = val_b;

        const uint rk = idx >> 5u;
        const uint cm = idx & 31u;
        const uint agk = k_base + rk;
        const uint agm = base_m + cm;
        float val_a = 0.0f;
        if (agm < M && agk < K) {
          val_a = float((transa == 0u) ? Ab[agm + agk * lda] : Ab[agk + agm * lda]);
        }
        s_right[rk * 32u + cm] = val_a;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint ks = 0; ks < 16u; ks += 8u) {
      simdgroup_float8x8 fl0;
      simdgroup_float8x8 fl1;
      simdgroup_float8x8 fr0;
      simdgroup_float8x8 fr1;
      simdgroup_load(fl0, s_left + (sg_rn + 0u) * 16u + ks, 16u);
      simdgroup_load(fl1, s_left + (sg_rn + 8u) * 16u + ks, 16u);
      simdgroup_load(fr0, s_right + ks * 32u + sg_cm + 0u, 32u);
      simdgroup_load(fr1, s_right + ks * 32u + sg_cm + 8u, 32u);

      simdgroup_multiply_accumulate(acc00, fl0, fr0, acc00);
      simdgroup_multiply_accumulate(acc01, fl0, fr1, acc01);
      simdgroup_multiply_accumulate(acc10, fl1, fr0, acc10);
      simdgroup_multiply_accumulate(acc11, fl1, fr1, acc11);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  threadgroup float* sg_scratch = (simd_id < 2u) ? (s_left + simd_id * 256u)
                                                 : (s_right + (simd_id - 2u) * 256u);
  simdgroup_store(acc00, sg_scratch + 0u, 16u);
  simdgroup_store(acc01, sg_scratch + 8u, 16u);
  simdgroup_store(acc10, sg_scratch + 128u, 16u);
  simdgroup_store(acc11, sg_scratch + 136u, 16u);
  simdgroup_barrier(mem_flags::mem_threadgroup);

  for (uint idx = simd_lane; idx < 256u; idx += 32u) {
    const uint rn = idx >> 4u;
    const uint cm = idx & 15u;
    const uint gm = base_m + sg_cm + cm;
    const uint gn = base_n + sg_rn + rn;
    if (gm < M && gn < N) {
      const uint c_idx = gm + gn * ldc;
      const float prev_c = (beta == 0.0f) ? 0.0f : (beta * float(Cb[c_idx]));
      Cb[c_idx] = half(fma(alpha, sg_scratch[rn * 16u + cm], prev_c));
    }
  }
}

kernel void muda_cublas_sgemm_strided_batched(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    constant uint& lda [[buffer(6)]],
    constant uint& ldb [[buffer(7)]],
    constant uint& ldc [[buffer(8)]],
    constant long& strideA [[buffer(9)]],
    constant long& strideB [[buffer(10)]],
    constant long& strideC [[buffer(11)]],
    constant float& alpha [[buffer(12)]],
    constant float& beta [[buffer(13)]],
    constant uint& transa [[buffer(14)]],
    constant uint& transb [[buffer(15)]],
    uint3 gid [[thread_position_in_grid]]) {
  const uint row = gid.x;
  const uint col = gid.y;
  const uint batch = gid.z;
  if (row >= M || col >= N) return;

  device const float* Ab = A + batch * strideA;
  device const float* Bb = B + batch * strideB;
  device float* Cb = C + batch * strideC;

  float acc = 0.0f;
  for (uint p = 0; p < K; ++p) {
    const float a_val = (transa == 0u) ? Ab[row + p * lda] : Ab[p + row * lda];
    const float b_val = (transb == 0u) ? Bb[p + col * ldb] : Bb[col + p * ldb];
    acc = fma(a_val, b_val, acc);
  }

  const uint c_idx = row + col * ldc;
  const float prev_c = (beta == 0.0f) ? 0.0f : (beta * Cb[c_idx]);
  Cb[c_idx] = fma(alpha, acc, prev_c);
}

kernel void muda_cublas_hgemm_strided_batched(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    constant uint& lda [[buffer(6)]],
    constant uint& ldb [[buffer(7)]],
    constant uint& ldc [[buffer(8)]],
    constant long& strideA [[buffer(9)]],
    constant long& strideB [[buffer(10)]],
    constant long& strideC [[buffer(11)]],
    constant float& alpha [[buffer(12)]],
    constant float& beta [[buffer(13)]],
    constant uint& transa [[buffer(14)]],
    constant uint& transb [[buffer(15)]],
    uint3 gid [[thread_position_in_grid]]) {
  const uint row = gid.x;
  const uint col = gid.y;
  const uint batch = gid.z;
  if (row >= M || col >= N) return;

  device const half* Ab = A + batch * strideA;
  device const half* Bb = B + batch * strideB;
  device half* Cb = C + batch * strideC;

  float acc = 0.0f;
  for (uint p = 0; p < K; ++p) {
    const float a_val = float((transa == 0u) ? Ab[row + p * lda] : Ab[p + row * lda]);
    const float b_val = float((transb == 0u) ? Bb[p + col * ldb] : Bb[col + p * ldb]);
    acc = fma(a_val, b_val, acc);
  }

  const uint c_idx = row + col * ldc;
  const float prev_c = (beta == 0.0f) ? 0.0f : (beta * float(Cb[c_idx]));
  Cb[c_idx] = half(fma(alpha, acc, prev_c));
}

kernel void muda_cublas_sgeam(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& lda [[buffer(5)]],
    constant uint& ldb [[buffer(6)]],
    constant uint& ldc [[buffer(7)]],
    constant float& alpha [[buffer(8)]],
    constant float& beta [[buffer(9)]],
    constant uint& transa [[buffer(10)]],
    constant uint& transb [[buffer(11)]],
    uint2 gid [[thread_position_in_grid]]) {
  const uint row = gid.x;
  const uint col = gid.y;
  if (row >= M || col >= N) return;

  float term_a = 0.0f;
  if (alpha != 0.0f) {
    const float a_val = (transa == 0u) ? A[row + col * lda] : A[col + row * lda];
    term_a = alpha * a_val;
  }
  float term_b = 0.0f;
  if (beta != 0.0f) {
    const float b_val = (transb == 0u) ? B[row + col * ldb] : B[col + row * ldb];
    term_b = beta * b_val;
  }
  C[row + col * ldc] = term_a + term_b;
}

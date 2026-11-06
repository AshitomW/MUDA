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
    uint tsize [[threads_per_threadgroup]]) {
  threadgroup float shared_sum[256];
  float acc = 0.0f;
  for (uint i = lid; i < n; i += tsize) {
    const int ix = (incx >= 0) ? (int(i) * incx) : ((int(i) - int(n) + 1) * incx);
    const int iy = (incy >= 0) ? (int(i) * incy) : ((int(i) - int(n) + 1) * incy);
    acc = fma(x[ix], y[iy], acc);
  }
  shared_sum[lid] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = tsize >> 1; s > 0; s >>= 1) {
    if (lid < s) {
      shared_sum[lid] += shared_sum[lid + s];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lid == 0) {
    out[0] = shared_sum[0];
  }
}

kernel void muda_cublas_snrm2(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant int& incx [[buffer(3)]],
    uint lid [[thread_position_in_threadgroup]],
    uint tsize [[threads_per_threadgroup]]) {
  threadgroup float shared_sum[256];
  float acc = 0.0f;
  if (incx > 0) {
    for (uint i = lid; i < n; i += tsize) {
      const float val = x[i * uint(incx)];
      acc = fma(val, val, acc);
    }
  }
  shared_sum[lid] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = tsize >> 1; s > 0; s >>= 1) {
    if (lid < s) {
      shared_sum[lid] += shared_sum[lid + s];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lid == 0) {
    out[0] = sqrt(shared_sum[0]);
  }
}

kernel void muda_cublas_sasum(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant int& incx [[buffer(3)]],
    uint lid [[thread_position_in_threadgroup]],
    uint tsize [[threads_per_threadgroup]]) {
  threadgroup float shared_sum[256];
  float acc = 0.0f;
  if (incx > 0) {
    for (uint i = lid; i < n; i += tsize) {
      acc += fabs(x[i * uint(incx)]);
    }
  }
  shared_sum[lid] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = tsize >> 1; s > 0; s >>= 1) {
    if (lid < s) {
      shared_sum[lid] += shared_sum[lid + s];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lid == 0) {
    out[0] = shared_sum[0];
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
  const uint sg_row = (simd_id & 1u) * 8u;
  const uint sg_col = (simd_id >> 1u) * 8u;
  const uint base_m = tgid.x * 16u + sg_row;
  const uint base_n = tgid.y * 16u + sg_col;
  if (base_m >= M || base_n >= N) return;

  threadgroup float scratch[256];
  threadgroup float* sg_scratch = scratch + simd_id * 64u;

  simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

  for (uint k_base = 0; k_base < K; k_base += 8u) {
    for (uint idx = simd_lane; idx < 64u; idx += 32u) {
      const uint r = idx >> 3u;
      const uint c = idx & 7u;
      const uint gm = base_m + r;
      const uint gk = k_base + c;
      float val = 0.0f;
      if (gm < M && gk < K) {
        val = (transa == 0u) ? A[gm + gk * lda] : A[gk + gm * lda];
      }
      sg_scratch[r * 8u + c] = val;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 matA;
    simdgroup_load(matA, sg_scratch, 8u);
    simdgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = simd_lane; idx < 64u; idx += 32u) {
      const uint r = idx >> 3u;
      const uint c = idx & 7u;
      const uint gk = k_base + r;
      const uint gn = base_n + c;
      float val = 0.0f;
      if (gk < K && gn < N) {
        val = (transb == 0u) ? B[gk + gn * ldb] : B[gn + gk * ldb];
      }
      sg_scratch[r * 8u + c] = val;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 matB;
    simdgroup_load(matB, sg_scratch, 8u);
    simdgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_multiply_accumulate(acc, matA, matB, acc);
  }

  simdgroup_store(acc, sg_scratch, 8u);
  simdgroup_barrier(mem_flags::mem_threadgroup);

  for (uint idx = simd_lane; idx < 64u; idx += 32u) {
    const uint r = idx >> 3u;
    const uint c = idx & 7u;
    const uint gm = base_m + r;
    const uint gn = base_n + c;
    if (gm < M && gn < N) {
      const uint c_idx = gm + gn * ldc;
      const float prev_c = (beta == 0.0f) ? 0.0f : (beta * C[c_idx]);
      C[c_idx] = fma(alpha, sg_scratch[r * 8u + c], prev_c);
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

/**
 * @file cusparse_kernels.metal
 * @brief Metal compute kernels for CSR SpMV and SpMM (FP32 and FP16, 0/1-based indexing).
 */

#include <metal_stdlib>
using namespace metal;

kernel void muda_cusparse_spmv_csr_f32(
    device const int* row_offsets [[buffer(0)]],
    device const int* col_indices [[buffer(1)]],
    device const float* values [[buffer(2)]],
    device const float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    constant uint& cols [[buffer(6)]],
    constant int& idx_base [[buffer(7)]],
    constant float& alpha [[buffer(8)]],
    constant float& beta [[buffer(9)]],
    constant uint& op_a [[buffer(10)]],
    uint tid [[thread_position_in_grid]]) {
  if (op_a == 0u) {
    if (tid >= rows) return;
    const int start = row_offsets[tid] - idx_base;
    const int end = row_offsets[tid + 1] - idx_base;
    float sum = 0.0f;
    for (int idx = start; idx < end; ++idx) {
      const int col = col_indices[idx] - idx_base;
      sum = fma(values[idx], x[col], sum);
    }
    const float prev = (beta == 0.0f) ? 0.0f : (beta * y[tid]);
    y[tid] = fma(alpha, sum, prev);
  } else {
    if (tid >= cols) return;
    float sum = 0.0f;
    for (uint r = 0; r < rows; ++r) {
      const int start = row_offsets[r] - idx_base;
      const int end = row_offsets[r + 1] - idx_base;
      for (int idx = start; idx < end; ++idx) {
        if (uint(col_indices[idx] - idx_base) == tid) {
          sum = fma(values[idx], x[r], sum);
        }
      }
    }
    const float prev = (beta == 0.0f) ? 0.0f : (beta * y[tid]);
    y[tid] = fma(alpha, sum, prev);
  }
}

kernel void muda_cusparse_spmv_csr_f16(
    device const int* row_offsets [[buffer(0)]],
    device const int* col_indices [[buffer(1)]],
    device const half* values [[buffer(2)]],
    device const half* x [[buffer(3)]],
    device half* y [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    constant uint& cols [[buffer(6)]],
    constant int& idx_base [[buffer(7)]],
    constant float& alpha [[buffer(8)]],
    constant float& beta [[buffer(9)]],
    constant uint& op_a [[buffer(10)]],
    uint tid [[thread_position_in_grid]]) {
  if (op_a == 0u) {
    if (tid >= rows) return;
    const int start = row_offsets[tid] - idx_base;
    const int end = row_offsets[tid + 1] - idx_base;
    float sum = 0.0f;
    for (int idx = start; idx < end; ++idx) {
      const int col = col_indices[idx] - idx_base;
      sum = fma(float(values[idx]), float(x[col]), sum);
    }
    const float prev = (beta == 0.0f) ? 0.0f : (beta * float(y[tid]));
    y[tid] = half(fma(alpha, sum, prev));
  } else {
    if (tid >= cols) return;
    float sum = 0.0f;
    for (uint r = 0; r < rows; ++r) {
      const int start = row_offsets[r] - idx_base;
      const int end = row_offsets[r + 1] - idx_base;
      for (int idx = start; idx < end; ++idx) {
        if (uint(col_indices[idx] - idx_base) == tid) {
          sum = fma(float(values[idx]), float(x[r]), sum);
        }
      }
    }
    const float prev = (beta == 0.0f) ? 0.0f : (beta * float(y[tid]));
    y[tid] = half(fma(alpha, sum, prev));
  }
}

kernel void muda_cusparse_spmv_csr_warp_f32(
    device const int* row_offsets [[buffer(0)]],
    device const int* col_indices [[buffer(1)]],
    device const float* values [[buffer(2)]],
    device const float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    constant int& idx_base [[buffer(6)]],
    constant float& alpha [[buffer(7)]],
    constant float& beta [[buffer(8)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]) {
  const uint row = tgid * 8u + simd_id;
  if (row >= rows) return;
  const int start = row_offsets[row] - idx_base;
  const int end = row_offsets[row + 1] - idx_base;
  float sum = 0.0f;
  for (int idx = start + int(simd_lane); idx < end; idx += 32) {
    const int col = col_indices[idx] - idx_base;
    sum = fma(values[idx], x[col], sum);
  }
  sum = simd_sum(sum);
  if (simd_lane == 0u) {
    const float prev = (beta == 0.0f) ? 0.0f : (beta * y[row]);
    y[row] = fma(alpha, sum, prev);
  }
}

kernel void muda_cusparse_spmv_csr_warp_f16(
    device const int* row_offsets [[buffer(0)]],
    device const int* col_indices [[buffer(1)]],
    device const half* values [[buffer(2)]],
    device const half* x [[buffer(3)]],
    device half* y [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    constant int& idx_base [[buffer(6)]],
    constant float& alpha [[buffer(7)]],
    constant float& beta [[buffer(8)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]) {
  const uint row = tgid * 8u + simd_id;
  if (row >= rows) return;
  const int start = row_offsets[row] - idx_base;
  const int end = row_offsets[row + 1] - idx_base;
  float sum = 0.0f;
  for (int idx = start + int(simd_lane); idx < end; idx += 32) {
    const int col = col_indices[idx] - idx_base;
    sum = fma(float(values[idx]), float(x[col]), sum);
  }
  sum = simd_sum(sum);
  if (simd_lane == 0u) {
    const float prev = (beta == 0.0f) ? 0.0f : (beta * float(y[row]));
    y[row] = half(fma(alpha, sum, prev));
  }
}

inline float load_dense_f32(device const float* mat, uint r, uint c, uint ld, uint order, uint op) {
  const uint eff_r = (op == 0u) ? r : c;
  const uint eff_c = (op == 0u) ? c : r;
  return (order == 1u) ? mat[eff_r + eff_c * ld] : mat[eff_r * ld + eff_c];
}

kernel void muda_cusparse_spmm_csr_f32(
    device const int* row_offsets [[buffer(0)]],
    device const int* col_indices [[buffer(1)]],
    device const float* values [[buffer(2)]],
    device const float* B [[buffer(3)]],
    device float* C [[buffer(4)]],
    constant uint& a_rows [[buffer(5)]],
    constant uint& a_cols [[buffer(6)]],
    constant uint& n_cols [[buffer(7)]],
    constant uint& ldb [[buffer(8)]],
    constant uint& ldc [[buffer(9)]],
    constant uint& order_b [[buffer(10)]],
    constant uint& order_c [[buffer(11)]],
    constant uint& op_a [[buffer(12)]],
    constant uint& op_b [[buffer(13)]],
    constant int& idx_base [[buffer(14)]],
    constant float& alpha [[buffer(15)]],
    constant float& beta [[buffer(16)]],
    uint2 gid [[thread_position_in_grid]]) {
  const uint out_rows = (op_a == 0u) ? a_rows : a_cols;
  const uint col = gid.x;
  const uint row = gid.y;
  if (row >= out_rows || col >= n_cols) return;

  float sum = 0.0f;
  if (op_a == 0u) {
    const int start = row_offsets[row] - idx_base;
    const int end = row_offsets[row + 1] - idx_base;
    for (int idx = start; idx < end; ++idx) {
      const uint k = uint(col_indices[idx] - idx_base);
      const float b_val = load_dense_f32(B, k, col, ldb, order_b, op_b);
      sum = fma(values[idx], b_val, sum);
    }
  } else {
    for (uint r = 0; r < a_rows; ++r) {
      const int start = row_offsets[r] - idx_base;
      const int end = row_offsets[r + 1] - idx_base;
      for (int idx = start; idx < end; ++idx) {
        if (uint(col_indices[idx] - idx_base) == row) {
          const float b_val = load_dense_f32(B, r, col, ldb, order_b, op_b);
          sum = fma(values[idx], b_val, sum);
        }
      }
    }
  }

  const uint c_idx = (order_c == 1u) ? (row + col * ldc) : (row * ldc + col);
  const float prev_c = (beta == 0.0f) ? 0.0f : (beta * C[c_idx]);
  C[c_idx] = fma(alpha, sum, prev_c);
}

kernel void muda_cusparse_spmm_csr_vec4_f32(
    device const int* row_offsets [[buffer(0)]],
    device const int* col_indices [[buffer(1)]],
    device const float* values [[buffer(2)]],
    device const float4* B4 [[buffer(3)]],
    device float4* C4 [[buffer(4)]],
    constant uint& a_rows [[buffer(5)]],
    constant uint& n_cols_vec4 [[buffer(6)]],
    constant uint& ldb_vec4 [[buffer(7)]],
    constant uint& ldc_vec4 [[buffer(8)]],
    constant int& idx_base [[buffer(9)]],
    constant float& alpha [[buffer(10)]],
    constant float& beta [[buffer(11)]],
    uint2 gid [[thread_position_in_grid]]) {
  const uint col4 = gid.x;
  const uint row = gid.y;
  if (row >= a_rows || col4 >= n_cols_vec4) return;

  const int start = row_offsets[row] - idx_base;
  const int end = row_offsets[row + 1] - idx_base;
  float4 sum = float4(0.0f);
  for (int idx = start; idx < end; ++idx) {
    const uint k = uint(col_indices[idx] - idx_base);
    const float4 b_val = B4[k * ldb_vec4 + col4];
    sum = fma(float4(values[idx]), b_val, sum);
  }

  const uint c_idx = row * ldc_vec4 + col4;
  const float4 prev_c = (beta == 0.0f) ? float4(0.0f) : (beta * C4[c_idx]);
  C4[c_idx] = fma(float4(alpha), sum, prev_c);
}


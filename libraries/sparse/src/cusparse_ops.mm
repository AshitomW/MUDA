/**
 * @file cusparse_ops.mm
 * @brief Implementation of cuSPARSE CSR SpMV and SpMM operations on Apple Silicon GPUs.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <array>
#include <string_view>

#include "cusparse_descriptors.hpp"
#include "muda/libraries/metal_executor.hpp"

namespace muda::libraries::sparse {
namespace {

[[nodiscard]] id<MTLComputePipelineState> GetCusparsePipeline(
    std::string_view function_name) noexcept {
  return LibraryExecutor::Instance().GetOrLoadPipeline(
      MUDA_CUSPARSE_METALLIB_PATH, MUDA_CUSPARSE_METAL_SOURCE_PATH, function_name);
}

[[nodiscard]] constexpr std::size_t DenseMatSpanElements(const DnMatDescriptor& m) noexcept {
  if (m.rows <= 0 || m.cols <= 0) {
    return 0;
  }
  if (m.order == CUSPARSE_ORDER_COL) {
    return static_cast<std::size_t>(m.ld) * static_cast<std::size_t>(m.cols - 1) +
           static_cast<std::size_t>(m.rows);
  }
  return static_cast<std::size_t>(m.ld) * static_cast<std::size_t>(m.rows - 1) +
         static_cast<std::size_t>(m.cols);
}

}  // namespace
}  // namespace muda::libraries::sparse

using namespace muda::libraries;
using namespace muda::libraries::sparse;

extern "C" {

cusparseStatus_t cusparseSpMV_bufferSize(cusparseHandle_t handle, cusparseOperation_t opA,
                                         const void* alpha, cusparseSpMatDescr_t matA,
                                         cusparseDnVecDescr_t vecX, const void* beta,
                                         cusparseDnVecDescr_t vecY, cudaDataType /*computeType*/,
                                         cusparseSpMVAlg_t /*alg*/, size_t* bufferSize) {
  if (bufferSize == nullptr || alpha == nullptr || beta == nullptr) {
    return CUSPARSE_STATUS_INVALID_VALUE;
  }
  auto ctx = GetCusparseHandleTable().Lookup(handle);
  auto a_d = GetSpMatDescTable().Lookup(matA);
  auto x_d = GetDnVecDescTable().Lookup(vecX);
  auto y_d = GetDnVecDescTable().Lookup(vecY);
  if (ctx == nullptr) {
    return CUSPARSE_STATUS_NOT_INITIALIZED;
  }
  if (a_d == nullptr || x_d == nullptr || y_d == nullptr) {
    return CUSPARSE_STATUS_INVALID_VALUE;
  }
  const int64_t expected_x = (opA == CUSPARSE_OPERATION_NON_TRANSPOSE) ? a_d->cols : a_d->rows;
  const int64_t expected_y = (opA == CUSPARSE_OPERATION_NON_TRANSPOSE) ? a_d->rows : a_d->cols;
  if (x_d->size != expected_x || y_d->size != expected_y) {
    return CUSPARSE_STATUS_INVALID_VALUE;
  }
  *bufferSize = 0;
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSpMV(cusparseHandle_t handle, cusparseOperation_t opA, const void* alpha,
                              cusparseSpMatDescr_t matA, cusparseDnVecDescr_t vecX,
                              const void* beta, cusparseDnVecDescr_t vecY,
                              cudaDataType /*computeType*/, cusparseSpMVAlg_t /*alg*/,
                              void* /*externalBuffer*/) {
  if (alpha == nullptr || beta == nullptr) {
    return CUSPARSE_STATUS_INVALID_VALUE;
  }
  auto ctx = GetCusparseHandleTable().Lookup(handle);
  if (ctx == nullptr) {
    return CUSPARSE_STATUS_NOT_INITIALIZED;
  }
  auto a_d = GetSpMatDescTable().Lookup(matA);
  auto x_d = GetDnVecDescTable().Lookup(vecX);
  auto y_d = GetDnVecDescTable().Lookup(vecY);
  if (a_d == nullptr || x_d == nullptr || y_d == nullptr) {
    return CUSPARSE_STATUS_INVALID_VALUE;
  }
  if (a_d->value_type != x_d->value_type || a_d->value_type != y_d->value_type) {
    return CUSPARSE_STATUS_NOT_SUPPORTED;
  }

  const int64_t expected_x = (opA == CUSPARSE_OPERATION_NON_TRANSPOSE) ? a_d->cols : a_d->rows;
  const int64_t expected_y = (opA == CUSPARSE_OPERATION_NON_TRANSPOSE) ? a_d->rows : a_d->cols;
  if (x_d->size != expected_x || y_d->size != expected_y) {
    return CUSPARSE_STATUS_INVALID_VALUE;
  }
  if (expected_y == 0) {
    return CUSPARSE_STATUS_SUCCESS;
  }

  const std::size_t elem_size = (a_d->value_type == CUDA_R_32F) ? sizeof(float) : sizeof(__half);
  auto& exec = LibraryExecutor::Instance();

  ResolvedBuffer buf_row;
  ResolvedBuffer buf_col;
  ResolvedBuffer buf_val;
  ResolvedBuffer buf_x;
  ResolvedBuffer buf_y;

  if (!exec.ResolveDeviceBuffer(a_d->row_offsets,
                                static_cast<std::size_t>(a_d->rows + 1) * sizeof(std::int32_t),
                                &buf_row) ||
      !exec.ResolveDeviceBuffer(x_d->values, static_cast<std::size_t>(x_d->size) * elem_size,
                                &buf_x) ||
      !exec.ResolveDeviceBuffer(y_d->values, static_cast<std::size_t>(y_d->size) * elem_size,
                                &buf_y)) {
    return CUSPARSE_STATUS_INVALID_VALUE;
  }
  if (a_d->nnz > 0) {
    if (!exec.ResolveDeviceBuffer(a_d->col_indices,
                                  static_cast<std::size_t>(a_d->nnz) * sizeof(std::int32_t),
                                  &buf_col) ||
        !exec.ResolveDeviceBuffer(a_d->values, static_cast<std::size_t>(a_d->nnz) * elem_size,
                                  &buf_val)) {
      return CUSPARSE_STATUS_INVALID_VALUE;
    }
  } else {
    buf_col = buf_row;
    buf_val = buf_y;
  }

  const auto u_rows = static_cast<std::uint32_t>(a_d->rows);
  const auto u_cols = static_cast<std::uint32_t>(a_d->cols);
  const int idx_base = (a_d->idx_base == CUSPARSE_INDEX_BASE_ONE) ? 1 : 0;
  const float alpha_val = *static_cast<const float*>(alpha);
  const float beta_val = *static_cast<const float*>(beta);
  const std::uint32_t u_op = (opA == CUSPARSE_OPERATION_NON_TRANSPOSE) ? 0u : 1u;

  cudaStream_t stream = nullptr;
  {
    std::lock_guard<std::mutex> lock(ctx->mutex);
    stream = ctx->stream;
  }

  // Fast Path: Coalesced warp-per-row CSR SpMV for non-transpose when average nnz/row >= 8
  if (u_op == 0u && a_d->rows > 0 && (a_d->nnz / a_d->rows) >= 8) {
    const std::string_view warp_k = (a_d->value_type == CUDA_R_32F)
                                        ? "muda_cusparse_spmv_csr_warp_f32"
                                        : "muda_cusparse_spmv_csr_warp_f16";
    id<MTLComputePipelineState> warp_pipe = GetCusparsePipeline(warp_k);
    if (warp_pipe != nil) {
      const std::array<ShaderArg, 9> warp_args = {
          ShaderArg::Buffer(buf_row), ShaderArg::Buffer(buf_col),  ShaderArg::Buffer(buf_val),
          ShaderArg::Buffer(buf_x),   ShaderArg::Buffer(buf_y),    ShaderArg::Value(u_rows),
          ShaderArg::Value(idx_base), ShaderArg::Value(alpha_val), ShaderArg::Value(beta_val),
      };
      const std::size_t gx = (static_cast<std::size_t>(u_rows) + 7u) / 8u;
      return (exec.DispatchThreadgroups(warp_pipe, MTLSizeMake(gx, 1, 1), MTLSizeMake(256, 1, 1),
                                        warp_args, 0, stream) == cudaSuccess)
                 ? CUSPARSE_STATUS_SUCCESS
                 : CUSPARSE_STATUS_EXECUTION_FAILED;
    }
  }

  const std::string_view k_name =
      (a_d->value_type == CUDA_R_32F) ? "muda_cusparse_spmv_csr_f32" : "muda_cusparse_spmv_csr_f16";
  id<MTLComputePipelineState> pipe = GetCusparsePipeline(k_name);
  if (pipe == nil) {
    return CUSPARSE_STATUS_INTERNAL_ERROR;
  }

  const std::array<ShaderArg, 11> args = {
      ShaderArg::Buffer(buf_row), ShaderArg::Buffer(buf_col), ShaderArg::Buffer(buf_val),
      ShaderArg::Buffer(buf_x),   ShaderArg::Buffer(buf_y),   ShaderArg::Value(u_rows),
      ShaderArg::Value(u_cols),   ShaderArg::Value(idx_base), ShaderArg::Value(alpha_val),
      ShaderArg::Value(beta_val), ShaderArg::Value(u_op),
  };

  const auto out_threads = static_cast<std::size_t>(expected_y);
  return (exec.DispatchThreads1D(pipe, out_threads, 256, args, 0, stream) == cudaSuccess)
             ? CUSPARSE_STATUS_SUCCESS
             : CUSPARSE_STATUS_EXECUTION_FAILED;
}

cusparseStatus_t cusparseSpMM_bufferSize(cusparseHandle_t handle, cusparseOperation_t opA,
                                         cusparseOperation_t opB, const void* alpha,
                                         cusparseSpMatDescr_t matA, cusparseDnMatDescr_t matB,
                                         const void* beta, cusparseDnMatDescr_t matC,
                                         cudaDataType /*computeType*/, cusparseSpMMAlg_t /*alg*/,
                                         size_t* bufferSize) {
  if (bufferSize == nullptr || alpha == nullptr || beta == nullptr) {
    return CUSPARSE_STATUS_INVALID_VALUE;
  }
  auto ctx = GetCusparseHandleTable().Lookup(handle);
  if (ctx == nullptr) {
    return CUSPARSE_STATUS_NOT_INITIALIZED;
  }
  auto a_d = GetSpMatDescTable().Lookup(matA);
  auto b_d = GetDnMatDescTable().Lookup(matB);
  auto c_d = GetDnMatDescTable().Lookup(matC);
  if (a_d == nullptr || b_d == nullptr || c_d == nullptr) {
    return CUSPARSE_STATUS_INVALID_VALUE;
  }
  const int64_t a_eff_r = (opA == CUSPARSE_OPERATION_NON_TRANSPOSE) ? a_d->rows : a_d->cols;
  const int64_t a_eff_c = (opA == CUSPARSE_OPERATION_NON_TRANSPOSE) ? a_d->cols : a_d->rows;
  const int64_t b_eff_r = (opB == CUSPARSE_OPERATION_NON_TRANSPOSE) ? b_d->rows : b_d->cols;
  const int64_t b_eff_c = (opB == CUSPARSE_OPERATION_NON_TRANSPOSE) ? b_d->cols : b_d->rows;
  if (a_eff_c != b_eff_r || c_d->rows != a_eff_r || c_d->cols != b_eff_c) {
    return CUSPARSE_STATUS_INVALID_VALUE;
  }
  *bufferSize = 0;
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSpMM(cusparseHandle_t handle, cusparseOperation_t opA,
                              cusparseOperation_t opB, const void* alpha, cusparseSpMatDescr_t matA,
                              cusparseDnMatDescr_t matB, const void* beta,
                              cusparseDnMatDescr_t matC, cudaDataType /*computeType*/,
                              cusparseSpMMAlg_t /*alg*/, void* /*externalBuffer*/) {
  if (alpha == nullptr || beta == nullptr) {
    return CUSPARSE_STATUS_INVALID_VALUE;
  }
  auto ctx = GetCusparseHandleTable().Lookup(handle);
  if (ctx == nullptr) {
    return CUSPARSE_STATUS_NOT_INITIALIZED;
  }
  auto a_d = GetSpMatDescTable().Lookup(matA);
  auto b_d = GetDnMatDescTable().Lookup(matB);
  auto c_d = GetDnMatDescTable().Lookup(matC);
  if (a_d == nullptr || b_d == nullptr || c_d == nullptr) {
    return CUSPARSE_STATUS_INVALID_VALUE;
  }
  if (a_d->value_type != CUDA_R_32F || b_d->value_type != CUDA_R_32F ||
      c_d->value_type != CUDA_R_32F) {
    return CUSPARSE_STATUS_NOT_SUPPORTED;
  }

  const int64_t a_eff_r = (opA == CUSPARSE_OPERATION_NON_TRANSPOSE) ? a_d->rows : a_d->cols;
  const int64_t a_eff_c = (opA == CUSPARSE_OPERATION_NON_TRANSPOSE) ? a_d->cols : a_d->rows;
  const int64_t b_eff_r = (opB == CUSPARSE_OPERATION_NON_TRANSPOSE) ? b_d->rows : b_d->cols;
  const int64_t b_eff_c = (opB == CUSPARSE_OPERATION_NON_TRANSPOSE) ? b_d->cols : b_d->rows;
  if (a_eff_c != b_eff_r || c_d->rows != a_eff_r || c_d->cols != b_eff_c) {
    return CUSPARSE_STATUS_INVALID_VALUE;
  }
  if (a_eff_r == 0 || b_eff_c == 0) {
    return CUSPARSE_STATUS_SUCCESS;
  }

  auto& exec = LibraryExecutor::Instance();
  ResolvedBuffer buf_row;
  ResolvedBuffer buf_col;
  ResolvedBuffer buf_val;
  ResolvedBuffer buf_b;
  ResolvedBuffer buf_c;

  if (!exec.ResolveDeviceBuffer(a_d->row_offsets,
                                static_cast<std::size_t>(a_d->rows + 1) * sizeof(std::int32_t),
                                &buf_row) ||
      !exec.ResolveDeviceBuffer(b_d->values, DenseMatSpanElements(*b_d) * sizeof(float), &buf_b) ||
      !exec.ResolveDeviceBuffer(c_d->values, DenseMatSpanElements(*c_d) * sizeof(float), &buf_c)) {
    return CUSPARSE_STATUS_INVALID_VALUE;
  }
  if (a_d->nnz > 0) {
    if (!exec.ResolveDeviceBuffer(a_d->col_indices,
                                  static_cast<std::size_t>(a_d->nnz) * sizeof(std::int32_t),
                                  &buf_col) ||
        !exec.ResolveDeviceBuffer(a_d->values, static_cast<std::size_t>(a_d->nnz) * sizeof(float),
                                  &buf_val)) {
      return CUSPARSE_STATUS_INVALID_VALUE;
    }
  } else {
    buf_col = buf_row;
    buf_val = buf_c;
  }

  id<MTLComputePipelineState> pipe = GetCusparsePipeline("muda_cusparse_spmm_csr_f32");
  if (pipe == nil) {
    return CUSPARSE_STATUS_INTERNAL_ERROR;
  }

  const auto u_arows = static_cast<std::uint32_t>(a_d->rows);
  const auto u_acols = static_cast<std::uint32_t>(a_d->cols);
  const auto u_ncols = static_cast<std::uint32_t>(b_eff_c);
  const auto u_ldb = static_cast<std::uint32_t>(b_d->ld);
  const auto u_ldc = static_cast<std::uint32_t>(c_d->ld);
  const auto u_ord_b = static_cast<std::uint32_t>(b_d->order);
  const auto u_ord_c = static_cast<std::uint32_t>(c_d->order);
  const std::uint32_t u_opa = (opA == CUSPARSE_OPERATION_NON_TRANSPOSE) ? 0u : 1u;
  const std::uint32_t u_opb = (opB == CUSPARSE_OPERATION_NON_TRANSPOSE) ? 0u : 1u;
  const int idx_base = (a_d->idx_base == CUSPARSE_INDEX_BASE_ONE) ? 1 : 0;
  const float alpha_val = *static_cast<const float*>(alpha);
  const float beta_val = *static_cast<const float*>(beta);

  const std::array<ShaderArg, 17> args = {
      ShaderArg::Buffer(buf_row),  ShaderArg::Buffer(buf_col), ShaderArg::Buffer(buf_val),
      ShaderArg::Buffer(buf_b),    ShaderArg::Buffer(buf_c),   ShaderArg::Value(u_arows),
      ShaderArg::Value(u_acols),   ShaderArg::Value(u_ncols),  ShaderArg::Value(u_ldb),
      ShaderArg::Value(u_ldc),     ShaderArg::Value(u_ord_b),  ShaderArg::Value(u_ord_c),
      ShaderArg::Value(u_opa),     ShaderArg::Value(u_opb),    ShaderArg::Value(idx_base),
      ShaderArg::Value(alpha_val), ShaderArg::Value(beta_val),
  };

  cudaStream_t stream = nullptr;
  {
    std::lock_guard<std::mutex> lock(ctx->mutex);
    stream = ctx->stream;
  }

  // Fast Path: Row-major non-transposed CSR SpMM using 128-bit float4 vectorization
  if (u_opa == 0u && u_opb == 0u && b_d->order == CUSPARSE_ORDER_ROW &&
      c_d->order == CUSPARSE_ORDER_ROW && (u_ncols & 3u) == 0u && (u_ldb & 3u) == 0u &&
      (u_ldc & 3u) == 0u && (buf_b.offset & 15u) == 0u && (buf_c.offset & 15u) == 0u) {
    id<MTLComputePipelineState> vec4_pipe = GetCusparsePipeline("muda_cusparse_spmm_csr_vec4_f32");
    if (vec4_pipe != nil) {
      const std::uint32_t ncols_v4 = u_ncols >> 2u;
      const std::uint32_t ldb_v4 = u_ldb >> 2u;
      const std::uint32_t ldc_v4 = u_ldc >> 2u;
      const std::array<ShaderArg, 12> v4_args = {
          ShaderArg::Buffer(buf_row), ShaderArg::Buffer(buf_col),  ShaderArg::Buffer(buf_val),
          ShaderArg::Buffer(buf_b),   ShaderArg::Buffer(buf_c),    ShaderArg::Value(u_arows),
          ShaderArg::Value(ncols_v4), ShaderArg::Value(ldb_v4),    ShaderArg::Value(ldc_v4),
          ShaderArg::Value(idx_base), ShaderArg::Value(alpha_val), ShaderArg::Value(beta_val),
      };
      const std::size_t bx4 = 32;
      const std::size_t by4 = 8;
      const std::size_t gx4 = (static_cast<std::size_t>(ncols_v4) + bx4 - 1) / bx4;
      const std::size_t gy4 = (static_cast<std::size_t>(u_arows) + by4 - 1) / by4;
      return (exec.DispatchThreadgroups(vec4_pipe, MTLSizeMake(gx4, gy4, 1),
                                        MTLSizeMake(bx4, by4, 1), v4_args, 0,
                                        stream) == cudaSuccess)
                 ? CUSPARSE_STATUS_SUCCESS
                 : CUSPARSE_STATUS_EXECUTION_FAILED;
    }
  }

  const std::size_t bx = 32;
  const std::size_t by = 8;
  const std::size_t gx = (static_cast<std::size_t>(b_eff_c) + bx - 1) / bx;
  const std::size_t gy = (static_cast<std::size_t>(a_eff_r) + by - 1) / by;
  return (exec.DispatchThreadgroups(pipe, MTLSizeMake(gx, gy, 1), MTLSizeMake(bx, by, 1), args, 0,
                                    stream) == cudaSuccess)
             ? CUSPARSE_STATUS_SUCCESS
             : CUSPARSE_STATUS_EXECUTION_FAILED;
}

}  // extern "C"

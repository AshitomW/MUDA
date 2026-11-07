/**
 * @file cublas_level3.mm
 * @brief Implementation of cuBLAS Level-3 GEMM and GEAM operations using LruPlanCache,
 *        MPSMatrixMultiplication (zero-copy col-major <-> row-major mapping), and custom
 *        Apple Silicon simdgroup_matrix Metal shaders.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>

#include "cublas_context.hpp"
#include "muda/libraries/metal_executor.hpp"
#include "muda/libraries/plan_cache.hpp"

namespace muda::libraries::blas {
namespace {

struct GemmPlanKey {
  int m = 0;
  int n = 0;
  int k = 0;
  int batch_count = 1;
  bool trans_left = false;
  bool trans_right = false;
  bool is_fp16 = false;
  std::uint32_t alpha_bits = 0;
  std::uint32_t beta_bits = 0;

  bool operator==(const GemmPlanKey& other) const noexcept {
    return m == other.m && n == other.n && k == other.k && batch_count == other.batch_count &&
           trans_left == other.trans_left && trans_right == other.trans_right &&
           is_fp16 == other.is_fp16 && alpha_bits == other.alpha_bits &&
           beta_bits == other.beta_bits;
  }
};

struct GemmPlanKeyHash {
  std::size_t operator()(const GemmPlanKey& key) const noexcept {
    std::size_t seed = 0;
    HashCombine(seed, key.m);
    HashCombine(seed, key.n);
    HashCombine(seed, key.k);
    HashCombine(seed, key.batch_count);
    HashCombine(seed, key.trans_left);
    HashCombine(seed, key.trans_right);
    HashCombine(seed, key.is_fp16);
    HashCombine(seed, key.alpha_bits);
    HashCombine(seed, key.beta_bits);
    return seed;
  }
};

struct CachedMpsGemmPlan {
  MPSMatrixMultiplication* kernel = nil;
  MPSMatrixDescriptor* desc_left = nil;
  MPSMatrixDescriptor* desc_right = nil;
  MPSMatrixDescriptor* desc_out = nil;
  int lda = 0;
  int ldb = 0;
  int ldc = 0;
  long long int stride_a = 0;
  long long int stride_b = 0;
  long long int stride_c = 0;
};

[[nodiscard]] LruPlanCache<GemmPlanKey, CachedMpsGemmPlan, GemmPlanKeyHash>&
GetMpsGemmPlanCache() noexcept {
  static LruPlanCache<GemmPlanKey, CachedMpsGemmPlan, GemmPlanKeyHash> cache(256);
  return cache;
}

[[nodiscard]] constexpr bool IsValidOperation(cublasOperation_t op) noexcept {
  return op == CUBLAS_OP_N || op == CUBLAS_OP_T || op == CUBLAS_OP_C;
}

[[nodiscard]] constexpr std::size_t MatrixSpanElements(int rows, int cols, int ld) noexcept {
  if (rows <= 0 || cols <= 0) {
    return 0;
  }
  return static_cast<std::size_t>(ld) * static_cast<std::size_t>(cols - 1) +
         static_cast<std::size_t>(rows);
}

[[nodiscard]] cublasStatus_t ExecuteStridedBatchedGemmF32(
    const std::shared_ptr<CublasContext>& ctx, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, float alpha_val, const float* A, int lda, long long int strideA,
    const float* B, int ldb, long long int strideB, float beta_val, float* C, int ldc,
    long long int strideC, int batch_count) noexcept {
  if (m == 0 || n == 0 || batch_count == 0) {
    return CUBLAS_STATUS_SUCCESS;
  }

  const int a_rows = (transa == CUBLAS_OP_N) ? m : k;
  const int a_cols = (transa == CUBLAS_OP_N) ? k : m;
  const int b_rows = (transb == CUBLAS_OP_N) ? k : n;
  const int b_cols = (transb == CUBLAS_OP_N) ? n : k;

  const std::size_t a_mat_elems = MatrixSpanElements(a_rows, a_cols, lda);
  const std::size_t b_mat_elems = MatrixSpanElements(b_rows, b_cols, ldb);
  const std::size_t c_mat_elems = MatrixSpanElements(m, n, ldc);

  auto& exec = LibraryExecutor::Instance();
  ResolvedBuffer buf_a;
  ResolvedBuffer buf_b;
  ResolvedBuffer buf_c;
  if (!exec.ResolveDeviceBuffer(A, a_mat_elems * sizeof(float), &buf_a) ||
      !exec.ResolveDeviceBuffer(B, b_mat_elems * sizeof(float), &buf_b) ||
      !exec.ResolveDeviceBuffer(C, c_mat_elems * sizeof(float), &buf_c)) {
    return CUBLAS_STATUS_INVALID_VALUE;
  }

  cudaStream_t stream = nullptr;
  {
    std::lock_guard<std::mutex> lock(ctx->mutex);
    stream = ctx->stream;
  }

  // Fast Path 1: Large single-batch or strided-batched aligned GEMM via cached
  // MPSMatrixMultiplication
  const bool valid_batch_strides =
      (batch_count == 1) || (strideA > 0 && strideB > 0 && strideC > 0);
  const bool can_use_mps = valid_batch_strides && (m >= 64 && n >= 64 && k >= 64) &&
                           ((buf_a.offset % 4u) == 0u) && ((buf_b.offset % 4u) == 0u) &&
                           ((buf_c.offset % 4u) == 0u);
  if (can_use_mps) {
    const bool trans_left = (transb != CUBLAS_OP_N);
    const bool trans_right = (transa != CUBLAS_OP_N);
    std::uint32_t alpha_bits = 0;
    std::uint32_t beta_bits = 0;
    std::memcpy(&alpha_bits, &alpha_val, sizeof(float));
    std::memcpy(&beta_bits, &beta_val, sizeof(float));

    const GemmPlanKey key{
        .m = m,
        .n = n,
        .k = k,
        .batch_count = batch_count,
        .trans_left = trans_left,
        .trans_right = trans_right,
        .is_fp16 = false,
        .alpha_bits = alpha_bits,
        .beta_bits = beta_bits,
    };

    @autoreleasepool {
      auto& cache = GetMpsGemmPlanCache();
      CachedMpsGemmPlan plan{};
      if (auto cached = cache.Lookup(key);
          cached.has_value() && cached->lda == lda && cached->ldb == ldb && cached->ldc == ldc &&
          cached->stride_a == strideA && cached->stride_b == strideB &&
          cached->stride_c == strideC) {
        plan = *cached;
      } else {
        id<MTLDevice> device = exec.Device();
        plan.kernel =
            [[MPSMatrixMultiplication alloc] initWithDevice:device
                                              transposeLeft:trans_left
                                             transposeRight:trans_right
                                                 resultRows:static_cast<NSUInteger>(n)
                                              resultColumns:static_cast<NSUInteger>(m)
                                            interiorColumns:static_cast<NSUInteger>(k)
                                                      alpha:static_cast<double>(alpha_val)
                                                       beta:static_cast<double>(beta_val)];
        if (plan.kernel != nil) {
          plan.kernel.batchStart = 0;
          plan.kernel.batchSize = static_cast<NSUInteger>(batch_count);
          const NSUInteger left_rows = static_cast<NSUInteger>(trans_left ? k : n);
          const NSUInteger left_cols = static_cast<NSUInteger>(trans_left ? n : k);
          const NSUInteger right_rows = static_cast<NSUInteger>(trans_right ? m : k);
          const NSUInteger right_cols = static_cast<NSUInteger>(trans_right ? k : m);
          const NSUInteger left_row_bytes = static_cast<NSUInteger>(ldb) * sizeof(float);
          const NSUInteger right_row_bytes = static_cast<NSUInteger>(lda) * sizeof(float);
          const NSUInteger out_row_bytes = static_cast<NSUInteger>(ldc) * sizeof(float);

          if (batch_count == 1) {
            plan.desc_left = [MPSMatrixDescriptor matrixDescriptorWithRows:left_rows
                                                                   columns:left_cols
                                                                  rowBytes:left_row_bytes
                                                                  dataType:MPSDataTypeFloat32];
            plan.desc_right = [MPSMatrixDescriptor matrixDescriptorWithRows:right_rows
                                                                    columns:right_cols
                                                                   rowBytes:right_row_bytes
                                                                   dataType:MPSDataTypeFloat32];
            plan.desc_out = [MPSMatrixDescriptor matrixDescriptorWithRows:static_cast<NSUInteger>(n)
                                                                  columns:static_cast<NSUInteger>(m)
                                                                 rowBytes:out_row_bytes
                                                                 dataType:MPSDataTypeFloat32];
          } else {
            plan.desc_left = [MPSMatrixDescriptor
                matrixDescriptorWithRows:left_rows
                                 columns:left_cols
                                matrices:static_cast<NSUInteger>(batch_count)
                                rowBytes:left_row_bytes
                             matrixBytes:static_cast<NSUInteger>(strideB) * sizeof(float)
                                dataType:MPSDataTypeFloat32];
            plan.desc_right = [MPSMatrixDescriptor
                matrixDescriptorWithRows:right_rows
                                 columns:right_cols
                                matrices:static_cast<NSUInteger>(batch_count)
                                rowBytes:right_row_bytes
                             matrixBytes:static_cast<NSUInteger>(strideA) * sizeof(float)
                                dataType:MPSDataTypeFloat32];
            plan.desc_out = [MPSMatrixDescriptor
                matrixDescriptorWithRows:static_cast<NSUInteger>(n)
                                 columns:static_cast<NSUInteger>(m)
                                matrices:static_cast<NSUInteger>(batch_count)
                                rowBytes:out_row_bytes
                             matrixBytes:static_cast<NSUInteger>(strideC) * sizeof(float)
                                dataType:MPSDataTypeFloat32];
          }
          plan.lda = lda;
          plan.ldb = ldb;
          plan.ldc = ldc;
          plan.stride_a = strideA;
          plan.stride_b = strideB;
          plan.stride_c = strideC;
          cache.Insert(key, plan);
        }
      }

      if (plan.kernel != nil && plan.desc_left != nil && plan.desc_right != nil &&
          plan.desc_out != nil) {
        MPSMatrix* mat_left = [[MPSMatrix alloc] initWithBuffer:buf_b.buffer
                                                         offset:buf_b.offset
                                                     descriptor:plan.desc_left];
        MPSMatrix* mat_right = [[MPSMatrix alloc] initWithBuffer:buf_a.buffer
                                                          offset:buf_a.offset
                                                      descriptor:plan.desc_right];
        MPSMatrix* mat_out = [[MPSMatrix alloc] initWithBuffer:buf_c.buffer
                                                        offset:buf_c.offset
                                                    descriptor:plan.desc_out];

        const cudaError_t enc_err =
            exec.EncodeCommandBuffer(stream, ^(id<MTLCommandBuffer> cmd_buf) {
              [plan.kernel encodeToCommandBuffer:cmd_buf
                                      leftMatrix:mat_left
                                     rightMatrix:mat_right
                                    resultMatrix:mat_out];
              return true;
            });
        if (enc_err == cudaSuccess) {
          return CUBLAS_STATUS_SUCCESS;
        }
      }
    }
  }

  const auto um = static_cast<std::uint32_t>(m);
  const auto un = static_cast<std::uint32_t>(n);
  const auto uk = static_cast<std::uint32_t>(k);
  const auto ulda = static_cast<std::uint32_t>(lda);
  const auto uldb = static_cast<std::uint32_t>(ldb);
  const auto uldc = static_cast<std::uint32_t>(ldc);
  const std::uint32_t utransa = (transa == CUBLAS_OP_N) ? 0u : 1u;
  const std::uint32_t utransb = (transb == CUBLAS_OP_N) ? 0u : 1u;

  // Fast Path 2: Single-batch Apple Silicon simdgroup_matrix 8x8 tiled microkernel
  if (batch_count == 1 && k > 0) {
    id<MTLComputePipelineState> sg_pipe = GetCublasPipeline("muda_cublas_sgemm_simdgroup");
    if (sg_pipe != nil) {
      const std::array<ShaderArg, 13> sg_args = {
          ShaderArg::Buffer(buf_a),    ShaderArg::Buffer(buf_b),   ShaderArg::Buffer(buf_c),
          ShaderArg::Value(um),        ShaderArg::Value(un),       ShaderArg::Value(uk),
          ShaderArg::Value(ulda),      ShaderArg::Value(uldb),     ShaderArg::Value(uldc),
          ShaderArg::Value(alpha_val), ShaderArg::Value(beta_val), ShaderArg::Value(utransa),
          ShaderArg::Value(utransb),
      };
      const std::size_t grid_x = (static_cast<std::size_t>(m) + 31u) / 32u;
      const std::size_t grid_y = (static_cast<std::size_t>(n) + 31u) / 32u;
      if (exec.DispatchThreadgroups(sg_pipe, MTLSizeMake(grid_x, grid_y, 1), MTLSizeMake(128, 1, 1),
                                    sg_args, 0, stream) == cudaSuccess) {
        return CUBLAS_STATUS_SUCCESS;
      }
    }
  }

  const auto s_a = static_cast<std::int64_t>(strideA);
  const auto s_b = static_cast<std::int64_t>(strideB);
  const auto s_c = static_cast<std::int64_t>(strideC);

  const std::array<ShaderArg, 16> args = {
      ShaderArg::Buffer(buf_a),    ShaderArg::Buffer(buf_b),   ShaderArg::Buffer(buf_c),
      ShaderArg::Value(um),        ShaderArg::Value(un),       ShaderArg::Value(uk),
      ShaderArg::Value(ulda),      ShaderArg::Value(uldb),     ShaderArg::Value(uldc),
      ShaderArg::Value(s_a),       ShaderArg::Value(s_b),      ShaderArg::Value(s_c),
      ShaderArg::Value(alpha_val), ShaderArg::Value(beta_val), ShaderArg::Value(utransa),
      ShaderArg::Value(utransb),
  };

  if (k > 0) {
    id<MTLComputePipelineState> sg_batched_pipe =
        GetCublasPipeline("muda_cublas_sgemm_strided_batched_simdgroup");
    if (sg_batched_pipe != nil) {
      const std::size_t grid_x = (static_cast<std::size_t>(m) + 31u) / 32u;
      const std::size_t grid_y = (static_cast<std::size_t>(n) + 31u) / 32u;
      const auto grid_z = static_cast<std::size_t>(batch_count);
      if (exec.DispatchThreadgroups(sg_batched_pipe, MTLSizeMake(grid_x, grid_y, grid_z),
                                    MTLSizeMake(128, 1, 1), args, 0, stream) == cudaSuccess) {
        return CUBLAS_STATUS_SUCCESS;
      }
    }
  }

  // General strided-batched MSL kernel fallback
  id<MTLComputePipelineState> pipe = GetCublasPipeline("muda_cublas_sgemm_strided_batched");
  if (pipe == nil) {
    return CUBLAS_STATUS_INTERNAL_ERROR;
  }

  const std::size_t bx = 16;
  const std::size_t by = 16;
  const std::size_t gx = (static_cast<std::size_t>(m) + bx - 1) / bx;
  const std::size_t gy = (static_cast<std::size_t>(n) + by - 1) / by;
  const auto gz = static_cast<std::size_t>(batch_count);

  return (exec.DispatchThreadgroups(pipe, MTLSizeMake(gx, gy, gz), MTLSizeMake(bx, by, 1), args, 0,
                                    stream) == cudaSuccess)
             ? CUBLAS_STATUS_SUCCESS
             : CUBLAS_STATUS_EXECUTION_FAILED;
}

[[nodiscard]] cublasStatus_t ExecuteStridedBatchedGemmF16(
    const std::shared_ptr<CublasContext>& ctx, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, float alpha_val, const void* A, int lda, long long int strideA,
    const void* B, int ldb, long long int strideB, float beta_val, void* C, int ldc,
    long long int strideC, int batch_count) noexcept {
  if (m == 0 || n == 0 || batch_count == 0) {
    return CUBLAS_STATUS_SUCCESS;
  }

  const int a_rows = (transa == CUBLAS_OP_N) ? m : k;
  const int a_cols = (transa == CUBLAS_OP_N) ? k : m;
  const int b_rows = (transb == CUBLAS_OP_N) ? k : n;
  const int b_cols = (transb == CUBLAS_OP_N) ? n : k;

  const std::size_t a_mat_elems = MatrixSpanElements(a_rows, a_cols, lda);
  const std::size_t b_mat_elems = MatrixSpanElements(b_rows, b_cols, ldb);
  const std::size_t c_mat_elems = MatrixSpanElements(m, n, ldc);

  auto& exec = LibraryExecutor::Instance();
  ResolvedBuffer buf_a;
  ResolvedBuffer buf_b;
  ResolvedBuffer buf_c;
  if (!exec.ResolveDeviceBuffer(A, a_mat_elems * sizeof(__half), &buf_a) ||
      !exec.ResolveDeviceBuffer(B, b_mat_elems * sizeof(__half), &buf_b) ||
      !exec.ResolveDeviceBuffer(C, c_mat_elems * sizeof(__half), &buf_c)) {
    return CUBLAS_STATUS_INVALID_VALUE;
  }

  cudaStream_t stream = nullptr;
  {
    std::lock_guard<std::mutex> lock(ctx->mutex);
    stream = ctx->stream;
  }

  // Fast Path 1: Cached MPSMatrixMultiplication (MPSDataTypeFloat16) when rowBytes & offsets are
  // 4B-aligned
  const bool valid_batch_strides =
      (batch_count == 1) || (strideA > 0 && strideB > 0 && strideC > 0 &&
                             ((static_cast<std::size_t>(strideA) * sizeof(__half)) % 4u == 0u) &&
                             ((static_cast<std::size_t>(strideB) * sizeof(__half)) % 4u == 0u) &&
                             ((static_cast<std::size_t>(strideC) * sizeof(__half)) % 4u == 0u));
  if (valid_batch_strides && k > 0 && m >= 64 && n >= 64 && k >= 64 &&
      ((static_cast<std::size_t>(lda) * sizeof(__half)) % 4u == 0u) &&
      ((static_cast<std::size_t>(ldb) * sizeof(__half)) % 4u == 0u) &&
      ((static_cast<std::size_t>(ldc) * sizeof(__half)) % 4u == 0u) && (buf_a.offset % 4u == 0u) &&
      (buf_b.offset % 4u == 0u) && (buf_c.offset % 4u == 0u)) {
    const bool trans_left = (transb != CUBLAS_OP_N);
    const bool trans_right = (transa != CUBLAS_OP_N);
    std::uint32_t alpha_bits = 0;
    std::uint32_t beta_bits = 0;
    std::memcpy(&alpha_bits, &alpha_val, sizeof(float));
    std::memcpy(&beta_bits, &beta_val, sizeof(float));

    const GemmPlanKey key{
        .m = m,
        .n = n,
        .k = k,
        .batch_count = batch_count,
        .trans_left = trans_left,
        .trans_right = trans_right,
        .is_fp16 = true,
        .alpha_bits = alpha_bits,
        .beta_bits = beta_bits,
    };

    @autoreleasepool {
      auto& cache = GetMpsGemmPlanCache();
      CachedMpsGemmPlan plan{};
      if (auto cached = cache.Lookup(key);
          cached.has_value() && cached->lda == lda && cached->ldb == ldb && cached->ldc == ldc &&
          cached->stride_a == strideA && cached->stride_b == strideB &&
          cached->stride_c == strideC) {
        plan = *cached;
      } else {
        id<MTLDevice> device = exec.Device();
        plan.kernel =
            [[MPSMatrixMultiplication alloc] initWithDevice:device
                                              transposeLeft:trans_left
                                             transposeRight:trans_right
                                                 resultRows:static_cast<NSUInteger>(n)
                                              resultColumns:static_cast<NSUInteger>(m)
                                            interiorColumns:static_cast<NSUInteger>(k)
                                                      alpha:static_cast<double>(alpha_val)
                                                       beta:static_cast<double>(beta_val)];
        if (plan.kernel != nil) {
          plan.kernel.batchStart = 0;
          plan.kernel.batchSize = static_cast<NSUInteger>(batch_count);
          const NSUInteger left_rows = static_cast<NSUInteger>(trans_left ? k : n);
          const NSUInteger left_cols = static_cast<NSUInteger>(trans_left ? n : k);
          const NSUInteger right_rows = static_cast<NSUInteger>(trans_right ? m : k);
          const NSUInteger right_cols = static_cast<NSUInteger>(trans_right ? k : m);
          const NSUInteger left_row_bytes = static_cast<NSUInteger>(ldb) * sizeof(__half);
          const NSUInteger right_row_bytes = static_cast<NSUInteger>(lda) * sizeof(__half);
          const NSUInteger out_row_bytes = static_cast<NSUInteger>(ldc) * sizeof(__half);

          if (batch_count == 1) {
            plan.desc_left = [MPSMatrixDescriptor matrixDescriptorWithRows:left_rows
                                                                   columns:left_cols
                                                                  rowBytes:left_row_bytes
                                                                  dataType:MPSDataTypeFloat16];
            plan.desc_right = [MPSMatrixDescriptor matrixDescriptorWithRows:right_rows
                                                                    columns:right_cols
                                                                   rowBytes:right_row_bytes
                                                                   dataType:MPSDataTypeFloat16];
            plan.desc_out = [MPSMatrixDescriptor matrixDescriptorWithRows:static_cast<NSUInteger>(n)
                                                                  columns:static_cast<NSUInteger>(m)
                                                                 rowBytes:out_row_bytes
                                                                 dataType:MPSDataTypeFloat16];
          } else {
            plan.desc_left = [MPSMatrixDescriptor
                matrixDescriptorWithRows:left_rows
                                 columns:left_cols
                                matrices:static_cast<NSUInteger>(batch_count)
                                rowBytes:left_row_bytes
                             matrixBytes:static_cast<NSUInteger>(strideB) * sizeof(__half)
                                dataType:MPSDataTypeFloat16];
            plan.desc_right = [MPSMatrixDescriptor
                matrixDescriptorWithRows:right_rows
                                 columns:right_cols
                                matrices:static_cast<NSUInteger>(batch_count)
                                rowBytes:right_row_bytes
                             matrixBytes:static_cast<NSUInteger>(strideA) * sizeof(__half)
                                dataType:MPSDataTypeFloat16];
            plan.desc_out = [MPSMatrixDescriptor
                matrixDescriptorWithRows:static_cast<NSUInteger>(n)
                                 columns:static_cast<NSUInteger>(m)
                                matrices:static_cast<NSUInteger>(batch_count)
                                rowBytes:out_row_bytes
                             matrixBytes:static_cast<NSUInteger>(strideC) * sizeof(__half)
                                dataType:MPSDataTypeFloat16];
          }
          plan.lda = lda;
          plan.ldb = ldb;
          plan.ldc = ldc;
          plan.stride_a = strideA;
          plan.stride_b = strideB;
          plan.stride_c = strideC;
          cache.Insert(key, plan);
        }
      }

      if (plan.kernel != nil && plan.desc_left != nil && plan.desc_right != nil &&
          plan.desc_out != nil) {
        MPSMatrix* mat_left = [[MPSMatrix alloc] initWithBuffer:buf_b.buffer
                                                         offset:buf_b.offset
                                                     descriptor:plan.desc_left];
        MPSMatrix* mat_right = [[MPSMatrix alloc] initWithBuffer:buf_a.buffer
                                                          offset:buf_a.offset
                                                      descriptor:plan.desc_right];
        MPSMatrix* mat_out = [[MPSMatrix alloc] initWithBuffer:buf_c.buffer
                                                        offset:buf_c.offset
                                                    descriptor:plan.desc_out];

        const cudaError_t enc_err =
            exec.EncodeCommandBuffer(stream, ^(id<MTLCommandBuffer> cmd_buf) {
              [plan.kernel encodeToCommandBuffer:cmd_buf
                                      leftMatrix:mat_left
                                     rightMatrix:mat_right
                                    resultMatrix:mat_out];
              return true;
            });
        if (enc_err == cudaSuccess) {
          return CUBLAS_STATUS_SUCCESS;
        }
      }
    }
  }

  const auto um = static_cast<std::uint32_t>(m);
  const auto un = static_cast<std::uint32_t>(n);
  const auto uk = static_cast<std::uint32_t>(k);
  const auto ulda = static_cast<std::uint32_t>(lda);
  const auto uldb = static_cast<std::uint32_t>(ldb);
  const auto uldc = static_cast<std::uint32_t>(ldc);
  const auto s_a = static_cast<std::int64_t>(strideA);
  const auto s_b = static_cast<std::int64_t>(strideB);
  const auto s_c = static_cast<std::int64_t>(strideC);
  const std::uint32_t utransa = (transa == CUBLAS_OP_N) ? 0u : 1u;
  const std::uint32_t utransb = (transb == CUBLAS_OP_N) ? 0u : 1u;

  const std::array<ShaderArg, 16> args = {
      ShaderArg::Buffer(buf_a),    ShaderArg::Buffer(buf_b),   ShaderArg::Buffer(buf_c),
      ShaderArg::Value(um),        ShaderArg::Value(un),       ShaderArg::Value(uk),
      ShaderArg::Value(ulda),      ShaderArg::Value(uldb),     ShaderArg::Value(uldc),
      ShaderArg::Value(s_a),       ShaderArg::Value(s_b),      ShaderArg::Value(s_c),
      ShaderArg::Value(alpha_val), ShaderArg::Value(beta_val), ShaderArg::Value(utransa),
      ShaderArg::Value(utransb),
  };

  if (k > 0) {
    id<MTLComputePipelineState> sg_pipe =
        GetCublasPipeline("muda_cublas_hgemm_strided_batched_simdgroup");
    if (sg_pipe != nil) {
      const std::size_t grid_x = (static_cast<std::size_t>(m) + 31u) / 32u;
      const std::size_t grid_y = (static_cast<std::size_t>(n) + 31u) / 32u;
      const auto grid_z = static_cast<std::size_t>(batch_count);
      if (exec.DispatchThreadgroups(sg_pipe, MTLSizeMake(grid_x, grid_y, grid_z),
                                    MTLSizeMake(128, 1, 1), args, 0, stream) == cudaSuccess) {
        return CUBLAS_STATUS_SUCCESS;
      }
    }
  }

  id<MTLComputePipelineState> pipe = GetCublasPipeline("muda_cublas_hgemm_strided_batched");
  if (pipe == nil) {
    return CUBLAS_STATUS_INTERNAL_ERROR;
  }

  const std::size_t bx = 16;
  const std::size_t by = 16;
  const std::size_t gx = (static_cast<std::size_t>(m) + bx - 1) / bx;
  const std::size_t gy = (static_cast<std::size_t>(n) + by - 1) / by;
  const auto gz = static_cast<std::size_t>(batch_count);

  return (exec.DispatchThreadgroups(pipe, MTLSizeMake(gx, gy, gz), MTLSizeMake(bx, by, 1), args, 0,
                                    stream) == cudaSuccess)
             ? CUBLAS_STATUS_SUCCESS
             : CUBLAS_STATUS_EXECUTION_FAILED;
}

}  // namespace
}  // namespace muda::libraries::blas

using namespace muda::libraries;
using namespace muda::libraries::blas;

extern "C" {

cublasStatus_t cublasSgemm_v2(cublasHandle_t handle, cublasOperation_t transa,
                              cublasOperation_t transb, int m, int n, int k, const float* alpha,
                              const float* A, int lda, const float* B, int ldb, const float* beta,
                              float* C, int ldc) {
  return cublasSgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, 0, B, ldb, 0,
                                   beta, C, ldc, 0, 1);
}

cublasStatus_t cublasHgemm(cublasHandle_t handle, cublasOperation_t transa,
                           cublasOperation_t transb, int m, int n, int k, const __half* alpha,
                           const __half* A, int lda, const __half* B, int ldb, const __half* beta,
                           __half* C, int ldc) {
  auto ctx = LookupContext(handle);
  if (ctx == nullptr) {
    return CUBLAS_STATUS_NOT_INITIALIZED;
  }
  if (!IsValidOperation(transa) || !IsValidOperation(transb) || m < 0 || n < 0 || k < 0 ||
      alpha == nullptr || beta == nullptr) {
    return CUBLAS_STATUS_INVALID_VALUE;
  }
  const int min_lda = (transa == CUBLAS_OP_N) ? (m > 1 ? m : 1) : (k > 1 ? k : 1);
  const int min_ldb = (transb == CUBLAS_OP_N) ? (k > 1 ? k : 1) : (n > 1 ? n : 1);
  const int min_ldc = (m > 1 ? m : 1);
  if (lda < min_lda || ldb < min_ldb || ldc < min_ldc) {
    return CUBLAS_STATUS_INVALID_VALUE;
  }

  float alpha_val = 0.0f;
  float beta_val = 0.0f;
  if (ResolveScalarHalfAsFloat(*ctx, alpha, &alpha_val) != CUBLAS_STATUS_SUCCESS ||
      ResolveScalarHalfAsFloat(*ctx, beta, &beta_val) != CUBLAS_STATUS_SUCCESS) {
    return CUBLAS_STATUS_INVALID_VALUE;
  }

  return ExecuteStridedBatchedGemmF16(ctx, transa, transb, m, n, k, alpha_val, A, lda, 0, B, ldb, 0,
                                      beta_val, C, ldc, 0, 1);
}

cublasStatus_t cublasSgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa,
                                         cublasOperation_t transb, int m, int n, int k,
                                         const float* alpha, const float* A, int lda,
                                         long long int strideA, const float* B, int ldb,
                                         long long int strideB, const float* beta, float* C,
                                         int ldc, long long int strideC, int batchCount) {
  auto ctx = LookupContext(handle);
  if (ctx == nullptr) {
    return CUBLAS_STATUS_NOT_INITIALIZED;
  }
  if (!IsValidOperation(transa) || !IsValidOperation(transb) || m < 0 || n < 0 || k < 0 ||
      batchCount < 0 || alpha == nullptr || beta == nullptr) {
    return CUBLAS_STATUS_INVALID_VALUE;
  }
  const int min_lda = (transa == CUBLAS_OP_N) ? (m > 1 ? m : 1) : (k > 1 ? k : 1);
  const int min_ldb = (transb == CUBLAS_OP_N) ? (k > 1 ? k : 1) : (n > 1 ? n : 1);
  const int min_ldc = (m > 1 ? m : 1);
  if (lda < min_lda || ldb < min_ldb || ldc < min_ldc) {
    return CUBLAS_STATUS_INVALID_VALUE;
  }

  float alpha_val = 0.0f;
  float beta_val = 0.0f;
  if (ResolveScalarFloat(*ctx, alpha, &alpha_val) != CUBLAS_STATUS_SUCCESS ||
      ResolveScalarFloat(*ctx, beta, &beta_val) != CUBLAS_STATUS_SUCCESS) {
    return CUBLAS_STATUS_INVALID_VALUE;
  }

  return ExecuteStridedBatchedGemmF32(ctx, transa, transb, m, n, k, alpha_val, A, lda, strideA, B,
                                      ldb, strideB, beta_val, C, ldc, strideC, batchCount);
}

cublasStatus_t cublasGemmStridedBatchedEx(cublasHandle_t handle, cublasOperation_t transa,
                                          cublasOperation_t transb, int m, int n, int k,
                                          const void* alpha, const void* A, cudaDataType Atype,
                                          int lda, long long int strideA, const void* B,
                                          cudaDataType Btype, int ldb, long long int strideB,
                                          const void* beta, void* C, cudaDataType Ctype, int ldc,
                                          long long int strideC, int batchCount, int computeType,
                                          cublasGemmAlgo_t /*algo*/) {
  auto ctx = LookupContext(handle);
  if (ctx == nullptr) {
    return CUBLAS_STATUS_NOT_INITIALIZED;
  }
  if (!IsValidOperation(transa) || !IsValidOperation(transb) || m < 0 || n < 0 || k < 0 ||
      batchCount < 0 || alpha == nullptr || beta == nullptr) {
    return CUBLAS_STATUS_INVALID_VALUE;
  }
  const int min_lda = (transa == CUBLAS_OP_N) ? (m > 1 ? m : 1) : (k > 1 ? k : 1);
  const int min_ldb = (transb == CUBLAS_OP_N) ? (k > 1 ? k : 1) : (n > 1 ? n : 1);
  const int min_ldc = (m > 1 ? m : 1);
  if (lda < min_lda || ldb < min_ldb || ldc < min_ldc) {
    return CUBLAS_STATUS_INVALID_VALUE;
  }

  if (Atype == CUDA_R_32F && Btype == CUDA_R_32F && Ctype == CUDA_R_32F) {
    float alpha_val = 0.0f;
    float beta_val = 0.0f;
    if (ResolveScalarFloat(*ctx, static_cast<const float*>(alpha), &alpha_val) !=
            CUBLAS_STATUS_SUCCESS ||
        ResolveScalarFloat(*ctx, static_cast<const float*>(beta), &beta_val) !=
            CUBLAS_STATUS_SUCCESS) {
      return CUBLAS_STATUS_INVALID_VALUE;
    }
    return ExecuteStridedBatchedGemmF32(ctx, transa, transb, m, n, k, alpha_val,
                                        static_cast<const float*>(A), lda, strideA,
                                        static_cast<const float*>(B), ldb, strideB, beta_val,
                                        static_cast<float*>(C), ldc, strideC, batchCount);
  }

  if (Atype == CUDA_R_16F && Btype == CUDA_R_16F && Ctype == CUDA_R_16F) {
    float alpha_val = 0.0f;
    float beta_val = 0.0f;
    const bool scalar_is_f16 =
        (computeType == CUBLAS_COMPUTE_16F || computeType == CUBLAS_COMPUTE_16F_PEDANTIC ||
         computeType == static_cast<int>(CUDA_R_16F));
    if (scalar_is_f16) {
      if (ResolveScalarHalfAsFloat(*ctx, static_cast<const __half*>(alpha), &alpha_val) !=
              CUBLAS_STATUS_SUCCESS ||
          ResolveScalarHalfAsFloat(*ctx, static_cast<const __half*>(beta), &beta_val) !=
              CUBLAS_STATUS_SUCCESS) {
        return CUBLAS_STATUS_INVALID_VALUE;
      }
    } else {
      if (ResolveScalarFloat(*ctx, static_cast<const float*>(alpha), &alpha_val) !=
              CUBLAS_STATUS_SUCCESS ||
          ResolveScalarFloat(*ctx, static_cast<const float*>(beta), &beta_val) !=
              CUBLAS_STATUS_SUCCESS) {
        return CUBLAS_STATUS_INVALID_VALUE;
      }
    }
    return ExecuteStridedBatchedGemmF16(ctx, transa, transb, m, n, k, alpha_val, A, lda, strideA, B,
                                        ldb, strideB, beta_val, C, ldc, strideC, batchCount);
  }

  return CUBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasGemmEx(cublasHandle_t handle, cublasOperation_t transa,
                            cublasOperation_t transb, int m, int n, int k, const void* alpha,
                            const void* A, cudaDataType Atype, int lda, const void* B,
                            cudaDataType Btype, int ldb, const void* beta, void* C,
                            cudaDataType Ctype, int ldc, int computeType, cublasGemmAlgo_t algo) {
  return cublasGemmStridedBatchedEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda, 0, B,
                                    Btype, ldb, 0, beta, C, Ctype, ldc, 0, 1, computeType, algo);
}

cublasStatus_t cublasSgeam(cublasHandle_t handle, cublasOperation_t transa,
                           cublasOperation_t transb, int m, int n, const float* alpha,
                           const float* A, int lda, const float* beta, const float* B, int ldb,
                           float* C, int ldc) {
  auto ctx = LookupContext(handle);
  if (ctx == nullptr) {
    return CUBLAS_STATUS_NOT_INITIALIZED;
  }
  if (!IsValidOperation(transa) || !IsValidOperation(transb) || m < 0 || n < 0 ||
      alpha == nullptr || beta == nullptr) {
    return CUBLAS_STATUS_INVALID_VALUE;
  }
  const int min_lda = (transa == CUBLAS_OP_N) ? (m > 1 ? m : 1) : (n > 1 ? n : 1);
  const int min_ldb = (transb == CUBLAS_OP_N) ? (m > 1 ? m : 1) : (n > 1 ? n : 1);
  const int min_ldc = (m > 1 ? m : 1);
  if (lda < min_lda || ldb < min_ldb || ldc < min_ldc) {
    return CUBLAS_STATUS_INVALID_VALUE;
  }
  if (m == 0 || n == 0) {
    return CUBLAS_STATUS_SUCCESS;
  }

  float alpha_val = 0.0f;
  float beta_val = 0.0f;
  if (ResolveScalarFloat(*ctx, alpha, &alpha_val) != CUBLAS_STATUS_SUCCESS ||
      ResolveScalarFloat(*ctx, beta, &beta_val) != CUBLAS_STATUS_SUCCESS) {
    return CUBLAS_STATUS_INVALID_VALUE;
  }

  auto& exec = LibraryExecutor::Instance();
  ResolvedBuffer buf_a;
  ResolvedBuffer buf_b;
  ResolvedBuffer buf_c;

  const int a_rows = (transa == CUBLAS_OP_N) ? m : n;
  const int a_cols = (transa == CUBLAS_OP_N) ? n : m;
  const int b_rows = (transb == CUBLAS_OP_N) ? m : n;
  const int b_cols = (transb == CUBLAS_OP_N) ? n : m;

  if (!exec.ResolveDeviceBuffer(C, MatrixSpanElements(m, n, ldc) * sizeof(float), &buf_c)) {
    return CUBLAS_STATUS_INVALID_VALUE;
  }
  if (alpha_val != 0.0f) {
    if (!exec.ResolveDeviceBuffer(A, MatrixSpanElements(a_rows, a_cols, lda) * sizeof(float),
                                  &buf_a)) {
      return CUBLAS_STATUS_INVALID_VALUE;
    }
  } else {
    buf_a = buf_c;
  }
  if (beta_val != 0.0f) {
    if (!exec.ResolveDeviceBuffer(B, MatrixSpanElements(b_rows, b_cols, ldb) * sizeof(float),
                                  &buf_b)) {
      return CUBLAS_STATUS_INVALID_VALUE;
    }
  } else {
    buf_b = buf_c;
  }

  id<MTLComputePipelineState> pipe = GetCublasPipeline("muda_cublas_sgeam");
  if (pipe == nil) {
    return CUBLAS_STATUS_INTERNAL_ERROR;
  }

  const auto um = static_cast<std::uint32_t>(m);
  const auto un = static_cast<std::uint32_t>(n);
  const auto ulda = static_cast<std::uint32_t>(lda);
  const auto uldb = static_cast<std::uint32_t>(ldb);
  const auto uldc = static_cast<std::uint32_t>(ldc);
  const std::uint32_t utransa = (transa == CUBLAS_OP_N) ? 0u : 1u;
  const std::uint32_t utransb = (transb == CUBLAS_OP_N) ? 0u : 1u;

  const std::array<ShaderArg, 12> args = {
      ShaderArg::Buffer(buf_a),   ShaderArg::Buffer(buf_b),  ShaderArg::Buffer(buf_c),
      ShaderArg::Value(um),       ShaderArg::Value(un),      ShaderArg::Value(ulda),
      ShaderArg::Value(uldb),     ShaderArg::Value(uldc),    ShaderArg::Value(alpha_val),
      ShaderArg::Value(beta_val), ShaderArg::Value(utransa), ShaderArg::Value(utransb),
  };

  cudaStream_t stream = nullptr;
  {
    std::lock_guard<std::mutex> lock(ctx->mutex);
    stream = ctx->stream;
  }

  const std::size_t bx = 16;
  const std::size_t by = 16;
  const std::size_t gx = (static_cast<std::size_t>(m) + bx - 1) / bx;
  const std::size_t gy = (static_cast<std::size_t>(n) + by - 1) / by;
  return (exec.DispatchThreadgroups(pipe, MTLSizeMake(gx, gy, 1), MTLSizeMake(bx, by, 1), args, 0,
                                    stream) == cudaSuccess)
             ? CUBLAS_STATUS_SUCCESS
             : CUBLAS_STATUS_EXECUTION_FAILED;
}

}  // extern "C"

/**
 * @file cublas_level2.mm
 * @brief Implementation of cuBLAS Level-2 matrix-vector operations on Apple Silicon GPUs.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <array>

#include "cublas_context.hpp"
#include "muda/libraries/metal_executor.hpp"

namespace muda::libraries::blas {
namespace {

[[nodiscard]] constexpr bool IsValidOperation(cublasOperation_t op) noexcept {
  return op == CUBLAS_OP_N || op == CUBLAS_OP_T || op == CUBLAS_OP_C;
}

[[nodiscard]] constexpr std::size_t SpanElementCount(int n, int inc) noexcept {
  if (n <= 0) {
    return 0;
  }
  const std::size_t abs_inc = (inc == 0) ? 1u : static_cast<std::size_t>(inc > 0 ? inc : -inc);
  return 1u + static_cast<std::size_t>(n - 1) * abs_inc;
}

}  // namespace
}  // namespace muda::libraries::blas

using namespace muda::libraries;
using namespace muda::libraries::blas;

extern "C" {

cublasStatus_t cublasSgemv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n,
                              const float* alpha, const float* A, int lda, const float* x, int incx,
                              const float* beta, float* y, int incy) {
  auto ctx = LookupContext(handle);
  if (ctx == nullptr) {
    return CUBLAS_STATUS_NOT_INITIALIZED;
  }
  if (!IsValidOperation(trans) || m < 0 || n < 0 || lda < (m > 1 ? m : 1) || incx == 0 ||
      incy == 0 || alpha == nullptr || beta == nullptr) {
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

  const int x_len = (trans == CUBLAS_OP_N) ? n : m;
  const int y_len = (trans == CUBLAS_OP_N) ? m : n;
  const std::size_t a_elems =
      static_cast<std::size_t>(lda) * static_cast<std::size_t>(n - 1) + static_cast<std::size_t>(m);

  auto& exec = LibraryExecutor::Instance();
  ResolvedBuffer buf_a;
  ResolvedBuffer buf_x;
  ResolvedBuffer buf_y;
  if (!exec.ResolveDeviceBuffer(A, a_elems * sizeof(float), &buf_a) ||
      !exec.ResolveDeviceBuffer(x, SpanElementCount(x_len, incx) * sizeof(float), &buf_x) ||
      !exec.ResolveDeviceBuffer(y, SpanElementCount(y_len, incy) * sizeof(float), &buf_y)) {
    return CUBLAS_STATUS_INVALID_VALUE;
  }

  const auto um = static_cast<std::uint32_t>(m);
  const auto un = static_cast<std::uint32_t>(n);
  const auto ulda = static_cast<std::uint32_t>(lda);
  const std::uint32_t utrans = (trans == CUBLAS_OP_N) ? 0u : 1u;
  const auto out_len = static_cast<std::size_t>(y_len);

  cudaStream_t stream = nullptr;
  {
    std::lock_guard<std::mutex> lock(ctx->mutex);
    stream = ctx->stream;
  }

  if (incx == 1 && incy == 1) {
    if (utrans == 0u) {
      id<MTLComputePipelineState> fast_n = GetCublasPipeline("muda_cublas_sgemv_n_fast");
      if (fast_n != nil) {
        const std::array<ShaderArg, 8> fast_args = {
            ShaderArg::Buffer(buf_a),    ShaderArg::Buffer(buf_x),   ShaderArg::Buffer(buf_y),
            ShaderArg::Value(um),        ShaderArg::Value(un),       ShaderArg::Value(ulda),
            ShaderArg::Value(alpha_val), ShaderArg::Value(beta_val),
        };
        const std::size_t gx = (out_len + 255u) / 256u;
        return (exec.DispatchThreadgroups(fast_n, MTLSizeMake(gx, 1, 1), MTLSizeMake(256, 1, 1),
                                          fast_args, 0, stream) == cudaSuccess)
                   ? CUBLAS_STATUS_SUCCESS
                   : CUBLAS_STATUS_EXECUTION_FAILED;
      }
    } else {
      id<MTLComputePipelineState> fast_t = GetCublasPipeline("muda_cublas_sgemv_t_fast");
      if (fast_t != nil) {
        const std::array<ShaderArg, 8> fast_args = {
            ShaderArg::Buffer(buf_a),    ShaderArg::Buffer(buf_x),   ShaderArg::Buffer(buf_y),
            ShaderArg::Value(um),        ShaderArg::Value(un),       ShaderArg::Value(ulda),
            ShaderArg::Value(alpha_val), ShaderArg::Value(beta_val),
        };
        const std::size_t gx = (out_len + 7u) / 8u;
        return (exec.DispatchThreadgroups(fast_t, MTLSizeMake(gx, 1, 1), MTLSizeMake(256, 1, 1),
                                          fast_args, 0, stream) == cudaSuccess)
                   ? CUBLAS_STATUS_SUCCESS
                   : CUBLAS_STATUS_EXECUTION_FAILED;
      }
    }
  }

  id<MTLComputePipelineState> pipe = GetCublasPipeline("muda_cublas_sgemv");
  if (pipe == nil) {
    return CUBLAS_STATUS_INTERNAL_ERROR;
  }

  const std::array<ShaderArg, 11> args = {
      ShaderArg::Buffer(buf_a),   ShaderArg::Buffer(buf_x), ShaderArg::Buffer(buf_y),
      ShaderArg::Value(um),       ShaderArg::Value(un),     ShaderArg::Value(ulda),
      ShaderArg::Value(incx),     ShaderArg::Value(incy),   ShaderArg::Value(alpha_val),
      ShaderArg::Value(beta_val), ShaderArg::Value(utrans),
  };
  return (exec.DispatchThreads1D(pipe, out_len, 256, args, 0, stream) == cudaSuccess)
             ? CUBLAS_STATUS_SUCCESS
             : CUBLAS_STATUS_EXECUTION_FAILED;
}

}  // extern "C"

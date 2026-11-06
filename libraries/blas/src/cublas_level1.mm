/**
 * @file cublas_level1.mm
 * @brief Implementation of cuBLAS Level-1 vector operations on Apple Silicon GPUs.
 */

#include "cublas_context.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <array>
#include <cstdlib>
#include <cstring>

#include "muda/libraries/metal_executor.hpp"

namespace muda::libraries::blas {
namespace {

[[nodiscard]] constexpr std::size_t SpanElementCount(int n, int inc) noexcept {
  if (n <= 0) {
    return 0;
  }
  const std::size_t abs_inc =
      (inc == 0) ? 1u : static_cast<std::size_t>(inc > 0 ? inc : -inc);
  return 1u + static_cast<std::size_t>(n - 1) * abs_inc;
}

template <typename OutT>
[[nodiscard]] cublasStatus_t DispatchReduction1Block(
    const std::shared_ptr<CublasContext>& ctx, std::string_view kernel_name,
    std::span<const ShaderArg> input_args, OutT zero_val, OutT* result) noexcept {
  if (result == nullptr) {
    return CUBLAS_STATUS_INVALID_VALUE;
  }

  cublasPointerMode_t mode = CUBLAS_POINTER_MODE_HOST;
  cudaStream_t stream = nullptr;
  id<MTLBuffer> scratch = nil;
  {
    std::lock_guard<std::mutex> lock(ctx->mutex);
    mode = ctx->pointer_mode;
    stream = ctx->stream;
    scratch = ctx->scratch_buffer;
  }

  auto& exec = LibraryExecutor::Instance();
  ResolvedBuffer dev_out;
  if (mode == CUBLAS_POINTER_MODE_DEVICE) {
    if (!exec.ResolveDeviceBuffer(result, sizeof(OutT), &dev_out)) {
      return CUBLAS_STATUS_INVALID_VALUE;
    }
  } else if (scratch == nil) {
    return CUBLAS_STATUS_INTERNAL_ERROR;
  }

  if (input_args.empty()) {
    if (mode == CUBLAS_POINTER_MODE_HOST) {
      *result = zero_val;
      return CUBLAS_STATUS_SUCCESS;
    }
    if (exec.SynchronizeStream(stream) != cudaSuccess) {
      return CUBLAS_STATUS_EXECUTION_FAILED;
    }
    auto* dst = static_cast<std::uint8_t*>([dev_out.buffer contents]);
    if (dst == nullptr) {
      return CUBLAS_STATUS_MAPPING_ERROR;
    }
    std::memcpy(dst + dev_out.offset, &zero_val, sizeof(OutT));
    return CUBLAS_STATUS_SUCCESS;
  }

  id<MTLComputePipelineState> pipe = GetCublasPipeline(kernel_name);
  if (pipe == nil) {
    return CUBLAS_STATUS_INTERNAL_ERROR;
  }

  std::array<ShaderArg, 8> bound_args{};
  if (input_args.size() > bound_args.size()) {
    return CUBLAS_STATUS_INTERNAL_ERROR;
  }
  for (std::size_t i = 0; i < input_args.size(); ++i) {
    bound_args[i] = input_args[i];
  }

  const std::size_t out_arg_idx = (kernel_name == "muda_cublas_sdot") ? 2u : 1u;
  bound_args[out_arg_idx] = (mode == CUBLAS_POINTER_MODE_DEVICE)
                                ? ShaderArg::Buffer(dev_out)
                                : ShaderArg::RawBuffer(scratch, 0);

  const cudaError_t disp_err = exec.DispatchThreadgroups(
      pipe, MTLSizeMake(1, 1, 1), MTLSizeMake(256, 1, 1),
      std::span<const ShaderArg>(bound_args.data(), input_args.size()), 0, stream);
  if (disp_err != cudaSuccess) {
    return CUBLAS_STATUS_EXECUTION_FAILED;
  }

  if (mode == CUBLAS_POINTER_MODE_HOST) {
    if (exec.SynchronizeStream(stream) != cudaSuccess) {
      return CUBLAS_STATUS_EXECUTION_FAILED;
    }
    const auto* src = static_cast<const std::uint8_t*>([scratch contents]);
    if (src == nullptr) {
      return CUBLAS_STATUS_MAPPING_ERROR;
    }
    std::memcpy(result, src, sizeof(OutT));
  }
  return CUBLAS_STATUS_SUCCESS;
}

}  // namespace
}  // namespace muda::libraries::blas

using namespace muda::libraries;
using namespace muda::libraries::blas;

extern "C" {

cublasStatus_t cublasSaxpy_v2(cublasHandle_t handle, int n, const float* alpha, const float* x,
                              int incx, float* y, int incy) {
  auto ctx = LookupContext(handle);
  if (ctx == nullptr) {
    return CUBLAS_STATUS_NOT_INITIALIZED;
  }
  if (alpha == nullptr) {
    return CUBLAS_STATUS_INVALID_VALUE;
  }
  if (n <= 0) {
    return CUBLAS_STATUS_SUCCESS;
  }

  float alpha_val = 0.0f;
  const cublasStatus_t s_status = ResolveScalarFloat(*ctx, alpha, &alpha_val);
  if (s_status != CUBLAS_STATUS_SUCCESS) {
    return s_status;
  }
  if (alpha_val == 0.0f) {
    return CUBLAS_STATUS_SUCCESS;
  }

  auto& exec = LibraryExecutor::Instance();
  ResolvedBuffer buf_x;
  ResolvedBuffer buf_y;
  if (!exec.ResolveDeviceBuffer(x, SpanElementCount(n, incx) * sizeof(float), &buf_x) ||
      !exec.ResolveDeviceBuffer(y, SpanElementCount(n, incy) * sizeof(float), &buf_y)) {
    return CUBLAS_STATUS_INVALID_VALUE;
  }

  id<MTLComputePipelineState> pipe = GetCublasPipeline("muda_cublas_saxpy");
  if (pipe == nil) {
    return CUBLAS_STATUS_INTERNAL_ERROR;
  }

  const auto un = static_cast<std::uint32_t>(n);
  const std::array<ShaderArg, 6> args = {
      ShaderArg::Buffer(buf_x), ShaderArg::Buffer(buf_y), ShaderArg::Value(un),
      ShaderArg::Value(alpha_val), ShaderArg::Value(incx), ShaderArg::Value(incy),
  };

  cudaStream_t stream = nullptr;
  {
    std::lock_guard<std::mutex> lock(ctx->mutex);
    stream = ctx->stream;
  }
  return (exec.DispatchThreads1D(pipe, un, 256, args, 0, stream) == cudaSuccess)
             ? CUBLAS_STATUS_SUCCESS
             : CUBLAS_STATUS_EXECUTION_FAILED;
}

cublasStatus_t cublasScopy_v2(cublasHandle_t handle, int n, const float* x, int incx, float* y,
                              int incy) {
  auto ctx = LookupContext(handle);
  if (ctx == nullptr) {
    return CUBLAS_STATUS_NOT_INITIALIZED;
  }
  if (n <= 0) {
    return CUBLAS_STATUS_SUCCESS;
  }

  auto& exec = LibraryExecutor::Instance();
  ResolvedBuffer buf_x;
  ResolvedBuffer buf_y;
  if (!exec.ResolveDeviceBuffer(x, SpanElementCount(n, incx) * sizeof(float), &buf_x) ||
      !exec.ResolveDeviceBuffer(y, SpanElementCount(n, incy) * sizeof(float), &buf_y)) {
    return CUBLAS_STATUS_INVALID_VALUE;
  }

  id<MTLComputePipelineState> pipe = GetCublasPipeline("muda_cublas_scopy");
  if (pipe == nil) {
    return CUBLAS_STATUS_INTERNAL_ERROR;
  }

  const auto un = static_cast<std::uint32_t>(n);
  const std::array<ShaderArg, 5> args = {
      ShaderArg::Buffer(buf_x), ShaderArg::Buffer(buf_y), ShaderArg::Value(un),
      ShaderArg::Value(incx),   ShaderArg::Value(incy),
  };

  cudaStream_t stream = nullptr;
  {
    std::lock_guard<std::mutex> lock(ctx->mutex);
    stream = ctx->stream;
  }
  return (exec.DispatchThreads1D(pipe, un, 256, args, 0, stream) == cudaSuccess)
             ? CUBLAS_STATUS_SUCCESS
             : CUBLAS_STATUS_EXECUTION_FAILED;
}

cublasStatus_t cublasSscal_v2(cublasHandle_t handle, int n, const float* alpha, float* x,
                              int incx) {
  auto ctx = LookupContext(handle);
  if (ctx == nullptr) {
    return CUBLAS_STATUS_NOT_INITIALIZED;
  }
  if (alpha == nullptr) {
    return CUBLAS_STATUS_INVALID_VALUE;
  }
  if (n <= 0 || incx <= 0) {
    return CUBLAS_STATUS_SUCCESS;
  }

  float alpha_val = 0.0f;
  const cublasStatus_t s_status = ResolveScalarFloat(*ctx, alpha, &alpha_val);
  if (s_status != CUBLAS_STATUS_SUCCESS) {
    return s_status;
  }

  auto& exec = LibraryExecutor::Instance();
  ResolvedBuffer buf_x;
  if (!exec.ResolveDeviceBuffer(x, SpanElementCount(n, incx) * sizeof(float), &buf_x)) {
    return CUBLAS_STATUS_INVALID_VALUE;
  }

  id<MTLComputePipelineState> pipe = GetCublasPipeline("muda_cublas_sscal");
  if (pipe == nil) {
    return CUBLAS_STATUS_INTERNAL_ERROR;
  }

  const auto un = static_cast<std::uint32_t>(n);
  const std::array<ShaderArg, 4> args = {
      ShaderArg::Buffer(buf_x),
      ShaderArg::Value(un),
      ShaderArg::Value(alpha_val),
      ShaderArg::Value(incx),
  };

  cudaStream_t stream = nullptr;
  {
    std::lock_guard<std::mutex> lock(ctx->mutex);
    stream = ctx->stream;
  }
  return (exec.DispatchThreads1D(pipe, un, 256, args, 0, stream) == cudaSuccess)
             ? CUBLAS_STATUS_SUCCESS
             : CUBLAS_STATUS_EXECUTION_FAILED;
}

cublasStatus_t cublasSswap_v2(cublasHandle_t handle, int n, float* x, int incx, float* y,
                              int incy) {
  auto ctx = LookupContext(handle);
  if (ctx == nullptr) {
    return CUBLAS_STATUS_NOT_INITIALIZED;
  }
  if (n <= 0) {
    return CUBLAS_STATUS_SUCCESS;
  }

  auto& exec = LibraryExecutor::Instance();
  ResolvedBuffer buf_x;
  ResolvedBuffer buf_y;
  if (!exec.ResolveDeviceBuffer(x, SpanElementCount(n, incx) * sizeof(float), &buf_x) ||
      !exec.ResolveDeviceBuffer(y, SpanElementCount(n, incy) * sizeof(float), &buf_y)) {
    return CUBLAS_STATUS_INVALID_VALUE;
  }

  id<MTLComputePipelineState> pipe = GetCublasPipeline("muda_cublas_sswap");
  if (pipe == nil) {
    return CUBLAS_STATUS_INTERNAL_ERROR;
  }

  const auto un = static_cast<std::uint32_t>(n);
  const std::array<ShaderArg, 5> args = {
      ShaderArg::Buffer(buf_x), ShaderArg::Buffer(buf_y), ShaderArg::Value(un),
      ShaderArg::Value(incx),   ShaderArg::Value(incy),
  };

  cudaStream_t stream = nullptr;
  {
    std::lock_guard<std::mutex> lock(ctx->mutex);
    stream = ctx->stream;
  }
  return (exec.DispatchThreads1D(pipe, un, 256, args, 0, stream) == cudaSuccess)
             ? CUBLAS_STATUS_SUCCESS
             : CUBLAS_STATUS_EXECUTION_FAILED;
}

cublasStatus_t cublasSdot_v2(cublasHandle_t handle, int n, const float* x, int incx,
                             const float* y, int incy, float* result) {
  auto ctx = LookupContext(handle);
  if (ctx == nullptr) {
    return CUBLAS_STATUS_NOT_INITIALIZED;
  }
  if (result == nullptr) {
    return CUBLAS_STATUS_INVALID_VALUE;
  }
  if (n <= 0) {
    return DispatchReduction1Block<float>(ctx, "muda_cublas_sdot", {}, 0.0f, result);
  }

  auto& exec = LibraryExecutor::Instance();
  ResolvedBuffer buf_x;
  ResolvedBuffer buf_y;
  if (!exec.ResolveDeviceBuffer(x, SpanElementCount(n, incx) * sizeof(float), &buf_x) ||
      !exec.ResolveDeviceBuffer(y, SpanElementCount(n, incy) * sizeof(float), &buf_y)) {
    return CUBLAS_STATUS_INVALID_VALUE;
  }

  const auto un = static_cast<std::uint32_t>(n);
  const std::array<ShaderArg, 6> args = {
      ShaderArg::Buffer(buf_x), ShaderArg::Buffer(buf_y), ShaderArg{},
      ShaderArg::Value(un),     ShaderArg::Value(incx),   ShaderArg::Value(incy),
  };
  return DispatchReduction1Block<float>(ctx, "muda_cublas_sdot", args, 0.0f, result);
}

cublasStatus_t cublasSnrm2_v2(cublasHandle_t handle, int n, const float* x, int incx,
                              float* result) {
  auto ctx = LookupContext(handle);
  if (ctx == nullptr) {
    return CUBLAS_STATUS_NOT_INITIALIZED;
  }
  if (result == nullptr) {
    return CUBLAS_STATUS_INVALID_VALUE;
  }
  if (n <= 0 || incx <= 0) {
    return DispatchReduction1Block<float>(ctx, "muda_cublas_snrm2", {}, 0.0f, result);
  }

  auto& exec = LibraryExecutor::Instance();
  ResolvedBuffer buf_x;
  if (!exec.ResolveDeviceBuffer(x, SpanElementCount(n, incx) * sizeof(float), &buf_x)) {
    return CUBLAS_STATUS_INVALID_VALUE;
  }

  const auto un = static_cast<std::uint32_t>(n);
  const std::array<ShaderArg, 4> args = {
      ShaderArg::Buffer(buf_x),
      ShaderArg{},
      ShaderArg::Value(un),
      ShaderArg::Value(incx),
  };
  return DispatchReduction1Block<float>(ctx, "muda_cublas_snrm2", args, 0.0f, result);
}

cublasStatus_t cublasSasum_v2(cublasHandle_t handle, int n, const float* x, int incx,
                              float* result) {
  auto ctx = LookupContext(handle);
  if (ctx == nullptr) {
    return CUBLAS_STATUS_NOT_INITIALIZED;
  }
  if (result == nullptr) {
    return CUBLAS_STATUS_INVALID_VALUE;
  }
  if (n <= 0 || incx <= 0) {
    return DispatchReduction1Block<float>(ctx, "muda_cublas_sasum", {}, 0.0f, result);
  }

  auto& exec = LibraryExecutor::Instance();
  ResolvedBuffer buf_x;
  if (!exec.ResolveDeviceBuffer(x, SpanElementCount(n, incx) * sizeof(float), &buf_x)) {
    return CUBLAS_STATUS_INVALID_VALUE;
  }

  const auto un = static_cast<std::uint32_t>(n);
  const std::array<ShaderArg, 4> args = {
      ShaderArg::Buffer(buf_x),
      ShaderArg{},
      ShaderArg::Value(un),
      ShaderArg::Value(incx),
  };
  return DispatchReduction1Block<float>(ctx, "muda_cublas_sasum", args, 0.0f, result);
}

cublasStatus_t cublasIsamax_v2(cublasHandle_t handle, int n, const float* x, int incx,
                               int* result) {
  auto ctx = LookupContext(handle);
  if (ctx == nullptr) {
    return CUBLAS_STATUS_NOT_INITIALIZED;
  }
  if (result == nullptr) {
    return CUBLAS_STATUS_INVALID_VALUE;
  }
  if (n <= 0 || incx <= 0) {
    return DispatchReduction1Block<int>(ctx, "muda_cublas_isamax", {}, 0, result);
  }

  auto& exec = LibraryExecutor::Instance();
  ResolvedBuffer buf_x;
  if (!exec.ResolveDeviceBuffer(x, SpanElementCount(n, incx) * sizeof(float), &buf_x)) {
    return CUBLAS_STATUS_INVALID_VALUE;
  }

  const auto un = static_cast<std::uint32_t>(n);
  const std::array<ShaderArg, 4> args = {
      ShaderArg::Buffer(buf_x),
      ShaderArg{},
      ShaderArg::Value(un),
      ShaderArg::Value(incx),
  };
  return DispatchReduction1Block<int>(ctx, "muda_cublas_isamax", args, 0, result);
}

}  // extern "C"

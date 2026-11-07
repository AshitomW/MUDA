/**
 * @file cudnn_ops.mm
 * @brief Implementation of cuDNN operator execution pipelines (Conv2D, Activation, Pooling,
 *        Softmax, BatchNorm, AddTensor) on Apple Silicon GPUs.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <array>
#include <string_view>

#include "cudnn_descriptors.hpp"
#include "cudnn_kernel_params.h"
#include "muda/libraries/metal_executor.hpp"

namespace muda::libraries::dnn {
namespace {

[[nodiscard]] id<MTLComputePipelineState> GetCudnnPipeline(
    std::string_view function_name) noexcept {
  return LibraryExecutor::Instance().GetOrLoadPipeline(MUDA_CUDNN_METALLIB_PATH,
                                                       MUDA_CUDNN_METAL_SOURCE_PATH, function_name);
}

[[nodiscard]] bool BuildConvParams(const TensorDescriptorState& x_desc,
                                   const FilterDescriptorState& w_desc,
                                   const ConvolutionDescriptorState& c_desc,
                                   const TensorDescriptorState& y_desc, float alpha, float beta,
                                   CudnnConvParams* out) noexcept {
  if (!x_desc.configured || !w_desc.configured || !c_desc.configured || !y_desc.configured) {
    return false;
  }
  if (x_desc.n != y_desc.n || y_desc.c != w_desc.k) {
    return false;
  }
  if (c_desc.group_count <= 0 || (x_desc.c % c_desc.group_count) != 0 ||
      (w_desc.k % c_desc.group_count) != 0) {
    return false;
  }
  const int expected_filter_c = x_desc.c / c_desc.group_count;
  if (w_desc.c != expected_filter_c) {
    return false;
  }

  const int eff_kh = 1 + (w_desc.h - 1) * c_desc.dilation_h;
  const int eff_kw = 1 + (w_desc.w - 1) * c_desc.dilation_w;
  const int expected_oh = 1 + (x_desc.h + 2 * c_desc.pad_h - eff_kh) / c_desc.stride_h;
  const int expected_ow = 1 + (x_desc.w + 2 * c_desc.pad_w - eff_kw) / c_desc.stride_w;
  if (y_desc.h != expected_oh || y_desc.w != expected_ow) {
    return false;
  }

  out->n = x_desc.n;
  out->c_in = x_desc.c;
  out->h_in = x_desc.h;
  out->w_in = x_desc.w;
  out->x_ns = x_desc.n_stride;
  out->x_cs = x_desc.c_stride;
  out->x_hs = x_desc.h_stride;
  out->x_ws = x_desc.w_stride;
  out->c_out = y_desc.c;
  out->h_out = y_desc.h;
  out->w_out = y_desc.w;
  out->groups = c_desc.group_count;
  out->y_ns = y_desc.n_stride;
  out->y_cs = y_desc.c_stride;
  out->y_hs = y_desc.h_stride;
  out->y_ws = y_desc.w_stride;
  out->kh = w_desc.h;
  out->kw = w_desc.w;
  out->pad_h = c_desc.pad_h;
  out->pad_w = c_desc.pad_w;
  out->stride_h = c_desc.stride_h;
  out->stride_w = c_desc.stride_w;
  out->dil_h = c_desc.dilation_h;
  out->dil_w = c_desc.dilation_w;
  out->w_format = (w_desc.format == CUDNN_TENSOR_NCHW) ? 0 : 1;
  out->conv_mode = (c_desc.mode == CUDNN_CONVOLUTION) ? 0 : 1;
  out->alpha = alpha;
  out->beta = beta;
  return true;
}

}  // namespace
}  // namespace muda::libraries::dnn

using namespace muda::libraries;
using namespace muda::libraries::dnn;

extern "C" {

cudnnStatus_t cudnnGetConvolutionForwardWorkspaceSize(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc, const cudnnFilterDescriptor_t wDesc,
    const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t yDesc,
    cudnnConvolutionFwdAlgo_t /*algo*/, size_t* sizeInBytes) {
  if (sizeInBytes == nullptr) {
    return CUDNN_STATUS_BAD_PARAM;
  }
  auto ctx = GetCudnnHandleTable().Lookup(handle);
  auto x_d = GetTensorDescTable().Lookup(xDesc);
  auto w_d = GetFilterDescTable().Lookup(wDesc);
  auto c_d = GetConvDescTable().Lookup(convDesc);
  auto y_d = GetTensorDescTable().Lookup(yDesc);
  if (ctx == nullptr || x_d == nullptr || w_d == nullptr || c_d == nullptr || y_d == nullptr) {
    return CUDNN_STATUS_BAD_PARAM;
  }
  CudnnConvParams params{};
  if (!BuildConvParams(*x_d, *w_d, *c_d, *y_d, 1.0f, 0.0f, &params)) {
    return CUDNN_STATUS_BAD_PARAM;
  }
  *sizeInBytes = 0;
  return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnConvolutionForward(cudnnHandle_t handle, const void* alpha,
                                      const cudnnTensorDescriptor_t xDesc, const void* x,
                                      const cudnnFilterDescriptor_t wDesc, const void* w,
                                      const cudnnConvolutionDescriptor_t convDesc,
                                      cudnnConvolutionFwdAlgo_t /*algo*/, void* /*workSpace*/,
                                      size_t /*workSpaceSizeInBytes*/, const void* beta,
                                      const cudnnTensorDescriptor_t yDesc, void* y) {
  if (alpha == nullptr || beta == nullptr || x == nullptr || w == nullptr || y == nullptr) {
    return CUDNN_STATUS_BAD_PARAM;
  }
  auto ctx = GetCudnnHandleTable().Lookup(handle);
  auto x_d = GetTensorDescTable().Lookup(xDesc);
  auto w_d = GetFilterDescTable().Lookup(wDesc);
  auto c_d = GetConvDescTable().Lookup(convDesc);
  auto y_d = GetTensorDescTable().Lookup(yDesc);
  if (ctx == nullptr || x_d == nullptr || w_d == nullptr || c_d == nullptr || y_d == nullptr) {
    return CUDNN_STATUS_BAD_PARAM;
  }
  if (x_d->data_type != CUDNN_DATA_FLOAT || w_d->data_type != CUDNN_DATA_FLOAT ||
      y_d->data_type != CUDNN_DATA_FLOAT) {
    return CUDNN_STATUS_NOT_SUPPORTED;
  }

  CudnnConvParams params{};
  const float alpha_val = *static_cast<const float*>(alpha);
  const float beta_val = *static_cast<const float*>(beta);
  if (!BuildConvParams(*x_d, *w_d, *c_d, *y_d, alpha_val, beta_val, &params)) {
    return CUDNN_STATUS_BAD_PARAM;
  }

  auto& exec = LibraryExecutor::Instance();
  ResolvedBuffer buf_x;
  ResolvedBuffer buf_w;
  ResolvedBuffer buf_y;
  if (!exec.ResolveDeviceBuffer(x, x_d->SpanElementCount() * sizeof(float), &buf_x) ||
      !exec.ResolveDeviceBuffer(w, w_d->ElementCount() * sizeof(float), &buf_w) ||
      !exec.ResolveDeviceBuffer(y, y_d->SpanElementCount() * sizeof(float), &buf_y)) {
    return CUDNN_STATUS_BAD_PARAM;
  }

  cudaStream_t stream = nullptr;
  {
    std::lock_guard<std::mutex> lock(ctx->mutex);
    stream = ctx->stream;
  }

  const std::array<ShaderArg, 4> args = {
      ShaderArg::Buffer(buf_x),
      ShaderArg::Buffer(buf_w),
      ShaderArg::Buffer(buf_y),
      ShaderArg::Value(params),
  };

  const std::size_t n_spatial = static_cast<std::size_t>(params.n) *
                                static_cast<std::size_t>(params.h_out) *
                                static_cast<std::size_t>(params.w_out);
  if (params.groups == 1 && params.w_format == 0 && (params.c_out >= 16 || n_spatial >= 32)) {
    id<MTLComputePipelineState> gemm_pipe = GetCudnnPipeline("muda_cudnn_conv2d_fwd_implicit_gemm");
    if (gemm_pipe != nil) {
      const std::size_t gx = (n_spatial + 31u) / 32u;
      const std::size_t gy = (static_cast<std::size_t>(params.c_out) + 31u) / 32u;
      if (exec.DispatchThreadgroups(gemm_pipe, MTLSizeMake(gx, gy, 1), MTLSizeMake(128, 1, 1), args,
                                    0, stream) == cudaSuccess) {
        return CUDNN_STATUS_SUCCESS;
      }
    }
  }

  id<MTLComputePipelineState> pipe = GetCudnnPipeline("muda_cudnn_conv2d_fwd");
  if (pipe == nil) {
    return CUDNN_STATUS_INTERNAL_ERROR;
  }

  const std::size_t total_out = n_spatial * static_cast<std::size_t>(params.c_out);
  return (exec.DispatchThreads1D(pipe, total_out, 256, args, 0, stream) == cudaSuccess)
             ? CUDNN_STATUS_SUCCESS
             : CUDNN_STATUS_EXECUTION_FAILED;
}

cudnnStatus_t cudnnConvolutionBackwardData(cudnnHandle_t handle, const void* alpha,
                                           const cudnnFilterDescriptor_t wDesc, const void* w,
                                           const cudnnTensorDescriptor_t dyDesc, const void* dy,
                                           const cudnnConvolutionDescriptor_t convDesc,
                                           cudnnConvolutionBwdDataAlgo_t /*algo*/,
                                           void* /*workSpace*/, size_t /*workSpaceSizeInBytes*/,
                                           const void* beta, const cudnnTensorDescriptor_t dxDesc,
                                           void* dx) {
  if (alpha == nullptr || beta == nullptr || w == nullptr || dy == nullptr || dx == nullptr) {
    return CUDNN_STATUS_BAD_PARAM;
  }
  auto ctx = GetCudnnHandleTable().Lookup(handle);
  auto dx_d = GetTensorDescTable().Lookup(dxDesc);
  auto w_d = GetFilterDescTable().Lookup(wDesc);
  auto c_d = GetConvDescTable().Lookup(convDesc);
  auto dy_d = GetTensorDescTable().Lookup(dyDesc);
  if (ctx == nullptr || dx_d == nullptr || w_d == nullptr || c_d == nullptr || dy_d == nullptr) {
    return CUDNN_STATUS_BAD_PARAM;
  }
  if (dx_d->data_type != CUDNN_DATA_FLOAT || w_d->data_type != CUDNN_DATA_FLOAT ||
      dy_d->data_type != CUDNN_DATA_FLOAT) {
    return CUDNN_STATUS_NOT_SUPPORTED;
  }

  CudnnConvParams params{};
  const float alpha_val = *static_cast<const float*>(alpha);
  const float beta_val = *static_cast<const float*>(beta);
  if (!BuildConvParams(*dx_d, *w_d, *c_d, *dy_d, alpha_val, beta_val, &params)) {
    return CUDNN_STATUS_BAD_PARAM;
  }

  auto& exec = LibraryExecutor::Instance();
  ResolvedBuffer buf_w;
  ResolvedBuffer buf_dy;
  ResolvedBuffer buf_dx;
  if (!exec.ResolveDeviceBuffer(w, w_d->ElementCount() * sizeof(float), &buf_w) ||
      !exec.ResolveDeviceBuffer(dy, dy_d->SpanElementCount() * sizeof(float), &buf_dy) ||
      !exec.ResolveDeviceBuffer(dx, dx_d->SpanElementCount() * sizeof(float), &buf_dx)) {
    return CUDNN_STATUS_BAD_PARAM;
  }

  id<MTLComputePipelineState> pipe = GetCudnnPipeline("muda_cudnn_conv2d_bwd_data");
  if (pipe == nil) {
    return CUDNN_STATUS_INTERNAL_ERROR;
  }

  const std::size_t total_in =
      static_cast<std::size_t>(params.n) * static_cast<std::size_t>(params.c_in) *
      static_cast<std::size_t>(params.h_in) * static_cast<std::size_t>(params.w_in);
  const std::array<ShaderArg, 4> args = {
      ShaderArg::Buffer(buf_w),
      ShaderArg::Buffer(buf_dy),
      ShaderArg::Buffer(buf_dx),
      ShaderArg::Value(params),
  };

  cudaStream_t stream = nullptr;
  {
    std::lock_guard<std::mutex> lock(ctx->mutex);
    stream = ctx->stream;
  }
  return (exec.DispatchThreads1D(pipe, total_in, 256, args, 0, stream) == cudaSuccess)
             ? CUDNN_STATUS_SUCCESS
             : CUDNN_STATUS_EXECUTION_FAILED;
}

cudnnStatus_t cudnnConvolutionBackwardFilter(cudnnHandle_t handle, const void* alpha,
                                             const cudnnTensorDescriptor_t xDesc, const void* x,
                                             const cudnnTensorDescriptor_t dyDesc, const void* dy,
                                             const cudnnConvolutionDescriptor_t convDesc,
                                             cudnnConvolutionBwdFilterAlgo_t /*algo*/,
                                             void* /*workSpace*/, size_t /*workSpaceSizeInBytes*/,
                                             const void* beta, const cudnnFilterDescriptor_t dwDesc,
                                             void* dw) {
  if (alpha == nullptr || beta == nullptr || x == nullptr || dy == nullptr || dw == nullptr) {
    return CUDNN_STATUS_BAD_PARAM;
  }
  auto ctx = GetCudnnHandleTable().Lookup(handle);
  auto x_d = GetTensorDescTable().Lookup(xDesc);
  auto dw_d = GetFilterDescTable().Lookup(dwDesc);
  auto c_d = GetConvDescTable().Lookup(convDesc);
  auto dy_d = GetTensorDescTable().Lookup(dyDesc);
  if (ctx == nullptr || x_d == nullptr || dw_d == nullptr || c_d == nullptr || dy_d == nullptr) {
    return CUDNN_STATUS_BAD_PARAM;
  }
  if (x_d->data_type != CUDNN_DATA_FLOAT || dw_d->data_type != CUDNN_DATA_FLOAT ||
      dy_d->data_type != CUDNN_DATA_FLOAT) {
    return CUDNN_STATUS_NOT_SUPPORTED;
  }

  CudnnConvParams params{};
  const float alpha_val = *static_cast<const float*>(alpha);
  const float beta_val = *static_cast<const float*>(beta);
  if (!BuildConvParams(*x_d, *dw_d, *c_d, *dy_d, alpha_val, beta_val, &params)) {
    return CUDNN_STATUS_BAD_PARAM;
  }

  auto& exec = LibraryExecutor::Instance();
  ResolvedBuffer buf_x;
  ResolvedBuffer buf_dy;
  ResolvedBuffer buf_dw;
  if (!exec.ResolveDeviceBuffer(x, x_d->SpanElementCount() * sizeof(float), &buf_x) ||
      !exec.ResolveDeviceBuffer(dy, dy_d->SpanElementCount() * sizeof(float), &buf_dy) ||
      !exec.ResolveDeviceBuffer(dw, dw_d->ElementCount() * sizeof(float), &buf_dw)) {
    return CUDNN_STATUS_BAD_PARAM;
  }

  id<MTLComputePipelineState> pipe = GetCudnnPipeline("muda_cudnn_conv2d_bwd_filter");
  if (pipe == nil) {
    return CUDNN_STATUS_INTERNAL_ERROR;
  }

  const std::size_t total_w = dw_d->ElementCount();
  const std::array<ShaderArg, 4> args = {
      ShaderArg::Buffer(buf_x),
      ShaderArg::Buffer(buf_dy),
      ShaderArg::Buffer(buf_dw),
      ShaderArg::Value(params),
  };

  cudaStream_t stream = nullptr;
  {
    std::lock_guard<std::mutex> lock(ctx->mutex);
    stream = ctx->stream;
  }
  return (exec.DispatchThreads1D(pipe, total_w, 256, args, 0, stream) == cudaSuccess)
             ? CUDNN_STATUS_SUCCESS
             : CUDNN_STATUS_EXECUTION_FAILED;
}

cudnnStatus_t cudnnConvolutionBackwardBias(cudnnHandle_t handle, const void* alpha,
                                           const cudnnTensorDescriptor_t dyDesc, const void* dy,
                                           const void* beta, const cudnnTensorDescriptor_t dbDesc,
                                           void* db) {
  if (alpha == nullptr || beta == nullptr || dy == nullptr || db == nullptr) {
    return CUDNN_STATUS_BAD_PARAM;
  }
  auto ctx = GetCudnnHandleTable().Lookup(handle);
  auto dy_d = GetTensorDescTable().Lookup(dyDesc);
  auto db_d = GetTensorDescTable().Lookup(dbDesc);
  if (ctx == nullptr || dy_d == nullptr || db_d == nullptr || !dy_d->configured ||
      !db_d->configured) {
    return CUDNN_STATUS_BAD_PARAM;
  }
  if (db_d->n != 1 || db_d->c != dy_d->c || db_d->h != 1 || db_d->w != 1) {
    return CUDNN_STATUS_BAD_PARAM;
  }

  auto& exec = LibraryExecutor::Instance();
  ResolvedBuffer buf_dy;
  ResolvedBuffer buf_db;
  if (!exec.ResolveDeviceBuffer(dy, dy_d->SpanElementCount() * sizeof(float), &buf_dy) ||
      !exec.ResolveDeviceBuffer(db, static_cast<std::size_t>(db_d->c) * sizeof(float), &buf_db)) {
    return CUDNN_STATUS_BAD_PARAM;
  }

  id<MTLComputePipelineState> pipe = GetCudnnPipeline("muda_cudnn_conv2d_bwd_bias");
  if (pipe == nil) {
    return CUDNN_STATUS_INTERNAL_ERROR;
  }

  CudnnConvParams params{};
  params.n = dy_d->n;
  params.c_out = dy_d->c;
  params.h_out = dy_d->h;
  params.w_out = dy_d->w;
  params.y_ns = dy_d->n_stride;
  params.y_cs = dy_d->c_stride;
  params.y_hs = dy_d->h_stride;
  params.y_ws = dy_d->w_stride;
  params.alpha = *static_cast<const float*>(alpha);
  params.beta = *static_cast<const float*>(beta);

  const std::array<ShaderArg, 3> args = {
      ShaderArg::Buffer(buf_dy),
      ShaderArg::Buffer(buf_db),
      ShaderArg::Value(params),
  };

  cudaStream_t stream = nullptr;
  {
    std::lock_guard<std::mutex> lock(ctx->mutex);
    stream = ctx->stream;
  }
  return (exec.DispatchThreads1D(pipe, static_cast<std::size_t>(dy_d->c), 256, args, 0, stream) ==
          cudaSuccess)
             ? CUDNN_STATUS_SUCCESS
             : CUDNN_STATUS_EXECUTION_FAILED;
}

cudnnStatus_t cudnnAddTensor(cudnnHandle_t handle, const void* alpha,
                             const cudnnTensorDescriptor_t aDesc, const void* A, const void* beta,
                             const cudnnTensorDescriptor_t cDesc, void* C) {
  if (alpha == nullptr || beta == nullptr || A == nullptr || C == nullptr) {
    return CUDNN_STATUS_BAD_PARAM;
  }
  auto ctx = GetCudnnHandleTable().Lookup(handle);
  auto a_d = GetTensorDescTable().Lookup(aDesc);
  auto c_d = GetTensorDescTable().Lookup(cDesc);
  if (ctx == nullptr || a_d == nullptr || c_d == nullptr || !a_d->configured || !c_d->configured) {
    return CUDNN_STATUS_BAD_PARAM;
  }
  if ((a_d->n != 1 && a_d->n != c_d->n) || (a_d->c != 1 && a_d->c != c_d->c) ||
      (a_d->h != 1 && a_d->h != c_d->h) || (a_d->w != 1 && a_d->w != c_d->w)) {
    return CUDNN_STATUS_BAD_PARAM;
  }

  auto& exec = LibraryExecutor::Instance();
  ResolvedBuffer buf_a;
  ResolvedBuffer buf_c;
  if (!exec.ResolveDeviceBuffer(A, a_d->SpanElementCount() * sizeof(float), &buf_a) ||
      !exec.ResolveDeviceBuffer(C, c_d->SpanElementCount() * sizeof(float), &buf_c)) {
    return CUDNN_STATUS_BAD_PARAM;
  }

  id<MTLComputePipelineState> pipe = GetCudnnPipeline("muda_cudnn_add_tensor");
  if (pipe == nil) {
    return CUDNN_STATUS_INTERNAL_ERROR;
  }

  CudnnTensorOpParams params{};
  params.n = c_d->n;
  params.c = c_d->c;
  params.h = c_d->h;
  params.w = c_d->w;
  params.x_ns = a_d->n_stride;
  params.x_cs = a_d->c_stride;
  params.x_hs = a_d->h_stride;
  params.x_ws = a_d->w_stride;
  params.y_ns = c_d->n_stride;
  params.y_cs = c_d->c_stride;
  params.y_hs = c_d->h_stride;
  params.y_ws = c_d->w_stride;
  params.a_n = a_d->n;
  params.a_c = a_d->c;
  params.a_h = a_d->h;
  params.a_w = a_d->w;
  params.alpha = *static_cast<const float*>(alpha);
  params.beta = *static_cast<const float*>(beta);

  const std::size_t total = static_cast<std::size_t>(c_d->n) * static_cast<std::size_t>(c_d->c) *
                            static_cast<std::size_t>(c_d->h) * static_cast<std::size_t>(c_d->w);
  const std::array<ShaderArg, 3> args = {
      ShaderArg::Buffer(buf_a),
      ShaderArg::Buffer(buf_c),
      ShaderArg::Value(params),
  };

  cudaStream_t stream = nullptr;
  {
    std::lock_guard<std::mutex> lock(ctx->mutex);
    stream = ctx->stream;
  }
  return (exec.DispatchThreads1D(pipe, total, 256, args, 0, stream) == cudaSuccess)
             ? CUDNN_STATUS_SUCCESS
             : CUDNN_STATUS_EXECUTION_FAILED;
}

cudnnStatus_t cudnnActivationForward(cudnnHandle_t handle,
                                     cudnnActivationDescriptor_t activationDesc, const void* alpha,
                                     const cudnnTensorDescriptor_t xDesc, const void* x,
                                     const void* beta, const cudnnTensorDescriptor_t yDesc,
                                     void* y) {
  if (alpha == nullptr || beta == nullptr || x == nullptr || y == nullptr) {
    return CUDNN_STATUS_BAD_PARAM;
  }
  auto ctx = GetCudnnHandleTable().Lookup(handle);
  auto act_d = GetActivationDescTable().Lookup(activationDesc);
  auto x_d = GetTensorDescTable().Lookup(xDesc);
  auto y_d = GetTensorDescTable().Lookup(yDesc);
  if (ctx == nullptr || act_d == nullptr || x_d == nullptr || y_d == nullptr ||
      !act_d->configured || !x_d->configured || !y_d->configured) {
    return CUDNN_STATUS_BAD_PARAM;
  }
  if (x_d->n != y_d->n || x_d->c != y_d->c || x_d->h != y_d->h || x_d->w != y_d->w) {
    return CUDNN_STATUS_BAD_PARAM;
  }

  auto& exec = LibraryExecutor::Instance();
  ResolvedBuffer buf_x;
  ResolvedBuffer buf_y;
  if (!exec.ResolveDeviceBuffer(x, x_d->SpanElementCount() * sizeof(float), &buf_x) ||
      !exec.ResolveDeviceBuffer(y, y_d->SpanElementCount() * sizeof(float), &buf_y)) {
    return CUDNN_STATUS_BAD_PARAM;
  }

  cudaStream_t stream = nullptr;
  {
    std::lock_guard<std::mutex> lock(ctx->mutex);
    stream = ctx->stream;
  }

  const std::size_t total = static_cast<std::size_t>(x_d->n) * static_cast<std::size_t>(x_d->c) *
                            static_cast<std::size_t>(x_d->h) * static_cast<std::size_t>(x_d->w);
  const int hw = x_d->h * x_d->w;
  const int chw = x_d->c * hw;
  const bool is_contiguous =
      (x_d->w_stride == 1 && x_d->h_stride == x_d->w && x_d->c_stride == hw &&
       x_d->n_stride == chw && y_d->w_stride == 1 && y_d->h_stride == y_d->w &&
       y_d->c_stride == hw && y_d->n_stride == chw);

  if (is_contiguous && (total & 3u) == 0u && (buf_x.offset & 15u) == 0u &&
      (buf_y.offset & 15u) == 0u) {
    id<MTLComputePipelineState> pipe4 = GetCudnnPipeline("muda_cudnn_activation_fwd_vec4");
    if (pipe4 != nil) {
      const auto n_vec4 = static_cast<std::uint32_t>(total >> 2u);
      const int mode_val = static_cast<int>(act_d->mode);
      const float alpha_val = *static_cast<const float*>(alpha);
      const float beta_val = *static_cast<const float*>(beta);
      const float coef_val = static_cast<float>(act_d->coef);
      const std::array<ShaderArg, 7> args4 = {
          ShaderArg::Buffer(buf_x),   ShaderArg::Buffer(buf_y),    ShaderArg::Value(n_vec4),
          ShaderArg::Value(mode_val), ShaderArg::Value(alpha_val), ShaderArg::Value(beta_val),
          ShaderArg::Value(coef_val),
      };
      return (exec.DispatchThreads1D(pipe4, n_vec4, 256, args4, 0, stream) == cudaSuccess)
                 ? CUDNN_STATUS_SUCCESS
                 : CUDNN_STATUS_EXECUTION_FAILED;
    }
  }

  id<MTLComputePipelineState> pipe = GetCudnnPipeline("muda_cudnn_activation_fwd");
  if (pipe == nil) {
    return CUDNN_STATUS_INTERNAL_ERROR;
  }

  CudnnTensorOpParams params{};
  params.n = x_d->n;
  params.c = x_d->c;
  params.h = x_d->h;
  params.w = x_d->w;
  params.x_ns = x_d->n_stride;
  params.x_cs = x_d->c_stride;
  params.x_hs = x_d->h_stride;
  params.x_ws = x_d->w_stride;
  params.y_ns = y_d->n_stride;
  params.y_cs = y_d->c_stride;
  params.y_hs = y_d->h_stride;
  params.y_ws = y_d->w_stride;
  params.mode = static_cast<int>(act_d->mode);
  params.alpha = *static_cast<const float*>(alpha);
  params.beta = *static_cast<const float*>(beta);
  params.coef = static_cast<float>(act_d->coef);

  const std::array<ShaderArg, 3> args = {
      ShaderArg::Buffer(buf_x),
      ShaderArg::Buffer(buf_y),
      ShaderArg::Value(params),
  };

  return (exec.DispatchThreads1D(pipe, total, 256, args, 0, stream) == cudaSuccess)
             ? CUDNN_STATUS_SUCCESS
             : CUDNN_STATUS_EXECUTION_FAILED;
}

cudnnStatus_t cudnnPoolingForward(cudnnHandle_t handle, const cudnnPoolingDescriptor_t poolingDesc,
                                  const void* alpha, const cudnnTensorDescriptor_t xDesc,
                                  const void* x, const void* beta,
                                  const cudnnTensorDescriptor_t yDesc, void* y) {
  if (alpha == nullptr || beta == nullptr || x == nullptr || y == nullptr) {
    return CUDNN_STATUS_BAD_PARAM;
  }
  auto ctx = GetCudnnHandleTable().Lookup(handle);
  auto p_d = GetPoolingDescTable().Lookup(poolingDesc);
  auto x_d = GetTensorDescTable().Lookup(xDesc);
  auto y_d = GetTensorDescTable().Lookup(yDesc);
  if (ctx == nullptr || p_d == nullptr || x_d == nullptr || y_d == nullptr || !p_d->configured ||
      !x_d->configured || !y_d->configured) {
    return CUDNN_STATUS_BAD_PARAM;
  }
  if (x_d->n != y_d->n || x_d->c != y_d->c) {
    return CUDNN_STATUS_BAD_PARAM;
  }

  auto& exec = LibraryExecutor::Instance();
  ResolvedBuffer buf_x;
  ResolvedBuffer buf_y;
  if (!exec.ResolveDeviceBuffer(x, x_d->SpanElementCount() * sizeof(float), &buf_x) ||
      !exec.ResolveDeviceBuffer(y, y_d->SpanElementCount() * sizeof(float), &buf_y)) {
    return CUDNN_STATUS_BAD_PARAM;
  }

  id<MTLComputePipelineState> pipe = GetCudnnPipeline("muda_cudnn_pooling2d_fwd");
  if (pipe == nil) {
    return CUDNN_STATUS_INTERNAL_ERROR;
  }

  CudnnPoolParams params{};
  params.n = x_d->n;
  params.c = x_d->c;
  params.h_in = x_d->h;
  params.w_in = x_d->w;
  params.x_ns = x_d->n_stride;
  params.x_cs = x_d->c_stride;
  params.x_hs = x_d->h_stride;
  params.x_ws = x_d->w_stride;
  params.h_out = y_d->h;
  params.w_out = y_d->w;
  params.y_ns = y_d->n_stride;
  params.y_cs = y_d->c_stride;
  params.y_hs = y_d->h_stride;
  params.y_ws = y_d->w_stride;
  params.win_h = p_d->window_h;
  params.win_w = p_d->window_w;
  params.pad_h = p_d->pad_h;
  params.pad_w = p_d->pad_w;
  params.stride_h = p_d->stride_h;
  params.stride_w = p_d->stride_w;
  params.mode = static_cast<int>(p_d->mode);
  params.alpha = *static_cast<const float*>(alpha);
  params.beta = *static_cast<const float*>(beta);

  const std::size_t total = static_cast<std::size_t>(y_d->n) * static_cast<std::size_t>(y_d->c) *
                            static_cast<std::size_t>(y_d->h) * static_cast<std::size_t>(y_d->w);
  const std::array<ShaderArg, 3> args = {
      ShaderArg::Buffer(buf_x),
      ShaderArg::Buffer(buf_y),
      ShaderArg::Value(params),
  };

  cudaStream_t stream = nullptr;
  {
    std::lock_guard<std::mutex> lock(ctx->mutex);
    stream = ctx->stream;
  }
  return (exec.DispatchThreads1D(pipe, total, 256, args, 0, stream) == cudaSuccess)
             ? CUDNN_STATUS_SUCCESS
             : CUDNN_STATUS_EXECUTION_FAILED;
}

cudnnStatus_t cudnnSoftmaxForward(cudnnHandle_t handle, cudnnSoftmaxAlgorithm_t algo,
                                  cudnnSoftmaxMode_t mode, const void* alpha,
                                  const cudnnTensorDescriptor_t xDesc, const void* x,
                                  const void* beta, const cudnnTensorDescriptor_t yDesc, void* y) {
  if (alpha == nullptr || beta == nullptr || x == nullptr || y == nullptr) {
    return CUDNN_STATUS_BAD_PARAM;
  }
  auto ctx = GetCudnnHandleTable().Lookup(handle);
  auto x_d = GetTensorDescTable().Lookup(xDesc);
  auto y_d = GetTensorDescTable().Lookup(yDesc);
  if (ctx == nullptr || x_d == nullptr || y_d == nullptr || !x_d->configured || !y_d->configured) {
    return CUDNN_STATUS_BAD_PARAM;
  }
  if (x_d->n != y_d->n || x_d->c != y_d->c || x_d->h != y_d->h || x_d->w != y_d->w) {
    return CUDNN_STATUS_BAD_PARAM;
  }

  auto& exec = LibraryExecutor::Instance();
  ResolvedBuffer buf_x;
  ResolvedBuffer buf_y;
  if (!exec.ResolveDeviceBuffer(x, x_d->SpanElementCount() * sizeof(float), &buf_x) ||
      !exec.ResolveDeviceBuffer(y, y_d->SpanElementCount() * sizeof(float), &buf_y)) {
    return CUDNN_STATUS_BAD_PARAM;
  }

  cudaStream_t stream = nullptr;
  {
    std::lock_guard<std::mutex> lock(ctx->mutex);
    stream = ctx->stream;
  }

  const int hw = x_d->h * x_d->w;
  const int chw = x_d->c * hw;
  const bool is_contiguous =
      (x_d->w_stride == 1 && x_d->h_stride == x_d->w && x_d->c_stride == hw &&
       x_d->n_stride == chw && y_d->w_stride == 1 && y_d->h_stride == y_d->w &&
       y_d->c_stride == hw && y_d->n_stride == chw);

  if (is_contiguous && (mode == CUDNN_SOFTMAX_MODE_INSTANCE || (x_d->h == 1 && x_d->w == 1))) {
    id<MTLComputePipelineState> warp_pipe = GetCudnnPipeline("muda_cudnn_softmax_warp_contiguous");
    if (warp_pipe != nil) {
      const auto num_slices = static_cast<std::uint32_t>(x_d->n);
      const auto slice_len = static_cast<std::uint32_t>(chw);
      const int algo_val = static_cast<int>(algo);
      const float alpha_val = *static_cast<const float*>(alpha);
      const float beta_val = *static_cast<const float*>(beta);
      const std::array<ShaderArg, 7> warp_args = {
          ShaderArg::Buffer(buf_x),    ShaderArg::Buffer(buf_y),   ShaderArg::Value(num_slices),
          ShaderArg::Value(slice_len), ShaderArg::Value(algo_val), ShaderArg::Value(alpha_val),
          ShaderArg::Value(beta_val),
      };
      const std::size_t gx = (static_cast<std::size_t>(num_slices) + 7u) / 8u;
      return (exec.DispatchThreadgroups(warp_pipe, MTLSizeMake(gx, 1, 1), MTLSizeMake(256, 1, 1),
                                        warp_args, 0, stream) == cudaSuccess)
                 ? CUDNN_STATUS_SUCCESS
                 : CUDNN_STATUS_EXECUTION_FAILED;
    }
  }

  id<MTLComputePipelineState> pipe = GetCudnnPipeline("muda_cudnn_softmax_fwd");
  if (pipe == nil) {
    return CUDNN_STATUS_INTERNAL_ERROR;
  }

  CudnnTensorOpParams params{};
  params.n = x_d->n;
  params.c = x_d->c;
  params.h = x_d->h;
  params.w = x_d->w;
  params.x_ns = x_d->n_stride;
  params.x_cs = x_d->c_stride;
  params.x_hs = x_d->h_stride;
  params.x_ws = x_d->w_stride;
  params.y_ns = y_d->n_stride;
  params.y_cs = y_d->c_stride;
  params.y_hs = y_d->h_stride;
  params.y_ws = y_d->w_stride;
  params.mode = (static_cast<int>(algo) << 1) | (mode == CUDNN_SOFTMAX_MODE_CHANNEL ? 1 : 0);
  params.alpha = *static_cast<const float*>(alpha);
  params.beta = *static_cast<const float*>(beta);

  const std::size_t threads =
      (mode == CUDNN_SOFTMAX_MODE_CHANNEL)
          ? (static_cast<std::size_t>(x_d->n) * static_cast<std::size_t>(x_d->h) *
             static_cast<std::size_t>(x_d->w))
          : static_cast<std::size_t>(x_d->n);
  const std::array<ShaderArg, 3> args = {
      ShaderArg::Buffer(buf_x),
      ShaderArg::Buffer(buf_y),
      ShaderArg::Value(params),
  };

  return (exec.DispatchThreads1D(pipe, threads, 256, args, 0, stream) == cudaSuccess)
             ? CUDNN_STATUS_SUCCESS
             : CUDNN_STATUS_EXECUTION_FAILED;
}

cudnnStatus_t cudnnBatchNormalizationForwardInference(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, const void* alpha, const void* beta,
    const cudnnTensorDescriptor_t xDesc, const void* x, const cudnnTensorDescriptor_t yDesc,
    void* y, const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void* bnScale,
    const void* bnBias, const void* estimatedMean, const void* estimatedVariance, double epsilon) {
  if (alpha == nullptr || beta == nullptr || x == nullptr || y == nullptr || bnScale == nullptr ||
      bnBias == nullptr || estimatedMean == nullptr || estimatedVariance == nullptr ||
      epsilon <= 0.0) {
    return CUDNN_STATUS_BAD_PARAM;
  }
  auto ctx = GetCudnnHandleTable().Lookup(handle);
  auto x_d = GetTensorDescTable().Lookup(xDesc);
  auto y_d = GetTensorDescTable().Lookup(yDesc);
  auto bn_d = GetTensorDescTable().Lookup(bnScaleBiasMeanVarDesc);
  if (ctx == nullptr || x_d == nullptr || y_d == nullptr || bn_d == nullptr || !x_d->configured ||
      !y_d->configured || !bn_d->configured) {
    return CUDNN_STATUS_BAD_PARAM;
  }

  const std::size_t param_elems =
      (mode == CUDNN_BATCHNORM_PER_ACTIVATION)
          ? (static_cast<std::size_t>(x_d->c) * static_cast<std::size_t>(x_d->h) *
             static_cast<std::size_t>(x_d->w))
          : static_cast<std::size_t>(x_d->c);

  auto& exec = LibraryExecutor::Instance();
  ResolvedBuffer buf_x;
  ResolvedBuffer buf_y;
  ResolvedBuffer buf_scale;
  ResolvedBuffer buf_bias;
  ResolvedBuffer buf_mean;
  ResolvedBuffer buf_var;
  if (!exec.ResolveDeviceBuffer(x, x_d->SpanElementCount() * sizeof(float), &buf_x) ||
      !exec.ResolveDeviceBuffer(y, y_d->SpanElementCount() * sizeof(float), &buf_y) ||
      !exec.ResolveDeviceBuffer(bnScale, param_elems * sizeof(float), &buf_scale) ||
      !exec.ResolveDeviceBuffer(bnBias, param_elems * sizeof(float), &buf_bias) ||
      !exec.ResolveDeviceBuffer(estimatedMean, param_elems * sizeof(float), &buf_mean) ||
      !exec.ResolveDeviceBuffer(estimatedVariance, param_elems * sizeof(float), &buf_var)) {
    return CUDNN_STATUS_BAD_PARAM;
  }

  id<MTLComputePipelineState> pipe = GetCudnnPipeline("muda_cudnn_batchnorm_fwd_inference");
  if (pipe == nil) {
    return CUDNN_STATUS_INTERNAL_ERROR;
  }

  CudnnTensorOpParams params{};
  params.n = x_d->n;
  params.c = x_d->c;
  params.h = x_d->h;
  params.w = x_d->w;
  params.x_ns = x_d->n_stride;
  params.x_cs = x_d->c_stride;
  params.x_hs = x_d->h_stride;
  params.x_ws = x_d->w_stride;
  params.y_ns = y_d->n_stride;
  params.y_cs = y_d->c_stride;
  params.y_hs = y_d->h_stride;
  params.y_ws = y_d->w_stride;
  params.mode = (mode == CUDNN_BATCHNORM_PER_ACTIVATION) ? 0 : 1;
  params.alpha = *static_cast<const float*>(alpha);
  params.beta = *static_cast<const float*>(beta);
  params.coef = static_cast<float>(epsilon);

  const std::size_t total = static_cast<std::size_t>(x_d->n) * static_cast<std::size_t>(x_d->c) *
                            static_cast<std::size_t>(x_d->h) * static_cast<std::size_t>(x_d->w);
  const std::array<ShaderArg, 7> args = {
      ShaderArg::Buffer(buf_x),    ShaderArg::Buffer(buf_y),    ShaderArg::Buffer(buf_scale),
      ShaderArg::Buffer(buf_bias), ShaderArg::Buffer(buf_mean), ShaderArg::Buffer(buf_var),
      ShaderArg::Value(params),
  };

  cudaStream_t stream = nullptr;
  {
    std::lock_guard<std::mutex> lock(ctx->mutex);
    stream = ctx->stream;
  }
  return (exec.DispatchThreads1D(pipe, total, 256, args, 0, stream) == cudaSuccess)
             ? CUDNN_STATUS_SUCCESS
             : CUDNN_STATUS_EXECUTION_FAILED;
}

cudnnStatus_t cudnnBatchNormalizationForwardTraining(
    cudnnHandle_t handle, cudnnBatchNormMode_t /*mode*/, const void* alpha, const void* beta,
    const cudnnTensorDescriptor_t xDesc, const void* x, const cudnnTensorDescriptor_t yDesc,
    void* y, const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void* bnScale,
    const void* bnBias, double exponentialAverageFactor, void* resultRunningMean,
    void* resultRunningVariance, double epsilon, void* resultSaveMean,
    void* resultSaveInvVariance) {
  if (alpha == nullptr || beta == nullptr || x == nullptr || y == nullptr || bnScale == nullptr ||
      bnBias == nullptr || epsilon <= 0.0) {
    return CUDNN_STATUS_BAD_PARAM;
  }
  auto ctx = GetCudnnHandleTable().Lookup(handle);
  auto x_d = GetTensorDescTable().Lookup(xDesc);
  auto y_d = GetTensorDescTable().Lookup(yDesc);
  auto bn_d = GetTensorDescTable().Lookup(bnScaleBiasMeanVarDesc);
  if (ctx == nullptr || x_d == nullptr || y_d == nullptr || bn_d == nullptr || !x_d->configured ||
      !y_d->configured || !bn_d->configured) {
    return CUDNN_STATUS_BAD_PARAM;
  }

  const auto c_elems = static_cast<std::size_t>(x_d->c);
  auto& exec = LibraryExecutor::Instance();
  ResolvedBuffer buf_x;
  ResolvedBuffer buf_y;
  ResolvedBuffer buf_scale;
  ResolvedBuffer buf_bias;
  if (!exec.ResolveDeviceBuffer(x, x_d->SpanElementCount() * sizeof(float), &buf_x) ||
      !exec.ResolveDeviceBuffer(y, y_d->SpanElementCount() * sizeof(float), &buf_y) ||
      !exec.ResolveDeviceBuffer(bnScale, c_elems * sizeof(float), &buf_scale) ||
      !exec.ResolveDeviceBuffer(bnBias, c_elems * sizeof(float), &buf_bias)) {
    return CUDNN_STATUS_BAD_PARAM;
  }

  ResolvedBuffer buf_rmean = buf_scale;
  ResolvedBuffer buf_rvar = buf_scale;
  std::uint32_t has_running = 0u;
  if (resultRunningMean != nullptr || resultRunningVariance != nullptr) {
    if (resultRunningMean == nullptr || resultRunningVariance == nullptr ||
        !exec.ResolveDeviceBuffer(resultRunningMean, c_elems * sizeof(float), &buf_rmean) ||
        !exec.ResolveDeviceBuffer(resultRunningVariance, c_elems * sizeof(float), &buf_rvar)) {
      return CUDNN_STATUS_BAD_PARAM;
    }
    has_running = 1u;
  }

  ResolvedBuffer buf_smean = buf_scale;
  ResolvedBuffer buf_sivar = buf_scale;
  std::uint32_t has_save = 0u;
  if (resultSaveMean != nullptr || resultSaveInvVariance != nullptr) {
    if (resultSaveMean == nullptr || resultSaveInvVariance == nullptr ||
        !exec.ResolveDeviceBuffer(resultSaveMean, c_elems * sizeof(float), &buf_smean) ||
        !exec.ResolveDeviceBuffer(resultSaveInvVariance, c_elems * sizeof(float), &buf_sivar)) {
      return CUDNN_STATUS_BAD_PARAM;
    }
    has_save = 1u;
  }

  id<MTLComputePipelineState> pipe = GetCudnnPipeline("muda_cudnn_batchnorm_fwd_training");
  if (pipe == nil) {
    return CUDNN_STATUS_INTERNAL_ERROR;
  }

  CudnnTensorOpParams params{};
  params.n = x_d->n;
  params.c = x_d->c;
  params.h = x_d->h;
  params.w = x_d->w;
  params.x_ns = x_d->n_stride;
  params.x_cs = x_d->c_stride;
  params.x_hs = x_d->h_stride;
  params.x_ws = x_d->w_stride;
  params.y_ns = y_d->n_stride;
  params.y_cs = y_d->c_stride;
  params.y_hs = y_d->h_stride;
  params.y_ws = y_d->w_stride;
  params.alpha = *static_cast<const float*>(alpha);
  params.beta = *static_cast<const float*>(beta);
  params.coef = static_cast<float>(epsilon);
  const auto exp_factor = static_cast<float>(exponentialAverageFactor);

  const std::array<ShaderArg, 12> args = {
      ShaderArg::Buffer(buf_x),     ShaderArg::Buffer(buf_y),      ShaderArg::Buffer(buf_scale),
      ShaderArg::Buffer(buf_bias),  ShaderArg::Buffer(buf_rmean),  ShaderArg::Buffer(buf_rvar),
      ShaderArg::Buffer(buf_smean), ShaderArg::Buffer(buf_sivar),  ShaderArg::Value(params),
      ShaderArg::Value(exp_factor), ShaderArg::Value(has_running), ShaderArg::Value(has_save),
  };

  cudaStream_t stream = nullptr;
  {
    std::lock_guard<std::mutex> lock(ctx->mutex);
    stream = ctx->stream;
  }
  return (exec.DispatchThreads1D(pipe, c_elems, 256, args, 0, stream) == cudaSuccess)
             ? CUDNN_STATUS_SUCCESS
             : CUDNN_STATUS_EXECUTION_FAILED;
}

}  // extern "C"

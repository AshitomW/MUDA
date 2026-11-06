/**
 * @file cudnn_descriptors.hpp
 * @brief Internal cuDNN handle and descriptor state definitions backed by HandleTable.
 */

#pragma once

#include <cstddef>
#include <memory>
#include <mutex>

#include "cudnn.h"
#include "handle_table.hpp"

namespace muda::libraries::dnn {

struct CudnnContext {
  mutable std::mutex mutex;
  cudaStream_t stream = nullptr;
};

struct TensorDescriptorState {
  mutable std::mutex mutex;
  bool configured = false;
  cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW;
  cudnnDataType_t data_type = CUDNN_DATA_FLOAT;
  int n = 0;
  int c = 0;
  int h = 0;
  int w = 0;
  int n_stride = 0;
  int c_stride = 0;
  int h_stride = 0;
  int w_stride = 0;

  [[nodiscard]] std::size_t SpanElementCount() const noexcept {
    if (!configured || n <= 0 || c <= 0 || h <= 0 || w <= 0) {
      return 0;
    }
    return 1u + static_cast<std::size_t>(n - 1) * static_cast<std::size_t>(n_stride) +
           static_cast<std::size_t>(c - 1) * static_cast<std::size_t>(c_stride) +
           static_cast<std::size_t>(h - 1) * static_cast<std::size_t>(h_stride) +
           static_cast<std::size_t>(w - 1) * static_cast<std::size_t>(w_stride);
  }
};

struct FilterDescriptorState {
  mutable std::mutex mutex;
  bool configured = false;
  cudnnDataType_t data_type = CUDNN_DATA_FLOAT;
  cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW;
  int k = 0;
  int c = 0;
  int h = 0;
  int w = 0;

  [[nodiscard]] std::size_t ElementCount() const noexcept {
    if (!configured || k <= 0 || c <= 0 || h <= 0 || w <= 0) {
      return 0;
    }
    return static_cast<std::size_t>(k) * static_cast<std::size_t>(c) *
           static_cast<std::size_t>(h) * static_cast<std::size_t>(w);
  }
};

struct ConvolutionDescriptorState {
  mutable std::mutex mutex;
  bool configured = false;
  int pad_h = 0;
  int pad_w = 0;
  int stride_h = 1;
  int stride_w = 1;
  int dilation_h = 1;
  int dilation_w = 1;
  int group_count = 1;
  cudnnConvolutionMode_t mode = CUDNN_CROSS_CORRELATION;
  cudnnDataType_t compute_type = CUDNN_DATA_FLOAT;
};

struct ActivationDescriptorState {
  mutable std::mutex mutex;
  bool configured = false;
  cudnnActivationMode_t mode = CUDNN_ACTIVATION_RELU;
  cudnnNanPropagation_t nan_opt = CUDNN_NOT_PROPAGATE_NAN;
  double coef = 0.0;
};

struct PoolingDescriptorState {
  mutable std::mutex mutex;
  bool configured = false;
  cudnnPoolingMode_t mode = CUDNN_POOLING_MAX;
  cudnnNanPropagation_t nan_opt = CUDNN_NOT_PROPAGATE_NAN;
  int window_h = 1;
  int window_w = 1;
  int pad_h = 0;
  int pad_w = 0;
  int stride_h = 1;
  int stride_w = 1;
};

using CudnnHandleTable =
    runtime::cudart::HandleTable<CudnnContext, runtime::cudart::HandleKind::kCudnnHandle>;
using TensorDescTable =
    runtime::cudart::HandleTable<TensorDescriptorState,
                                 runtime::cudart::HandleKind::kCudnnTensorDesc>;
using FilterDescTable =
    runtime::cudart::HandleTable<FilterDescriptorState,
                                 runtime::cudart::HandleKind::kCudnnFilterDesc>;
using ConvDescTable =
    runtime::cudart::HandleTable<ConvolutionDescriptorState,
                                 runtime::cudart::HandleKind::kCudnnConvDesc>;
using ActivationDescTable =
    runtime::cudart::HandleTable<ActivationDescriptorState,
                                 runtime::cudart::HandleKind::kCudnnActivationDesc>;
using PoolingDescTable =
    runtime::cudart::HandleTable<PoolingDescriptorState,
                                 runtime::cudart::HandleKind::kCudnnPoolingDesc>;

[[nodiscard]] CudnnHandleTable& GetCudnnHandleTable() noexcept;
[[nodiscard]] TensorDescTable& GetTensorDescTable() noexcept;
[[nodiscard]] FilterDescTable& GetFilterDescTable() noexcept;
[[nodiscard]] ConvDescTable& GetConvDescTable() noexcept;
[[nodiscard]] ActivationDescTable& GetActivationDescTable() noexcept;
[[nodiscard]] PoolingDescTable& GetPoolingDescTable() noexcept;

[[nodiscard]] constexpr std::size_t DataTypeSizeBytes(cudnnDataType_t dtype) noexcept {
  switch (dtype) {
    case CUDNN_DATA_FLOAT:
      return 4;
    case CUDNN_DATA_HALF:
      return 2;
    case CUDNN_DATA_INT8:
      return 1;
    case CUDNN_DATA_INT32:
      return 4;
    case CUDNN_DATA_DOUBLE:
      return 8;
  }
  return 0;
}

}  // namespace muda::libraries::dnn


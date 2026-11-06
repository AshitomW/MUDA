/**
 * @file cudnn_descriptors.mm
 * @brief Implementation of cuDNN context and descriptor creation, configuration, and inspection.
 */

#include "cudnn_descriptors.hpp"

#include "muda/libraries/metal_executor.hpp"

namespace muda::libraries::dnn {
namespace {

[[nodiscard]] constexpr bool IsSupportedDataType(cudnnDataType_t dt) noexcept {
  return dt == CUDNN_DATA_FLOAT || dt == CUDNN_DATA_HALF;
}

}  // namespace

CudnnHandleTable& GetCudnnHandleTable() noexcept {
  static CudnnHandleTable table;
  return table;
}

TensorDescTable& GetTensorDescTable() noexcept {
  static TensorDescTable table;
  return table;
}

FilterDescTable& GetFilterDescTable() noexcept {
  static FilterDescTable table;
  return table;
}

ConvDescTable& GetConvDescTable() noexcept {
  static ConvDescTable table;
  return table;
}

ActivationDescTable& GetActivationDescTable() noexcept {
  static ActivationDescTable table;
  return table;
}

PoolingDescTable& GetPoolingDescTable() noexcept {
  static PoolingDescTable table;
  return table;
}

}  // namespace muda::libraries::dnn

using namespace muda::libraries;
using namespace muda::libraries::dnn;

extern "C" {

size_t cudnnGetVersion(void) { return CUDNN_VERSION; }

const char* cudnnGetErrorString(cudnnStatus_t status) {
  switch (status) {
    case CUDNN_STATUS_SUCCESS:
      return "CUDNN_STATUS_SUCCESS";
    case CUDNN_STATUS_NOT_INITIALIZED:
      return "CUDNN_STATUS_NOT_INITIALIZED";
    case CUDNN_STATUS_ALLOC_FAILED:
      return "CUDNN_STATUS_ALLOC_FAILED";
    case CUDNN_STATUS_BAD_PARAM:
      return "CUDNN_STATUS_BAD_PARAM";
    case CUDNN_STATUS_INTERNAL_ERROR:
      return "CUDNN_STATUS_INTERNAL_ERROR";
    case CUDNN_STATUS_INVALID_VALUE:
      return "CUDNN_STATUS_INVALID_VALUE";
    case CUDNN_STATUS_ARCH_MISMATCH:
      return "CUDNN_STATUS_ARCH_MISMATCH";
    case CUDNN_STATUS_MAPPING_ERROR:
      return "CUDNN_STATUS_MAPPING_ERROR";
    case CUDNN_STATUS_EXECUTION_FAILED:
      return "CUDNN_STATUS_EXECUTION_FAILED";
    case CUDNN_STATUS_NOT_SUPPORTED:
      return "CUDNN_STATUS_NOT_SUPPORTED";
  }
  return "CUDNN_STATUS_UNKNOWN";
}

cudnnStatus_t cudnnCreate(cudnnHandle_t* handle) {
  if (handle == nullptr) {
    return CUDNN_STATUS_BAD_PARAM;
  }
  if (LibraryExecutor::Instance().EnsureInitialized() != cudaSuccess) {
    return CUDNN_STATUS_NOT_INITIALIZED;
  }
  auto ctx = std::make_shared<CudnnContext>();
  *handle = GetCudnnHandleTable().Insert<cudnnHandle_t>(std::move(ctx));
  return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroy(cudnnHandle_t handle) {
  auto removed = GetCudnnHandleTable().Remove(handle);
  if (removed == nullptr) {
    return CUDNN_STATUS_BAD_PARAM;
  }
  return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetStream(cudnnHandle_t handle, cudaStream_t streamId) {
  auto ctx = GetCudnnHandleTable().Lookup(handle);
  if (ctx == nullptr) {
    return CUDNN_STATUS_BAD_PARAM;
  }
  if (!LibraryExecutor::Instance().IsValidStream(streamId)) {
    return CUDNN_STATUS_BAD_PARAM;
  }
  std::lock_guard<std::mutex> lock(ctx->mutex);
  ctx->stream = streamId;
  return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetStream(cudnnHandle_t handle, cudaStream_t* streamId) {
  if (streamId == nullptr) {
    return CUDNN_STATUS_BAD_PARAM;
  }
  auto ctx = GetCudnnHandleTable().Lookup(handle);
  if (ctx == nullptr) {
    return CUDNN_STATUS_BAD_PARAM;
  }
  std::lock_guard<std::mutex> lock(ctx->mutex);
  *streamId = ctx->stream;
  return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t* tensorDesc) {
  if (tensorDesc == nullptr) {
    return CUDNN_STATUS_BAD_PARAM;
  }
  auto desc = std::make_shared<TensorDescriptorState>();
  *tensorDesc = GetTensorDescTable().Insert<cudnnTensorDescriptor_t>(std::move(desc));
  return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetTensor4dDescriptor(cudnnTensorDescriptor_t tensorDesc,
                                         cudnnTensorFormat_t format, cudnnDataType_t dataType,
                                         int n, int c, int h, int w) {
  auto desc = GetTensorDescTable().Lookup(tensorDesc);
  if (desc == nullptr || n <= 0 || c <= 0 || h <= 0 || w <= 0) {
    return CUDNN_STATUS_BAD_PARAM;
  }
  if (format != CUDNN_TENSOR_NCHW && format != CUDNN_TENSOR_NHWC) {
    return CUDNN_STATUS_BAD_PARAM;
  }
  if (dataType == CUDNN_DATA_DOUBLE) {
    return CUDNN_STATUS_NOT_SUPPORTED;
  }
  if (!IsSupportedDataType(dataType)) {
    return CUDNN_STATUS_BAD_PARAM;
  }

  std::lock_guard<std::mutex> lock(desc->mutex);
  desc->configured = true;
  desc->format = format;
  desc->data_type = dataType;
  desc->n = n;
  desc->c = c;
  desc->h = h;
  desc->w = w;
  if (format == CUDNN_TENSOR_NCHW) {
    desc->w_stride = 1;
    desc->h_stride = w;
    desc->c_stride = h * w;
    desc->n_stride = c * h * w;
  } else {
    desc->c_stride = 1;
    desc->w_stride = c;
    desc->h_stride = w * c;
    desc->n_stride = h * w * c;
  }
  return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetTensor4dDescriptorEx(cudnnTensorDescriptor_t tensorDesc,
                                           cudnnDataType_t dataType, int n, int c, int h, int w,
                                           int nStride, int cStride, int hStride, int wStride) {
  auto desc = GetTensorDescTable().Lookup(tensorDesc);
  if (desc == nullptr || n <= 0 || c <= 0 || h <= 0 || w <= 0 || nStride <= 0 || cStride <= 0 ||
      hStride <= 0 || wStride <= 0) {
    return CUDNN_STATUS_BAD_PARAM;
  }
  if (dataType == CUDNN_DATA_DOUBLE) {
    return CUDNN_STATUS_NOT_SUPPORTED;
  }
  if (!IsSupportedDataType(dataType)) {
    return CUDNN_STATUS_BAD_PARAM;
  }

  std::lock_guard<std::mutex> lock(desc->mutex);
  desc->configured = true;
  desc->format = (cStride == 1 && wStride >= c) ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW;
  desc->data_type = dataType;
  desc->n = n;
  desc->c = c;
  desc->h = h;
  desc->w = w;
  desc->n_stride = nStride;
  desc->c_stride = cStride;
  desc->h_stride = hStride;
  desc->w_stride = wStride;
  return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetTensor4dDescriptor(const cudnnTensorDescriptor_t tensorDesc,
                                         cudnnDataType_t* dataType, int* n, int* c, int* h, int* w,
                                         int* nStride, int* cStride, int* hStride, int* wStride) {
  auto desc = GetTensorDescTable().Lookup(tensorDesc);
  if (desc == nullptr || !desc->configured) {
    return CUDNN_STATUS_BAD_PARAM;
  }
  std::lock_guard<std::mutex> lock(desc->mutex);
  if (dataType != nullptr) *dataType = desc->data_type;
  if (n != nullptr) *n = desc->n;
  if (c != nullptr) *c = desc->c;
  if (h != nullptr) *h = desc->h;
  if (w != nullptr) *w = desc->w;
  if (nStride != nullptr) *nStride = desc->n_stride;
  if (cStride != nullptr) *cStride = desc->c_stride;
  if (hStride != nullptr) *hStride = desc->h_stride;
  if (wStride != nullptr) *wStride = desc->w_stride;
  return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t tensorDesc) {
  return (GetTensorDescTable().Remove(tensorDesc) != nullptr) ? CUDNN_STATUS_SUCCESS
                                                              : CUDNN_STATUS_BAD_PARAM;
}

cudnnStatus_t cudnnCreateFilterDescriptor(cudnnFilterDescriptor_t* filterDesc) {
  if (filterDesc == nullptr) {
    return CUDNN_STATUS_BAD_PARAM;
  }
  auto desc = std::make_shared<FilterDescriptorState>();
  *filterDesc = GetFilterDescTable().Insert<cudnnFilterDescriptor_t>(std::move(desc));
  return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetFilter4dDescriptor(cudnnFilterDescriptor_t filterDesc,
                                         cudnnDataType_t dataType, cudnnTensorFormat_t format,
                                         int k, int c, int h, int w) {
  auto desc = GetFilterDescTable().Lookup(filterDesc);
  if (desc == nullptr || k <= 0 || c <= 0 || h <= 0 || w <= 0) {
    return CUDNN_STATUS_BAD_PARAM;
  }
  if (format != CUDNN_TENSOR_NCHW && format != CUDNN_TENSOR_NHWC) {
    return CUDNN_STATUS_BAD_PARAM;
  }
  if (dataType == CUDNN_DATA_DOUBLE) {
    return CUDNN_STATUS_NOT_SUPPORTED;
  }
  if (!IsSupportedDataType(dataType)) {
    return CUDNN_STATUS_BAD_PARAM;
  }
  std::lock_guard<std::mutex> lock(desc->mutex);
  desc->configured = true;
  desc->data_type = dataType;
  desc->format = format;
  desc->k = k;
  desc->c = c;
  desc->h = h;
  desc->w = w;
  return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetFilter4dDescriptor(const cudnnFilterDescriptor_t filterDesc,
                                         cudnnDataType_t* dataType, cudnnTensorFormat_t* format,
                                         int* k, int* c, int* h, int* w) {
  auto desc = GetFilterDescTable().Lookup(filterDesc);
  if (desc == nullptr || !desc->configured) {
    return CUDNN_STATUS_BAD_PARAM;
  }
  std::lock_guard<std::mutex> lock(desc->mutex);
  if (dataType != nullptr) *dataType = desc->data_type;
  if (format != nullptr) *format = desc->format;
  if (k != nullptr) *k = desc->k;
  if (c != nullptr) *c = desc->c;
  if (h != nullptr) *h = desc->h;
  if (w != nullptr) *w = desc->w;
  return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyFilterDescriptor(cudnnFilterDescriptor_t filterDesc) {
  return (GetFilterDescTable().Remove(filterDesc) != nullptr) ? CUDNN_STATUS_SUCCESS
                                                              : CUDNN_STATUS_BAD_PARAM;
}

cudnnStatus_t cudnnCreateConvolutionDescriptor(cudnnConvolutionDescriptor_t* convDesc) {
  if (convDesc == nullptr) {
    return CUDNN_STATUS_BAD_PARAM;
  }
  auto desc = std::make_shared<ConvolutionDescriptorState>();
  *convDesc = GetConvDescTable().Insert<cudnnConvolutionDescriptor_t>(std::move(desc));
  return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetConvolution2dDescriptor(cudnnConvolutionDescriptor_t convDesc, int pad_h,
                                              int pad_w, int u, int v, int dilation_h,
                                              int dilation_w, cudnnConvolutionMode_t mode,
                                              cudnnDataType_t computeType) {
  auto desc = GetConvDescTable().Lookup(convDesc);
  if (desc == nullptr || pad_h < 0 || pad_w < 0 || u <= 0 || v <= 0 || dilation_h <= 0 ||
      dilation_w <= 0) {
    return CUDNN_STATUS_BAD_PARAM;
  }
  if (mode != CUDNN_CONVOLUTION && mode != CUDNN_CROSS_CORRELATION) {
    return CUDNN_STATUS_BAD_PARAM;
  }
  if (computeType == CUDNN_DATA_DOUBLE) {
    return CUDNN_STATUS_NOT_SUPPORTED;
  }
  std::lock_guard<std::mutex> lock(desc->mutex);
  desc->configured = true;
  desc->pad_h = pad_h;
  desc->pad_w = pad_w;
  desc->stride_h = u;
  desc->stride_w = v;
  desc->dilation_h = dilation_h;
  desc->dilation_w = dilation_w;
  desc->mode = mode;
  desc->compute_type = computeType;
  return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetConvolutionGroupCount(cudnnConvolutionDescriptor_t convDesc, int groupCount) {
  auto desc = GetConvDescTable().Lookup(convDesc);
  if (desc == nullptr || groupCount <= 0) {
    return CUDNN_STATUS_BAD_PARAM;
  }
  std::lock_guard<std::mutex> lock(desc->mutex);
  desc->group_count = groupCount;
  return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionGroupCount(cudnnConvolutionDescriptor_t convDesc,
                                            int* groupCount) {
  if (groupCount == nullptr) {
    return CUDNN_STATUS_BAD_PARAM;
  }
  auto desc = GetConvDescTable().Lookup(convDesc);
  if (desc == nullptr) {
    return CUDNN_STATUS_BAD_PARAM;
  }
  std::lock_guard<std::mutex> lock(desc->mutex);
  *groupCount = desc->group_count;
  return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolution2dForwardOutputDim(const cudnnConvolutionDescriptor_t convDesc,
                                                    const cudnnTensorDescriptor_t inputTensorDesc,
                                                    const cudnnFilterDescriptor_t filterDesc,
                                                    int* n, int* c, int* h, int* w) {
  auto c_desc = GetConvDescTable().Lookup(convDesc);
  auto x_desc = GetTensorDescTable().Lookup(inputTensorDesc);
  auto w_desc = GetFilterDescTable().Lookup(filterDesc);
  if (c_desc == nullptr || x_desc == nullptr || w_desc == nullptr || !c_desc->configured ||
      !x_desc->configured || !w_desc->configured) {
    return CUDNN_STATUS_BAD_PARAM;
  }
  const int eff_kh = 1 + (w_desc->h - 1) * c_desc->dilation_h;
  const int eff_kw = 1 + (w_desc->w - 1) * c_desc->dilation_w;
  const int out_h = 1 + (x_desc->h + 2 * c_desc->pad_h - eff_kh) / c_desc->stride_h;
  const int out_w = 1 + (x_desc->w + 2 * c_desc->pad_w - eff_kw) / c_desc->stride_w;
  if (out_h <= 0 || out_w <= 0) {
    return CUDNN_STATUS_BAD_PARAM;
  }
  if (n != nullptr) *n = x_desc->n;
  if (c != nullptr) *c = w_desc->k;
  if (h != nullptr) *h = out_h;
  if (w != nullptr) *w = out_w;
  return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyConvolutionDescriptor(cudnnConvolutionDescriptor_t convDesc) {
  return (GetConvDescTable().Remove(convDesc) != nullptr) ? CUDNN_STATUS_SUCCESS
                                                          : CUDNN_STATUS_BAD_PARAM;
}

cudnnStatus_t cudnnCreateActivationDescriptor(cudnnActivationDescriptor_t* activationDesc) {
  if (activationDesc == nullptr) {
    return CUDNN_STATUS_BAD_PARAM;
  }
  auto desc = std::make_shared<ActivationDescriptorState>();
  *activationDesc = GetActivationDescTable().Insert<cudnnActivationDescriptor_t>(std::move(desc));
  return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetActivationDescriptor(cudnnActivationDescriptor_t activationDesc,
                                           cudnnActivationMode_t mode,
                                           cudnnNanPropagation_t reluNanOpt, double coef) {
  auto desc = GetActivationDescTable().Lookup(activationDesc);
  if (desc == nullptr || mode < CUDNN_ACTIVATION_SIGMOID || mode > CUDNN_ACTIVATION_SWISH) {
    return CUDNN_STATUS_BAD_PARAM;
  }
  std::lock_guard<std::mutex> lock(desc->mutex);
  desc->configured = true;
  desc->mode = mode;
  desc->nan_opt = reluNanOpt;
  desc->coef = coef;
  return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetActivationDescriptor(const cudnnActivationDescriptor_t activationDesc,
                                           cudnnActivationMode_t* mode,
                                           cudnnNanPropagation_t* reluNanOpt, double* coef) {
  auto desc = GetActivationDescTable().Lookup(activationDesc);
  if (desc == nullptr || !desc->configured) {
    return CUDNN_STATUS_BAD_PARAM;
  }
  std::lock_guard<std::mutex> lock(desc->mutex);
  if (mode != nullptr) *mode = desc->mode;
  if (reluNanOpt != nullptr) *reluNanOpt = desc->nan_opt;
  if (coef != nullptr) *coef = desc->coef;
  return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyActivationDescriptor(cudnnActivationDescriptor_t activationDesc) {
  return (GetActivationDescTable().Remove(activationDesc) != nullptr) ? CUDNN_STATUS_SUCCESS
                                                                      : CUDNN_STATUS_BAD_PARAM;
}

cudnnStatus_t cudnnCreatePoolingDescriptor(cudnnPoolingDescriptor_t* poolingDesc) {
  if (poolingDesc == nullptr) {
    return CUDNN_STATUS_BAD_PARAM;
  }
  auto desc = std::make_shared<PoolingDescriptorState>();
  *poolingDesc = GetPoolingDescTable().Insert<cudnnPoolingDescriptor_t>(std::move(desc));
  return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetPooling2dDescriptor(cudnnPoolingDescriptor_t poolingDesc,
                                          cudnnPoolingMode_t mode,
                                          cudnnNanPropagation_t maxpoolingNanOpt, int windowHeight,
                                          int windowWidth, int verticalPadding,
                                          int horizontalPadding, int verticalStride,
                                          int horizontalStride) {
  auto desc = GetPoolingDescTable().Lookup(poolingDesc);
  if (desc == nullptr || windowHeight <= 0 || windowWidth <= 0 || verticalPadding < 0 ||
      horizontalPadding < 0 || verticalStride <= 0 || horizontalStride <= 0) {
    return CUDNN_STATUS_BAD_PARAM;
  }
  if (mode < CUDNN_POOLING_MAX || mode > CUDNN_POOLING_MAX_DETERMINISTIC) {
    return CUDNN_STATUS_BAD_PARAM;
  }
  std::lock_guard<std::mutex> lock(desc->mutex);
  desc->configured = true;
  desc->mode = mode;
  desc->nan_opt = maxpoolingNanOpt;
  desc->window_h = windowHeight;
  desc->window_w = windowWidth;
  desc->pad_h = verticalPadding;
  desc->pad_w = horizontalPadding;
  desc->stride_h = verticalStride;
  desc->stride_w = horizontalStride;
  return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetPooling2dForwardOutputDim(const cudnnPoolingDescriptor_t poolingDesc,
                                                const cudnnTensorDescriptor_t inputTensorDesc,
                                                int* n, int* c, int* h, int* w) {
  auto p_desc = GetPoolingDescTable().Lookup(poolingDesc);
  auto x_desc = GetTensorDescTable().Lookup(inputTensorDesc);
  if (p_desc == nullptr || x_desc == nullptr || !p_desc->configured || !x_desc->configured) {
    return CUDNN_STATUS_BAD_PARAM;
  }
  const int out_h = 1 + (x_desc->h + 2 * p_desc->pad_h - p_desc->window_h) / p_desc->stride_h;
  const int out_w = 1 + (x_desc->w + 2 * p_desc->pad_w - p_desc->window_w) / p_desc->stride_w;
  if (out_h <= 0 || out_w <= 0) {
    return CUDNN_STATUS_BAD_PARAM;
  }
  if (n != nullptr) *n = x_desc->n;
  if (c != nullptr) *c = x_desc->c;
  if (h != nullptr) *h = out_h;
  if (w != nullptr) *w = out_w;
  return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyPoolingDescriptor(cudnnPoolingDescriptor_t poolingDesc) {
  return (GetPoolingDescTable().Remove(poolingDesc) != nullptr) ? CUDNN_STATUS_SUCCESS
                                                                : CUDNN_STATUS_BAD_PARAM;
}

cudnnStatus_t cudnnDeriveBNTensorDescriptor(cudnnTensorDescriptor_t derivedBnDesc,
                                            const cudnnTensorDescriptor_t xDesc,
                                            cudnnBatchNormMode_t mode) {
  auto x = GetTensorDescTable().Lookup(xDesc);
  if (x == nullptr || !x->configured) {
    return CUDNN_STATUS_BAD_PARAM;
  }
  const cudnnDataType_t bn_type =
      (x->data_type == CUDNN_DATA_HALF) ? CUDNN_DATA_FLOAT : x->data_type;
  if (mode == CUDNN_BATCHNORM_PER_ACTIVATION) {
    return cudnnSetTensor4dDescriptor(derivedBnDesc, CUDNN_TENSOR_NCHW, bn_type, 1, x->c, x->h,
                                      x->w);
  }
  return cudnnSetTensor4dDescriptor(derivedBnDesc, CUDNN_TENSOR_NCHW, bn_type, 1, x->c, 1, 1);
}

}  // extern "C"

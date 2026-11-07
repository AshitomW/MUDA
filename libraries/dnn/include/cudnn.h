/**
 * @file cudnn.h
 * @brief Clean-room C API header for the MUDA cuDNN compatibility shim.
 */

#pragma once

#include <cstddef>
#include <cstdint>

#include "cuda_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

#define CUDNN_MAJOR 8
#define CUDNN_MINOR 9
#define CUDNN_PATCHLEVEL 0
#define CUDNN_VERSION (CUDNN_MAJOR * 1000 + CUDNN_MINOR * 100 + CUDNN_PATCHLEVEL)

/**
 * @brief cuDNN status return codes.
 */
enum cudnnStatus_t : std::uint8_t {
  CUDNN_STATUS_SUCCESS = 0,
  CUDNN_STATUS_NOT_INITIALIZED = 1,
  CUDNN_STATUS_ALLOC_FAILED = 2,
  CUDNN_STATUS_BAD_PARAM = 3,
  CUDNN_STATUS_INTERNAL_ERROR = 4,
  CUDNN_STATUS_INVALID_VALUE = 5,
  CUDNN_STATUS_ARCH_MISMATCH = 6,
  CUDNN_STATUS_MAPPING_ERROR = 7,
  CUDNN_STATUS_EXECUTION_FAILED = 8,
  CUDNN_STATUS_NOT_SUPPORTED = 9
};

/**
 * @brief cuDNN element data types.
 */
enum cudnnDataType_t : std::uint8_t {
  CUDNN_DATA_FLOAT = 0,
  CUDNN_DATA_DOUBLE = 1,
  CUDNN_DATA_HALF = 2,
  CUDNN_DATA_INT8 = 3,
  CUDNN_DATA_INT32 = 4
};

/**
 * @brief cuDNN 4D tensor memory layout formats.
 */
enum cudnnTensorFormat_t : std::uint8_t { CUDNN_TENSOR_NCHW = 0, CUDNN_TENSOR_NHWC = 1 };

/**
 * @brief cuDNN convolution mode (true convolution with filter flip vs. cross-correlation).
 */
enum cudnnConvolutionMode_t : std::uint8_t { CUDNN_CONVOLUTION = 0, CUDNN_CROSS_CORRELATION = 1 };

/**
 * @brief cuDNN forward convolution algorithm selectors.
 */
enum cudnnConvolutionFwdAlgo_t : std::uint8_t {
  CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM = 0,
  CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM = 1,
  CUDNN_CONVOLUTION_FWD_ALGO_GEMM = 2,
  CUDNN_CONVOLUTION_FWD_ALGO_DIRECT = 3
};

/**
 * @brief cuDNN backward-data convolution algorithm selectors.
 */
enum cudnnConvolutionBwdDataAlgo_t : std::uint8_t {
  CUDNN_CONVOLUTION_BWD_DATA_ALGO_0 = 0,
  CUDNN_CONVOLUTION_BWD_DATA_ALGO_1 = 1
};

/**
 * @brief cuDNN backward-filter convolution algorithm selectors.
 */
enum cudnnConvolutionBwdFilterAlgo_t : std::uint8_t {
  CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0 = 0,
  CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1 = 1
};

/**
 * @brief cuDNN activation modes.
 */
enum cudnnActivationMode_t : std::uint8_t {
  CUDNN_ACTIVATION_SIGMOID = 0,
  CUDNN_ACTIVATION_RELU = 1,
  CUDNN_ACTIVATION_TANH = 2,
  CUDNN_ACTIVATION_CLIPPED_RELU = 3,
  CUDNN_ACTIVATION_ELU = 4,
  CUDNN_ACTIVATION_IDENTITY = 5,
  CUDNN_ACTIVATION_SWISH = 6
};

/**
 * @brief cuDNN NaN propagation policy.
 */
enum cudnnNanPropagation_t : std::uint8_t { CUDNN_NOT_PROPAGATE_NAN = 0, CUDNN_PROPAGATE_NAN = 1 };

/**
 * @brief cuDNN 2D pooling modes.
 */
enum cudnnPoolingMode_t : std::uint8_t {
  CUDNN_POOLING_MAX = 0,
  CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING = 1,
  CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING = 2,
  CUDNN_POOLING_MAX_DETERMINISTIC = 3
};

/**
 * @brief cuDNN softmax evaluation algorithms.
 */
enum cudnnSoftmaxAlgorithm_t : std::uint8_t {
  CUDNN_SOFTMAX_FAST = 0,
  CUDNN_SOFTMAX_ACCURATE = 1,
  CUDNN_SOFTMAX_LOG = 2
};

/**
 * @brief cuDNN softmax dimension modes.
 */
enum cudnnSoftmaxMode_t : std::uint8_t {
  CUDNN_SOFTMAX_MODE_INSTANCE = 0,
  CUDNN_SOFTMAX_MODE_CHANNEL = 1
};

/**
 * @brief cuDNN batch normalization modes.
 */
enum cudnnBatchNormMode_t : std::uint8_t {
  CUDNN_BATCHNORM_PER_ACTIVATION = 0,
  CUDNN_BATCHNORM_SPATIAL = 1,
  CUDNN_BATCHNORM_SPATIAL_PERSISTENT = 2
};

/**
 * @brief Opaque cuDNN handles and descriptors.
 */
using cudnnHandle_t = struct cudnnContext*;
using cudnnTensorDescriptor_t = struct cudnnTensorStruct*;
using cudnnFilterDescriptor_t = struct cudnnFilterStruct*;
using cudnnConvolutionDescriptor_t = struct cudnnConvolutionStruct*;
using cudnnActivationDescriptor_t = struct cudnnActivationStruct*;
using cudnnPoolingDescriptor_t = struct cudnnPoolingStruct*;

/* Context & Version */
size_t cudnnGetVersion(void);
const char* cudnnGetErrorString(cudnnStatus_t status);
cudnnStatus_t cudnnCreate(cudnnHandle_t* handle);
cudnnStatus_t cudnnDestroy(cudnnHandle_t handle);
cudnnStatus_t cudnnSetStream(cudnnHandle_t handle, cudaStream_t streamId);
cudnnStatus_t cudnnGetStream(cudnnHandle_t handle, cudaStream_t* streamId);

/* Tensor Descriptors */
cudnnStatus_t cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t* tensorDesc);
cudnnStatus_t cudnnSetTensor4dDescriptor(cudnnTensorDescriptor_t tensorDesc,
                                         cudnnTensorFormat_t format, cudnnDataType_t dataType,
                                         int n, int c, int h, int w);
cudnnStatus_t cudnnSetTensor4dDescriptorEx(cudnnTensorDescriptor_t tensorDesc,
                                           cudnnDataType_t dataType, int n, int c, int h, int w,
                                           int nStride, int cStride, int hStride, int wStride);
cudnnStatus_t cudnnGetTensor4dDescriptor(cudnnTensorDescriptor_t tensorDesc,
                                         cudnnDataType_t* dataType, int* n, int* c, int* h, int* w,
                                         int* nStride, int* cStride, int* hStride, int* wStride);
cudnnStatus_t cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t tensorDesc);

/* Filter Descriptors */
cudnnStatus_t cudnnCreateFilterDescriptor(cudnnFilterDescriptor_t* filterDesc);
cudnnStatus_t cudnnSetFilter4dDescriptor(cudnnFilterDescriptor_t filterDesc,
                                         cudnnDataType_t dataType, cudnnTensorFormat_t format,
                                         int k, int c, int h, int w);
cudnnStatus_t cudnnGetFilter4dDescriptor(cudnnFilterDescriptor_t filterDesc,
                                         cudnnDataType_t* dataType, cudnnTensorFormat_t* format,
                                         int* k, int* c, int* h, int* w);
cudnnStatus_t cudnnDestroyFilterDescriptor(cudnnFilterDescriptor_t filterDesc);

/* Convolution Descriptors */
cudnnStatus_t cudnnCreateConvolutionDescriptor(cudnnConvolutionDescriptor_t* convDesc);
cudnnStatus_t cudnnSetConvolution2dDescriptor(cudnnConvolutionDescriptor_t convDesc, int pad_h,
                                              int pad_w, int u, int v, int dilation_h,
                                              int dilation_w, cudnnConvolutionMode_t mode,
                                              cudnnDataType_t computeType);
cudnnStatus_t cudnnSetConvolutionGroupCount(cudnnConvolutionDescriptor_t convDesc, int groupCount);
cudnnStatus_t cudnnGetConvolutionGroupCount(cudnnConvolutionDescriptor_t convDesc, int* groupCount);
cudnnStatus_t cudnnGetConvolution2dForwardOutputDim(cudnnConvolutionDescriptor_t convDesc,
                                                    cudnnTensorDescriptor_t inputTensorDesc,
                                                    cudnnFilterDescriptor_t filterDesc, int* n,
                                                    int* c, int* h, int* w);
cudnnStatus_t cudnnDestroyConvolutionDescriptor(cudnnConvolutionDescriptor_t convDesc);

/* Activation Descriptors */
cudnnStatus_t cudnnCreateActivationDescriptor(cudnnActivationDescriptor_t* activationDesc);
cudnnStatus_t cudnnSetActivationDescriptor(cudnnActivationDescriptor_t activationDesc,
                                           cudnnActivationMode_t mode,
                                           cudnnNanPropagation_t reluNanOpt, double coef);
cudnnStatus_t cudnnGetActivationDescriptor(cudnnActivationDescriptor_t activationDesc,
                                           cudnnActivationMode_t* mode,
                                           cudnnNanPropagation_t* reluNanOpt, double* coef);
cudnnStatus_t cudnnDestroyActivationDescriptor(cudnnActivationDescriptor_t activationDesc);

/* Pooling Descriptors */
cudnnStatus_t cudnnCreatePoolingDescriptor(cudnnPoolingDescriptor_t* poolingDesc);
cudnnStatus_t cudnnSetPooling2dDescriptor(cudnnPoolingDescriptor_t poolingDesc,
                                          cudnnPoolingMode_t mode,
                                          cudnnNanPropagation_t maxpoolingNanOpt, int windowHeight,
                                          int windowWidth, int verticalPadding,
                                          int horizontalPadding, int verticalStride,
                                          int horizontalStride);
cudnnStatus_t cudnnGetPooling2dForwardOutputDim(cudnnPoolingDescriptor_t poolingDesc,
                                                cudnnTensorDescriptor_t inputTensorDesc, int* n,
                                                int* c, int* h, int* w);
cudnnStatus_t cudnnDestroyPoolingDescriptor(cudnnPoolingDescriptor_t poolingDesc);

/* Convolution Operations */
cudnnStatus_t cudnnGetConvolutionForwardWorkspaceSize(
    cudnnHandle_t handle, cudnnTensorDescriptor_t xDesc, cudnnFilterDescriptor_t wDesc,
    cudnnConvolutionDescriptor_t convDesc, cudnnTensorDescriptor_t yDesc,
    cudnnConvolutionFwdAlgo_t algo, size_t* sizeInBytes);
cudnnStatus_t cudnnConvolutionForward(cudnnHandle_t handle, const void* alpha,
                                      cudnnTensorDescriptor_t xDesc, const void* x,
                                      cudnnFilterDescriptor_t wDesc, const void* w,
                                      cudnnConvolutionDescriptor_t convDesc,
                                      cudnnConvolutionFwdAlgo_t algo, void* workSpace,
                                      size_t workSpaceSizeInBytes, const void* beta,
                                      cudnnTensorDescriptor_t yDesc, void* y);
cudnnStatus_t cudnnConvolutionBackwardData(cudnnHandle_t handle, const void* alpha,
                                           cudnnFilterDescriptor_t wDesc, const void* w,
                                           cudnnTensorDescriptor_t dyDesc, const void* dy,
                                           cudnnConvolutionDescriptor_t convDesc,
                                           cudnnConvolutionBwdDataAlgo_t algo, void* workSpace,
                                           size_t workSpaceSizeInBytes, const void* beta,
                                           cudnnTensorDescriptor_t dxDesc, void* dx);
cudnnStatus_t cudnnConvolutionBackwardFilter(cudnnHandle_t handle, const void* alpha,
                                             cudnnTensorDescriptor_t xDesc, const void* x,
                                             cudnnTensorDescriptor_t dyDesc, const void* dy,
                                             cudnnConvolutionDescriptor_t convDesc,
                                             cudnnConvolutionBwdFilterAlgo_t algo, void* workSpace,
                                             size_t workSpaceSizeInBytes, const void* beta,
                                             cudnnFilterDescriptor_t dwDesc, void* dw);
cudnnStatus_t cudnnConvolutionBackwardBias(cudnnHandle_t handle, const void* alpha,
                                           cudnnTensorDescriptor_t dyDesc, const void* dy,
                                           const void* beta, cudnnTensorDescriptor_t dbDesc,
                                           void* db);

/* Tensor, Activation, Pooling, Softmax & BatchNorm Operations */
cudnnStatus_t cudnnAddTensor(cudnnHandle_t handle, const void* alpha, cudnnTensorDescriptor_t aDesc,
                             const void* A, const void* beta, cudnnTensorDescriptor_t cDesc,
                             void* C);
cudnnStatus_t cudnnActivationForward(cudnnHandle_t handle,
                                     cudnnActivationDescriptor_t activationDesc, const void* alpha,
                                     cudnnTensorDescriptor_t xDesc, const void* x, const void* beta,
                                     cudnnTensorDescriptor_t yDesc, void* y);
cudnnStatus_t cudnnPoolingForward(cudnnHandle_t handle, cudnnPoolingDescriptor_t poolingDesc,
                                  const void* alpha, cudnnTensorDescriptor_t xDesc, const void* x,
                                  const void* beta, cudnnTensorDescriptor_t yDesc, void* y);
cudnnStatus_t cudnnSoftmaxForward(cudnnHandle_t handle, cudnnSoftmaxAlgorithm_t algo,
                                  cudnnSoftmaxMode_t mode, const void* alpha,
                                  cudnnTensorDescriptor_t xDesc, const void* x, const void* beta,
                                  cudnnTensorDescriptor_t yDesc, void* y);
cudnnStatus_t cudnnBatchNormalizationForwardInference(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, const void* alpha, const void* beta,
    cudnnTensorDescriptor_t xDesc, const void* x, cudnnTensorDescriptor_t yDesc, void* y,
    cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void* bnScale, const void* bnBias,
    const void* estimatedMean, const void* estimatedVariance, double epsilon);
cudnnStatus_t cudnnDeriveBNTensorDescriptor(cudnnTensorDescriptor_t derivedBnDesc,
                                            cudnnTensorDescriptor_t xDesc,
                                            cudnnBatchNormMode_t mode);
cudnnStatus_t cudnnBatchNormalizationForwardTraining(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, const void* alpha, const void* beta,
    cudnnTensorDescriptor_t xDesc, const void* x, cudnnTensorDescriptor_t yDesc, void* y,
    cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void* bnScale, const void* bnBias,
    double exponentialAverageFactor, void* resultRunningMean, void* resultRunningVariance,
    double epsilon, void* resultSaveMean, void* resultSaveInvVariance);

#ifdef __cplusplus
}  // extern "C"
#endif

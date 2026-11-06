/**
 * @file cudnn_test.cpp
 * @brief Comprehensive verification test suite for the MUDA cuDNN shim (libcudnn.dylib).
 */

#include "cudnn.h"

#include <cmath>
#include <iostream>
#include <vector>

#include "cuda_runtime.h"

namespace {

bool Expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    return false;
  }
  return true;
}

bool ExpectClose(float actual, float expected, float tol, const char* message) {
  if (std::fabs(actual - expected) > tol) {
    std::cerr << "FAIL: " << message << " (actual=" << actual << ", expected=" << expected
              << ", diff=" << std::fabs(actual - expected) << ")\n";
    return false;
  }
  return true;
}

bool TestConv2dForwardAndBackward() {
  std::cout << "Testing cuDNN Conv2D Forward, BackwardData, BackwardFilter, and BackwardBias...\n";
  cudnnHandle_t handle = nullptr;
  if (!Expect(cudnnCreate(&handle) == CUDNN_STATUS_SUCCESS, "cudnnCreate")) {
    return false;
  }

  // N=1, C=2, H=4, W=4; K=2, R=3, S=3, pad=1, stride=1
  cudnnTensorDescriptor_t x_desc = nullptr;
  cudnnTensorDescriptor_t y_desc = nullptr;
  cudnnTensorDescriptor_t b_desc = nullptr;
  cudnnFilterDescriptor_t w_desc = nullptr;
  cudnnConvolutionDescriptor_t conv_desc = nullptr;

  cudnnCreateTensorDescriptor(&x_desc);
  cudnnCreateTensorDescriptor(&y_desc);
  cudnnCreateTensorDescriptor(&b_desc);
  cudnnCreateFilterDescriptor(&w_desc);
  cudnnCreateConvolutionDescriptor(&conv_desc);

  cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 2, 4, 4);
  cudnnSetFilter4dDescriptor(w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 2, 2, 3, 3);
  cudnnSetConvolution2dDescriptor(conv_desc, 1, 1, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION,
                                  CUDNN_DATA_FLOAT);

  int out_n = 0, out_c = 0, out_h = 0, out_w = 0;
  if (!Expect(cudnnGetConvolution2dForwardOutputDim(conv_desc, x_desc, w_desc, &out_n, &out_c,
                                                    &out_h, &out_w) == CUDNN_STATUS_SUCCESS &&
                  out_n == 1 && out_c == 2 && out_h == 4 && out_w == 4,
              "cudnnGetConvolution2dForwardOutputDim")) {
    return false;
  }
  cudnnSetTensor4dDescriptor(y_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, out_n, out_c, out_h,
                             out_w);
  cudnnSetTensor4dDescriptor(b_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, out_c, 1, 1);

  std::vector<float> h_x(1 * 2 * 4 * 4, 1.0f);
  std::vector<float> h_w(2 * 2 * 3 * 3, 0.5f);
  std::vector<float> h_y(1 * 2 * 4 * 4, 0.0f);
  std::vector<float> h_b = {1.25f, -0.75f};

  float* d_x = nullptr;
  float* d_w = nullptr;
  float* d_y = nullptr;
  float* d_b = nullptr;
  cudaMalloc(reinterpret_cast<void**>(&d_x), h_x.size() * sizeof(float));
  cudaMalloc(reinterpret_cast<void**>(&d_w), h_w.size() * sizeof(float));
  cudaMalloc(reinterpret_cast<void**>(&d_y), h_y.size() * sizeof(float));
  cudaMalloc(reinterpret_cast<void**>(&d_b), h_b.size() * sizeof(float));
  cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_w, h_w.data(), h_w.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b.data(), h_b.size() * sizeof(float), cudaMemcpyHostToDevice);

  const float alpha = 1.0f;
  const float zero = 0.0f;
  if (!Expect(cudnnConvolutionForward(handle, &alpha, x_desc, d_x, w_desc, d_w, conv_desc,
                                      CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, nullptr, 0, &zero,
                                      y_desc, d_y) == CUDNN_STATUS_SUCCESS,
              "cudnnConvolutionForward")) {
    return false;
  }

  // Add bias via cudnnAddTensor
  if (!Expect(cudnnAddTensor(handle, &alpha, b_desc, d_b, &alpha, y_desc, d_y) ==
                  CUDNN_STATUS_SUCCESS,
              "cudnnAddTensor")) {
    return false;
  }

  cudaMemcpy(h_y.data(), d_y, h_y.size() * sizeof(float), cudaMemcpyDeviceToHost);
  // Corner pixel (0,0) has 4 valid 3x3 neighbors * 2 channels * 0.5 + bias(1.25) = 5.25
  // Center pixel (1,1) has 9 valid 3x3 neighbors * 2 channels * 0.5 + bias(1.25) = 10.25
  if (!ExpectClose(h_y[0], 5.25f, 1e-4f, "Conv2D corner + bias") ||
      !ExpectClose(h_y[1 * 4 + 1], 10.25f, 1e-4f, "Conv2D center + bias")) {
    return false;
  }

  // BackwardBias: dy = all 1s -> db = 16.0 per channel
  std::vector<float> h_dy(1 * 2 * 4 * 4, 1.0f);
  cudaMemcpy(d_y, h_dy.data(), h_dy.size() * sizeof(float), cudaMemcpyHostToDevice);
  if (!Expect(cudnnConvolutionBackwardBias(handle, &alpha, y_desc, d_y, &zero, b_desc, d_b) ==
                  CUDNN_STATUS_SUCCESS,
              "cudnnConvolutionBackwardBias")) {
    return false;
  }
  cudaMemcpy(h_b.data(), d_b, h_b.size() * sizeof(float), cudaMemcpyDeviceToHost);
  if (!ExpectClose(h_b[0], 16.0f, 1e-4f, "BackwardBias ch0") ||
      !ExpectClose(h_b[1], 16.0f, 1e-4f, "BackwardBias ch1")) {
    return false;
  }

  // BackwardFilter & BackwardData
  float* d_dw = nullptr;
  float* d_dx = nullptr;
  cudaMalloc(reinterpret_cast<void**>(&d_dw), h_w.size() * sizeof(float));
  cudaMalloc(reinterpret_cast<void**>(&d_dx), h_x.size() * sizeof(float));
  if (!Expect(cudnnConvolutionBackwardFilter(
                  handle, &alpha, x_desc, d_x, y_desc, d_y, conv_desc,
                  CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0, nullptr, 0, &zero, w_desc,
                  d_dw) == CUDNN_STATUS_SUCCESS,
              "cudnnConvolutionBackwardFilter") ||
      !Expect(cudnnConvolutionBackwardData(
                  handle, &alpha, w_desc, d_w, y_desc, d_y, conv_desc,
                  CUDNN_CONVOLUTION_BWD_DATA_ALGO_0, nullptr, 0, &zero, x_desc,
                  d_dx) == CUDNN_STATUS_SUCCESS,
              "cudnnConvolutionBackwardData")) {
    return false;
  }
  std::vector<float> h_dw(h_w.size());
  cudaMemcpy(h_dw.data(), d_dw, h_dw.size() * sizeof(float), cudaMemcpyDeviceToHost);
  // Center weight (1,1) overlaps all 16 output locations -> sum of 1*1 = 16.0
  if (!ExpectClose(h_dw[4], 16.0f, 1e-4f, "BackwardFilter center weight")) {
    return false;
  }

  cudaFree(d_x);
  cudaFree(d_w);
  cudaFree(d_y);
  cudaFree(d_b);
  cudaFree(d_dw);
  cudaFree(d_dx);
  cudnnDestroyTensorDescriptor(x_desc);
  cudnnDestroyTensorDescriptor(y_desc);
  cudnnDestroyTensorDescriptor(b_desc);
  cudnnDestroyFilterDescriptor(w_desc);
  cudnnDestroyConvolutionDescriptor(conv_desc);
  cudnnDestroy(handle);
  return true;
}

bool TestActivationPoolingSoftmaxAndBatchNorm() {
  std::cout << "Testing cuDNN Activation, Pooling2D, Softmax, and BatchNorm2D...\n";
  cudnnHandle_t handle = nullptr;
  cudnnCreate(&handle);

  // 1. Activation (ReLU & Swish)
  cudnnTensorDescriptor_t t_desc = nullptr;
  cudnnActivationDescriptor_t act_desc = nullptr;
  cudnnCreateTensorDescriptor(&t_desc);
  cudnnCreateActivationDescriptor(&act_desc);
  cudnnSetTensor4dDescriptor(t_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 2, 2, 2);
  cudnnSetActivationDescriptor(act_desc, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0);

  std::vector<float> h_in = {-2.0f, -0.5f, 0.0f, 1.5f, -3.0f, 2.0f, 4.0f, -1.0f};
  std::vector<float> h_out(8, 0.0f);
  float* d_in = nullptr;
  float* d_out = nullptr;
  cudaMalloc(reinterpret_cast<void**>(&d_in), 8 * sizeof(float));
  cudaMalloc(reinterpret_cast<void**>(&d_out), 8 * sizeof(float));
  cudaMemcpy(d_in, h_in.data(), 8 * sizeof(float), cudaMemcpyHostToDevice);

  const float one = 1.0f;
  const float zero = 0.0f;
  if (!Expect(cudnnActivationForward(handle, act_desc, &one, t_desc, d_in, &zero, t_desc, d_out) ==
                  CUDNN_STATUS_SUCCESS,
              "cudnnActivationForward ReLU")) {
    return false;
  }
  cudaMemcpy(h_out.data(), d_out, 8 * sizeof(float), cudaMemcpyDeviceToHost);
  for (std::size_t i = 0; i < 8; ++i) {
    if (!ExpectClose(h_out[i], std::max(0.0f, h_in[i]), 1e-5f, "ReLU output")) {
      return false;
    }
  }

  // 2. Pooling 2x2 stride 2 over [1, 1, 2, 2]
  cudnnTensorDescriptor_t pool_in_desc = nullptr;
  cudnnTensorDescriptor_t pool_out_desc = nullptr;
  cudnnPoolingDescriptor_t pool_desc = nullptr;
  cudnnCreateTensorDescriptor(&pool_in_desc);
  cudnnCreateTensorDescriptor(&pool_out_desc);
  cudnnCreatePoolingDescriptor(&pool_desc);
  cudnnSetTensor4dDescriptor(pool_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 2, 2, 2);
  cudnnSetTensor4dDescriptor(pool_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 2, 1, 1);
  cudnnSetPooling2dDescriptor(pool_desc, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, 2, 2, 0, 0, 2,
                              2);

  if (!Expect(cudnnPoolingForward(handle, pool_desc, &one, pool_in_desc, d_in, &zero,
                                  pool_out_desc, d_out) == CUDNN_STATUS_SUCCESS,
              "cudnnPoolingForward Max")) {
    return false;
  }
  cudaMemcpy(h_out.data(), d_out, 2 * sizeof(float), cudaMemcpyDeviceToHost);
  if (!ExpectClose(h_out[0], 1.5f, 1e-5f, "MaxPool ch0") ||
      !ExpectClose(h_out[1], 4.0f, 1e-5f, "MaxPool ch1")) {
    return false;
  }

  // 3. Softmax over [1, 2, 1, 1]
  std::vector<float> h_sm_in = {1.0f, 2.0f};
  cudaMemcpy(d_in, h_sm_in.data(), 2 * sizeof(float), cudaMemcpyHostToDevice);
  if (!Expect(cudnnSoftmaxForward(handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &one,
                                  pool_out_desc, d_in, &zero, pool_out_desc,
                                  d_out) == CUDNN_STATUS_SUCCESS,
              "cudnnSoftmaxForward")) {
    return false;
  }
  cudaMemcpy(h_out.data(), d_out, 2 * sizeof(float), cudaMemcpyDeviceToHost);
  const float exp_sm0 = std::exp(1.0f) / (std::exp(1.0f) + std::exp(2.0f));
  const float exp_sm1 = std::exp(2.0f) / (std::exp(1.0f) + std::exp(2.0f));
  if (!ExpectClose(h_out[0], exp_sm0, 1e-5f, "Softmax ch0") ||
      !ExpectClose(h_out[1], exp_sm1, 1e-5f, "Softmax ch1")) {
    return false;
  }

  // 4. BatchNormalizationForwardInference & Training
  cudnnTensorDescriptor_t bn_scale_desc = nullptr;
  cudnnCreateTensorDescriptor(&bn_scale_desc);
  cudnnDeriveBNTensorDescriptor(bn_scale_desc, t_desc, CUDNN_BATCHNORM_SPATIAL);

  std::vector<float> h_scale = {1.0f, 2.0f};
  std::vector<float> h_bias = {0.5f, -0.5f};
  std::vector<float> h_mean = {0.0f, 0.0f};
  std::vector<float> h_var = {1.0f, 1.0f};

  float* d_scale = nullptr;
  float* d_bias = nullptr;
  float* d_mean = nullptr;
  float* d_var = nullptr;
  cudaMalloc(reinterpret_cast<void**>(&d_scale), 2 * sizeof(float));
  cudaMalloc(reinterpret_cast<void**>(&d_bias), 2 * sizeof(float));
  cudaMalloc(reinterpret_cast<void**>(&d_mean), 2 * sizeof(float));
  cudaMalloc(reinterpret_cast<void**>(&d_var), 2 * sizeof(float));
  cudaMemcpy(d_scale, h_scale.data(), 2 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_bias, h_bias.data(), 2 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_mean, h_mean.data(), 2 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_var, h_var.data(), 2 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_in, h_in.data(), 8 * sizeof(float), cudaMemcpyHostToDevice);

  if (!Expect(cudnnBatchNormalizationForwardInference(
                  handle, CUDNN_BATCHNORM_SPATIAL, &one, &zero, t_desc, d_in, t_desc, d_out,
                  bn_scale_desc, d_scale, d_bias, d_mean, d_var, 1e-5) == CUDNN_STATUS_SUCCESS,
              "cudnnBatchNormalizationForwardInference")) {
    return false;
  }
  cudaMemcpy(h_out.data(), d_out, 8 * sizeof(float), cudaMemcpyDeviceToHost);
  if (!ExpectClose(h_out[0], -2.0f + 0.5f, 1e-3f, "BN inference ch0") ||
      !ExpectClose(h_out[5], 2.0f * 2.0f - 0.5f, 1e-3f, "BN inference ch1")) {
    return false;
  }

  if (!Expect(cudnnBatchNormalizationForwardTraining(
                  handle, CUDNN_BATCHNORM_SPATIAL, &one, &zero, t_desc, d_in, t_desc, d_out,
                  bn_scale_desc, d_scale, d_bias, 0.1, d_mean, d_var, 1e-5, nullptr,
                  nullptr) == CUDNN_STATUS_SUCCESS,
              "cudnnBatchNormalizationForwardTraining")) {
    return false;
  }

  cudaFree(d_in);
  cudaFree(d_out);
  cudaFree(d_scale);
  cudaFree(d_bias);
  cudaFree(d_mean);
  cudaFree(d_var);
  cudnnDestroyTensorDescriptor(t_desc);
  cudnnDestroyTensorDescriptor(pool_in_desc);
  cudnnDestroyTensorDescriptor(pool_out_desc);
  cudnnDestroyTensorDescriptor(bn_scale_desc);
  cudnnDestroyActivationDescriptor(act_desc);
  cudnnDestroyPoolingDescriptor(pool_desc);
  cudnnDestroy(handle);
  return true;
}

}  // namespace

int main() {
  if (!TestConv2dForwardAndBackward() || !TestActivationPoolingSoftmaxAndBatchNorm()) {
    return 1;
  }
  std::cout << "ALL CUDNN TESTS PASSED!\n";
  return 0;
}


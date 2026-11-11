/**
 * @file library_benchmarks.cpp
 * @brief Implementation of MUDA library and subsystem benchmarks (cuBLAS, cuFFT, cuDNN,
 *        cuSPARSE, NVRTC, Compiler, and BufferPool).
 */

#include "library_benchmarks.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "cudnn.h"
#include "cufft.h"
#include "curand.h"
#include "cusparse.h"
#include "muda/compiler/translator.hpp"
#include "nvrtc.h"

namespace muda::bench {
namespace {

__half FloatToHalf(float value) noexcept {
  const __fp16 h = static_cast<__fp16>(value);
  __half out{};
  std::memcpy(&out.x, &h, sizeof(out.x));
  return out;
}

}  // namespace

LibraryBenchmarkSuiteResults RunLibraryBenchmarks(const BenchmarkConfig& base_config) {
  LibraryBenchmarkSuiteResults out{};

  auto record = [&](std::string_view suite, std::string_view impl, std::string_view domain,
                    std::string_view unit, const BenchmarkConfig& cfg,
                    const BenchmarkStatistics& stats) {
    PrintBenchmarkReport(suite, impl, cfg, unit, stats, domain);
    out.records.push_back(BenchmarkResultRecord{
        .suite = std::string(suite),
        .implementation = std::string(impl),
        .timing_domain = std::string(domain),
        .metric_unit = std::string(unit),
        .config = cfg,
        .stats = stats,
    });
  };

  // =========================================================================
  // 1. Runtime BufferPool & Async Memset
  // =========================================================================
  {
    BenchmarkConfig pool_cfg = base_config;
    pool_cfg.warmup_iterations = 20;
    pool_cfg.samples = 50;
    pool_cfg.iterations_per_sample = 50;

    constexpr std::size_t kPoolBytes = 4ull * 1024ull * 1024ull;
    void* warm_ptr = nullptr;
    (void)cudaMalloc(&warm_ptr, kPoolBytes);
    (void)cudaFree(warm_ptr);

    const auto pool_sec = MeasureCpuDuration(
        pool_cfg,
        [&]() {
          void* ptr = nullptr;
          (void)cudaMalloc(&ptr, kPoolBytes);
          (void)cudaFree(ptr);
        },
        []() {});
    record("1. Runtime BufferPool & Memory Ops", "BufferPool Reuse (cudaMalloc+cudaFree 4 MiB)",
           "CPU Wall-Clock", "us", pool_cfg, pool_sec.Transform([](double s) { return s * 1e6; }));

    BenchmarkConfig memset_cfg = base_config;
    memset_cfg.warmup_iterations = 10;
    memset_cfg.samples = 40;
    memset_cfg.iterations_per_sample = 15;
    constexpr std::size_t kMemsetBytes = 64ull * 1024ull * 1024ull;
    void* d_memset = nullptr;
    (void)cudaMalloc(&d_memset, kMemsetBytes);
    const auto memset_sec = MeasureGpuDuration(
        memset_cfg, [&]() { (void)cudaMemsetAsync(d_memset, 0x3F, kMemsetBytes, nullptr); });
    const double memset_gb = static_cast<double>(kMemsetBytes) / 1e9;
    record("1. Runtime BufferPool & Memory Ops", "cudaMemsetAsync (64 MiB)", "GPU Execution",
           "GB/s", memset_cfg, memset_sec.Transform([=](double s) { return memset_gb / s; }));
    (void)cudaFree(d_memset);
  }

  // =========================================================================
  // 2. cuBLAS Level-1, Level-2, Level-3 (FP16 & Strided Batched)
  // =========================================================================
  {
    cublasHandle_t blas = nullptr;
    (void)cublasCreate(&blas);

    BenchmarkConfig l1_cfg = base_config;
    l1_cfg.warmup_iterations = 10;
    l1_cfg.samples = 40;
    l1_cfg.iterations_per_sample = 15;

    constexpr int kVecN = 16 * 1024 * 1024;  // 16M floats (64 MiB per vector)
    float* d_x = nullptr;
    float* d_y = nullptr;
    float* d_dot = nullptr;
    (void)cudaMalloc(reinterpret_cast<void**>(&d_x),
                     static_cast<std::size_t>(kVecN) * sizeof(float));
    (void)cudaMalloc(reinterpret_cast<void**>(&d_y),
                     static_cast<std::size_t>(kVecN) * sizeof(float));
    (void)cudaMalloc(reinterpret_cast<void**>(&d_dot), sizeof(float));

    const float alpha = 2.0f;
    const float beta = 1.0f;

    // SAXPY: reads X (64 MiB) + reads Y (64 MiB) + writes Y (64 MiB) = 192 MiB
    const double saxpy_gb = (3.0 * static_cast<double>(kVecN) * sizeof(float)) / 1e9;
    const auto saxpy_sec = MeasureGpuDuration(
        l1_cfg, [&]() { (void)cublasSaxpy(blas, kVecN, &alpha, d_x, 1, d_y, 1); });
    record("2. cuBLAS Suite (Level 1/2/3)", "cublasSaxpy (N=16M, 192 MiB, float4)", "GPU Execution",
           "GB/s", l1_cfg, saxpy_sec.Transform([=](double s) { return saxpy_gb / s; }));

    // SCOPY: reads X (64 MiB) + writes Y (64 MiB) = 128 MiB
    const double scopy_gb = (2.0 * static_cast<double>(kVecN) * sizeof(float)) / 1e9;
    const auto scopy_sec =
        MeasureGpuDuration(l1_cfg, [&]() { (void)cublasScopy(blas, kVecN, d_x, 1, d_y, 1); });
    record("2. cuBLAS Suite (Level 1/2/3)", "cublasScopy (N=16M, 128 MiB, float4)", "GPU Execution",
           "GB/s", l1_cfg, scopy_sec.Transform([=](double s) { return scopy_gb / s; }));

    // SSCAL: reads X (64 MiB) + writes X (64 MiB) = 128 MiB
    const double sscal_gb = (2.0 * static_cast<double>(kVecN) * sizeof(float)) / 1e9;
    const auto sscal_sec =
        MeasureGpuDuration(l1_cfg, [&]() { (void)cublasSscal(blas, kVecN, &alpha, d_x, 1); });
    record("2. cuBLAS Suite (Level 1/2/3)", "cublasSscal (N=16M, 128 MiB, float4)", "GPU Execution",
           "GB/s", l1_cfg, sscal_sec.Transform([=](double s) { return sscal_gb / s; }));

    // SDOT (Device pointer mode): reads X (64 MiB) + reads Y (64 MiB) = 128 MiB
    (void)cublasSetPointerMode(blas, CUBLAS_POINTER_MODE_DEVICE);
    const double sdot_gb = (2.0 * static_cast<double>(kVecN) * sizeof(float)) / 1e9;
    const auto sdot_sec =
        MeasureGpuDuration(l1_cfg, [&]() { (void)cublasSdot(blas, kVecN, d_x, 1, d_y, 1, d_dot); });
    record("2. cuBLAS Suite (Level 1/2/3)", "cublasSdot (N=16M, 128 MiB, 2-Stage Warp)",
           "GPU Execution", "GB/s", l1_cfg,
           sdot_sec.Transform([=](double s) { return sdot_gb / s; }));
    (void)cublasSetPointerMode(blas, CUBLAS_POINTER_MODE_HOST);

    // SGEMV (4096 x 4096 = 64 MiB matrix)
    constexpr int kGemvDim = 4096;
    const double sgemv_gb =
        (static_cast<double>(kGemvDim) * static_cast<double>(kGemvDim) * sizeof(float)) / 1e9;
    const auto sgemv_n_sec = MeasureGpuDuration(l1_cfg, [&]() {
      (void)cublasSgemv(blas, CUBLAS_OP_N, kGemvDim, kGemvDim, &alpha, d_x, kGemvDim, d_y, 1, &beta,
                        d_y + kGemvDim, 1);
    });
    record("2. cuBLAS Suite (Level 1/2/3)", "cublasSgemv (OP_N, 4096x4096, 64 MiB)",
           "GPU Execution", "GB/s", l1_cfg,
           sgemv_n_sec.Transform([=](double s) { return sgemv_gb / s; }));

    const auto sgemv_t_sec = MeasureGpuDuration(l1_cfg, [&]() {
      (void)cublasSgemv(blas, CUBLAS_OP_T, kGemvDim, kGemvDim, &alpha, d_x, kGemvDim, d_y, 1, &beta,
                        d_y + kGemvDim, 1);
    });
    record("2. cuBLAS Suite (Level 1/2/3)", "cublasSgemv (OP_T, 4096x4096, 64 MiB)",
           "GPU Execution", "GB/s", l1_cfg,
           sgemv_t_sec.Transform([=](double s) { return sgemv_gb / s; }));

    (void)cudaFree(d_dot);
    (void)cudaFree(d_y);
    (void)cudaFree(d_x);

    // HGEMM FP16 (1024x1024 and 2048x2048)
    BenchmarkConfig hgemm_cfg = base_config;
    hgemm_cfg.warmup_iterations = 8;
    hgemm_cfg.samples = 30;
    hgemm_cfg.iterations_per_sample = 10;

    for (const int n_dim : {1024, 2048}) {
      const std::size_t elems = static_cast<std::size_t>(n_dim) * static_cast<std::size_t>(n_dim);
      __half* d_ha = nullptr;
      __half* d_hb = nullptr;
      __half* d_hc = nullptr;
      (void)cudaMalloc(reinterpret_cast<void**>(&d_ha), elems * sizeof(__half));
      (void)cudaMalloc(reinterpret_cast<void**>(&d_hb), elems * sizeof(__half));
      (void)cudaMalloc(reinterpret_cast<void**>(&d_hc), elems * sizeof(__half));
      const __half h_alpha = FloatToHalf(1.0f);
      const __half h_beta = FloatToHalf(0.0f);
      const double gflops = (2.0 * static_cast<double>(n_dim) * static_cast<double>(n_dim) *
                             static_cast<double>(n_dim)) /
                            1e9;
      const auto hgemm_sec = MeasureGpuDuration(hgemm_cfg, [&]() {
        (void)cublasHgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, n_dim, n_dim, n_dim, &h_alpha, d_ha,
                          n_dim, d_hb, n_dim, &h_beta, d_hc, n_dim);
      });
      const std::string label =
          "cublasHgemm (FP16, " + std::to_string(n_dim) + "x" + std::to_string(n_dim) + ")";
      record("2. cuBLAS Suite (Level 1/2/3)", label, "GPU Execution", "GFLOP/s", hgemm_cfg,
             hgemm_sec.Transform([=](double s) { return gflops / s; }));
      (void)cudaFree(d_hc);
      (void)cudaFree(d_hb);
      (void)cudaFree(d_ha);
    }

    // StridedBatched SGEMM (Batch=16, 512x512)
    {
      constexpr int kBatch = 16;
      constexpr int kDim = 512;
      constexpr long long int kStride = kDim * kDim;
      const std::size_t total_elems =
          static_cast<std::size_t>(kBatch) * static_cast<std::size_t>(kStride);
      float* d_ba = nullptr;
      float* d_bb = nullptr;
      float* d_bc = nullptr;
      (void)cudaMalloc(reinterpret_cast<void**>(&d_ba), total_elems * sizeof(float));
      (void)cudaMalloc(reinterpret_cast<void**>(&d_bb), total_elems * sizeof(float));
      (void)cudaMalloc(reinterpret_cast<void**>(&d_bc), total_elems * sizeof(float));
      const float f_alpha = 1.0f;
      const float f_beta = 0.0f;
      const double batched_gflops = (2.0 * static_cast<double>(kBatch) * static_cast<double>(kDim) *
                                     static_cast<double>(kDim) * static_cast<double>(kDim)) /
                                    1e9;
      const auto b_sec = MeasureGpuDuration(hgemm_cfg, [&]() {
        (void)cublasSgemmStridedBatched(blas, CUBLAS_OP_N, CUBLAS_OP_N, kDim, kDim, kDim, &f_alpha,
                                        d_ba, kDim, kStride, d_bb, kDim, kStride, &f_beta, d_bc,
                                        kDim, kStride, kBatch);
      });
      record("2. cuBLAS Suite (Level 1/2/3)",
             "cublasSgemmStridedBatched (FP32, B=16, 512x512, simdgroup)", "GPU Execution",
             "GFLOP/s", hgemm_cfg, b_sec.Transform([=](double s) { return batched_gflops / s; }));
      (void)cudaFree(d_bc);
      (void)cudaFree(d_bb);
      (void)cudaFree(d_ba);
    }

    (void)cublasDestroy(blas);
  }

  // =========================================================================
  // 3. cuFFT Suite (1D Batched C2C, R2C, C2R)
  // =========================================================================
  {
    BenchmarkConfig fft_cfg = base_config;
    fft_cfg.warmup_iterations = 8;
    fft_cfg.samples = 30;
    fft_cfg.iterations_per_sample = 15;

    constexpr int kFftN = 1024;
    constexpr int kFftBatch = 256;
    const std::size_t total_c2c =
        static_cast<std::size_t>(kFftN) * static_cast<std::size_t>(kFftBatch);
    cufftComplex* d_in = nullptr;
    cufftComplex* d_out = nullptr;
    cufftReal* d_real = nullptr;
    (void)cudaMalloc(reinterpret_cast<void**>(&d_in), total_c2c * sizeof(cufftComplex));
    (void)cudaMalloc(reinterpret_cast<void**>(&d_out), total_c2c * sizeof(cufftComplex));
    (void)cudaMalloc(reinterpret_cast<void**>(&d_real), total_c2c * sizeof(cufftReal));

    // Standard Cooley-Tukey FFT FLOP estimate: 5 * N * log2(N) * batch
    const double c2c_gflops =
        (5.0 * static_cast<double>(kFftN) * std::log2(static_cast<double>(kFftN)) *
         static_cast<double>(kFftBatch)) /
        1e9;
    const double r2c_gflops = 0.5 * c2c_gflops;

    cufftHandle plan_c2c = 0;
    int dims[1] = {kFftN};
    (void)cufftPlanMany(&plan_c2c, 1, dims, nullptr, 1, kFftN, nullptr, 1, kFftN, CUFFT_C2C,
                        kFftBatch);
    const auto c2c_sec = MeasureGpuDuration(
        fft_cfg, [&]() { (void)cufftExecC2C(plan_c2c, d_in, d_out, CUFFT_FORWARD); });
    record("3. cuFFT Spectral Suite", "cufftExecC2C (1D N=1024, Batch=256, Fused Shared)",
           "GPU Execution", "GFLOP/s", fft_cfg,
           c2c_sec.Transform([=](double s) { return c2c_gflops / s; }));
    (void)cufftDestroy(plan_c2c);

    cufftHandle plan_r2c = 0;
    (void)cufftPlanMany(&plan_r2c, 1, dims, nullptr, 1, kFftN, nullptr, 1, (kFftN / 2) + 1,
                        CUFFT_R2C, kFftBatch);
    const auto r2c_sec =
        MeasureGpuDuration(fft_cfg, [&]() { (void)cufftExecR2C(plan_r2c, d_real, d_out); });
    record("3. cuFFT Spectral Suite", "cufftExecR2C (1D N=1024, Batch=256)", "GPU Execution",
           "GFLOP/s", fft_cfg, r2c_sec.Transform([=](double s) { return r2c_gflops / s; }));
    (void)cufftDestroy(plan_r2c);

    (void)cudaFree(d_real);
    (void)cudaFree(d_out);
    (void)cudaFree(d_in);
  }

  // =========================================================================
  // 4. cuDNN Suite (Activation, BatchNorm, Softmax, Pooling, Conv2D)
  // =========================================================================
  {
    cudnnHandle_t dnn = nullptr;
    (void)cudnnCreate(&dnn);

    BenchmarkConfig dnn_cfg = base_config;
    dnn_cfg.warmup_iterations = 8;
    dnn_cfg.samples = 30;
    dnn_cfg.iterations_per_sample = 15;

    // Elementwise / Spatial tensors: N=8, C=64, H=128, W=128 (8M floats = 32 MiB per tensor)
    constexpr int kN = 8;
    constexpr int kC = 64;
    constexpr int kH = 128;
    constexpr int kW = 128;
    const std::size_t elems = static_cast<std::size_t>(kN) * kC * kH * kW;
    float* d_x = nullptr;
    float* d_y = nullptr;
    float* d_bn_param = nullptr;
    (void)cudaMalloc(reinterpret_cast<void**>(&d_x), elems * sizeof(float));
    (void)cudaMalloc(reinterpret_cast<void**>(&d_y), elems * sizeof(float));
    (void)cudaMalloc(reinterpret_cast<void**>(&d_bn_param),
                     static_cast<std::size_t>(kC) * 4u * sizeof(float));

    cudnnTensorDescriptor_t x_desc = nullptr;
    cudnnTensorDescriptor_t y_desc = nullptr;
    cudnnTensorDescriptor_t bn_desc = nullptr;
    (void)cudnnCreateTensorDescriptor(&x_desc);
    (void)cudnnCreateTensorDescriptor(&y_desc);
    (void)cudnnCreateTensorDescriptor(&bn_desc);
    (void)cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, kN, kC, kH, kW);
    (void)cudnnSetTensor4dDescriptor(y_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, kN, kC, kH, kW);
    (void)cudnnSetTensor4dDescriptor(bn_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, kC, 1, 1);

    const float alpha = 1.0f;
    const float beta = 0.0f;
    const double rw_gb = (2.0 * static_cast<double>(elems) * sizeof(float)) / 1e9;

    // Activation Forward (ReLU, float4)
    cudnnActivationDescriptor_t act_desc = nullptr;
    (void)cudnnCreateActivationDescriptor(&act_desc);
    (void)cudnnSetActivationDescriptor(act_desc, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN,
                                       0.0);
    const auto act_sec = MeasureGpuDuration(dnn_cfg, [&]() {
      (void)cudnnActivationForward(dnn, act_desc, &alpha, x_desc, d_x, &beta, y_desc, d_y);
    });
    record("4. cuDNN Deep Learning Suite", "cudnnActivationForward (ReLU, 8x64x128x128, float4)",
           "GPU Execution", "GB/s", dnn_cfg,
           act_sec.Transform([=](double s) { return rw_gb / s; }));
    (void)cudnnDestroyActivationDescriptor(act_desc);

    // BatchNorm Inference
    const auto bn_sec = MeasureGpuDuration(dnn_cfg, [&]() {
      (void)cudnnBatchNormalizationForwardInference(
          dnn, CUDNN_BATCHNORM_SPATIAL, &alpha, &beta, x_desc, d_x, y_desc, d_y, bn_desc,
          d_bn_param, d_bn_param + kC, d_bn_param + 2 * kC, d_bn_param + 3 * kC, 1e-5);
    });
    record("4. cuDNN Deep Learning Suite", "cudnnBatchNormalizationForwardInference (8x64x128x128)",
           "GPU Execution", "GB/s", dnn_cfg, bn_sec.Transform([=](double s) { return rw_gb / s; }));

    // Softmax Forward (Warp-reduced contiguous slices: 4096 x 1024 = 4M floats = 16 MiB each)
    cudnnTensorDescriptor_t sm_desc = nullptr;
    (void)cudnnCreateTensorDescriptor(&sm_desc);
    (void)cudnnSetTensor4dDescriptor(sm_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 4096, 1024, 1,
                                     1);
    const double sm_gb = (2.0 * 4096.0 * 1024.0 * sizeof(float)) / 1e9;
    const auto sm_sec = MeasureGpuDuration(dnn_cfg, [&]() {
      (void)cudnnSoftmaxForward(dnn, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE, &alpha,
                                sm_desc, d_x, &beta, sm_desc, d_y);
    });
    record("4. cuDNN Deep Learning Suite",
           "cudnnSoftmaxForward (4096x1024, Warp simd_max/simd_sum)", "GPU Execution", "GB/s",
           dnn_cfg, sm_sec.Transform([=](double s) { return sm_gb / s; }));
    (void)cudnnDestroyTensorDescriptor(sm_desc);

    // Conv2D Forward (Implicit GEMM simdgroup_float8x8: N=4, Cin=64, Cout=128, H=56, W=56, 3x3)
    {
      constexpr int cN = 4;
      constexpr int cIn = 64;
      constexpr int cOut = 128;
      constexpr int cH = 56;
      constexpr int cW = 56;
      constexpr int cK = 3;
      cudnnTensorDescriptor_t cx_desc = nullptr;
      cudnnTensorDescriptor_t cy_desc = nullptr;
      cudnnFilterDescriptor_t cw_desc = nullptr;
      cudnnConvolutionDescriptor_t conv_desc = nullptr;
      (void)cudnnCreateTensorDescriptor(&cx_desc);
      (void)cudnnCreateTensorDescriptor(&cy_desc);
      (void)cudnnCreateFilterDescriptor(&cw_desc);
      (void)cudnnCreateConvolutionDescriptor(&conv_desc);

      (void)cudnnSetTensor4dDescriptor(cx_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, cN, cIn, cH,
                                       cW);
      (void)cudnnSetFilter4dDescriptor(cw_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, cOut, cIn, cK,
                                       cK);
      (void)cudnnSetConvolution2dDescriptor(conv_desc, 1, 1, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION,
                                            CUDNN_DATA_FLOAT);
      (void)cudnnSetTensor4dDescriptor(cy_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, cN, cOut, cH,
                                       cW);

      float* d_w = nullptr;
      (void)cudaMalloc(reinterpret_cast<void**>(&d_w),
                       static_cast<std::size_t>(cOut) * cIn * cK * cK * sizeof(float));

      const double conv_gflops =
          (2.0 * static_cast<double>(cN) * cOut * cH * cW * cIn * cK * cK) / 1e9;
      const auto conv_sec = MeasureGpuDuration(dnn_cfg, [&]() {
        (void)cudnnConvolutionForward(dnn, &alpha, cx_desc, d_x, cw_desc, d_w, conv_desc,
                                      CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, nullptr, 0, &beta,
                                      cy_desc, d_y);
      });
      record("4. cuDNN Deep Learning Suite",
             "cudnnConvolutionForward (N=4, 64->128, 56x56, 3x3, Implicit GEMM)", "GPU Execution",
             "GFLOP/s", dnn_cfg, conv_sec.Transform([=](double s) { return conv_gflops / s; }));

      (void)cudaFree(d_w);
      (void)cudnnDestroyConvolutionDescriptor(conv_desc);
      (void)cudnnDestroyFilterDescriptor(cw_desc);
      (void)cudnnDestroyTensorDescriptor(cy_desc);
      (void)cudnnDestroyTensorDescriptor(cx_desc);
    }

    (void)cudnnDestroyTensorDescriptor(bn_desc);
    (void)cudnnDestroyTensorDescriptor(y_desc);
    (void)cudnnDestroyTensorDescriptor(x_desc);
    (void)cudaFree(d_bn_param);
    (void)cudaFree(d_y);
    (void)cudaFree(d_x);
    (void)cudnnDestroy(dnn);
  }

  // =========================================================================
  // 5. cuSPARSE Suite (CSR SpMV & SpMM)
  // =========================================================================
  {
    cusparseHandle_t sp = nullptr;
    (void)cusparseCreate(&sp);

    BenchmarkConfig sp_cfg = base_config;
    sp_cfg.warmup_iterations = 8;
    sp_cfg.samples = 30;
    sp_cfg.iterations_per_sample = 15;

    constexpr int kRows = 16384;
    constexpr int kCols = 16384;
    constexpr int kNnzPerRow = 64;
    constexpr int kNnz = kRows * kNnzPerRow;

    std::vector<int> h_row_offsets(static_cast<std::size_t>(kRows + 1));
    std::vector<int> h_col_indices(static_cast<std::size_t>(kNnz));
    std::vector<float> h_values(static_cast<std::size_t>(kNnz), 1.0f);
    for (int r = 0; r <= kRows; ++r) {
      h_row_offsets[static_cast<std::size_t>(r)] = r * kNnzPerRow;
    }
    for (int r = 0; r < kRows; ++r) {
      for (int j = 0; j < kNnzPerRow; ++j) {
        h_col_indices[static_cast<std::size_t>(r * kNnzPerRow + j)] = (r + j) % kCols;
      }
    }

    int* d_row = nullptr;
    int* d_col = nullptr;
    float* d_val = nullptr;
    float* d_x = nullptr;
    float* d_y = nullptr;
    (void)cudaMalloc(reinterpret_cast<void**>(&d_row), h_row_offsets.size() * sizeof(int));
    (void)cudaMalloc(reinterpret_cast<void**>(&d_col), h_col_indices.size() * sizeof(int));
    (void)cudaMalloc(reinterpret_cast<void**>(&d_val), h_values.size() * sizeof(float));
    (void)cudaMalloc(reinterpret_cast<void**>(&d_x),
                     static_cast<std::size_t>(kCols) * sizeof(float));
    (void)cudaMalloc(reinterpret_cast<void**>(&d_y),
                     static_cast<std::size_t>(kRows) * sizeof(float));
    (void)cudaMemcpy(d_row, h_row_offsets.data(), h_row_offsets.size() * sizeof(int),
                     cudaMemcpyHostToDevice);
    (void)cudaMemcpy(d_col, h_col_indices.data(), h_col_indices.size() * sizeof(int),
                     cudaMemcpyHostToDevice);
    (void)cudaMemcpy(d_val, h_values.data(), h_values.size() * sizeof(float),
                     cudaMemcpyHostToDevice);

    cusparseSpMatDescr_t mat_a = nullptr;
    cusparseDnVecDescr_t vec_x = nullptr;
    cusparseDnVecDescr_t vec_y = nullptr;
    (void)cusparseCreateCsr(&mat_a, kRows, kCols, kNnz, d_row, d_col, d_val, CUSPARSE_INDEX_32I,
                            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    (void)cusparseCreateDnVec(&vec_x, kCols, d_x, CUDA_R_32F);
    (void)cusparseCreateDnVec(&vec_y, kRows, d_y, CUDA_R_32F);

    const float alpha = 1.0f;
    const float beta = 0.0f;
    const double spmv_bytes = static_cast<double>(kNnz) * (sizeof(int) + sizeof(float)) +
                              static_cast<double>(kRows + kCols) * sizeof(float);
    const double spmv_gb = spmv_bytes / 1e9;

    const auto spmv_sec = MeasureGpuDuration(sp_cfg, [&]() {
      (void)cusparseSpMV(sp, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat_a, vec_x, &beta, vec_y,
                         CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, nullptr);
    });
    record("5. cuSPARSE Suite (SpMV & SpMM)",
           "cusparseSpMV (CSR 16384x16384, 64 nnz/row, Warp-Reduced)", "GPU Execution", "GB/s",
           sp_cfg, spmv_sec.Transform([=](double s) { return spmv_gb / s; }));

    // SpMM (CSR 4096x4096, 64 nnz/row, Dense K=128)
    constexpr int kSpmmRows = 4096;
    constexpr int kSpmmDenseCols = 128;
    constexpr int kSpmmNnz = kSpmmRows * kNnzPerRow;
    float* d_b = nullptr;
    float* d_c = nullptr;
    (void)cudaMalloc(reinterpret_cast<void**>(&d_b),
                     static_cast<std::size_t>(kCols) * kSpmmDenseCols * sizeof(float));
    (void)cudaMalloc(reinterpret_cast<void**>(&d_c),
                     static_cast<std::size_t>(kSpmmRows) * kSpmmDenseCols * sizeof(float));

    cusparseSpMatDescr_t mat_a_spmm = nullptr;
    cusparseDnMatDescr_t mat_b = nullptr;
    cusparseDnMatDescr_t mat_c = nullptr;
    (void)cusparseCreateCsr(&mat_a_spmm, kSpmmRows, kCols, kSpmmNnz, d_row, d_col, d_val,
                            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                            CUDA_R_32F);
    (void)cusparseCreateDnMat(&mat_b, kCols, kSpmmDenseCols, kSpmmDenseCols, d_b, CUDA_R_32F,
                              CUSPARSE_ORDER_ROW);
    (void)cusparseCreateDnMat(&mat_c, kSpmmRows, kSpmmDenseCols, kSpmmDenseCols, d_c, CUDA_R_32F,
                              CUSPARSE_ORDER_ROW);

    const double spmm_gflops =
        (2.0 * static_cast<double>(kSpmmNnz) * static_cast<double>(kSpmmDenseCols)) / 1e9;
    const auto spmm_sec = MeasureGpuDuration(sp_cfg, [&]() {
      (void)cusparseSpMM(sp, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                         &alpha, mat_a_spmm, mat_b, &beta, mat_c, CUDA_R_32F,
                         CUSPARSE_SPMM_ALG_DEFAULT, nullptr);
    });
    record("5. cuSPARSE Suite (SpMV & SpMM)",
           "cusparseSpMM (CSR 4096x16384 * Dense 16384x128, Coalesced)", "GPU Execution", "GFLOP/s",
           sp_cfg, spmm_sec.Transform([=](double s) { return spmm_gflops / s; }));

    (void)cusparseDestroyDnMat(mat_c);
    (void)cusparseDestroyDnMat(mat_b);
    (void)cusparseDestroySpMat(mat_a_spmm);
    (void)cusparseDestroyDnVec(vec_y);
    (void)cusparseDestroyDnVec(vec_x);
    (void)cusparseDestroySpMat(mat_a);
    (void)cudaFree(d_c);
    (void)cudaFree(d_b);
    (void)cudaFree(d_y);
    (void)cudaFree(d_x);
    (void)cudaFree(d_val);
    (void)cudaFree(d_col);
    (void)cudaFree(d_row);
    (void)cusparseDestroy(sp);
  }

  // =========================================================================
  // 6. Compiler & NVRTC JIT Suite
  // =========================================================================
  {
    BenchmarkConfig jit_cfg = base_config;
    jit_cfg.warmup_iterations = 10;
    jit_cfg.samples = 40;
    jit_cfg.iterations_per_sample = 20;

    constexpr std::string_view kSampleKernel = R"CUDA(
__global__ void saxpy_jit(float a, const float* x, float* y, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) {
    y[tid] = fmaf(a, x[tid], y[tid]);
  }
}
)CUDA";

    const auto trans_sec = MeasureCpuDuration(
        jit_cfg,
        [&]() {
          const auto res = muda::compiler::TranslateCudaToMsl(kSampleKernel);
          (void)res;
        },
        []() {});
    record("6. Compiler & NVRTC JIT Suite", "TranslateCudaToMsl (AST -> MSL Source)",
           "CPU Wall-Clock", "us", jit_cfg, trans_sec.Transform([](double s) { return s * 1e6; }));

    const auto nvrtc_sec = MeasureCpuDuration(
        jit_cfg,
        [&]() {
          nvrtcProgram prog = nullptr;
          (void)nvrtcCreateProgram(&prog, kSampleKernel.data(), "saxpy_jit.cu", 0, nullptr,
                                   nullptr);
          (void)nvrtcCompileProgram(prog, 0, nullptr);
          (void)nvrtcDestroyProgram(&prog);
        },
        []() {});
    record("6. Compiler & NVRTC JIT Suite", "nvrtcCompileProgram (Full NVRTC JIT Pipeline)",
           "CPU Wall-Clock", "us", jit_cfg, nvrtc_sec.Transform([](double s) { return s * 1e6; }));
  }

  // =========================================================================
  // 7. cuRAND Random Number Generation Suite
  // =========================================================================
  {
    constexpr std::size_t kRandN = 16 * 1024 * 1024;  // 16M floats (64 MiB)
    float* d_rand = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&d_rand), kRandN * sizeof(float)) == cudaSuccess) {
      curandGenerator_t gen = nullptr;
      if (curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) == CURAND_STATUS_SUCCESS) {
        (void)curandSetPseudoRandomGeneratorSeed(gen, 12345ull);

        BenchmarkConfig rand_cfg = base_config;
        rand_cfg.warmup_iterations = 5;
        rand_cfg.samples = 20;
        rand_cfg.iterations_per_sample = 5;

        const auto uniform_sec = MeasureGpuDuration(
            rand_cfg, [&]() { (void)curandGenerateUniform(gen, d_rand, kRandN); });
        const double bytes = static_cast<double>(kRandN * sizeof(float));
        record("7. cuRAND Random Number Suite", "curandGenerateUniform (16M floats, Philox4x32)",
               "GPU Kernel Execution", "GB/s", rand_cfg, uniform_sec.Transform([bytes](double s) {
                 return (bytes / s) / (1024.0 * 1024.0 * 1024.0);
               }));

        const auto normal_sec = MeasureGpuDuration(
            rand_cfg, [&]() { (void)curandGenerateNormal(gen, d_rand, kRandN, 0.0f, 1.0f); });
        record("7. cuRAND Random Number Suite", "curandGenerateNormal (16M floats, Box-Muller)",
               "GPU Kernel Execution", "GB/s", rand_cfg, normal_sec.Transform([bytes](double s) {
                 return (bytes / s) / (1024.0 * 1024.0 * 1024.0);
               }));

        (void)curandDestroyGenerator(gen);
      }
      (void)cudaFree(d_rand);
    }
  }

  return out;
}

}  // namespace muda::bench

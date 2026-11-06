/**
 * @file cublas_test.cpp
 * @brief Comprehensive verification test suite for the MUDA cuBLAS shim (libcublas.dylib).
 */

#include "cublas_v2.h"

#include <cmath>
#include <cstdint>
#include <cstring>
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

std::uint16_t FloatToHalfBits(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(float));
  const std::uint16_t sign = static_cast<std::uint16_t>((bits >> 16) & 0x8000u);
  const int32_t exp = static_cast<int32_t>((bits >> 23) & 0xFFu) - 127 + 15;
  std::uint32_t mant = bits & 0x7FFFFFu;
  if (exp <= 0) {
    return sign;
  }
  if (exp >= 31) {
    return static_cast<std::uint16_t>(sign | 0x7C00u);
  }
  return static_cast<std::uint16_t>(sign | (static_cast<std::uint16_t>(exp) << 10) |
                                    static_cast<std::uint16_t>(mant >> 13));
}

float HalfBitsToFloat(std::uint16_t h) {
  const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
  const std::uint32_t exp = (h >> 10) & 0x1Fu;
  const std::uint32_t mant = h & 0x03FFu;
  std::uint32_t bits = 0;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign;
    } else {
      float val = static_cast<float>(mant) / 1024.0f * std::ldexp(1.0f, -14);
      return (sign != 0) ? -val : val;
    }
  } else if (exp == 31) {
    bits = sign | 0x7F800000u | (mant << 13);
  } else {
    bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
  }
  float out = 0.0f;
  std::memcpy(&out, &bits, sizeof(float));
  return out;
}

bool TestHandleAndConfig() {
  std::cout << "Testing cuBLAS handle lifecycle, stream, pointer mode, and math mode...\n";
  cublasHandle_t handle = nullptr;
  if (!Expect(cublasCreate(&handle) == CUBLAS_STATUS_SUCCESS, "cublasCreate")) {
    return false;
  }

  int version = 0;
  if (!Expect(cublasGetVersion(handle, &version) == CUBLAS_STATUS_SUCCESS && version > 0,
              "cublasGetVersion")) {
    return false;
  }

  cudaStream_t stream = nullptr;
  if (!Expect(cudaStreamCreate(&stream) == cudaSuccess, "cudaStreamCreate")) {
    return false;
  }
  if (!Expect(cublasSetStream(handle, stream) == CUBLAS_STATUS_SUCCESS, "cublasSetStream")) {
    return false;
  }
  cudaStream_t queried_stream = nullptr;
  if (!Expect(cublasGetStream(handle, &queried_stream) == CUBLAS_STATUS_SUCCESS &&
                  queried_stream == stream,
              "cublasGetStream")) {
    return false;
  }

  if (!Expect(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE) == CUBLAS_STATUS_SUCCESS,
              "cublasSetPointerMode")) {
    return false;
  }
  cublasPointerMode_t mode = CUBLAS_POINTER_MODE_HOST;
  if (!Expect(cublasGetPointerMode(handle, &mode) == CUBLAS_STATUS_SUCCESS &&
                  mode == CUBLAS_POINTER_MODE_DEVICE,
              "cublasGetPointerMode")) {
    return false;
  }

  if (!Expect(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH) == CUBLAS_STATUS_SUCCESS,
              "cublasSetMathMode")) {
    return false;
  }
  cublasMath_t math_mode = CUBLAS_DEFAULT_MATH;
  if (!Expect(cublasGetMathMode(handle, &math_mode) == CUBLAS_STATUS_SUCCESS &&
                  math_mode == CUBLAS_TENSOR_OP_MATH,
              "cublasGetMathMode")) {
    return false;
  }

  cudaStreamDestroy(stream);
  return Expect(cublasDestroy(handle) == CUBLAS_STATUS_SUCCESS, "cublasDestroy");
}

bool TestLevel1Ops() {
  std::cout << "Testing cuBLAS Level-1 vector operations...\n";
  cublasHandle_t handle = nullptr;
  if (!Expect(cublasCreate(&handle) == CUBLAS_STATUS_SUCCESS, "cublasCreate")) {
    return false;
  }

  constexpr int kN = 128;
  std::vector<float> h_x(kN);
  std::vector<float> h_y(kN);
  for (int i = 0; i < kN; ++i) {
    h_x[static_cast<std::size_t>(i)] = static_cast<float>(i + 1) * 0.25f;
    h_y[static_cast<std::size_t>(i)] = static_cast<float>(kN - i) * 0.5f;
  }
  h_x[42] = -99.5f;  // Distinct max magnitude at 0-based index 42 -> 1-based index 43

  float* d_x = nullptr;
  float* d_y = nullptr;
  if (!Expect(cudaMalloc(reinterpret_cast<void**>(&d_x), kN * sizeof(float)) == cudaSuccess,
              "cudaMalloc d_x") ||
      !Expect(cudaMalloc(reinterpret_cast<void**>(&d_y), kN * sizeof(float)) == cudaSuccess,
              "cudaMalloc d_y")) {
    return false;
  }

  cudaMemcpy(d_x, h_x.data(), kN * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, h_y.data(), kN * sizeof(float), cudaMemcpyHostToDevice);

  // Isamax (1-based index)
  int max_idx = 0;
  if (!Expect(cublasIsamax(handle, kN, d_x, 1, &max_idx) == CUBLAS_STATUS_SUCCESS,
              "cublasIsamax")) {
    return false;
  }
  if (!Expect(max_idx == 43, "cublasIsamax 1-based index == 43")) {
    return false;
  }

  // Sasum & Snrm2 & Sdot
  float expected_asum = 0.0f;
  float expected_dot = 0.0f;
  for (int i = 0; i < kN; ++i) {
    expected_asum += std::fabs(h_x[static_cast<std::size_t>(i)]);
    expected_dot += h_x[static_cast<std::size_t>(i)] * h_y[static_cast<std::size_t>(i)];
  }
  float actual_asum = 0.0f;
  float actual_nrm2 = 0.0f;
  float actual_dot = 0.0f;
  if (!Expect(cublasSasum(handle, kN, d_x, 1, &actual_asum) == CUBLAS_STATUS_SUCCESS,
              "cublasSasum") ||
      !ExpectClose(actual_asum, expected_asum, 1e-2f, "cublasSasum value")) {
    return false;
  }
  if (!Expect(cublasSdot(handle, kN, d_x, 1, d_y, 1, &actual_dot) == CUBLAS_STATUS_SUCCESS,
              "cublasSdot") ||
      !ExpectClose(actual_dot, expected_dot, 1e-1f, "cublasSdot value")) {
    return false;
  }

  // Test device pointer mode for Sdot
  float* d_dot = nullptr;
  cudaMalloc(reinterpret_cast<void**>(&d_dot), sizeof(float));
  cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
  if (!Expect(cublasSdot(handle, kN, d_x, 1, d_y, 1, d_dot) == CUBLAS_STATUS_SUCCESS,
              "cublasSdot device pointer mode")) {
    return false;
  }
  float dev_dot_host = 0.0f;
  cudaMemcpy(&dev_dot_host, d_dot, sizeof(float), cudaMemcpyDeviceToHost);
  if (!ExpectClose(dev_dot_host, expected_dot, 1e-1f, "cublasSdot device result")) {
    return false;
  }
  cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
  cudaFree(d_dot);

  // Saxpy: y = 2.0 * x + y
  const float alpha = 2.0f;
  if (!Expect(cublasSaxpy(handle, kN, &alpha, d_x, 1, d_y, 1) == CUBLAS_STATUS_SUCCESS,
              "cublasSaxpy")) {
    return false;
  }
  std::vector<float> h_y_out(kN);
  cudaMemcpy(h_y_out.data(), d_y, kN * sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i < kN; ++i) {
    const float exp =
        alpha * h_x[static_cast<std::size_t>(i)] + h_y[static_cast<std::size_t>(i)];
    if (!ExpectClose(h_y_out[static_cast<std::size_t>(i)], exp, 1e-4f, "Saxpy element")) {
      return false;
    }
  }

  // Sscal: x = 0.5 * x
  const float scale = 0.5f;
  if (!Expect(cublasSscal(handle, kN, &scale, d_x, 1) == CUBLAS_STATUS_SUCCESS, "cublasSscal") ||
      !Expect(cublasSnrm2(handle, kN, d_x, 1, &actual_nrm2) == CUBLAS_STATUS_SUCCESS,
              "cublasSnrm2")) {
    return false;
  }

  // Scopy & Sswap
  if (!Expect(cublasSswap(handle, kN, d_x, 1, d_y, 1) == CUBLAS_STATUS_SUCCESS, "cublasSswap") ||
      !Expect(cublasScopy(handle, kN, d_x, 1, d_y, 1) == CUBLAS_STATUS_SUCCESS, "cublasScopy")) {
    return false;
  }

  cudaFree(d_x);
  cudaFree(d_y);
  cublasDestroy(handle);
  return true;
}

bool TestLevel2Gemv() {
  std::cout << "Testing cuBLAS Level-2 SGEMV...\n";
  cublasHandle_t handle = nullptr;
  cublasCreate(&handle);

  constexpr int kM = 16;
  constexpr int kN = 12;
  std::vector<float> h_a(kM * kN);
  std::vector<float> h_x(kN);
  std::vector<float> h_y(kM, 1.0f);
  for (int c = 0; c < kN; ++c) {
    h_x[static_cast<std::size_t>(c)] = static_cast<float>(c + 1) * 0.5f;
    for (int r = 0; r < kM; ++r) {
      h_a[static_cast<std::size_t>(r + c * kM)] = static_cast<float>((r + 1) * (c + 1)) * 0.1f;
    }
  }

  float* d_a = nullptr;
  float* d_x = nullptr;
  float* d_y = nullptr;
  cudaMalloc(reinterpret_cast<void**>(&d_a), h_a.size() * sizeof(float));
  cudaMalloc(reinterpret_cast<void**>(&d_x), h_x.size() * sizeof(float));
  cudaMalloc(reinterpret_cast<void**>(&d_y), h_y.size() * sizeof(float));
  cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, h_y.data(), h_y.size() * sizeof(float), cudaMemcpyHostToDevice);

  const float alpha = 1.5f;
  const float beta = 0.5f;
  if (!Expect(cublasSgemv(handle, CUBLAS_OP_N, kM, kN, &alpha, d_a, kM, d_x, 1, &beta, d_y, 1) ==
                  CUBLAS_STATUS_SUCCESS,
              "cublasSgemv OP_N")) {
    return false;
  }

  std::vector<float> h_y_out(kM);
  cudaMemcpy(h_y_out.data(), d_y, h_y_out.size() * sizeof(float), cudaMemcpyDeviceToHost);
  for (int r = 0; r < kM; ++r) {
    float sum = 0.0f;
    for (int c = 0; c < kN; ++c) {
      sum += h_a[static_cast<std::size_t>(r + c * kM)] * h_x[static_cast<std::size_t>(c)];
    }
    const float exp = alpha * sum + beta * 1.0f;
    if (!ExpectClose(h_y_out[static_cast<std::size_t>(r)], exp, 1e-3f, "cublasSgemv result")) {
      return false;
    }
  }

  cudaFree(d_a);
  cudaFree(d_x);
  cudaFree(d_y);
  cublasDestroy(handle);
  return true;
}

bool RunSgemmCase(cublasHandle_t handle, int m, int n, int k, cublasOperation_t transa,
                  cublasOperation_t transb) {
  const int lda = (transa == CUBLAS_OP_N) ? m : k;
  const int ldb = (transb == CUBLAS_OP_N) ? k : n;
  const int ldc = m;

  std::vector<float> h_a(static_cast<std::size_t>(lda * ((transa == CUBLAS_OP_N) ? k : m)));
  std::vector<float> h_b(static_cast<std::size_t>(ldb * ((transb == CUBLAS_OP_N) ? n : k)));
  std::vector<float> h_c(static_cast<std::size_t>(ldc * n), 2.0f);

  for (std::size_t i = 0; i < h_a.size(); ++i) {
    h_a[i] = static_cast<float>((i % 13) + 1) * 0.1f;
  }
  for (std::size_t i = 0; i < h_b.size(); ++i) {
    h_b[i] = static_cast<float>((i % 11) + 1) * 0.1f;
  }

  float* d_a = nullptr;
  float* d_b = nullptr;
  float* d_c = nullptr;
  cudaMalloc(reinterpret_cast<void**>(&d_a), h_a.size() * sizeof(float));
  cudaMalloc(reinterpret_cast<void**>(&d_b), h_b.size() * sizeof(float));
  cudaMalloc(reinterpret_cast<void**>(&d_c), h_c.size() * sizeof(float));
  cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b.data(), h_b.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, h_c.data(), h_c.size() * sizeof(float), cudaMemcpyHostToDevice);

  const float alpha = 1.25f;
  const float beta = 0.5f;
  if (!Expect(cublasSgemm(handle, transa, transb, m, n, k, &alpha, d_a, lda, d_b, ldb, &beta, d_c,
                          ldc) == CUBLAS_STATUS_SUCCESS,
              "cublasSgemm")) {
    return false;
  }

  std::vector<float> h_out(h_c.size());
  cudaMemcpy(h_out.data(), d_c, h_out.size() * sizeof(float), cudaMemcpyDeviceToHost);

  for (int col = 0; col < n; ++col) {
    for (int row = 0; row < m; ++row) {
      float acc = 0.0f;
      for (int p = 0; p < k; ++p) {
        const float a_val =
            (transa == CUBLAS_OP_N) ? h_a[static_cast<std::size_t>(row + p * lda)]
                                    : h_a[static_cast<std::size_t>(p + row * lda)];
        const float b_val =
            (transb == CUBLAS_OP_N) ? h_b[static_cast<std::size_t>(p + col * ldb)]
                                    : h_b[static_cast<std::size_t>(col + p * ldb)];
        acc += a_val * b_val;
      }
      const float expected = alpha * acc + beta * 2.0f;
      if (!ExpectClose(h_out[static_cast<std::size_t>(row + col * ldc)], expected, 1e-2f,
                       "SGEMM output mismatch")) {
        return false;
      }
    }
  }

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  return true;
}

bool TestLevel3GemmAndGeam() {
  std::cout << "Testing cuBLAS Level-3 SGEMM (simdgroup + MPS fast path), HGEMM, StridedBatched, "
               "and SGEAM...\n";
  cublasHandle_t handle = nullptr;
  cublasCreate(&handle);

  // Small matrices -> simdgroup_matrix path
  if (!RunSgemmCase(handle, 32, 24, 40, CUBLAS_OP_N, CUBLAS_OP_N) ||
      !RunSgemmCase(handle, 32, 24, 40, CUBLAS_OP_T, CUBLAS_OP_N) ||
      !RunSgemmCase(handle, 32, 24, 40, CUBLAS_OP_N, CUBLAS_OP_T) ||
      !RunSgemmCase(handle, 32, 24, 40, CUBLAS_OP_T, CUBLAS_OP_T)) {
    return false;
  }

  // Large matrices (M,N,K >= 64) -> MPSMatrixMultiplication zero-copy fast path + LruPlanCache
  if (!RunSgemmCase(handle, 128, 96, 64, CUBLAS_OP_N, CUBLAS_OP_N) ||
      !RunSgemmCase(handle, 128, 96, 64, CUBLAS_OP_N, CUBLAS_OP_N) ||  // Cache hit
      !RunSgemmCase(handle, 96, 128, 64, CUBLAS_OP_T, CUBLAS_OP_N)) {
    return false;
  }

  // Hgemm & GemmEx (FP16 inputs)
  constexpr int kM = 16;
  constexpr int kN = 16;
  constexpr int kK = 16;
  std::vector<std::uint16_t> h_a16(kM * kK, FloatToHalfBits(0.5f));
  std::vector<std::uint16_t> h_b16(kK * kN, FloatToHalfBits(0.25f));
  std::vector<std::uint16_t> h_c16(kM * kN, FloatToHalfBits(0.0f));
  void* d_a16 = nullptr;
  void* d_b16 = nullptr;
  void* d_c16 = nullptr;
  cudaMalloc(&d_a16, h_a16.size() * sizeof(std::uint16_t));
  cudaMalloc(&d_b16, h_b16.size() * sizeof(std::uint16_t));
  cudaMalloc(&d_c16, h_c16.size() * sizeof(std::uint16_t));
  cudaMemcpy(d_a16, h_a16.data(), h_a16.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b16, h_b16.data(), h_b16.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_c16, h_c16.data(), h_c16.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice);

  const __half alpha16{FloatToHalfBits(1.0f)};
  const __half beta16{FloatToHalfBits(0.0f)};
  if (!Expect(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, kM, kN, kK, &alpha16,
                           static_cast<const __half*>(d_a16), kM, static_cast<const __half*>(d_b16),
                          kK, &beta16, static_cast<__half*>(d_c16), kM) == CUBLAS_STATUS_SUCCESS,
              "cublasHgemm")) {
    return false;
  }
  cudaMemcpy(h_c16.data(), d_c16, h_c16.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost);
  if (!ExpectClose(HalfBitsToFloat(h_c16[0]), 2.0f, 1e-2f, "cublasHgemm value")) {
    return false;
  }
  cudaFree(d_a16);
  cudaFree(d_b16);
  cudaFree(d_c16);

  // SgemmStridedBatched
  constexpr int kBatch = 3;
  std::vector<float> h_ba(kBatch * 8 * 8, 1.0f);
  std::vector<float> h_bb(kBatch * 8 * 8, 0.5f);
  std::vector<float> h_bc(kBatch * 8 * 8, 0.0f);
  float* d_ba = nullptr;
  float* d_bb = nullptr;
  float* d_bc = nullptr;
  cudaMalloc(reinterpret_cast<void**>(&d_ba), h_ba.size() * sizeof(float));
  cudaMalloc(reinterpret_cast<void**>(&d_bb), h_bb.size() * sizeof(float));
  cudaMalloc(reinterpret_cast<void**>(&d_bc), h_bc.size() * sizeof(float));
  cudaMemcpy(d_ba, h_ba.data(), h_ba.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_bb, h_bb.data(), h_bb.size() * sizeof(float), cudaMemcpyHostToDevice);
  const float one = 1.0f;
  const float zero = 0.0f;
  if (!Expect(cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, 8, 8, 8, &one, d_ba, 8,
                                        64, d_bb, 8, 64, &zero, d_bc, 8, 64,
                                        kBatch) == CUBLAS_STATUS_SUCCESS,
              "cublasSgemmStridedBatched")) {
    return false;
  }
  cudaMemcpy(h_bc.data(), d_bc, h_bc.size() * sizeof(float), cudaMemcpyDeviceToHost);
  for (float v : h_bc) {
    if (!ExpectClose(v, 4.0f, 1e-4f, "cublasSgemmStridedBatched element")) {
      return false;
    }
  }

  // Sgeam: C = 2.0 * A^T + 3.0 * B
  const float alpha_geam = 2.0f;
  const float beta_geam = 3.0f;
  if (!Expect(cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, 8, 8, &alpha_geam, d_ba, 8, &beta_geam,
                          d_bb, 8, d_bc, 8) == CUBLAS_STATUS_SUCCESS,
              "cublasSgeam")) {
    return false;
  }
  cudaMemcpy(h_bc.data(), d_bc, 64 * sizeof(float), cudaMemcpyDeviceToHost);
  if (!ExpectClose(h_bc[0], 3.5f, 1e-4f, "cublasSgeam element")) {
    return false;
  }

  cudaFree(d_ba);
  cudaFree(d_bb);
  cudaFree(d_bc);
  cublasDestroy(handle);
  return true;
}

}  // namespace

int main() {
  if (!TestHandleAndConfig() || !TestLevel1Ops() || !TestLevel2Gemv() ||
      !TestLevel3GemmAndGeam()) {
    return 1;
  }
  std::cout << "ALL CUBLAS TESTS PASSED!\n";
  return 0;
}

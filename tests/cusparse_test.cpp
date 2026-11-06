/**
 * @file cusparse_test.cpp
 * @brief Comprehensive verification test suite for the MUDA cuSPARSE shim (libcusparse.dylib).
 */

#include "cusparse.h"

#include <cmath>
#include <cstdint>
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

bool TestCsrSpMVAndSpMM() {
  std::cout << "Testing cuSPARSE CSR SpMV (non-transpose & transpose) and SpMM...\n";
  cusparseHandle_t handle = nullptr;
  if (!Expect(cusparseCreate(&handle) == CUSPARSE_STATUS_SUCCESS, "cusparseCreate")) {
    return false;
  }

  // 4x4 sparse matrix:
  // [ 10   0  20   0 ]
  // [  0  30   0  40 ]
  // [ 50  60   0   0 ]
  // [  0   0  70  80 ]
  const std::vector<std::int32_t> h_offsets = {0, 2, 4, 6, 8};
  const std::vector<std::int32_t> h_cols = {0, 2, 1, 3, 0, 1, 2, 3};
  const std::vector<float> h_vals = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f};
  const std::vector<float> h_x = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> h_y = {0.0f, 0.0f, 0.0f, 0.0f};

  std::int32_t* d_offsets = nullptr;
  std::int32_t* d_cols = nullptr;
  float* d_vals = nullptr;
  float* d_x = nullptr;
  float* d_y = nullptr;
  cudaMalloc(reinterpret_cast<void**>(&d_offsets), h_offsets.size() * sizeof(std::int32_t));
  cudaMalloc(reinterpret_cast<void**>(&d_cols), h_cols.size() * sizeof(std::int32_t));
  cudaMalloc(reinterpret_cast<void**>(&d_vals), h_vals.size() * sizeof(float));
  cudaMalloc(reinterpret_cast<void**>(&d_x), h_x.size() * sizeof(float));
  cudaMalloc(reinterpret_cast<void**>(&d_y), h_y.size() * sizeof(float));

  cudaMemcpy(d_offsets, h_offsets.data(), h_offsets.size() * sizeof(std::int32_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_cols, h_cols.data(), h_cols.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_vals, h_vals.data(), h_vals.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, h_y.data(), h_y.size() * sizeof(float), cudaMemcpyHostToDevice);

  cusparseSpMatDescr_t mat_a = nullptr;
  cusparseDnVecDescr_t vec_x = nullptr;
  cusparseDnVecDescr_t vec_y = nullptr;
  if (!Expect(cusparseCreateCsr(&mat_a, 4, 4, 8, d_offsets, d_cols, d_vals, CUSPARSE_INDEX_32I,
                                CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                                CUDA_R_32F) == CUSPARSE_STATUS_SUCCESS,
              "cusparseCreateCsr") ||
      !Expect(cusparseCreateDnVec(&vec_x, 4, d_x, CUDA_R_32F) == CUSPARSE_STATUS_SUCCESS,
              "cusparseCreateDnVec x") ||
      !Expect(cusparseCreateDnVec(&vec_y, 4, d_y, CUDA_R_32F) == CUSPARSE_STATUS_SUCCESS,
              "cusparseCreateDnVec y")) {
    return false;
  }

  const float alpha = 1.0f;
  const float beta = 0.0f;
  std::size_t buf_size = 0;
  if (!Expect(cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat_a,
                                      vec_x, &beta, vec_y, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT,
                                      &buf_size) == CUSPARSE_STATUS_SUCCESS,
              "cusparseSpMV_bufferSize") ||
      !Expect(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat_a, vec_x, &beta,
                           vec_y, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT,
                           nullptr) == CUSPARSE_STATUS_SUCCESS,
              "cusparseSpMV non-transpose")) {
    return false;
  }

  cudaMemcpy(h_y.data(), d_y, h_y.size() * sizeof(float), cudaMemcpyDeviceToHost);
  // Row 0: 10*1 + 20*3 = 70
  // Row 1: 30*2 + 40*4 = 220
  // Row 2: 50*1 + 60*2 = 170
  // Row 3: 70*3 + 80*4 = 530
  if (!ExpectClose(h_y[0], 70.0f, 1e-4f, "SpMV row 0") ||
      !ExpectClose(h_y[1], 220.0f, 1e-4f, "SpMV row 1") ||
      !ExpectClose(h_y[2], 170.0f, 1e-4f, "SpMV row 2") ||
      !ExpectClose(h_y[3], 530.0f, 1e-4f, "SpMV row 3")) {
    return false;
  }

  // Transpose SpMV: y = A^T * x
  if (!Expect(cusparseSpMV(handle, CUSPARSE_OPERATION_TRANSPOSE, &alpha, mat_a, vec_x, &beta, vec_y,
                           CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT,
                           nullptr) == CUSPARSE_STATUS_SUCCESS,
              "cusparseSpMV transpose")) {
    return false;
  }
  cudaMemcpy(h_y.data(), d_y, h_y.size() * sizeof(float), cudaMemcpyDeviceToHost);
  // Col 0: 10*1 + 50*3 = 160
  // Col 1: 30*2 + 60*3 = 240
  // Col 2: 20*1 + 70*4 = 300
  // Col 3: 40*2 + 80*4 = 400
  if (!ExpectClose(h_y[0], 160.0f, 1e-4f, "SpMV^T col 0") ||
      !ExpectClose(h_y[1], 240.0f, 1e-4f, "SpMV^T col 1") ||
      !ExpectClose(h_y[2], 300.0f, 1e-4f, "SpMV^T col 2") ||
      !ExpectClose(h_y[3], 400.0f, 1e-4f, "SpMV^T col 3")) {
    return false;
  }

  // SpMM: C (4x2, col-major) = A (4x4) * B (4x2, col-major)
  const std::vector<float> h_b = {1.0f, 2.0f, 3.0f, 4.0f, 0.5f, 1.0f, 1.5f, 2.0f};
  std::vector<float> h_c(8, 0.0f);
  float* d_b = nullptr;
  float* d_c = nullptr;
  cudaMalloc(reinterpret_cast<void**>(&d_b), h_b.size() * sizeof(float));
  cudaMalloc(reinterpret_cast<void**>(&d_c), h_c.size() * sizeof(float));
  cudaMemcpy(d_b, h_b.data(), h_b.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, h_c.data(), h_c.size() * sizeof(float), cudaMemcpyHostToDevice);

  cusparseDnMatDescr_t mat_b = nullptr;
  cusparseDnMatDescr_t mat_c = nullptr;
  cusparseCreateDnMat(&mat_b, 4, 2, 4, d_b, CUDA_R_32F, CUSPARSE_ORDER_COL);
  cusparseCreateDnMat(&mat_c, 4, 2, 4, d_c, CUDA_R_32F, CUSPARSE_ORDER_COL);

  if (!Expect(cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                           CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat_a, mat_b, &beta, mat_c,
                           CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT,
                           nullptr) == CUSPARSE_STATUS_SUCCESS,
              "cusparseSpMM")) {
    return false;
  }
  cudaMemcpy(h_c.data(), d_c, h_c.size() * sizeof(float), cudaMemcpyDeviceToHost);
  if (!ExpectClose(h_c[0], 70.0f, 1e-4f, "SpMM (0,0)") ||
      !ExpectClose(h_c[1], 220.0f, 1e-4f, "SpMM (1,0)") ||
      !ExpectClose(h_c[4], 35.0f, 1e-4f, "SpMM (0,1)") ||
      !ExpectClose(h_c[7], 265.0f, 1e-4f, "SpMM (3,1)")) {
    return false;
  }

  cusparseDestroyDnMat(mat_b);
  cusparseDestroyDnMat(mat_c);
  cusparseDestroyDnVec(vec_x);
  cusparseDestroyDnVec(vec_y);
  cusparseDestroySpMat(mat_a);
  cudaFree(d_offsets);
  cudaFree(d_cols);
  cudaFree(d_vals);
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_b);
  cudaFree(d_c);
  cusparseDestroy(handle);
  return true;
}

}  // namespace

int main() {
  if (!TestCsrSpMVAndSpMM()) {
    return 1;
  }
  std::cout << "ALL CUSPARSE TESTS PASSED!\n";
  return 0;
}


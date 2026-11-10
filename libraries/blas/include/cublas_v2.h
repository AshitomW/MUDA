/**
 * @file cublas_v2.h
 * @brief Clean-room C API header for the MUDA cuBLAS v2 compatibility shim.
 */

#pragma once

#include <cstddef>
#include <cstdint>

#include "cuda_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief cuBLAS status return codes.
 */
enum cublasStatus_t : std::uint8_t {
  CUBLAS_STATUS_SUCCESS = 0,
  CUBLAS_STATUS_NOT_INITIALIZED = 1,
  CUBLAS_STATUS_ALLOC_FAILED = 3,
  CUBLAS_STATUS_INVALID_VALUE = 7,
  CUBLAS_STATUS_ARCH_MISMATCH = 8,
  CUBLAS_STATUS_MAPPING_ERROR = 11,
  CUBLAS_STATUS_EXECUTION_FAILED = 13,
  CUBLAS_STATUS_INTERNAL_ERROR = 14,
  CUBLAS_STATUS_NOT_SUPPORTED = 15
};

/**
 * @brief Matrix transpose operation specifiers.
 */
enum cublasOperation_t : std::uint8_t { CUBLAS_OP_N = 0, CUBLAS_OP_T = 1, CUBLAS_OP_C = 2 };

/**
 * @brief Scalar pointer mode (host vs. device memory).
 */
enum cublasPointerMode_t : std::uint8_t {
  CUBLAS_POINTER_MODE_HOST = 0,
  CUBLAS_POINTER_MODE_DEVICE = 1
};

/**
 * @brief Math mode flags for tensor core / pedantic execution.
 */
enum cublasMath_t : std::uint8_t {
  CUBLAS_DEFAULT_MATH = 0,
  CUBLAS_TENSOR_OP_MATH = 1,
  CUBLAS_PEDANTIC_MATH = 2,
  CUBLAS_TF32_TENSOR_OP_MATH = 3
};

/**
 * @brief GEMM algorithm selection identifiers.
 */
enum cublasGemmAlgo_t : std::int8_t {
  CUBLAS_GEMM_DFALT = -1,
  CUBLAS_GEMM_DEFAULT = -1,
  CUBLAS_GEMM_DEFAULT_TENSOR_OP = 99
};

/**
 * @brief Compute precision types for cublasGemmEx.
 */
enum cublasComputeType_t : std::uint8_t {
  CUBLAS_COMPUTE_16F = 64,
  CUBLAS_COMPUTE_16F_PEDANTIC = 65,
  CUBLAS_COMPUTE_32F = 68,
  CUBLAS_COMPUTE_32F_PEDANTIC = 69,
  CUBLAS_COMPUTE_32F_FAST_16F = 74,
  CUBLAS_COMPUTE_32F_FAST_TF32 = 77
};

#ifndef MUDA_CUDA_DATA_TYPE_DEFINED
#define MUDA_CUDA_DATA_TYPE_DEFINED
/**
 * @brief Shared CUDA library element data types.
 */
enum cudaDataType_t : std::uint8_t {
  CUDA_R_16F = 2,
  CUDA_C_16F = 6,
  CUDA_R_32F = 0,
  CUDA_C_32F = 4,
  CUDA_R_64F = 1,
  CUDA_C_64F = 5,
  CUDA_R_8I = 3,
  CUDA_R_32I = 10
};
using cudaDataType = cudaDataType_t;
#endif

/**
 * @brief Opaque cuBLAS context handle.
 */
using cublasHandle_t = struct cublasContext*;

/* Handle Lifecycle & Configuration */
cublasStatus_t cublasCreate_v2(cublasHandle_t* handle);
cublasStatus_t cublasDestroy_v2(cublasHandle_t handle);
cublasStatus_t cublasGetVersion_v2(cublasHandle_t handle, int* version);
cublasStatus_t cublasSetStream_v2(cublasHandle_t handle, cudaStream_t streamId);
cublasStatus_t cublasGetStream_v2(cublasHandle_t handle, cudaStream_t* streamId);
cublasStatus_t cublasSetPointerMode_v2(cublasHandle_t handle, cublasPointerMode_t mode);
cublasStatus_t cublasGetPointerMode_v2(cublasHandle_t handle, cublasPointerMode_t* mode);
cublasStatus_t cublasSetMathMode(cublasHandle_t handle, cublasMath_t mode);
cublasStatus_t cublasGetMathMode(cublasHandle_t handle, cublasMath_t* mode);
const char* cublasGetStatusName(cublasStatus_t status);
const char* cublasGetStatusString(cublasStatus_t status);

/* Level-1 BLAS */
cublasStatus_t cublasSaxpy_v2(cublasHandle_t handle, int n, const float* alpha, const float* x,
                              int incx, float* y, int incy);
cublasStatus_t cublasDaxpy_v2(cublasHandle_t handle, int n, const double* alpha, const double* x,
                              int incx, double* y, int incy);
cublasStatus_t cublasScopy_v2(cublasHandle_t handle, int n, const float* x, int incx, float* y,
                              int incy);
cublasStatus_t cublasDcopy_v2(cublasHandle_t handle, int n, const double* x, int incx, double* y,
                              int incy);
cublasStatus_t cublasSscal_v2(cublasHandle_t handle, int n, const float* alpha, float* x, int incx);
cublasStatus_t cublasDscal_v2(cublasHandle_t handle, int n, const double* alpha, double* x,
                              int incx);
cublasStatus_t cublasSswap_v2(cublasHandle_t handle, int n, float* x, int incx, float* y, int incy);
cublasStatus_t cublasDswap_v2(cublasHandle_t handle, int n, double* x, int incx, double* y,
                              int incy);
cublasStatus_t cublasSdot_v2(cublasHandle_t handle, int n, const float* x, int incx, const float* y,
                             int incy, float* result);
cublasStatus_t cublasDdot_v2(cublasHandle_t handle, int n, const double* x, int incx,
                             const double* y, int incy, double* result);
cublasStatus_t cublasSnrm2_v2(cublasHandle_t handle, int n, const float* x, int incx,
                              float* result);
cublasStatus_t cublasDnrm2_v2(cublasHandle_t handle, int n, const double* x, int incx,
                              double* result);
cublasStatus_t cublasSasum_v2(cublasHandle_t handle, int n, const float* x, int incx,
                              float* result);
cublasStatus_t cublasDasum_v2(cublasHandle_t handle, int n, const double* x, int incx,
                              double* result);
cublasStatus_t cublasIsamax_v2(cublasHandle_t handle, int n, const float* x, int incx, int* result);
cublasStatus_t cublasIdamax_v2(cublasHandle_t handle, int n, const double* x, int incx,
                               int* result);

/* Level-2 BLAS */
cublasStatus_t cublasSgemv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n,
                              const float* alpha, const float* A, int lda, const float* x, int incx,
                              const float* beta, float* y, int incy);
cublasStatus_t cublasDgemv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n,
                              const double* alpha, const double* A, int lda, const double* x,
                              int incx, const double* beta, double* y, int incy);

/* Level-3 BLAS */
cublasStatus_t cublasSgemm_v2(cublasHandle_t handle, cublasOperation_t transa,
                              cublasOperation_t transb, int m, int n, int k, const float* alpha,
                              const float* A, int lda, const float* B, int ldb, const float* beta,
                              float* C, int ldc);
cublasStatus_t cublasDgemm_v2(cublasHandle_t handle, cublasOperation_t transa,
                              cublasOperation_t transb, int m, int n, int k, const double* alpha,
                              const double* A, int lda, const double* B, int ldb,
                              const double* beta, double* C, int ldc);

cublasStatus_t cublasHgemm(cublasHandle_t handle, cublasOperation_t transa,
                           cublasOperation_t transb, int m, int n, int k, const __half* alpha,
                           const __half* A, int lda, const __half* B, int ldb, const __half* beta,
                           __half* C, int ldc);

cublasStatus_t cublasGemmEx(cublasHandle_t handle, cublasOperation_t transa,
                            cublasOperation_t transb, int m, int n, int k, const void* alpha,
                            const void* A, cudaDataType Atype, int lda, const void* B,
                            cudaDataType Btype, int ldb, const void* beta, void* C,
                            cudaDataType Ctype, int ldc, int computeType, cublasGemmAlgo_t algo);

cublasStatus_t cublasSgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa,
                                         cublasOperation_t transb, int m, int n, int k,
                                         const float* alpha, const float* A, int lda,
                                         long long int strideA, const float* B, int ldb,
                                         long long int strideB, const float* beta, float* C,
                                         int ldc, long long int strideC, int batchCount);
cublasStatus_t cublasDgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa,
                                         cublasOperation_t transb, int m, int n, int k,
                                         const double* alpha, const double* A, int lda,
                                         long long int strideA, const double* B, int ldb,
                                         long long int strideB, const double* beta, double* C,
                                         int ldc, long long int strideC, int batchCount);

cublasStatus_t cublasGemmStridedBatchedEx(cublasHandle_t handle, cublasOperation_t transa,
                                          cublasOperation_t transb, int m, int n, int k,
                                          const void* alpha, const void* A, cudaDataType Atype,
                                          int lda, long long int strideA, const void* B,
                                          cudaDataType Btype, int ldb, long long int strideB,
                                          const void* beta, void* C, cudaDataType Ctype, int ldc,
                                          long long int strideC, int batchCount, int computeType,
                                          cublasGemmAlgo_t algo);

cublasStatus_t cublasSgeam(cublasHandle_t handle, cublasOperation_t transa,
                           cublasOperation_t transb, int m, int n, const float* alpha,
                           const float* A, int lda, const float* beta, const float* B, int ldb,
                           float* C, int ldc);
cublasStatus_t cublasDgeam(cublasHandle_t handle, cublasOperation_t transa,
                           cublasOperation_t transb, int m, int n, const double* alpha,
                           const double* A, int lda, const double* beta, const double* B, int ldb,
                           double* C, int ldc);

#define cublasCreate cublasCreate_v2
#define cublasDestroy cublasDestroy_v2
#define cublasGetVersion cublasGetVersion_v2
#define cublasSetStream cublasSetStream_v2
#define cublasGetStream cublasGetStream_v2
#define cublasSetPointerMode cublasSetPointerMode_v2
#define cublasGetPointerMode cublasGetPointerMode_v2
#define cublasSaxpy cublasSaxpy_v2
#define cublasDaxpy cublasDaxpy_v2
#define cublasScopy cublasScopy_v2
#define cublasDcopy cublasDcopy_v2
#define cublasSscal cublasSscal_v2
#define cublasDscal cublasDscal_v2
#define cublasSswap cublasSswap_v2
#define cublasDswap cublasDswap_v2
#define cublasSdot cublasSdot_v2
#define cublasDdot cublasDdot_v2
#define cublasSnrm2 cublasSnrm2_v2
#define cublasDnrm2 cublasDnrm2_v2
#define cublasSasum cublasSasum_v2
#define cublasDasum cublasDasum_v2
#define cublasIsamax cublasIsamax_v2
#define cublasIdamax cublasIdamax_v2
#define cublasSgemv cublasSgemv_v2
#define cublasDgemv cublasDgemv_v2
#define cublasSgemm cublasSgemm_v2
#define cublasDgemm cublasDgemm_v2

#ifdef __cplusplus
}  // extern "C"
#endif

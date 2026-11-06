/**
 * @file cusparse.h
 * @brief Clean-room C API header for the MUDA cuSPARSE compatibility shim.
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "cuda_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief cuSPARSE status return codes.
 */
typedef enum cusparseStatus_t {
  CUSPARSE_STATUS_SUCCESS = 0,
  CUSPARSE_STATUS_NOT_INITIALIZED = 1,
  CUSPARSE_STATUS_ALLOC_FAILED = 2,
  CUSPARSE_STATUS_INVALID_VALUE = 3,
  CUSPARSE_STATUS_ARCH_MISMATCH = 4,
  CUSPARSE_STATUS_MAPPING_ERROR = 5,
  CUSPARSE_STATUS_EXECUTION_FAILED = 6,
  CUSPARSE_STATUS_INTERNAL_ERROR = 7,
  CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED = 8,
  CUSPARSE_STATUS_ZERO_PIVOT = 9,
  CUSPARSE_STATUS_NOT_SUPPORTED = 10,
  CUSPARSE_STATUS_INSUFFICIENT_RESOURCES = 11
} cusparseStatus_t;

/**
 * @brief Sparse matrix transpose operation specifiers.
 */
typedef enum cusparseOperation_t {
  CUSPARSE_OPERATION_NON_TRANSPOSE = 0,
  CUSPARSE_OPERATION_TRANSPOSE = 1,
  CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE = 2
} cusparseOperation_t;

/**
 * @brief Sparse index integer width specifiers.
 */
typedef enum cusparseIndexType_t {
  CUSPARSE_INDEX_16U = 1,
  CUSPARSE_INDEX_32I = 2,
  CUSPARSE_INDEX_64I = 3
} cusparseIndexType_t;

/**
 * @brief Sparse index base (0-based C vs. 1-based Fortran).
 */
typedef enum cusparseIndexBase_t {
  CUSPARSE_INDEX_BASE_ZERO = 0,
  CUSPARSE_INDEX_BASE_ONE = 1
} cusparseIndexBase_t;

/**
 * @brief Dense matrix storage order.
 */
typedef enum cusparseOrder_t {
  CUSPARSE_ORDER_COL = 1,
  CUSPARSE_ORDER_ROW = 2
} cusparseOrder_t;

/**
 * @brief SpMV algorithm selectors.
 */
typedef enum cusparseSpMVAlg_t {
  CUSPARSE_SPMV_ALG_DEFAULT = 0,
  CUSPARSE_SPMV_CSR_ALG1 = 2,
  CUSPARSE_SPMV_CSR_ALG2 = 3
} cusparseSpMVAlg_t;

/**
 * @brief SpMM algorithm selectors.
 */
typedef enum cusparseSpMMAlg_t {
  CUSPARSE_SPMM_ALG_DEFAULT = 0,
  CUSPARSE_SPMM_CSR_ALG1 = 1,
  CUSPARSE_SPMM_CSR_ALG2 = 2
} cusparseSpMMAlg_t;

#ifndef MUDA_CUDA_DATA_TYPE_DEFINED
#define MUDA_CUDA_DATA_TYPE_DEFINED
typedef enum cudaDataType_t {
  CUDA_R_16F = 2,
  CUDA_C_16F = 6,
  CUDA_R_32F = 0,
  CUDA_C_32F = 4,
  CUDA_R_64F = 1,
  CUDA_C_64F = 5,
  CUDA_R_8I = 3,
  CUDA_R_32I = 10
} cudaDataType_t;
typedef cudaDataType_t cudaDataType;
#endif

/**
 * @brief Opaque cuSPARSE context and generic API descriptors.
 */
typedef struct cusparseContext* cusparseHandle_t;
typedef struct cusparseSpMatDescr* cusparseSpMatDescr_t;
typedef struct cusparseDnVecDescr* cusparseDnVecDescr_t;
typedef struct cusparseDnMatDescr* cusparseDnMatDescr_t;

/* Context & Version */
cusparseStatus_t cusparseCreate(cusparseHandle_t* handle);
cusparseStatus_t cusparseDestroy(cusparseHandle_t handle);
cusparseStatus_t cusparseGetVersion(cusparseHandle_t handle, int* version);
cusparseStatus_t cusparseSetStream(cusparseHandle_t handle, cudaStream_t streamId);
cusparseStatus_t cusparseGetStream(cusparseHandle_t handle, cudaStream_t* streamId);
const char* cusparseGetErrorName(cusparseStatus_t status);
const char* cusparseGetErrorString(cusparseStatus_t status);

/* Generic Descriptors */
cusparseStatus_t cusparseCreateCsr(cusparseSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols,
                                   int64_t nnz, void* csrRowOffsets, void* csrColInd,
                                   void* csrValues, cusparseIndexType_t csrRowOffsetsType,
                                   cusparseIndexType_t csrColIndType, cusparseIndexBase_t idxBase,
                                   cudaDataType valueType);
cusparseStatus_t cusparseDestroySpMat(cusparseSpMatDescr_t spMatDescr);

cusparseStatus_t cusparseCreateDnVec(cusparseDnVecDescr_t* dnVecDescr, int64_t size, void* values,
                                     cudaDataType valueType);
cusparseStatus_t cusparseDestroyDnVec(cusparseDnVecDescr_t dnVecDescr);

cusparseStatus_t cusparseCreateDnMat(cusparseDnMatDescr_t* dnMatDescr, int64_t rows, int64_t cols,
                                     int64_t ld, void* values, cudaDataType valueType,
                                     cusparseOrder_t order);
cusparseStatus_t cusparseDestroyDnMat(cusparseDnMatDescr_t dnMatDescr);

/* Generic Sparse Linear Algebra Operations */
cusparseStatus_t cusparseSpMV_bufferSize(cusparseHandle_t handle, cusparseOperation_t opA,
                                         const void* alpha, cusparseSpMatDescr_t matA,
                                         cusparseDnVecDescr_t vecX, const void* beta,
                                         cusparseDnVecDescr_t vecY, cudaDataType computeType,
                                         cusparseSpMVAlg_t alg, size_t* bufferSize);

cusparseStatus_t cusparseSpMV(cusparseHandle_t handle, cusparseOperation_t opA, const void* alpha,
                              cusparseSpMatDescr_t matA, cusparseDnVecDescr_t vecX,
                              const void* beta, cusparseDnVecDescr_t vecY,
                              cudaDataType computeType, cusparseSpMVAlg_t alg,
                              void* externalBuffer);

cusparseStatus_t cusparseSpMM_bufferSize(cusparseHandle_t handle, cusparseOperation_t opA,
                                         cusparseOperation_t opB, const void* alpha,
                                         cusparseSpMatDescr_t matA, cusparseDnMatDescr_t matB,
                                         const void* beta, cusparseDnMatDescr_t matC,
                                         cudaDataType computeType, cusparseSpMMAlg_t alg,
                                         size_t* bufferSize);

cusparseStatus_t cusparseSpMM(cusparseHandle_t handle, cusparseOperation_t opA,
                              cusparseOperation_t opB, const void* alpha,
                              cusparseSpMatDescr_t matA, cusparseDnMatDescr_t matB,
                              const void* beta, cusparseDnMatDescr_t matC,
                              cudaDataType computeType, cusparseSpMMAlg_t alg,
                              void* externalBuffer);

#ifdef __cplusplus
}  // extern "C"
#endif


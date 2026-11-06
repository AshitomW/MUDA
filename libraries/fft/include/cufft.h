/**
 * @file cufft.h
 * @brief Clean-room C API header for the MUDA cuFFT compatibility shim.
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "cuda_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Transform direction constants matching standard cuFFT semantics.
 */
#define CUFFT_FORWARD -1
#define CUFFT_INVERSE 1

/**
 * @brief cuFFT status return codes.
 */
typedef enum cufftResult_t {
  CUFFT_SUCCESS = 0,
  CUFFT_INVALID_PLAN = 1,
  CUFFT_ALLOC_FAILED = 2,
  CUFFT_INVALID_TYPE = 3,
  CUFFT_INVALID_VALUE = 4,
  CUFFT_INTERNAL_ERROR = 5,
  CUFFT_EXEC_FAILED = 6,
  CUFFT_SETUP_FAILED = 7,
  CUFFT_INVALID_SIZE = 8,
  CUFFT_UNALIGNED_DATA = 9,
  CUFFT_INCOMPLETE_PARAMETER_LIST = 10,
  CUFFT_INVALID_DEVICE = 11,
  CUFFT_PARSE_ERROR = 12,
  CUFFT_NO_WORKSPACE = 13,
  CUFFT_NOT_IMPLEMENTED = 14,
  CUFFT_NOT_SUPPORTED = 16
} cufftResult;

/**
 * @brief cuFFT transform data types.
 */
typedef enum cufftType_t {
  CUFFT_R2C = 0x2a,
  CUFFT_C2R = 0x2c,
  CUFFT_C2C = 0x29,
  CUFFT_D2Z = 0x6a,
  CUFFT_Z2D = 0x6c,
  CUFFT_Z2Z = 0x69
} cufftType;

/**
 * @brief Single-precision real and complex types.
 */
typedef float cufftReal;
typedef float2 cufftComplex;

/**
 * @brief Opaque cuFFT plan handle (pointer-sized or integer-compatible token).
 */
typedef struct cufftContext* cufftHandle;

/* Plan Lifecycle & Configuration */
cufftResult cufftCreate(cufftHandle* handle);
cufftResult cufftPlan1d(cufftHandle* plan, int nx, cufftType type, int batch);
cufftResult cufftPlan2d(cufftHandle* plan, int nx, int ny, cufftType type);
cufftResult cufftPlan3d(cufftHandle* plan, int nx, int ny, int nz, cufftType type);
cufftResult cufftPlanMany(cufftHandle* plan, int rank, int* n, int* inembed, int istride,
                          int idist, int* onembed, int ostride, int odist, cufftType type,
                          int batch);

cufftResult cufftMakePlan1d(cufftHandle plan, int nx, cufftType type, int batch,
                            size_t* workSize);
cufftResult cufftMakePlan2d(cufftHandle plan, int nx, int ny, cufftType type, size_t* workSize);
cufftResult cufftMakePlan3d(cufftHandle plan, int nx, int ny, int nz, cufftType type,
                            size_t* workSize);
cufftResult cufftMakePlanMany(cufftHandle plan, int rank, int* n, int* inembed, int istride,
                              int idist, int* onembed, int ostride, int odist, cufftType type,
                              int batch, size_t* workSize);

cufftResult cufftSetStream(cufftHandle plan, cudaStream_t stream);
cufftResult cufftGetSize(cufftHandle handle, size_t* workSize);
cufftResult cufftDestroy(cufftHandle plan);

/* Transform Execution */
cufftResult cufftExecC2C(cufftHandle plan, cufftComplex* idata, cufftComplex* odata, int direction);
cufftResult cufftExecR2C(cufftHandle plan, cufftReal* idata, cufftComplex* odata);
cufftResult cufftExecC2R(cufftHandle plan, cufftComplex* idata, cufftReal* odata);

#ifdef __cplusplus
}  // extern "C"
#endif


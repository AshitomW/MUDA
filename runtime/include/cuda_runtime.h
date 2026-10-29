/**
 * @file cuda_runtime.h
 * @brief Clean-room CUDA Runtime C API subset for memory, device, and error management.
 */

#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Managed memory attachment flags.
 */
#define cudaMemAttachGlobal 0x01u
#define cudaMemAttachHost 0x02u
#define cudaMemAttachSingle 0x04u

/**
 * @brief CUDA runtime error codes.
 */
typedef enum cudaError {
  cudaSuccess = 0,
  cudaErrorInvalidValue = 1,
  cudaErrorMemoryAllocation = 2,
  cudaErrorInitializationError = 3,
  cudaErrorInvalidDevice = 10,
  cudaErrorInvalidDevicePointer = 17,
  cudaErrorInvalidMemcpyDirection = 21,
  cudaErrorNoDevice = 100,
  cudaErrorNotSupported = 801,
  cudaErrorInternalError = 998,
  cudaErrorUnknown = 999
} cudaError_t;

/**
 * @brief CUDA memory copy direction specifiers.
 */
typedef enum cudaMemcpyKind {
  cudaMemcpyHostToHost = 0,
  cudaMemcpyHostToDevice = 1,
  cudaMemcpyDeviceToHost = 2,
  cudaMemcpyDeviceToDevice = 3,
  cudaMemcpyDefault = 4
} cudaMemcpyKind;

cudaError_t cudaGetDeviceCount(int* count);
cudaError_t cudaSetDevice(int device);
cudaError_t cudaGetDevice(int* device);
cudaError_t cudaDeviceSynchronize(void);
cudaError_t cudaDeviceReset(void);

cudaError_t cudaMalloc(void** devPtr, size_t size);
#ifdef __cplusplus
cudaError_t cudaMallocManaged(void** devPtr, size_t size, unsigned int flags = cudaMemAttachGlobal);
#else
cudaError_t cudaMallocManaged(void** devPtr, size_t size, unsigned int flags);
#endif
cudaError_t cudaFree(void* devPtr);
cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind);
cudaError_t cudaMemset(void* devPtr, int value, size_t count);

cudaError_t cudaGetLastError(void);
cudaError_t cudaPeekAtLastError(void);
const char* cudaGetErrorName(cudaError_t error);
const char* cudaGetErrorString(cudaError_t error);

#ifdef __cplusplus
}  // extern "C"
#endif

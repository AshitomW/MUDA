/**
 * @file cuda_runtime.h
 * @brief Clean-room CUDA Runtime C API subset for memory, device, kernel launch, and errors.
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
  cudaErrorInvalidConfiguration = 9,
  cudaErrorInvalidDevice = 10,
  cudaErrorInvalidDevicePointer = 17,
  cudaErrorInvalidMemcpyDirection = 21,
  cudaErrorInvalidDeviceFunction = 98,
  cudaErrorNoDevice = 100,
  cudaErrorInvalidResourceHandle = 400,
  cudaErrorLaunchOutOfResources = 701,
  cudaErrorLaunchFailure = 719,
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

/**
 * @brief Three-dimensional grid or thread-block extent.
 */
typedef struct dim3 {
  unsigned int x;
  unsigned int y;
  unsigned int z;
#ifdef __cplusplus
  constexpr dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1)
      : x(vx), y(vy), z(vz) {}
#endif
} dim3;

/**
 * @brief Opaque CUDA stream handle.
 */
typedef struct CUstream_st* cudaStream_t;

/**
 * @brief Parameter binding kind for registered MUDA kernels.
 */
typedef enum mudaKernelParamKind {
  mudaKernelParamBuffer = 0,
  mudaKernelParamBytes = 1
} mudaKernelParamKind;

/**
 * @brief Metadata for a single kernel parameter binding.
 */
typedef struct mudaKernelParamDesc {
  mudaKernelParamKind kind;
  size_t size;
} mudaKernelParamDesc;

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

cudaError_t mudaRegisterKernelFromFile(const void* hostFunc, const char* metallibPath,
                                       const char* kernelName, const mudaKernelParamDesc* params,
                                       size_t paramCount);
cudaError_t mudaRegisterKernelFromData(const void* hostFunc, const void* metallibData,
                                       size_t metallibSize, const char* kernelName,
                                       const mudaKernelParamDesc* params, size_t paramCount);
cudaError_t cudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim, void** args,
                             size_t sharedMem, cudaStream_t stream);

cudaError_t cudaGetLastError(void);
cudaError_t cudaPeekAtLastError(void);
const char* cudaGetErrorName(cudaError_t error);
const char* cudaGetErrorString(cudaError_t error);

#ifdef __cplusplus
}  // extern "C"
#endif

/**
 * @file cuda_runtime.h
 * @brief Clean-room CUDA Runtime C API subset for memory, device, streams, events, kernel launch,
 *        and errors.
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
 * @brief Stream creation flags.
 */
#define cudaStreamDefault 0x00u
#define cudaStreamNonBlocking 0x01u

/**
 * @brief Event creation flags.
 */
#define cudaEventDefault 0x00u
#define cudaEventBlockingSync 0x01u
#define cudaEventDisableTiming 0x02u
#define cudaEventInterprocess 0x04u

/**
 * @brief Event wait flags.
 */
#define cudaEventWaitDefault 0x00u
#define cudaEventWaitExternal 0x01u

/**
 * @brief Host allocation flags.
 */
#define cudaHostAllocDefault 0x00u
#define cudaHostAllocPortable 0x01u
#define cudaHostAllocMapped 0x02u
#define cudaHostAllocWriteCombined 0x04u

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
  cudaErrorInvalidPitchValue = 12,
  cudaErrorSymbolNotFound = 13,
  cudaErrorInvalidDevicePointer = 17,
  cudaErrorInvalidMemcpyDirection = 21,
  cudaErrorInvalidDeviceFunction = 98,
  cudaErrorNoDevice = 100,
  cudaErrorInvalidResourceHandle = 400,
  cudaErrorNotReady = 600,
  cudaErrorIllegalAddress = 700,
  cudaErrorLaunchOutOfResources = 701,
  cudaErrorLaunchFailure = 719,
  cudaErrorNotSupported = 801,
  cudaErrorUnsupportedLimit = 802,
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
 * @brief CUDA memory space type classification.
 */
typedef enum cudaMemoryType {
  cudaMemoryTypeUnregistered = 0,
  cudaMemoryTypeHost = 1,
  cudaMemoryTypeDevice = 2,
  cudaMemoryTypeManaged = 3
} cudaMemoryType;

/**
 * @brief CUDA compute mode.
 */
typedef enum cudaComputeMode {
  cudaComputeModeDefault = 0,
  cudaComputeModeExclusive = 1,
  cudaComputeModeProhibited = 2,
  cudaComputeModeExclusiveProcess = 3
} cudaComputeMode;

/**
 * @brief CUDA device attribute query identifiers.
 */
typedef enum cudaDeviceAttr {
  cudaDevAttrMaxThreadsPerBlock = 1,
  cudaDevAttrMaxBlockDimX = 2,
  cudaDevAttrMaxBlockDimY = 3,
  cudaDevAttrMaxBlockDimZ = 4,
  cudaDevAttrMaxGridDimX = 5,
  cudaDevAttrMaxGridDimY = 6,
  cudaDevAttrMaxGridDimZ = 7,
  cudaDevAttrMaxSharedMemoryPerBlock = 8,
  cudaDevAttrTotalConstantMemory = 9,
  cudaDevAttrWarpSize = 10,
  cudaDevAttrMaxPitch = 11,
  cudaDevAttrClockRate = 13,
  cudaDevAttrMultiProcessorCount = 16,
  cudaDevAttrComputeMode = 20,
  cudaDevAttrConcurrentKernels = 31,
  cudaDevAttrUnifiedAddressing = 41,
  cudaDevAttrComputeCapabilityMajor = 75,
  cudaDevAttrComputeCapabilityMinor = 76,
  cudaDevAttrManagedMemory = 83,
  cudaDevAttrHostNativeAtomicSupported = 86,
  cudaDevAttrPageableMemoryAccess = 88,
  cudaDevAttrConcurrentManagedAccess = 89,
  cudaDevAttrMaxSharedMemoryPerBlockOptin = 97
} cudaDeviceAttr;

/**
 * @brief Synthesized CUDA device properties.
 */
typedef struct cudaDeviceProp {
  char name[256];
  size_t totalGlobalMem;
  size_t sharedMemPerBlock;
  int regsPerBlock;
  int warpSize;
  size_t memPitch;
  int maxThreadsPerBlock;
  int maxThreadsDim[3];
  int maxGridSize[3];
  int clockRate;
  size_t totalConstMem;
  int major;
  int minor;
  int multiProcessorCount;
  int integrated;
  int canMapHostMemory;
  int computeMode;
  int concurrentKernels;
  int unifiedAddressing;
  int memoryClockRate;
  int memoryBusWidth;
  int l2CacheSize;
  int maxThreadsPerMultiProcessor;
  int managedMemory;
  int concurrentManagedAccess;
  int pageableMemoryAccess;
  size_t sharedMemPerMultiprocessor;
  size_t sharedMemPerBlockOptin;
} cudaDeviceProp;

/**
 * @brief Pointer attributes returned by cudaPointerGetAttributes.
 */
typedef struct cudaPointerAttributes {
  cudaMemoryType type;
  int device;
  void* devicePointer;
  void* hostPointer;
} cudaPointerAttributes;

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

typedef struct float2 {
  float x, y;
} float2;
typedef struct float3 {
  float x, y, z;
} float3;
typedef struct float4 {
  float x, y, z, w;
} float4;
typedef struct int2 {
  int x, y;
} int2;
typedef struct int3 {
  int x, y, z;
} int3;
typedef struct int4 {
  int x, y, z, w;
} int4;
typedef struct uint2 {
  unsigned int x, y;
} uint2;
typedef struct uint3 {
  unsigned int x, y, z;
} uint3;
typedef struct uint4 {
  unsigned int x, y, z, w;
} uint4;

/**
 * @brief Opaque CUDA stream handle.
 */
typedef struct CUstream_st* cudaStream_t;

/**
 * @brief Opaque CUDA event handle.
 */
typedef struct CUevent_st* cudaEvent_t;

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

/* Device and Version Management */
cudaError_t cudaGetDeviceCount(int* count);
cudaError_t cudaSetDevice(int device);
cudaError_t cudaGetDevice(int* device);
cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int device);
cudaError_t cudaDeviceGetAttribute(int* value, cudaDeviceAttr attr, int device);
cudaError_t cudaDeviceSynchronize(void);
cudaError_t cudaDeviceReset(void);
cudaError_t cudaRuntimeGetVersion(int* runtimeVersion);
cudaError_t cudaDriverGetVersion(int* driverVersion);

/* Memory Management */
cudaError_t cudaMalloc(void** devPtr, size_t size);
#ifdef __cplusplus
cudaError_t cudaMallocManaged(void** devPtr, size_t size, unsigned int flags = cudaMemAttachGlobal);
#else
cudaError_t cudaMallocManaged(void** devPtr, size_t size, unsigned int flags);
#endif
cudaError_t cudaMallocHost(void** ptr, size_t size);
cudaError_t cudaHostAlloc(void** pHost, size_t size, unsigned int flags);
cudaError_t cudaMallocPitch(void** devPtr, size_t* pitch, size_t width, size_t height);
cudaError_t cudaFree(void* devPtr);
cudaError_t cudaFreeHost(void* ptr);
cudaError_t cudaMemGetInfo(size_t* free, size_t* total);
cudaError_t cudaPointerGetAttributes(cudaPointerAttributes* attributes, const void* ptr);

/* Copies and Fills */
cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind);
#ifdef __cplusplus
cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind,
                            cudaStream_t stream = nullptr);
#else
cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind,
                            cudaStream_t stream);
#endif
cudaError_t cudaMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                         size_t height, cudaMemcpyKind kind);
#ifdef __cplusplus
cudaError_t cudaMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch,
                              size_t width, size_t height, cudaMemcpyKind kind,
                              cudaStream_t stream = nullptr);
#else
cudaError_t cudaMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch,
                              size_t width, size_t height, cudaMemcpyKind kind,
                              cudaStream_t stream);
#endif
cudaError_t cudaMemset(void* devPtr, int value, size_t count);
#ifdef __cplusplus
cudaError_t cudaMemsetAsync(void* devPtr, int value, size_t count, cudaStream_t stream = nullptr);
#else
cudaError_t cudaMemsetAsync(void* devPtr, int value, size_t count, cudaStream_t stream);
#endif
cudaError_t cudaMemset2D(void* devPtr, size_t pitch, int value, size_t width, size_t height);

/* Streams */
cudaError_t cudaStreamCreate(cudaStream_t* pStream);
cudaError_t cudaStreamCreateWithFlags(cudaStream_t* pStream, unsigned int flags);
cudaError_t cudaStreamDestroy(cudaStream_t stream);
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
cudaError_t cudaStreamQuery(cudaStream_t stream);
#ifdef __cplusplus
cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event,
                                unsigned int flags = cudaEventWaitDefault);
#else
cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags);
#endif

/* Events */
cudaError_t cudaEventCreate(cudaEvent_t* event);
cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags);
cudaError_t cudaEventDestroy(cudaEvent_t event);
#ifdef __cplusplus
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream = nullptr);
#else
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream);
#endif
cudaError_t cudaEventSynchronize(cudaEvent_t event);
cudaError_t cudaEventQuery(cudaEvent_t event);
cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end);

/* Kernel Registration and Launch */
cudaError_t mudaRegisterKernelFromFile(const void* hostFunc, const char* metallibPath,
                                       const char* kernelName, const mudaKernelParamDesc* params,
                                       size_t paramCount);
cudaError_t mudaRegisterKernelFromData(const void* hostFunc, const void* metallibData,
                                       size_t metallibSize, const char* kernelName,
                                       const mudaKernelParamDesc* params, size_t paramCount);
#ifdef __cplusplus
cudaError_t cudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim, void** args,
                             size_t sharedMem = 0, cudaStream_t stream = nullptr);
#else
cudaError_t cudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim, void** args,
                             size_t sharedMem, cudaStream_t stream);
#endif

/* Error Handling */
cudaError_t cudaGetLastError(void);
cudaError_t cudaPeekAtLastError(void);
const char* cudaGetErrorName(cudaError_t error);
const char* cudaGetErrorString(cudaError_t error);

#ifdef __cplusplus
}  // extern "C"
#endif

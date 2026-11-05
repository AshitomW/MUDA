/**
 * @file cuda.h
 * @brief Clean-room CUDA Driver C API subset for device initialization, context management,
 *        memory operations, streams, events, module loading, and kernel execution.
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int CUdevice;
typedef unsigned long long CUdeviceptr;

typedef struct CUctx_st* CUcontext;
typedef struct CUmod_st* CUmodule;
typedef struct CUfunc_st* CUfunction;
typedef struct CUstream_st* CUstream;
typedef struct CUevent_st* CUevent;
typedef struct CUarray_st* CUarray;

typedef struct CUuuid_st {
  char bytes[16];
} CUuuid;

typedef struct CUipcMemHandle_st {
  char reserved[64];
} CUipcMemHandle;

typedef struct CUlinkState_st* CUlinkState;

#ifdef __cplusplus
#define CU_LAUNCH_PARAM_END (reinterpret_cast<void*>(0x00))
#define CU_LAUNCH_PARAM_BUFFER_POINTER (reinterpret_cast<void*>(0x01))
#define CU_LAUNCH_PARAM_BUFFER_SIZE (reinterpret_cast<void*>(0x02))
#else
#define CU_LAUNCH_PARAM_END ((void*)0x00)
#define CU_LAUNCH_PARAM_BUFFER_POINTER ((void*)0x01)
#define CU_LAUNCH_PARAM_BUFFER_SIZE ((void*)0x02)
#endif

#define CU_STREAM_DEFAULT 0x00u
#define CU_STREAM_NON_BLOCKING 0x01u

#define CU_EVENT_DEFAULT 0x00u
#define CU_EVENT_BLOCKING_SYNC 0x01u
#define CU_EVENT_DISABLE_TIMING 0x02u
#define CU_EVENT_INTERPROCESS 0x04u

#define CU_MEMHOSTALLOC_PORTABLE 0x01u
#define CU_MEMHOSTALLOC_DEVICEMAP 0x02u
#define CU_MEMHOSTALLOC_WRITECOMBINED 0x04u

typedef enum cudaError_enum {
  CUDA_SUCCESS = 0,
  CUDA_ERROR_INVALID_VALUE = 1,
  CUDA_ERROR_OUT_OF_MEMORY = 2,
  CUDA_ERROR_NOT_INITIALIZED = 3,
  CUDA_ERROR_DEINITIALIZED = 4,
  CUDA_ERROR_PROFILER_DISABLED = 5,
  CUDA_ERROR_PROFILER_NOT_INITIALIZED = 6,
  CUDA_ERROR_PROFILER_ALREADY_STARTED = 7,
  CUDA_ERROR_PROFILER_ALREADY_STOPPED = 8,
  CUDA_ERROR_NO_DEVICE = 100,
  CUDA_ERROR_INVALID_DEVICE = 101,
  CUDA_ERROR_INVALID_IMAGE = 200,
  CUDA_ERROR_INVALID_CONTEXT = 201,
  CUDA_ERROR_CONTEXT_ALREADY_CURRENT = 202,
  CUDA_ERROR_MAP_FAILED = 205,
  CUDA_ERROR_UNMAP_FAILED = 206,
  CUDA_ERROR_ARRAY_IS_MAPPED = 207,
  CUDA_ERROR_ALREADY_MAPPED = 208,
  CUDA_ERROR_NO_BINARY_FOR_GPU = 209,
  CUDA_ERROR_ALREADY_ACQUIRED = 210,
  CUDA_ERROR_NOT_MAPPED = 211,
  CUDA_ERROR_NOT_MAPPED_AS_ARRAY = 212,
  CUDA_ERROR_NOT_MAPPED_AS_POINTER = 213,
  CUDA_ERROR_ECC_UNCORRECTABLE = 214,
  CUDA_ERROR_UNSUPPORTED_LIMIT = 215,
  CUDA_ERROR_CONTEXT_ALREADY_IN_USE = 216,
  CUDA_ERROR_PEER_ACCESS_UNSUPPORTED = 217,
  CUDA_ERROR_INVALID_PTX = 218,
  CUDA_ERROR_INVALID_GRAPHICS_CONTEXT = 219,
  CUDA_ERROR_NVLINK_UNCORRECTABLE = 220,
  CUDA_ERROR_JIT_COMPILER_NOT_FOUND = 221,
  CUDA_ERROR_UNSUPPORTED_PTX_VERSION = 222,
  CUDA_ERROR_JIT_COMPILATION_DISABLED = 223,
  CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY = 224,
  CUDA_ERROR_INVALID_SOURCE = 300,
  CUDA_ERROR_FILE_NOT_FOUND = 301,
  CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = 302,
  CUDA_ERROR_SHARED_OBJECT_INIT_FAILED = 303,
  CUDA_ERROR_OPERATING_SYSTEM = 304,
  CUDA_ERROR_INVALID_HANDLE = 400,
  CUDA_ERROR_ILLEGAL_STATE = 401,
  CUDA_ERROR_NOT_FOUND = 500,
  CUDA_ERROR_NOT_READY = 600,
  CUDA_ERROR_ILLEGAL_ADDRESS = 700,
  CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES = 701,
  CUDA_ERROR_LAUNCH_TIMEOUT = 702,
  CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING = 703,
  CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED = 704,
  CUDA_ERROR_PEER_ACCESS_NOT_ENABLED = 705,
  CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE = 708,
  CUDA_ERROR_CONTEXT_IS_DESTROYED = 709,
  CUDA_ERROR_ASSERT = 710,
  CUDA_ERROR_TOO_MANY_PEERS = 711,
  CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 712,
  CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED = 713,
  CUDA_ERROR_HARDWARE_STACK_ERROR = 714,
  CUDA_ERROR_ILLEGAL_INSTRUCTION = 715,
  CUDA_ERROR_MISALIGNED_ADDRESS = 716,
  CUDA_ERROR_INVALID_ADDRESS_SPACE = 717,
  CUDA_ERROR_INVALID_PC = 718,
  CUDA_ERROR_LAUNCH_FAILED = 719,
  CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE = 720,
  CUDA_ERROR_NOT_PERMITTED = 800,
  CUDA_ERROR_NOT_SUPPORTED = 801,
  CUDA_ERROR_SYSTEM_NOT_READY = 802,
  CUDA_ERROR_SYSTEM_DRIVER_MISMATCH = 803,
  CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE = 804,
  CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED = 900,
  CUDA_ERROR_STREAM_CAPTURE_INVALIDATED = 901,
  CUDA_ERROR_STREAM_CAPTURE_MERGE = 902,
  CUDA_ERROR_STREAM_CAPTURE_UNMATCHED = 903,
  CUDA_ERROR_STREAM_CAPTURE_UNJOINED = 904,
  CUDA_ERROR_STREAM_CAPTURE_ISOLATION = 905,
  CUDA_ERROR_STREAM_CAPTURE_IMPLICIT = 906,
  CUDA_ERROR_CAPTURED_EVENT = 907,
  CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD = 908,
  CUDA_ERROR_TIMEOUT = 909,
  CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE = 910,
  CUDA_ERROR_UNKNOWN = 999
} CUresult;

typedef enum CUdevice_attribute_enum {
  CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,
  CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2,
  CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3,
  CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4,
  CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5,
  CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6,
  CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7,
  CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8,
  CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK = 8,
  CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = 9,
  CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10,
  CU_DEVICE_ATTRIBUTE_MAX_PITCH = 11,
  CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12,
  CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK = 12,
  CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13,
  CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = 14,
  CU_DEVICE_ATTRIBUTE_GPU_OVERLAP = 15,
  CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16,
  CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT = 17,
  CU_DEVICE_ATTRIBUTE_INTEGRATED = 18,
  CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = 19,
  CU_DEVICE_ATTRIBUTE_COMPUTE_MODE = 20,
  CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT = 30,
  CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS = 31,
  CU_DEVICE_ATTRIBUTE_ECC_ENABLED = 32,
  CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33,
  CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34,
  CU_DEVICE_ATTRIBUTE_TCC_DRIVER = 35,
  CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36,
  CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37,
  CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38,
  CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39,
  CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = 40,
  CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = 41,
  CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = 50,
  CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT = 51,
  CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
  CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76,
  CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED = 78,
  CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED = 79,
  CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED = 80,
  CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81,
  CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR = 82,
  CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY = 83,
  CU_DEVICE_ATTRIBUTE_IS_MULTI_GPU_BOARD = 84,
  CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID = 85,
  CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED = 86,
  CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO = 87,
  CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS = 88,
  CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS = 89,
  CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED = 90,
  CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = 91,
  CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH = 95,
  CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH = 96,
  CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97
} CUdevice_attribute;

typedef enum CUpointer_attribute_enum {
  CU_POINTER_ATTRIBUTE_CONTEXT = 1,
  CU_POINTER_ATTRIBUTE_MEMORY_TYPE = 2,
  CU_POINTER_ATTRIBUTE_DEVICE_POINTER = 3,
  CU_POINTER_ATTRIBUTE_HOST_POINTER = 4,
  CU_POINTER_ATTRIBUTE_P2P_TOKENS = 5,
  CU_POINTER_ATTRIBUTE_SYNC_MEMOPS = 6,
  CU_POINTER_ATTRIBUTE_BUFFER_ID = 7,
  CU_POINTER_ATTRIBUTE_IS_MANAGED = 8,
  CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL = 9,
  CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE = 10,
  CU_POINTER_ATTRIBUTE_RANGE_START_ADDR = 11,
  CU_POINTER_ATTRIBUTE_RANGE_SIZE = 12,
  CU_POINTER_ATTRIBUTE_MAPPED = 13,
  CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES = 14,
  CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE = 15,
  CU_POINTER_ATTRIBUTE_ACCESS_FLAGS = 16,
  CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE = 17
} CUpointer_attribute;

typedef enum CUfunction_attribute_enum {
  CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0,
  CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1,
  CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES = 2,
  CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = 3,
  CU_FUNC_ATTRIBUTE_NUM_REGS = 4,
  CU_FUNC_ATTRIBUTE_PTX_VERSION = 5,
  CU_FUNC_ATTRIBUTE_BINARY_VERSION = 6,
  CU_FUNC_ATTRIBUTE_CACHE_MODE_CA = 7,
  CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8,
  CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = 9,
  CU_FUNC_ATTRIBUTE_MAX = 10
} CUfunction_attribute;

typedef enum CUjit_option_enum {
  CU_JIT_MAX_REGISTERS = 0,
  CU_JIT_THREADS_PER_BLOCK = 1,
  CU_JIT_WALL_TIME = 2,
  CU_JIT_INFO_LOG_BUFFER = 3,
  CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 4,
  CU_JIT_ERROR_LOG_BUFFER = 5,
  CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6,
  CU_JIT_OPTIMIZATION_LEVEL = 7,
  CU_JIT_TARGET_FROM_CUCONTEXT = 8,
  CU_JIT_TARGET = 9,
  CU_JIT_FALLBACK_STRATEGY = 10,
  CU_JIT_GENERATE_DEBUG_INFO = 11,
  CU_JIT_LOG_VERBOSE = 12,
  CU_JIT_GENERATE_LINE_INFO = 13,
  CU_JIT_CACHE_MODE = 14
} CUjit_option;

typedef enum CUjitInputType_enum {
  CU_JIT_INPUT_CUBIN = 0,
  CU_JIT_INPUT_PTX = 1,
  CU_JIT_INPUT_FATBINARY = 2,
  CU_JIT_INPUT_OBJECT = 3,
  CU_JIT_INPUT_LIBRARY = 4,
  CU_JIT_NUM_INPUT_TYPES = 5
} CUjitInputType;

typedef enum CUfunc_cache_enum {
  CU_FUNC_CACHE_PREFER_NONE = 0,
  CU_FUNC_CACHE_PREFER_SHARED = 1,
  CU_FUNC_CACHE_PREFER_L1 = 2,
  CU_FUNC_CACHE_PREFER_EQUAL = 3
} CUfunc_cache;

typedef enum CUsharedconfig_enum {
  CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE = 0,
  CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE = 1,
  CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE = 2
} CUsharedconfig;

typedef struct CUDA_MEMCPY2D_st {
  size_t srcXInBytes;
  size_t srcY;
  int srcMemoryType;
  const void* srcHost;
  CUdeviceptr srcDevice;
  CUarray srcArray;
  size_t srcPitch;

  size_t dstXInBytes;
  size_t dstY;
  int dstMemoryType;
  void* dstHost;
  CUdeviceptr dstDevice;
  CUarray dstArray;
  size_t dstPitch;

  size_t WidthInBytes;
  size_t Height;
} CUDA_MEMCPY2D;

typedef void (*CUstreamCallback)(CUstream hStream, CUresult status, void* userData);
typedef size_t (*CUoccupancyB2DSize)(int blockSize);

/* Initialization and Device Management */
CUresult cuInit(unsigned int Flags);
CUresult cuDriverGetVersion(int* driverVersion);
CUresult cuDeviceGet(CUdevice* device, int ordinal);
CUresult cuDeviceGetCount(int* count);
CUresult cuDeviceGetName(char* name, int len, CUdevice dev);
CUresult cuDeviceGetUuid(CUuuid* uuid, CUdevice dev);
CUresult cuDeviceTotalMem(size_t* bytes, CUdevice dev);
CUresult cuDeviceTotalMem_v2(size_t* bytes, CUdevice dev);
CUresult cuDeviceGetAttribute(int* pi, CUdevice_attribute attrib, CUdevice dev);
CUresult cuDeviceComputeCapability(int* major, int* minor, CUdevice dev);
CUresult cuDeviceCanAccessPeer(int* canAccessPeer, CUdevice dev, CUdevice peerDev);

/* Primary Context Management */
CUresult cuDevicePrimaryCtxRetain(CUcontext* pctx, CUdevice dev);
CUresult cuDevicePrimaryCtxRelease(CUdevice dev);
CUresult cuDevicePrimaryCtxReset(CUdevice dev);
CUresult cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int* flags, int* active);
CUresult cuDevicePrimaryCtxSetFlags(CUdevice dev, unsigned int flags);

/* Context Management */
CUresult cuCtxCreate(CUcontext* pctx, unsigned int flags, CUdevice dev);
CUresult cuCtxCreate_v2(CUcontext* pctx, unsigned int flags, CUdevice dev);
CUresult cuCtxDestroy(CUcontext ctx);
CUresult cuCtxDestroy_v2(CUcontext ctx);
CUresult cuCtxPushCurrent(CUcontext ctx);
CUresult cuCtxPushCurrent_v2(CUcontext ctx);
CUresult cuCtxPopCurrent(CUcontext* pctx);
CUresult cuCtxPopCurrent_v2(CUcontext* pctx);
CUresult cuCtxSetCurrent(CUcontext ctx);
CUresult cuCtxGetCurrent(CUcontext* pctx);
CUresult cuCtxGetCurrent_v2(CUcontext* pctx);
CUresult cuCtxGetDevice(CUdevice* device);
CUresult cuCtxSynchronize(void);
CUresult cuCtxEnablePeerAccess(CUcontext peerContext, unsigned int Flags);
CUresult cuCtxDisablePeerAccess(CUcontext peerContext);
CUresult cuCtxGetFlags(unsigned int* flags);
CUresult cuCtxGetApiVersion(CUcontext ctx, unsigned int* version);

/* Memory Management */
CUresult cuMemAlloc(CUdeviceptr* dptr, size_t bytesize);
CUresult cuMemAlloc_v2(CUdeviceptr* dptr, size_t bytesize);
CUresult cuMemAllocPitch(CUdeviceptr* dptr, size_t* pPitch, size_t WidthInBytes, size_t Height,
                         unsigned int ElementSizeBytes);
CUresult cuMemAllocPitch_v2(CUdeviceptr* dptr, size_t* pPitch, size_t WidthInBytes, size_t Height,
                            unsigned int ElementSizeBytes);
CUresult cuMemAllocManaged(CUdeviceptr* dptr, size_t bytesize, unsigned int flags);
CUresult cuMemFree(CUdeviceptr dptr);
CUresult cuMemFree_v2(CUdeviceptr dptr);
CUresult cuMemAllocHost(void** pp, size_t bytesize);
CUresult cuMemAllocHost_v2(void** pp, size_t bytesize);
CUresult cuMemFreeHost(void* p);
CUresult cuMemHostAlloc(void** pp, size_t bytesize, unsigned int Flags);
CUresult cuMemHostRegister(void* p, size_t bytesize, unsigned int Flags);
CUresult cuMemHostRegister_v2(void* p, size_t bytesize, unsigned int Flags);
CUresult cuMemHostUnregister(void* p);
CUresult cuMemHostGetDevicePointer(CUdeviceptr* pdptr, void* p, unsigned int Flags);
CUresult cuMemHostGetDevicePointer_v2(CUdeviceptr* pdptr, void* p, unsigned int Flags);
CUresult cuMemHostGetFlags(unsigned int* pFlags, void* p);
CUresult cuMemGetInfo(size_t* free, size_t* total);
CUresult cuMemGetInfo_v2(size_t* free, size_t* total);
CUresult cuMemGetAddressRange(CUdeviceptr* pbase, size_t* psize, CUdeviceptr dptr);
CUresult cuMemGetAddressRange_v2(CUdeviceptr* pbase, size_t* psize, CUdeviceptr dptr);
CUresult cuPointerGetAttribute(void* data, CUpointer_attribute attribute, CUdeviceptr ptr);
CUresult cuPointerGetAttributes(unsigned int numAttributes, CUpointer_attribute* attributes,
                                void** data, CUdeviceptr ptr);

/* Memory Transfers */
CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount);
CUresult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount);
CUresult cuMemcpyDtoH(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount);
CUresult cuMemcpyDtoH_v2(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount);
CUresult cuMemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount);
CUresult cuMemcpyDtoD_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount);
CUresult cuMemcpyHtoDAsync(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount,
                           CUstream hStream);
CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount,
                              CUstream hStream);
CUresult cuMemcpyDtoHAsync(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount,
                           CUstream hStream);
CUresult cuMemcpyDtoHAsync_v2(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount,
                              CUstream hStream);
CUresult cuMemcpyDtoDAsync(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount,
                           CUstream hStream);
CUresult cuMemcpyDtoDAsync_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount,
                              CUstream hStream);
CUresult cuMemcpyAsync(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount,
                       CUstream hStream);
CUresult cuMemcpy2D(const CUDA_MEMCPY2D* pCopy);
CUresult cuMemcpy2D_v2(const CUDA_MEMCPY2D* pCopy);
CUresult cuMemcpy2DAsync(const CUDA_MEMCPY2D* pCopy, CUstream hStream);
CUresult cuMemcpy2DAsync_v2(const CUDA_MEMCPY2D* pCopy, CUstream hStream);
CUresult cuMemsetD8(CUdeviceptr dstDevice, unsigned char uc, size_t N);
CUresult cuMemsetD8_v2(CUdeviceptr dstDevice, unsigned char uc, size_t N);
CUresult cuMemsetD16(CUdeviceptr dstDevice, unsigned short us, size_t N);
CUresult cuMemsetD16_v2(CUdeviceptr dstDevice, unsigned short us, size_t N);
CUresult cuMemsetD32(CUdeviceptr dstDevice, unsigned int ui, size_t N);
CUresult cuMemsetD32_v2(CUdeviceptr dstDevice, unsigned int ui, size_t N);
CUresult cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream hStream);
CUresult cuMemsetD32Async(CUdeviceptr dstDevice, unsigned int ui, size_t N, CUstream hStream);

/* Stream Management */
CUresult cuStreamCreate(CUstream* phStream, unsigned int Flags);
CUresult cuStreamCreateWithPriority(CUstream* phStream, unsigned int flags, int priority);
CUresult cuStreamDestroy(CUstream hStream);
CUresult cuStreamDestroy_v2(CUstream hStream);
CUresult cuStreamSynchronize(CUstream hStream);
CUresult cuStreamQuery(CUstream hStream);
CUresult cuStreamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int Flags);
CUresult cuStreamAddCallback(CUstream hStream, CUstreamCallback callback, void* userData,
                             unsigned int flags);

/* Event Management */
CUresult cuEventCreate(CUevent* phEvent, unsigned int Flags);
CUresult cuEventDestroy(CUevent hEvent);
CUresult cuEventDestroy_v2(CUevent hEvent);
CUresult cuEventRecord(CUevent hEvent, CUstream hStream);
CUresult cuEventSynchronize(CUevent hEvent);
CUresult cuEventQuery(CUevent hEvent);
CUresult cuEventElapsedTime(float* pMilliseconds, CUevent hStart, CUevent hEnd);

/* Module & Function Management */
CUresult cuModuleLoad(CUmodule* module, const char* fname);
CUresult cuModuleLoadData(CUmodule* module, const void* image);
CUresult cuModuleLoadDataEx(CUmodule* module, const void* image, unsigned int numOptions,
                            CUjit_option* options, void** optionValues);
CUresult cuModuleUnload(CUmodule hmod);
CUresult cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, const char* name);
CUresult cuModuleGetGlobal(CUdeviceptr* dptr, size_t* bytes, CUmodule hmod, const char* name);
CUresult cuModuleGetGlobal_v2(CUdeviceptr* dptr, size_t* bytes, CUmodule hmod, const char* name);
CUresult cuFuncGetAttribute(int* pi, CUfunction_attribute attrib, CUfunction hfunc);
CUresult cuFuncSetCacheConfig(CUfunction hfunc, CUfunc_cache config);
CUresult cuFuncSetSharedMemConfig(CUfunction hfunc, CUsharedconfig config);

/* Occupancy Calculation */
CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, CUfunction func, int blockSize,
                                                     size_t dynamicSMemSize);
CUresult cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* numBlocks, CUfunction func,
                                                              int blockSize, size_t dynamicSMemSize,
                                                              unsigned int flags);
CUresult cuOccupancyMaxPotentialBlockSize(int* minGridSize, int* blockSize, CUfunction func,
                                          CUoccupancyB2DSize blockSizeToDynamicSMemSize,
                                          size_t dynamicSMemSize, int blockSizeLimit);
CUresult cuOccupancyMaxPotentialBlockSizeWithFlags(int* minGridSize, int* blockSize,
                                                   CUfunction func,
                                                   CUoccupancyB2DSize blockSizeToDynamicSMemSize,
                                                   size_t dynamicSMemSize, int blockSizeLimit,
                                                   unsigned int flags);

/* Inter-Process Communication (IPC) */
CUresult cuIpcGetMemHandle(CUipcMemHandle* pHandle, CUdeviceptr dptr);
CUresult cuIpcOpenMemHandle(CUdeviceptr* pdptr, CUipcMemHandle handle, unsigned int Flags);
CUresult cuIpcCloseMemHandle(CUdeviceptr dptr);

/* Profiler Support */
CUresult cuProfilerStart(void);
CUresult cuProfilerStop(void);
CUresult cuProfilerInitialize(const char* configFile, const char* outputFile, int outputMode);

/* JIT Linker */
CUresult cuLinkCreate(unsigned int numOptions, CUjit_option* options, void** optionValues,
                      CUlinkState* stateOut);
CUresult cuLinkCreate_v2(unsigned int numOptions, CUjit_option* options, void** optionValues,
                         CUlinkState* stateOut);
CUresult cuLinkAddData(CUlinkState state, CUjitInputType type, void* data, size_t size,
                       const char* name, unsigned int numOptions, CUjit_option* options,
                       void** optionValues);
CUresult cuLinkAddData_v2(CUlinkState state, CUjitInputType type, void* data, size_t size,
                          const char* name, unsigned int numOptions, CUjit_option* options,
                          void** optionValues);
CUresult cuLinkAddFile(CUlinkState state, CUjitInputType type, const char* path,
                       unsigned int numOptions, CUjit_option* options, void** optionValues);
CUresult cuLinkAddFile_v2(CUlinkState state, CUjitInputType type, const char* path,
                          unsigned int numOptions, CUjit_option* options, void** optionValues);
CUresult cuLinkComplete(CUlinkState state, void** cubinOut, size_t* sizeOut);
CUresult cuLinkDestroy(CUlinkState state);

/* Kernel Launch */
CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
                        unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
                        unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream,
                        void** kernelParams, void** extra);
CUresult cuLaunchCooperativeKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
                                   unsigned int gridDimZ, unsigned int blockDimX,
                                   unsigned int blockDimY, unsigned int blockDimZ,
                                   unsigned int sharedMemBytes, CUstream hStream,
                                   void** kernelParams);

/* Error Handling */
CUresult cuGetErrorString(CUresult error, const char** pStr);
CUresult cuGetErrorName(CUresult error, const char** pStr);

#ifdef __cplusplus
}
#endif

/**
 * @file driver_context.mm
 * @brief CUDA Driver API initialization, device queries, context management, and error strings.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <atomic>
#include <cstring>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_manager.hpp"
#include "driver/driver_internal.hpp"
#include "muda/runtime/driver_types.hpp"
#include "runtime_context.hpp"

namespace muda::runtime::driver {

thread_local CUcontext g_current_ctx = nullptr;
std::atomic<int> g_primary_ctx_refcount{0};
CUctx_st g_primary_ctx{0, 0, nil, nil};

CUresult MapCudaError(cudaError_t err) noexcept {
  switch (err) {
    case cudaSuccess:
      return CUDA_SUCCESS;
    case cudaErrorInvalidValue:
      return CUDA_ERROR_INVALID_VALUE;
    case cudaErrorMemoryAllocation:
      return CUDA_ERROR_OUT_OF_MEMORY;
    case cudaErrorInitializationError:
      return CUDA_ERROR_NOT_INITIALIZED;
    case cudaErrorInvalidDevice:
      return CUDA_ERROR_INVALID_DEVICE;
    case cudaErrorNoDevice:
      return CUDA_ERROR_NO_DEVICE;
    case cudaErrorInvalidResourceHandle:
      return CUDA_ERROR_INVALID_HANDLE;
    case cudaErrorNotReady:
      return CUDA_ERROR_NOT_READY;
    case cudaErrorLaunchFailure:
      return CUDA_ERROR_LAUNCH_FAILED;
    case cudaErrorLaunchOutOfResources:
      return CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES;
    case cudaErrorNotSupported:
      return CUDA_ERROR_NOT_SUPPORTED;
    case cudaErrorInvalidConfiguration:
      return CUDA_ERROR_INVALID_VALUE;
    default:
      return CUDA_ERROR_UNKNOWN;
  }
}

CUcontext EnsureContext() noexcept {
  if (g_current_ctx != nullptr) {
    return g_current_ctx;
  }
  auto& rt = muda::runtime::cudart::RuntimeContext::Instance();
  (void)rt.EnsureInitialized();
  g_primary_ctx.device = rt.Device();
  g_primary_ctx.queue = rt.DefaultQueue();
  g_current_ctx = &g_primary_ctx;
  return g_current_ctx;
}

}  // namespace muda::runtime::driver

extern "C" {

CUresult cuInit([[maybe_unused]] unsigned int Flags) {
  auto& rt = muda::runtime::cudart::RuntimeContext::Instance();
  const cudaError_t err = rt.EnsureInitialized();
  return muda::runtime::driver::MapCudaError(err);
}

CUresult cuDriverGetVersion(int* driverVersion) {
  if (driverVersion == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  auto& rt = muda::runtime::cudart::RuntimeContext::Instance();
  return muda::runtime::driver::MapCudaError(rt.Devices().GetDriverVersion(driverVersion));
}

CUresult cuDeviceGet(CUdevice* device, int ordinal) {
  if (device == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (ordinal != 0) {
    return CUDA_ERROR_INVALID_DEVICE;
  }
  *device = 0;
  return CUDA_SUCCESS;
}

CUresult cuDeviceGetCount(int* count) {
  if (count == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  *count = 1;
  return CUDA_SUCCESS;
}

CUresult cuDeviceGetName(char* name, int len, CUdevice dev) {
  if (name == nullptr || len <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (dev != 0) {
    return CUDA_ERROR_INVALID_DEVICE;
  }
  auto& rt = muda::runtime::cudart::RuntimeContext::Instance();
  cudaDeviceProp prop{};
  const cudaError_t err = rt.Devices().GetDeviceProperties(&prop, dev);
  if (err != cudaSuccess) {
    return muda::runtime::driver::MapCudaError(err);
  }
  std::strncpy(name, prop.name, static_cast<std::size_t>(len) - 1);
  name[len - 1] = '\0';
  return CUDA_SUCCESS;
}

CUresult cuDeviceGetUuid(CUuuid* uuid, CUdevice dev) {
  if (uuid == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (dev != 0) {
    return CUDA_ERROR_INVALID_DEVICE;
  }
  // Deterministic 16-byte UUID for device 0
  static const char kDeviceUuid[16] = {'M', 'U', 'D', 'A', '-', 'G', 'P', 'U',
                                       '-', 'A', 'P', 'P', 'L', 'E', '0', '0'};
  std::memcpy(uuid->bytes, kDeviceUuid, sizeof(uuid->bytes));
  return CUDA_SUCCESS;
}

CUresult cuDeviceTotalMem(size_t* bytes, CUdevice dev) {
  if (bytes == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (dev != 0) {
    return CUDA_ERROR_INVALID_DEVICE;
  }
  auto& rt = muda::runtime::cudart::RuntimeContext::Instance();
  cudaDeviceProp prop{};
  const cudaError_t err = rt.Devices().GetDeviceProperties(&prop, dev);
  if (err != cudaSuccess) {
    return muda::runtime::driver::MapCudaError(err);
  }
  *bytes = prop.totalGlobalMem;
  return CUDA_SUCCESS;
}

CUresult cuDeviceTotalMem_v2(size_t* bytes, CUdevice dev) { return cuDeviceTotalMem(bytes, dev); }

CUresult cuDeviceGetAttribute(int* value, CUdevice_attribute attrib, CUdevice dev) {
  if (value == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (dev != 0) {
    return CUDA_ERROR_INVALID_DEVICE;
  }

  // Handle specific CUDA Driver attributes
  switch (attrib) {
    case CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID:
    case CU_DEVICE_ATTRIBUTE_PCI_BUS_ID:
    case CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID:
    case CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID:
    case CU_DEVICE_ATTRIBUTE_IS_MULTI_GPU_BOARD:
    case CU_DEVICE_ATTRIBUTE_ECC_ENABLED:
    case CU_DEVICE_ATTRIBUTE_TCC_DRIVER:
    case CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED:
    case CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH:
    case CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT:
    case CU_DEVICE_ATTRIBUTE_COMPUTE_MODE:
      *value = 0;
      return CUDA_SUCCESS;

    case CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK:
    case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X:
    case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y:
    case CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR:
      *value = 1024;
      return CUDA_SUCCESS;

    case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z:
      *value = 64;
      return CUDA_SUCCESS;

    case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X:
      *value = 2147483647;
      return CUDA_SUCCESS;

    case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y:
    case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z:
      *value = 65535;
      return CUDA_SUCCESS;

    case CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK:
      *value = 32768;
      return CUDA_SUCCESS;

    case CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN:
    case CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR:
      *value = 65536;
      return CUDA_SUCCESS;

    case CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY:
      *value = 65536;
      return CUDA_SUCCESS;

    case CU_DEVICE_ATTRIBUTE_WARP_SIZE:
      *value = 32;
      return CUDA_SUCCESS;

    case CU_DEVICE_ATTRIBUTE_MAX_PITCH:
      *value = 2147483647;
      return CUDA_SUCCESS;

    case CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK:
    case CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR:
      *value = 65536;
      return CUDA_SUCCESS;

    case CU_DEVICE_ATTRIBUTE_CLOCK_RATE:
      *value = 1390000;
      return CUDA_SUCCESS;

    case CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE:
      *value = 1000000;
      return CUDA_SUCCESS;

    case CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH:
      *value = 256;
      return CUDA_SUCCESS;

    case CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE:
      *value = 4194304;
      return CUDA_SUCCESS;

    case CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT:
      *value = 16;
      return CUDA_SUCCESS;

    case CU_DEVICE_ATTRIBUTE_GPU_OVERLAP:
    case CU_DEVICE_ATTRIBUTE_INTEGRATED:
    case CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY:
    case CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS:
    case CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING:
    case CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY:
    case CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS:
    case CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS:
    case CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED:
    case CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED:
    case CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED:
    case CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED:
    case CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH:
    case CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM:
      *value = 1;
      return CUDA_SUCCESS;

    case CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT:
    case CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO:
      *value = 2;
      return CUDA_SUCCESS;

    case CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT:
    case CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT:
      *value = 512;
      return CUDA_SUCCESS;

    case CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT:
      *value = 32;
      return CUDA_SUCCESS;

    case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR:
      *value = 8;
      return CUDA_SUCCESS;

    case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR:
      *value = 6;
      return CUDA_SUCCESS;

    default:
      *value = 0;
      return CUDA_SUCCESS;
  }
}

CUresult cuDeviceComputeCapability(int* major, int* minor, CUdevice dev) {
  if (major == nullptr || minor == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (dev != 0) {
    return CUDA_ERROR_INVALID_DEVICE;
  }
  *major = 8;
  *minor = 6;
  return CUDA_SUCCESS;
}

CUresult cuDeviceCanAccessPeer(int* canAccessPeer, CUdevice dev, CUdevice peerDev) {
  if (canAccessPeer == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  *canAccessPeer = (dev == peerDev) ? 1 : 0;
  return CUDA_SUCCESS;
}

/* Primary Context Management */
CUresult cuDevicePrimaryCtxRetain(CUcontext* pctx, CUdevice dev) {
  if (pctx == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (dev != 0) {
    return CUDA_ERROR_INVALID_DEVICE;
  }
  muda::runtime::driver::EnsureContext();
  muda::runtime::driver::g_primary_ctx_refcount.fetch_add(1, std::memory_order_relaxed);
  *pctx = &muda::runtime::driver::g_primary_ctx;
  return CUDA_SUCCESS;
}

CUresult cuDevicePrimaryCtxRelease(CUdevice dev) {
  if (dev != 0) {
    return CUDA_ERROR_INVALID_DEVICE;
  }
  int count = muda::runtime::driver::g_primary_ctx_refcount.load(std::memory_order_relaxed);
  if (count > 0) {
    muda::runtime::driver::g_primary_ctx_refcount.fetch_sub(1, std::memory_order_relaxed);
  }
  return CUDA_SUCCESS;
}

CUresult cuDevicePrimaryCtxReset(CUdevice dev) {
  if (dev != 0) {
    return CUDA_ERROR_INVALID_DEVICE;
  }
  auto& rt = muda::runtime::cudart::RuntimeContext::Instance();
  (void)rt.Reset();
  muda::runtime::driver::g_primary_ctx_refcount.store(0, std::memory_order_relaxed);
  return CUDA_SUCCESS;
}

CUresult cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int* flags, int* active) {
  if (dev != 0) {
    return CUDA_ERROR_INVALID_DEVICE;
  }
  if (flags != nullptr) {
    *flags = 0;
  }
  if (active != nullptr) {
    *active = (muda::runtime::driver::g_primary_ctx_refcount.load(std::memory_order_relaxed) > 0) ? 1 : 0;
  }
  return CUDA_SUCCESS;
}

CUresult cuDevicePrimaryCtxSetFlags(CUdevice dev, [[maybe_unused]] unsigned int flags) {
  if (dev != 0) {
    return CUDA_ERROR_INVALID_DEVICE;
  }
  return CUDA_SUCCESS;
}

/* Context Management */
CUresult cuCtxCreate(CUcontext* pctx, unsigned int flags, CUdevice dev) {
  if (pctx == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (dev != 0) {
    return CUDA_ERROR_INVALID_DEVICE;
  }
  auto& rt = muda::runtime::cudart::RuntimeContext::Instance();
  (void)rt.EnsureInitialized();

  auto ctx = new CUctx_st{dev, flags, rt.Device(), rt.DefaultQueue()};
  *pctx = ctx;
  muda::runtime::driver::g_current_ctx = ctx;
  return CUDA_SUCCESS;
}

CUresult cuCtxCreate_v2(CUcontext* pctx, unsigned int flags, CUdevice dev) {
  return cuCtxCreate(pctx, flags, dev);
}

CUresult cuCtxDestroy(CUcontext ctx) {
  if (ctx == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (muda::runtime::driver::g_current_ctx == ctx) {
    muda::runtime::driver::g_current_ctx = nullptr;
  }
  if (ctx != &muda::runtime::driver::g_primary_ctx) {
    delete ctx;
  }
  return CUDA_SUCCESS;
}

CUresult cuCtxDestroy_v2(CUcontext ctx) { return cuCtxDestroy(ctx); }

CUresult cuCtxPushCurrent(CUcontext ctx) {
  muda::runtime::driver::g_current_ctx = ctx;
  return CUDA_SUCCESS;
}

CUresult cuCtxPushCurrent_v2(CUcontext ctx) { return cuCtxPushCurrent(ctx); }

CUresult cuCtxPopCurrent(CUcontext* pctx) {
  if (pctx != nullptr) {
    *pctx = muda::runtime::driver::g_current_ctx;
  }
  muda::runtime::driver::g_current_ctx = nullptr;
  return CUDA_SUCCESS;
}

CUresult cuCtxPopCurrent_v2(CUcontext* pctx) { return cuCtxPopCurrent(pctx); }

CUresult cuCtxSetCurrent(CUcontext ctx) {
  muda::runtime::driver::g_current_ctx = ctx;
  return CUDA_SUCCESS;
}

CUresult cuCtxGetCurrent(CUcontext* pctx) {
  if (pctx == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  *pctx = muda::runtime::driver::g_current_ctx;
  return CUDA_SUCCESS;
}

CUresult cuCtxGetCurrent_v2(CUcontext* pctx) { return cuCtxGetCurrent(pctx); }

CUresult cuCtxGetDevice(CUdevice* device) {
  if (device == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  *device = muda::runtime::driver::g_current_ctx ? muda::runtime::driver::g_current_ctx->device_ordinal : 0;
  return CUDA_SUCCESS;
}

CUresult cuCtxSynchronize(void) {
  auto& rt = muda::runtime::cudart::RuntimeContext::Instance();
  return muda::runtime::driver::MapCudaError(rt.Streams().SynchronizeDevice(rt.DefaultQueue()));
}

CUresult cuCtxEnablePeerAccess([[maybe_unused]] CUcontext peerContext,
                               [[maybe_unused]] unsigned int Flags) {
  return CUDA_SUCCESS;
}

CUresult cuCtxDisablePeerAccess([[maybe_unused]] CUcontext peerContext) { return CUDA_SUCCESS; }

CUresult cuCtxGetFlags(unsigned int* flags) {
  if (flags == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  *flags = muda::runtime::driver::g_current_ctx ? muda::runtime::driver::g_current_ctx->flags : 0;
  return CUDA_SUCCESS;
}

CUresult cuCtxGetApiVersion([[maybe_unused]] CUcontext ctx, unsigned int* version) {
  if (version == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  *version = 12000;
  return CUDA_SUCCESS;
}

/* Memory Management */

CUresult cuGetErrorString(CUresult error, const char** pStr) {
  if (pStr == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  switch (error) {
    case CUDA_SUCCESS:
      *pStr = "CUDA_SUCCESS";
      break;
    case CUDA_ERROR_INVALID_VALUE:
      *pStr = "CUDA_ERROR_INVALID_VALUE";
      break;
    case CUDA_ERROR_OUT_OF_MEMORY:
      *pStr = "CUDA_ERROR_OUT_OF_MEMORY";
      break;
    case CUDA_ERROR_NOT_INITIALIZED:
      *pStr = "CUDA_ERROR_NOT_INITIALIZED";
      break;
    case CUDA_ERROR_NO_DEVICE:
      *pStr = "CUDA_ERROR_NO_DEVICE";
      break;
    case CUDA_ERROR_INVALID_DEVICE:
      *pStr = "CUDA_ERROR_INVALID_DEVICE";
      break;
    case CUDA_ERROR_INVALID_IMAGE:
      *pStr = "CUDA_ERROR_INVALID_IMAGE";
      break;
    case CUDA_ERROR_INVALID_CONTEXT:
      *pStr = "CUDA_ERROR_INVALID_CONTEXT";
      break;
    case CUDA_ERROR_INVALID_PTX:
      *pStr = "CUDA_ERROR_INVALID_PTX";
      break;
    case CUDA_ERROR_NOT_FOUND:
      *pStr = "CUDA_ERROR_NOT_FOUND";
      break;
    case CUDA_ERROR_NOT_READY:
      *pStr = "CUDA_ERROR_NOT_READY";
      break;
    case CUDA_ERROR_LAUNCH_FAILED:
      *pStr = "CUDA_ERROR_LAUNCH_FAILED";
      break;
    case CUDA_ERROR_NOT_SUPPORTED:
      *pStr = "CUDA_ERROR_NOT_SUPPORTED";
      break;
    default:
      *pStr = "CUDA_ERROR_UNKNOWN";
      break;
  }
  return CUDA_SUCCESS;
}

CUresult cuGetErrorName(CUresult error, const char** pStr) { return cuGetErrorString(error, pStr); }

}  // extern "C"

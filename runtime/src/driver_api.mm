/**
 * @file driver_api.mm
 * @brief Implementation of the clean-room CUDA Driver C API over MUDA runtime subsystems.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_manager.hpp"
#include "error_state.hpp"
#include "event_manager.hpp"
#include "launch_engine.hpp"
#include "memory_manager.hpp"
#include "muda/runtime/driver_types.hpp"
#include "muda/runtime/module_manager.hpp"
#include "runtime_context.hpp"
#include "stream_manager.hpp"
#include "transfer_manager.hpp"

struct CUlinkState_st {
  std::string accumulated_ptx;
};

namespace {

thread_local CUcontext g_current_ctx = nullptr;
static std::atomic<int> s_primary_ctx_refcount{0};
static CUctx_st s_primary_ctx{0, 0, nil, nil};

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
  s_primary_ctx.device = rt.Device();
  s_primary_ctx.queue = rt.DefaultQueue();
  g_current_ctx = &s_primary_ctx;
  return g_current_ctx;
}

}  // namespace

extern "C" {

CUresult cuInit([[maybe_unused]] unsigned int Flags) {
  auto& rt = muda::runtime::cudart::RuntimeContext::Instance();
  const cudaError_t err = rt.EnsureInitialized();
  return MapCudaError(err);
}

CUresult cuDriverGetVersion(int* driverVersion) {
  if (driverVersion == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  auto& rt = muda::runtime::cudart::RuntimeContext::Instance();
  return MapCudaError(rt.Devices().GetDriverVersion(driverVersion));
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
    return MapCudaError(err);
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
    return MapCudaError(err);
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
  EnsureContext();
  s_primary_ctx_refcount.fetch_add(1, std::memory_order_relaxed);
  *pctx = &s_primary_ctx;
  return CUDA_SUCCESS;
}

CUresult cuDevicePrimaryCtxRelease(CUdevice dev) {
  if (dev != 0) {
    return CUDA_ERROR_INVALID_DEVICE;
  }
  int count = s_primary_ctx_refcount.load(std::memory_order_relaxed);
  if (count > 0) {
    s_primary_ctx_refcount.fetch_sub(1, std::memory_order_relaxed);
  }
  return CUDA_SUCCESS;
}

CUresult cuDevicePrimaryCtxReset(CUdevice dev) {
  if (dev != 0) {
    return CUDA_ERROR_INVALID_DEVICE;
  }
  auto& rt = muda::runtime::cudart::RuntimeContext::Instance();
  (void)rt.Reset();
  s_primary_ctx_refcount.store(0, std::memory_order_relaxed);
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
    *active = (s_primary_ctx_refcount.load(std::memory_order_relaxed) > 0) ? 1 : 0;
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
  g_current_ctx = ctx;
  return CUDA_SUCCESS;
}

CUresult cuCtxCreate_v2(CUcontext* pctx, unsigned int flags, CUdevice dev) {
  return cuCtxCreate(pctx, flags, dev);
}

CUresult cuCtxDestroy(CUcontext ctx) {
  if (ctx == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (g_current_ctx == ctx) {
    g_current_ctx = nullptr;
  }
  if (ctx != &s_primary_ctx) {
    delete ctx;
  }
  return CUDA_SUCCESS;
}

CUresult cuCtxDestroy_v2(CUcontext ctx) { return cuCtxDestroy(ctx); }

CUresult cuCtxPushCurrent(CUcontext ctx) {
  g_current_ctx = ctx;
  return CUDA_SUCCESS;
}

CUresult cuCtxPushCurrent_v2(CUcontext ctx) { return cuCtxPushCurrent(ctx); }

CUresult cuCtxPopCurrent(CUcontext* pctx) {
  if (pctx != nullptr) {
    *pctx = g_current_ctx;
  }
  g_current_ctx = nullptr;
  return CUDA_SUCCESS;
}

CUresult cuCtxPopCurrent_v2(CUcontext* pctx) { return cuCtxPopCurrent(pctx); }

CUresult cuCtxSetCurrent(CUcontext ctx) {
  g_current_ctx = ctx;
  return CUDA_SUCCESS;
}

CUresult cuCtxGetCurrent(CUcontext* pctx) {
  if (pctx == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  *pctx = g_current_ctx;
  return CUDA_SUCCESS;
}

CUresult cuCtxGetCurrent_v2(CUcontext* pctx) { return cuCtxGetCurrent(pctx); }

CUresult cuCtxGetDevice(CUdevice* device) {
  if (device == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  *device = g_current_ctx ? g_current_ctx->device_ordinal : 0;
  return CUDA_SUCCESS;
}

CUresult cuCtxSynchronize(void) {
  auto& rt = muda::runtime::cudart::RuntimeContext::Instance();
  return MapCudaError(rt.Streams().SynchronizeDevice(rt.DefaultQueue()));
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
  *flags = g_current_ctx ? g_current_ctx->flags : 0;
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
CUresult cuMemAlloc(CUdeviceptr* dptr, size_t bytesize) {
  if (dptr == nullptr || bytesize == 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  EnsureContext();
  auto& rt = muda::runtime::cudart::RuntimeContext::Instance();
  void* ptr = nullptr;
  const cudaError_t err =
      rt.Memory().Allocate(&ptr, bytesize, cudaMemAttachGlobal, cudaMemoryTypeDevice, rt.Device());
  if (err != cudaSuccess) {
    return MapCudaError(err);
  }
  *dptr = reinterpret_cast<uintptr_t>(ptr);
  return CUDA_SUCCESS;
}

CUresult cuMemAlloc_v2(CUdeviceptr* dptr, size_t bytesize) { return cuMemAlloc(dptr, bytesize); }

CUresult cuMemAllocPitch(CUdeviceptr* dptr, size_t* pPitch, size_t WidthInBytes, size_t Height,
                         [[maybe_unused]] unsigned int ElementSizeBytes) {
  if (dptr == nullptr || pPitch == nullptr || WidthInBytes == 0 || Height == 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  EnsureContext();
  auto& rt = muda::runtime::cudart::RuntimeContext::Instance();
  void* ptr = nullptr;
  const cudaError_t err =
      rt.Memory().AllocatePitch(&ptr, pPitch, WidthInBytes, Height, rt.Device());
  if (err != cudaSuccess) {
    return MapCudaError(err);
  }
  *dptr = reinterpret_cast<uintptr_t>(ptr);
  return CUDA_SUCCESS;
}

CUresult cuMemAllocPitch_v2(CUdeviceptr* dptr, size_t* pPitch, size_t WidthInBytes, size_t Height,
                            unsigned int ElementSizeBytes) {
  return cuMemAllocPitch(dptr, pPitch, WidthInBytes, Height, ElementSizeBytes);
}

CUresult cuMemAllocManaged(CUdeviceptr* dptr, size_t bytesize,
                           [[maybe_unused]] unsigned int flags) {
  if (dptr == nullptr || bytesize == 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  EnsureContext();
  auto& rt = muda::runtime::cudart::RuntimeContext::Instance();
  void* ptr = nullptr;
  const cudaError_t err =
      rt.Memory().Allocate(&ptr, bytesize, cudaMemAttachGlobal, cudaMemoryTypeManaged, rt.Device());
  if (err != cudaSuccess) {
    return MapCudaError(err);
  }
  *dptr = reinterpret_cast<uintptr_t>(ptr);
  return CUDA_SUCCESS;
}

CUresult cuMemFree(CUdeviceptr dptr) {
  if (dptr == 0) {
    return CUDA_SUCCESS;
  }
  auto& rt = muda::runtime::cudart::RuntimeContext::Instance();
  void* ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(dptr));
  return MapCudaError(rt.Memory().Free(ptr, rt.Streams(), rt.DefaultQueue()));
}

CUresult cuMemFree_v2(CUdeviceptr dptr) { return cuMemFree(dptr); }

CUresult cuMemAllocHost(void** pp, size_t bytesize) {
  if (pp == nullptr || bytesize == 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  auto& rt = muda::runtime::cudart::RuntimeContext::Instance();
  return MapCudaError(rt.Memory().AllocateHost(pp, bytesize, cudaHostAllocDefault, rt.Device()));
}

CUresult cuMemAllocHost_v2(void** pp, size_t bytesize) { return cuMemAllocHost(pp, bytesize); }

CUresult cuMemFreeHost(void* p) {
  if (p == nullptr) {
    return CUDA_SUCCESS;
  }
  auto& rt = muda::runtime::cudart::RuntimeContext::Instance();
  return MapCudaError(rt.Memory().FreeHost(p, rt.Streams(), rt.DefaultQueue()));
}

CUresult cuMemHostAlloc(void** pp, size_t bytesize, unsigned int Flags) {
  if (pp == nullptr || bytesize == 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  auto& rt = muda::runtime::cudart::RuntimeContext::Instance();
  return MapCudaError(rt.Memory().AllocateHost(pp, bytesize, Flags, rt.Device()));
}

CUresult cuMemHostRegister(void* p, size_t bytesize, [[maybe_unused]] unsigned int Flags) {
  if (p == nullptr || bytesize == 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  return CUDA_SUCCESS;
}

CUresult cuMemHostRegister_v2(void* p, size_t bytesize, unsigned int Flags) {
  return cuMemHostRegister(p, bytesize, Flags);
}

CUresult cuMemHostUnregister(void* p) {
  if (p == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  return CUDA_SUCCESS;
}

CUresult cuMemHostGetDevicePointer(CUdeviceptr* pdptr, void* p,
                                   [[maybe_unused]] unsigned int Flags) {
  if (pdptr == nullptr || p == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  *pdptr = reinterpret_cast<uintptr_t>(p);
  return CUDA_SUCCESS;
}

CUresult cuMemHostGetDevicePointer_v2(CUdeviceptr* pdptr, void* p, unsigned int Flags) {
  return cuMemHostGetDevicePointer(pdptr, p, Flags);
}

CUresult cuMemHostGetFlags(unsigned int* pFlags, void* p) {
  if (pFlags == nullptr || p == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  *pFlags = 0;
  return CUDA_SUCCESS;
}

CUresult cuMemGetInfo(size_t* free, size_t* total) {
  if (free == nullptr || total == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  auto& rt = muda::runtime::cudart::RuntimeContext::Instance();
  return MapCudaError(rt.Memory().MemGetInfo(free, total, rt.Device()));
}

CUresult cuMemGetInfo_v2(size_t* free, size_t* total) { return cuMemGetInfo(free, total); }

CUresult cuMemGetAddressRange(CUdeviceptr* pbase, size_t* psize, CUdeviceptr dptr) {
  if (dptr == 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  auto& rt = muda::runtime::cudart::RuntimeContext::Instance();
  const void* ptr = reinterpret_cast<const void*>(static_cast<uintptr_t>(dptr));
  auto rec = rt.Memory().SnapshotRecord(muda::runtime::cudart::DeviceAddress::FromPointer(ptr));
  if (!rec.has_value()) {
    return CUDA_ERROR_NOT_FOUND;
  }
  if (pbase != nullptr) {
    *pbase = rec->base_address.value;
  }
  if (psize != nullptr) {
    *psize = rec->size;
  }
  return CUDA_SUCCESS;
}

CUresult cuMemGetAddressRange_v2(CUdeviceptr* pbase, size_t* psize, CUdeviceptr dptr) {
  return cuMemGetAddressRange(pbase, psize, dptr);
}

CUresult cuPointerGetAttribute(void* data, CUpointer_attribute attribute, CUdeviceptr ptr) {
  if (data == nullptr || ptr == 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  auto& rt = muda::runtime::cudart::RuntimeContext::Instance();
  const void* raw_ptr = reinterpret_cast<const void*>(static_cast<uintptr_t>(ptr));
  auto rec = rt.Memory().SnapshotRecord(muda::runtime::cudart::DeviceAddress::FromPointer(raw_ptr));
  if (!rec.has_value()) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  switch (attribute) {
    case CU_POINTER_ATTRIBUTE_CONTEXT:
      *static_cast<CUcontext*>(data) = EnsureContext();
      return CUDA_SUCCESS;
    case CU_POINTER_ATTRIBUTE_MEMORY_TYPE:
      *static_cast<unsigned int*>(data) = static_cast<unsigned int>(rec->memory_type);
      return CUDA_SUCCESS;
    case CU_POINTER_ATTRIBUTE_DEVICE_POINTER:
      *static_cast<CUdeviceptr*>(data) = ptr;
      return CUDA_SUCCESS;
    case CU_POINTER_ATTRIBUTE_HOST_POINTER:
      *static_cast<void**>(data) = const_cast<void*>(raw_ptr);
      return CUDA_SUCCESS;
    case CU_POINTER_ATTRIBUTE_IS_MANAGED:
      *static_cast<unsigned int*>(data) = (rec->memory_type == cudaMemoryTypeManaged) ? 1 : 0;
      return CUDA_SUCCESS;
    case CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL:
      *static_cast<int*>(data) = 0;
      return CUDA_SUCCESS;
    case CU_POINTER_ATTRIBUTE_RANGE_START_ADDR:
      *static_cast<CUdeviceptr*>(data) = rec->base_address.value;
      return CUDA_SUCCESS;
    case CU_POINTER_ATTRIBUTE_RANGE_SIZE:
      *static_cast<size_t*>(data) = rec->size;
      return CUDA_SUCCESS;
    default:
      return CUDA_SUCCESS;
  }
}

CUresult cuPointerGetAttributes(unsigned int numAttributes, CUpointer_attribute* attributes,
                                void** data, CUdeviceptr ptr) {
  if (attributes == nullptr || data == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  for (unsigned int i = 0; i < numAttributes; ++i) {
    const CUresult res = cuPointerGetAttribute(data[i], attributes[i], ptr);
    if (res != CUDA_SUCCESS) {
      return res;
    }
  }
  return CUDA_SUCCESS;
}

CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount) {
  if (ByteCount == 0)
    return CUDA_SUCCESS;
  if (dstDevice == 0 || srcHost == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  auto& rt = muda::runtime::cudart::RuntimeContext::Instance();
  void* dst = reinterpret_cast<void*>(static_cast<uintptr_t>(dstDevice));
  return MapCudaError(rt.Transfers().Memcpy(dst, srcHost, ByteCount, cudaMemcpyHostToDevice,
                                            rt.Memory(), rt.Streams(), rt.DefaultQueue()));
}

CUresult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount) {
  return cuMemcpyHtoD(dstDevice, srcHost, ByteCount);
}

CUresult cuMemcpyDtoH(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount) {
  if (ByteCount == 0)
    return CUDA_SUCCESS;
  if (dstHost == nullptr || srcDevice == 0)
    return CUDA_ERROR_INVALID_VALUE;
  auto& rt = muda::runtime::cudart::RuntimeContext::Instance();
  const void* src = reinterpret_cast<const void*>(static_cast<uintptr_t>(srcDevice));
  return MapCudaError(rt.Transfers().Memcpy(dstHost, src, ByteCount, cudaMemcpyDeviceToHost,
                                            rt.Memory(), rt.Streams(), rt.DefaultQueue()));
}

CUresult cuMemcpyDtoH_v2(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount) {
  return cuMemcpyDtoH(dstHost, srcDevice, ByteCount);
}

CUresult cuMemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount) {
  if (ByteCount == 0)
    return CUDA_SUCCESS;
  if (dstDevice == 0 || srcDevice == 0)
    return CUDA_ERROR_INVALID_VALUE;
  auto& rt = muda::runtime::cudart::RuntimeContext::Instance();
  void* dst = reinterpret_cast<void*>(static_cast<uintptr_t>(dstDevice));
  const void* src = reinterpret_cast<const void*>(static_cast<uintptr_t>(srcDevice));
  return MapCudaError(rt.Transfers().Memcpy(dst, src, ByteCount, cudaMemcpyDeviceToDevice,
                                            rt.Memory(), rt.Streams(), rt.DefaultQueue()));
}

CUresult cuMemcpyDtoD_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount) {
  return cuMemcpyDtoD(dstDevice, srcDevice, ByteCount);
}

CUresult cuMemcpyHtoDAsync(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount,
                           CUstream hStream) {
  if (ByteCount == 0)
    return CUDA_SUCCESS;
  if (dstDevice == 0 || srcHost == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  auto& rt = muda::runtime::cudart::RuntimeContext::Instance();
  void* dst = reinterpret_cast<void*>(static_cast<uintptr_t>(dstDevice));
  return MapCudaError(rt.Transfers().MemcpyAsync(dst, srcHost, ByteCount, cudaMemcpyHostToDevice,
                                                 hStream, rt.Memory(), rt.Streams(),
                                                 rt.DefaultQueue()));
}

CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount,
                              CUstream hStream) {
  return cuMemcpyHtoDAsync(dstDevice, srcHost, ByteCount, hStream);
}

CUresult cuMemcpyDtoHAsync(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount,
                           CUstream hStream) {
  if (ByteCount == 0)
    return CUDA_SUCCESS;
  if (dstHost == nullptr || srcDevice == 0)
    return CUDA_ERROR_INVALID_VALUE;
  auto& rt = muda::runtime::cudart::RuntimeContext::Instance();
  const void* src = reinterpret_cast<const void*>(static_cast<uintptr_t>(srcDevice));
  return MapCudaError(rt.Transfers().MemcpyAsync(dstHost, src, ByteCount, cudaMemcpyDeviceToHost,
                                                 hStream, rt.Memory(), rt.Streams(),
                                                 rt.DefaultQueue()));
}

CUresult cuMemcpyDtoHAsync_v2(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount,
                              CUstream hStream) {
  return cuMemcpyDtoHAsync(dstHost, srcDevice, ByteCount, hStream);
}

CUresult cuMemcpyDtoDAsync(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount,
                           CUstream hStream) {
  if (ByteCount == 0)
    return CUDA_SUCCESS;
  if (dstDevice == 0 || srcDevice == 0)
    return CUDA_ERROR_INVALID_VALUE;
  auto& rt = muda::runtime::cudart::RuntimeContext::Instance();
  void* dst = reinterpret_cast<void*>(static_cast<uintptr_t>(dstDevice));
  const void* src = reinterpret_cast<const void*>(static_cast<uintptr_t>(srcDevice));
  return MapCudaError(rt.Transfers().MemcpyAsync(dst, src, ByteCount, cudaMemcpyDeviceToDevice,
                                                 hStream, rt.Memory(), rt.Streams(),
                                                 rt.DefaultQueue()));
}

CUresult cuMemcpyDtoDAsync_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount,
                              CUstream hStream) {
  return cuMemcpyDtoDAsync(dstDevice, srcDevice, ByteCount, hStream);
}

CUresult cuMemcpyAsync(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount,
                       CUstream hStream) {
  return cuMemcpyDtoDAsync(dstDevice, srcDevice, ByteCount, hStream);
}

CUresult cuMemcpy2D(const CUDA_MEMCPY2D* pCopy) {
  if (pCopy == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  auto& rt = muda::runtime::cudart::RuntimeContext::Instance();
  const char* src_base =
      pCopy->srcHost ? static_cast<const char*>(pCopy->srcHost)
                     : reinterpret_cast<const char*>(static_cast<uintptr_t>(pCopy->srcDevice));
  char* dst_base = pCopy->dstHost
                       ? static_cast<char*>(pCopy->dstHost)
                       : reinterpret_cast<char*>(static_cast<uintptr_t>(pCopy->dstDevice));
  if (src_base == nullptr || dst_base == nullptr)
    return CUDA_ERROR_INVALID_VALUE;

  const void* src = src_base + pCopy->srcY * pCopy->srcPitch + pCopy->srcXInBytes;
  void* dst = dst_base + pCopy->dstY * pCopy->dstPitch + pCopy->dstXInBytes;

  return MapCudaError(rt.Transfers().Memcpy2D(dst, pCopy->dstPitch, src, pCopy->srcPitch,
                                              pCopy->WidthInBytes, pCopy->Height, cudaMemcpyDefault,
                                              rt.Memory(), rt.Streams(), rt.DefaultQueue()));
}

CUresult cuMemcpy2D_v2(const CUDA_MEMCPY2D* pCopy) { return cuMemcpy2D(pCopy); }

CUresult cuMemcpy2DAsync(const CUDA_MEMCPY2D* pCopy, CUstream hStream) {
  if (pCopy == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  auto& rt = muda::runtime::cudart::RuntimeContext::Instance();
  const char* src_base =
      pCopy->srcHost ? static_cast<const char*>(pCopy->srcHost)
                     : reinterpret_cast<const char*>(static_cast<uintptr_t>(pCopy->srcDevice));
  char* dst_base = pCopy->dstHost
                       ? static_cast<char*>(pCopy->dstHost)
                       : reinterpret_cast<char*>(static_cast<uintptr_t>(pCopy->dstDevice));
  if (src_base == nullptr || dst_base == nullptr)
    return CUDA_ERROR_INVALID_VALUE;

  const void* src = src_base + pCopy->srcY * pCopy->srcPitch + pCopy->srcXInBytes;
  void* dst = dst_base + pCopy->dstY * pCopy->dstPitch + pCopy->dstXInBytes;

  return MapCudaError(rt.Transfers().Memcpy2DAsync(
      dst, pCopy->dstPitch, src, pCopy->srcPitch, pCopy->WidthInBytes, pCopy->Height,
      cudaMemcpyDefault, hStream, rt.Memory(), rt.Streams(), rt.DefaultQueue()));
}

CUresult cuMemcpy2DAsync_v2(const CUDA_MEMCPY2D* pCopy, CUstream hStream) {
  return cuMemcpy2DAsync(pCopy, hStream);
}

CUresult cuMemsetD8(CUdeviceptr dstDevice, unsigned char uc, size_t N) {
  if (N == 0)
    return CUDA_SUCCESS;
  if (dstDevice == 0)
    return CUDA_ERROR_INVALID_VALUE;
  auto& rt = muda::runtime::cudart::RuntimeContext::Instance();
  void* dst = reinterpret_cast<void*>(static_cast<uintptr_t>(dstDevice));
  return MapCudaError(rt.Transfers().Memset(dst, static_cast<int>(uc), N, rt.Memory(), rt.Streams(),
                                            rt.DefaultQueue()));
}

CUresult cuMemsetD8_v2(CUdeviceptr dstDevice, unsigned char uc, size_t N) {
  return cuMemsetD8(dstDevice, uc, N);
}

CUresult cuMemsetD16(CUdeviceptr dstDevice, unsigned short us, size_t N) {
  if (N == 0)
    return CUDA_SUCCESS;
  if (dstDevice == 0)
    return CUDA_ERROR_INVALID_VALUE;
  std::vector<unsigned short> host_buf(N, us);
  return cuMemcpyHtoD(dstDevice, host_buf.data(), N * sizeof(unsigned short));
}

CUresult cuMemsetD16_v2(CUdeviceptr dstDevice, unsigned short us, size_t N) {
  return cuMemsetD16(dstDevice, us, N);
}

CUresult cuMemsetD32(CUdeviceptr dstDevice, unsigned int ui, size_t N) {
  if (N == 0)
    return CUDA_SUCCESS;
  if (dstDevice == 0)
    return CUDA_ERROR_INVALID_VALUE;
  std::vector<unsigned int> host_buf(N, ui);
  return cuMemcpyHtoD(dstDevice, host_buf.data(), N * sizeof(unsigned int));
}

CUresult cuMemsetD32_v2(CUdeviceptr dstDevice, unsigned int ui, size_t N) {
  return cuMemsetD32(dstDevice, ui, N);
}

CUresult cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream hStream) {
  if (N == 0)
    return CUDA_SUCCESS;
  if (dstDevice == 0)
    return CUDA_ERROR_INVALID_VALUE;
  auto& rt = muda::runtime::cudart::RuntimeContext::Instance();
  void* dst = reinterpret_cast<void*>(static_cast<uintptr_t>(dstDevice));
  return MapCudaError(rt.Transfers().MemsetAsync(dst, static_cast<int>(uc), N, hStream, rt.Memory(),
                                                 rt.Streams(), rt.DefaultQueue()));
}

CUresult cuMemsetD32Async(CUdeviceptr dstDevice, unsigned int ui, size_t N, CUstream hStream) {
  if (N == 0)
    return CUDA_SUCCESS;
  if (dstDevice == 0)
    return CUDA_ERROR_INVALID_VALUE;
  std::vector<unsigned int> host_buf(N, ui);
  return cuMemcpyHtoDAsync(dstDevice, host_buf.data(), N * sizeof(unsigned int), hStream);
}

/* Stream Management */
CUresult cuStreamCreate(CUstream* phStream, unsigned int Flags) {
  if (phStream == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  auto& rt = muda::runtime::cudart::RuntimeContext::Instance();
  return MapCudaError(rt.Streams().StreamCreate(phStream, Flags, rt.Device()));
}

CUresult cuStreamCreateWithPriority(CUstream* phStream, unsigned int flags,
                                    [[maybe_unused]] int priority) {
  if (phStream == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  auto& rt = muda::runtime::cudart::RuntimeContext::Instance();
  return MapCudaError(rt.Streams().StreamCreate(phStream, flags, rt.Device()));
}

CUresult cuStreamDestroy(CUstream hStream) {
  if (hStream == nullptr)
    return CUDA_SUCCESS;
  auto& rt = muda::runtime::cudart::RuntimeContext::Instance();
  return MapCudaError(rt.Streams().StreamDestroy(hStream));
}

CUresult cuStreamDestroy_v2(CUstream hStream) { return cuStreamDestroy(hStream); }

CUresult cuStreamSynchronize(CUstream hStream) {
  auto& rt = muda::runtime::cudart::RuntimeContext::Instance();
  return MapCudaError(rt.Streams().StreamSynchronize(hStream, rt.DefaultQueue()));
}

CUresult cuStreamQuery(CUstream hStream) {
  auto& rt = muda::runtime::cudart::RuntimeContext::Instance();
  return MapCudaError(rt.Streams().StreamQuery(hStream));
}

CUresult cuStreamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int Flags) {
  if (hEvent == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  auto& rt = muda::runtime::cudart::RuntimeContext::Instance();
  return MapCudaError(
      rt.Events().StreamWaitEvent(hStream, hEvent, Flags, rt.Streams(), rt.DefaultQueue()));
}

CUresult cuStreamAddCallback(CUstream hStream, CUstreamCallback callback, void* userData,
                             [[maybe_unused]] unsigned int flags) {
  if (callback == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  auto& rt = muda::runtime::cudart::RuntimeContext::Instance();
  (void)rt.Streams().StreamSynchronize(hStream, rt.DefaultQueue());
  callback(hStream, CUDA_SUCCESS, userData);
  return CUDA_SUCCESS;
}

/* Event Management */
CUresult cuEventCreate(CUevent* phEvent, unsigned int Flags) {
  if (phEvent == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  auto& rt = muda::runtime::cudart::RuntimeContext::Instance();
  return MapCudaError(rt.Events().EventCreate(phEvent, Flags, rt.Device()));
}

CUresult cuEventDestroy(CUevent hEvent) {
  if (hEvent == nullptr)
    return CUDA_SUCCESS;
  auto& rt = muda::runtime::cudart::RuntimeContext::Instance();
  return MapCudaError(rt.Events().EventDestroy(hEvent));
}

CUresult cuEventDestroy_v2(CUevent hEvent) { return cuEventDestroy(hEvent); }

CUresult cuEventRecord(CUevent hEvent, CUstream hStream) {
  if (hEvent == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  auto& rt = muda::runtime::cudart::RuntimeContext::Instance();
  return MapCudaError(rt.Events().EventRecord(hEvent, hStream, rt.Streams(), rt.DefaultQueue()));
}

CUresult cuEventSynchronize(CUevent hEvent) {
  if (hEvent == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  auto& rt = muda::runtime::cudart::RuntimeContext::Instance();
  return MapCudaError(rt.Events().EventSynchronize(hEvent, rt.Streams()));
}

CUresult cuEventQuery(CUevent hEvent) {
  if (hEvent == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  auto& rt = muda::runtime::cudart::RuntimeContext::Instance();
  return MapCudaError(rt.Events().EventQuery(hEvent));
}

CUresult cuEventElapsedTime(float* pMilliseconds, CUevent hStart, CUevent hEnd) {
  if (pMilliseconds == nullptr || hStart == nullptr || hEnd == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  auto& rt = muda::runtime::cudart::RuntimeContext::Instance();
  return MapCudaError(rt.Events().EventElapsedTime(pMilliseconds, hStart, hEnd, rt.Streams()));
}

/* Module & Function Management */
CUresult cuModuleLoad(CUmodule* module, const char* fname) {
  if (module == nullptr || fname == nullptr || fname[0] == '\0') {
    return CUDA_ERROR_INVALID_VALUE;
  }
  std::ifstream f(fname);
  if (!f.is_open()) {
    return CUDA_ERROR_FILE_NOT_FOUND;
  }
  std::stringstream ss;
  ss << f.rdbuf();
  const std::string content = ss.str();
  return cuModuleLoadData(module, content.c_str());
}

CUresult cuModuleLoadData(CUmodule* module, const void* image) {
  if (module == nullptr || image == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  EnsureContext();
  auto& rt = muda::runtime::cudart::RuntimeContext::Instance();

  const char* str = static_cast<const char*>(image);
  // Check if image is a null-terminated PTX string
  if (std::strstr(str, ".version") != nullptr || std::strstr(str, ".target") != nullptr ||
      std::strstr(str, ".entry") != nullptr) {
    return muda::runtime::driver::ModuleManager::LoadPtx(str, module, rt.Device(), rt.Launcher());
  }

  // Treat as binary metallib data
  return muda::runtime::driver::ModuleManager::LoadMetallibData(image, 4096, module, rt.Device(),
                                                                rt.Launcher());
}

CUresult cuModuleLoadDataEx(CUmodule* module, const void* image,
                            [[maybe_unused]] unsigned int numOptions,
                            [[maybe_unused]] CUjit_option* options,
                            [[maybe_unused]] void** optionValues) {
  return cuModuleLoadData(module, image);
}

CUresult cuModuleUnload(CUmodule hmod) {
  return muda::runtime::driver::ModuleManager::Unload(hmod);
}

CUresult cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, const char* name) {
  return muda::runtime::driver::ModuleManager::GetFunction(hfunc, hmod, name);
}

CUresult cuModuleGetGlobal([[maybe_unused]] CUdeviceptr* dptr, [[maybe_unused]] size_t* bytes,
                           [[maybe_unused]] CUmodule hmod, [[maybe_unused]] const char* name) {
  return CUDA_ERROR_NOT_FOUND;
}

CUresult cuModuleGetGlobal_v2(CUdeviceptr* dptr, size_t* bytes, CUmodule hmod, const char* name) {
  return cuModuleGetGlobal(dptr, bytes, hmod, name);
}

CUresult cuFuncGetAttribute(int* value, CUfunction_attribute attrib, CUfunction hfunc) {
  if (value == nullptr || hfunc == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (attrib == CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK) {
    *value = static_cast<int>(hfunc->pipeline.maxTotalThreadsPerThreadgroup);
    return CUDA_SUCCESS;
  }
  if (attrib == CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES) {
    *value = static_cast<int>(hfunc->static_shared_bytes);
    return CUDA_SUCCESS;
  }
  if (attrib == CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES) {
    *value = static_cast<int>(hfunc->pipeline.device.maxThreadgroupMemoryLength);
    return CUDA_SUCCESS;
  }
  return CUDA_SUCCESS;
}

CUresult cuFuncSetCacheConfig([[maybe_unused]] CUfunction hfunc,
                              [[maybe_unused]] CUfunc_cache config) {
  return CUDA_SUCCESS;
}

CUresult cuFuncSetSharedMemConfig([[maybe_unused]] CUfunction hfunc,
                                  [[maybe_unused]] CUsharedconfig config) {
  return CUDA_SUCCESS;
}

/* Occupancy Calculation */
CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, CUfunction func, int blockSize,
                                                     [[maybe_unused]] size_t dynamicSMemSize) {
  if (numBlocks == nullptr || func == nullptr || blockSize <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const int maxThreads = static_cast<int>(func->pipeline.maxTotalThreadsPerThreadgroup);
  *numBlocks = std::max(1, maxThreads / blockSize);
  return CUDA_SUCCESS;
}

CUresult cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* numBlocks, CUfunction func,
                                                              int blockSize, size_t dynamicSMemSize,
                                                              [[maybe_unused]] unsigned int flags) {
  return cuOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, func, blockSize, dynamicSMemSize);
}

CUresult cuOccupancyMaxPotentialBlockSize(
    int* minGridSize, int* blockSize, [[maybe_unused]] CUfunction func,
    [[maybe_unused]] CUoccupancyB2DSize blockSizeToDynamicSMemSize,
    [[maybe_unused]] size_t dynamicSMemSize, int blockSizeLimit) {
  if (minGridSize == nullptr || blockSize == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  int recommendedBlock = 256;
  if (blockSizeLimit > 0 && recommendedBlock > blockSizeLimit) {
    recommendedBlock = blockSizeLimit;
  }
  *blockSize = recommendedBlock;
  *minGridSize = 32;
  return CUDA_SUCCESS;
}

CUresult cuOccupancyMaxPotentialBlockSizeWithFlags(int* minGridSize, int* blockSize,
                                                   CUfunction func,
                                                   CUoccupancyB2DSize blockSizeToDynamicSMemSize,
                                                   size_t dynamicSMemSize, int blockSizeLimit,
                                                   [[maybe_unused]] unsigned int flags) {
  return cuOccupancyMaxPotentialBlockSize(minGridSize, blockSize, func, blockSizeToDynamicSMemSize,
                                          dynamicSMemSize, blockSizeLimit);
}

/* Inter-Process Communication (IPC) Stubs */
CUresult cuIpcGetMemHandle([[maybe_unused]] CUipcMemHandle* pHandle,
                           [[maybe_unused]] CUdeviceptr dptr) {
  return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult cuIpcOpenMemHandle([[maybe_unused]] CUdeviceptr* pdptr,
                            [[maybe_unused]] CUipcMemHandle handle,
                            [[maybe_unused]] unsigned int Flags) {
  return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult cuIpcCloseMemHandle([[maybe_unused]] CUdeviceptr dptr) { return CUDA_ERROR_NOT_SUPPORTED; }

/* Profiler Support */
CUresult cuProfilerStart(void) { return CUDA_SUCCESS; }
CUresult cuProfilerStop(void) { return CUDA_SUCCESS; }
CUresult cuProfilerInitialize([[maybe_unused]] const char* configFile,
                              [[maybe_unused]] const char* outputFile,
                              [[maybe_unused]] int outputMode) {
  return CUDA_SUCCESS;
}

/* JIT Linker */
CUresult cuLinkCreate([[maybe_unused]] unsigned int numOptions,
                      [[maybe_unused]] CUjit_option* options, [[maybe_unused]] void** optionValues,
                      CUlinkState* stateOut) {
  if (stateOut == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  *stateOut = new CUlinkState_st{};
  return CUDA_SUCCESS;
}

CUresult cuLinkCreate_v2(unsigned int numOptions, CUjit_option* options, void** optionValues,
                         CUlinkState* stateOut) {
  return cuLinkCreate(numOptions, options, optionValues, stateOut);
}

CUresult cuLinkAddData(CUlinkState state, [[maybe_unused]] CUjitInputType type, void* data,
                       size_t size, [[maybe_unused]] const char* name,
                       [[maybe_unused]] unsigned int numOptions,
                       [[maybe_unused]] CUjit_option* options,
                       [[maybe_unused]] void** optionValues) {
  if (state == nullptr || data == nullptr || size == 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  state->accumulated_ptx.append(static_cast<const char*>(data), size);
  return CUDA_SUCCESS;
}

CUresult cuLinkAddData_v2(CUlinkState state, CUjitInputType type, void* data, size_t size,
                          const char* name, unsigned int numOptions, CUjit_option* options,
                          void** optionValues) {
  return cuLinkAddData(state, type, data, size, name, numOptions, options, optionValues);
}

CUresult cuLinkAddFile(CUlinkState state, CUjitInputType type, const char* path,
                       unsigned int numOptions, CUjit_option* options, void** optionValues) {
  if (state == nullptr || path == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  std::ifstream f(path);
  if (!f.is_open()) {
    return CUDA_ERROR_FILE_NOT_FOUND;
  }
  std::stringstream ss;
  ss << f.rdbuf();
  const std::string content = ss.str();
  return cuLinkAddData(state, type, const_cast<char*>(content.data()), content.size(), path,
                       numOptions, options, optionValues);
}

CUresult cuLinkAddFile_v2(CUlinkState state, CUjitInputType type, const char* path,
                          unsigned int numOptions, CUjit_option* options, void** optionValues) {
  return cuLinkAddFile(state, type, path, numOptions, options, optionValues);
}

CUresult cuLinkComplete(CUlinkState state, void** cubinOut, size_t* sizeOut) {
  if (state == nullptr || cubinOut == nullptr || sizeOut == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  *cubinOut = const_cast<char*>(state->accumulated_ptx.data());
  *sizeOut = state->accumulated_ptx.size();
  return CUDA_SUCCESS;
}

CUresult cuLinkDestroy(CUlinkState state) {
  if (state == nullptr) {
    return CUDA_SUCCESS;
  }
  delete state;
  return CUDA_SUCCESS;
}

CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
                        unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
                        unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream,
                        void** kernelParams, void** extra) {
  if (f == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  EnsureContext();
  auto& rt = muda::runtime::cudart::RuntimeContext::Instance();

  const dim3 grid(gridDimX, gridDimY, gridDimZ);
  const dim3 block(blockDimX, blockDimY, blockDimZ);

  void** launch_args = kernelParams;
  std::vector<void*> extra_args;

  if (launch_args == nullptr && extra != nullptr) {
    // Unpack extra configuration array
    for (std::size_t i = 0; extra[i] != CU_LAUNCH_PARAM_END; ++i) {
      if (extra[i] == CU_LAUNCH_PARAM_BUFFER_POINTER) {
        char* packed_buf = static_cast<char*>(extra[i + 1]);
        std::size_t offset = 0;
        for (const mudaKernelParamDesc& desc : f->params) {
          extra_args.push_back(reinterpret_cast<void*>(packed_buf + offset));
          offset += desc.size;
        }
        launch_args = extra_args.data();
        break;
      }
    }
  }

  const cudaError_t err =
      rt.Launcher().LaunchKernel(f, grid, block, launch_args, sharedMemBytes, hStream, rt.Memory(),
                                 rt.Streams(), rt.DefaultQueue());

  return MapCudaError(err);
}

CUresult cuLaunchCooperativeKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
                                   unsigned int gridDimZ, unsigned int blockDimX,
                                   unsigned int blockDimY, unsigned int blockDimZ,
                                   unsigned int sharedMemBytes, CUstream hStream,
                                   void** kernelParams) {
  return cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
                        sharedMemBytes, hStream, kernelParams, nullptr);
}

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

/**
 * @file driver_api.mm
 * @brief Implementation of the clean-room CUDA Driver C API over MUDA runtime subsystems.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

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

namespace {

thread_local CUcontext g_current_ctx = nullptr;

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

cudaDeviceAttr MapDeviceAttribute(CUdevice_attribute attrib) noexcept {
  switch (attrib) {
    case CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK:
      return cudaDevAttrMaxThreadsPerBlock;
    case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X:
      return cudaDevAttrMaxBlockDimX;
    case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y:
      return cudaDevAttrMaxBlockDimY;
    case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z:
      return cudaDevAttrMaxBlockDimZ;
    case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X:
      return cudaDevAttrMaxGridDimX;
    case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y:
      return cudaDevAttrMaxGridDimY;
    case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z:
      return cudaDevAttrMaxGridDimZ;
    case CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK:
      return cudaDevAttrMaxSharedMemoryPerBlock;
    case CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY:
      return cudaDevAttrTotalConstantMemory;
    case CU_DEVICE_ATTRIBUTE_WARP_SIZE:
      return cudaDevAttrWarpSize;
    case CU_DEVICE_ATTRIBUTE_MAX_PITCH:
      return cudaDevAttrMaxPitch;
    case CU_DEVICE_ATTRIBUTE_CLOCK_RATE:
      return cudaDevAttrClockRate;
    case CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT:
      return cudaDevAttrMultiProcessorCount;
    case CU_DEVICE_ATTRIBUTE_COMPUTE_MODE:
      return cudaDevAttrComputeMode;
    case CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS:
      return cudaDevAttrConcurrentKernels;
    case CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING:
      return cudaDevAttrUnifiedAddressing;
    case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR:
      return cudaDevAttrComputeCapabilityMajor;
    case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR:
      return cudaDevAttrComputeCapabilityMinor;
    case CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY:
      return cudaDevAttrManagedMemory;
    case CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED:
      return cudaDevAttrHostNativeAtomicSupported;
    case CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS:
      return cudaDevAttrPageableMemoryAccess;
    case CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS:
      return cudaDevAttrConcurrentManagedAccess;
    case CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN:
      return cudaDevAttrMaxSharedMemoryPerBlockOptin;
    default:
      return static_cast<cudaDeviceAttr>(attrib);
  }
}

CUcontext EnsureContext() noexcept {
  if (g_current_ctx != nullptr) {
    return g_current_ctx;
  }
  static CUctx_st s_primary_ctx{0, 0, nil, nil};
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

CUresult cuDeviceGetAttribute(int* value, CUdevice_attribute attrib, CUdevice dev) {
  if (value == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (dev != 0) {
    return CUDA_ERROR_INVALID_DEVICE;
  }
  auto& rt = muda::runtime::cudart::RuntimeContext::Instance();
  return MapCudaError(rt.Devices().GetDeviceAttribute(value, MapDeviceAttribute(attrib), dev));
}

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

CUresult cuCtxDestroy(CUcontext ctx) {
  if (ctx == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (g_current_ctx == ctx) {
    g_current_ctx = nullptr;
  }
  delete ctx;
  return CUDA_SUCCESS;
}

CUresult cuCtxPushCurrent(CUcontext ctx) {
  g_current_ctx = ctx;
  return CUDA_SUCCESS;
}

CUresult cuCtxPopCurrent(CUcontext* pctx) {
  if (pctx != nullptr) {
    *pctx = g_current_ctx;
  }
  g_current_ctx = nullptr;
  return CUDA_SUCCESS;
}

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

CUresult cuMemFree(CUdeviceptr dptr) {
  if (dptr == 0) {
    return CUDA_SUCCESS;
  }
  auto& rt = muda::runtime::cudart::RuntimeContext::Instance();
  void* ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(dptr));
  return MapCudaError(rt.Memory().Free(ptr, rt.Streams(), rt.DefaultQueue()));
}

CUresult cuMemAllocHost(void** pp, size_t bytesize) {
  if (pp == nullptr || bytesize == 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  auto& rt = muda::runtime::cudart::RuntimeContext::Instance();
  return MapCudaError(rt.Memory().AllocateHost(pp, bytesize, cudaHostAllocDefault, rt.Device()));
}

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

CUresult cuMemcpyAsync(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount,
                       CUstream hStream) {
  return cuMemcpyDtoDAsync(dstDevice, srcDevice, ByteCount, hStream);
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

CUresult cuMemsetD16(CUdeviceptr dstDevice, unsigned short us, size_t N) {
  if (N == 0)
    return CUDA_SUCCESS;
  if (dstDevice == 0)
    return CUDA_ERROR_INVALID_VALUE;
  std::vector<unsigned short> host_buf(N, us);
  return cuMemcpyHtoD(dstDevice, host_buf.data(), N * sizeof(unsigned short));
}

CUresult cuMemsetD32(CUdeviceptr dstDevice, unsigned int ui, size_t N) {
  if (N == 0)
    return CUDA_SUCCESS;
  if (dstDevice == 0)
    return CUDA_ERROR_INVALID_VALUE;
  std::vector<unsigned int> host_buf(N, ui);
  return cuMemcpyHtoD(dstDevice, host_buf.data(), N * sizeof(unsigned int));
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

/**
 * @file driver_memory.mm
 * @brief CUDA Driver API memory allocation, pointer queries, copies, and memset operations.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstring>
#include <string.h>

#include "cuda.h"
#include "cuda_runtime.h"
#include "driver/driver_internal.hpp"
#include "memory_manager.hpp"
#include "muda/runtime/driver_types.hpp"
#include "runtime_context.hpp"
#include "transfer_manager.hpp"

extern "C" {

CUresult cuMemAlloc(CUdeviceptr* dptr, size_t bytesize) {
  if (dptr == nullptr || bytesize == 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  muda::runtime::driver::EnsureContext();
  auto& rt = muda::runtime::cudart::RuntimeContext::Instance();
  void* ptr = nullptr;
  const cudaError_t err =
      rt.Memory().Allocate(&ptr, bytesize, cudaMemAttachGlobal, cudaMemoryTypeDevice, rt.Device());
  if (err != cudaSuccess) {
    return muda::runtime::driver::MapCudaError(err);
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
  muda::runtime::driver::EnsureContext();
  auto& rt = muda::runtime::cudart::RuntimeContext::Instance();
  void* ptr = nullptr;
  const cudaError_t err =
      rt.Memory().AllocatePitch(&ptr, pPitch, WidthInBytes, Height, rt.Device());
  if (err != cudaSuccess) {
    return muda::runtime::driver::MapCudaError(err);
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
  muda::runtime::driver::EnsureContext();
  auto& rt = muda::runtime::cudart::RuntimeContext::Instance();
  void* ptr = nullptr;
  const cudaError_t err =
      rt.Memory().Allocate(&ptr, bytesize, cudaMemAttachGlobal, cudaMemoryTypeManaged, rt.Device());
  if (err != cudaSuccess) {
    return muda::runtime::driver::MapCudaError(err);
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
  return muda::runtime::driver::MapCudaError(rt.Memory().Free(ptr, rt.Streams(), rt.DefaultQueue()));
}

CUresult cuMemFree_v2(CUdeviceptr dptr) { return cuMemFree(dptr); }

CUresult cuMemAllocHost(void** pp, size_t bytesize) {
  if (pp == nullptr || bytesize == 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  auto& rt = muda::runtime::cudart::RuntimeContext::Instance();
  return muda::runtime::driver::MapCudaError(rt.Memory().AllocateHost(pp, bytesize, cudaHostAllocDefault, rt.Device()));
}

CUresult cuMemAllocHost_v2(void** pp, size_t bytesize) { return cuMemAllocHost(pp, bytesize); }

CUresult cuMemFreeHost(void* p) {
  if (p == nullptr) {
    return CUDA_SUCCESS;
  }
  auto& rt = muda::runtime::cudart::RuntimeContext::Instance();
  return muda::runtime::driver::MapCudaError(rt.Memory().FreeHost(p, rt.Streams(), rt.DefaultQueue()));
}

CUresult cuMemHostAlloc(void** pp, size_t bytesize, unsigned int Flags) {
  if (pp == nullptr || bytesize == 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  auto& rt = muda::runtime::cudart::RuntimeContext::Instance();
  return muda::runtime::driver::MapCudaError(rt.Memory().AllocateHost(pp, bytesize, Flags, rt.Device()));
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
  return muda::runtime::driver::MapCudaError(rt.Memory().MemGetInfo(free, total, rt.Device()));
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
      *static_cast<CUcontext*>(data) = muda::runtime::driver::EnsureContext();
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
  return muda::runtime::driver::MapCudaError(rt.Transfers().Memcpy(dst, srcHost, ByteCount, cudaMemcpyHostToDevice,
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
  return muda::runtime::driver::MapCudaError(rt.Transfers().Memcpy(dstHost, src, ByteCount, cudaMemcpyDeviceToHost,
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
  return muda::runtime::driver::MapCudaError(rt.Transfers().Memcpy(dst, src, ByteCount, cudaMemcpyDeviceToDevice,
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
  return muda::runtime::driver::MapCudaError(rt.Transfers().MemcpyAsync(dst, srcHost, ByteCount, cudaMemcpyHostToDevice,
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
  return muda::runtime::driver::MapCudaError(rt.Transfers().MemcpyAsync(dstHost, src, ByteCount, cudaMemcpyDeviceToHost,
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
  return muda::runtime::driver::MapCudaError(rt.Transfers().MemcpyAsync(dst, src, ByteCount, cudaMemcpyDeviceToDevice,
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

  return muda::runtime::driver::MapCudaError(rt.Transfers().Memcpy2D(dst, pCopy->dstPitch, src, pCopy->srcPitch,
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

  return muda::runtime::driver::MapCudaError(rt.Transfers().Memcpy2DAsync(
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
  return muda::runtime::driver::MapCudaError(rt.Transfers().Memset(dst, static_cast<int>(uc), N, rt.Memory(), rt.Streams(),
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
  auto& rt = muda::runtime::cudart::RuntimeContext::Instance();
  void* dst = reinterpret_cast<void*>(static_cast<uintptr_t>(dstDevice));
  const std::size_t bytes = N * sizeof(unsigned short);
  if (!rt.Memory().ValidateRange(muda::runtime::cudart::DeviceAddress::FromPointer(dst), bytes)) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (rt.Streams().HasPendingWork()) {
    const cudaError_t sync_err = rt.Streams().SynchronizeDevice(rt.DefaultQueue());
    if (sync_err != cudaSuccess)
      return muda::runtime::driver::MapCudaError(sync_err);
  }
  const std::uint32_t pat = (static_cast<std::uint32_t>(us) << 16) | static_cast<std::uint32_t>(us);
  const std::size_t even_bytes = (N / 2) * sizeof(std::uint32_t);
  if (even_bytes > 0) {
    memset_pattern4(dst, &pat, even_bytes);
  }
  if ((N & 1u) != 0u) {
    static_cast<unsigned short*>(dst)[N - 1] = us;
  }
  return CUDA_SUCCESS;
}

CUresult cuMemsetD16_v2(CUdeviceptr dstDevice, unsigned short us, size_t N) {
  return cuMemsetD16(dstDevice, us, N);
}

CUresult cuMemsetD32(CUdeviceptr dstDevice, unsigned int ui, size_t N) {
  return cuMemsetD32Async(dstDevice, ui, N, nullptr);
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
  return muda::runtime::driver::MapCudaError(rt.Transfers().MemsetAsync(dst, static_cast<int>(uc), N, hStream, rt.Memory(),
                                                 rt.Streams(), rt.DefaultQueue()));
}

CUresult cuMemsetD32Async(CUdeviceptr dstDevice, unsigned int ui, size_t N, CUstream hStream) {
  if (N == 0)
    return CUDA_SUCCESS;
  if (dstDevice == 0)
    return CUDA_ERROR_INVALID_VALUE;
  auto& rt = muda::runtime::cudart::RuntimeContext::Instance();
  void* dst = reinterpret_cast<void*>(static_cast<uintptr_t>(dstDevice));
  const std::size_t bytes = N * sizeof(unsigned int);
  if (!rt.Memory().ValidateRange(muda::runtime::cudart::DeviceAddress::FromPointer(dst), bytes)) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (rt.Streams().HasPendingWork()) {
    const cudaError_t sync_err = rt.Streams().StreamSynchronize(hStream, rt.DefaultQueue());
    if (sync_err != cudaSuccess)
      return muda::runtime::driver::MapCudaError(sync_err);
  }
  const std::uint32_t pat = static_cast<std::uint32_t>(ui);
  memset_pattern4(dst, &pat, bytes);
  return CUDA_SUCCESS;
}


}  // extern "C"

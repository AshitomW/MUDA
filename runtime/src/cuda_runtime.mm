/**
 * @file cuda_runtime.mm
 * @brief Implements the CUDA Runtime C API over modular Metal subcomponents.
 */

#include "cuda_runtime.h"

#include <cstddef>
#include <mutex>
#include <new>

#include "device_manager.hpp"
#include "error_state.hpp"
#include "event_manager.hpp"
#include "launch_engine.hpp"
#include "memory_manager.hpp"
#include "stream_manager.hpp"
#include "transfer_manager.hpp"

namespace muda::runtime::cudart {

class RuntimeContext {
 public:
  static RuntimeContext& Instance() noexcept {
    static std::once_flag init_flag;
    alignas(RuntimeContext) static std::byte storage[sizeof(RuntimeContext)];
    static RuntimeContext* instance = nullptr;
    std::call_once(init_flag, [&]() { instance = new (storage) RuntimeContext(); });
    return *instance;
  }

  cudaError_t EnsureInitialized() noexcept { return devices_.EnsureInitialized(); }
  id<MTLDevice> Device() const noexcept { return devices_.Device(); }
  id<MTLCommandQueue> DefaultQueue() const noexcept { return devices_.DefaultQueue(); }

  DeviceManager& Devices() noexcept { return devices_; }
  MemoryManager& Memory() noexcept { return memory_; }
  StreamManager& Streams() noexcept { return streams_; }
  EventManager& Events() noexcept { return events_; }
  TransferManager& Transfers() noexcept { return transfers_; }
  LaunchEngine& Launcher() noexcept { return launcher_; }

  cudaError_t Reset() noexcept {
    streams_.Reset();
    events_.Reset();
    memory_.Reset();
    launcher_.Clear();
    devices_.Reset();
    return cudaSuccess;
  }

 private:
  RuntimeContext() = default;

  DeviceManager devices_;
  MemoryManager memory_;
  StreamManager streams_;
  EventManager events_;
  TransferManager transfers_;
  LaunchEngine launcher_;
};

namespace {

template <typename Fn>
cudaError_t InvokeCAbi(Fn&& fn) noexcept {
  @try {
    return fn();
  } @catch (...) {
    return RecordError(cudaErrorInternalError);
  }
}

template <typename Fn>
cudaError_t RunWithInit(Fn&& fn) noexcept {
  return InvokeCAbi([&] {
    auto& ctx = RuntimeContext::Instance();
    const cudaError_t init = ctx.EnsureInitialized();
    return (init != cudaSuccess) ? init : fn(ctx);
  });
}

}  // namespace
}  // namespace muda::runtime::cudart

extern "C" {

cudaError_t cudaGetDeviceCount(int* count) {
  return muda::runtime::cudart::InvokeCAbi([&] {
    return muda::runtime::cudart::RuntimeContext::Instance().Devices().GetDeviceCount(count);
  });
}

cudaError_t cudaSetDevice(int device) {
  return muda::runtime::cudart::InvokeCAbi([&] {
    return muda::runtime::cudart::RuntimeContext::Instance().Devices().SetDevice(device);
  });
}

cudaError_t cudaGetDevice(int* device) {
  return muda::runtime::cudart::RunWithInit(
      [&](auto& ctx) { return ctx.Devices().GetDevice(device); });
}

cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int device) {
  return muda::runtime::cudart::RunWithInit(
      [&](auto& ctx) { return ctx.Devices().GetDeviceProperties(prop, device); });
}

cudaError_t cudaDeviceGetAttribute(int* value, cudaDeviceAttr attr, int device) {
  return muda::runtime::cudart::RunWithInit(
      [&](auto& ctx) { return ctx.Devices().GetDeviceAttribute(value, attr, device); });
}

cudaError_t cudaDeviceSynchronize(void) {
  return muda::runtime::cudart::RunWithInit(
      [&](auto& ctx) { return ctx.Streams().SynchronizeDevice(ctx.DefaultQueue()); });
}

cudaError_t cudaDeviceReset(void) {
  return muda::runtime::cudart::InvokeCAbi(
      [&] { return muda::runtime::cudart::RuntimeContext::Instance().Reset(); });
}

cudaError_t cudaRuntimeGetVersion(int* runtimeVersion) {
  return muda::runtime::cudart::InvokeCAbi([&] {
    return muda::runtime::cudart::RuntimeContext::Instance().Devices().GetRuntimeVersion(
        runtimeVersion);
  });
}

cudaError_t cudaDriverGetVersion(int* driverVersion) {
  return muda::runtime::cudart::InvokeCAbi([&] {
    return muda::runtime::cudart::RuntimeContext::Instance().Devices().GetDriverVersion(
        driverVersion);
  });
}

cudaError_t cudaMalloc(void** devPtr, size_t size) {
  return muda::runtime::cudart::RunWithInit([&](auto& ctx) {
    return ctx.Memory().Allocate(devPtr, size, cudaMemAttachGlobal, cudaMemoryTypeDevice,
                                 ctx.Device());
  });
}

cudaError_t cudaMallocManaged(void** devPtr, size_t size, unsigned int flags) {
  return muda::runtime::cudart::RunWithInit([&](auto& ctx) {
    return ctx.Memory().Allocate(devPtr, size, flags, cudaMemoryTypeManaged, ctx.Device());
  });
}

cudaError_t cudaMallocHost(void** ptr, size_t size) {
  return muda::runtime::cudart::RunWithInit([&](auto& ctx) {
    return ctx.Memory().AllocateHost(ptr, size, cudaHostAllocDefault, ctx.Device());
  });
}

cudaError_t cudaHostAlloc(void** pHost, size_t size, unsigned int flags) {
  return muda::runtime::cudart::RunWithInit(
      [&](auto& ctx) { return ctx.Memory().AllocateHost(pHost, size, flags, ctx.Device()); });
}

cudaError_t cudaMallocPitch(void** devPtr, size_t* pitch, size_t width, size_t height) {
  return muda::runtime::cudart::RunWithInit([&](auto& ctx) {
    return ctx.Memory().AllocatePitch(devPtr, pitch, width, height, ctx.Device());
  });
}

cudaError_t cudaFree(void* devPtr) {
  return muda::runtime::cudart::InvokeCAbi([&] {
    auto& ctx = muda::runtime::cudart::RuntimeContext::Instance();
    return ctx.Memory().Free(devPtr, ctx.Streams(), ctx.DefaultQueue());
  });
}

cudaError_t cudaFreeHost(void* ptr) {
  return muda::runtime::cudart::InvokeCAbi([&] {
    auto& ctx = muda::runtime::cudart::RuntimeContext::Instance();
    return ctx.Memory().FreeHost(ptr, ctx.Streams(), ctx.DefaultQueue());
  });
}

cudaError_t cudaMemGetInfo(size_t* free, size_t* total) {
  return muda::runtime::cudart::RunWithInit(
      [&](auto& ctx) { return ctx.Memory().MemGetInfo(free, total, ctx.Device()); });
}

cudaError_t cudaPointerGetAttributes(cudaPointerAttributes* attributes, const void* ptr) {
  return muda::runtime::cudart::RunWithInit(
      [&](auto& ctx) { return ctx.Memory().PointerGetAttributes(attributes, ptr); });
}

cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) {
  return muda::runtime::cudart::RunWithInit([&](auto& ctx) {
    return ctx.Transfers().Memcpy(dst, src, count, kind, ctx.Memory(), ctx.Streams(),
                                  ctx.DefaultQueue());
  });
}

cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind,
                            cudaStream_t stream) {
  return muda::runtime::cudart::RunWithInit([&](auto& ctx) {
    return ctx.Transfers().MemcpyAsync(dst, src, count, kind, stream, ctx.Memory(), ctx.Streams(),
                                       ctx.DefaultQueue());
  });
}

cudaError_t cudaMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                         size_t height, cudaMemcpyKind kind) {
  return muda::runtime::cudart::RunWithInit([&](auto& ctx) {
    return ctx.Transfers().Memcpy2D(dst, dpitch, src, spitch, width, height, kind, ctx.Memory(),
                                    ctx.Streams(), ctx.DefaultQueue());
  });
}

cudaError_t cudaMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch,
                              size_t width, size_t height, cudaMemcpyKind kind,
                              cudaStream_t stream) {
  return muda::runtime::cudart::RunWithInit([&](auto& ctx) {
    return ctx.Transfers().Memcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, stream,
                                         ctx.Memory(), ctx.Streams(), ctx.DefaultQueue());
  });
}

cudaError_t cudaMemset(void* devPtr, int value, size_t count) {
  return muda::runtime::cudart::RunWithInit([&](auto& ctx) {
    return ctx.Transfers().Memset(devPtr, value, count, ctx.Memory(), ctx.Streams(),
                                  ctx.DefaultQueue());
  });
}

cudaError_t cudaMemsetAsync(void* devPtr, int value, size_t count, cudaStream_t stream) {
  return muda::runtime::cudart::RunWithInit([&](auto& ctx) {
    return ctx.Transfers().MemsetAsync(devPtr, value, count, stream, ctx.Memory(), ctx.Streams(),
                                       ctx.DefaultQueue());
  });
}

cudaError_t cudaMemset2D(void* devPtr, size_t pitch, int value, size_t width, size_t height) {
  return muda::runtime::cudart::RunWithInit([&](auto& ctx) {
    return ctx.Transfers().Memset2D(devPtr, pitch, value, width, height, ctx.Memory(),
                                    ctx.Streams(), ctx.DefaultQueue());
  });
}

cudaError_t cudaStreamCreate(cudaStream_t* pStream) {
  return cudaStreamCreateWithFlags(pStream, cudaStreamDefault);
}

cudaError_t cudaStreamCreateWithFlags(cudaStream_t* pStream, unsigned int flags) {
  return muda::runtime::cudart::RunWithInit(
      [&](auto& ctx) { return ctx.Streams().StreamCreate(pStream, flags, ctx.Device()); });
}

cudaError_t cudaStreamDestroy(cudaStream_t stream) {
  return muda::runtime::cudart::InvokeCAbi([&] {
    return muda::runtime::cudart::RuntimeContext::Instance().Streams().StreamDestroy(stream);
  });
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
  return muda::runtime::cudart::RunWithInit(
      [&](auto& ctx) { return ctx.Streams().StreamSynchronize(stream, ctx.DefaultQueue()); });
}

cudaError_t cudaStreamQuery(cudaStream_t stream) {
  return muda::runtime::cudart::RunWithInit(
      [&](auto& ctx) { return ctx.Streams().StreamQuery(stream); });
}

cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags) {
  return muda::runtime::cudart::RunWithInit([&](auto& ctx) {
    return ctx.Events().StreamWaitEvent(stream, event, flags, ctx.Streams(), ctx.DefaultQueue());
  });
}

cudaError_t cudaEventCreate(cudaEvent_t* event) {
  return cudaEventCreateWithFlags(event, cudaEventDefault);
}

cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags) {
  return muda::runtime::cudart::RunWithInit(
      [&](auto& ctx) { return ctx.Events().EventCreate(event, flags, ctx.Device()); });
}

cudaError_t cudaEventDestroy(cudaEvent_t event) {
  return muda::runtime::cudart::InvokeCAbi([&] {
    return muda::runtime::cudart::RuntimeContext::Instance().Events().EventDestroy(event);
  });
}

cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
  return muda::runtime::cudart::RunWithInit([&](auto& ctx) {
    return ctx.Events().EventRecord(event, stream, ctx.Streams(), ctx.DefaultQueue());
  });
}

cudaError_t cudaEventSynchronize(cudaEvent_t event) {
  return muda::runtime::cudart::InvokeCAbi([&] {
    auto& ctx = muda::runtime::cudart::RuntimeContext::Instance();
    return ctx.Events().EventSynchronize(event, ctx.Streams());
  });
}

cudaError_t cudaEventQuery(cudaEvent_t event) {
  return muda::runtime::cudart::InvokeCAbi(
      [&] { return muda::runtime::cudart::RuntimeContext::Instance().Events().EventQuery(event); });
}

cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end) {
  return muda::runtime::cudart::InvokeCAbi([&] {
    auto& ctx = muda::runtime::cudart::RuntimeContext::Instance();
    return ctx.Events().EventElapsedTime(ms, start, end, ctx.Streams());
  });
}

cudaError_t mudaRegisterKernelFromFile(const void* hostFunc, const char* metallibPath,
                                       const char* kernelName, const mudaKernelParamDesc* params,
                                       size_t paramCount) {
  return muda::runtime::cudart::RunWithInit([&](auto& ctx) {
    return ctx.Launcher().RegisterKernelFromFile(hostFunc, metallibPath, kernelName, params,
                                                 paramCount, ctx.Device());
  });
}

cudaError_t mudaRegisterKernelFromData(const void* hostFunc, const void* metallibData,
                                       size_t metallibSize, const char* kernelName,
                                       const mudaKernelParamDesc* params, size_t paramCount) {
  return muda::runtime::cudart::RunWithInit([&](auto& ctx) {
    return ctx.Launcher().RegisterKernelFromData(hostFunc, metallibData, metallibSize, kernelName,
                                                 params, paramCount, ctx.Device());
  });
}

cudaError_t cudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim, void** args,
                             size_t sharedMem, cudaStream_t stream) {
  return muda::runtime::cudart::RunWithInit([&](auto& ctx) {
    return ctx.Launcher().LaunchKernel(func, gridDim, blockDim, args, sharedMem, stream,
                                       ctx.Memory(), ctx.Streams(), ctx.DefaultQueue());
  });
}

cudaError_t cudaGetLastError(void) { return muda::runtime::cudart::ConsumeLastError(); }

cudaError_t cudaPeekAtLastError(void) { return muda::runtime::cudart::PeekLastError(); }

const char* cudaGetErrorName(cudaError_t error) { return muda::runtime::cudart::ErrorName(error); }

const char* cudaGetErrorString(cudaError_t error) {
  return muda::runtime::cudart::ErrorString(error);
}

}  // extern "C"

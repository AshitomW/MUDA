/**
 * @file driver_exec.mm
 * @brief CUDA Driver API streams, events, modules, JIT linker, and kernel launch.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "cuda.h"
#include "cuda_runtime.h"
#include "driver/driver_internal.hpp"
#include "event_manager.hpp"
#include "launch_engine.hpp"
#include "muda/compiler/cubin/cubin_translator.hpp"
#include "muda/runtime/driver_types.hpp"
#include "muda/runtime/module_manager.hpp"
#include "runtime_context.hpp"
#include "stream_manager.hpp"

struct CUlinkState_st {
  std::string accumulated_ptx;
};

extern "C" {

/* Stream Management */
CUresult cuStreamCreate(CUstream* phStream, unsigned int Flags) {
  if (phStream == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  auto& rt = muda::runtime::cudart::RuntimeContext::Instance();
  return muda::runtime::driver::MapCudaError(rt.Streams().StreamCreate(phStream, Flags, rt.Device()));
}

CUresult cuStreamCreateWithPriority(CUstream* phStream, unsigned int flags,
                                    [[maybe_unused]] int priority) {
  if (phStream == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  auto& rt = muda::runtime::cudart::RuntimeContext::Instance();
  return muda::runtime::driver::MapCudaError(rt.Streams().StreamCreate(phStream, flags, rt.Device()));
}

CUresult cuStreamDestroy(CUstream hStream) {
  if (hStream == nullptr)
    return CUDA_SUCCESS;
  auto& rt = muda::runtime::cudart::RuntimeContext::Instance();
  return muda::runtime::driver::MapCudaError(rt.Streams().StreamDestroy(hStream));
}

CUresult cuStreamDestroy_v2(CUstream hStream) { return cuStreamDestroy(hStream); }

CUresult cuStreamSynchronize(CUstream hStream) {
  auto& rt = muda::runtime::cudart::RuntimeContext::Instance();
  return muda::runtime::driver::MapCudaError(rt.Streams().StreamSynchronize(hStream, rt.DefaultQueue()));
}

CUresult cuStreamQuery(CUstream hStream) {
  auto& rt = muda::runtime::cudart::RuntimeContext::Instance();
  return muda::runtime::driver::MapCudaError(rt.Streams().StreamQuery(hStream));
}

CUresult cuStreamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int Flags) {
  if (hEvent == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  auto& rt = muda::runtime::cudart::RuntimeContext::Instance();
  return muda::runtime::driver::MapCudaError(
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
  return muda::runtime::driver::MapCudaError(rt.Events().EventCreate(phEvent, Flags, rt.Device()));
}

CUresult cuEventDestroy(CUevent hEvent) {
  if (hEvent == nullptr)
    return CUDA_SUCCESS;
  auto& rt = muda::runtime::cudart::RuntimeContext::Instance();
  return muda::runtime::driver::MapCudaError(rt.Events().EventDestroy(hEvent));
}

CUresult cuEventDestroy_v2(CUevent hEvent) { return cuEventDestroy(hEvent); }

CUresult cuEventRecord(CUevent hEvent, CUstream hStream) {
  if (hEvent == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  auto& rt = muda::runtime::cudart::RuntimeContext::Instance();
  return muda::runtime::driver::MapCudaError(rt.Events().EventRecord(hEvent, hStream, rt.Streams(), rt.DefaultQueue()));
}

CUresult cuEventSynchronize(CUevent hEvent) {
  if (hEvent == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  auto& rt = muda::runtime::cudart::RuntimeContext::Instance();
  return muda::runtime::driver::MapCudaError(rt.Events().EventSynchronize(hEvent, rt.Streams()));
}

CUresult cuEventQuery(CUevent hEvent) {
  if (hEvent == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  auto& rt = muda::runtime::cudart::RuntimeContext::Instance();
  return muda::runtime::driver::MapCudaError(rt.Events().EventQuery(hEvent));
}

CUresult cuEventElapsedTime(float* pMilliseconds, CUevent hStart, CUevent hEnd) {
  if (pMilliseconds == nullptr || hStart == nullptr || hEnd == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  auto& rt = muda::runtime::cudart::RuntimeContext::Instance();
  return muda::runtime::driver::MapCudaError(rt.Events().EventElapsedTime(pMilliseconds, hStart, hEnd, rt.Streams()));
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
  muda::runtime::driver::EnsureContext();
  auto& rt = muda::runtime::cudart::RuntimeContext::Instance();

  const auto* bytes = static_cast<const std::uint8_t*>(image);
  // Check if image is an NVIDIA Fatbinary or ELF CUBIN
  if (muda::compiler::cubin::IsFatbinary(std::span<const std::uint8_t>(bytes, 16)) ||
      muda::compiler::cubin::IsElfCubin(std::span<const std::uint8_t>(bytes, 64))) {
    return muda::runtime::driver::ModuleManager::LoadCubin(
        std::span<const std::uint8_t>(bytes, 1024 * 1024 * 64), module, rt.Device(),
        rt.Launcher());
  }

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
  muda::runtime::driver::EnsureContext();
  auto& rt = muda::runtime::cudart::RuntimeContext::Instance();

  const dim3 grid(gridDimX, gridDimY, gridDimZ);
  const dim3 block(blockDimX, blockDimY, blockDimZ);

  void** launch_args = kernelParams;
  std::array<void*, 32> inline_extra_args{};
  std::vector<void*> heap_extra_args;

  if (launch_args == nullptr && extra != nullptr) {
    // Unpack extra configuration array
    for (std::size_t i = 0; extra[i] != CU_LAUNCH_PARAM_END; ++i) {
      if (extra[i] == CU_LAUNCH_PARAM_BUFFER_POINTER) {
        char* packed_buf = static_cast<char*>(extra[i + 1]);
        std::size_t offset = 0;
        const std::size_t param_count = f->params.size();
        if (param_count <= inline_extra_args.size()) {
          for (std::size_t p = 0; p < param_count; ++p) {
            inline_extra_args[p] = reinterpret_cast<void*>(packed_buf + offset);
            offset += f->params[p].size;
          }
          launch_args = inline_extra_args.data();
        } else {
          heap_extra_args.reserve(param_count);
          for (const mudaKernelParamDesc& desc : f->params) {
            heap_extra_args.push_back(reinterpret_cast<void*>(packed_buf + offset));
            offset += desc.size;
          }
          launch_args = heap_extra_args.data();
        }
        break;
      }
    }
  }

  const cudaError_t err =
      rt.Launcher().LaunchKernel(f, grid, block, launch_args, sharedMemBytes, hStream, rt.Memory(),
                                 rt.Streams(), rt.DefaultQueue());

  return muda::runtime::driver::MapCudaError(err);
}

CUresult cuLaunchCooperativeKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
                                   unsigned int gridDimZ, unsigned int blockDimX,
                                   unsigned int blockDimY, unsigned int blockDimZ,
                                   unsigned int sharedMemBytes, CUstream hStream,
                                   void** kernelParams) {
  return cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
                        sharedMemBytes, hStream, kernelParams, nullptr);
}

}  // extern "C"

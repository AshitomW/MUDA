/**
 * @file driver_api_test.cpp
 * @brief Comprehensive verification test suite for the MUDA CUDA Driver API (libcuda).
 */

#include <cmath>
#include <iostream>
#include <string_view>
#include <vector>

#include "cuda.h"

namespace {

bool Expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    return false;
  }
  return true;
}

bool ExpectCuSuccess(CUresult res, const char* message) {
  if (res != CUDA_SUCCESS) {
    const char* err_name = "UNKNOWN";
    cuGetErrorName(res, &err_name);
    std::cerr << "FAIL: " << message << " (error: " << err_name << " / " << res << ")\n";
    return false;
  }
  return true;
}

constexpr std::string_view kVectorAddPtx = R"PTX(
.version 6.5
.target sm_70
.address_size 64

.visible .entry vector_add(
    .param .u64 param_a,
    .param .u64 param_b,
    .param .u64 param_c,
    .param .u32 param_n
)
{
    .reg .pred  %p<2>;
    .reg .b32   %r<5>;
    .reg .b64   %rd<9>;
    .reg .f32   %f<4>;

    ld.param.u64    %rd1, [param_a];
    ld.param.u64    %rd2, [param_b];
    ld.param.u64    %rd3, [param_c];
    ld.param.u32    %r1, [param_n];

    mov.u32         %r2, %ctaid.x;
    mov.u32         %r3, %ntid.x;
    mov.u32         %r4, %tid.x;
    mad.lo.s32      %r0, %r2, %r3, %r4;

    setp.ge.s32     %p1, %r0, %r1;
    @%p1 bra        $L__exit;

    mul.wide.s32    %rd4, %r0, 4;
    add.s64         %rd5, %rd1, %rd4;
    add.s64         %rd6, %rd2, %rd4;
    ld.global.f32   %f1, [%rd5];
    ld.global.f32   %f2, [%rd6];
    add.f32         %f3, %f1, %f2;

    add.s64         %rd7, %rd3, %rd4;
    st.global.f32   [%rd7], %f3;

$L__exit:
    ret;
}
)PTX";

bool TestDriverInitAndDevice() {
  std::cout << "Testing cuInit, cuDevice*, cuDriverGetVersion...\n";
  if (!ExpectCuSuccess(cuInit(0), "cuInit(0) should succeed"))
    return false;

  int driver_version = 0;
  if (!ExpectCuSuccess(cuDriverGetVersion(&driver_version), "cuDriverGetVersion"))
    return false;
  if (!Expect(driver_version > 0, "driver_version > 0"))
    return false;

  int count = 0;
  if (!ExpectCuSuccess(cuDeviceGetCount(&count), "cuDeviceGetCount"))
    return false;
  if (!Expect(count >= 1, "device count >= 1"))
    return false;

  CUdevice dev = -1;
  if (!ExpectCuSuccess(cuDeviceGet(&dev, 0), "cuDeviceGet"))
    return false;
  if (!Expect(dev == 0, "dev == 0"))
    return false;

  char name[256]{};
  if (!ExpectCuSuccess(cuDeviceGetName(name, sizeof(name), dev), "cuDeviceGetName"))
    return false;
  if (!Expect(name[0] != '\0', "device name not empty"))
    return false;
  std::cout << "  Device name: " << name << "\n";

  size_t total_mem = 0;
  if (!ExpectCuSuccess(cuDeviceTotalMem(&total_mem, dev), "cuDeviceTotalMem"))
    return false;
  if (!Expect(total_mem > 0, "totalGlobalMem > 0"))
    return false;

  int max_threads = 0;
  if (!ExpectCuSuccess(
          cuDeviceGetAttribute(&max_threads, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, dev),
          "cuDeviceGetAttribute max threads")) {
    return false;
  }
  if (!Expect(max_threads >= 256, "max_threads >= 256"))
    return false;

  return true;
}

bool TestDriverContextAndMemory() {
  std::cout << "Testing cuCtxCreate, cuMemAlloc, cuMemcpy, cuMemFree, cuCtxDestroy...\n";
  CUdevice dev = 0;
  CUcontext ctx = nullptr;
  if (!ExpectCuSuccess(cuCtxCreate(&ctx, 0, dev), "cuCtxCreate"))
    return false;
  if (!Expect(ctx != nullptr, "ctx != nullptr"))
    return false;

  CUcontext current = nullptr;
  if (!ExpectCuSuccess(cuCtxGetCurrent(&current), "cuCtxGetCurrent"))
    return false;
  if (!Expect(current == ctx, "current == ctx"))
    return false;

  constexpr size_t kBytes = 1024 * sizeof(float);
  CUdeviceptr d_ptr = 0;
  if (!ExpectCuSuccess(cuMemAlloc(&d_ptr, kBytes), "cuMemAlloc"))
    return false;
  if (!Expect(d_ptr != 0, "d_ptr != 0"))
    return false;

  std::vector<float> host_in(1024, 42.0f);
  std::vector<float> host_out(1024, 0.0f);

  if (!ExpectCuSuccess(cuMemcpyHtoD(d_ptr, host_in.data(), kBytes), "cuMemcpyHtoD"))
    return false;
  if (!ExpectCuSuccess(cuMemcpyDtoH(host_out.data(), d_ptr, kBytes), "cuMemcpyDtoH"))
    return false;

  for (size_t i = 0; i < 1024; ++i) {
    if (host_out[i] != 42.0f) {
      std::cerr << "FAIL: Memory mismatch at index " << i << ": expected 42.0, got " << host_out[i]
                << "\n";
      return false;
    }
  }

  if (!ExpectCuSuccess(cuMemFree(d_ptr), "cuMemFree"))
    return false;
  if (!ExpectCuSuccess(cuCtxDestroy(ctx), "cuCtxDestroy"))
    return false;

  return true;
}

bool TestDriverPtxExecution() {
  std::cout
      << "Testing cuModuleLoadData (in-memory PTX JIT), cuModuleGetFunction, cuLaunchKernel...\n";
  CUdevice dev = 0;
  CUcontext ctx = nullptr;
  if (!ExpectCuSuccess(cuCtxCreate(&ctx, 0, dev), "cuCtxCreate"))
    return false;

  CUmodule module = nullptr;
  if (!ExpectCuSuccess(cuModuleLoadData(&module, kVectorAddPtx.data()), "cuModuleLoadData")) {
    return false;
  }
  if (!Expect(module != nullptr, "module != nullptr"))
    return false;

  CUfunction func = nullptr;
  if (!ExpectCuSuccess(cuModuleGetFunction(&func, module, "vector_add"), "cuModuleGetFunction")) {
    return false;
  }
  if (!Expect(func != nullptr, "func != nullptr"))
    return false;

  constexpr unsigned int kN = 2048;
  constexpr size_t kBytes = kN * sizeof(float);

  std::vector<float> h_a(kN);
  std::vector<float> h_b(kN);
  std::vector<float> h_c(kN, 0.0f);
  for (unsigned int i = 0; i < kN; ++i) {
    h_a[i] = static_cast<float>(i) * 1.5f;
    h_b[i] = static_cast<float>(i) * 2.5f;
  }

  CUdeviceptr d_a = 0;
  CUdeviceptr d_b = 0;
  CUdeviceptr d_c = 0;
  if (!ExpectCuSuccess(cuMemAlloc(&d_a, kBytes), "cuMemAlloc d_a"))
    return false;
  if (!ExpectCuSuccess(cuMemAlloc(&d_b, kBytes), "cuMemAlloc d_b"))
    return false;
  if (!ExpectCuSuccess(cuMemAlloc(&d_c, kBytes), "cuMemAlloc d_c"))
    return false;

  if (!ExpectCuSuccess(cuMemcpyHtoD(d_a, h_a.data(), kBytes), "cuMemcpyHtoD d_a"))
    return false;
  if (!ExpectCuSuccess(cuMemcpyHtoD(d_b, h_b.data(), kBytes), "cuMemcpyHtoD d_b"))
    return false;

  unsigned int n = kN;
  void* args[] = {&d_a, &d_b, &d_c, &n};

  unsigned int block_dim = 256;
  unsigned int grid_dim = (kN + block_dim - 1) / block_dim;

  if (!ExpectCuSuccess(
          cuLaunchKernel(func, grid_dim, 1, 1, block_dim, 1, 1, 0, nullptr, args, nullptr),
          "cuLaunchKernel")) {
    return false;
  }

  if (!ExpectCuSuccess(cuCtxSynchronize(), "cuCtxSynchronize"))
    return false;
  if (!ExpectCuSuccess(cuMemcpyDtoH(h_c.data(), d_c, kBytes), "cuMemcpyDtoH d_c"))
    return false;

  for (unsigned int i = 0; i < kN; ++i) {
    const float expected = h_a[i] + h_b[i];
    if (std::abs(h_c[i] - expected) > 1e-4f) {
      std::cerr << "FAIL: PTX execution mismatch at " << i << ": expected " << expected << ", got "
                << h_c[i] << "\n";
      return false;
    }
  }

  std::cout << "  Verified " << kN << " floats correctly computed via JIT-compiled PTX on GPU!\n";

  if (!ExpectCuSuccess(cuMemFree(d_a), "cuMemFree d_a"))
    return false;
  if (!ExpectCuSuccess(cuMemFree(d_b), "cuMemFree d_b"))
    return false;
  if (!ExpectCuSuccess(cuMemFree(d_c), "cuMemFree d_c"))
    return false;
  if (!ExpectCuSuccess(cuModuleUnload(module), "cuModuleUnload"))
    return false;
  if (!ExpectCuSuccess(cuCtxDestroy(ctx), "cuCtxDestroy"))
    return false;

  return true;
}

bool TestDriverStreamsAndEvents() {
  std::cout << "Testing cuStream*, cuEvent*...\n";
  CUdevice dev = 0;
  CUcontext ctx = nullptr;
  if (!ExpectCuSuccess(cuCtxCreate(&ctx, 0, dev), "cuCtxCreate"))
    return false;

  CUstream stream = nullptr;
  if (!ExpectCuSuccess(cuStreamCreate(&stream, CU_STREAM_DEFAULT), "cuStreamCreate"))
    return false;
  if (!Expect(stream != nullptr, "stream != nullptr"))
    return false;

  CUevent start = nullptr;
  CUevent stop = nullptr;
  if (!ExpectCuSuccess(cuEventCreate(&start, CU_EVENT_DEFAULT), "cuEventCreate start"))
    return false;
  if (!ExpectCuSuccess(cuEventCreate(&stop, CU_EVENT_DEFAULT), "cuEventCreate stop"))
    return false;

  if (!ExpectCuSuccess(cuEventRecord(start, stream), "cuEventRecord start"))
    return false;
  if (!ExpectCuSuccess(cuEventRecord(stop, stream), "cuEventRecord stop"))
    return false;

  if (!ExpectCuSuccess(cuEventSynchronize(stop), "cuEventSynchronize stop"))
    return false;
  if (!ExpectCuSuccess(cuStreamSynchronize(stream), "cuStreamSynchronize"))
    return false;

  float elapsed_ms = 0.0f;
  if (!ExpectCuSuccess(cuEventElapsedTime(&elapsed_ms, start, stop), "cuEventElapsedTime")) {
    return false;
  }
  if (!Expect(elapsed_ms >= 0.0f, "elapsed_ms >= 0.0f"))
    return false;

  if (!ExpectCuSuccess(cuEventDestroy(start), "cuEventDestroy start"))
    return false;
  if (!ExpectCuSuccess(cuEventDestroy(stop), "cuEventDestroy stop"))
    return false;
  if (!ExpectCuSuccess(cuStreamDestroy(stream), "cuStreamDestroy"))
    return false;
  if (!ExpectCuSuccess(cuCtxDestroy(ctx), "cuCtxDestroy"))
    return false;

  return true;
}

}  // namespace

int main() {
  if (!TestDriverInitAndDevice())
    return 1;
  if (!TestDriverContextAndMemory())
    return 1;
  if (!TestDriverPtxExecution())
    return 1;
  if (!TestDriverStreamsAndEvents())
    return 1;

  std::cout << "driver_api_test: all tests passed!\n";
  return 0;
}

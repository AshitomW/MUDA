/**
 * @file nvrtc_test.cpp
 * @brief End-to-end tests for NVRTC runtime compilation and GPU execution via cuModuleLoadData.
 */

#include "nvrtc.h"

#include <array>
#include <iostream>
#include <string>
#include <vector>

#include "cuda.h"
#include "test_utils.hpp"

namespace {

bool Expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    return false;
  }
  return true;
}

bool TestNvrtcVersionAndErrors() {
  int major = 0;
  int minor = 0;
  if (!Expect(nvrtcVersion(&major, &minor) == NVRTC_SUCCESS, "nvrtcVersion failed")) {
    return false;
  }
  if (!Expect(major >= 11, "Expected NVRTC major version >= 11")) {
    return false;
  }
  if (!Expect(nvrtcVersion(nullptr, &minor) == NVRTC_ERROR_INVALID_INPUT,
              "nvrtcVersion(nullptr) must fail")) {
    return false;
  }
  const std::string_view err_str = nvrtcGetErrorString(NVRTC_SUCCESS);
  return Expect(err_str == "NVRTC_SUCCESS", "nvrtcGetErrorString(NVRTC_SUCCESS) mismatch");
}

bool TestNvrtcCompileAndLaunch() {
  const char* header_src =
      "static inline __device__ float add_offset(float x) { return x + 2.5f; }\n";
  const char* header_name = "my_math_helper.h";

  const char* cuda_src = R"(
#include "my_math_helper.h"
__global__ void nvrtc_axpy_offset(float* out, const float* x, float alpha, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = add_offset(x[idx] * alpha);
  }
}
)";

  nvrtcProgram prog = nullptr;
  if (!Expect(nvrtcCreateProgram(&prog, cuda_src, "axpy.cu", 1, &header_src, &header_name) ==
                  NVRTC_SUCCESS,
              "nvrtcCreateProgram failed")) {
    return false;
  }

  if (!Expect(nvrtcAddNameExpression(prog, "nvrtc_axpy_offset") == NVRTC_SUCCESS,
              "nvrtcAddNameExpression failed")) {
    return false;
  }

  const std::array<const char*, 2> opts = {"-std=c++20", "-arch=sm_86"};
  if (!Expect(
          nvrtcCompileProgram(prog, static_cast<int>(opts.size()), opts.data()) == NVRTC_SUCCESS,
          "nvrtcCompileProgram failed")) {
    return false;
  }

  const char* lowered = nullptr;
  if (!Expect(nvrtcGetLoweredName(prog, "nvrtc_axpy_offset", &lowered) == NVRTC_SUCCESS &&
                  lowered != nullptr && std::string_view(lowered) == "nvrtc_axpy_offset",
              "nvrtcGetLoweredName failed")) {
    return false;
  }

  std::size_t ptx_size = 0;
  if (!Expect(nvrtcGetPTXSize(prog, &ptx_size) == NVRTC_SUCCESS && ptx_size > 16,
              "nvrtcGetPTXSize failed")) {
    return false;
  }
  std::vector<char> ptx_buf(ptx_size, '\0');
  if (!Expect(nvrtcGetPTX(prog, ptx_buf.data()) == NVRTC_SUCCESS, "nvrtcGetPTX failed")) {
    return false;
  }

  if (!Expect(nvrtcDestroyProgram(&prog) == NVRTC_SUCCESS && prog == nullptr,
              "nvrtcDestroyProgram failed")) {
    return false;
  }

  // Load emitted PTX into CUDA Driver API and execute on Apple Silicon GPU
  if (!Expect(cuInit(0) == CUDA_SUCCESS, "cuInit failed")) {
    return false;
  }
  CUdevice dev = 0;
  CUcontext ctx = nullptr;
  if (!Expect(cuDeviceGet(&dev, 0) == CUDA_SUCCESS &&
                  cuDevicePrimaryCtxRetain(&ctx, dev) == CUDA_SUCCESS,
              "Primary context retain failed")) {
    return false;
  }

  CUmodule mod = nullptr;
  if (!Expect(cuModuleLoadData(&mod, ptx_buf.data()) == CUDA_SUCCESS,
              "cuModuleLoadData with NVRTC output failed")) {
    return false;
  }
  CUfunction fn = nullptr;
  if (!Expect(cuModuleGetFunction(&fn, mod, "nvrtc_axpy_offset") == CUDA_SUCCESS,
              "cuModuleGetFunction failed")) {
    return false;
  }

  constexpr int kCount = 64;
  std::vector<float> h_x(kCount);
  std::vector<float> h_out(kCount, 0.0f);
  for (int i = 0; i < kCount; ++i) {
    h_x[static_cast<std::size_t>(i)] = static_cast<float>(i + 1);
  }

  CUdeviceptr d_x = 0;
  CUdeviceptr d_out = 0;
  const std::size_t bytes = static_cast<std::size_t>(kCount) * sizeof(float);
  if (!Expect(cuMemAlloc(&d_x, bytes) == CUDA_SUCCESS && cuMemAlloc(&d_out, bytes) == CUDA_SUCCESS,
              "cuMemAlloc failed")) {
    return false;
  }
  if (!Expect(cuMemcpyHtoD(d_x, h_x.data(), bytes) == CUDA_SUCCESS, "cuMemcpyHtoD failed")) {
    return false;
  }

  float alpha = 3.0f;
  int n = kCount;
  void* args[] = {&d_out, &d_x, &alpha, &n};
  if (!Expect(cuLaunchKernel(fn, 1, 1, 1, kCount, 1, 1, 0, nullptr, args, nullptr) == CUDA_SUCCESS,
              "cuLaunchKernel failed")) {
    return false;
  }
  if (!Expect(cuCtxSynchronize() == CUDA_SUCCESS, "cuCtxSynchronize failed")) {
    return false;
  }
  if (!Expect(cuMemcpyDtoH(h_out.data(), d_out, bytes) == CUDA_SUCCESS, "cuMemcpyDtoH failed")) {
    return false;
  }

  for (int i = 0; i < kCount; ++i) {
    const float expected = h_x[static_cast<std::size_t>(i)] * alpha + 2.5f;
    if (!Expect(muda::testing::AreClose(h_out[static_cast<std::size_t>(i)], expected),
                "GPU numerical output mismatch for NVRTC kernel")) {
      return false;
    }
  }

  (void)cuMemFree(d_x);
  (void)cuMemFree(d_out);
  (void)cuModuleUnload(mod);
  (void)cuDevicePrimaryCtxRelease(dev);
  return true;
}

bool TestNvrtcDiagnosticsOnUnsupported() {
  const char* invalid_src = R"(
__global__ void bad_double_kernel(double* out) {
  out[0] = 1.0;
}
)";
  nvrtcProgram prog = nullptr;
  if (!Expect(
          nvrtcCreateProgram(&prog, invalid_src, "bad.cu", 0, nullptr, nullptr) == NVRTC_SUCCESS,
          "nvrtcCreateProgram failed")) {
    return false;
  }
  if (!Expect(nvrtcCompileProgram(prog, 0, nullptr) == NVRTC_ERROR_COMPILATION,
              "Unsupported double kernel must fail with NVRTC_ERROR_COMPILATION")) {
    return false;
  }
  std::size_t log_size = 0;
  if (!Expect(nvrtcGetProgramLogSize(prog, &log_size) == NVRTC_SUCCESS && log_size > 4,
              "nvrtcGetProgramLogSize failed")) {
    return false;
  }
  std::string log(log_size, '\0');
  if (!Expect(nvrtcGetProgramLog(prog, log.data()) == NVRTC_SUCCESS, "nvrtcGetProgramLog failed")) {
    return false;
  }
  if (!Expect(log.find("ALY-") != std::string::npos,
              "NVRTC failure log should include ALY-* diagnostic code")) {
    return false;
  }
  (void)nvrtcDestroyProgram(&prog);
  return true;
}

}  // namespace

int main() {
  if (!TestNvrtcVersionAndErrors()) {
    return 1;
  }
  if (!TestNvrtcCompileAndLaunch()) {
    return 1;
  }
  if (!TestNvrtcDiagnosticsOnUnsupported()) {
    return 1;
  }
  std::cout << "All NVRTC tests passed!\n";
  return 0;
}

/**
 * @file fp64_test.cpp
 * @brief Comprehensive verification suite for MUDA FP64 (double precision) emulation.
 */

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "cuda.h"
#include "cuda_runtime.h"
#include "nvrtc.h"

namespace {

bool Expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    return false;
  }
  return true;
}

bool ExpectCloseDouble(double actual, double expected, double rel_tol, const char* message) {
  const double diff = std::fabs(actual - expected);
  const double denom = std::max(std::fabs(expected), 1.0);
  if (diff / denom > rel_tol) {
    std::cerr << "FAIL: " << message << " (actual=" << actual << ", expected=" << expected
              << ", rel_diff=" << (diff / denom) << ")\n";
    return false;
  }
  return true;
}

bool TestFp64OffRejection() {
  std::cout << "Testing FP64 rejection under default policy (kOff)...\n";

  const char* cuda_src = R"(
__global__ void fp64_kernel(const double* a, const double* b, double* c, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}
)";

  nvrtcProgram prog = nullptr;
  if (!Expect(
          nvrtcCreateProgram(&prog, cuda_src, "fp64_test.cu", 0, nullptr, nullptr) == NVRTC_SUCCESS,
          "nvrtcCreateProgram failed")) {
    return false;
  }

  // Compile with default options (policy = kOff)
  const char* opts[] = {"-std=c++20"};
  const nvrtcResult res = nvrtcCompileProgram(prog, 1, opts);
  if (!Expect(res == NVRTC_ERROR_COMPILATION,
              "Expected compilation failure for double under kOff policy")) {
    nvrtcDestroyProgram(&prog);
    return false;
  }

  std::size_t log_size = 0;
  nvrtcGetProgramLogSize(prog, &log_size);
  std::string log(log_size, '\0');
  nvrtcGetProgramLog(prog, log.data());

  nvrtcDestroyProgram(&prog);

  if (!Expect(log.find("ALY-C-401") != std::string::npos,
              "Log should contain diagnostic ALY-C-401 for native FP64 rejection")) {
    std::cerr << "Actual log:\n" << log << "\n";
    return false;
  }

  return true;
}

bool TestFp64Fast48Math() {
  std::cout << "Testing FP64 fast48 double-single arithmetic and math intrinsics on GPU...\n";

  const char* cuda_src = R"(
__global__ void fp64_math_ops(const double* a, const double* b, double* c, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    double x = a[i];
    double y = b[i];
    double sum = x + y;
    double diff = x - y;
    double prod = x * y;
    double quot = x / y;
    double fma_res = fma(x, y, x);
    double s = sqrt(x);
    c[i] = (sum + diff + prod + quot + fma_res + s);
  }
}
)";

  nvrtcProgram prog = nullptr;
  if (!Expect(
          nvrtcCreateProgram(&prog, cuda_src, "fp64_math.cu", 0, nullptr, nullptr) == NVRTC_SUCCESS,
          "nvrtcCreateProgram failed")) {
    return false;
  }

  const char* opts[] = {"--muda-fp64=fast48", "-std=c++20"};
  if (!Expect(nvrtcCompileProgram(prog, 2, opts) == NVRTC_SUCCESS,
              "nvrtcCompileProgram with --muda-fp64=fast48 failed")) {
    std::size_t log_size = 0;
    nvrtcGetProgramLogSize(prog, &log_size);
    std::string log(log_size, '\0');
    nvrtcGetProgramLog(prog, log.data());
    std::cerr << "Compile log:\n" << log << "\n";
    nvrtcDestroyProgram(&prog);
    return false;
  }

  std::size_t ptx_size = 0;
  nvrtcGetPTXSize(prog, &ptx_size);
  std::vector<char> ptx(ptx_size, '\0');
  nvrtcGetPTX(prog, ptx.data());
  nvrtcDestroyProgram(&prog);

  // Initialize Driver API
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
  if (!Expect(cuModuleLoadData(&mod, ptx.data()) == CUDA_SUCCESS,
              "cuModuleLoadData for FP64 kernel failed")) {
    return false;
  }

  CUfunction fn = nullptr;
  if (!Expect(cuModuleGetFunction(&fn, mod, "fp64_math_ops") == CUDA_SUCCESS,
              "cuModuleGetFunction failed")) {
    return false;
  }

  constexpr int kN = 256;
  std::vector<double> h_a(kN);
  std::vector<double> h_b(kN);
  std::vector<double> h_c(kN, 0.0);
  std::vector<double> h_gold(kN, 0.0);

  for (int i = 0; i < kN; ++i) {
    h_a[static_cast<std::size_t>(i)] = 10.0 + static_cast<double>(i) * 0.25;
    h_b[static_cast<std::size_t>(i)] = 2.0 + static_cast<double>(i) * 0.05;

    const double x = h_a[static_cast<std::size_t>(i)];
    const double y = h_b[static_cast<std::size_t>(i)];
    const double sum = x + y;
    const double diff = x - y;
    const double prod = x * y;
    const double quot = x / y;
    const double fma_res = std::fma(x, y, x);
    const double s = std::sqrt(x);
    h_gold[static_cast<std::size_t>(i)] = (sum + diff + prod + quot + fma_res + s);
  }

  CUdeviceptr d_a = 0;
  CUdeviceptr d_b = 0;
  CUdeviceptr d_c = 0;
  const std::size_t bytes = static_cast<std::size_t>(kN) * sizeof(double);
  if (!Expect(cuMemAlloc(&d_a, bytes) == CUDA_SUCCESS && cuMemAlloc(&d_b, bytes) == CUDA_SUCCESS &&
                  cuMemAlloc(&d_c, bytes) == CUDA_SUCCESS,
              "cuMemAlloc failed")) {
    return false;
  }

  cuMemcpyHtoD(d_a, h_a.data(), bytes);
  cuMemcpyHtoD(d_b, h_b.data(), bytes);

  int n = kN;
  void* args[] = {&d_a, &d_b, &d_c, &n};
  if (!Expect(cuLaunchKernel(fn, 1, 1, 1, kN, 1, 1, 0, nullptr, args, nullptr) == CUDA_SUCCESS,
              "cuLaunchKernel failed for FP64 kernel")) {
    return false;
  }
  if (!Expect(cuCtxSynchronize() == CUDA_SUCCESS, "cuCtxSynchronize failed")) {
    return false;
  }

  cuMemcpyDtoH(h_c.data(), d_c, bytes);

  for (int i = 0; i < kN; ++i) {
    if (!ExpectCloseDouble(h_c[static_cast<std::size_t>(i)], h_gold[static_cast<std::size_t>(i)],
                           1e-12, "FP64 math op numerical accuracy mismatch")) {
      return false;
    }
  }

  cuMemFree(d_a);
  cuMemFree(d_b);
  cuMemFree(d_c);
  cuModuleUnload(mod);
  return true;
}

bool TestFp64SharedMemory() {
  std::cout << "Testing FP64 shared memory allocation and synchronization...\n";

  const char* cuda_src = R"(
__global__ void fp64_shmem_reverse(const double* in, double* out, int n) {
  __shared__ double sdata[64];
  int tid = threadIdx.x;
  if (tid < n) {
    sdata[tid] = in[tid];
  }
  __syncthreads();
  if (tid < n) {
    out[tid] = sdata[n - 1 - tid];
  }
}
)";

  nvrtcProgram prog = nullptr;
  if (!Expect(nvrtcCreateProgram(&prog, cuda_src, "fp64_shmem.cu", 0, nullptr, nullptr) ==
                  NVRTC_SUCCESS,
              "nvrtcCreateProgram failed")) {
    return false;
  }

  const char* opts[] = {"--muda-fp64=fast48", "-std=c++20"};
  if (!Expect(nvrtcCompileProgram(prog, 2, opts) == NVRTC_SUCCESS,
              "nvrtcCompileProgram for FP64 shmem failed")) {
    nvrtcDestroyProgram(&prog);
    return false;
  }

  std::size_t ptx_size = 0;
  nvrtcGetPTXSize(prog, &ptx_size);
  std::vector<char> ptx(ptx_size, '\0');
  nvrtcGetPTX(prog, ptx.data());
  nvrtcDestroyProgram(&prog);

  CUmodule mod = nullptr;
  if (!Expect(cuModuleLoadData(&mod, ptx.data()) == CUDA_SUCCESS, "cuModuleLoadData failed")) {
    return false;
  }
  CUfunction fn = nullptr;
  if (!Expect(cuModuleGetFunction(&fn, mod, "fp64_shmem_reverse") == CUDA_SUCCESS,
              "cuModuleGetFunction failed")) {
    return false;
  }

  constexpr int kN = 64;
  std::vector<double> h_in(kN);
  std::vector<double> h_out(kN, 0.0);
  for (int i = 0; i < kN; ++i) {
    h_in[static_cast<std::size_t>(i)] = 100.0 + static_cast<double>(i) * 1.5;
  }

  CUdeviceptr d_in = 0;
  CUdeviceptr d_out = 0;
  const std::size_t bytes = static_cast<std::size_t>(kN) * sizeof(double);
  cuMemAlloc(&d_in, bytes);
  cuMemAlloc(&d_out, bytes);
  cuMemcpyHtoD(d_in, h_in.data(), bytes);

  int n = kN;
  void* args[] = {&d_in, &d_out, &n};
  if (!Expect(cuLaunchKernel(fn, 1, 1, 1, kN, 1, 1, 0, nullptr, args, nullptr) == CUDA_SUCCESS,
              "cuLaunchKernel failed for shmem reverse")) {
    return false;
  }
  cuCtxSynchronize();

  cuMemcpyDtoH(h_out.data(), d_out, bytes);

  for (int i = 0; i < kN; ++i) {
    const double expected = h_in[static_cast<std::size_t>(kN - 1 - i)];
    if (!ExpectCloseDouble(h_out[static_cast<std::size_t>(i)], expected, 1e-15,
                           "FP64 shared memory reverse element mismatch")) {
      return false;
    }
  }

  cuMemFree(d_in);
  cuMemFree(d_out);
  cuModuleUnload(mod);
  return true;
}

bool TestFp64Ieee64Math() {
  std::cout << "Testing FP64 ieee64 bit-exact integer bitfield emulation on GPU...\n";

  const char* cuda_src = R"(
__global__ void fp64_ieee_ops(const double* a, const double* b,
                              double* c_add, double* c_sub, double* c_mul,
                              double* c_div, double* c_sqrt, int* c_cmp,
                              int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    double x = a[i];
    double y = b[i];
    c_add[i] = x + y;
    c_sub[i] = x - y;
    c_mul[i] = x * y;
    c_div[i] = x / y;
    c_sqrt[i] = sqrt(x);
    c_cmp[i] = (x < y) ? 1 : 0;
  }
}
)";

  nvrtcProgram prog = nullptr;
  if (!Expect(
          nvrtcCreateProgram(&prog, cuda_src, "fp64_ieee.cu", 0, nullptr, nullptr) == NVRTC_SUCCESS,
          "nvrtcCreateProgram failed")) {
    return false;
  }

  const char* opts[] = {"--muda-fp64=ieee64", "-std=c++20"};
  if (!Expect(nvrtcCompileProgram(prog, 2, opts) == NVRTC_SUCCESS,
              "nvrtcCompileProgram with --muda-fp64=ieee64 failed")) {
    std::size_t log_size = 0;
    nvrtcGetProgramLogSize(prog, &log_size);
    std::string log(log_size, '\0');
    nvrtcGetProgramLog(prog, log.data());
    std::cerr << "Compile log:\n" << log << "\n";
    nvrtcDestroyProgram(&prog);
    return false;
  }

  std::size_t ptx_size = 0;
  nvrtcGetPTXSize(prog, &ptx_size);
  std::vector<char> ptx(ptx_size, '\0');
  nvrtcGetPTX(prog, ptx.data());
  nvrtcDestroyProgram(&prog);

  CUmodule mod = nullptr;
  if (!Expect(cuModuleLoadData(&mod, ptx.data()) == CUDA_SUCCESS,
              "cuModuleLoadData for IEEE-754 FP64 kernel failed")) {
    return false;
  }

  CUfunction fn = nullptr;
  if (!Expect(cuModuleGetFunction(&fn, mod, "fp64_ieee_ops") == CUDA_SUCCESS,
              "cuModuleGetFunction failed")) {
    return false;
  }

  constexpr int kN = 128;
  std::vector<double> h_a(kN);
  std::vector<double> h_b(kN);
  std::vector<double> h_add(kN, 0.0);
  std::vector<double> h_sub(kN, 0.0);
  std::vector<double> h_mul(kN, 0.0);
  std::vector<double> h_div(kN, 0.0);
  std::vector<double> h_sqrt(kN, 0.0);
  std::vector<int> h_cmp(kN, 0);

  for (int i = 0; i < kN; ++i) {
    h_a[static_cast<std::size_t>(i)] = 12.34567890123456 + static_cast<double>(i) * 1.5;
    h_b[static_cast<std::size_t>(i)] = 3.14159265358979 + static_cast<double>(i) * 0.125;
  }
  // Edge cases
  h_a[0] = 1.0;
  h_b[0] = 3.0;
  h_a[1] = 1000000.0;
  h_b[1] = 0.000001;
  h_a[2] = 2.0;
  h_b[2] = 2.0;
  h_a[3] = 0.25;
  h_b[3] = 4.0;
  h_a[4] = 7.0;
  h_b[4] = 5.0;

  const std::size_t bytes_d = static_cast<std::size_t>(kN) * sizeof(double);
  const std::size_t bytes_i = static_cast<std::size_t>(kN) * sizeof(int);

  CUdeviceptr d_a = 0;
  CUdeviceptr d_b = 0;
  CUdeviceptr d_add = 0;
  CUdeviceptr d_sub = 0;
  CUdeviceptr d_mul = 0;
  CUdeviceptr d_div = 0;
  CUdeviceptr d_sqrt = 0;
  CUdeviceptr d_cmp = 0;
  cuMemAlloc(&d_a, bytes_d);
  cuMemAlloc(&d_b, bytes_d);
  cuMemAlloc(&d_add, bytes_d);
  cuMemAlloc(&d_sub, bytes_d);
  cuMemAlloc(&d_mul, bytes_d);
  cuMemAlloc(&d_div, bytes_d);
  cuMemAlloc(&d_sqrt, bytes_d);
  cuMemAlloc(&d_cmp, bytes_i);

  cuMemcpyHtoD(d_a, h_a.data(), bytes_d);
  cuMemcpyHtoD(d_b, h_b.data(), bytes_d);

  int n = kN;
  void* args[] = {&d_a, &d_b, &d_add, &d_sub, &d_mul, &d_div, &d_sqrt, &d_cmp, &n};
  if (!Expect(cuLaunchKernel(fn, 1, 1, 1, kN, 1, 1, 0, nullptr, args, nullptr) == CUDA_SUCCESS,
              "cuLaunchKernel failed for IEEE FP64 kernel")) {
    return false;
  }
  cuCtxSynchronize();

  cuMemcpyDtoH(h_add.data(), d_add, bytes_d);
  cuMemcpyDtoH(h_sub.data(), d_sub, bytes_d);
  cuMemcpyDtoH(h_mul.data(), d_mul, bytes_d);
  cuMemcpyDtoH(h_div.data(), d_div, bytes_d);
  cuMemcpyDtoH(h_sqrt.data(), d_sqrt, bytes_d);
  cuMemcpyDtoH(h_cmp.data(), d_cmp, bytes_i);

  auto to_u64 = [](double d) {
    uint64_t u = 0;
    std::memcpy(&u, &d, sizeof(double));
    return u;
  };

  for (int i = 0; i < kN; ++i) {
    const double x = h_a[static_cast<std::size_t>(i)];
    const double y = h_b[static_cast<std::size_t>(i)];

    const double exp_add = x + y;
    const double exp_sub = x - y;
    const double exp_mul = x * y;
    const double exp_div = x / y;
    const double exp_sqrt = std::sqrt(x);
    const int exp_cmp = (x < y) ? 1 : 0;

    // Check addition
    const uint64_t u_add = to_u64(h_add[static_cast<std::size_t>(i)]);
    const uint64_t u_exp_add = to_u64(exp_add);
    const int64_t diff_add =
        std::abs(static_cast<int64_t>(u_add) - static_cast<int64_t>(u_exp_add));
    if (!Expect(diff_add == 0, "IEEE FP64 add bit mismatch")) {
      std::cerr << "i=" << i << " x=" << x << " y=" << y << " gpu=" << u_add << " cpu=" << u_exp_add
                << "\n";
      return false;
    }

    // Check subtraction
    const uint64_t u_sub = to_u64(h_sub[static_cast<std::size_t>(i)]);
    const uint64_t u_exp_sub = to_u64(exp_sub);
    const int64_t diff_sub =
        std::abs(static_cast<int64_t>(u_sub) - static_cast<int64_t>(u_exp_sub));
    if (!Expect(diff_sub == 0, "IEEE FP64 sub bit mismatch")) {
      std::cerr << "i=" << i << " x=" << x << " y=" << y << " gpu=" << u_sub << " cpu=" << u_exp_sub
                << "\n";
      return false;
    }

    // Check multiplication
    const uint64_t u_mul = to_u64(h_mul[static_cast<std::size_t>(i)]);
    const uint64_t u_exp_mul = to_u64(exp_mul);
    const int64_t diff_mul =
        std::abs(static_cast<int64_t>(u_mul) - static_cast<int64_t>(u_exp_mul));
    if (!Expect(diff_mul <= 1, "IEEE FP64 mul bit mismatch (>1 ULP)")) {
      std::cerr << "i=" << i << " x=" << x << " y=" << y << " gpu=" << u_mul << " cpu=" << u_exp_mul
                << "\n";
      return false;
    }

    // Check division
    const uint64_t u_div = to_u64(h_div[static_cast<std::size_t>(i)]);
    const uint64_t u_exp_div = to_u64(exp_div);
    const int64_t diff_div =
        std::abs(static_cast<int64_t>(u_div) - static_cast<int64_t>(u_exp_div));
    if (!Expect(diff_div <= 1, "IEEE FP64 div bit mismatch (>1 ULP)")) {
      std::cerr << "i=" << i << " x=" << x << " y=" << y << " gpu=" << u_div << " cpu=" << u_exp_div
                << "\n";
      return false;
    }

    // Check sqrt
    const uint64_t u_sqrt = to_u64(h_sqrt[static_cast<std::size_t>(i)]);
    const uint64_t u_exp_sqrt = to_u64(exp_sqrt);
    const int64_t diff_sqrt =
        std::abs(static_cast<int64_t>(u_sqrt) - static_cast<int64_t>(u_exp_sqrt));
    if (!Expect(diff_sqrt <= 1, "IEEE FP64 sqrt bit mismatch (>1 ULP)")) {
      std::cerr << "i=" << i << " x=" << x << " gpu=" << u_sqrt << " cpu=" << u_exp_sqrt << "\n";
      return false;
    }

    // Check comparison
    if (!Expect(h_cmp[static_cast<std::size_t>(i)] == exp_cmp, "IEEE FP64 comparison mismatch")) {
      return false;
    }
  }

  cuMemFree(d_a);
  cuMemFree(d_b);
  cuMemFree(d_add);
  cuMemFree(d_sub);
  cuMemFree(d_mul);
  cuMemFree(d_div);
  cuMemFree(d_sqrt);
  cuMemFree(d_cmp);
  cuModuleUnload(mod);
  return true;
}

}  // namespace

int main() {
  if (!TestFp64OffRejection() || !TestFp64Fast48Math() || !TestFp64SharedMemory() ||
      !TestFp64Ieee64Math()) {
    return 1;
  }
  std::cout << "ALL FP64 TESTS PASSED!\n";
  return 0;
}

/**
 * @file half_vector_test.cpp
 * @brief Tests half-precision (FP16) and vector data types (float4, int4, half2) on Apple Silicon GPU.
 */

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "cuda_runtime.h"
#include "muda/compiler/driver.hpp"
#include "muda/compiler/toolchain.hpp"
#include "muda/compiler/translator.hpp"
#include "test_utils.hpp"

namespace {

bool Expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    return false;
  }
  return true;
}

constexpr const char* kHalfVectorCudaSource = R"CUDA(
// 1. FP16 half scalar arithmetic and conversions
__global__ void test_half_scalar(const float* in_a, const float* in_b, float* out, int n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    half ha = __float2half(in_a[idx]);
    half hb = __float2half(in_b[idx]);
    half h_sum = __hadd(ha, hb);
    half h_prod = __hmul(ha, hb);
    half h_res = __hfma(h_sum, hb, h_prod);
    out[idx] = __half2float(h_res);
  }
}

// 2. FP16 half2 SIMD vector operations
__global__ void test_half2_vector(const float* in_a, const float* in_b, float* out, int n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    half2 h2_a = __floats2half2_rn(in_a[idx * 2], in_a[idx * 2 + 1]);
    half2 h2_b = __floats2half2_rn(in_b[idx * 2], in_b[idx * 2 + 1]);
    half2 h2_res = __hfma2(h2_a, h2_b, h2_a);
    half h_x = h2_res.x;
    half h_y = h2_res.y;
    out[idx * 2] = __half2float(h_x);
    out[idx * 2 + 1] = __half2float(h_y);
  }
}

// 3. float4 vectorized memory loads, stores, and arithmetic
__global__ void test_float4_vector(const float4* in_a, const float4* in_b, float4* out, int n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float4 a = in_a[idx];
    float4 b = in_b[idx];
    float4 c = make_float4(a.x + b.x, a.y - b.y, a.z * b.z, a.w + 10.0f);
    out[idx] = c;
  }
}

// 4. int4 vector arithmetic
__global__ void test_int4_vector(const int4* in_a, const int4* in_b, int4* out, int n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    int4 a = in_a[idx];
    int4 b = in_b[idx];
    int4 c = make_int4(a.x + b.x, a.y * b.y, a.z - b.z, a.w ^ b.w);
    out[idx] = c;
  }
}
)CUDA";

void HostStubHalfScalar() {}
void HostStubHalf2Vector() {}
void HostStubFloat4Vector() {}
void HostStubInt4Vector() {}

bool RegisterKernel(const void* host_func, const std::filesystem::path& metallib,
                    const muda::compiler::TranslatedKernel& kernel) {
  std::vector<mudaKernelParamDesc> descs;
  descs.reserve(kernel.parameters.size());
  for (const auto& param : kernel.parameters) {
    descs.push_back({
        .kind = param.is_pointer() ? mudaKernelParamBuffer : mudaKernelParamBytes,
        .size = param.byte_size(),
    });
  }
  return mudaRegisterKernelFromFile(host_func, metallib.c_str(), kernel.name.c_str(), descs.data(),
                                    descs.size()) == cudaSuccess;
}

}  // namespace

int main() {
  if (!muda::compiler::MetalToolchainAvailable()) {
    std::cout << "skip: Metal toolchain unavailable\n";
    return 77;
  }

  const muda::testing::ScopedTempDir temp_dir("muda_half_vector_test");
  if (!temp_dir.valid()) {
    std::cerr << "cannot create temp directory\n";
    return 1;
  }

  const std::filesystem::path cu_path = temp_dir.path() / "half_vec.cu";
  const std::filesystem::path metallib_path = temp_dir.path() / "half_vec.metallib";

  {
    std::ofstream cu_file(cu_path, std::ios::binary);
    cu_file.write(kHalfVectorCudaSource,
                 static_cast<std::streamsize>(std::string_view(kHalfVectorCudaSource).size()));
  }

  const muda::compiler::TranslationResult tr = muda::compiler::TranslateCudaToMsl(kHalfVectorCudaSource);
  if (!Expect(tr.ok(), "half/vector translation should succeed") ||
      !Expect(tr.kernels.size() == 4u, "expected 4 translated kernels")) {
    return 1;
  }

  muda::compiler::CompileOptions options;
  options.input = cu_path;
  options.output = metallib_path;
  options.output_kind = muda::compiler::OutputKind::kMetallib;

  const muda::compiler::CompileResult compiled = muda::compiler::Compile(options);
  if (!Expect(compiled.status == muda::compiler::Status::kOk, "half/vector compilation should succeed")) {
    return 1;
  }

  if (!Expect(RegisterKernel(reinterpret_cast<const void*>(&HostStubHalfScalar), metallib_path, tr.kernels[0]),
              "register half_scalar kernel failed") ||
      !Expect(RegisterKernel(reinterpret_cast<const void*>(&HostStubHalf2Vector), metallib_path, tr.kernels[1]),
              "register half2_vector kernel failed") ||
      !Expect(RegisterKernel(reinterpret_cast<const void*>(&HostStubFloat4Vector), metallib_path, tr.kernels[2]),
              "register float4_vector kernel failed") ||
      !Expect(RegisterKernel(reinterpret_cast<const void*>(&HostStubInt4Vector), metallib_path, tr.kernels[3]),
              "register int4_vector kernel failed")) {
    return 1;
  }

  // -------------------------------------------------------------------------
  // Test 1: FP16 half scalar arithmetic
  // -------------------------------------------------------------------------
  {
    constexpr std::size_t kN = 256;
    std::vector<float> h_a(kN), h_b(kN);
    for (std::size_t i = 0; i < kN; ++i) {
      h_a[i] = static_cast<float>(i % 8) * 0.5f + 1.0f;
      h_b[i] = 2.0f;
    }

    float *d_a = nullptr, *d_b = nullptr, *d_out = nullptr;
    (void)cudaMalloc(reinterpret_cast<void**>(&d_a), kN * sizeof(float));
    (void)cudaMalloc(reinterpret_cast<void**>(&d_b), kN * sizeof(float));
    (void)cudaMalloc(reinterpret_cast<void**>(&d_out), kN * sizeof(float));

    (void)cudaMemcpy(d_a, h_a.data(), kN * sizeof(float), cudaMemcpyHostToDevice);
    (void)cudaMemcpy(d_b, h_b.data(), kN * sizeof(float), cudaMemcpyHostToDevice);

    int n_arg = static_cast<int>(kN);
    void* args[] = {&d_a, &d_b, &d_out, &n_arg};
    const auto err = cudaLaunchKernel(reinterpret_cast<const void*>(&HostStubHalfScalar), dim3(1),
                                      dim3(static_cast<unsigned int>(kN)), args, 0, nullptr);
    if (!Expect(err == cudaSuccess, "launch half_scalar kernel failed")) {
      return 1;
    }
    (void)cudaDeviceSynchronize();

    std::vector<float> h_out(kN, 0.0f);
    (void)cudaMemcpy(h_out.data(), d_out, kN * sizeof(float), cudaMemcpyDeviceToHost);

    // ha = a, hb = 2, h_sum = a + 2, h_prod = a * 2, h_res = (a + 2) * 2 + (a * 2) = 4a + 4
    for (std::size_t i = 0; i < kN; ++i) {
      const float expected = 4.0f * h_a[i] + 4.0f;
      if (!Expect(std::fabs(h_out[i] - expected) < 1e-2f, "half scalar result mismatch")) {
        std::cerr << "half[" << i << "] got " << h_out[i] << " expected " << expected << "\n";
        return 1;
      }
    }

    (void)cudaFree(d_a);
    (void)cudaFree(d_b);
    (void)cudaFree(d_out);
  }

  // -------------------------------------------------------------------------
  // Test 2: FP16 half2 vector operations
  // -------------------------------------------------------------------------
  {
    constexpr std::size_t kN = 128;
    constexpr std::size_t kFloats = kN * 2;
    std::vector<float> h_a(kFloats), h_b(kFloats);
    for (std::size_t i = 0; i < kFloats; ++i) {
      h_a[i] = static_cast<float>(i + 1) * 0.25f;
      h_b[i] = 1.5f;
    }

    float *d_a = nullptr, *d_b = nullptr, *d_out = nullptr;
    (void)cudaMalloc(reinterpret_cast<void**>(&d_a), kFloats * sizeof(float));
    (void)cudaMalloc(reinterpret_cast<void**>(&d_b), kFloats * sizeof(float));
    (void)cudaMalloc(reinterpret_cast<void**>(&d_out), kFloats * sizeof(float));

    (void)cudaMemcpy(d_a, h_a.data(), kFloats * sizeof(float), cudaMemcpyHostToDevice);
    (void)cudaMemcpy(d_b, h_b.data(), kFloats * sizeof(float), cudaMemcpyHostToDevice);

    int n_arg = static_cast<int>(kN);
    void* args[] = {&d_a, &d_b, &d_out, &n_arg};
    const auto err = cudaLaunchKernel(reinterpret_cast<const void*>(&HostStubHalf2Vector), dim3(1),
                                      dim3(static_cast<unsigned int>(kN)), args, 0, nullptr);
    if (!Expect(err == cudaSuccess, "launch half2_vector kernel failed")) {
      return 1;
    }
    (void)cudaDeviceSynchronize();

    std::vector<float> h_out(kFloats, 0.0f);
    (void)cudaMemcpy(h_out.data(), d_out, kFloats * sizeof(float), cudaMemcpyDeviceToHost);

    // h2_res = a * 1.5f + a = 2.5f * a
    for (std::size_t i = 0; i < kFloats; ++i) {
      const float expected = 2.5f * h_a[i];
      if (!Expect(std::fabs(h_out[i] - expected) < 2e-2f, "half2 result mismatch")) {
        std::cerr << "half2[" << i << "] got " << h_out[i] << " expected " << expected << "\n";
        return 1;
      }
    }

    (void)cudaFree(d_a);
    (void)cudaFree(d_b);
    (void)cudaFree(d_out);
  }

  // -------------------------------------------------------------------------
  // Test 3: float4 vector arithmetic
  // -------------------------------------------------------------------------
  {
    constexpr std::size_t kN = 64;
    std::vector<float4> h_a(kN), h_b(kN);
    for (std::size_t i = 0; i < kN; ++i) {
      h_a[i] = make_float4(static_cast<float>(i), 10.0f, 2.0f, 5.0f);
      h_b[i] = make_float4(1.0f, 4.0f, 3.0f, 0.0f);
    }

    float4 *d_a = nullptr, *d_b = nullptr, *d_out = nullptr;
    (void)cudaMalloc(reinterpret_cast<void**>(&d_a), kN * sizeof(float4));
    (void)cudaMalloc(reinterpret_cast<void**>(&d_b), kN * sizeof(float4));
    (void)cudaMalloc(reinterpret_cast<void**>(&d_out), kN * sizeof(float4));

    (void)cudaMemcpy(d_a, h_a.data(), kN * sizeof(float4), cudaMemcpyHostToDevice);
    (void)cudaMemcpy(d_b, h_b.data(), kN * sizeof(float4), cudaMemcpyHostToDevice);

    int n_arg = static_cast<int>(kN);
    void* args[] = {&d_a, &d_b, &d_out, &n_arg};
    const auto err = cudaLaunchKernel(reinterpret_cast<const void*>(&HostStubFloat4Vector), dim3(1),
                                      dim3(static_cast<unsigned int>(kN)), args, 0, nullptr);
    if (!Expect(err == cudaSuccess, "launch float4_vector kernel failed")) {
      return 1;
    }
    (void)cudaDeviceSynchronize();

    std::vector<float4> h_out(kN);
    (void)cudaMemcpy(h_out.data(), d_out, kN * sizeof(float4), cudaMemcpyDeviceToHost);

    for (std::size_t i = 0; i < kN; ++i) {
      if (!Expect(std::fabs(h_out[i].x - (static_cast<float>(i) + 1.0f)) < 1e-4f, "float4.x mismatch") ||
          !Expect(std::fabs(h_out[i].y - 6.0f) < 1e-4f, "float4.y mismatch") ||
          !Expect(std::fabs(h_out[i].z - 6.0f) < 1e-4f, "float4.z mismatch") ||
          !Expect(std::fabs(h_out[i].w - 15.0f) < 1e-4f, "float4.w mismatch")) {
        std::cerr << "float4 mismatch at " << i << "\n";
        return 1;
      }
    }

    (void)cudaFree(d_a);
    (void)cudaFree(d_b);
    (void)cudaFree(d_out);
  }

  // -------------------------------------------------------------------------
  // Test 4: int4 vector arithmetic
  // -------------------------------------------------------------------------
  {
    constexpr std::size_t kN = 64;
    std::vector<int4> h_a(kN), h_b(kN);
    for (std::size_t i = 0; i < kN; ++i) {
      h_a[i] = make_int4(static_cast<int>(i), 5, 20, 0x0F);
      h_b[i] = make_int4(10, 4, 3, 0xFF);
    }

    int4 *d_a = nullptr, *d_b = nullptr, *d_out = nullptr;
    (void)cudaMalloc(reinterpret_cast<void**>(&d_a), kN * sizeof(int4));
    (void)cudaMalloc(reinterpret_cast<void**>(&d_b), kN * sizeof(int4));
    (void)cudaMalloc(reinterpret_cast<void**>(&d_out), kN * sizeof(int4));

    (void)cudaMemcpy(d_a, h_a.data(), kN * sizeof(int4), cudaMemcpyHostToDevice);
    (void)cudaMemcpy(d_b, h_b.data(), kN * sizeof(int4), cudaMemcpyHostToDevice);

    int n_arg = static_cast<int>(kN);
    void* args[] = {&d_a, &d_b, &d_out, &n_arg};
    const auto err = cudaLaunchKernel(reinterpret_cast<const void*>(&HostStubInt4Vector), dim3(1),
                                      dim3(static_cast<unsigned int>(kN)), args, 0, nullptr);
    if (!Expect(err == cudaSuccess, "launch int4_vector kernel failed")) {
      return 1;
    }
    (void)cudaDeviceSynchronize();

    std::vector<int4> h_out(kN);
    (void)cudaMemcpy(h_out.data(), d_out, kN * sizeof(int4), cudaMemcpyDeviceToHost);

    for (std::size_t i = 0; i < kN; ++i) {
      if (!Expect(h_out[i].x == (static_cast<int>(i) + 10), "int4.x mismatch") ||
          !Expect(h_out[i].y == 20, "int4.y mismatch") ||
          !Expect(h_out[i].z == 17, "int4.z mismatch") ||
          !Expect(h_out[i].w == (0x0F ^ 0xFF), "int4.w mismatch")) {
        std::cerr << "int4 mismatch at " << i << "\n";
        return 1;
      }
    }

    (void)cudaFree(d_a);
    (void)cudaFree(d_b);
    (void)cudaFree(d_out);
  }

  std::cout << "half_vector_test: ALL PASSED\n";
  return 0;
}


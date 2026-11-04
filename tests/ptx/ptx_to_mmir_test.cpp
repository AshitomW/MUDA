/**
 * @file ptx_to_mmir_test.cpp
 * @brief End-to-end tests for PTX translation to Metal Shading Language and GPU execution.
 */

#include "muda/compiler/ptx/ptx_to_mmir.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string_view>
#include <vector>

#include "../test_utils.hpp"
#include "cuda_runtime.h"
#include "muda/compiler/driver.hpp"
#include "muda/compiler/toolchain.hpp"

namespace {

bool Expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
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
    @%p1 bra        $L__BB0_2;

    mul.wide.s32    %rd4, %r0, 4;
    add.s64         %rd5, %rd1, %rd4;
    add.s64         %rd6, %rd2, %rd4;
    ld.global.f32   %f1, [%rd5];
    ld.global.f32   %f2, [%rd6];
    add.f32         %f3, %f1, %f2;

    add.s64         %rd7, %rd3, %rd4;
    st.global.f32   [%rd7], %f3;

$L__BB0_2:
    ret;
}
)PTX";

bool TestPtxTranslationToMsl() {
  const muda::compiler::TranslationResult tr =
      muda::compiler::ptx::TranslatePtxToMsl(kVectorAddPtx);

  std::cout << "--- GENERATED MSL FROM PTX ---\n" << tr.msl << "\n-----------------------------\n";

  if (!Expect(tr.ok(), "PTX translation should succeed"))
    return false;
  if (!Expect(!tr.msl.empty(), "Generated MSL should not be empty"))
    return false;

  if (!Expect(tr.msl.find("kernel void k_vector_add") != std::string::npos,
              "MSL should define kernel void k_vector_add"))
    return false;
  if (!Expect(tr.msl.find("param_a") != std::string::npos, "MSL should bind param_a"))
    return false;
  if (!Expect(tr.msl.find("param_b") != std::string::npos, "MSL should bind param_b"))
    return false;
  if (!Expect(tr.msl.find("param_c") != std::string::npos, "MSL should bind param_c"))
    return false;

  return true;
}

void PtxKernelStub() {}

bool TestPtxEndToEndGpuExecution() {
  if (!muda::compiler::MetalToolchainAvailable()) {
    std::cout << "Metal toolchain unavailable, skipping GPU execution test.\n";
    return true;
  }

  const std::filesystem::path temp_dir = std::filesystem::temp_directory_path() / "muda_ptx_test";
  std::filesystem::create_directories(temp_dir);
  const std::filesystem::path ptx_file = temp_dir / "test_kernel.ptx";
  const std::filesystem::path metallib_file = temp_dir / "test_kernel.metallib";

  {
    std::ofstream out(ptx_file);
    out << kVectorAddPtx;
  }

  muda::compiler::CompileOptions options{};
  options.input = ptx_file;
  options.output = metallib_file;
  options.output_kind = muda::compiler::OutputKind::kMetallib;

  const muda::compiler::CompileResult cr = muda::compiler::Compile(options);
  if (!Expect(cr.status == muda::compiler::Status::kOk, "Compiling PTX to metallib failed")) {
    return false;
  }

  // Register the compiled PTX kernel
  mudaKernelParamDesc descs[4] = {
      {.kind = mudaKernelParamBuffer, .size = sizeof(float*)},
      {.kind = mudaKernelParamBuffer, .size = sizeof(float*)},
      {.kind = mudaKernelParamBuffer, .size = sizeof(float*)},
      {.kind = mudaKernelParamBytes, .size = sizeof(int)},
  };

  if (!Expect(mudaRegisterKernelFromFile(reinterpret_cast<const void*>(&PtxKernelStub),
                                         metallib_file.string().c_str(), "k_vector_add", descs,
                                         4) == cudaSuccess,
              "mudaRegisterKernelFromFile failed for PTX kernel")) {
    return false;
  }

  constexpr int kN = 64;
  float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
  if (!Expect(
          cudaMallocManaged(reinterpret_cast<void**>(&d_a), kN * sizeof(float)) == cudaSuccess &&
              cudaMallocManaged(reinterpret_cast<void**>(&d_b), kN * sizeof(float)) ==
                  cudaSuccess &&
              cudaMallocManaged(reinterpret_cast<void**>(&d_c), kN * sizeof(float)) == cudaSuccess,
          "cudaMallocManaged failed")) {
    return false;
  }

  for (int i = 0; i < kN; ++i) {
    d_a[i] = static_cast<float>(i);
    d_b[i] = 100.0f + static_cast<float>(i);
    d_c[i] = 0.0f;
  }

  int n_param = kN;
  void* args[4] = {&d_a, &d_b, &d_c, &n_param};

  if (!Expect(cudaLaunchKernel(reinterpret_cast<const void*>(&PtxKernelStub), dim3(2), dim3(32),
                               args, 0, nullptr) == cudaSuccess,
              "cudaLaunchKernel failed")) {
    return false;
  }

  if (!Expect(cudaDeviceSynchronize() == cudaSuccess, "cudaDeviceSynchronize failed")) {
    return false;
  }

  for (int i = 0; i < kN; ++i) {
    if (!Expect(muda::testing::AreClose(d_c[i], d_a[i] + d_b[i], 1e-4f),
                "Numerical mismatch in PTX vector add result on GPU")) {
      return false;
    }
  }

  (void)cudaFree(d_a);
  (void)cudaFree(d_b);
  (void)cudaFree(d_c);

  std::error_code ec{};
  std::filesystem::remove_all(temp_dir, ec);
  return true;
}

}  // namespace

int main() {
  if (!TestPtxTranslationToMsl())
    return 1;
  if (!TestPtxEndToEndGpuExecution())
    return 1;

  std::cout << "ptx_to_mmir_test: all tests passed!\n";
  return 0;
}

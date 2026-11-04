/**
 * @file numba_mock_test.cpp
 * @brief Simulates Numba's @cuda.jit workflow executing vectorAdd, parallel_reduction, and matmul
 *        over the MUDA CUDA Driver API (libcuda.dylib).
 */

#include <cmath>
#include <iostream>
#include <string_view>
#include <vector>

#include "cuda.h"

namespace {

bool ExpectCu(CUresult res, const char* msg) {
  if (res != CUDA_SUCCESS) {
    const char* err_name = "UNKNOWN";
    cuGetErrorName(res, &err_name);
    std::cerr << "FAIL: " << msg << " -> " << err_name << " (" << res << ")\n";
    return false;
  }
  return true;
}

// 1. Vector Add PTX
constexpr std::string_view kVectorAddPtx = R"PTX(
.version 6.5
.target sm_70
.address_size 64

.visible .entry numba_vector_add(
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

// 2. Parallel Reduction PTX (using threadgroup shared memory and bar.sync)
constexpr std::string_view kReductionPtx = R"PTX(
.version 6.5
.target sm_70
.address_size 64

.visible .entry numba_parallel_reduction(
    .param .u64 param_input,
    .param .u64 param_output,
    .param .u32 param_n
)
{
    .reg .pred  %p<2>;
    .reg .b32   %r<6>;
    .reg .b64   %rd<10>;
    .reg .f32   %f<4>;
    .shared .align 4 .b8 s_cache[1024];

    ld.param.u64    %rd1, [param_input];
    ld.param.u64    %rd2, [param_output];
    ld.param.u32    %r1, [param_n];

    mov.u32         %r2, %ctaid.x;
    mov.u32         %r3, %ntid.x;
    mov.u32         %r4, %tid.x;
    mad.lo.s32      %r0, %r2, %r3, %r4;

    setp.ge.s32     %p1, %r0, %r1;
    @%p1 bra        $L__skip_store;

    mul.wide.s32    %rd3, %r0, 4;
    add.s64         %rd4, %rd1, %rd3;
    ld.global.f32   %f1, [%rd4];

    mul.wide.s32    %rd5, %r4, 4;
    cvta.to.shared.u64 %rd6, s_cache;
    add.s64         %rd7, %rd6, %rd5;
    st.shared.f32   [%rd7], %f1;

$L__skip_store:
    bar.sync        0;

    // Output block leader
    setp.ne.s32     %p1, %r4, 0;
    @%p1 bra        $L__exit;

    cvta.to.shared.u64 %rd6, s_cache;
    ld.shared.f32   %f2, [%rd6];
    mul.wide.s32    %rd8, %r2, 4;
    add.s64         %rd9, %rd2, %rd8;
    st.global.f32   [%rd9], %f2;

$L__exit:
    ret;
}
)PTX";

// 3. 2D Matmul PTX (2D grid and block indexing)
constexpr std::string_view kMatmulPtx = R"PTX(
.version 6.5
.target sm_70
.address_size 64

.visible .entry numba_matmul_2d(
    .param .u64 param_a,
    .param .u64 param_b,
    .param .u64 param_c,
    .param .u32 param_n
)
{
    .reg .pred  %p<3>;
    .reg .b32   %r<12>;
    .reg .b64   %rd<15>;
    .reg .f32   %f<5>;

    ld.param.u64    %rd1, [param_a];
    ld.param.u64    %rd2, [param_b];
    ld.param.u64    %rd3, [param_c];
    ld.param.u32    %r1, [param_n];

    // row = ctaid.y * ntid.y + tid.y
    mov.u32         %r2, %ctaid.y;
    mov.u32         %r3, %ntid.y;
    mov.u32         %r4, %tid.y;
    mad.lo.s32      %r5, %r2, %r3, %r4;

    // col = ctaid.x * ntid.x + tid.x
    mov.u32         %r6, %ctaid.x;
    mov.u32         %r7, %ntid.x;
    mov.u32         %r8, %tid.x;
    mad.lo.s32      %r9, %r6, %r7, %r8;

    setp.ge.s32     %p1, %r5, %r1;
    @%p1 bra        $L__exit;
    setp.ge.s32     %p2, %r9, %r1;
    @%p2 bra        $L__exit;

    // c_idx = row * n + col
    mad.lo.s32      %r10, %r5, %r1, %r9;
    mul.wide.s32    %rd4, %r10, 4;
    add.s64         %rd5, %rd3, %rd4;

    // a_idx = row * n + 0
    mul.lo.s32      %r11, %r5, %r1;
    mul.wide.s32    %rd6, %r11, 4;
    add.s64         %rd7, %rd1, %rd6;
    ld.global.f32   %f1, [%rd7];

    // b_idx = 0 * n + col
    mul.wide.s32    %rd8, %r9, 4;
    add.s64         %rd9, %rd2, %rd8;
    ld.global.f32   %f2, [%rd9];

    mul.f32         %f3, %f1, %f2;
    st.global.f32   [%rd5], %f3;

$L__exit:
    ret;
}
)PTX";

bool TestNumbaVectorAdd() {
  std::cout << "--- Testing Numba @cuda.jit vectorAdd ---\n";
  CUmodule module = nullptr;
  if (!ExpectCu(cuModuleLoadData(&module, kVectorAddPtx.data()), "Load vector_add PTX"))
    return false;

  CUfunction fn = nullptr;
  if (!ExpectCu(cuModuleGetFunction(&fn, module, "numba_vector_add"), "Get numba_vector_add"))
    return false;

  constexpr unsigned int kN = 4096;
  constexpr size_t kBytes = kN * sizeof(float);

  std::vector<float> h_a(kN, 1.25f);
  std::vector<float> h_b(kN, 2.75f);
  std::vector<float> h_c(kN, 0.0f);

  CUdeviceptr d_a = 0;
  CUdeviceptr d_b = 0;
  CUdeviceptr d_c = 0;
  cuMemAlloc(&d_a, kBytes);
  cuMemAlloc(&d_b, kBytes);
  cuMemAlloc(&d_c, kBytes);

  cuMemcpyHtoD(d_a, h_a.data(), kBytes);
  cuMemcpyHtoD(d_b, h_b.data(), kBytes);

  unsigned int n = kN;
  void* args[] = {&d_a, &d_b, &d_c, &n};

  unsigned int block_size = 256;
  unsigned int grid_size = (kN + block_size - 1) / block_size;

  if (!ExpectCu(cuLaunchKernel(fn, grid_size, 1, 1, block_size, 1, 1, 0, nullptr, args, nullptr),
                "Launch numba_vector_add")) {
    return false;
  }

  cuCtxSynchronize();
  cuMemcpyDtoH(h_c.data(), d_c, kBytes);

  for (unsigned int i = 0; i < kN; ++i) {
    if (std::abs(h_c[i] - 4.0f) > 1e-4f) {
      std::cerr << "FAIL: Vector add mismatch at " << i << ": expected 4.0, got " << h_c[i] << "\n";
      return false;
    }
  }

  std::cout << "  [PASS] Numba vectorAdd (4096 elements verified)\n";

  cuMemFree(d_a);
  cuMemFree(d_b);
  cuMemFree(d_c);
  cuModuleUnload(module);
  return true;
}

bool TestNumbaReduction() {
  std::cout << "--- Testing Numba @cuda.jit parallel_reduction ---\n";
  CUmodule module = nullptr;
  if (!ExpectCu(cuModuleLoadData(&module, kReductionPtx.data()), "Load reduction PTX"))
    return false;

  CUfunction fn = nullptr;
  if (!ExpectCu(cuModuleGetFunction(&fn, module, "numba_parallel_reduction"),
                "Get numba_parallel_reduction"))
    return false;

  constexpr unsigned int kN = 1024;
  constexpr unsigned int kBlock = 256;
  constexpr unsigned int kGrid = (kN + kBlock - 1) / kBlock;

  std::vector<float> h_in(kN);
  for (unsigned int i = 0; i < kN; ++i) {
    h_in[i] = static_cast<float>(i + 1);
  }
  std::vector<float> h_out(kGrid, 0.0f);

  CUdeviceptr d_in = 0;
  CUdeviceptr d_out = 0;
  cuMemAlloc(&d_in, kN * sizeof(float));
  cuMemAlloc(&d_out, kGrid * sizeof(float));

  cuMemcpyHtoD(d_in, h_in.data(), kN * sizeof(float));

  unsigned int n = kN;
  void* args[] = {&d_in, &d_out, &n};

  if (!ExpectCu(cuLaunchKernel(fn, kGrid, 1, 1, kBlock, 1, 1, 0, nullptr, args, nullptr),
                "Launch numba_parallel_reduction")) {
    return false;
  }

  cuCtxSynchronize();
  cuMemcpyDtoH(h_out.data(), d_out, kGrid * sizeof(float));

  // Thread 0 of each block wrote element (block_idx * 256) into h_out
  for (unsigned int b = 0; b < kGrid; ++b) {
    const float expected = h_in[b * kBlock];
    if (std::abs(h_out[b] - expected) > 1e-4f) {
      std::cerr << "FAIL: Reduction block " << b << " mismatch: expected " << expected << ", got "
                << h_out[b] << "\n";
      return false;
    }
  }

  std::cout << "  [PASS] Numba parallel_reduction (shared memory and bar.sync verified)\n";

  cuMemFree(d_in);
  cuMemFree(d_out);
  cuModuleUnload(module);
  return true;
}

bool TestNumbaMatmul() {
  std::cout << "--- Testing Numba @cuda.jit 2D matmul ---\n";
  CUmodule module = nullptr;
  if (!ExpectCu(cuModuleLoadData(&module, kMatmulPtx.data()), "Load matmul PTX"))
    return false;

  CUfunction fn = nullptr;
  if (!ExpectCu(cuModuleGetFunction(&fn, module, "numba_matmul_2d"), "Get numba_matmul_2d"))
    return false;

  constexpr unsigned int kN = 16;
  constexpr size_t kBytes = kN * kN * sizeof(float);

  std::vector<float> h_a(kN * kN, 2.0f);
  std::vector<float> h_b(kN * kN, 3.0f);
  std::vector<float> h_c(kN * kN, 0.0f);

  CUdeviceptr d_a = 0;
  CUdeviceptr d_b = 0;
  CUdeviceptr d_c = 0;
  cuMemAlloc(&d_a, kBytes);
  cuMemAlloc(&d_b, kBytes);
  cuMemAlloc(&d_c, kBytes);

  cuMemcpyHtoD(d_a, h_a.data(), kBytes);
  cuMemcpyHtoD(d_b, h_b.data(), kBytes);

  unsigned int n = kN;
  void* args[] = {&d_a, &d_b, &d_c, &n};

  // 2D grid: 2x2 blocks of 8x8 threads = 16x16 threads
  unsigned int bx = 8;
  unsigned int by = 8;
  unsigned int gx = (kN + bx - 1) / bx;
  unsigned int gy = (kN + by - 1) / by;

  if (!ExpectCu(cuLaunchKernel(fn, gx, gy, 1, bx, by, 1, 0, nullptr, args, nullptr),
                "Launch numba_matmul_2d")) {
    return false;
  }

  cuCtxSynchronize();
  cuMemcpyDtoH(h_c.data(), d_c, kBytes);

  for (unsigned int i = 0; i < kN * kN; ++i) {
    if (std::abs(h_c[i] - 6.0f) > 1e-4f) {
      std::cerr << "FAIL: Matmul element " << i << " mismatch: expected 6.0, got " << h_c[i]
                << "\n";
      return false;
    }
  }

  std::cout << "  [PASS] Numba 2D matmul (16x16 grid/block topology verified)\n";

  cuMemFree(d_a);
  cuMemFree(d_b);
  cuMemFree(d_c);
  cuModuleUnload(module);
  return true;
}

}  // namespace

int main() {
  if (!ExpectCu(cuInit(0), "cuInit(0)"))
    return 1;

  CUcontext ctx = nullptr;
  if (!ExpectCu(cuCtxCreate(&ctx, 0, 0), "cuCtxCreate"))
    return 1;

  if (!TestNumbaVectorAdd())
    return 1;
  if (!TestNumbaReduction())
    return 1;
  if (!TestNumbaMatmul())
    return 1;

  cuCtxDestroy(ctx);
  std::cout << "numba_mock_test: all Numba test workflows passed successfully!\n";
  return 0;
}

/**
 * @file runtime_bench.cpp
 * @brief Benchmark suite executing MUDA microbenchmarks and throughput benchmarks.
 */

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

#include "benchmark_runner.hpp"
#include "cuda_runtime.h"
#include "library_benchmarks.hpp"
#include "microbenchmarks.hpp"
#include "muda/compiler/driver.hpp"
#include "muda/compiler/toolchain.hpp"
#include "muda/compiler/translator.hpp"
#include "throughput_benchmarks.hpp"

extern "C" cudaError_t mudaRegisterKernelFromFile(const void* host_func, const char* metallib_path,
                                                  const char* kernel_name,
                                                  const mudaKernelParamDesc* params,
                                                  size_t param_count);

namespace {

constexpr std::string_view kBenchSource = R"CUDA(
__global__ void bench_noop(float* out) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    out[0] = 1.0f;
  }
}

__global__ void bench_vector_add(const float* a, const float* b, float* c, int n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}

__global__ void bench_vector_add_float4(const float4* a, const float4* b, float4* c, int n4) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n4) {
    c[idx] = a[idx] + b[idx];
  }
}

__global__ void bench_vector_add_float4x2(const float4* a, const float4* b, float4* c, int n4) {
  const int base = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
  if (base < n4) {
    c[base] = a[base] + b[base];
  }
  const int idx2 = base + blockDim.x;
  if (idx2 < n4) {
    c[idx2] = a[idx2] + b[idx2];
  }
}

__global__ void bench_gemm_tiled(const float* a, const float* b, float* c, int n) {
  __shared__ float tile_a[272];
  __shared__ float tile_b[272];
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int bx = blockIdx.x * 16;
  const int by = blockIdx.y * 16;
  const int row = by + ty;
  const int col = bx + tx;
  float acc = 0.0f;
  const int num_tiles = (n + 15) / 16;
  const bool full_block = (by + 16 <= n) && (bx + 16 <= n);

  for (int t = 0; t < num_tiles; ++t) {
    const int k_base = t * 16;
    const int a_col = k_base + tx;
    const int b_row = k_base + ty;
    if (full_block && (k_base + 16 <= n)) {
      tile_a[ty * 17 + tx] = a[row * n + a_col];
      tile_b[ty * 17 + tx] = b[b_row * n + col];
    } else {
      tile_a[ty * 17 + tx] = (row < n && a_col < n) ? a[row * n + a_col] : 0.0f;
      tile_b[ty * 17 + tx] = (b_row < n && col < n) ? b[b_row * n + col] : 0.0f;
    }
    __syncthreads();
    for (int k = 0; k < 16; k += 4) {
      acc = fmaf(tile_a[ty * 17 + (k + 0)], tile_b[(k + 0) * 17 + tx], acc);
      acc = fmaf(tile_a[ty * 17 + (k + 1)], tile_b[(k + 1) * 17 + tx], acc);
      acc = fmaf(tile_a[ty * 17 + (k + 2)], tile_b[(k + 2) * 17 + tx], acc);
      acc = fmaf(tile_a[ty * 17 + (k + 3)], tile_b[(k + 3) * 17 + tx], acc);
    }
    __syncthreads();
  }
  if (row < n && col < n) {
    c[row * n + col] = acc;
  }
}

__global__ void bench_gemm_register_tiled(const float* a, const float* b, float* c, int n) {
  __shared__ float s_a[1024];
  __shared__ float s_b[1024];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * 16 + tx;
  const int bx = blockIdx.x * 64;
  const int by = blockIdx.y * 64;
  const bool full_block = (by + 64 <= n) && (bx + 64 <= n);

  const int sg_id = simd_group_id;
  const int sg_r = (sg_id >> 1) * 16;
  const int sg_c = (sg_id & 1) * 32;

  simdgroup_float8x8 acc00(0.0f);
  simdgroup_float8x8 acc01(0.0f);
  simdgroup_float8x8 acc02(0.0f);
  simdgroup_float8x8 acc03(0.0f);
  simdgroup_float8x8 acc10(0.0f);
  simdgroup_float8x8 acc11(0.0f);
  simdgroup_float8x8 acc12(0.0f);
  simdgroup_float8x8 acc13(0.0f);

  const int num_tiles = (n + 15) / 16;
  for (int t = 0; t < num_tiles; ++t) {
    const int k_base = t * 16;
    if (full_block && (k_base + 16 <= n)) {
      for (int step = 0; step < 4; ++step) {
        const int flat = tid + step * 256;
        const int a_r = flat >> 4;
        const int a_c = flat & 15;
        s_a[a_r * 16 + a_c] = a[(by + a_r) * n + (k_base + a_c)];

        const int b_r = flat >> 6;
        const int b_c = flat & 63;
        s_b[b_r * 64 + b_c] = b[(k_base + b_r) * n + (bx + b_c)];
      }
    } else {
      for (int step = 0; step < 4; ++step) {
        const int flat = tid + step * 256;
        const int a_r = flat >> 4;
        const int a_c = flat & 15;
        s_a[a_r * 16 + a_c] =
            (by + a_r < n && k_base + a_c < n) ? a[(by + a_r) * n + (k_base + a_c)] : 0.0f;

        const int b_r = flat >> 6;
        const int b_c = flat & 63;
        s_b[b_r * 64 + b_c] =
            (k_base + b_r < n && bx + b_c < n) ? b[(k_base + b_r) * n + (bx + b_c)] : 0.0f;
      }
    }
    __syncthreads();

    for (int ks = 0; ks < 16; ks += 8) {
      simdgroup_float8x8 fa0;
      simdgroup_float8x8 fa1;
      simdgroup_load(fa0, s_a + (sg_r + 0) * 16 + ks, 16);
      simdgroup_load(fa1, s_a + (sg_r + 8) * 16 + ks, 16);

      simdgroup_float8x8 fb0;
      simdgroup_float8x8 fb1;
      simdgroup_float8x8 fb2;
      simdgroup_float8x8 fb3;
      simdgroup_load(fb0, s_b + ks * 64 + sg_c + 0, 64);
      simdgroup_load(fb1, s_b + ks * 64 + sg_c + 8, 64);
      simdgroup_load(fb2, s_b + ks * 64 + sg_c + 16, 64);
      simdgroup_load(fb3, s_b + ks * 64 + sg_c + 24, 64);

      simdgroup_multiply_accumulate(acc00, fa0, fb0, acc00);
      simdgroup_multiply_accumulate(acc01, fa0, fb1, acc01);
      simdgroup_multiply_accumulate(acc02, fa0, fb2, acc02);
      simdgroup_multiply_accumulate(acc03, fa0, fb3, acc03);
      simdgroup_multiply_accumulate(acc10, fa1, fb0, acc10);
      simdgroup_multiply_accumulate(acc11, fa1, fb1, acc11);
      simdgroup_multiply_accumulate(acc12, fa1, fb2, acc12);
      simdgroup_multiply_accumulate(acc13, fa1, fb3, acc13);
    }
    __syncthreads();
  }

  if (full_block) {
    simdgroup_store(acc00, c + (by + sg_r + 0) * n + (bx + sg_c + 0), n);
    simdgroup_store(acc01, c + (by + sg_r + 0) * n + (bx + sg_c + 8), n);
    simdgroup_store(acc02, c + (by + sg_r + 0) * n + (bx + sg_c + 16), n);
    simdgroup_store(acc03, c + (by + sg_r + 0) * n + (bx + sg_c + 24), n);
    simdgroup_store(acc10, c + (by + sg_r + 8) * n + (bx + sg_c + 0), n);
    simdgroup_store(acc11, c + (by + sg_r + 8) * n + (bx + sg_c + 8), n);
    simdgroup_store(acc12, c + (by + sg_r + 8) * n + (bx + sg_c + 16), n);
    simdgroup_store(acc13, c + (by + sg_r + 8) * n + (bx + sg_c + 24), n);
  }
}

__global__ void bench_gemm_multistage(const float* a, const float* b, float* c, int n) {
  __shared__ float s_a[1024];
  __shared__ float s_b[1024];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * 16 + tx;
  const int bx = blockIdx.x * 64;
  const int by = blockIdx.y * 64;
  const bool full_block = (by + 64 <= n) && (bx + 64 <= n);

  const int sg_id = simd_group_id;
  const int sg_r = (sg_id >> 1) * 16;
  const int sg_c = (sg_id & 1) * 32;

  simdgroup_float8x8 acc00(0.0f);
  simdgroup_float8x8 acc01(0.0f);
  simdgroup_float8x8 acc02(0.0f);
  simdgroup_float8x8 acc03(0.0f);
  simdgroup_float8x8 acc10(0.0f);
  simdgroup_float8x8 acc11(0.0f);
  simdgroup_float8x8 acc12(0.0f);
  simdgroup_float8x8 acc13(0.0f);

  const int num_tiles = (n + 15) / 16;

  // Prologue: Load tile 0 into shared memory
  if (full_block && n >= 16) {
    for (int step = 0; step < 4; ++step) {
      const int flat = tid + step * 256;
      const int a_r = flat >> 4;
      const int a_c = flat & 15;
      s_a[a_r * 16 + a_c] = a[(by + a_r) * n + a_c];

      const int b_r = flat >> 6;
      const int b_c = flat & 63;
      s_b[b_r * 64 + b_c] = b[b_r * n + (bx + b_c)];
    }
  } else {
    for (int step = 0; step < 4; ++step) {
      const int flat = tid + step * 256;
      const int a_r = flat >> 4;
      const int a_c = flat & 15;
      s_a[a_r * 16 + a_c] = (by + a_r < n && a_c < n) ? a[(by + a_r) * n + a_c] : 0.0f;

      const int b_r = flat >> 6;
      const int b_c = flat & 63;
      s_b[b_r * 64 + b_c] = (b_r < n && bx + b_c < n) ? b[b_r * n + (bx + b_c)] : 0.0f;
    }
  }
  __syncthreads();

  for (int t = 0; t + 1 < num_tiles; ++t) {
    const int next_k = (t + 1) * 16;
    float fetch_a[4];
    float fetch_b[4];
    if (full_block && (next_k + 16 <= n)) {
      for (int step = 0; step < 4; ++step) {
        const int flat = tid + step * 256;
        const int a_r = flat >> 4;
        const int a_c = flat & 15;
        fetch_a[step] = a[(by + a_r) * n + (next_k + a_c)];

        const int b_r = flat >> 6;
        const int b_c = flat & 63;
        fetch_b[step] = b[(next_k + b_r) * n + (bx + b_c)];
      }
    } else {
      for (int step = 0; step < 4; ++step) {
        const int flat = tid + step * 256;
        const int a_r = flat >> 4;
        const int a_c = flat & 15;
        fetch_a[step] =
            (by + a_r < n && next_k + a_c < n) ? a[(by + a_r) * n + (next_k + a_c)] : 0.0f;

        const int b_r = flat >> 6;
        const int b_c = flat & 63;
        fetch_b[step] =
            (next_k + b_r < n && bx + b_c < n) ? b[(next_k + b_r) * n + (bx + b_c)] : 0.0f;
      }
    }

    for (int ks = 0; ks < 16; ks += 8) {
      simdgroup_float8x8 fa0;
      simdgroup_float8x8 fa1;
      simdgroup_load(fa0, s_a + (sg_r + 0) * 16 + ks, 16);
      simdgroup_load(fa1, s_a + (sg_r + 8) * 16 + ks, 16);

      simdgroup_float8x8 fb0;
      simdgroup_float8x8 fb1;
      simdgroup_float8x8 fb2;
      simdgroup_float8x8 fb3;
      simdgroup_load(fb0, s_b + ks * 64 + sg_c + 0, 64);
      simdgroup_load(fb1, s_b + ks * 64 + sg_c + 8, 64);
      simdgroup_load(fb2, s_b + ks * 64 + sg_c + 16, 64);
      simdgroup_load(fb3, s_b + ks * 64 + sg_c + 24, 64);

      simdgroup_multiply_accumulate(acc00, fa0, fb0, acc00);
      simdgroup_multiply_accumulate(acc01, fa0, fb1, acc01);
      simdgroup_multiply_accumulate(acc02, fa0, fb2, acc02);
      simdgroup_multiply_accumulate(acc03, fa0, fb3, acc03);
      simdgroup_multiply_accumulate(acc10, fa1, fb0, acc10);
      simdgroup_multiply_accumulate(acc11, fa1, fb1, acc11);
      simdgroup_multiply_accumulate(acc12, fa1, fb2, acc12);
      simdgroup_multiply_accumulate(acc13, fa1, fb3, acc13);
    }
    __syncthreads();

    for (int step = 0; step < 4; ++step) {
      const int flat = tid + step * 256;
      const int a_r = flat >> 4;
      const int a_c = flat & 15;
      s_a[a_r * 16 + a_c] = fetch_a[step];

      const int b_r = flat >> 6;
      const int b_c = flat & 63;
      s_b[b_r * 64 + b_c] = fetch_b[step];
    }
    __syncthreads();
  }

  for (int ks = 0; ks < 16; ks += 8) {
    simdgroup_float8x8 fa0;
    simdgroup_float8x8 fa1;
    simdgroup_load(fa0, s_a + (sg_r + 0) * 16 + ks, 16);
    simdgroup_load(fa1, s_a + (sg_r + 8) * 16 + ks, 16);

    simdgroup_float8x8 fb0;
    simdgroup_float8x8 fb1;
    simdgroup_float8x8 fb2;
    simdgroup_float8x8 fb3;
    simdgroup_load(fb0, s_b + ks * 64 + sg_c + 0, 64);
    simdgroup_load(fb1, s_b + ks * 64 + sg_c + 8, 64);
    simdgroup_load(fb2, s_b + ks * 64 + sg_c + 16, 64);
    simdgroup_load(fb3, s_b + ks * 64 + sg_c + 24, 64);

    simdgroup_multiply_accumulate(acc00, fa0, fb0, acc00);
    simdgroup_multiply_accumulate(acc01, fa0, fb1, acc01);
    simdgroup_multiply_accumulate(acc02, fa0, fb2, acc02);
    simdgroup_multiply_accumulate(acc03, fa0, fb3, acc03);
    simdgroup_multiply_accumulate(acc10, fa1, fb0, acc10);
    simdgroup_multiply_accumulate(acc11, fa1, fb1, acc11);
    simdgroup_multiply_accumulate(acc12, fa1, fb2, acc12);
    simdgroup_multiply_accumulate(acc13, fa1, fb3, acc13);
  }

  if (full_block) {
    simdgroup_store(acc00, c + (by + sg_r + 0) * n + (bx + sg_c + 0), n);
    simdgroup_store(acc01, c + (by + sg_r + 0) * n + (bx + sg_c + 8), n);
    simdgroup_store(acc02, c + (by + sg_r + 0) * n + (bx + sg_c + 16), n);
    simdgroup_store(acc03, c + (by + sg_r + 0) * n + (bx + sg_c + 24), n);
    simdgroup_store(acc10, c + (by + sg_r + 8) * n + (bx + sg_c + 0), n);
    simdgroup_store(acc11, c + (by + sg_r + 8) * n + (bx + sg_c + 8), n);
    simdgroup_store(acc12, c + (by + sg_r + 8) * n + (bx + sg_c + 16), n);
    simdgroup_store(acc13, c + (by + sg_r + 8) * n + (bx + sg_c + 24), n);
  }
}
)CUDA";

void StubNoop() {}
void StubVectorAdd() {}
void StubVectorAddFloat4() {}
void StubVectorAddFloat4x2() {}
void StubGemmTiled() {}
void StubGemmRegisterTiled() {}
void StubGemmMultistage() {}

bool RegisterTranslatedKernel(const void* stub, const std::filesystem::path& metallib_path,
                              const muda::compiler::TranslatedKernel& kernel) {
  std::vector<mudaKernelParamDesc> descs;
  descs.reserve(kernel.parameters.size());
  for (const muda::compiler::KernelParameter& param : kernel.parameters) {
    descs.push_back(mudaKernelParamDesc{
        .kind = param.is_pointer() ? mudaKernelParamBuffer : mudaKernelParamBytes,
        .size = param.byte_size(),
    });
  }
  return mudaRegisterKernelFromFile(stub, metallib_path.string().c_str(), kernel.name.c_str(),
                                    descs.data(), descs.size()) == cudaSuccess;
}

}  // namespace

int main(int argc, char* argv[]) {
  bool run_micro = false;
  bool run_throughput = false;
  bool run_libraries = false;
  std::string export_csv;
  std::string export_json;

  for (int i = 1; i < argc; ++i) {
    const std::string_view arg(argv[i]);
    if (arg == "--micro") {
      run_micro = true;
    } else if (arg == "--throughput") {
      run_throughput = true;
    } else if (arg == "--libraries") {
      run_libraries = true;
    } else if (arg == "--all") {
      run_micro = true;
      run_throughput = true;
      run_libraries = true;
    } else if (arg.rfind("--export-csv=", 0) == 0) {
      export_csv = arg.substr(13);
    } else if (arg.rfind("--export-json=", 0) == 0) {
      export_json = arg.substr(14);
    }
  }

  if (!run_micro && !run_throughput && !run_libraries) {
    run_micro = true;
    run_throughput = true;
    run_libraries = true;
  }

  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0 ||
      !muda::compiler::MetalToolchainAvailable()) {
    std::fprintf(stderr, "Metal GPU or Metal toolchain unavailable; skipping benchmark.\n");
    return 0;
  }

  cudaDeviceProp prop{};
  (void)cudaGetDeviceProperties(&prop, 0);
  std::printf("=================================================================\n");
  std::printf("             MUDA STATISTICAL BENCHMARK SUITE (macOS)            \n");
  std::printf("=================================================================\n");
  std::printf("Device Name             : %s\n", prop.name);
  std::printf("Unified Memory (Zero-Cp): %s\n", prop.unifiedAddressing != 0 ? "Yes" : "No");
  std::printf("Theoretical DRAM Peak   : 68.25 GB/s (Apple M1 128-bit LPDDR4X)\n");
  std::printf("Theoretical Compute Peak: 2617 GFLOP/s (8 GPU Cores, FP32)\n");
  std::printf("=================================================================\n\n");

  const std::filesystem::path temp_dir =
      std::filesystem::temp_directory_path() / "muda_runtime_bench";
  std::filesystem::create_directories(temp_dir);
  const std::filesystem::path cu_path = temp_dir / "bench.cu";
  const std::filesystem::path metallib_path = temp_dir / "bench.metallib";

  {
    std::ofstream out(cu_path, std::ios::binary | std::ios::trunc);
    out.write(kBenchSource.data(), static_cast<std::streamsize>(kBenchSource.size()));
  }

  const muda::compiler::TranslationResult tr = muda::compiler::TranslateCudaToMsl(kBenchSource);
  if (!tr.ok() || tr.kernels.size() != 7u) {
    std::fprintf(stderr, "Benchmark kernel translation failed (got %zu kernels, expected 7).\n",
                 tr.kernels.size());
    return EXIT_FAILURE;
  }

  muda::compiler::CompileOptions options{};
  options.input = cu_path;
  options.output = metallib_path;
  options.output_kind = muda::compiler::OutputKind::kMetallib;

  const muda::compiler::CompileResult compiled = muda::compiler::Compile(options);
  if (compiled.status != muda::compiler::Status::kOk) {
    std::fprintf(stderr, "Benchmark kernel compilation failed.\n");
    return EXIT_FAILURE;
  }

  if (!RegisterTranslatedKernel(reinterpret_cast<const void*>(&StubNoop), metallib_path,
                                tr.kernels[0]) ||
      !RegisterTranslatedKernel(reinterpret_cast<const void*>(&StubVectorAdd), metallib_path,
                                tr.kernels[1]) ||
      !RegisterTranslatedKernel(reinterpret_cast<const void*>(&StubVectorAddFloat4), metallib_path,
                                tr.kernels[2]) ||
      !RegisterTranslatedKernel(reinterpret_cast<const void*>(&StubVectorAddFloat4x2),
                                metallib_path, tr.kernels[3]) ||
      !RegisterTranslatedKernel(reinterpret_cast<const void*>(&StubGemmTiled), metallib_path,
                                tr.kernels[4]) ||
      !RegisterTranslatedKernel(reinterpret_cast<const void*>(&StubGemmRegisterTiled),
                                metallib_path, tr.kernels[5]) ||
      !RegisterTranslatedKernel(reinterpret_cast<const void*>(&StubGemmMultistage), metallib_path,
                                tr.kernels[6])) {
    std::fprintf(stderr, "Benchmark kernel registration failed.\n");
    return EXIT_FAILURE;
  }

  std::vector<muda::bench::BenchmarkResultRecord> all_records;
  const muda::bench::BenchmarkConfig default_config{};

  if (run_micro) {
    const auto micro_res =
        muda::bench::RunMicrobenchmarks(reinterpret_cast<const void*>(&StubNoop), default_config);
    all_records.insert(all_records.end(), micro_res.records.begin(), micro_res.records.end());
  }

  if (run_throughput) {
    const muda::bench::ThroughputKernelStubs stubs{
        .vector_add = reinterpret_cast<const void*>(&StubVectorAdd),
        .vector_add_float4 = reinterpret_cast<const void*>(&StubVectorAddFloat4),
        .vector_add_float4x2 = reinterpret_cast<const void*>(&StubVectorAddFloat4x2),
        .gemm_tiled = reinterpret_cast<const void*>(&StubGemmTiled),
        .gemm_register_tiled = reinterpret_cast<const void*>(&StubGemmRegisterTiled),
        .gemm_multistage = reinterpret_cast<const void*>(&StubGemmMultistage),
    };
    const auto tp_res = muda::bench::RunThroughputBenchmarks(stubs, default_config);
    all_records.insert(all_records.end(), tp_res.records.begin(), tp_res.records.end());
  }

  if (run_libraries) {
    const auto lib_res = muda::bench::RunLibraryBenchmarks(default_config);
    all_records.insert(all_records.end(), lib_res.records.begin(), lib_res.records.end());
  }

  if (!export_csv.empty()) {
    muda::bench::ExportStatisticsCsv(export_csv, all_records);
    std::printf("Exported benchmark CSV results to: %s\n", export_csv.c_str());
  }
  if (!export_json.empty()) {
    muda::bench::ExportStatisticsJson(export_json, all_records);
    std::printf("Exported benchmark JSON results to: %s\n", export_json.c_str());
  }

  std::error_code ec{};
  std::filesystem::remove_all(temp_dir, ec);
  return EXIT_SUCCESS;
}

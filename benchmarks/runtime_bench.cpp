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
  const int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
  if (idx < n4) {
    c[idx] = a[idx] + b[idx];
  }
  if (idx + 1 < n4) {
    c[idx + 1] = a[idx + 1] + b[idx + 1];
  }
}

__global__ void bench_gemm_tiled(const float* a, const float* b, float* c, int n) {
  __shared__ float tile_a[256];
  __shared__ float tile_b[256];
  const int row = blockIdx.y * 16 + threadIdx.y;
  const int col = blockIdx.x * 16 + threadIdx.x;
  float acc = 0.0f;
  const int num_tiles = (n + 15) / 16;
  for (int t = 0; t < num_tiles; ++t) {
    const int a_col = t * 16 + threadIdx.x;
    const int b_row = t * 16 + threadIdx.y;
    if (row < n && a_col < n) {
      tile_a[threadIdx.y * 16 + threadIdx.x] = a[row * n + a_col];
    } else {
      tile_a[threadIdx.y * 16 + threadIdx.x] = 0.0f;
    }
    if (b_row < n && col < n) {
      tile_b[threadIdx.y * 16 + threadIdx.x] = b[b_row * n + col];
    } else {
      tile_b[threadIdx.y * 16 + threadIdx.x] = 0.0f;
    }
    __syncthreads();
    for (int k = 0; k < 16; ++k) {
      acc = fmaf(tile_a[threadIdx.y * 16 + k], tile_b[k * 16 + threadIdx.x], acc);
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

  float acc[16];
  for (int i = 0; i < 16; ++i) {
    acc[i] = 0.0f;
  }

  const int num_tiles = (n + 15) / 16;
  for (int t = 0; t < num_tiles; ++t) {
    for (int step = 0; step < 4; ++step) {
      const int a_r = (tid + step * 256) / 16;
      const int a_c = ((tid + step * 256) % 16) + t * 16;
      if (by + a_r < n && a_c < n) {
        s_a[a_r * 16 + ((tid + step * 256) % 16)] = a[(by + a_r) * n + a_c];
      } else {
        s_a[a_r * 16 + ((tid + step * 256) % 16)] = 0.0f;
      }

      const int b_r = ((tid + step * 256) / 64) + t * 16;
      const int b_c = (tid + step * 256) % 64;
      if (b_r < n && bx + b_c < n) {
        s_b[((tid + step * 256) / 64) * 64 + b_c] = b[b_r * n + bx + b_c];
      } else {
        s_b[((tid + step * 256) / 64) * 64 + b_c] = 0.0f;
      }
    }
    __syncthreads();

    for (int k = 0; k < 16; ++k) {
      float reg_b[4];
      for (int j = 0; j < 4; ++j) {
        reg_b[j] = s_b[k * 64 + (tx * 4 + j)];
      }
      float reg_a[4];
      for (int i = 0; i < 4; ++i) {
        reg_a[i] = s_a[(ty * 4 + i) * 16 + k];
      }
      for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
          acc[i * 4 + j] = fmaf(reg_a[i], reg_b[j], acc[i * 4 + j]);
        }
      }
    }
    __syncthreads();
  }

  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      const int out_r = by + ty * 4 + i;
      const int out_c = bx + tx * 4 + j;
      if (out_r < n && out_c < n) {
        c[out_r * n + out_c] = acc[i * 4 + j];
      }
    }
  }
}

__global__ void bench_gemm_multistage(const float* a, const float* b, float* c, int n) {
  __shared__ float s_a[2048];
  __shared__ float s_b[2048];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * 16 + tx;
  const int bx = blockIdx.x * 64;
  const int by = blockIdx.y * 64;

  float acc[16];
  for (int i = 0; i < 16; ++i) {
    acc[i] = 0.0f;
  }

  const int num_tiles = (n + 15) / 16;

  // Prologue: Load tile 0 into buffer 0
  for (int step = 0; step < 4; ++step) {
    const int a_r = (tid + step * 256) / 16;
    const int a_c = (tid + step * 256) % 16;
    if (by + a_r < n && a_c < n) {
      s_a[a_r * 16 + a_c] = a[(by + a_r) * n + a_c];
    } else {
      s_a[a_r * 16 + a_c] = 0.0f;
    }

    const int b_r = (tid + step * 256) / 64;
    const int b_c = (tid + step * 256) % 64;
    if (b_r < n && bx + b_c < n) {
      s_b[b_r * 64 + b_c] = b[b_r * n + bx + b_c];
    } else {
      s_b[b_r * 64 + b_c] = 0.0f;
    }
  }
  __syncthreads();

  for (int t = 0; t < num_tiles; ++t) {
    const int cur_off = (t & 1) * 1024;
    const int next_off = ((t + 1) & 1) * 1024;

    float fetch_a[4];
    float fetch_b[4];
    if (t + 1 < num_tiles) {
      for (int step = 0; step < 4; ++step) {
        const int a_r = (tid + step * 256) / 16;
        const int a_c = ((tid + step * 256) % 16) + (t + 1) * 16;
        if (by + a_r < n && a_c < n) {
          fetch_a[step] = a[(by + a_r) * n + a_c];
        } else {
          fetch_a[step] = 0.0f;
        }

        const int b_r = ((tid + step * 256) / 64) + (t + 1) * 16;
        const int b_c = (tid + step * 256) % 64;
        if (b_r < n && bx + b_c < n) {
          fetch_b[step] = b[b_r * n + bx + b_c];
        } else {
          fetch_b[step] = 0.0f;
        }
      }
    }

    for (int k = 0; k < 16; ++k) {
      float reg_b[4];
      for (int j = 0; j < 4; ++j) {
        reg_b[j] = s_b[cur_off + k * 64 + (tx * 4 + j)];
      }
      float reg_a[4];
      for (int i = 0; i < 4; ++i) {
        reg_a[i] = s_a[cur_off + (ty * 4 + i) * 16 + k];
      }
      for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
          acc[i * 4 + j] = fmaf(reg_a[i], reg_b[j], acc[i * 4 + j]);
        }
      }
    }

    if (t + 1 < num_tiles) {
      for (int step = 0; step < 4; ++step) {
        const int a_r = (tid + step * 256) / 16;
        const int a_c = (tid + step * 256) % 16;
        s_a[next_off + a_r * 16 + a_c] = fetch_a[step];

        const int b_r = (tid + step * 256) / 64;
        const int b_c = (tid + step * 256) % 64;
        s_b[next_off + b_r * 64 + b_c] = fetch_b[step];
      }
      __syncthreads();
    }
  }

  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      const int out_r = by + ty * 4 + i;
      const int out_c = bx + tx * 4 + j;
      if (out_r < n && out_c < n) {
        c[out_r * n + out_c] = acc[i * 4 + j];
      }
    }
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
  std::string export_csv;
  std::string export_json;

  for (int i = 1; i < argc; ++i) {
    const std::string_view arg(argv[i]);
    if (arg == "--micro") {
      run_micro = true;
    } else if (arg == "--throughput") {
      run_throughput = true;
    } else if (arg == "--all") {
      run_micro = true;
      run_throughput = true;
    } else if (arg.rfind("--export-csv=", 0) == 0) {
      export_csv = arg.substr(13);
    } else if (arg.rfind("--export-json=", 0) == 0) {
      export_json = arg.substr(14);
    }
  }

  if (!run_micro && !run_throughput) {
    run_micro = true;
    run_throughput = true;
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

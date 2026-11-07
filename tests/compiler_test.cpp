/**
 * @file compiler_test.cpp
 * @brief Tests CUDA-to-Metal compilation, MSL emission, object compilation, and executable linking.
 */

#include <dlfcn.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "cuda_runtime.h"
#include "muda/compiler/driver.hpp"
#include "muda/compiler/toolchain.hpp"
#include "test_utils.hpp"

namespace {

bool RunExecutableAndCheckOutput(const std::filesystem::path& exe_path,
                                 std::string_view expected_prefix) {
  const std::string cmd = "\"" + exe_path.string() + "\"";
  FILE* pipe = popen(cmd.c_str(), "r");
  if (pipe == nullptr) {
    return false;
  }
  char buffer[256] = {};
  std::string output;
  while (std::fgets(buffer, sizeof(buffer), pipe) != nullptr) {
    output += buffer;
  }
  const int status = pclose(pipe);
  if (status != 0) {
    std::cerr << "executable exited with non-zero status: " << status << "\n";
    return false;
  }
  if (output.find(expected_prefix) == std::string::npos) {
    std::cerr << "unexpected executable output: " << output << "\n";
    return false;
  }
  return true;
}

}  // namespace

int main() {
  const muda::testing::ScopedTempDir temp_dir("muda_compiler_test");
  if (!temp_dir.valid()) {
    std::cerr << "cannot create temp directory\n";
    return 1;
  }

  const std::filesystem::path sample_cu =
      std::filesystem::path(MUDA_SOURCE_DIR) / "samples/vector_add.cu";

  // 1. Test MSL source emission (-S mode) — works even without Metal toolchain
  {
    muda::compiler::CompileOptions msl_options;
    msl_options.input = sample_cu;
    msl_options.output = temp_dir.path() / "vector_add_only.metal";
    msl_options.output_kind = muda::compiler::OutputKind::kMslSource;

    const muda::compiler::CompileResult msl_result = muda::compiler::Compile(msl_options);
    if (msl_result.status != muda::compiler::Status::kOk) {
      std::cerr << "MSL emission failed [" << msl_result.diagnostic_id()
                << "]: " << msl_result.message() << "\n";
      return 1;
    }
    if (!std::filesystem::exists(msl_result.output_path)) {
      std::cerr << "missing emitted .metal file\n";
      return 1;
    }
  }

  if (!muda::compiler::MetalToolchainAvailable()) {
    std::cout << "skip: Metal toolchain unavailable\n";
    return 77;
  }

  // 2. Test .metallib compilation with kept intermediates
  {
    muda::compiler::CompileOptions options;
    options.input = sample_cu;
    options.output = temp_dir.path() / "vector_add.metallib";
    options.intermediates = muda::compiler::IntermediatePolicy::kKeep;

    const muda::compiler::CompileResult result = muda::compiler::Compile(options);
    if (result.status != muda::compiler::Status::kOk) {
      std::cerr << "compile failed [" << result.diagnostic_id() << "]: " << result.message()
                << "\n";
      return 1;
    }
    if (result.kernel_name != "vector_add") {
      std::cerr << "unexpected kernel name: " << result.kernel_name << "\n";
      return 1;
    }
    if (!std::filesystem::exists(result.metallib_path)) {
      std::cerr << "missing metallib\n";
      return 1;
    }
    if (!std::filesystem::exists(temp_dir.path() / "vector_add.metal")) {
      std::cerr << "missing generated Metal source\n";
      return 1;
    }
    if (!std::filesystem::exists(temp_dir.path() / "vector_add.air")) {
      std::cerr << "missing intermediate AIR file\n";
      return 1;
    }
  }

  // 3. Test compiler artifact cache v1 (miss on first compile, hit on second compile)
  {
    const std::filesystem::path cache_dir = temp_dir.path() / "cache_v1";
    muda::compiler::CompileOptions cache_opts;
    cache_opts.input = sample_cu;
    cache_opts.output = temp_dir.path() / "cached_1.metallib";
    cache_opts.output_kind = muda::compiler::OutputKind::kMetallib;
    cache_opts.cache_dir = cache_dir;

    const muda::compiler::CompileResult first = muda::compiler::Compile(cache_opts);
    if (first.status != muda::compiler::Status::kOk || first.cache_hit) {
      std::cerr << "first compile into fresh cache should succeed as a cache miss\n";
      return 1;
    }

    cache_opts.output = temp_dir.path() / "cached_2.metallib";
    const muda::compiler::CompileResult second = muda::compiler::Compile(cache_opts);
    if (second.status != muda::compiler::Status::kOk || !second.cache_hit) {
      std::cerr << "second compile with identical input should be a cache hit\n";
      return 1;
    }
    if (!std::filesystem::exists(second.metallib_path)) {
      std::cerr << "missing cached metallib output\n";
      return 1;
    }
  }

  // 4. Test host object (-c) compilation
  {
    muda::compiler::CompileOptions obj_options;
    obj_options.input = sample_cu;
    obj_options.output = temp_dir.path() / "vector_add.o";
    obj_options.output_kind = muda::compiler::OutputKind::kObject;

    const muda::compiler::CompileResult obj_result = muda::compiler::Compile(obj_options);
    if (obj_result.status != muda::compiler::Status::kOk) {
      std::cerr << "object compile failed [" << obj_result.diagnostic_id()
                << "]: " << obj_result.message() << "\n";
      return 1;
    }
    if (!std::filesystem::exists(obj_result.output_path)) {
      std::cerr << "missing compiled object file\n";
      return 1;
    }
  }

  // 4. Test standalone executable compilation and GPU execution
  {
    muda::compiler::CompileOptions exe_options;
    exe_options.input = sample_cu;
    exe_options.output = temp_dir.path() / "vector_add_bin";
    exe_options.output_kind = muda::compiler::OutputKind::kExecutable;

    const muda::compiler::CompileResult exe_result = muda::compiler::Compile(exe_options);
    if (exe_result.status != muda::compiler::Status::kOk) {
      std::cerr << "executable compile failed [" << exe_result.diagnostic_id()
                << "]: " << exe_result.message() << "\n";
      return 1;
    }
    if (!std::filesystem::exists(exe_result.output_path)) {
      std::cerr << "missing compiled executable\n";
      return 1;
    }
    if (!RunExecutableAndCheckOutput(exe_result.output_path, "3.000000")) {
      return 1;
    }
  }

  // 5. Test multi-kernel .cu file, preprocessor defines (-D), and shared library (-shared)
  {
    const std::filesystem::path multi_cu = temp_dir.path() / "multi_kernel_ext.cu";
    {
      std::ofstream out(multi_cu);
      out << "#include <cuda_runtime.h>\n"
          << "#ifndef SCALE_FACTOR\n"
          << "#define SCALE_FACTOR 1.0f\n"
          << "#endif\n"
          << "__global__ void k_scale(float* data, int n) {\n"
          << "  int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
          << "  if (idx < n) { data[idx] = data[idx] * SCALE_FACTOR; }\n"
          << "}\n"
          << "__global__ void k_bias(float* data, float bias, int n) {\n"
          << "  int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
          << "  if (idx < n) { data[idx] = data[idx] + bias; }\n"
          << "}\n"
          << "extern \"C\" int run_scale_and_bias(float* host_io, int n, float bias) {\n"
          << "  float* dev = nullptr;\n"
          << "  if (cudaMalloc(reinterpret_cast<void**>(&dev), static_cast<size_t>(n) * "
             "sizeof(float)) != cudaSuccess) return 1;\n"
          << "  if (cudaMemcpy(dev, host_io, static_cast<size_t>(n) * sizeof(float), "
             "cudaMemcpyHostToDevice) != cudaSuccess) return 2;\n"
          << "  k_scale<<<(n + 63) / 64, 64>>>(dev, n);\n"
          << "  k_bias<<<(n + 63) / 64, 64>>>(dev, bias, n);\n"
          << "  if (cudaDeviceSynchronize() != cudaSuccess) return 3;\n"
          << "  if (cudaMemcpy(host_io, dev, static_cast<size_t>(n) * sizeof(float), "
             "cudaMemcpyDeviceToHost) != cudaSuccess) return 4;\n"
          << "  cudaFree(dev);\n"
          << "  return 0;\n"
          << "}\n";
    }

    const std::filesystem::path dylib_path = temp_dir.path() / "libmulti_kernel_ext.dylib";
    muda::compiler::CompileOptions shared_opts;
    shared_opts.input = multi_cu;
    shared_opts.output = dylib_path;
    shared_opts.output_kind = muda::compiler::OutputKind::kSharedLibrary;
    shared_opts.position_independent = true;
    shared_opts.defines.emplace_back("-DSCALE_FACTOR=3.0f");

    const muda::compiler::CompileResult shared_res = muda::compiler::Compile(shared_opts);
    if (shared_res.status != muda::compiler::Status::kOk) {
      std::cerr << "shared library compile failed [" << shared_res.diagnostic_id()
                << "]: " << shared_res.message() << "\n";
      return 1;
    }

    void* handle = dlopen(dylib_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
      std::cerr << "dlopen failed: " << dlerror() << "\n";
      return 1;
    }
    using RunFn = int (*)(float*, int, float);
    auto run_fn = reinterpret_cast<RunFn>(dlsym(handle, "run_scale_and_bias"));
    if (run_fn == nullptr) {
      std::cerr << "dlsym failed for run_scale_and_bias\n";
      dlclose(handle);
      return 1;
    }

    std::vector<float> values = {1.0f, 2.0f, 3.0f, 4.0f};
    const int rc = run_fn(values.data(), static_cast<int>(values.size()), 0.5f);
    dlclose(handle);
    if (rc != 0) {
      std::cerr << "run_scale_and_bias returned error: " << rc << "\n";
      return 1;
    }
    for (std::size_t i = 0; i < values.size(); ++i) {
      const float expected = static_cast<float>(i + 1) * 3.0f + 0.5f;
      if (!muda::testing::AreClose(values[i], expected)) {
        std::cerr << "mismatch at index " << i << ": got " << values[i] << ", expected " << expected
                  << "\n";
        return 1;
      }
    }
  }

  std::cout << "compiled and executed vector_add and multi-kernel shared library end-to-end\n";
  return 0;
}

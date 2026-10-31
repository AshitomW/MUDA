/**
 * @file compiler_test.cpp
 * @brief Tests CUDA-to-Metal compilation, MSL emission, object compilation, and executable linking.
 */

#include <cstdio>
#include <filesystem>
#include <iostream>
#include <string>

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

  std::cout << "compiled and executed vector_add end-to-end\n";
  return 0;
}

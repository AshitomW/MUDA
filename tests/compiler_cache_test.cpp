/**
 * @file compiler_cache_test.cpp
 * @brief Unit tests for Compiler Cache v2 (dependency hashing, toolchain keying, fast-math).
 */

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "muda/compiler/cache.hpp"
#include "muda/compiler/driver.hpp"
#include "muda/compiler/toolchain.hpp"
#include "muda/version.hpp"

namespace {

bool Expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    return false;
  }
  return true;
}

bool TestCacheKeySensitivity() {
  muda::compiler::CacheKeyInput base{
      .msl_source = "kernel void foo() {}",
      .muda_version = muda::kVersion,
      .toolchain_version = "metal-32023.921",
      .metal_standard = muda::compiler::MetalStandard::kMetal32,
      .optimization = muda::compiler::OptimizationLevel::kO3,
      .dependencies_hash = "abcdef123456",
  };
  const std::string key_base = muda::compiler::ArtifactCache::ComputeKey(base);

  // Fast-math vs O3
  auto with_fast_math = base;
  with_fast_math.optimization = muda::compiler::OptimizationLevel::kFast;
  const std::string key_fast_math = muda::compiler::ArtifactCache::ComputeKey(with_fast_math);
  if (!Expect(key_base != key_fast_math, "Fast-math must alter cache key")) {
    return false;
  }

  // Toolchain SDK version change
  auto with_diff_toolchain = base;
  with_diff_toolchain.toolchain_version = "metal-99999.0";
  const std::string key_toolchain = muda::compiler::ArtifactCache::ComputeKey(with_diff_toolchain);
  if (!Expect(key_base != key_toolchain, "Toolchain version must alter cache key")) {
    return false;
  }

  // Header dependency change
  auto with_diff_deps = base;
  with_diff_deps.dependencies_hash = "111122223333";
  const std::string key_deps = muda::compiler::ArtifactCache::ComputeKey(with_diff_deps);
  if (!Expect(key_base != key_deps, "Dependency hash must alter cache key")) {
    return false;
  }

  return true;
}

bool TestIncludeDependencyInvalidation() {
  const std::filesystem::path temp_dir =
      std::filesystem::temp_directory_path() / "muda_dep_test_cache";
  const std::filesystem::path cache_dir = temp_dir / "cache";
  std::error_code ec;
  std::filesystem::remove_all(temp_dir, ec);
  std::filesystem::create_directories(temp_dir, ec);
  std::filesystem::create_directories(cache_dir, ec);

  const std::filesystem::path header_path = temp_dir / "kernel_defs.h";
  const std::filesystem::path source_path = temp_dir / "test_kernel.cu";
  const std::filesystem::path output_path = temp_dir / "test_kernel.metallib";

  // Initial header content
  {
    std::ofstream header(header_path);
    header << "#define BLOCK_DIM 64\n";
  }

  // Source including header
  {
    std::ofstream source(source_path);
    source << "#include \"kernel_defs.h\"\n"
           << "__global__ void test_dep_k(float* out) {\n"
           << "  int idx = threadIdx.x;\n"
           << "  out[idx] = (float)BLOCK_DIM;\n"
           << "}\n";
  }

  muda::compiler::CompileOptions options;
  options.input = source_path;
  options.output = output_path;
  options.output_kind = muda::compiler::OutputKind::kMetallib;
  options.cache_dir = cache_dir;
  options.include_dirs.push_back(temp_dir);

  // First compilation: must be cache miss
  auto result1 = muda::compiler::Compile(options);
  if (!Expect(result1.status == muda::compiler::Status::kOk, "First compilation failed")) {
    return false;
  }
  if (!Expect(!result1.cache_hit, "First compilation should be cache miss")) {
    return false;
  }

  // Second compilation with untouched header: must be cache hit
  auto result2 = muda::compiler::Compile(options);
  if (!Expect(result2.status == muda::compiler::Status::kOk, "Second compilation failed")) {
    return false;
  }
  if (!Expect(result2.cache_hit, "Second compilation should be cache hit")) {
    return false;
  }

  // Modify included header!
  {
    std::ofstream header(header_path, std::ios::trunc);
    header << "#define BLOCK_DIM 128\n";
  }

  // Third compilation: header changed -> must invalidate cache and produce cache miss!
  auto result3 = muda::compiler::Compile(options);
  if (!Expect(result3.status == muda::compiler::Status::kOk, "Third compilation failed")) {
    return false;
  }
  if (!Expect(!result3.cache_hit,
              "Compilation after header modification must be cache miss")) {
    return false;
  }

  // Fourth compilation: must be cache hit
  auto result4 = muda::compiler::Compile(options);
  if (!Expect(result4.status == muda::compiler::Status::kOk, "Fourth compilation failed")) {
    return false;
  }
  if (!Expect(result4.cache_hit, "Fourth compilation should be cache hit")) {
    return false;
  }

  std::filesystem::remove_all(temp_dir, ec);
  return true;
}

bool TestFastMathModeCompilation() {
  const std::filesystem::path temp_dir =
      std::filesystem::temp_directory_path() / "muda_fast_math_test";
  const std::filesystem::path source_path = temp_dir / "fast_math.cu";
  const std::filesystem::path output_path = temp_dir / "fast_math.metallib";
  std::error_code ec;
  std::filesystem::remove_all(temp_dir, ec);
  std::filesystem::create_directories(temp_dir, ec);

  {
    std::ofstream source(source_path);
    source << "__global__ void fast_math_k(float* out, const float* in, float a, float b) {\n"
           << "  int idx = threadIdx.x;\n"
           << "  out[idx] = in[idx] * a + b;\n"
           << "}\n";
  }

  muda::compiler::CompileOptions options;
  options.input = source_path;
  options.output = output_path;
  options.output_kind = muda::compiler::OutputKind::kMetallib;
  options.optimization = muda::compiler::OptimizationLevel::kFast;

  auto result = muda::compiler::Compile(options);
  if (!Expect(result.status == muda::compiler::Status::kOk,
              "Fast-math mode compilation must succeed")) {
    return false;
  }
  if (!Expect(std::filesystem::exists(output_path),
              "Fast-math metallib output must exist on disk")) {
    return false;
  }

  std::filesystem::remove_all(temp_dir, ec);
  return true;
}

}  // namespace

int main() {
  std::cout << "Running compiler cache v2 and fast-math tests...\n";
  if (!TestCacheKeySensitivity()) {
    return 1;
  }
  if (!muda::compiler::MetalToolchainAvailable()) {
    std::cout << "skip: Metal toolchain unavailable for AOT metallib compilation\n";
    return 77;
  }
  if (!TestIncludeDependencyInvalidation()) {
    return 1;
  }
  if (!TestFastMathModeCompilation()) {
    return 1;
  }
  std::cout << "All compiler cache tests passed successfully!\n";
  return 0;
}


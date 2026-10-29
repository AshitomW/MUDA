/**
 * @file compiler_test.cpp
 * @brief Tests CUDA-to-Metal compilation.
 */

#include <unistd.h>

#include <filesystem>
#include <iostream>
#include <string>

#include "muda/compiler/driver.hpp"
#include "muda/compiler/toolchain.hpp"

int main() {
  if (!muda::compiler::MetalToolchainAvailable()) {
    std::cout << "skip: Metal toolchain unavailable\n";
    return 77;
  }

  const std::filesystem::path temp =
      std::filesystem::temp_directory_path() / ("muda_compiler_test_" + std::to_string(getpid()));
  std::error_code error;
  std::filesystem::remove_all(temp, error);
  std::filesystem::create_directories(temp, error);
  if (error) {
    std::cerr << "cannot create temp directory: " << error.message() << "\n";
    return 1;
  }

  muda::compiler::CompileOptions options;
  options.input = std::filesystem::path(MUDA_SOURCE_DIR) / "samples/vector_add.cu";
  options.output = temp / "vector_add.metallib";
  options.intermediates = muda::compiler::IntermediatePolicy::kKeep;

  const muda::compiler::CompileResult result = muda::compiler::Compile(options);
  if (result.status != muda::compiler::Status::kOk) {
    std::cerr << "compile failed [" << result.diagnostic_id() << "]: " << result.message() << "\n";
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
  if (!std::filesystem::exists(temp / "vector_add.metal")) {
    std::cerr << "missing generated Metal source\n";
    return 1;
  }
  if (!std::filesystem::exists(temp / "vector_add.air")) {
    std::cerr << "missing intermediate AIR file\n";
    return 1;
  }

  std::filesystem::remove_all(temp, error);
  std::cout << "compiled vector_add.metallib\n";
  return 0;
}

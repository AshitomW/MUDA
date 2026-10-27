/**
 * @file driver.hpp
 * @brief Public interface for compiling CUDA source into a Metal library.
 */

#pragma once

#include <filesystem>
#include <string>

namespace muda::compiler {

enum class Status {
  kOk,
  kError,
};

struct CompileOptions {
  std::filesystem::path input;
  std::filesystem::path output;
  bool keep_intermediates = false;
};

struct CompileResult {
  Status status = Status::kError;
  std::string message;
  std::filesystem::path metallib_path;
  std::string kernel_name;
};

[[nodiscard]] CompileResult Compile(const CompileOptions& options);

}

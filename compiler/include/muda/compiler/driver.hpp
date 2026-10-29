/**
 * @file driver.hpp
 * @brief Public interface for compiling CUDA source into a Metal library.
 */

#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>

#include "muda/compiler/types.hpp"

namespace muda::compiler {

enum class Status : std::uint8_t {
  kOk,
  kError,
};

struct CompileOptions {
  std::filesystem::path input;
  std::filesystem::path output;
  IntermediatePolicy intermediates = IntermediatePolicy::kRemove;
};

struct CompileResult {
  Status status = Status::kError;
  DiagnosticCode code = DiagnosticCode::kNoInputFile;
  std::filesystem::path metallib_path;
  std::string kernel_name;

  [[nodiscard]] constexpr std::string_view diagnostic_id() const noexcept {
    return DiagnosticId(code);
  }

  [[nodiscard]] constexpr std::string_view message() const noexcept {
    return DiagnosticMessage(code);
  }
};

[[nodiscard]] CompileResult Compile(const CompileOptions& options);

}  // namespace muda::compiler

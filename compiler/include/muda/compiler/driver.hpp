/**
 * @file driver.hpp
 * @brief Public interface for compiling CUDA source into a Metal library.
 */

#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include "muda/compiler/types.hpp"

namespace muda::compiler {

enum class Status : std::uint8_t {
  kOk,
  kError,
};

struct CompileOptions {
  std::filesystem::path input;
  std::filesystem::path output;
  OutputKind output_kind = OutputKind::kAuto;
  IntermediatePolicy intermediates = IntermediatePolicy::kRemove;
  MetalStandard metal_standard = MetalStandard::kMetal32;
  OptimizationLevel optimization = OptimizationLevel::kO3;
  std::filesystem::path cache_dir;
  bool disable_cache = false;
  std::vector<std::filesystem::path> include_dirs;
  std::vector<std::filesystem::path> library_dirs;
  std::vector<std::string> libraries;
};

struct CompileResult {
  Status status = Status::kError;
  DiagnosticCode code = DiagnosticCode::kNoInputFile;
  std::filesystem::path output_path;
  std::filesystem::path metallib_path;
  std::string kernel_name;
  bool cache_hit = false;

  [[nodiscard]] constexpr std::string_view diagnostic_id() const noexcept {
    return DiagnosticId(code);
  }

  [[nodiscard]] constexpr std::string_view message() const noexcept {
    return DiagnosticMessage(code);
  }
};

[[nodiscard]] CompileResult Compile(const CompileOptions& options);

}  // namespace muda::compiler

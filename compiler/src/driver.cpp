/**
 * @file driver.cpp
 * @brief Orchestrates translation, Metal compilation, and linking.
 */

#include "muda/compiler/driver.hpp"

#include <unistd.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

#include "muda/compiler/toolchain.hpp"
#include "muda/compiler/translator.hpp"
#include "process.hpp"

namespace muda::compiler {
namespace {

std::atomic<std::uint64_t> g_temp_counter{0};

CompileResult Failure(DiagnosticCode code) noexcept {
  CompileResult result;
  result.status = Status::kError;
  result.code = code;
  return result;
}

void RemoveFile(const std::filesystem::path& path) noexcept {
  std::error_code error;
  std::filesystem::remove(path, error);
}

class ScopedIntermediateCleaner {
 public:
  ScopedIntermediateCleaner(std::filesystem::path metal_path, std::filesystem::path air_path,
                            IntermediatePolicy policy)
      : metal_path_(std::move(metal_path)), air_path_(std::move(air_path)), policy_(policy) {}

  ~ScopedIntermediateCleaner() {
    if (policy_ == IntermediatePolicy::kRemove) {
      RemoveFile(metal_path_);
      RemoveFile(air_path_);
    }
  }

  ScopedIntermediateCleaner(const ScopedIntermediateCleaner&) = delete;
  ScopedIntermediateCleaner& operator=(const ScopedIntermediateCleaner&) = delete;

 private:
  std::filesystem::path metal_path_;
  std::filesystem::path air_path_;
  IntermediatePolicy policy_;
};

std::filesystem::path ResolveBaseDirectory(const std::filesystem::path& output,
                                           IntermediatePolicy policy) {
  if (policy == IntermediatePolicy::kKeep) {
    return output.has_parent_path() ? output.parent_path() : std::filesystem::current_path();
  }
  return std::filesystem::temp_directory_path();
}

std::string MakeIntermediateStem(const std::filesystem::path& output, IntermediatePolicy policy) {
  if (policy == IntermediatePolicy::kKeep) {
    return output.stem().string();
  }
  const std::uint64_t id = g_temp_counter.fetch_add(1, std::memory_order_relaxed);
  return output.stem().string() + "_" + std::to_string(getpid()) + "_" + std::to_string(id);
}

}  // namespace

CompileResult Compile(const CompileOptions& options) {
  if (options.input.empty()) {
    return Failure(DiagnosticCode::kNoInputFile);
  }

  if (!MetalToolchainAvailable()) {
    return Failure(DiagnosticCode::kToolchainNotFound);
  }

  std::ifstream input(options.input, std::ios::binary);
  if (!input) {
    return Failure(DiagnosticCode::kCannotOpenInput);
  }

  std::ostringstream source_buffer;
  source_buffer << input.rdbuf();
  const std::string source = source_buffer.str();

  TranslationResult translation = TranslateCudaToMsl(source);
  if (!translation.ok()) {
    return Failure(translation.code);
  }

  std::filesystem::path output = options.output;
  if (output.empty()) {
    output = options.input;
    output.replace_extension(".metallib");
  }

  std::error_code fs_error;
  if (output.has_parent_path() && !output.parent_path().empty()) {
    std::filesystem::create_directories(output.parent_path(), fs_error);
    if (fs_error) {
      return Failure(DiagnosticCode::kCannotCreateOutputDir);
    }
  }

  const std::filesystem::path base_dir = ResolveBaseDirectory(output, options.intermediates);
  const std::string stem = MakeIntermediateStem(output, options.intermediates);
  const std::filesystem::path metal_path = base_dir / (stem + ".metal");
  const std::filesystem::path air_path = base_dir / (stem + ".air");

  const ScopedIntermediateCleaner cleaner(metal_path, air_path, options.intermediates);

  {
    std::ofstream metal_file(metal_path, std::ios::binary);
    if (!metal_file) {
      return Failure(DiagnosticCode::kCannotWriteMetalSource);
    }
    metal_file << translation.kernel.msl;
    if (!metal_file) {
      return Failure(DiagnosticCode::kCannotWriteMetalSource);
    }
  }

  const std::string metal_path_str = metal_path.string();
  const std::string air_path_str = air_path.string();
  const std::string output_str = output.string();

  const std::array<std::string_view, 9> compile_args = {
      "xcrun", "--sdk",        "macosx", "metal",     "-std=metal3.2",
      "-c",    metal_path_str, "-o",     air_path_str};
  if (!detail::RunProcess(compile_args).succeeded()) {
    return Failure(DiagnosticCode::kMetalCompilationFailed);
  }

  const std::array<std::string_view, 7> link_args = {"xcrun",      "--sdk", "macosx",  "metallib",
                                                     air_path_str, "-o",    output_str};
  if (!detail::RunProcess(link_args).succeeded()) {
    RemoveFile(output);
    return Failure(DiagnosticCode::kMetallibLinkFailed);
  }

  CompileResult result;
  result.status = Status::kOk;
  result.code = DiagnosticCode::kOk;
  result.metallib_path = std::move(output);
  result.kernel_name = std::move(translation.kernel.name);
  return result;
}

}  // namespace muda::compiler

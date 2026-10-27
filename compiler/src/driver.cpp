/**
 * @file driver.cpp
 * @brief Orchestrates translation, Metal compilation, and linking.
 */

#include "muda/compiler/driver.hpp"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>

#include "muda/compiler/toolchain.hpp"
#include "muda/compiler/translator.hpp"
#include "process.hpp"

namespace muda::compiler {
namespace {

CompileResult Failure(std::string message) {
  CompileResult result;
  result.status = Status::kError;
  result.message = std::move(message);
  return result;
}

void RemoveFile(const std::filesystem::path& path) {
  std::error_code error;
  std::filesystem::remove(path, error);
}

}

CompileResult Compile(const CompileOptions& options) {
  if (options.input.empty()) {
    return Failure("no input file");
  }

  if (!MetalToolchainAvailable()) {
    return Failure(
        "Metal toolchain not found; install Xcode and run "
        "xcodebuild -downloadComponent metalToolchain");
  }

  std::ifstream input(options.input, std::ios::binary);
  if (!input) {
    return Failure("cannot open input: " + options.input.string());
  }

  std::ostringstream source_buffer;
  source_buffer << input.rdbuf();
  const std::string source = source_buffer.str();

  const TranslationResult translation = TranslateCudaToMsl(source);
  if (!translation.ok) {
    return Failure(translation.error);
  }

  std::filesystem::path output = options.output;
  if (output.empty()) {
    output = options.input;
    output.replace_extension(".metallib");
  }

  std::error_code error;
  if (output.has_parent_path()) {
    std::filesystem::create_directories(output.parent_path(), error);
    if (error) {
      return Failure("cannot create output directory: " + error.message());
    }
  }

  const std::filesystem::path metal_path =
      output.parent_path() / (output.stem().string() + ".metal");
  const std::filesystem::path air_path = output.parent_path() / (output.stem().string() + ".air");

  std::ofstream metal_file(metal_path, std::ios::binary);
  if (!metal_file) {
    return Failure("cannot write Metal source: " + metal_path.string());
  }
  metal_file << translation.kernel.msl;
  if (!metal_file) {
    return Failure("failed to write Metal source: " + metal_path.string());
  }
  metal_file.close();

  const int metal_status = detail::RunProcess({"xcrun", "--sdk", "macosx", "metal", "-std=metal3.2",
                                               "-c", metal_path.string(), "-o", air_path.string()});
  if (metal_status != 0) {
    if (!options.keep_intermediates) {
      RemoveFile(metal_path);
      RemoveFile(air_path);
    }
    return Failure("Metal compilation failed");
  }

  const int link_status = detail::RunProcess(
      {"xcrun", "--sdk", "macosx", "metallib", air_path.string(), "-o", output.string()});
  if (link_status != 0) {
    if (!options.keep_intermediates) {
      RemoveFile(metal_path);
      RemoveFile(air_path);
      RemoveFile(output);
    }
    return Failure("Metal library link failed");
  }

  if (!options.keep_intermediates) {
    RemoveFile(metal_path);
    RemoveFile(air_path);
  }

  CompileResult result;
  result.status = Status::kOk;
  result.metallib_path = output;
  result.kernel_name = translation.kernel.name;
  return result;
}

}

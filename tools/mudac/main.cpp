/**
 * @file main.cpp
 * @brief Implements the MUDA CUDA-to-Metal compiler CLI.
 */

#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "muda/compiler/driver.hpp"
#include "muda/version.hpp"

namespace {

void PrintUsage(std::ostream& out) {
  out << "mudac " << muda::kVersion << "\n"
      << "Usage: mudac [options] <file.cu>\n"
      << "\n"
      << "Options:\n"
      << "  -o <path>                 Write output to <path>\n"
      << "  -c                        Compile to host object file (.o) with embedded metallib\n"
      << "  -S                        Emit translated Metal Shading Language (.metal) source\n"
      << "  --emit-metallib           Emit standalone Metal library (.metallib)\n"
      << "  -I <dir>, -I<dir>         Add header include directory\n"
      << "  -L <dir>, -L<dir>         Add library search directory\n"
      << "  -l <lib>, -l<lib>         Link against library <lib>\n"
      << "  -k, --keep-intermediates  Keep generated .metal, .air, and .host.cpp files\n"
      << "  --muda-keep-msl           Keep generated .metal and intermediate files\n"
      << "  --muda-save-temps         Keep all intermediate compilation artifacts\n"
      << "  --muda-cache=<dir>        Use <dir> for the compiler artifact cache\n"
      << "  --muda-no-cache           Disable the compiler artifact cache\n"
      << "  -h, --help                Show this help\n"
      << "  --version                 Show version\n";
}

bool StartsWith(std::string_view text, std::string_view prefix) noexcept {
  return text.size() >= prefix.size() && text.substr(0, prefix.size()) == prefix;
}

}  // namespace

int main(int argc, char** argv) {
  muda::compiler::CompileOptions options;

  for (int index = 1; index < argc; ++index) {
    const std::string_view argument = argv[index];
    if (argument == "--version") {
      std::cout << "mudac " << muda::kVersion << "\n";
      return 0;
    }
    if (argument == "-h" || argument == "--help") {
      PrintUsage(std::cout);
      return 0;
    }
    if (argument == "-k" || argument == "--keep-intermediates" || argument == "--muda-keep-msl" ||
        argument == "--muda-save-temps") {
      options.intermediates = muda::compiler::IntermediatePolicy::kKeep;
      continue;
    }
    if (argument == "-c") {
      options.output_kind = muda::compiler::OutputKind::kObject;
      continue;
    }
    if (argument == "-S") {
      options.output_kind = muda::compiler::OutputKind::kMslSource;
      continue;
    }
    if (argument == "--emit-metallib") {
      options.output_kind = muda::compiler::OutputKind::kMetallib;
      continue;
    }
    if (argument == "--muda-no-cache") {
      options.disable_cache = true;
      continue;
    }
    if (StartsWith(argument, "--muda-cache=")) {
      options.cache_dir = argument.substr(std::string_view("--muda-cache=").size());
      continue;
    }
    if (argument == "-o") {
      if (index + 1 >= argc) {
        std::cerr << "mudac: missing output path\n";
        return 2;
      }
      options.output = argv[++index];
      continue;
    }
    if (argument == "-I") {
      if (index + 1 >= argc) {
        std::cerr << "mudac: missing include directory\n";
        return 2;
      }
      options.include_dirs.emplace_back(argv[++index]);
      continue;
    }
    if (StartsWith(argument, "-I") && argument.size() > 2) {
      options.include_dirs.emplace_back(argument.substr(2));
      continue;
    }
    if (argument == "-L") {
      if (index + 1 >= argc) {
        std::cerr << "mudac: missing library directory\n";
        return 2;
      }
      options.library_dirs.emplace_back(argv[++index]);
      continue;
    }
    if (StartsWith(argument, "-L") && argument.size() > 2) {
      options.library_dirs.emplace_back(argument.substr(2));
      continue;
    }
    if (argument == "-l") {
      if (index + 1 >= argc) {
        std::cerr << "mudac: missing library name\n";
        return 2;
      }
      options.libraries.emplace_back(argv[++index]);
      continue;
    }
    if (StartsWith(argument, "-l") && argument.size() > 2) {
      options.libraries.emplace_back(argument.substr(2));
      continue;
    }
    if (argument == "-O0") {
      options.optimization = muda::compiler::OptimizationLevel::kO0;
      continue;
    }
    if (argument == "-O1") {
      options.optimization = muda::compiler::OptimizationLevel::kO1;
      continue;
    }
    if (argument == "-O2") {
      options.optimization = muda::compiler::OptimizationLevel::kO2;
      continue;
    }
    if (argument == "-O3") {
      options.optimization = muda::compiler::OptimizationLevel::kO3;
      continue;
    }
    if (argument == "-Ofast" || argument == "--use_fast_math") {
      options.optimization = muda::compiler::OptimizationLevel::kFast;
      continue;
    }
    if (argument == "-std=metal3.2" || argument == "--metal-standard=3.2") {
      options.metal_standard = muda::compiler::MetalStandard::kMetal32;
      continue;
    }
    if (argument == "-std=metal3.1" || argument == "--metal-standard=3.1") {
      options.metal_standard = muda::compiler::MetalStandard::kMetal31;
      continue;
    }
    if (argument == "-std=metal3.0" || argument == "--metal-standard=3.0") {
      options.metal_standard = muda::compiler::MetalStandard::kMetal30;
      continue;
    }
    if (argument == "-std=metal2.4" || argument == "--metal-standard=2.4") {
      options.metal_standard = muda::compiler::MetalStandard::kMetal24;
      continue;
    }
    if (StartsWith(argument, "-std=") || argument == "-g" || argument == "-lineinfo" ||
        StartsWith(argument, "-arch=")) {
      continue;
    }
    if (!argument.empty() && argument.front() == '-') {
      std::cerr << "mudac: unknown option: " << argument << "\n";
      return 2;
    }
    if (!options.input.empty()) {
      std::cerr << "mudac: multiple input files are not supported\n";
      return 2;
    }
    options.input = argument;
  }

  if (options.input.empty()) {
    PrintUsage(std::cerr);
    return 2;
  }

  const muda::compiler::CompileResult result = muda::compiler::Compile(options);
  if (result.status != muda::compiler::Status::kOk) {
    std::cerr << "mudac [" << result.diagnostic_id() << "]: " << result.message() << "\n";
    return 1;
  }

  std::cout << "wrote " << result.output_path.string() << " (" << result.kernel_name << ")\n";
  return 0;
}

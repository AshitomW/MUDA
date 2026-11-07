/**
 * @file main.cpp
 * @brief Implements the MUDA CUDA-to-Metal compiler CLI (`mudac`), compatible with `nvcc` flags.
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
      << "Usage: mudac [options] <file.cu|file.ptx>\n"
      << "\n"
      << "Options:\n"
      << "  -o <path>                 Write output to <path>\n"
      << "  -c                        Compile to host object file (.o) with embedded metallib\n"
      << "  -shared                   Compile to a dynamically loadable shared library (.dylib)\n"
      << "  -fPIC                     Generate position-independent code\n"
      << "  -S                        Emit translated Metal Shading Language (.metal) source\n"
      << "  --emit-metallib           Emit standalone Metal library (.metallib)\n"
      << "  -I <dir>, -I<dir>         Add header include directory\n"
      << "  -D <macro>[=<val>]        Define preprocessor macro\n"
      << "  -U <macro>                Undefine preprocessor macro\n"
      << "  -L <dir>, -L<dir>         Add library search directory\n"
      << "  -l <lib>, -l<lib>         Link against library <lib>\n"
      << "  -Xcompiler <flag>         Pass <flag> to the host C++ compiler\n"
      << "  -Xlinker <flag>           Pass <flag> to the host linker\n"
      << "  -k, --keep-intermediates  Keep generated .metal, .air, and .host.cpp files\n"
      << "  --muda-keep-msl           Keep generated .metal and intermediate files\n"
      << "  --muda-save-temps         Keep all intermediate compilation artifacts\n"
      << "  --muda-print-metadata     Print extracted kernel metadata JSON\n"
      << "  --muda-cache=<dir>        Use <dir> for the compiler artifact cache\n"
      << "  --muda-no-cache           Disable the compiler artifact cache\n"
      << "  -h, --help                Show this help\n"
      << "  --version, -V             Show version\n";
}

bool StartsWith(std::string_view text, std::string_view prefix) noexcept {
  return text.size() >= prefix.size() && text.substr(0, prefix.size()) == prefix;
}

void AppendCommaSeparated(std::string_view csv, std::vector<std::string>* target) {
  std::size_t start = 0;
  while (start <= csv.size()) {
    const std::size_t comma = csv.find(',', start);
    const std::string_view token =
        (comma == std::string_view::npos) ? csv.substr(start) : csv.substr(start, comma - start);
    if (!token.empty()) {
      target->emplace_back(token);
    }
    if (comma == std::string_view::npos) {
      break;
    }
    start = comma + 1;
  }
}

}  // namespace

int main(int argc, char** argv) {
  muda::compiler::CompileOptions options;

  for (int index = 1; index < argc; ++index) {
    const std::string_view argument = argv[index];
    if (argument == "--version" || argument == "-V") {
      std::cout << "mudac " << muda::kVersion << "\n"
                << "nvcc: MUDA clean-room CUDA-to-Metal compiler driver\n"
                << "Cuda compilation tools, release 12.2, V12.2.140\n";
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
    if (argument == "--muda-print-metadata") {
      options.print_metadata = true;
      continue;
    }
    if (argument == "-c" || argument == "--compile") {
      options.output_kind = muda::compiler::OutputKind::kObject;
      continue;
    }
    if (argument == "-shared" || argument == "--shared" || argument == "-dynamiclib") {
      options.output_kind = muda::compiler::OutputKind::kSharedLibrary;
      options.position_independent = true;
      continue;
    }
    if (argument == "-fPIC" || argument == "-fpic") {
      options.position_independent = true;
      continue;
    }
    if (argument == "-S" || argument == "-ptx" || argument == "--ptx") {
      options.output_kind = muda::compiler::OutputKind::kMslSource;
      continue;
    }
    if (argument == "--emit-metallib" || argument == "-cubin" || argument == "--cubin") {
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
    if (argument == "-o" || argument == "--output-file") {
      if (index + 1 >= argc) {
        std::cerr << "mudac: missing output path\n";
        return 2;
      }
      options.output = argv[++index];
      continue;
    }
    if (StartsWith(argument, "-o") && argument.size() > 2) {
      options.output = argument.substr(2);
      continue;
    }
    if (argument == "-I" || argument == "--include-path" || argument == "-isystem") {
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
    if (StartsWith(argument, "--include-path=")) {
      options.include_dirs.emplace_back(
          argument.substr(std::string_view("--include-path=").size()));
      continue;
    }
    if (argument == "-D" || argument == "--define-macro") {
      if (index + 1 >= argc) {
        std::cerr << "mudac: missing macro definition\n";
        return 2;
      }
      options.defines.push_back("-D" + std::string(argv[++index]));
      continue;
    }
    if (StartsWith(argument, "-D") && argument.size() > 2) {
      options.defines.emplace_back(argument);
      continue;
    }
    if (argument == "-U" || argument == "--undefine-macro") {
      if (index + 1 >= argc) {
        std::cerr << "mudac: missing macro name for -U\n";
        return 2;
      }
      options.defines.push_back("-U" + std::string(argv[++index]));
      continue;
    }
    if (StartsWith(argument, "-U") && argument.size() > 2) {
      options.defines.emplace_back(argument);
      continue;
    }
    if (argument == "-L" || argument == "--library-path") {
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
    if (argument == "-l" || argument == "--library") {
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
    if (argument == "-Xcompiler" || argument == "--compiler-options") {
      if (index + 1 >= argc) {
        std::cerr << "mudac: missing argument for " << argument << "\n";
        return 2;
      }
      AppendCommaSeparated(argv[++index], &options.host_compiler_flags);
      continue;
    }
    if (StartsWith(argument, "-Xcompiler=")) {
      AppendCommaSeparated(argument.substr(std::string_view("-Xcompiler=").size()),
                           &options.host_compiler_flags);
      continue;
    }
    if (StartsWith(argument, "--compiler-options=")) {
      AppendCommaSeparated(argument.substr(std::string_view("--compiler-options=").size()),
                           &options.host_compiler_flags);
      continue;
    }
    if (argument == "-Xlinker" || argument == "--linker-options") {
      if (index + 1 >= argc) {
        std::cerr << "mudac: missing argument for " << argument << "\n";
        return 2;
      }
      AppendCommaSeparated(argv[++index], &options.linker_flags);
      continue;
    }
    if (StartsWith(argument, "-Xlinker=") || StartsWith(argument, "-Wl,")) {
      options.linker_flags.emplace_back(argument);
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
    // NVIDIA nvcc compatibility flags that take 1 value argument
    if (argument == "-gencode" || argument == "--generate-code" || argument == "-arch" ||
        argument == "--gpu-architecture" || argument == "-code" || argument == "--gpu-code" ||
        argument == "-Xptxas" || argument == "--ptxas-options" || argument == "-Xnvlink" ||
        argument == "-cudart" || argument == "-ccbin" || argument == "--compiler-bindir") {
      if (index + 1 < argc) {
        ++index;
      }
      continue;
    }
    // NVIDIA nvcc single-token compatibility flags
    if (StartsWith(argument, "-std=") || argument == "-g" || argument == "-G" ||
        argument == "-lineinfo" || argument == "--generate-line-info" ||
        StartsWith(argument, "-arch=") || StartsWith(argument, "--gpu-architecture=") ||
        StartsWith(argument, "-gencode=") || StartsWith(argument, "--generate-code=") ||
        StartsWith(argument, "-Xptxas=") || StartsWith(argument, "--ptxas-options=") ||
        StartsWith(argument, "--expt-") || StartsWith(argument, "-Xcudafe") ||
        argument == "-rdc=true" || argument == "-rdc=false" || argument == "-w") {
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

/**
 * @file main.cpp
 * @brief Implements the MUDA CUDA-to-Metal compiler CLI.
 */

#include <filesystem>
#include <iostream>
#include <string_view>

#include "muda/compiler/driver.hpp"
#include "muda/version.hpp"

namespace {

void PrintUsage(std::ostream& out) {
  out << "mudac " << muda::kVersion << "\n"
      << "Usage: mudac [options] <file.cu>\n"
      << "\n"
      << "Options:\n"
      << "  -o <path>                 Write the Metal library to <path>\n"
      << "  -k, --keep-intermediates  Keep generated .metal and .air files\n"
      << "  -h, --help                Show this help\n"
      << "  --version                 Show version\n";
}

}  // namespace

int main(int argc, char** argv) {
  std::filesystem::path input;
  std::filesystem::path output;
  muda::compiler::IntermediatePolicy intermediates = muda::compiler::IntermediatePolicy::kRemove;

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
    if (argument == "-k" || argument == "--keep-intermediates") {
      intermediates = muda::compiler::IntermediatePolicy::kKeep;
      continue;
    }
    if (argument == "-o") {
      if (index + 1 >= argc) {
        std::cerr << "mudac: missing output path\n";
        return 2;
      }
      output = argv[++index];
      continue;
    }
    if (!argument.empty() && argument.front() == '-') {
      std::cerr << "mudac: unknown option: " << argument << "\n";
      return 2;
    }
    if (!input.empty()) {
      std::cerr << "mudac: multiple input files are not supported\n";
      return 2;
    }
    input = argument;
  }

  if (input.empty()) {
    PrintUsage(std::cerr);
    return 2;
  }

  if (output.empty()) {
    output = input;
    output.replace_extension(".metallib");
  }

  muda::compiler::CompileOptions options;
  options.input = input;
  options.output = output;
  options.intermediates = intermediates;

  const muda::compiler::CompileResult result = muda::compiler::Compile(options);
  if (result.status != muda::compiler::Status::kOk) {
    std::cerr << "mudac [" << result.diagnostic_id() << "]: " << result.message() << "\n";
    return 1;
  }

  std::cout << "wrote " << result.metallib_path.string() << " (" << result.kernel_name << ")\n";
  return 0;
}

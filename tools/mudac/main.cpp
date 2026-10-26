#include "muda/compiler/driver.hpp"
#include "muda/version.hpp"

#include <filesystem>
#include <iostream>
#include <string_view>

namespace {

void PrintUsage(std::ostream& out) {
  out << "mudac " << muda::kVersion << "\n"
      << "Usage: mudac [options] <file.cu>\n"
      << "\n"
      << "Options:\n"
      << "  -h, --help    Show this help\n"
      << "  --version     Show version\n";
}

}

int main(int argc, char** argv) {
  if (argc == 2) {
    const std::string_view arg = argv[1];
    if (arg == "--version") {
      std::cout << "mudac " << muda::kVersion << "\n";
      return 0;
    }
    if (arg == "-h" || arg == "--help") {
      PrintUsage(std::cout);
      return 0;
    }
  }

  if (argc != 2) {
    PrintUsage(std::cerr);
    return 2;
  }

  muda::compiler::CompileOptions options;
  options.input = std::filesystem::path(argv[1]);
  options.output = options.input;
  options.output.replace_extension();

  const muda::compiler::CompileResult result = muda::compiler::Compile(options);
  if (result.status != muda::compiler::Status::kOk) {
    std::cerr << "mudac: " << result.message << "\n";
    return 2;
  }

  return 0;
}

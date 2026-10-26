#include "muda/version.hpp"

#include <iostream>

int main() {
  if (muda::kProjectName != "MUDA") {
    std::cerr << "unexpected project name\n";
    return 1;
  }
  if (muda::kVersion.empty() || muda::kVersion.find('.') == std::string_view::npos) {
    std::cerr << "unexpected version string: " << muda::kVersion << "\n";
    return 1;
  }
  return 0;
}

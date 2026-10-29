/**
 * @file version_test.cpp
 * @brief Tests generated version metadata.
 */

#include "muda/version.hpp"

#include <iostream>
#include <string_view>

static_assert(muda::kProjectName == "MUDA", "unexpected project name");
static_assert(!muda::kVersion.empty(), "version string must not be empty");
static_assert(muda::kVersion.find('.') != std::string_view::npos,
              "version string must contain a dot separator");

int main() {
  const std::string_view runtime_version = muda::kVersion;
  if (runtime_version.empty() || runtime_version.find('.') == std::string_view::npos) {
    std::cerr << "unexpected version string: " << runtime_version << "\n";
    return 1;
  }
  return 0;
}

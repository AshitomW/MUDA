/**
 * @file toolchain.cpp
 * @brief Locates the Metal compiler and library tools.
 */

#include "muda/compiler/toolchain.hpp"

#include <array>
#include <cstdio>
#include <string>
#include <string_view>

#include "process.hpp"

namespace muda::compiler {

bool MetalToolchainAvailable() {
  constexpr std::array<std::string_view, 3> kCheckMetal = {"xcrun", "-f", "metal"};
  constexpr std::array<std::string_view, 3> kCheckMetallib = {"xcrun", "-f", "metallib"};
  return detail::RunProcess(kCheckMetal, detail::ProcessOutputMode::kSilent).succeeded() &&
         detail::RunProcess(kCheckMetallib, detail::ProcessOutputMode::kSilent).succeeded();
}

std::string GetMetalToolchainVersion() noexcept {
  static const std::string kVersion = []() {
    FILE* const pipe = popen("xcrun --sdk macosx metal --version 2>/dev/null", "r");
    if (pipe == nullptr) {
      return std::string("apple_metal_unknown");
    }
    std::array<char, 256> buffer{};
    std::string result;
    if (std::fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
      result = buffer.data();
      while (!result.empty() && (result.back() == '\n' || result.back() == '\r')) {
        result.pop_back();
      }
    }
    (void)pclose(pipe);
    return result.empty() ? std::string("apple_metal_unknown") : result;
  }();
  return kVersion;
}

}  // namespace muda::compiler

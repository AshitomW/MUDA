/**
 * @file toolchain.cpp
 * @brief Locates the Metal compiler and library tools.
 */

#include "muda/compiler/toolchain.hpp"

#include <array>
#include <string_view>

#include "process.hpp"

namespace muda::compiler {

bool MetalToolchainAvailable() {
  constexpr std::array<std::string_view, 3> kCheckMetal = {"xcrun", "-f", "metal"};
  constexpr std::array<std::string_view, 3> kCheckMetallib = {"xcrun", "-f", "metallib"};
  return detail::RunProcess(kCheckMetal, detail::ProcessOutputMode::kSilent).succeeded() &&
         detail::RunProcess(kCheckMetallib, detail::ProcessOutputMode::kSilent).succeeded();
}

}  // namespace muda::compiler

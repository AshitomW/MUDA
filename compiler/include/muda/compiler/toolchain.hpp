/**
 * @file toolchain.hpp
 * @brief Detects the installed Metal toolchain.
 */

#pragma once

#include <string>

namespace muda::compiler {

[[nodiscard]] bool MetalToolchainAvailable();
[[nodiscard]] std::string GetMetalToolchainVersion() noexcept;

}  // namespace muda::compiler

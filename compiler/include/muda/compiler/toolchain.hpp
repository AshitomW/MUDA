/**
 * @file toolchain.hpp
 * @brief Detects the installed Metal toolchain.
 */

#pragma once

namespace muda::compiler {

[[nodiscard]] bool MetalToolchainAvailable();

}  // namespace muda::compiler

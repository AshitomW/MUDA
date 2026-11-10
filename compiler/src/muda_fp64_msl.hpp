/**
 * @file muda_fp64_msl.hpp
 * @brief Declaration of MSL Double-Single 64-bit floating point emulation preamble.
 */

#pragma once

#include <string_view>

#include "muda/compiler/types.hpp"

namespace muda::compiler {

/**
 * @brief Returns the MSL source preamble defining muda_fp64 and emulation.
 * @param policy Emulation mode: kFast48 (Double-Single) or kIeee64 (bit-exact IEEE-754).
 */
[[nodiscard]] std::string_view GetMudaFp64MslPreamble(
    Fp64Policy policy = Fp64Policy::kFast48) noexcept;

}  // namespace muda::compiler

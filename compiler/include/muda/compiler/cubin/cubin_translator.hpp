/**
 * @file cubin_translator.hpp
 * @brief Unified translation entry point for NVIDIA Fatbin containers and ELF CUBIN binaries to MSL.
 */

#pragma once

#include <cstdint>
#include <span>
#include <string_view>

#include "muda/compiler/translator.hpp"

namespace muda::compiler::cubin {

/**
 * @brief Checks if a buffer represents an NVIDIA Fatbinary container.
 */
[[nodiscard]] bool IsFatbinary(std::span<const std::uint8_t> data) noexcept;

/**
 * @brief Checks if a buffer represents an NVIDIA ELF CUBIN binary.
 */
[[nodiscard]] bool IsElfCubin(std::span<const std::uint8_t> data) noexcept;

/**
 * @brief Translates an NVIDIA Fatbinary or ELF CUBIN binary directly into verified Metal Shading Language.
 */
[[nodiscard]] TranslationResult TranslateCubinOrFatbinToMsl(std::span<const std::uint8_t> data,
                                                           const TranslationOptions& options = {});

}  // namespace muda::compiler::cubin


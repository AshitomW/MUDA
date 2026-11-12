/**
 * @file sass_decoder.hpp
 * @brief Clean-room 128-bit SASS instruction decoder for NVIDIA Turing (SM 7.5) and Ampere (SM 8.0).
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "muda/compiler/cubin/sass_types.hpp"

namespace muda::compiler::cubin {

class SassDecoder final {
 public:
  /**
   * @brief Decodes a stream of 128-bit SASS machine instructions into structured SassInstruction objects.
   *
   * @param text_bytes Raw bytecode from an ELF .text section.
   * @param arch_sm Target SM architecture (e.g. 70, 75, 80, 86).
   * @return std::vector<SassInstruction> Decoded instructions with resolved operands and targets.
   */
  [[nodiscard]] static std::vector<SassInstruction> Decode(std::span<const std::uint8_t> text_bytes,
                                                          std::uint32_t arch_sm = 75);

  /**
   * @brief Decodes a single 128-bit SASS instruction word at the given program counter (PC).
   */
  [[nodiscard]] static SassInstruction DecodeInstruction(std::span<const std::uint8_t> word_bytes,
                                                         std::uint64_t pc,
                                                         std::uint32_t arch_sm = 75);
};

}  // namespace muda::compiler::cubin


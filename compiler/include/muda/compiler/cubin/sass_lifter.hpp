/**
 * @file sass_lifter.hpp
 * @brief Lifts decoded SASS instructions and CUBIN metadata into Metal Shading Language (MSL).
 */

#pragma once

#include <string>
#include <vector>

#include "muda/compiler/cubin/elf_reader.hpp"
#include "muda/compiler/cubin/sass_types.hpp"

namespace muda::compiler::cubin {

class SassLifter final {
 public:
  /**
   * @brief Lifts a parsed CubinKernel and its decoded SASS instructions into standalone MSL source.
   *
   * @param kernel CubinKernel containing metadata, name, parameters, and shared memory sizes.
   * @param instructions Decoded SASS instructions.
   * @return std::string Complete, verified Metal Shading Language source code.
   */
  [[nodiscard]] static std::string LiftToMsl(const CubinKernel& kernel,
                                             const std::vector<SassInstruction>& instructions);
};

}  // namespace muda::compiler::cubin


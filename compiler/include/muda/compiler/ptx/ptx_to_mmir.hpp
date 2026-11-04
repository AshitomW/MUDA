/**
 * @file ptx_to_mmir.hpp
 * @brief Lowers PTX AST, CFG, and inferred types into MMIR Module and MSL.
 */

#pragma once

#include <string_view>

#include "muda/compiler/mmir.hpp"
#include "muda/compiler/ptx/ast.hpp"
#include "muda/compiler/translator.hpp"

namespace muda::compiler::ptx {

/**
 * @brief Lowers a parsed PTX Program into an MMIR Module.
 */
[[nodiscard]] mmir::Module LowerPtxToMmir(const Program& program);

/**
 * @brief End-to-end translation entry point converting PTX source into verified Metal Shading
 * Language.
 */
[[nodiscard]] TranslationResult TranslatePtxToMsl(std::string_view ptx_source);

}  // namespace muda::compiler::ptx

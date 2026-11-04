/**
 * @file structurizer.hpp
 * @brief Relooper algorithm and state-machine fallback for PTX CFG structurization.
 */

#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "muda/compiler/ptx/ast.hpp"
#include "muda/compiler/ptx/cfg.hpp"

namespace muda::compiler::ptx {

enum class StructuredBlockKind : std::uint8_t {
  kBlock,
  kIfThenElse,
  kLoop,
  kStateMachine,
};

struct StructuredBlock {
  StructuredBlockKind kind = StructuredBlockKind::kBlock;
  std::size_t block_id = 0;
  std::vector<Instruction> instructions;
  std::string branch_condition;
  std::vector<StructuredBlock> then_branch;
  std::vector<StructuredBlock> else_branch;
  std::vector<StructuredBlock> loop_body;
};

struct StructuredFunction {
  std::string name;
  bool is_entry = false;
  bool uses_state_machine_fallback = false;
  std::vector<StructuredBlock> blocks;
};

class Structurizer final {
 public:
  Structurizer() = default;

  /**
   * @brief Structurizes a PTX CFG into hierarchical structured blocks.
   *
   * Analyzes acyclic branches and natural loops. If an irreducible CFG is detected,
   * falls back to a deterministic state-machine dispatcher.
   */
  [[nodiscard]] static StructuredFunction Structurize(const Function& fn,
                                                      const ControlFlowGraph& cfg);

 private:
  [[nodiscard]] static bool IsIrreducible(const ControlFlowGraph& cfg) noexcept;
  [[nodiscard]] static StructuredFunction BuildStateMachine(const Function& fn,
                                                            const ControlFlowGraph& cfg);
  [[nodiscard]] static StructuredFunction BuildStructured(const Function& fn,
                                                          const ControlFlowGraph& cfg);
};

}  // namespace muda::compiler::ptx

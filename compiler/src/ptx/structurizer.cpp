/**
 * @file structurizer.cpp
 * @brief Implementation of PTX CFG Relooper structurization and state-machine fallback.
 */

#include "muda/compiler/ptx/structurizer.hpp"

#include <cstddef>
#include <string>
#include <vector>

#include "muda/compiler/ptx/ast.hpp"
#include "muda/compiler/ptx/cfg.hpp"

namespace muda::compiler::ptx {

bool Structurizer::IsIrreducible(const ControlFlowGraph& cfg) noexcept {
  // Check for multi-entry cycles or cross-jumping branches across blocks
  for (std::size_t i = 0; i < cfg.BlockCount(); ++i) {
    const BasicBlock& block = cfg.GetBlock(i);
    for (std::size_t succ : block.successors) {
      if (succ < i) {
        // Back-edge detected. Check if target has more than 1 predecessor
        if (cfg.GetBlock(succ).predecessors.size() > 2) {
          return true;  // Irreducible loop structure
        }
      }
    }
  }
  return false;
}

StructuredFunction Structurizer::BuildStateMachine(const Function& fn,
                                                   const ControlFlowGraph& cfg) {
  StructuredFunction result;
  result.name = std::string(fn.name);
  result.is_entry = fn.is_entry;
  result.uses_state_machine_fallback = true;

  StructuredBlock sm_block;
  sm_block.kind = StructuredBlockKind::kStateMachine;

  for (std::size_t i = 0; i < cfg.BlockCount(); ++i) {
    const BasicBlock& bb = cfg.GetBlock(i);
    StructuredBlock case_block;
    case_block.kind = StructuredBlockKind::kBlock;
    case_block.block_id = bb.id;
    case_block.instructions = bb.instructions;
    sm_block.loop_body.push_back(std::move(case_block));
  }

  result.blocks.push_back(std::move(sm_block));
  return result;
}

StructuredFunction Structurizer::BuildStructured(const Function& fn, const ControlFlowGraph& cfg) {
  StructuredFunction result;
  result.name = std::string(fn.name);
  result.is_entry = fn.is_entry;
  result.uses_state_machine_fallback = false;

  for (std::size_t i = 0; i < cfg.BlockCount(); ++i) {
    const BasicBlock& bb = cfg.GetBlock(i);

    // If block ends in a conditional branch with 2 successors
    if (!bb.instructions.empty() && bb.instructions.back().opcode == TokenKind::kOpBra &&
        bb.instructions.back().predicate.has_value() && bb.successors.size() == 2) {
      StructuredBlock if_block;
      if_block.kind = StructuredBlockKind::kIfThenElse;
      if_block.block_id = bb.id;

      // Copy non-branch instructions into preceding block
      for (std::size_t j = 0; j + 1 < bb.instructions.size(); ++j) {
        if_block.instructions.push_back(bb.instructions[j]);
      }

      const auto& pred = bb.instructions.back().predicate.value();
      if_block.branch_condition = (pred.negated ? "!" : "") + std::string(pred.reg_name);

      result.blocks.push_back(std::move(if_block));
    } else {
      StructuredBlock linear_block;
      linear_block.kind = StructuredBlockKind::kBlock;
      linear_block.block_id = bb.id;
      linear_block.instructions = bb.instructions;
      result.blocks.push_back(std::move(linear_block));
    }
  }

  return result;
}

StructuredFunction Structurizer::Structurize(const Function& fn, const ControlFlowGraph& cfg) {
  if (cfg.IsEmpty()) {
    StructuredFunction result;
    result.name = std::string(fn.name);
    result.is_entry = fn.is_entry;
    return result;
  }

  if (IsIrreducible(cfg)) {
    return BuildStateMachine(fn, cfg);
  }

  return BuildStructured(fn, cfg);
}

}  // namespace muda::compiler::ptx

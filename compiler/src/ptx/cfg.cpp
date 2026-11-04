/**
 * @file cfg.cpp
 * @brief Implementation of Control Flow Graph construction.
 */

#include "muda/compiler/ptx/cfg.hpp"

#include <algorithm>
#include <cstddef>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "muda/compiler/ptx/ast.hpp"

namespace muda::compiler::ptx {

bool BasicBlock::HasSuccessor(std::size_t block_id) const noexcept {
  return std::find(successors.begin(), successors.end(), block_id) != successors.end();
}

bool BasicBlock::HasPredecessor(std::size_t block_id) const noexcept {
  return std::find(predecessors.begin(), predecessors.end(), block_id) != predecessors.end();
}

void ControlFlowGraph::ConnectBlocks(std::size_t from_id, std::size_t to_id) {
  if (from_id < blocks_.size() && to_id < blocks_.size()) {
    if (!blocks_[from_id].HasSuccessor(to_id)) {
      blocks_[from_id].successors.push_back(to_id);
    }
    if (!blocks_[to_id].HasPredecessor(from_id)) {
      blocks_[to_id].predecessors.push_back(from_id);
    }
  }
}

ControlFlowGraph ControlFlowGraph::Build(const Function& fn) {
  ControlFlowGraph cfg;
  if (fn.statements.empty()) {
    return cfg;
  }

  // Pass 1: Chunk statements into basic blocks by labels and branch boundaries
  std::unordered_map<std::string_view, std::size_t> label_to_block_id;
  BasicBlock current_block;
  current_block.id = 0;

  for (const Statement& stmt : fn.statements) {
    if (stmt.kind == StatementKind::kLabel) {
      if (!current_block.instructions.empty() || !current_block.label.empty()) {
        cfg.blocks_.push_back(std::move(current_block));
        current_block = BasicBlock{};
        current_block.id = cfg.blocks_.size();
      }
      current_block.label = stmt.label_name;
      label_to_block_id[stmt.label_name] = current_block.id;
    } else {
      current_block.instructions.push_back(stmt.instruction);
      // If terminal instruction, close block
      if (stmt.instruction.opcode == TokenKind::kOpBra ||
          stmt.instruction.opcode == TokenKind::kOpRet ||
          stmt.instruction.opcode == TokenKind::kOpExit) {
        cfg.blocks_.push_back(std::move(current_block));
        current_block = BasicBlock{};
        current_block.id = cfg.blocks_.size();
      }
    }
  }

  if (!current_block.instructions.empty() || !current_block.label.empty()) {
    cfg.blocks_.push_back(std::move(current_block));
  }

  // Re-map label_to_block_id in case labels were associated with empty leading blocks
  for (std::size_t i = 0; i < cfg.blocks_.size(); ++i) {
    if (!cfg.blocks_[i].label.empty()) {
      label_to_block_id[cfg.blocks_[i].label] = i;
    }
  }

  // Pass 2: Connect control flow edges
  for (std::size_t i = 0; i < cfg.blocks_.size(); ++i) {
    BasicBlock& block = cfg.blocks_[i];
    if (block.instructions.empty()) {
      if (i + 1 < cfg.blocks_.size()) {
        cfg.ConnectBlocks(i, i + 1);
      }
      continue;
    }

    const Instruction& last_inst = block.instructions.back();
    if (last_inst.opcode == TokenKind::kOpRet || last_inst.opcode == TokenKind::kOpExit) {
      // Terminal instruction: no successors
      continue;
    }

    if (last_inst.opcode == TokenKind::kOpBra) {
      // Find branch target
      if (!last_inst.operands.empty()) {
        const std::string_view target_label = last_inst.operands[0].name;
        const auto it = label_to_block_id.find(target_label);
        if (it != label_to_block_id.end()) {
          cfg.ConnectBlocks(i, it->second);
        }
      }

      // If branch is conditional (has predicate), also add fallthrough edge
      if (last_inst.predicate.has_value() && i + 1 < cfg.blocks_.size()) {
        cfg.ConnectBlocks(i, i + 1);
      }
    } else {
      // Normal sequential fallthrough
      if (i + 1 < cfg.blocks_.size()) {
        cfg.ConnectBlocks(i, i + 1);
      }
    }
  }

  return cfg;
}

}  // namespace muda::compiler::ptx

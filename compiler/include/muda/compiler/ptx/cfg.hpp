/**
 * @file cfg.hpp
 * @brief Control Flow Graph (CFG) construction and basic block analysis for PTX.
 */

#pragma once

#include <cstddef>
#include <string_view>
#include <vector>

#include "muda/compiler/ptx/ast.hpp"

namespace muda::compiler::ptx {

struct BasicBlock {
  std::size_t id = 0;
  std::string_view label;
  std::vector<Instruction> instructions;
  std::vector<std::size_t> predecessors;
  std::vector<std::size_t> successors;

  [[nodiscard]] bool HasSuccessor(std::size_t block_id) const noexcept;
  [[nodiscard]] bool HasPredecessor(std::size_t block_id) const noexcept;
};

class ControlFlowGraph final {
 public:
  ControlFlowGraph() = default;

  /**
   * @brief Constructs a Control Flow Graph from a parsed PTX Function.
   */
  [[nodiscard]] static ControlFlowGraph Build(const Function& fn);

  [[nodiscard]] std::size_t BlockCount() const noexcept { return blocks_.size(); }
  [[nodiscard]] bool IsEmpty() const noexcept { return blocks_.empty(); }

  [[nodiscard]] const BasicBlock& GetBlock(std::size_t index) const noexcept {
    return blocks_[index];
  }
  [[nodiscard]] BasicBlock& GetBlock(std::size_t index) noexcept { return blocks_[index]; }

  [[nodiscard]] const std::vector<BasicBlock>& blocks() const noexcept { return blocks_; }

  [[nodiscard]] std::size_t entry_block_id() const noexcept { return entry_block_id_; }

 private:
  void ConnectBlocks(std::size_t from_id, std::size_t to_id);

  std::vector<BasicBlock> blocks_;
  std::size_t entry_block_id_ = 0;
};

}  // namespace muda::compiler::ptx

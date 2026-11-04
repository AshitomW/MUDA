/**
 * @file parser.hpp
 * @brief Table-driven LALR(1) shift-reduce parser for PTX programs.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "muda/compiler/ptx/arena.hpp"
#include "muda/compiler/ptx/ast.hpp"
#include "muda/compiler/ptx/lexer.hpp"
#include "muda/compiler/ptx/tokens.hpp"

namespace muda::compiler::ptx {

struct ParseResult {
  bool success = false;
  Program program;
  std::string error_message;
  SourceLocation error_location;

  [[nodiscard]] constexpr bool ok() const noexcept { return success; }
};

class Parser final {
 public:
  Parser(Lexer& lexer, Arena& arena) noexcept;

  /**
   * @brief Parses an entire PTX compilation unit.
   */
  [[nodiscard]] ParseResult ParseProgram();

  [[nodiscard]] Arena& arena() noexcept { return arena_; }

 private:
  using StateId = std::uint16_t;

  enum class ActionKind : std::uint8_t {
    kError = 0,
    kShift,
    kReduce,
    kAccept,
  };

  struct Action {
    ActionKind kind = ActionKind::kError;
    std::uint16_t target = 0;  // next StateId for Shift, rule ID for Reduce
  };

  struct SemanticSymbol {
    Token token;
    VariableDecl decl;
    std::vector<VariableDecl> decls;
    Operand operand;
    std::vector<Operand> operands;
    Instruction instruction;
    Statement statement;
    std::vector<Statement> statements;
    Function function;
  };

  [[nodiscard]] Action QueryAction(StateId state, const Token& lookahead) const noexcept;
  [[nodiscard]] StateId QueryGoto(StateId state, TokenKind non_terminal) const noexcept;

  void ExecuteReduce(std::uint16_t rule_id, Program* program, Function* current_function);

  Lexer& lexer_;
  Arena& arena_;
  std::vector<StateId> state_stack_;
  std::vector<SemanticSymbol> symbol_stack_;
};

}  // namespace muda::compiler::ptx

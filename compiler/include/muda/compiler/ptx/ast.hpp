/**
 * @file ast.hpp
 * @brief Strongly-typed PTX Abstract Syntax Tree (AST) definitions.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

#include "muda/compiler/ptx/tokens.hpp"

namespace muda::compiler::ptx {

enum class StateSpace : std::uint8_t {
  kReg,
  kParam,
  kShared,
  kGlobal,
  kConst,
  kLocal,
  kGeneric,
};

enum class TypeKind : std::uint8_t {
  kPred,
  kB8,
  kB16,
  kB32,
  kB64,
  kU8,
  kU16,
  kU32,
  kU64,
  kS8,
  kS16,
  kS32,
  kS64,
  kF16,
  kF16x2,
  kF32,
  kF64,
};

[[nodiscard]] constexpr std::size_t TypeByteSize(TypeKind kind) noexcept {
  switch (kind) {
    case TypeKind::kPred:
    case TypeKind::kB8:
    case TypeKind::kU8:
    case TypeKind::kS8:
      return 1;
    case TypeKind::kB16:
    case TypeKind::kU16:
    case TypeKind::kS16:
    case TypeKind::kF16:
      return 2;
    case TypeKind::kB32:
    case TypeKind::kU32:
    case TypeKind::kS32:
    case TypeKind::kF16x2:
    case TypeKind::kF32:
      return 4;
    case TypeKind::kB64:
    case TypeKind::kU64:
    case TypeKind::kS64:
    case TypeKind::kF64:
      return 8;
  }
  return 4;
}

[[nodiscard]] constexpr bool IsFloatType(TypeKind kind) noexcept {
  return kind == TypeKind::kF16 || kind == TypeKind::kF16x2 || kind == TypeKind::kF32 ||
         kind == TypeKind::kF64;
}

[[nodiscard]] constexpr bool IsIntegerType(TypeKind kind) noexcept {
  return (kind >= TypeKind::kU8 && kind <= TypeKind::kS64) ||
         (kind >= TypeKind::kB8 && kind <= TypeKind::kB64);
}

enum class OperandKind : std::uint8_t {
  kRegister,
  kSpecialRegister,
  kImmediateInt,
  kImmediateFloat,
  kAddress,
  kLabel,
  kVectorList,
};

struct Operand {
  OperandKind kind = OperandKind::kRegister;
  std::string_view name;
  std::int64_t int_val = 0;
  double float_val = 0.0;
  std::int64_t offset = 0;
  std::vector<Operand> vector_elements;

  [[nodiscard]] static Operand MakeRegister(std::string_view reg) noexcept {
    Operand op;
    op.kind = OperandKind::kRegister;
    op.name = reg;
    return op;
  }

  [[nodiscard]] static Operand MakeSpecialRegister(std::string_view reg) noexcept {
    Operand op;
    op.kind = OperandKind::kSpecialRegister;
    op.name = reg;
    return op;
  }

  [[nodiscard]] static Operand MakeImmediateInt(std::int64_t val) noexcept {
    Operand op;
    op.kind = OperandKind::kImmediateInt;
    op.int_val = val;
    return op;
  }

  [[nodiscard]] static Operand MakeImmediateFloat(double val) noexcept {
    Operand op;
    op.kind = OperandKind::kImmediateFloat;
    op.float_val = val;
    return op;
  }

  [[nodiscard]] static Operand MakeAddress(std::string_view base_reg,
                                           std::int64_t offset = 0) noexcept {
    Operand op;
    op.kind = OperandKind::kAddress;
    op.name = base_reg;
    op.offset = offset;
    return op;
  }

  [[nodiscard]] static Operand MakeLabel(std::string_view label) noexcept {
    Operand op;
    op.kind = OperandKind::kLabel;
    op.name = label;
    return op;
  }
};

struct Predicate {
  std::string_view reg_name;
  bool negated = false;
};

struct Instruction {
  TokenKind opcode = TokenKind::kUnknown;
  TypeKind type = TypeKind::kB32;
  std::vector<std::string_view> modifiers;
  std::optional<Predicate> predicate;
  std::vector<Operand> operands;
  SourceLocation location;
};

enum class StatementKind : std::uint8_t {
  kInstruction,
  kLabel,
};

struct Statement {
  StatementKind kind = StatementKind::kInstruction;
  std::string_view label_name;
  Instruction instruction;

  [[nodiscard]] static Statement MakeLabel(std::string_view label) noexcept {
    Statement stmt;
    stmt.kind = StatementKind::kLabel;
    stmt.label_name = label;
    return stmt;
  }

  [[nodiscard]] static Statement MakeInstruction(Instruction inst) noexcept {
    Statement stmt;
    stmt.kind = StatementKind::kInstruction;
    stmt.instruction = std::move(inst);
    return stmt;
  }
};

struct VariableDecl {
  std::string_view name;
  StateSpace space = StateSpace::kReg;
  TypeKind type = TypeKind::kB32;
  std::size_t count = 1;
  std::size_t alignment = 0;
  bool is_array = false;
};

struct Function {
  std::string_view name;
  bool is_entry = false;
  std::vector<VariableDecl> return_params;
  std::vector<VariableDecl> params;
  std::vector<VariableDecl> declarations;
  std::vector<Statement> statements;
};

struct Program {
  std::uint32_t version_major = 6;
  std::uint32_t version_minor = 0;
  std::string_view target = "sm_70";
  std::uint32_t address_size = 64;
  std::vector<VariableDecl> globals;
  std::vector<Function> functions;
};

}  // namespace muda::compiler::ptx

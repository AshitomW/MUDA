/**
 * @file tokens.hpp
 * @brief Strongly-typed PTX tokens, directives, and token classifications.
 */

#pragma once

#include <cstdint>
#include <string_view>

namespace muda::compiler::ptx {

enum class TokenKind : std::uint16_t {
  // Special / control
  kUnknown = 0,
  kEndOfFile,

  // Directives
  kDirectiveVersion,
  kDirectiveTarget,
  kDirectiveAddressSize,
  kDirectiveVisible,
  kDirectiveEntry,
  kDirectiveFunc,
  kDirectiveParam,
  kDirectiveReg,
  kDirectiveShared,
  kDirectiveGlobal,
  kDirectiveConst,
  kDirectiveLocal,
  kDirectiveAlign,
  kDirectiveWeak,
  kDirectiveExtern,
  kDirectiveFile,
  kDirectiveLoc,

  // Data types
  kTypePred,
  kTypeB8,
  kTypeB16,
  kTypeB32,
  kTypeB64,
  kTypeU8,
  kTypeU16,
  kTypeU32,
  kTypeU64,
  kTypeS8,
  kTypeS16,
  kTypeS32,
  kTypeS64,
  kTypeF16,
  kTypeF16x2,
  kTypeF32,
  kTypeF64,

  // Instruction opcodes
  kOpMov,
  kOpLd,
  kOpSt,
  kOpCvta,
  kOpCvt,
  kOpSelp,
  kOpAdd,
  kOpSub,
  kOpMul,
  kOpMad,
  kOpFma,
  kOpDiv,
  kOpRem,
  kOpNeg,
  kOpAbs,
  kOpMin,
  kOpMax,
  kOpRsqrt,
  kOpSqrt,
  kOpSin,
  kOpCos,
  kOpRcp,
  kOpEx2,
  kOpLg2,
  kOpAnd,
  kOpOr,
  kOpXor,
  kOpNot,
  kOpShl,
  kOpShr,
  kOpCnot,
  kOpSetp,
  kOpBra,
  kOpRet,
  kOpCall,
  kOpExit,
  kOpBar,
  kOpAtom,
  kOpRed,
  kOpVote,
  kOpShfl,

  // Literals and identifiers
  kIdentifier,
  kRegister,
  kIntegerLiteral,
  kFloatLiteral,
  kStringLiteral,

  // Special registers
  kSpecialTid,
  kSpecialNtid,
  kSpecialCtaid,
  kSpecialNctaid,
  kSpecialLaneId,
  kSpecialWarpId,
  kSpecialClock,
  kSpecialClock64,

  // Punctuation and symbols
  kSemicolon,     // ;
  kComma,         // ,
  kColon,         // :
  kLeftBrace,     // {
  kRightBrace,    // }
  kLeftParen,     // (
  kRightParen,    // )
  kLeftBracket,   // [
  kRightBracket,  // ]
  kLeftAngle,     // <
  kRightAngle,    // >
  kEquals,        // =
  kAt,            // @
  kExclamation,   // !
  kDot,           // .
  kPlus,          // +
  kMinus,         // -
};

struct SourceLocation {
  std::uint32_t line = 1;
  std::uint32_t column = 1;
};

struct Token {
  TokenKind kind = TokenKind::kUnknown;
  std::string_view text;
  SourceLocation location;

  [[nodiscard]] constexpr bool Is(TokenKind expected) const noexcept { return kind == expected; }
  [[nodiscard]] constexpr bool IsNot(TokenKind expected) const noexcept { return kind != expected; }
};

[[nodiscard]] std::string_view TokenKindName(TokenKind kind) noexcept;
[[nodiscard]] bool IsDirective(TokenKind kind) noexcept;
[[nodiscard]] bool IsType(TokenKind kind) noexcept;
[[nodiscard]] bool IsInstruction(TokenKind kind) noexcept;
[[nodiscard]] bool IsSpecialRegister(TokenKind kind) noexcept;

}  // namespace muda::compiler::ptx

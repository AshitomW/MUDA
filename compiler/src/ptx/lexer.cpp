/**
 * @file lexer.cpp
 * @brief Implementation of high-performance zero-copy PTX lexer.
 */

#include "muda/compiler/ptx/lexer.hpp"

#include <cctype>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "muda/compiler/ptx/tokens.hpp"

namespace muda::compiler::ptx {
namespace {

bool IsIdentifierChar(char c) noexcept {
  return std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '_' || c == '$';
}

TokenKind LookupDirectiveOrType(std::string_view text) noexcept {
  static const std::unordered_map<std::string_view, TokenKind> kMap = {
      {".version", TokenKind::kDirectiveVersion},
      {".target", TokenKind::kDirectiveTarget},
      {".address_size", TokenKind::kDirectiveAddressSize},
      {".visible", TokenKind::kDirectiveVisible},
      {".entry", TokenKind::kDirectiveEntry},
      {".func", TokenKind::kDirectiveFunc},
      {".param", TokenKind::kDirectiveParam},
      {".reg", TokenKind::kDirectiveReg},
      {".shared", TokenKind::kDirectiveShared},
      {".global", TokenKind::kDirectiveGlobal},
      {".const", TokenKind::kDirectiveConst},
      {".local", TokenKind::kDirectiveLocal},
      {".align", TokenKind::kDirectiveAlign},
      {".weak", TokenKind::kDirectiveWeak},
      {".extern", TokenKind::kDirectiveExtern},
      {".file", TokenKind::kDirectiveFile},
      {".loc", TokenKind::kDirectiveLoc},

      {".pred", TokenKind::kTypePred},
      {".b8", TokenKind::kTypeB8},
      {".b16", TokenKind::kTypeB16},
      {".b32", TokenKind::kTypeB32},
      {".b64", TokenKind::kTypeB64},
      {".u8", TokenKind::kTypeU8},
      {".u16", TokenKind::kTypeU16},
      {".u32", TokenKind::kTypeU32},
      {".u64", TokenKind::kTypeU64},
      {".s8", TokenKind::kTypeS8},
      {".s16", TokenKind::kTypeS16},
      {".s32", TokenKind::kTypeS32},
      {".s64", TokenKind::kTypeS64},
      {".f16", TokenKind::kTypeF16},
      {".f16x2", TokenKind::kTypeF16x2},
      {".f32", TokenKind::kTypeF32},
      {".f64", TokenKind::kTypeF64},
  };

  const auto it = kMap.find(text);
  return (it != kMap.end()) ? it->second : TokenKind::kIdentifier;
}

TokenKind LookupOpcode(std::string_view text) noexcept {
  static const std::unordered_map<std::string_view, TokenKind> kMap = {
      {"mov", TokenKind::kOpMov},   {"ld", TokenKind::kOpLd},     {"st", TokenKind::kOpSt},
      {"cvta", TokenKind::kOpCvta}, {"cvt", TokenKind::kOpCvt},   {"selp", TokenKind::kOpSelp},
      {"add", TokenKind::kOpAdd},   {"sub", TokenKind::kOpSub},   {"mul", TokenKind::kOpMul},
      {"mad", TokenKind::kOpMad},   {"fma", TokenKind::kOpFma},   {"div", TokenKind::kOpDiv},
      {"rem", TokenKind::kOpRem},   {"neg", TokenKind::kOpNeg},   {"abs", TokenKind::kOpAbs},
      {"min", TokenKind::kOpMin},   {"max", TokenKind::kOpMax},   {"rsqrt", TokenKind::kOpRsqrt},
      {"sqrt", TokenKind::kOpSqrt}, {"sin", TokenKind::kOpSin},   {"cos", TokenKind::kOpCos},
      {"rcp", TokenKind::kOpRcp},   {"ex2", TokenKind::kOpEx2},   {"lg2", TokenKind::kOpLg2},
      {"and", TokenKind::kOpAnd},   {"or", TokenKind::kOpOr},     {"xor", TokenKind::kOpXor},
      {"not", TokenKind::kOpNot},   {"shl", TokenKind::kOpShl},   {"shr", TokenKind::kOpShr},
      {"cnot", TokenKind::kOpCnot}, {"setp", TokenKind::kOpSetp}, {"bra", TokenKind::kOpBra},
      {"ret", TokenKind::kOpRet},   {"call", TokenKind::kOpCall}, {"exit", TokenKind::kOpExit},
      {"bar", TokenKind::kOpBar},   {"atom", TokenKind::kOpAtom}, {"red", TokenKind::kOpRed},
      {"vote", TokenKind::kOpVote}, {"shfl", TokenKind::kOpShfl},
  };

  const auto it = kMap.find(text);
  return (it != kMap.end()) ? it->second : TokenKind::kIdentifier;
}

TokenKind LookupSpecialRegister(std::string_view text) noexcept {
  if (text.starts_with("%tid"))
    return TokenKind::kSpecialTid;
  if (text.starts_with("%ntid"))
    return TokenKind::kSpecialNtid;
  if (text.starts_with("%ctaid"))
    return TokenKind::kSpecialCtaid;
  if (text.starts_with("%nctaid"))
    return TokenKind::kSpecialNctaid;
  if (text.starts_with("%laneid"))
    return TokenKind::kSpecialLaneId;
  if (text.starts_with("%warpid"))
    return TokenKind::kSpecialWarpId;
  if (text == "%clock")
    return TokenKind::kSpecialClock;
  if (text == "%clock64")
    return TokenKind::kSpecialClock64;
  return TokenKind::kRegister;
}

}  // namespace

Lexer::Lexer(std::string_view source) noexcept : source_(source) {}

char Lexer::Current() const noexcept {
  return (offset_ < source_.size()) ? source_[offset_] : '\0';
}

char Lexer::Peek(std::size_t forward) const noexcept {
  const std::size_t target = offset_ + forward;
  return (target < source_.size()) ? source_[target] : '\0';
}

char Lexer::Advance() noexcept {
  if (offset_ >= source_.size()) {
    return '\0';
  }
  const char c = source_[offset_++];
  if (c == '\n') {
    ++line_;
    column_ = 1;
  } else {
    ++column_;
  }
  return c;
}

void Lexer::SkipLineComment() {
  while (!IsAtEnd() && Current() != '\n') {
    Advance();
  }
}

void Lexer::SkipBlockComment() {
  Advance();  // consume '*'
  while (!IsAtEnd()) {
    if (Current() == '*' && Peek(1) == '/') {
      Advance();  // consume '*'
      Advance();  // consume '/'
      return;
    }
    Advance();
  }
}

void Lexer::SkipWhitespaceAndComments() {
  while (!IsAtEnd()) {
    const char c = Current();
    if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
      Advance();
    } else if (c == '/' && Peek(1) == '/') {
      Advance();
      Advance();
      SkipLineComment();
    } else if (c == '/' && Peek(1) == '*') {
      Advance();
      SkipBlockComment();
    } else {
      break;
    }
  }
}

Token Lexer::NextToken() {
  SkipWhitespaceAndComments();

  if (IsAtEnd()) {
    return Token{
        .kind = TokenKind::kEndOfFile,
        .text = std::string_view{},
        .location = SourceLocation{.line = line_, .column = column_},
    };
  }

  const SourceLocation loc{.line = line_, .column = column_};
  const char c = Current();

  if (c == '.') {
    return LexDirectiveOrType();
  }
  if (c == '%') {
    return LexRegisterOrSpecial();
  }
  if (std::isalpha(static_cast<unsigned char>(c)) != 0 || c == '_' || c == '$') {
    return LexIdentifierOrOpcode();
  }
  if (std::isdigit(static_cast<unsigned char>(c)) != 0) {
    return LexNumber();
  }
  if (c == '"') {
    return LexString();
  }

  // Punctuation and symbols
  Advance();
  switch (c) {
    case ';':
      return Token{TokenKind::kSemicolon, ";", loc};
    case ',':
      return Token{TokenKind::kComma, ",", loc};
    case ':':
      return Token{TokenKind::kColon, ":", loc};
    case '{':
      return Token{TokenKind::kLeftBrace, "{", loc};
    case '}':
      return Token{TokenKind::kRightBrace, "}", loc};
    case '(':
      return Token{TokenKind::kLeftParen, "(", loc};
    case ')':
      return Token{TokenKind::kRightParen, ")", loc};
    case '[':
      return Token{TokenKind::kLeftBracket, "[", loc};
    case ']':
      return Token{TokenKind::kRightBracket, "]", loc};
    case '<':
      return Token{TokenKind::kLeftAngle, "<", loc};
    case '>':
      return Token{TokenKind::kRightAngle, ">", loc};
    case '=':
      return Token{TokenKind::kEquals, "=", loc};
    case '@':
      return Token{TokenKind::kAt, "@", loc};
    case '!':
      return Token{TokenKind::kExclamation, "!", loc};
    case '+':
      return Token{TokenKind::kPlus, "+", loc};
    case '-':
      return Token{TokenKind::kMinus, "-", loc};
    default:
      return Token{TokenKind::kUnknown, std::string_view(source_.data() + offset_ - 1, 1), loc};
  }
}

Token Lexer::PeekToken() {
  const std::size_t saved_offset = offset_;
  const std::uint32_t saved_line = line_;
  const std::uint32_t saved_column = column_;

  const Token tok = NextToken();

  offset_ = saved_offset;
  line_ = saved_line;
  column_ = saved_column;
  return tok;
}

std::vector<Token> Lexer::TokenizeAll() {
  std::vector<Token> tokens;
  while (true) {
    Token tok = NextToken();
    tokens.push_back(tok);
    if (tok.kind == TokenKind::kEndOfFile) {
      break;
    }
  }
  return tokens;
}

Token Lexer::LexDirectiveOrType() {
  const SourceLocation loc{.line = line_, .column = column_};
  const std::size_t start = offset_;
  Advance();  // consume '.'

  while (!IsAtEnd() && IsIdentifierChar(Current())) {
    Advance();
  }

  const std::string_view text(source_.data() + start, offset_ - start);
  const TokenKind kind = LookupDirectiveOrType(text);
  return Token{kind, text, loc};
}

Token Lexer::LexRegisterOrSpecial() {
  const SourceLocation loc{.line = line_, .column = column_};
  const std::size_t start = offset_;
  Advance();  // consume '%'

  while (!IsAtEnd() && (IsIdentifierChar(Current()) || Current() == '.')) {
    Advance();
  }

  const std::string_view text(source_.data() + start, offset_ - start);
  const TokenKind kind = LookupSpecialRegister(text);
  return Token{kind, text, loc};
}

Token Lexer::LexIdentifierOrOpcode() {
  const SourceLocation loc{.line = line_, .column = column_};
  const std::size_t start = offset_;

  while (!IsAtEnd() && (IsIdentifierChar(Current()) || Current() == '.')) {
    // If it's a dot immediately followed by modifier chars (like add.f32),
    // stop before dot so that modifiers are lexed as types/modifiers
    if (Current() == '.') {
      break;
    }
    Advance();
  }

  const std::string_view text(source_.data() + start, offset_ - start);
  const TokenKind kind = LookupOpcode(text);
  return Token{kind, text, loc};
}

Token Lexer::LexNumber() {
  const SourceLocation loc{.line = line_, .column = column_};
  const std::size_t start = offset_;

  // Check for hex float 0f... or 0d... or standard hex 0x...
  if (Current() == '0' && (Peek(1) == 'x' || Peek(1) == 'X')) {
    Advance();  // '0'
    Advance();  // 'x'
    while (!IsAtEnd() && std::isxdigit(static_cast<unsigned char>(Current())) != 0) {
      Advance();
    }
    return Token{TokenKind::kIntegerLiteral,
                 std::string_view(source_.data() + start, offset_ - start), loc};
  }

  if (Current() == '0' && (Peek(1) == 'f' || Peek(1) == 'F' || Peek(1) == 'd' || Peek(1) == 'D')) {
    Advance();  // '0'
    Advance();  // 'f' / 'd'
    while (!IsAtEnd() && std::isxdigit(static_cast<unsigned char>(Current())) != 0) {
      Advance();
    }
    return Token{TokenKind::kFloatLiteral,
                 std::string_view(source_.data() + start, offset_ - start), loc};
  }

  bool is_float = false;
  while (!IsAtEnd()) {
    const char c = Current();
    if (std::isdigit(static_cast<unsigned char>(c)) != 0) {
      Advance();
    } else if (c == '.' && std::isdigit(static_cast<unsigned char>(Peek(1))) != 0) {
      is_float = true;
      Advance();
    } else if (c == 'e' || c == 'E') {
      is_float = true;
      Advance();
      if (Current() == '+' || Current() == '-') {
        Advance();
      }
    } else if (c == 'f' && is_float) {
      Advance();
      break;
    } else {
      break;
    }
  }

  return Token{is_float ? TokenKind::kFloatLiteral : TokenKind::kIntegerLiteral,
               std::string_view(source_.data() + start, offset_ - start), loc};
}

Token Lexer::LexString() {
  const SourceLocation loc{.line = line_, .column = column_};
  const std::size_t start = offset_;
  Advance();  // consume '"'

  while (!IsAtEnd()) {
    const char c = Current();
    if (c == '\\' && !IsAtEnd()) {
      Advance();
      Advance();
    } else if (c == '"') {
      Advance();
      break;
    } else {
      Advance();
    }
  }

  return Token{TokenKind::kStringLiteral, std::string_view(source_.data() + start, offset_ - start),
               loc};
}

}  // namespace muda::compiler::ptx

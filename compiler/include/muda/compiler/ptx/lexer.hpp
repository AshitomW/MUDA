/**
 * @file lexer.hpp
 * @brief High-performance zero-copy lexer for PTX source code.
 */

#pragma once

#include <cstddef>
#include <string_view>
#include <vector>

#include "muda/compiler/ptx/tokens.hpp"

namespace muda::compiler::ptx {

class Lexer final {
 public:
  explicit Lexer(std::string_view source) noexcept;

  /**
   * @brief Returns the next token in the input stream and advances position.
   */
  [[nodiscard]] Token NextToken();

  /**
   * @brief Peeks at the next token without advancing position.
   */
  [[nodiscard]] Token PeekToken();

  /**
   * @brief Tokenizes the entire input into a vector of tokens up to kEndOfFile.
   */
  [[nodiscard]] std::vector<Token> TokenizeAll();

  /**
   * @brief Returns true if end of input has been reached.
   */
  [[nodiscard]] bool IsAtEnd() const noexcept { return offset_ >= source_.size(); }

 private:
  void SkipWhitespaceAndComments();
  void SkipLineComment();
  void SkipBlockComment();

  [[nodiscard]] char Current() const noexcept;
  [[nodiscard]] char Peek(std::size_t forward = 1) const noexcept;
  char Advance() noexcept;

  [[nodiscard]] Token LexDirectiveOrType();
  [[nodiscard]] Token LexRegisterOrSpecial();
  [[nodiscard]] Token LexIdentifierOrOpcode();
  [[nodiscard]] Token LexNumber();
  [[nodiscard]] Token LexString();

  std::string_view source_;
  std::size_t offset_ = 0;
  std::uint32_t line_ = 1;
  std::uint32_t column_ = 1;
};

}  // namespace muda::compiler::ptx

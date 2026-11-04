/**
 * @file ptx_lexer_test.cpp
 * @brief Unit tests for PTX zero-copy lexer and token classification.
 */

#include <iostream>
#include <string_view>
#include <vector>

#include "muda/compiler/ptx/lexer.hpp"
#include "muda/compiler/ptx/tokens.hpp"

namespace {

bool Expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    return false;
  }
  return true;
}

bool TestDirectivesAndTypes() {
  constexpr std::string_view kSource =
      ".version 6.5\n"
      ".target sm_70\n"
      ".address_size 64\n"
      ".visible .entry my_kernel(\n"
      "  .param .u64 in_ptr,\n"
      "  .param .f32 in_val\n"
      ");\n";

  muda::compiler::ptx::Lexer lexer(kSource);
  const std::vector<muda::compiler::ptx::Token> tokens = lexer.TokenizeAll();

  if (!Expect(!tokens.empty(), "Token list should not be empty"))
    return false;
  if (!Expect(tokens[0].Is(muda::compiler::ptx::TokenKind::kDirectiveVersion),
              "First token should be .version"))
    return false;
  if (!Expect(tokens[1].Is(muda::compiler::ptx::TokenKind::kFloatLiteral),
              "Second token should be float literal 6.5"))
    return false;
  if (!Expect(tokens[2].Is(muda::compiler::ptx::TokenKind::kDirectiveTarget),
              "Third token should be .target"))
    return false;
  if (!Expect(
          tokens[3].Is(muda::compiler::ptx::TokenKind::kIdentifier) && tokens[3].text == "sm_70",
          "Fourth token should be identifier sm_70"))
    return false;
  if (!Expect(tokens[4].Is(muda::compiler::ptx::TokenKind::kDirectiveAddressSize),
              "Fifth token should be .address_size"))
    return false;
  if (!Expect(
          tokens[5].Is(muda::compiler::ptx::TokenKind::kIntegerLiteral) && tokens[5].text == "64",
          "Sixth token should be 64"))
    return false;

  return true;
}

bool TestRegistersAndInstructions() {
  constexpr std::string_view kSource =
      "// Compute thread index\n"
      "mov.u32 %r1, %tid.x;\n"
      "ld.param.u64 %rd1, [in_ptr];\n"
      "add.f32 %f3, %f1, %f2;\n"
      "@%p1 bra $L__BB0_1;\n"
      "ret;\n";

  muda::compiler::ptx::Lexer lexer(kSource);
  const std::vector<muda::compiler::ptx::Token> tokens = lexer.TokenizeAll();

  if (!Expect(!tokens.empty(), "Tokens should not be empty"))
    return false;

  // Verify mov.u32 %r1, %tid.x;
  if (!Expect(tokens[0].Is(muda::compiler::ptx::TokenKind::kOpMov), "Should start with mov"))
    return false;
  if (!Expect(tokens[1].Is(muda::compiler::ptx::TokenKind::kTypeU32), "Should have .u32 modifier"))
    return false;
  if (!Expect(tokens[2].Is(muda::compiler::ptx::TokenKind::kRegister) && tokens[2].text == "%r1",
              "Should have %r1 register"))
    return false;
  if (!Expect(tokens[3].Is(muda::compiler::ptx::TokenKind::kComma), "Should have comma"))
    return false;
  if (!Expect(
          tokens[4].Is(muda::compiler::ptx::TokenKind::kSpecialTid) && tokens[4].text == "%tid.x",
          "Should have %tid.x special register"))
    return false;
  if (!Expect(tokens[5].Is(muda::compiler::ptx::TokenKind::kSemicolon), "Should have semicolon"))
    return false;

  // Verify @%p1 bra $L__BB0_1;
  // Tokens:
  // 6: ld, 7: .param, 8: .u64, 9: %rd1, 10: comma, 11: [, 12: in_ptr, 13: ], 14: ;
  // 15: add, 16: .f32, 17: %f3, 18: comma, 19: %f1, 20: comma, 21: %f2, 22: ;
  // 23: @, 24: %p1, 25: bra, 26: $L__BB0_1, 27: ;
  if (!Expect(tokens[23].Is(muda::compiler::ptx::TokenKind::kAt), "Should have @"))
    return false;
  if (!Expect(tokens[24].Is(muda::compiler::ptx::TokenKind::kRegister) && tokens[24].text == "%p1",
              "Should have %p1"))
    return false;
  if (!Expect(tokens[25].Is(muda::compiler::ptx::TokenKind::kOpBra), "Should have bra"))
    return false;
  if (!Expect(tokens[26].Is(muda::compiler::ptx::TokenKind::kIdentifier) &&
                  tokens[26].text == "$L__BB0_1",
              "Should have $L__BB0_1 label target"))
    return false;

  return true;
}

bool TestHexAndFloatLiterals() {
  constexpr std::string_view kSource = "0x1f 0x2A 0f3f800000 0d400921fb54442d18 3.14159 100\n";

  muda::compiler::ptx::Lexer lexer(kSource);
  const std::vector<muda::compiler::ptx::Token> tokens = lexer.TokenizeAll();

  if (!Expect(
          tokens[0].Is(muda::compiler::ptx::TokenKind::kIntegerLiteral) && tokens[0].text == "0x1f",
          "0x1f should be integer literal"))
    return false;
  if (!Expect(
          tokens[1].Is(muda::compiler::ptx::TokenKind::kIntegerLiteral) && tokens[1].text == "0x2A",
          "0x2A should be integer literal"))
    return false;
  if (!Expect(tokens[2].Is(muda::compiler::ptx::TokenKind::kFloatLiteral) &&
                  tokens[2].text == "0f3f800000",
              "0f3f800000 should be float literal"))
    return false;
  if (!Expect(tokens[3].Is(muda::compiler::ptx::TokenKind::kFloatLiteral) &&
                  tokens[3].text == "0d400921fb54442d18",
              "0d... should be float literal"))
    return false;
  if (!Expect(tokens[4].Is(muda::compiler::ptx::TokenKind::kFloatLiteral) &&
                  tokens[4].text == "3.14159",
              "3.14159 should be float literal"))
    return false;
  if (!Expect(
          tokens[5].Is(muda::compiler::ptx::TokenKind::kIntegerLiteral) && tokens[5].text == "100",
          "100 should be integer literal"))
    return false;

  return true;
}

}  // namespace

int main() {
  if (!TestDirectivesAndTypes())
    return 1;
  if (!TestRegistersAndInstructions())
    return 1;
  if (!TestHexAndFloatLiterals())
    return 1;

  std::cout << "ptx_lexer_test: all tests passed!\n";
  return 0;
}

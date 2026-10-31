/**
 * @file ast_extractor.hpp
 * @brief Lexical scanning and AST function slice extraction for CUDA sources.
 */

#pragma once

#include <cctype>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "muda/compiler/mmir.hpp"

namespace muda::compiler {

[[nodiscard]] inline bool IsSpace(char value) noexcept {
  return std::isspace(static_cast<unsigned char>(value)) != 0;
}

[[nodiscard]] inline bool IsIdentifierStart(char value) noexcept {
  return std::isalpha(static_cast<unsigned char>(value)) != 0 || value == '_';
}

[[nodiscard]] inline bool IsIdentifierChar(char value) noexcept {
  return std::isalnum(static_cast<unsigned char>(value)) != 0 || value == '_';
}

[[nodiscard]] std::string_view Trim(std::string_view text) noexcept;
[[nodiscard]] std::string NormalizeSpace(std::string_view text);
[[nodiscard]] std::size_t FindMatching(std::string_view text, std::size_t open, char open_char,
                                       char close_char) noexcept;
[[nodiscard]] std::vector<std::string_view> SplitParameters(std::string_view text);
void RemoveIdentifier(std::string& text, std::string_view identifier);

struct ParsedParameter {
  DiagnosticCode code = DiagnosticCode::kOk;
  KernelParameter parameter;
};

[[nodiscard]] ParsedParameter ParseParameter(std::string_view text, std::size_t index);
[[nodiscard]] DiagnosticCode ParseParameterList(std::string_view parameter_list,
                                                std::vector<KernelParameter>* out_params);

enum class QualifierKind : std::uint8_t {
  kGlobal,
  kDevice,
};

struct RawFunctionSlice {
  QualifierKind kind = QualifierKind::kGlobal;
  std::string return_type;
  std::string name;
  std::string_view parameter_list;
  std::string_view body;
};

[[nodiscard]] DiagnosticCode ExtractFunctionsFromSource(
    std::string_view source, std::vector<RawFunctionSlice>* out_functions);

[[nodiscard]] DiagnosticCode ExtractSharedMemoryAndLowerBody(
    std::string_view raw_body, std::vector<mmir::SharedAllocation>* out_shared,
    std::string* out_lowered_body);

}  // namespace muda::compiler

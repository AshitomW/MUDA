/**
 * @file translator.cpp
 * @brief Implements CUDA-to-MSL translation.
 */

#include "muda/compiler/translator.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace muda::compiler {
namespace {

struct TypeMappingEntry {
  std::string_view cuda_spelling;
  DataType type;
};

constexpr std::array<TypeMappingEntry, 33> kTypeTable = {{
    {"bool", {ScalarType::kBool, VectorWidth::k1}},
    {"char", {ScalarType::kInt8, VectorWidth::k1}},
    {"signed char", {ScalarType::kInt8, VectorWidth::k1}},
    {"uchar", {ScalarType::kUInt8, VectorWidth::k1}},
    {"unsigned char", {ScalarType::kUInt8, VectorWidth::k1}},
    {"short", {ScalarType::kInt16, VectorWidth::k1}},
    {"ushort", {ScalarType::kUInt16, VectorWidth::k1}},
    {"unsigned short", {ScalarType::kUInt16, VectorWidth::k1}},
    {"int", {ScalarType::kInt32, VectorWidth::k1}},
    {"int2", {ScalarType::kInt32, VectorWidth::k2}},
    {"int3", {ScalarType::kInt32, VectorWidth::k3}},
    {"int4", {ScalarType::kInt32, VectorWidth::k4}},
    {"uint", {ScalarType::kUInt32, VectorWidth::k1}},
    {"unsigned", {ScalarType::kUInt32, VectorWidth::k1}},
    {"unsigned int", {ScalarType::kUInt32, VectorWidth::k1}},
    {"uint2", {ScalarType::kUInt32, VectorWidth::k2}},
    {"uint3", {ScalarType::kUInt32, VectorWidth::k3}},
    {"uint4", {ScalarType::kUInt32, VectorWidth::k4}},
    {"long", {ScalarType::kInt64, VectorWidth::k1}},
    {"long long", {ScalarType::kInt64, VectorWidth::k1}},
    {"long2", {ScalarType::kInt64, VectorWidth::k2}},
    {"long3", {ScalarType::kInt64, VectorWidth::k3}},
    {"long4", {ScalarType::kInt64, VectorWidth::k4}},
    {"ulong", {ScalarType::kUInt64, VectorWidth::k1}},
    {"unsigned long", {ScalarType::kUInt64, VectorWidth::k1}},
    {"unsigned long long", {ScalarType::kUInt64, VectorWidth::k1}},
    {"half", {ScalarType::kFloat16, VectorWidth::k1}},
    {"half2", {ScalarType::kFloat16, VectorWidth::k2}},
    {"half3", {ScalarType::kFloat16, VectorWidth::k3}},
    {"half4", {ScalarType::kFloat16, VectorWidth::k4}},
    {"float", {ScalarType::kFloat32, VectorWidth::k1}},
    {"float2", {ScalarType::kFloat32, VectorWidth::k2}},
    {"float3", {ScalarType::kFloat32, VectorWidth::k3}},
}};

struct BuiltinMappingEntry {
  std::string_view cuda_name;
  std::string_view msl_name;
};

constexpr std::array<BuiltinMappingEntry, 4> kBuiltinTable = {{
    {"blockIdx", "threadgroup_position_in_grid"},
    {"blockDim", "threads_per_threadgroup"},
    {"threadIdx", "thread_position_in_threadgroup"},
    {"gridDim", "threadgroups_per_grid"},
}};

constexpr std::array<std::string_view, 20> kUnsupportedConstructs = {{
    "__shared__", "__device__",  "__constant__",   "__managed__",      "cooperative_groups",
    "atomicAdd",  "atomicSub",   "atomicCAS",      "atomicExch",       "atomicMin",
    "atomicMax",  "atomicInc",   "atomicDec",      "atomicAnd",        "atomicOr",
    "atomicXor",  "__shfl_sync", "__shfl_up_sync", "__shfl_down_sync", "printf",
}};

bool IsIdentifierStart(char value) noexcept {
  return std::isalpha(static_cast<unsigned char>(value)) != 0 || value == '_';
}

bool IsIdentifierChar(char value) noexcept {
  return std::isalnum(static_cast<unsigned char>(value)) != 0 || value == '_';
}

bool IsSpace(char value) noexcept { return std::isspace(static_cast<unsigned char>(value)) != 0; }

std::string_view Trim(std::string_view text) noexcept {
  while (!text.empty() && IsSpace(text.front())) {
    text.remove_prefix(1);
  }
  while (!text.empty() && IsSpace(text.back())) {
    text.remove_suffix(1);
  }
  return text;
}

std::string NormalizeSpace(std::string_view text) {
  std::string result;
  result.reserve(text.size());
  bool pending_space = false;
  for (const char value : text) {
    if (IsSpace(value)) {
      pending_space = !result.empty();
      continue;
    }
    if (pending_space) {
      result.push_back(' ');
      pending_space = false;
    }
    result.push_back(value);
  }
  return result;
}

std::size_t FindMatching(std::string_view text, std::size_t open, char open_char,
                         char close_char) noexcept {
  int depth = 0;
  char quote = '\0';
  bool escape = false;
  bool line_comment = false;
  bool block_comment = false;

  for (std::size_t index = open; index < text.size(); ++index) {
    const char value = text[index];
    if (line_comment) {
      if (value == '\n') {
        line_comment = false;
      }
      continue;
    }
    if (block_comment) {
      if (value == '*' && index + 1 < text.size() && text[index + 1] == '/') {
        block_comment = false;
        ++index;
      }
      continue;
    }
    if (quote != '\0') {
      if (escape) {
        escape = false;
        continue;
      }
      if (value == '\\') {
        escape = true;
        continue;
      }
      if (value == quote) {
        quote = '\0';
      }
      continue;
    }
    if (value == '/' && index + 1 < text.size() && text[index + 1] == '/') {
      line_comment = true;
      ++index;
      continue;
    }
    if (value == '/' && index + 1 < text.size() && text[index + 1] == '*') {
      block_comment = true;
      ++index;
      continue;
    }
    if (value == '"' || value == '\'') {
      quote = value;
      continue;
    }
    if (value == open_char) {
      ++depth;
    } else if (value == close_char) {
      --depth;
      if (depth == 0) {
        return index;
      }
    }
  }

  return std::string_view::npos;
}

std::vector<std::string_view> SplitParameters(std::string_view text) {
  std::vector<std::string_view> result;
  const std::string_view trimmed = Trim(text);
  if (trimmed.empty()) {
    return result;
  }

  std::size_t start = 0;
  int depth = 0;
  char quote = '\0';
  bool escape = false;

  for (std::size_t index = 0; index < text.size(); ++index) {
    const char value = text[index];
    if (quote != '\0') {
      if (escape) {
        escape = false;
        continue;
      }
      if (value == '\\') {
        escape = true;
        continue;
      }
      if (value == quote) {
        quote = '\0';
      }
      continue;
    }
    if (value == '"' || value == '\'') {
      quote = value;
      continue;
    }
    if (value == '(' || value == '[' || value == '{' || value == '<') {
      ++depth;
    } else if (value == ')' || value == ']' || value == '}' || value == '>') {
      --depth;
    } else if (value == ',' && depth == 0) {
      result.push_back(Trim(text.substr(start, index - start)));
      start = index + 1;
    }
  }

  result.push_back(Trim(text.substr(start)));
  return result;
}

void RemoveIdentifier(std::string& text, std::string_view identifier) {
  std::string result;
  result.reserve(text.size());
  std::size_t index = 0;

  while (index < text.size()) {
    if (IsIdentifierStart(text[index])) {
      const std::size_t start = index;
      ++index;
      while (index < text.size() && IsIdentifierChar(text[index])) {
        ++index;
      }
      const std::string_view word(text.data() + start, index - start);
      if (word == identifier) {
        result.push_back(' ');
      } else {
        result.append(word);
      }
      continue;
    }
    result.push_back(text[index]);
    ++index;
  }

  text = NormalizeSpace(result);
}

std::optional<DataType> ResolveCudaType(std::string_view spelling) noexcept {
  if (spelling == "float4") {
    return DataType{ScalarType::kFloat32, VectorWidth::k4};
  }
  for (const TypeMappingEntry& entry : kTypeTable) {
    if (entry.cuda_spelling == spelling) {
      return entry.type;
    }
  }
  return std::nullopt;
}

struct ParsedParameter {
  DiagnosticCode code = DiagnosticCode::kOk;
  KernelParameter parameter;
};

ParsedParameter ParseParameter(std::string_view text, std::size_t index) {
  const std::string normalized = NormalizeSpace(text);
  if (normalized.empty()) {
    return {DiagnosticCode::kEmptyKernelParameter, {}};
  }

  std::size_t declaration_end = normalized.size();
  for (std::size_t position = 0; position < normalized.size(); ++position) {
    if (normalized[position] == '[' || normalized[position] == '=') {
      declaration_end = position;
      break;
    }
  }

  const std::string declaration =
      NormalizeSpace(std::string_view(normalized).substr(0, declaration_end));
  std::size_t name_end = declaration.size();
  while (name_end > 0 && !IsIdentifierChar(declaration[name_end - 1])) {
    --name_end;
  }
  std::size_t name_start = name_end;
  while (name_start > 0 && IsIdentifierChar(declaration[name_start - 1])) {
    --name_start;
  }
  if (name_start == name_end) {
    return {DiagnosticCode::kMissingParameterName, {}};
  }

  KernelParameter parameter;
  parameter.name = declaration.substr(name_start, name_end - name_start);
  parameter.buffer_index = index;

  std::string type_spelling = declaration.substr(0, name_start);
  const bool is_pointer = type_spelling.find('*') != std::string::npos;
  const bool is_const = type_spelling.find("const") != std::string::npos;

  parameter.binding_kind =
      is_pointer ? ParameterBindingKind::kBufferPointer : ParameterBindingKind::kConstantValue;
  parameter.address_space = is_pointer ? AddressSpace::kDevice : AddressSpace::kConstant;
  parameter.access =
      (!is_pointer || is_const) ? AccessQualifier::kReadOnly : AccessQualifier::kReadWrite;

  type_spelling.erase(std::remove(type_spelling.begin(), type_spelling.end(), '*'),
                      type_spelling.end());
  RemoveIdentifier(type_spelling, "const");
  RemoveIdentifier(type_spelling, "volatile");
  RemoveIdentifier(type_spelling, "__restrict__");
  RemoveIdentifier(type_spelling, "__restrict");
  RemoveIdentifier(type_spelling, "restrict");

  const std::optional<DataType> resolved_type = ResolveCudaType(NormalizeSpace(type_spelling));
  if (!resolved_type.has_value()) {
    return {DiagnosticCode::kUnsupportedParameterType, {}};
  }
  parameter.type = *resolved_type;

  return {DiagnosticCode::kOk, std::move(parameter)};
}

std::string_view BuiltinReplacement(std::string_view word) noexcept {
  for (const BuiltinMappingEntry& entry : kBuiltinTable) {
    if (entry.cuda_name == word) {
      return entry.msl_name;
    }
  }
  return {};
}

std::string TranslateBuiltins(std::string_view body) {
  std::string result;
  result.reserve(body.size());

  for (std::size_t index = 0; index < body.size();) {
    if (!IsIdentifierStart(body[index])) {
      result.push_back(body[index]);
      ++index;
      continue;
    }

    const std::size_t start = index;
    ++index;
    while (index < body.size() && IsIdentifierChar(body[index])) {
      ++index;
    }
    const std::string_view word = body.substr(start, index - start);
    const std::string_view replacement = BuiltinReplacement(word);

    std::size_t next = index;
    while (next < body.size() && IsSpace(body[next])) {
      ++next;
    }

    if (!replacement.empty() && next < body.size() && body[next] == '.') {
      std::size_t axis = next + 1;
      while (axis < body.size() && IsSpace(body[axis])) {
        ++axis;
      }
      if (axis < body.size() && (body[axis] == 'x' || body[axis] == 'y' || body[axis] == 'z')) {
        result.append(replacement);
        result.push_back('.');
        result.push_back(body[axis]);
        index = axis + 1;
        continue;
      }
    }

    if (word == "__syncthreads" && next < body.size() && body[next] == '(') {
      const std::size_t close = FindMatching(body, next, '(', ')');
      if (close != std::string_view::npos) {
        result.append("threadgroup_barrier(mem_flags::mem_threadgroup)");
        index = close + 1;
        continue;
      }
    }

    result.append(word);
  }

  return result;
}

TranslationResult Failure(DiagnosticCode code) noexcept {
  TranslationResult result;
  result.code = code;
  return result;
}

}  // namespace

TranslationResult TranslateCudaToMsl(std::string_view source) {
  constexpr std::string_view kMarker = "__global__";
  const std::size_t marker_position = source.find(kMarker);
  if (marker_position == std::string_view::npos) {
    return Failure(DiagnosticCode::kMissingGlobalKernel);
  }
  if (source.find(kMarker, marker_position + kMarker.size()) != std::string_view::npos) {
    return Failure(DiagnosticCode::kMultipleKernelsUnsupported);
  }

  const std::size_t open = source.find('(', marker_position + kMarker.size());
  const std::size_t semicolon = source.find(';', marker_position + kMarker.size());
  if (open == std::string_view::npos || (semicolon != std::string_view::npos && semicolon < open)) {
    return Failure(DiagnosticCode::kMalformedKernelDeclaration);
  }

  const std::size_t close = FindMatching(source, open, '(', ')');
  if (close == std::string_view::npos) {
    return Failure(DiagnosticCode::kUnterminatedParameterList);
  }

  std::size_t name_end = open;
  while (name_end > 0 && IsSpace(source[name_end - 1])) {
    --name_end;
  }
  std::size_t name_start = name_end;
  while (name_start > 0 && IsIdentifierChar(source[name_start - 1])) {
    --name_start;
  }
  if (name_start == name_end) {
    return Failure(DiagnosticCode::kMissingKernelName);
  }

  const std::string name(source.substr(name_start, name_end - name_start));
  const std::string return_type = NormalizeSpace(source.substr(
      marker_position + kMarker.size(), name_start - (marker_position + kMarker.size())));
  if (return_type != "void") {
    return Failure(DiagnosticCode::kNonVoidKernelReturn);
  }

  std::size_t body_open = close + 1;
  while (body_open < source.size() && IsSpace(source[body_open])) {
    ++body_open;
  }
  if (body_open >= source.size() || source[body_open] != '{') {
    return Failure(DiagnosticCode::kMissingKernelBody);
  }

  const std::size_t body_close = FindMatching(source, body_open, '{', '}');
  if (body_close == std::string_view::npos) {
    return Failure(DiagnosticCode::kUnterminatedKernelBody);
  }

  const std::string_view body = source.substr(body_open + 1, body_close - body_open - 1);
  for (const std::string_view token : kUnsupportedConstructs) {
    if (body.find(token) != std::string_view::npos) {
      return Failure(DiagnosticCode::kUnsupportedConstruct);
    }
  }

  TranslatedKernel kernel;
  kernel.name = name;

  const std::vector<std::string_view> parameters =
      SplitParameters(source.substr(open + 1, close - open - 1));
  for (std::size_t index = 0; index < parameters.size(); ++index) {
    ParsedParameter parsed = ParseParameter(parameters[index], index);
    if (parsed.code != DiagnosticCode::kOk) {
      return Failure(parsed.code);
    }
    kernel.parameters.push_back(std::move(parsed.parameter));
  }

  std::string msl;
  msl.reserve(body.size() + 512);
  msl += "#include <metal_stdlib>\n";
  msl += "using namespace metal;\n\n";
  msl += "kernel void ";
  msl += kernel.name;
  msl += "(\n";

  for (const KernelParameter& parameter : kernel.parameters) {
    msl += "    ";
    msl += AddressSpaceName(parameter.address_space);
    msl.push_back(' ');
    if (parameter.is_pointer()) {
      if (parameter.is_const()) {
        msl += "const ";
      }
      msl += parameter.metal_type();
      msl += "* ";
    } else {
      msl += parameter.metal_type();
      msl += "& ";
    }
    msl += parameter.name;
    msl += " [[buffer(";
    msl += std::to_string(parameter.buffer_index);
    msl += ")]],\n";
  }

  msl += "    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],\n";
  msl += "    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],\n";
  msl += "    uint3 threads_per_threadgroup [[threads_per_threadgroup]],\n";
  msl += "    uint3 threadgroups_per_grid [[threadgroups_per_grid]])\n";
  msl += "{\n";
  msl += "  ";
  msl += TranslateBuiltins(Trim(body));
  msl += "\n}\n";

  kernel.msl = std::move(msl);

  TranslationResult result;
  result.code = DiagnosticCode::kOk;
  result.kernel = std::move(kernel);
  return result;
}

}  // namespace muda::compiler

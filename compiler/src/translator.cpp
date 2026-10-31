/**
 * @file translator.cpp
 * @brief Implements CUDA-to-MMIR extraction and MSL translation.
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

#include "muda/compiler/mmir.hpp"

namespace muda::compiler {
namespace {

struct TypeMappingEntry {
  std::string_view cuda_spelling;
  DataType type;
};

constexpr std::array<TypeMappingEntry, 45> kTypeTable = {{
    {"bool", {ScalarType::kBool, VectorWidth::k1}},
    {"char", {ScalarType::kInt8, VectorWidth::k1}},
    {"signed char", {ScalarType::kInt8, VectorWidth::k1}},
    {"int8_t", {ScalarType::kInt8, VectorWidth::k1}},
    {"uchar", {ScalarType::kUInt8, VectorWidth::k1}},
    {"unsigned char", {ScalarType::kUInt8, VectorWidth::k1}},
    {"uint8_t", {ScalarType::kUInt8, VectorWidth::k1}},
    {"short", {ScalarType::kInt16, VectorWidth::k1}},
    {"int16_t", {ScalarType::kInt16, VectorWidth::k1}},
    {"ushort", {ScalarType::kUInt16, VectorWidth::k1}},
    {"unsigned short", {ScalarType::kUInt16, VectorWidth::k1}},
    {"uint16_t", {ScalarType::kUInt16, VectorWidth::k1}},
    {"int", {ScalarType::kInt32, VectorWidth::k1}},
    {"int32_t", {ScalarType::kInt32, VectorWidth::k1}},
    {"int2", {ScalarType::kInt32, VectorWidth::k2}},
    {"int3", {ScalarType::kInt32, VectorWidth::k3}},
    {"int4", {ScalarType::kInt32, VectorWidth::k4}},
    {"uint", {ScalarType::kUInt32, VectorWidth::k1}},
    {"unsigned", {ScalarType::kUInt32, VectorWidth::k1}},
    {"unsigned int", {ScalarType::kUInt32, VectorWidth::k1}},
    {"uint32_t", {ScalarType::kUInt32, VectorWidth::k1}},
    {"uint2", {ScalarType::kUInt32, VectorWidth::k2}},
    {"uint3", {ScalarType::kUInt32, VectorWidth::k3}},
    {"uint4", {ScalarType::kUInt32, VectorWidth::k4}},
    {"long", {ScalarType::kInt64, VectorWidth::k1}},
    {"long long", {ScalarType::kInt64, VectorWidth::k1}},
    {"int64_t", {ScalarType::kInt64, VectorWidth::k1}},
    {"ptrdiff_t", {ScalarType::kInt64, VectorWidth::k1}},
    {"long2", {ScalarType::kInt64, VectorWidth::k2}},
    {"long3", {ScalarType::kInt64, VectorWidth::k3}},
    {"long4", {ScalarType::kInt64, VectorWidth::k4}},
    {"ulong", {ScalarType::kUInt64, VectorWidth::k1}},
    {"unsigned long", {ScalarType::kUInt64, VectorWidth::k1}},
    {"unsigned long long", {ScalarType::kUInt64, VectorWidth::k1}},
    {"uint64_t", {ScalarType::kUInt64, VectorWidth::k1}},
    {"size_t", {ScalarType::kUInt64, VectorWidth::k1}},
    {"ulong2", {ScalarType::kUInt64, VectorWidth::k2}},
    {"ulong3", {ScalarType::kUInt64, VectorWidth::k3}},
    {"ulong4", {ScalarType::kUInt64, VectorWidth::k4}},
    {"half", {ScalarType::kFloat16, VectorWidth::k1}},
    {"__half", {ScalarType::kFloat16, VectorWidth::k1}},
    {"half2", {ScalarType::kFloat16, VectorWidth::k2}},
    {"__half2", {ScalarType::kFloat16, VectorWidth::k2}},
    {"half3", {ScalarType::kFloat16, VectorWidth::k3}},
    {"half4", {ScalarType::kFloat16, VectorWidth::k4}},
}};

constexpr std::array<TypeMappingEntry, 4> kFloatTypeTable = {{
    {"float", {ScalarType::kFloat32, VectorWidth::k1}},
    {"float2", {ScalarType::kFloat32, VectorWidth::k2}},
    {"float3", {ScalarType::kFloat32, VectorWidth::k3}},
    {"float4", {ScalarType::kFloat32, VectorWidth::k4}},
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

constexpr std::array<BuiltinMappingEntry, 18> kMathIntrinsicTable = {{
    {"fmaf", "fma"},
    {"__fmaf_rn", "fma"},
    {"sqrtf", "sqrt"},
    {"__fsqrt_rn", "sqrt"},
    {"rsqrtf", "rsqrt"},
    {"fabsf", "abs"},
    {"fminf", "min"},
    {"fmaxf", "max"},
    {"expf", "exp"},
    {"__expf", "fast::exp"},
    {"logf", "log"},
    {"__logf", "fast::log"},
    {"sinf", "sin"},
    {"__sinf", "fast::sin"},
    {"cosf", "cos"},
    {"__cosf", "fast::cos"},
    {"tanf", "tan"},
    {"powf", "pow"},
}};

constexpr std::array<std::string_view, 18> kUnsupportedConstructs = {{
    "__constant__",
    "__managed__",
    "cooperative_groups",
    "atomicAdd",
    "atomicSub",
    "atomicCAS",
    "atomicExch",
    "atomicMin",
    "atomicMax",
    "atomicInc",
    "atomicDec",
    "atomicAnd",
    "atomicOr",
    "atomicXor",
    "__shfl_sync",
    "__shfl_up_sync",
    "__shfl_down_sync",
    "printf",
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
  if (trimmed.empty() || trimmed == "void") {
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
  for (const TypeMappingEntry& entry : kTypeTable) {
    if (entry.cuda_spelling == spelling) {
      return entry.type;
    }
  }
  for (const TypeMappingEntry& entry : kFloatTypeTable) {
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
  if (normalized.find("(*") != std::string::npos) {
    return {DiagnosticCode::kFunctionPointerUnsupported, {}};
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

std::string_view MathIntrinsicReplacement(std::string_view word) noexcept {
  for (const BuiltinMappingEntry& entry : kMathIntrinsicTable) {
    if (entry.cuda_name == word) {
      return entry.msl_name;
    }
  }
  return {};
}

DiagnosticCode ExtractSharedMemoryAndLowerBody(std::string_view raw_body,
                                               std::vector<mmir::SharedAllocation>* out_shared,
                                               std::string* out_lowered_body) {
  std::string body_without_shared;
  body_without_shared.reserve(raw_body.size());

  std::size_t cursor = 0;
  while (cursor < raw_body.size()) {
    // Check for extern __shared__ or __shared__ declaration
    if (IsIdentifierStart(raw_body[cursor])) {
      const std::size_t word_start = cursor;
      std::size_t word_end = cursor + 1;
      while (word_end < raw_body.size() && IsIdentifierChar(raw_body[word_end])) {
        ++word_end;
      }
      const std::string_view first_word = raw_body.substr(word_start, word_end - word_start);

      bool is_extern_shared = false;
      std::size_t after_shared = word_end;
      if (first_word == "extern") {
        std::size_t next = word_end;
        while (next < raw_body.size() && IsSpace(raw_body[next])) {
          ++next;
        }
        std::size_t next_end = next;
        while (next_end < raw_body.size() && IsIdentifierChar(raw_body[next_end])) {
          ++next_end;
        }
        if (raw_body.substr(next, next_end - next) == "__shared__") {
          is_extern_shared = true;
          after_shared = next_end;
        }
      }

      if (first_word == "__shared__" || is_extern_shared) {
        if (out_shared == nullptr) {
          return DiagnosticCode::kUnsupportedConstruct;
        }
        const std::size_t semi = raw_body.find(';', after_shared);
        if (semi == std::string_view::npos) {
          return DiagnosticCode::kInvalidSharedMemoryDeclaration;
        }
        const std::string_view decl_text = Trim(raw_body.substr(after_shared, semi - after_shared));
        const std::size_t bracket_open = decl_text.find('[');
        const std::size_t bracket_close = decl_text.rfind(']');
        if (bracket_open == std::string_view::npos || bracket_close == std::string_view::npos ||
            bracket_close < bracket_open) {
          return DiagnosticCode::kInvalidSharedMemoryDeclaration;
        }

        const std::string_view count_expr =
            Trim(decl_text.substr(bracket_open + 1, bracket_close - bracket_open - 1));
        if (is_extern_shared && !count_expr.empty()) {
          return DiagnosticCode::kInvalidSharedMemoryDeclaration;
        }
        if (!is_extern_shared && count_expr.empty()) {
          return DiagnosticCode::kInvalidSharedMemoryDeclaration;
        }

        const std::string_view type_and_name = Trim(decl_text.substr(0, bracket_open));
        std::size_t name_end = type_and_name.size();
        while (name_end > 0 && IsSpace(type_and_name[name_end - 1])) {
          --name_end;
        }
        std::size_t name_start = name_end;
        while (name_start > 0 && IsIdentifierChar(type_and_name[name_start - 1])) {
          --name_start;
        }
        if (name_start == name_end) {
          return DiagnosticCode::kInvalidSharedMemoryDeclaration;
        }

        const std::string type_str = NormalizeSpace(type_and_name.substr(0, name_start));
        const std::optional<DataType> elem_type = ResolveCudaType(type_str);
        if (!elem_type.has_value()) {
          return DiagnosticCode::kUnsupportedParameterType;
        }

        out_shared->push_back(mmir::SharedAllocation{
            .name = std::string(type_and_name.substr(name_start, name_end - name_start)),
            .element_type = *elem_type,
            .count_expression = std::string(count_expr),
            .is_dynamic = is_extern_shared,
        });

        cursor = semi + 1;
        continue;
      }
    }

    body_without_shared.push_back(raw_body[cursor]);
    ++cursor;
  }

  std::string result;
  const std::string_view body = Trim(body_without_shared);
  result.reserve(body.size() + 64);

  for (std::size_t index = 0; index < body.size();) {
    if (body[index] == '<' && index + 2 < body.size() && body[index + 1] == '<' &&
        body[index + 2] == '<') {
      return DiagnosticCode::kDynamicParallelismUnsupported;
    }

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

    for (const std::string_view unsupported : kUnsupportedConstructs) {
      if (word == unsupported) {
        return DiagnosticCode::kUnsupportedConstruct;
      }
    }
    if (word == "double") {
      return DiagnosticCode::kUnsupportedConstruct;
    }

    const std::string_view builtin_rep = BuiltinReplacement(word);
    std::size_t next = index;
    while (next < body.size() && IsSpace(body[next])) {
      ++next;
    }

    if (!builtin_rep.empty() && next < body.size() && body[next] == '.') {
      std::size_t axis = next + 1;
      while (axis < body.size() && IsSpace(body[axis])) {
        ++axis;
      }
      if (axis < body.size() && (body[axis] == 'x' || body[axis] == 'y' || body[axis] == 'z')) {
        result.append(builtin_rep);
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

    if (word == "__syncwarp" && next < body.size() && body[next] == '(') {
      const std::size_t close = FindMatching(body, next, '(', ')');
      if (close != std::string_view::npos) {
        result.append("simdgroup_barrier(mem_flags::mem_threadgroup)");
        index = close + 1;
        continue;
      }
    }

    const std::string_view math_rep = MathIntrinsicReplacement(word);
    if (!math_rep.empty() && next < body.size() && body[next] == '(') {
      result.append(math_rep);
      continue;
    }

    result.append(word);
  }

  *out_lowered_body = std::move(result);
  return DiagnosticCode::kOk;
}

DiagnosticCode ParseParameterList(std::string_view parameter_list,
                                  std::vector<KernelParameter>* out_parameters) {
  const std::vector<std::string_view> parameters = SplitParameters(parameter_list);
  out_parameters->reserve(parameters.size());
  for (std::size_t index = 0; index < parameters.size(); ++index) {
    ParsedParameter parsed = ParseParameter(parameters[index], index);
    if (parsed.code != DiagnosticCode::kOk) {
      return parsed.code;
    }
    out_parameters->push_back(std::move(parsed.parameter));
  }
  return DiagnosticCode::kOk;
}

TranslationResult Failure(DiagnosticCode code) noexcept {
  TranslationResult result;
  result.code = code;
  return result;
}

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

DiagnosticCode ExtractFunctionsFromSource(std::string_view source,
                                          std::vector<RawFunctionSlice>* out_slices) {
  std::size_t cursor = 0;
  while (cursor < source.size()) {
    if (source[cursor] == '/' && cursor + 1 < source.size() && source[cursor + 1] == '/') {
      const std::size_t nl = source.find('\n', cursor + 2);
      if (nl == std::string_view::npos) {
        break;
      }
      cursor = nl + 1;
      continue;
    }
    if (source[cursor] == '/' && cursor + 1 < source.size() && source[cursor + 1] == '*') {
      const std::size_t end_c = source.find("*/", cursor + 2);
      if (end_c == std::string_view::npos) {
        break;
      }
      cursor = end_c + 2;
      continue;
    }
    if (source[cursor] == '"' || source[cursor] == '\'') {
      const char quote = source[cursor++];
      while (cursor < source.size()) {
        const char ch = source[cursor++];
        if (ch == '\\' && cursor < source.size()) {
          ++cursor;
          continue;
        }
        if (ch == quote) {
          break;
        }
      }
      continue;
    }

    if (!IsIdentifierStart(source[cursor])) {
      ++cursor;
      continue;
    }

    const std::size_t word_start = cursor;
    ++cursor;
    while (cursor < source.size() && IsIdentifierChar(source[cursor])) {
      ++cursor;
    }
    const std::string_view word = source.substr(word_start, cursor - word_start);

    if (word != "__global__" && word != "__device__") {
      continue;
    }

    const QualifierKind kind =
        (word == "__global__") ? QualifierKind::kGlobal : QualifierKind::kDevice;

    const std::size_t open = source.find('(', cursor);
    const std::size_t semicolon = source.find(';', cursor);
    if (open == std::string_view::npos ||
        (semicolon != std::string_view::npos && semicolon < open)) {
      if (kind == QualifierKind::kGlobal) {
        return DiagnosticCode::kMalformedKernelDeclaration;
      }
      // __device__ variable declaration inside or outside function; let kernel body validation
      // handle it if inside a kernel, or reject global __device__ variable.
      return DiagnosticCode::kUnsupportedConstruct;
    }

    const std::size_t close = FindMatching(source, open, '(', ')');
    if (close == std::string_view::npos) {
      return DiagnosticCode::kUnterminatedParameterList;
    }

    std::size_t name_end = open;
    while (name_end > cursor && IsSpace(source[name_end - 1])) {
      --name_end;
    }
    std::size_t name_start = name_end;
    while (name_start > cursor && IsIdentifierChar(source[name_start - 1])) {
      --name_start;
    }
    if (name_start == name_end) {
      return DiagnosticCode::kMissingKernelName;
    }

    std::string return_type = NormalizeSpace(source.substr(cursor, name_start - cursor));
    RemoveIdentifier(return_type, "__host__");
    RemoveIdentifier(return_type, "__device__");
    RemoveIdentifier(return_type, "__forceinline__");
    RemoveIdentifier(return_type, "__noinline__");
    RemoveIdentifier(return_type, "inline");
    RemoveIdentifier(return_type, "static");
    RemoveIdentifier(return_type, "constexpr");

    if (kind == QualifierKind::kGlobal && return_type != "void") {
      return DiagnosticCode::kNonVoidKernelReturn;
    }

    std::size_t body_open = close + 1;
    while (body_open < source.size() && IsSpace(source[body_open])) {
      ++body_open;
    }
    if (body_open >= source.size() || source[body_open] != '{') {
      return DiagnosticCode::kMissingKernelBody;
    }

    const std::size_t body_close = FindMatching(source, body_open, '{', '}');
    if (body_close == std::string_view::npos) {
      return DiagnosticCode::kUnterminatedKernelBody;
    }

    out_slices->push_back(RawFunctionSlice{
        .kind = kind,
        .return_type = std::move(return_type),
        .name = std::string(source.substr(name_start, name_end - name_start)),
        .parameter_list = source.substr(open + 1, close - open - 1),
        .body = source.substr(body_open + 1, body_close - body_open - 1),
    });

    cursor = body_close + 1;
  }

  return DiagnosticCode::kOk;
}

}  // namespace

TranslationResult TranslateCudaToMsl(std::string_view source) {
  std::vector<RawFunctionSlice> slices;
  const DiagnosticCode extract_status = ExtractFunctionsFromSource(source, &slices);
  if (extract_status != DiagnosticCode::kOk) {
    return Failure(extract_status);
  }

  mmir::Module module;
  for (const RawFunctionSlice& slice : slices) {
    if (slice.kind == QualifierKind::kDevice) {
      mmir::DeviceFunction dev_fn;
      dev_fn.name = slice.name;
      if (slice.return_type == "void") {
        dev_fn.returns_void = true;
      } else {
        const std::optional<DataType> ret_type = ResolveCudaType(slice.return_type);
        if (!ret_type.has_value()) {
          return Failure(DiagnosticCode::kUnsupportedParameterType);
        }
        dev_fn.return_type = *ret_type;
      }

      const DiagnosticCode param_status =
          ParseParameterList(slice.parameter_list, &dev_fn.parameters);
      if (param_status != DiagnosticCode::kOk) {
        return Failure(param_status);
      }

      const DiagnosticCode body_status =
          ExtractSharedMemoryAndLowerBody(slice.body, nullptr, &dev_fn.lowered_body);
      if (body_status != DiagnosticCode::kOk) {
        return Failure(body_status);
      }
      module.device_functions.push_back(std::move(dev_fn));
    } else {
      mmir::KernelFunction kernel_fn;
      kernel_fn.name = slice.name;
      kernel_fn.msl_name = slice.name;

      const DiagnosticCode param_status =
          ParseParameterList(slice.parameter_list, &kernel_fn.parameters);
      if (param_status != DiagnosticCode::kOk) {
        return Failure(param_status);
      }

      const DiagnosticCode body_status = ExtractSharedMemoryAndLowerBody(
          slice.body, &kernel_fn.shared_allocations, &kernel_fn.lowered_body);
      if (body_status != DiagnosticCode::kOk) {
        return Failure(body_status);
      }
      module.kernels.push_back(std::move(kernel_fn));
    }
  }

  const DiagnosticCode verify_status = mmir::VerifyModule(module);
  if (verify_status != DiagnosticCode::kOk) {
    return Failure(verify_status);
  }

  TranslationResult result;
  result.code = DiagnosticCode::kOk;
  result.msl = mmir::EmitMslModule(module);

  result.kernels.reserve(module.kernels.size());
  for (const mmir::KernelFunction& kernel_fn : module.kernels) {
    TranslatedKernel tk;
    tk.name = kernel_fn.name;
    tk.msl_name = kernel_fn.msl_name;
    tk.parameters = kernel_fn.parameters;
    tk.uses_dynamic_shared_memory = kernel_fn.uses_dynamic_shared_memory();
    tk.msl = mmir::EmitMslKernel(module, kernel_fn);
    result.kernels.push_back(std::move(tk));
  }

  result.kernel = result.kernels.front();
  result.kernel.msl = result.msl;
  return result;
}

}  // namespace muda::compiler

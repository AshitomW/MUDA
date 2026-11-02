/**
 * @file ast_extractor.cpp
 * @brief Lexical scanning and AST function slice extraction for CUDA sources.
 */

#include "ast_extractor.hpp"

#include <algorithm>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "type_table.hpp"

namespace muda::compiler {

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
  bool in_space = false;
  for (char ch : text) {
    if (IsSpace(ch)) {
      if (!in_space && !result.empty()) {
        result.push_back(' ');
        in_space = true;
      }
      continue;
    }
    result.push_back(ch);
    in_space = false;
  }
  while (!result.empty() && IsSpace(result.back())) {
    result.pop_back();
  }
  return result;
}

std::size_t FindMatching(std::string_view text, std::size_t open, char open_char,
                         char close_char) noexcept {
  int depth = 0;
  bool in_line_comment = false;
  bool in_block_comment = false;
  bool in_string = false;
  bool in_char = false;

  for (std::size_t index = open; index < text.size(); ++index) {
    const char ch = text[index];
    const char next = (index + 1 < text.size()) ? text[index + 1] : '\0';

    if (in_line_comment) {
      if (ch == '\n')
        in_line_comment = false;
      continue;
    }
    if (in_block_comment) {
      if (ch == '*' && next == '/') {
        in_block_comment = false;
        ++index;
      }
      continue;
    }
    if (in_string) {
      if (ch == '\\')
        ++index;
      else if (ch == '"')
        in_string = false;
      continue;
    }
    if (in_char) {
      if (ch == '\\')
        ++index;
      else if (ch == '\'')
        in_char = false;
      continue;
    }

    if (ch == '/' && next == '/') {
      in_line_comment = true;
      ++index;
      continue;
    }
    if (ch == '/' && next == '*') {
      in_block_comment = true;
      ++index;
      continue;
    }
    if (ch == '"') {
      in_string = true;
      continue;
    }
    if (ch == '\'') {
      in_char = true;
      continue;
    }

    if (ch == open_char) {
      ++depth;
    } else if (ch == close_char) {
      --depth;
      if (depth == 0)
        return index;
    }
  }
  return std::string_view::npos;
}

std::vector<std::string_view> SplitParameters(std::string_view text) {
  std::vector<std::string_view> parameters;
  int paren_depth = 0;
  int template_depth = 0;
  std::size_t start = 0;

  for (std::size_t index = 0; index < text.size(); ++index) {
    const char ch = text[index];
    if (ch == '(')
      ++paren_depth;
    else if (ch == ')')
      --paren_depth;
    else if (ch == '<')
      ++template_depth;
    else if (ch == '>')
      --template_depth;
    else if (ch == ',' && paren_depth == 0 && template_depth == 0) {
      std::string_view slice = Trim(text.substr(start, index - start));
      if (!slice.empty())
        parameters.push_back(slice);
      start = index + 1;
    }
  }

  std::string_view trailing = Trim(text.substr(start));
  if (!trailing.empty())
    parameters.push_back(trailing);
  return parameters;
}

void RemoveIdentifier(std::string& text, std::string_view identifier) {
  std::size_t pos = 0;
  while ((pos = text.find(identifier, pos)) != std::string::npos) {
    const bool left_ok = (pos == 0) || !IsIdentifierChar(text[pos - 1]);
    const std::size_t right = pos + identifier.size();
    const bool right_ok = (right >= text.size()) || !IsIdentifierChar(text[right]);
    if (left_ok && right_ok) {
      text.erase(pos, identifier.size());
    } else {
      pos = right;
    }
  }
}

ParsedParameter ParseParameter(std::string_view text, std::size_t index) {
  const std::string normalized = NormalizeSpace(text);
  if (normalized.empty()) {
    return {.code=DiagnosticCode::kEmptyKernelParameter, .parameter={}};
  }
  if (normalized.find("(*") != std::string::npos) {
    return {.code=DiagnosticCode::kFunctionPointerUnsupported, .parameter={}};
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

DiagnosticCode ParseParameterList(std::string_view parameter_list,
                                  std::vector<KernelParameter>* out_params) {
  if (out_params == nullptr) {
    return DiagnosticCode::kEmptyKernelParameter;
  }
  out_params->clear();
  const std::vector<std::string_view> slices = SplitParameters(parameter_list);
  for (std::size_t index = 0; index < slices.size(); ++index) {
    ParsedParameter parsed = ParseParameter(slices[index], index);
    if (parsed.code != DiagnosticCode::kOk) {
      return parsed.code;
    }
    out_params->push_back(std::move(parsed.parameter));
  }
  return DiagnosticCode::kOk;
}

namespace {

DiagnosticCode ParseSharedMemoryDecl(std::string_view decl_text, bool is_extern,
                                     mmir::SharedAllocation* out_alloc) {
  const std::size_t bracket_open = decl_text.find('[');
  const std::size_t bracket_close = decl_text.rfind(']');
  if (bracket_open == std::string_view::npos || bracket_close == std::string_view::npos ||
      bracket_close < bracket_open) {
    return DiagnosticCode::kInvalidSharedMemoryDeclaration;
  }

  const std::string_view count_expr =
      Trim(decl_text.substr(bracket_open + 1, bracket_close - bracket_open - 1));
  if (is_extern && !count_expr.empty()) {
    return DiagnosticCode::kInvalidSharedMemoryDeclaration;
  }
  if (!is_extern && count_expr.empty()) {
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

  out_alloc->name = std::string(type_and_name.substr(name_start, name_end - name_start));
  out_alloc->element_type = *elem_type;
  out_alloc->count_expression = std::string(count_expr);
  out_alloc->is_dynamic = is_extern;
  return DiagnosticCode::kOk;
}

void CleanReturnType(std::string& return_type) {
  RemoveIdentifier(return_type, "__host__");
  RemoveIdentifier(return_type, "__device__");
  RemoveIdentifier(return_type, "__forceinline__");
  RemoveIdentifier(return_type, "__noinline__");
  RemoveIdentifier(return_type, "inline");
  RemoveIdentifier(return_type, "static");
  RemoveIdentifier(return_type, "constexpr");
}

std::string LowerExpression(std::string_view expr) {
  std::string result;
  result.reserve(expr.size() + 16);
  for (std::size_t i = 0; i < expr.size();) {
    if (!IsIdentifierStart(expr[i])) {
      result.push_back(expr[i]);
      ++i;
      continue;
    }
    const std::size_t start = i;
    ++i;
    while (i < expr.size() && IsIdentifierChar(expr[i])) {
      ++i;
    }
    const std::string_view word = expr.substr(start, i - start);
    const std::string_view builtin_rep = BuiltinReplacement(word);
    std::size_t next = i;
    while (next < expr.size() && IsSpace(expr[next])) {
      ++next;
    }
    if (!builtin_rep.empty() && next < expr.size() && expr[next] == '.') {
      std::size_t axis = next + 1;
      while (axis < expr.size() && IsSpace(expr[axis])) {
        ++axis;
      }
      if (axis < expr.size() && (expr[axis] == 'x' || expr[axis] == 'y' || expr[axis] == 'z')) {
        result.append(builtin_rep);
        result.push_back('.');
        result.push_back(expr[axis]);
        i = axis + 1;
        continue;
      }
    }
    if (word == "__half") {
      result.append("half");
      continue;
    }
    if (word == "__half2") {
      result.append("half2");
      continue;
    }
    const std::string_view math_rep = MathIntrinsicReplacement(word);
    if (!math_rep.empty() && next < expr.size() && expr[next] == '(') {
      result.append(math_rep);
      continue;
    }
    result.append(word);
  }
  return result;
}

std::string_view ExtractBaseIdentifier(std::string_view expr) {
  std::string_view trimmed = Trim(expr);
  if (!trimmed.empty() && trimmed.front() == '&') {
    trimmed = Trim(trimmed.substr(1));
  }
  std::size_t start = 0;
  while (start < trimmed.size() && !IsIdentifierStart(trimmed[start])) {
    ++start;
  }
  std::size_t end = start;
  while (end < trimmed.size() && IsIdentifierChar(trimmed[end])) {
    ++end;
  }
  return (end > start) ? trimmed.substr(start, end - start) : std::string_view{};
}

struct AtomicTargetInfo {
  std::string_view addr_space = "device";
  std::string_view atomic_type = "atomic_int";
  bool is_float = false;
};

AtomicTargetInfo ResolveAtomicTarget(
    std::string_view addr_expr,
    std::string_view val_expr,
    const std::vector<mmir::SharedAllocation>* shared_allocs,
    std::span<const KernelParameter> parameters) {
  const std::string_view base_id = ExtractBaseIdentifier(addr_expr);
  AtomicTargetInfo info;

  bool found = false;
  if (shared_allocs != nullptr) {
    for (const auto& alloc : *shared_allocs) {
      if (alloc.name == base_id) {
        info.addr_space = "threadgroup";
        info.is_float = (alloc.element_type.scalar == ScalarType::kFloat32);
        const bool is_uint = (alloc.element_type.scalar == ScalarType::kUInt32 ||
                              alloc.element_type.scalar == ScalarType::kUInt64);
        const bool is_64bit = (alloc.element_type.scalar == ScalarType::kInt64 ||
                               alloc.element_type.scalar == ScalarType::kUInt64);
        info.atomic_type = is_64bit ? (is_uint ? "atomic_ulong" : "atomic_long")
                                    : (is_uint ? "atomic_uint" : "atomic_int");
        found = true;
        break;
      }
    }
  }

  if (!found) {
    for (const auto& param : parameters) {
      if (param.name == base_id) {
        info.addr_space = "device";
        info.is_float = (param.type.scalar == ScalarType::kFloat32);
        const bool is_uint = (param.type.scalar == ScalarType::kUInt32 ||
                              param.type.scalar == ScalarType::kUInt64);
        const bool is_64bit = (param.type.scalar == ScalarType::kInt64 ||
                               param.type.scalar == ScalarType::kUInt64);
        info.atomic_type = is_64bit ? (is_uint ? "atomic_ulong" : "atomic_long")
                                    : (is_uint ? "atomic_uint" : "atomic_int");
        found = true;
        break;
      }
    }
  }

  if (!found) {
    info.addr_space = "device";
    if (val_expr.find('.') != std::string_view::npos ||
        (!val_expr.empty() && (val_expr.back() == 'f' || val_expr.back() == 'F'))) {
      info.is_float = true;
      info.atomic_type = "atomic_uint";
    } else if (!val_expr.empty() && (val_expr.back() == 'u' || val_expr.back() == 'U')) {
      info.atomic_type = "atomic_uint";
    } else {
      info.atomic_type = "atomic_int";
    }
  }

  return info;
}

}  // namespace

DiagnosticCode ExtractSharedMemoryAndLowerBody(std::string_view raw_body,
                                               std::vector<mmir::SharedAllocation>* out_shared,
                                               std::string* out_lowered_body,
                                               std::span<const KernelParameter> parameters) {
  std::string body_without_shared;
  body_without_shared.reserve(raw_body.size());

  std::size_t cursor = 0;
  while (cursor < raw_body.size()) {
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
        while (next < raw_body.size() && IsSpace(raw_body[next]))
          ++next;
        std::size_t next_end = next;
        while (next_end < raw_body.size() && IsIdentifierChar(raw_body[next_end]))
          ++next_end;
        if (raw_body.substr(next, next_end - next) == "__shared__") {
          is_extern_shared = true;
          after_shared = next_end;
        }
      }

      if (first_word == "__shared__" || is_extern_shared) {
        if (out_shared == nullptr)
          return DiagnosticCode::kUnsupportedConstruct;
        const std::size_t semi = raw_body.find(';', after_shared);
        if (semi == std::string_view::npos)
          return DiagnosticCode::kInvalidSharedMemoryDeclaration;

        mmir::SharedAllocation alloc;
        const auto diag = ParseSharedMemoryDecl(
            Trim(raw_body.substr(after_shared, semi - after_shared)), is_extern_shared, &alloc);
        if (diag != DiagnosticCode::kOk)
          return diag;

        out_shared->push_back(std::move(alloc));
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
    while (index < body.size() && IsIdentifierChar(body[index]))
      ++index;
    const std::string_view word = body.substr(start, index - start);

    for (const std::string_view unsupported : kUnsupportedConstructs) {
      if (word == unsupported)
        return DiagnosticCode::kUnsupportedConstruct;
    }
    if (word == "double")
      return DiagnosticCode::kUnsupportedConstruct;

    const std::string_view builtin_rep = BuiltinReplacement(word);
    std::size_t next = index;
    while (next < body.size() && IsSpace(body[next]))
      ++next;

    if (!builtin_rep.empty() && next < body.size() && body[next] == '.') {
      std::size_t axis = next + 1;
      while (axis < body.size() && IsSpace(body[axis]))
        ++axis;
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

    if ((word == "__shfl_sync" || word == "__shfl_down_sync" || word == "__shfl_up_sync" ||
         word == "__shfl_xor_sync" || word == "__ballot_sync" || word == "__any_sync" ||
         word == "__all_sync" || word == "__activemask" || word == "__ffs" || word == "__ffsll") &&
        next < body.size() && body[next] == '(') {
      const std::size_t close = FindMatching(body, next, '(', ')');
      if (close != std::string_view::npos) {
        const std::string_view arg_str = body.substr(next + 1, close - next - 1);
        const std::vector<std::string_view> args = SplitParameters(arg_str);
        if (word == "__shfl_sync" && args.size() >= 3) {
          result.append("simd_shuffle(");
          result.append(LowerExpression(args[1]));
          result.append(", static_cast<ushort>(");
          result.append(LowerExpression(args[2]));
          result.append("))");
          index = close + 1;
          continue;
        }
        if (word == "__shfl_down_sync" && args.size() >= 3) {
          result.append("simd_shuffle_down(");
          result.append(LowerExpression(args[1]));
          result.append(", static_cast<ushort>(");
          result.append(LowerExpression(args[2]));
          result.append("))");
          index = close + 1;
          continue;
        }
        if (word == "__shfl_up_sync" && args.size() >= 3) {
          result.append("simd_shuffle_up(");
          result.append(LowerExpression(args[1]));
          result.append(", static_cast<ushort>(");
          result.append(LowerExpression(args[2]));
          result.append("))");
          index = close + 1;
          continue;
        }
        if (word == "__shfl_xor_sync" && args.size() >= 3) {
          result.append("simd_shuffle_xor(");
          result.append(LowerExpression(args[1]));
          result.append(", static_cast<ushort>(");
          result.append(LowerExpression(args[2]));
          result.append("))");
          index = close + 1;
          continue;
        }
        if (word == "__ballot_sync" && args.size() >= 2) {
          result.append("static_cast<uint>(static_cast<ulong>(simd_ballot(");
          result.append(LowerExpression(args[1]));
          result.append(" != 0)))");
          index = close + 1;
          continue;
        }
        if (word == "__any_sync" && args.size() >= 2) {
          result.append("(simd_any(");
          result.append(LowerExpression(args[1]));
          result.append(" != 0) ? 1 : 0)");
          index = close + 1;
          continue;
        }
        if (word == "__all_sync" && args.size() >= 2) {
          result.append("(simd_all(");
          result.append(LowerExpression(args[1]));
          result.append(" != 0) ? 1 : 0)");
          index = close + 1;
          continue;
        }
        if (word == "__activemask") {
          result.append("static_cast<uint>(static_cast<ulong>(simd_ballot(true)))");
          index = close + 1;
          continue;
        }
        if (word == "__ffs" && !args.empty()) {
          result.append("([&](auto __val) { return (__val == 0) ? 0 : static_cast<int>(ctz(__val) + 1u); }(");
          result.append(LowerExpression(args[0]));
          result.append("))");
          index = close + 1;
          continue;
        }
        if (word == "__ffsll" && !args.empty()) {
          result.append("([&](auto __val) { return (__val == 0ull) ? 0 : static_cast<int>(ctz(__val) + 1u); }(");
          result.append(LowerExpression(args[0]));
          result.append("))");
          index = close + 1;
          continue;
        }
      }
    }

    if ((word == "__float2half" || word == "__half2float" || word == "__floats2half2_rn" ||
         word == "__float2half2_rn" || word == "__hadd" || word == "__hsub" || word == "__hmul" ||
         word == "__hadd2" || word == "__hsub2" || word == "__hmul2") &&
        next < body.size() && body[next] == '(') {
      const std::size_t close = FindMatching(body, next, '(', ')');
      if (close != std::string_view::npos) {
        const std::string_view arg_str = body.substr(next + 1, close - next - 1);
        const std::vector<std::string_view> args = SplitParameters(arg_str);
        if (word == "__float2half" && !args.empty()) {
          result.append("static_cast<half>(");
          result.append(LowerExpression(args[0]));
          result.append(")");
          index = close + 1;
          continue;
        }
        if (word == "__half2float" && !args.empty()) {
          result.append("static_cast<float>(");
          result.append(LowerExpression(args[0]));
          result.append(")");
          index = close + 1;
          continue;
        }
        if (word == "__floats2half2_rn" && args.size() >= 2) {
          result.append("half2(static_cast<half>(");
          result.append(LowerExpression(args[0]));
          result.append("), static_cast<half>(");
          result.append(LowerExpression(args[1]));
          result.append("))");
          index = close + 1;
          continue;
        }
        if (word == "__float2half2_rn" && !args.empty()) {
          result.append("half2(static_cast<half>(");
          result.append(LowerExpression(args[0]));
          result.append("), static_cast<half>(");
          result.append(LowerExpression(args[0]));
          result.append("))");
          index = close + 1;
          continue;
        }
        if ((word == "__hadd" || word == "__hadd2") && args.size() >= 2) {
          result.append("((");
          result.append(LowerExpression(args[0]));
          result.append(") + (");
          result.append(LowerExpression(args[1]));
          result.append("))");
          index = close + 1;
          continue;
        }
        if ((word == "__hsub" || word == "__hsub2") && args.size() >= 2) {
          result.append("((");
          result.append(LowerExpression(args[0]));
          result.append(") - (");
          result.append(LowerExpression(args[1]));
          result.append("))");
          index = close + 1;
          continue;
        }
        if ((word == "__hmul" || word == "__hmul2") && args.size() >= 2) {
          result.append("((");
          result.append(LowerExpression(args[0]));
          result.append(") * (");
          result.append(LowerExpression(args[1]));
          result.append("))");
          index = close + 1;
          continue;
        }
      }
    }

    if ((word == "atomicAdd" || word == "atomicSub" || word == "atomicExch" ||
         word == "atomicMin" || word == "atomicMax" || word == "atomicAnd" ||
         word == "atomicOr" || word == "atomicXor" || word == "atomicInc" ||
         word == "atomicDec" || word == "atomicCAS") &&
        next < body.size() && body[next] == '(') {
      const std::size_t close = FindMatching(body, next, '(', ')');
      if (close != std::string_view::npos) {
        const std::string_view arg_str = body.substr(next + 1, close - next - 1);
        const std::vector<std::string_view> args = SplitParameters(arg_str);
        if (args.size() >= 2) {
          const AtomicTargetInfo target = ResolveAtomicTarget(args[0], args[1], out_shared, parameters);
          const std::string lowered_addr = LowerExpression(args[0]);
          const std::string lowered_val = LowerExpression(args[1]);

          if (word == "atomicAdd") {
            if (target.is_float) {
              result.append("([&](");
              result.append(target.addr_space);
              result.append(" auto* __a, float __v) { ");
              result.append(target.addr_space);
              result.append(" atomic_uint* __u = reinterpret_cast<");
              result.append(target.addr_space);
              result.append(" atomic_uint*>(__a); uint __exp = atomic_load_explicit(__u, memory_order_relaxed); while (!atomic_compare_exchange_weak_explicit(__u, &__exp, as_type<uint>(as_type<float>(__exp) + __v), memory_order_relaxed, memory_order_relaxed)) {} return as_type<float>(__exp); }(");
              result.append(lowered_addr);
              result.append(", ");
              result.append(lowered_val);
              result.append("))");
              index = close + 1;
              continue;
            }
            result.append("atomic_fetch_add_explicit(reinterpret_cast<");
            result.append(target.addr_space);
            result.push_back(' ');
            result.append(target.atomic_type);
            result.append("*>(");
            result.append(lowered_addr);
            result.append("), ");
            result.append(lowered_val);
            result.append(", memory_order_relaxed)");
            index = close + 1;
            continue;
          }

          if (word == "atomicSub") {
            result.append("atomic_fetch_sub_explicit(reinterpret_cast<");
            result.append(target.addr_space);
            result.push_back(' ');
            result.append(target.atomic_type);
            result.append("*>(");
            result.append(lowered_addr);
            result.append("), ");
            result.append(lowered_val);
            result.append(", memory_order_relaxed)");
            index = close + 1;
            continue;
          }

          if (word == "atomicExch") {
            if (target.is_float) {
              result.append("([&](");
              result.append(target.addr_space);
              result.append(" auto* __a, float __v) { ");
              result.append(target.addr_space);
              result.append(" atomic_uint* __u = reinterpret_cast<");
              result.append(target.addr_space);
              result.append(" atomic_uint*>(__a); return as_type<float>(atomic_exchange_explicit(__u, as_type<uint>(__v), memory_order_relaxed)); }(");
              result.append(lowered_addr);
              result.append(", ");
              result.append(lowered_val);
              result.append("))");
              index = close + 1;
              continue;
            }
            result.append("atomic_exchange_explicit(reinterpret_cast<");
            result.append(target.addr_space);
            result.push_back(' ');
            result.append(target.atomic_type);
            result.append("*>(");
            result.append(lowered_addr);
            result.append("), ");
            result.append(lowered_val);
            result.append(", memory_order_relaxed)");
            index = close + 1;
            continue;
          }

          if (word == "atomicMin") {
            result.append("atomic_fetch_min_explicit(reinterpret_cast<");
            result.append(target.addr_space);
            result.push_back(' ');
            result.append(target.atomic_type);
            result.append("*>(");
            result.append(lowered_addr);
            result.append("), ");
            result.append(lowered_val);
            result.append(", memory_order_relaxed)");
            index = close + 1;
            continue;
          }

          if (word == "atomicMax") {
            result.append("atomic_fetch_max_explicit(reinterpret_cast<");
            result.append(target.addr_space);
            result.push_back(' ');
            result.append(target.atomic_type);
            result.append("*>(");
            result.append(lowered_addr);
            result.append("), ");
            result.append(lowered_val);
            result.append(", memory_order_relaxed)");
            index = close + 1;
            continue;
          }

          if (word == "atomicAnd") {
            result.append("atomic_fetch_and_explicit(reinterpret_cast<");
            result.append(target.addr_space);
            result.push_back(' ');
            result.append(target.atomic_type);
            result.append("*>(");
            result.append(lowered_addr);
            result.append("), ");
            result.append(lowered_val);
            result.append(", memory_order_relaxed)");
            index = close + 1;
            continue;
          }

          if (word == "atomicOr") {
            result.append("atomic_fetch_or_explicit(reinterpret_cast<");
            result.append(target.addr_space);
            result.push_back(' ');
            result.append(target.atomic_type);
            result.append("*>(");
            result.append(lowered_addr);
            result.append("), ");
            result.append(lowered_val);
            result.append(", memory_order_relaxed)");
            index = close + 1;
            continue;
          }

          if (word == "atomicXor") {
            result.append("atomic_fetch_xor_explicit(reinterpret_cast<");
            result.append(target.addr_space);
            result.push_back(' ');
            result.append(target.atomic_type);
            result.append("*>(");
            result.append(lowered_addr);
            result.append("), ");
            result.append(lowered_val);
            result.append(", memory_order_relaxed)");
            index = close + 1;
            continue;
          }

          if (word == "atomicInc") {
            result.append("([&](");
            result.append(target.addr_space);
            result.append(" auto* __a, auto __v) { ");
            result.append(target.addr_space);
            result.push_back(' ');
            result.append(target.atomic_type);
            result.append("* __u = reinterpret_cast<");
            result.append(target.addr_space);
            result.push_back(' ');
            result.append(target.atomic_type);
            result.append("*>(__a); auto __exp = atomic_load_explicit(__u, memory_order_relaxed); while (true) { auto __desired = (__exp >= __v) ? 0u : (__exp + 1u); if (atomic_compare_exchange_weak_explicit(__u, &__exp, __desired, memory_order_relaxed, memory_order_relaxed)) { return __exp; } } }(");
            result.append(lowered_addr);
            result.append(", ");
            result.append(lowered_val);
            result.append("))");
            index = close + 1;
            continue;
          }

          if (word == "atomicDec") {
            result.append("([&](");
            result.append(target.addr_space);
            result.append(" auto* __a, auto __v) { ");
            result.append(target.addr_space);
            result.push_back(' ');
            result.append(target.atomic_type);
            result.append("* __u = reinterpret_cast<");
            result.append(target.addr_space);
            result.push_back(' ');
            result.append(target.atomic_type);
            result.append("*>(__a); auto __exp = atomic_load_explicit(__u, memory_order_relaxed); while (true) { auto __desired = ((__exp == 0u) || (__exp > __v)) ? __v : (__exp - 1u); if (atomic_compare_exchange_weak_explicit(__u, &__exp, __desired, memory_order_relaxed, memory_order_relaxed)) { return __exp; } } }(");
            result.append(lowered_addr);
            result.append(", ");
            result.append(lowered_val);
            result.append("))");
            index = close + 1;
            continue;
          }

          if (word == "atomicCAS" && args.size() >= 3) {
            const std::string lowered_cmp = LowerExpression(args[1]);
            const std::string lowered_desired = LowerExpression(args[2]);
            if (target.is_float) {
              result.append("([&](");
              result.append(target.addr_space);
              result.append(" auto* __p, float __cmp, float __v) { ");
              result.append(target.addr_space);
              result.append(" atomic_uint* __u = reinterpret_cast<");
              result.append(target.addr_space);
              result.append(" atomic_uint*>(__p); uint __exp = as_type<uint>(__cmp); while (!atomic_compare_exchange_weak_explicit(__u, &__exp, as_type<uint>(__v), memory_order_relaxed, memory_order_relaxed)) { if (__exp != as_type<uint>(__cmp)) { break; } } return as_type<float>(__exp); }(");
              result.append(lowered_addr);
              result.append(", ");
              result.append(lowered_cmp);
              result.append(", ");
              result.append(lowered_desired);
              result.append("))");
              index = close + 1;
              continue;
            }
            result.append("([&](");
            result.append(target.addr_space);
            result.append(" auto* __p, auto __cmp, auto __v) { auto __exp = __cmp; while (!atomic_compare_exchange_weak_explicit(reinterpret_cast<");
            result.append(target.addr_space);
            result.push_back(' ');
            result.append(target.atomic_type);
            result.append("*>(__p), &__exp, __v, memory_order_relaxed, memory_order_relaxed)) { if (__exp != __cmp) { break; } } return __exp; }(");
            result.append(lowered_addr);
            result.append(", ");
            result.append(lowered_cmp);
            result.append(", ");
            result.append(lowered_desired);
            result.append("))");
            index = close + 1;
            continue;
          }
        }
      }
    }

    if (word == "__half") {
      result.append("half");
      continue;
    }
    if (word == "__half2") {
      result.append("half2");
      continue;
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

DiagnosticCode ExtractFunctionsFromSource(std::string_view source,
                                          std::vector<RawFunctionSlice>* out_slices) {
  if (out_slices == nullptr)
    return DiagnosticCode::kMalformedKernelDeclaration;
  out_slices->clear();

  std::size_t cursor = 0;
  while (cursor < source.size()) {
    if (source[cursor] == '/' && cursor + 1 < source.size() && source[cursor + 1] == '/') {
      const std::size_t nl = source.find('\n', cursor + 2);
      if (nl == std::string_view::npos)
        break;
      cursor = nl + 1;
      continue;
    }
    if (source[cursor] == '/' && cursor + 1 < source.size() && source[cursor + 1] == '*') {
      const std::size_t end_c = source.find("*/", cursor + 2);
      if (end_c == std::string_view::npos)
        break;
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
        if (ch == quote)
          break;
      }
      continue;
    }

    if (!IsIdentifierStart(source[cursor])) {
      ++cursor;
      continue;
    }

    const std::size_t word_start = cursor;
    ++cursor;
    while (cursor < source.size() && IsIdentifierChar(source[cursor]))
      ++cursor;
    const std::string_view word = source.substr(word_start, cursor - word_start);

    if (word != "__global__" && word != "__device__")
      continue;
    const QualifierKind kind =
        (word == "__global__") ? QualifierKind::kGlobal : QualifierKind::kDevice;

    const std::size_t open = source.find('(', cursor);
    const std::size_t semicolon = source.find(';', cursor);
    if (open == std::string_view::npos ||
        (semicolon != std::string_view::npos && semicolon < open)) {
      if (kind == QualifierKind::kGlobal)
        return DiagnosticCode::kMalformedKernelDeclaration;
      return DiagnosticCode::kUnsupportedConstruct;
    }

    const std::size_t close = FindMatching(source, open, '(', ')');
    if (close == std::string_view::npos)
      return DiagnosticCode::kUnterminatedParameterList;

    std::size_t name_end = open;
    while (name_end > cursor && IsSpace(source[name_end - 1]))
      --name_end;
    std::size_t name_start = name_end;
    while (name_start > cursor && IsIdentifierChar(source[name_start - 1]))
      --name_start;
    if (name_start == name_end)
      return DiagnosticCode::kMissingKernelName;

    std::string return_type = NormalizeSpace(source.substr(cursor, name_start - cursor));
    CleanReturnType(return_type);

    if (kind == QualifierKind::kGlobal && return_type != "void") {
      return DiagnosticCode::kNonVoidKernelReturn;
    }

    std::size_t body_open = close + 1;
    while (body_open < source.size() && IsSpace(source[body_open]))
      ++body_open;
    if (body_open >= source.size() || source[body_open] != '{') {
      return DiagnosticCode::kMissingKernelBody;
    }

    const std::size_t body_close = FindMatching(source, body_open, '{', '}');
    if (body_close == std::string_view::npos)
      return DiagnosticCode::kUnterminatedKernelBody;

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

void ExtractIncludes(std::string_view source, std::vector<std::string>* out_includes) {
  if (out_includes == nullptr) {
    return;
  }
  std::size_t cursor = 0;
  while (cursor < source.size()) {
    const std::size_t line_end = source.find('\n', cursor);
    const std::string_view line = (line_end == std::string_view::npos)
                                      ? source.substr(cursor)
                                      : source.substr(cursor, line_end - cursor);
    std::string_view sv = line;
    while (!sv.empty() && (sv.front() == ' ' || sv.front() == '\t')) {
      sv.remove_prefix(1);
    }
    if (sv.starts_with('#')) {
      sv.remove_prefix(1);
      while (!sv.empty() && (sv.front() == ' ' || sv.front() == '\t')) {
        sv.remove_prefix(1);
      }
      if (sv.starts_with("include")) {
        sv.remove_prefix(7);
        while (!sv.empty() && (sv.front() == ' ' || sv.front() == '\t')) {
          sv.remove_prefix(1);
        }
        if (sv.starts_with('"')) {
          if (line.find("cuda_runtime.h") == std::string_view::npos &&
              line.find("cuda.h") == std::string_view::npos &&
              line.find("cuda_fp16.h") == std::string_view::npos) {
            out_includes->emplace_back(Trim(line));
          }
        }
      }
    }
    if (line_end == std::string_view::npos) {
      break;
    }
    cursor = line_end + 1;
  }
}

}  // namespace muda::compiler

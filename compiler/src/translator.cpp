/**
 * @file translator.cpp
 * @brief Implements the CUDA-to-MSL translation prototype.
 */

#include "muda/compiler/translator.hpp"

#include <algorithm>
#include <cctype>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace muda::compiler {
namespace {

bool IsIdentifierStart(char value) {
  return std::isalpha(static_cast<unsigned char>(value)) != 0 || value == '_';
}

bool IsIdentifierChar(char value) {
  return std::isalnum(static_cast<unsigned char>(value)) != 0 || value == '_';
}

bool IsSpace(char value) { return std::isspace(static_cast<unsigned char>(value)) != 0; }

std::string_view Trim(std::string_view text) {
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

std::size_t FindMatching(std::string_view text, std::size_t open, char open_char, char close_char) {
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

std::string MapType(std::string_view type) {
  if (type == "float" || type == "float2" || type == "float3" || type == "float4") {
    return std::string(type);
  }
  if (type == "half" || type == "half2" || type == "half3" || type == "half4") {
    return std::string(type);
  }
  if (type == "int" || type == "int2" || type == "int3" || type == "int4") {
    return std::string(type);
  }
  if (type == "uint" || type == "uint2" || type == "uint3" || type == "uint4") {
    return std::string(type);
  }
  if (type == "unsigned" || type == "unsigned int") {
    return "uint";
  }
  if (type == "short" || type == "ushort" || type == "char" || type == "uchar") {
    return std::string(type);
  }
  if (type == "unsigned char") {
    return "uchar";
  }
  if (type == "unsigned short") {
    return "ushort";
  }
  if (type == "bool") {
    return "bool";
  }
  if (type == "long" || type == "unsigned long" || type == "long2" || type == "long3" ||
      type == "long4") {
    return std::string(type);
  }
  return {};
}

struct ParsedParameter {
  bool ok = false;
  KernelParameter parameter;
  std::string error;
};

ParsedParameter ParseParameter(std::string_view text, std::size_t index) {
  const std::string normalized = NormalizeSpace(text);
  if (normalized.empty()) {
    return {false, {}, "empty kernel parameter"};
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
    return {false, {}, "missing kernel parameter name"};
  }

  KernelParameter parameter;
  parameter.name = declaration.substr(name_start, name_end - name_start);
  parameter.buffer_index = index;

  std::string type = declaration.substr(0, name_start);
  parameter.is_pointer = type.find('*') != std::string::npos;
  parameter.is_const = type.find("const") != std::string::npos;
  type.erase(std::remove(type.begin(), type.end(), '*'), type.end());
  RemoveIdentifier(type, "const");
  RemoveIdentifier(type, "volatile");
  RemoveIdentifier(type, "__restrict__");
  RemoveIdentifier(type, "__restrict");
  RemoveIdentifier(type, "restrict");

  parameter.cuda_type = NormalizeSpace(type);
  parameter.metal_type = MapType(parameter.cuda_type);
  if (parameter.metal_type.empty()) {
    return {false, {}, "unsupported kernel parameter type: " + parameter.cuda_type};
  }

  return {true, std::move(parameter), {}};
}

std::string_view BuiltinReplacement(std::string_view word) {
  if (word == "blockIdx") {
    return "threadgroup_position_in_grid";
  }
  if (word == "blockDim") {
    return "threads_per_threadgroup";
  }
  if (word == "threadIdx") {
    return "thread_position_in_threadgroup";
  }
  if (word == "gridDim") {
    return "threadgroups_per_grid";
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

TranslationResult Failure(std::string message) {
  TranslationResult result;
  result.ok = false;
  result.error = std::move(message);
  return result;
}

}

TranslationResult TranslateCudaToMsl(std::string_view source) {
  constexpr std::string_view kMarker = "__global__";
  const std::size_t marker_position = source.find(kMarker);
  if (marker_position == std::string_view::npos) {
    return Failure("no __global__ kernel found");
  }
  if (source.find(kMarker, marker_position + kMarker.size()) != std::string_view::npos) {
    return Failure("only one kernel is supported");
  }

  const std::size_t open = source.find('(', marker_position + kMarker.size());
  const std::size_t semicolon = source.find(';', marker_position + kMarker.size());
  if (open == std::string_view::npos || (semicolon != std::string_view::npos && semicolon < open)) {
    return Failure("malformed kernel declaration");
  }

  const std::size_t close = FindMatching(source, open, '(', ')');
  if (close == std::string_view::npos) {
    return Failure("unterminated kernel parameter list");
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
    return Failure("missing kernel name");
  }

  const std::string name(source.substr(name_start, name_end - name_start));
  const std::string return_type = NormalizeSpace(source.substr(
      marker_position + kMarker.size(), name_start - (marker_position + kMarker.size())));
  if (return_type != "void") {
    return Failure("kernel return type must be void");
  }

  std::size_t body_open = close + 1;
  while (body_open < source.size() && IsSpace(source[body_open])) {
    ++body_open;
  }
  if (body_open >= source.size() || source[body_open] != '{') {
    return Failure("missing kernel body");
  }

  const std::size_t body_close = FindMatching(source, body_open, '{', '}');
  if (body_close == std::string_view::npos) {
    return Failure("unterminated kernel body");
  }

  const std::string_view body = source.substr(body_open + 1, body_close - body_open - 1);
  const std::string_view unsupported[] = {
      "__shared__", "__device__",  "__constant__",   "__managed__",      "cooperative_groups",
      "atomicAdd",  "atomicSub",   "atomicCAS",      "atomicExch",       "atomicMin",
      "atomicMax",  "atomicInc",   "atomicDec",      "atomicAnd",        "atomicOr",
      "atomicXor",  "__shfl_sync", "__shfl_up_sync", "__shfl_down_sync", "printf",
  };
  for (const std::string_view token : unsupported) {
    if (body.find(token) != std::string_view::npos) {
      return Failure("unsupported kernel construct: " + std::string(token));
    }
  }

  TranslatedKernel kernel;
  kernel.name = name;

  const std::vector<std::string_view> parameters =
      SplitParameters(source.substr(open + 1, close - open - 1));
  for (std::size_t index = 0; index < parameters.size(); ++index) {
    ParsedParameter parsed = ParseParameter(parameters[index], index);
    if (!parsed.ok) {
      return Failure(std::move(parsed.error));
    }
    kernel.parameters.push_back(std::move(parsed.parameter));
  }

  std::string msl;
  msl += "#include <metal_stdlib>\n";
  msl += "using namespace metal;\n\n";
  msl += "kernel void ";
  msl += kernel.name;
  msl += "(\n";

  for (const KernelParameter& parameter : kernel.parameters) {
    msl += "    ";
    if (parameter.is_pointer) {
      msl += "device ";
      if (parameter.is_const) {
        msl += "const ";
      }
      msl += parameter.metal_type;
      msl += "* ";
    } else {
      msl += "constant ";
      msl += parameter.metal_type;
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
  result.ok = true;
  result.kernel = std::move(kernel);
  return result;
}

}

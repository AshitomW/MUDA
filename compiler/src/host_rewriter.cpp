/**
 * @file host_rewriter.cpp
 * @brief Implements CUDA host-side C++20 lowering and embedded metallib registration.
 */

#include "muda/compiler/host_rewriter.hpp"

#include <cctype>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace muda::compiler {
namespace {

bool IsSpace(char value) noexcept { return std::isspace(static_cast<unsigned char>(value)) != 0; }

bool IsIdentifierStart(char value) noexcept {
  return std::isalpha(static_cast<unsigned char>(value)) != 0 || value == '_';
}

bool IsIdentifierChar(char value) noexcept {
  return std::isalnum(static_cast<unsigned char>(value)) != 0 || value == '_';
}

std::string_view Trim(std::string_view text) noexcept {
  while (!text.empty() && IsSpace(text.front())) {
    text.remove_prefix(1);
  }
  while (!text.empty() && IsSpace(text.back())) {
    text.remove_suffix(1);
  }
  return text;
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

std::size_t FindMatchingTripleAngle(std::string_view text, std::size_t open_triple) noexcept {
  int paren_depth = 0;
  char quote = '\0';
  bool escape = false;

  for (std::size_t index = open_triple + 3; index + 2 < text.size(); ++index) {
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
    if (value == '(' || value == '[' || value == '{') {
      ++paren_depth;
      continue;
    }
    if (value == ')' || value == ']' || value == '}') {
      if (paren_depth == 0) {
        return std::string_view::npos;
      }
      --paren_depth;
      continue;
    }
    if (paren_depth == 0 && value == '>' && text[index + 1] == '>' && text[index + 2] == '>') {
      return index;
    }
  }
  return std::string_view::npos;
}

std::vector<std::string_view> SplitTopLevelCommas(std::string_view text) {
  std::vector<std::string_view> parts;
  const std::string_view trimmed = Trim(text);
  if (trimmed.empty()) {
    return parts;
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
    if (value == '(' || value == '[' || value == '{') {
      ++depth;
    } else if (value == ')' || value == ']' || value == '}') {
      --depth;
    } else if (value == ',' && depth == 0) {
      parts.push_back(Trim(text.substr(start, index - start)));
      start = index + 1;
    }
  }
  parts.push_back(Trim(text.substr(start)));
  return parts;
}

std::string FormatParameterCppDeclaration(const KernelParameter& param) {
  std::string decl;
  if (param.is_pointer()) {
    if (param.is_const()) {
      decl += "const ";
    }
    decl += CudaTypeName(param.type);
    decl += "* ";
  } else {
    decl += CudaTypeName(param.type);
    decl += " ";
  }
  decl += param.name;
  return decl;
}

std::string GenerateKernelStubAndLauncher(const TranslatedKernel& kernel) {
  std::string code;
  code += "static void __muda_host_stub_" + kernel.name + "() {}\n";
  code += "static inline void __muda_launch_" + kernel.name +
          "(dim3 __muda_grid, dim3 __muda_block, "
          "size_t __muda_shared, cudaStream_t __muda_stream";
  for (const KernelParameter& param : kernel.parameters) {
    code += ", ";
    code += FormatParameterCppDeclaration(param);
  }
  code += ") {\n";
  if (kernel.parameters.empty()) {
    code += "  (void)cudaLaunchKernel(reinterpret_cast<const void*>(&__muda_host_stub_" +
            kernel.name + "), __muda_grid, __muda_block, nullptr, __muda_shared, __muda_stream);\n";
  } else {
    code += "  void* __muda_args[" + std::to_string(kernel.parameters.size()) + "] = {";
    for (std::size_t index = 0; index < kernel.parameters.size(); ++index) {
      if (index > 0) {
        code += ", ";
      }
      code += "const_cast<void*>(static_cast<const void*>(&" + kernel.parameters[index].name + "))";
    }
    code += "};\n";
    code += "  (void)cudaLaunchKernel(reinterpret_cast<const void*>(&__muda_host_stub_" +
            kernel.name +
            "), __muda_grid, __muda_block, __muda_args, __muda_shared, __muda_stream);\n";
  }
  code += "}\n";
  return code;
}

std::string GenerateEmbeddedRegistration(std::span<const TranslatedKernel> kernels,
                                         std::span<const std::uint8_t> metallib_bytes) {
  std::string code;
  code += "\nnamespace {\n";
  if (!metallib_bytes.empty()) {
    code += "alignas(16) constexpr unsigned char __muda_embedded_metallib[] = {";
    for (std::size_t index = 0; index < metallib_bytes.size(); ++index) {
      if (index % 16 == 0) {
        code += "\n  ";
      } else {
        code += " ";
      }
      code += std::to_string(static_cast<unsigned int>(metallib_bytes[index])) + "u,";
    }
    code += "\n};\n";
  }

  for (const TranslatedKernel& kernel : kernels) {
    if (!kernel.parameters.empty()) {
      code += "constexpr mudaKernelParamDesc __muda_params_" + kernel.name + "[] = {\n";
      for (const KernelParameter& param : kernel.parameters) {
        const char* kind_str =
            param.is_pointer() ? "mudaKernelParamBuffer" : "mudaKernelParamBytes";
        code += "  {" + std::string(kind_str) + ", " + std::to_string(param.byte_size()) + "u},\n";
      }
      code += "};\n";
    }
  }

  if (!metallib_bytes.empty()) {
    code += "__attribute__((constructor)) void __muda_register_all_kernels() {\n";
    for (const TranslatedKernel& kernel : kernels) {
      const std::string& msl_entry = kernel.msl_name.empty() ? kernel.name : kernel.msl_name;
      const std::string params_ptr =
          kernel.parameters.empty() ? "nullptr" : ("__muda_params_" + kernel.name);
      code +=
          "  (void)mudaRegisterKernelFromData(reinterpret_cast<const void*>(&__muda_host_stub_" +
          kernel.name + "), __muda_embedded_metallib, sizeof(__muda_embedded_metallib), \"" +
          msl_entry + "\", " + params_ptr + ", " + std::to_string(kernel.parameters.size()) +
          "u);\n";
    }
    code += "}\n";
  }
  code += "}  // namespace\n";
  return code;
}

const TranslatedKernel* FindKernelByName(std::span<const TranslatedKernel> kernels,
                                         std::string_view name) noexcept {
  for (const TranslatedKernel& kernel : kernels) {
    if (kernel.name == name) {
      return &kernel;
    }
  }
  return nullptr;
}

DiagnosticCode StripDeviceAndRewriteKernels(std::string_view source,
                                            std::span<const TranslatedKernel> kernels,
                                            std::string* out_rewritten) {
  std::string output;
  output.reserve(source.size() + 1024);

  std::size_t cursor = 0;
  while (cursor < source.size()) {
    // Handle comments and string literals verbatim
    if (source[cursor] == '/' && cursor + 1 < source.size() && source[cursor + 1] == '/') {
      const std::size_t newline = source.find('\n', cursor + 2);
      if (newline == std::string_view::npos) {
        output.append(source.substr(cursor));
        break;
      }
      output.append(source.substr(cursor, newline - cursor + 1));
      cursor = newline + 1;
      continue;
    }
    if (source[cursor] == '/' && cursor + 1 < source.size() && source[cursor + 1] == '*') {
      const std::size_t end_comment = source.find("*/", cursor + 2);
      if (end_comment == std::string_view::npos) {
        output.append(source.substr(cursor));
        break;
      }
      output.append(source.substr(cursor, end_comment - cursor + 2));
      cursor = end_comment + 2;
      continue;
    }
    if (source[cursor] == '"' || source[cursor] == '\'') {
      const char quote = source[cursor];
      output.push_back(quote);
      ++cursor;
      while (cursor < source.size()) {
        const char ch = source[cursor++];
        output.push_back(ch);
        if (ch == '\\' && cursor < source.size()) {
          output.push_back(source[cursor++]);
          continue;
        }
        if (ch == quote) {
          break;
        }
      }
      continue;
    }

    if (IsIdentifierStart(source[cursor])) {
      const std::size_t word_start = cursor;
      ++cursor;
      while (cursor < source.size() && IsIdentifierChar(source[cursor])) {
        ++cursor;
      }
      const std::string_view word = source.substr(word_start, cursor - word_start);

      if (word == "__global__") {
        const std::size_t open_paren = source.find('(', cursor);
        if (open_paren == std::string_view::npos) {
          return DiagnosticCode::kMalformedKernelDeclaration;
        }
        const std::size_t close_paren = FindMatching(source, open_paren, '(', ')');
        if (close_paren == std::string_view::npos) {
          return DiagnosticCode::kUnterminatedParameterList;
        }
        std::size_t name_end = open_paren;
        while (name_end > cursor && IsSpace(source[name_end - 1])) {
          --name_end;
        }
        std::size_t name_start = name_end;
        while (name_start > cursor && IsIdentifierChar(source[name_start - 1])) {
          --name_start;
        }
        const std::string_view kernel_name = source.substr(name_start, name_end - name_start);
        std::size_t body_open = close_paren + 1;
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
        const TranslatedKernel* kernel = FindKernelByName(kernels, kernel_name);
        if (kernel != nullptr) {
          output += GenerateKernelStubAndLauncher(*kernel);
        }
        cursor = body_close + 1;
        continue;
      }

      if (word == "__device__" || word == "__host__" || word == "__forceinline__" ||
          word == "__noinline__" || word == "__restrict__" || word == "__restrict") {
        // Strip CUDA qualifier for host compilation; if __device__ function is not __host__,
        // make it [[maybe_unused]] inline on host or strip its body if device-only.
        if (word == "__forceinline__") {
          output += "inline";
        }
        continue;
      }

      // Check if word is followed by <<<
      std::size_t lookahead = cursor;
      while (lookahead < source.size() && IsSpace(source[lookahead])) {
        ++lookahead;
      }
      if (lookahead + 2 < source.size() && source[lookahead] == '<' &&
          source[lookahead + 1] == '<' && source[lookahead + 2] == '<') {
        const std::size_t close_triple = FindMatchingTripleAngle(source, lookahead);
        if (close_triple == std::string_view::npos) {
          return DiagnosticCode::kMalformedLaunchConfiguration;
        }
        const std::string_view config_text =
            source.substr(lookahead + 3, close_triple - (lookahead + 3));
        const std::vector<std::string_view> config_parts = SplitTopLevelCommas(config_text);
        if (config_parts.size() < 2 || config_parts.size() > 4) {
          return DiagnosticCode::kMalformedLaunchConfiguration;
        }
        for (const std::string_view part : config_parts) {
          if (part.empty()) {
            return DiagnosticCode::kMalformedLaunchConfiguration;
          }
        }

        std::size_t call_open = close_triple + 3;
        while (call_open < source.size() && IsSpace(source[call_open])) {
          ++call_open;
        }
        if (call_open >= source.size() || source[call_open] != '(') {
          return DiagnosticCode::kMalformedLaunchConfiguration;
        }
        const std::size_t call_close = FindMatching(source, call_open, '(', ')');
        if (call_close == std::string_view::npos) {
          return DiagnosticCode::kMalformedLaunchConfiguration;
        }
        const std::string_view args_text =
            Trim(source.substr(call_open + 1, call_close - call_open - 1));

        const std::string_view grid_expr = config_parts[0];
        const std::string_view block_expr = config_parts[1];
        const std::string_view shared_expr =
            config_parts.size() >= 3 ? config_parts[2] : std::string_view{"0u"};
        const std::string_view stream_expr =
            config_parts.size() == 4 ? config_parts[3] : std::string_view{"nullptr"};

        output += "__muda_launch_";
        output.append(word);
        output += "(dim3(";
        output.append(grid_expr);
        output += "), dim3(";
        output.append(block_expr);
        output += "), static_cast<size_t>(";
        output.append(shared_expr);
        output += "), static_cast<cudaStream_t>(";
        output.append(stream_expr);
        output += ")";
        if (!args_text.empty()) {
          output += ", ";
          output.append(args_text);
        }
        output.push_back(')');
        cursor = call_close + 1;
        continue;
      }

      output.append(word);
      continue;
    }

    output.push_back(source[cursor]);
    ++cursor;
  }

  *out_rewritten = std::move(output);
  return DiagnosticCode::kOk;
}

}  // namespace

HostRewriteResult RewriteHostCudaToCpp(std::string_view cuda_source,
                                       std::span<const TranslatedKernel> kernels,
                                       std::span<const std::uint8_t> metallib_bytes) {
  if (kernels.empty()) {
    return {DiagnosticCode::kMissingGlobalKernel, {}};
  }

  std::string body_rewritten;
  const DiagnosticCode status = StripDeviceAndRewriteKernels(cuda_source, kernels, &body_rewritten);
  if (status != DiagnosticCode::kOk) {
    return {status, {}};
  }

  std::string full_cpp;
  full_cpp.reserve(body_rewritten.size() + metallib_bytes.size() * 5 + 1024);
  full_cpp += "#include <cuda_runtime.h>\n";
  full_cpp += "#include <cstddef>\n";
  full_cpp += body_rewritten;
  full_cpp += GenerateEmbeddedRegistration(kernels, metallib_bytes);

  return {DiagnosticCode::kOk, std::move(full_cpp)};
}

}  // namespace muda::compiler

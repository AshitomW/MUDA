/**
 * @file nvrtc.cpp
 * @brief Clean-room implementation of NVRTC runtime compilation backed by MUDA's CUDA-to-MSL
 * engine.
 */

#include "nvrtc.h"

#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "handle_table.hpp"
#include "muda/compiler/translator.hpp"
#include "muda/compiler/types.hpp"

namespace {

struct NvrtcProgramState {
  std::mutex mutex;
  std::string name;
  std::string source;
  std::unordered_map<std::string, std::string> virtual_headers;
  std::unordered_map<std::string, std::string> lowered_names;
  std::string log;
  std::string emitted_ptx;
  bool compiled = false;
  bool compile_succeeded = false;
};

using NvrtcHandleTable =
    muda::runtime::cudart::HandleTable<NvrtcProgramState,
                                       muda::runtime::cudart::HandleKind::kNvrtcProgram>;

NvrtcHandleTable& ProgramTable() {
  static NvrtcHandleTable table;
  return table;
}

std::string InlineVirtualHeaders(std::string_view src,
                                 const std::unordered_map<std::string, std::string>& headers) {
  if (headers.empty()) {
    return std::string(src);
  }
  std::string out;
  for (const auto& [inc_name, content] : headers) {
    out += "// inlined virtual header: " + inc_name + "\n";
    out += content;
    out += "\n";
  }
  std::size_t pos = 0;
  while (pos < src.size()) {
    const std::size_t nl = src.find('\n', pos);
    const std::size_t end = (nl == std::string_view::npos) ? src.size() : nl;
    const std::string_view line = src.substr(pos, end - pos);

    bool matched_virtual_include = false;
    for (const auto& [inc_name, _] : headers) {
      const std::string quoted = "\"" + inc_name + "\"";
      const std::string angled = "<" + inc_name + ">";
      if (line.find("#include") != std::string_view::npos &&
          (line.find(quoted) != std::string_view::npos ||
           line.find(angled) != std::string_view::npos)) {
        matched_virtual_include = true;
        break;
      }
    }
    if (!matched_virtual_include) {
      out.append(line);
      if (nl != std::string_view::npos) {
        out.push_back('\n');
      }
    }
    if (nl == std::string_view::npos) {
      break;
    }
    pos = nl + 1;
  }
  return out;
}

std::string ExtractCleanIdentifier(std::string_view expr) {
  std::string_view trimmed = expr;
  while (!trimmed.empty() && (trimmed.front() == ' ' || trimmed.front() == '\t' ||
                              trimmed.front() == '&' || trimmed.front() == '(')) {
    trimmed.remove_prefix(1);
  }
  while (!trimmed.empty() &&
         (trimmed.back() == ' ' || trimmed.back() == '\t' || trimmed.back() == ')')) {
    trimmed.remove_suffix(1);
  }
  return std::string(trimmed);
}

}  // namespace

extern "C" {

const char* nvrtcGetErrorString(nvrtcResult result) {
  switch (result) {
    case NVRTC_SUCCESS:
      return "NVRTC_SUCCESS";
    case NVRTC_ERROR_OUT_OF_MEMORY:
      return "NVRTC_ERROR_OUT_OF_MEMORY";
    case NVRTC_ERROR_PROGRAM_CREATION_FAILURE:
      return "NVRTC_ERROR_PROGRAM_CREATION_FAILURE";
    case NVRTC_ERROR_INVALID_INPUT:
      return "NVRTC_ERROR_INVALID_INPUT";
    case NVRTC_ERROR_INVALID_PROGRAM:
      return "NVRTC_ERROR_INVALID_PROGRAM";
    case NVRTC_ERROR_INVALID_OPTION:
      return "NVRTC_ERROR_INVALID_OPTION";
    case NVRTC_ERROR_COMPILATION:
      return "NVRTC_ERROR_COMPILATION";
    case NVRTC_ERROR_BUILTIN_OPERATION_FAILURE:
      return "NVRTC_ERROR_BUILTIN_OPERATION_FAILURE";
    case NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION:
      return "NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION";
    case NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION:
      return "NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION";
    case NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID:
      return "NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID";
    case NVRTC_ERROR_INTERNAL_ERROR:
      return "NVRTC_ERROR_INTERNAL_ERROR";
  }
  return "NVRTC_ERROR_UNKNOWN";
}

nvrtcResult nvrtcVersion(int* major, int* minor) {
  if (major == nullptr || minor == nullptr) {
    return NVRTC_ERROR_INVALID_INPUT;
  }
  *major = 12;
  *minor = 2;
  return NVRTC_SUCCESS;
}

nvrtcResult nvrtcCreateProgram(nvrtcProgram* prog, const char* src, const char* name,
                               int numHeaders, const char* const* headers,
                               const char* const* includeNames) {
  if (prog == nullptr || src == nullptr || numHeaders < 0) {
    return NVRTC_ERROR_INVALID_INPUT;
  }
  if (numHeaders > 0 && (headers == nullptr || includeNames == nullptr)) {
    return NVRTC_ERROR_INVALID_INPUT;
  }

  auto state = std::make_shared<NvrtcProgramState>();
  state->name = (name != nullptr && name[0] != '\0') ? name : "default_program";
  state->source = src;

  for (int i = 0; i < numHeaders; ++i) {
    if (headers[i] == nullptr || includeNames[i] == nullptr) {
      return NVRTC_ERROR_INVALID_INPUT;
    }
    state->virtual_headers[includeNames[i]] = headers[i];
  }

  *prog = ProgramTable().Insert<nvrtcProgram>(std::move(state));
  return NVRTC_SUCCESS;
}

nvrtcResult nvrtcDestroyProgram(nvrtcProgram* prog) {
  if (prog == nullptr || *prog == nullptr) {
    return NVRTC_ERROR_INVALID_PROGRAM;
  }
  auto removed = ProgramTable().Remove(*prog);
  if (removed == nullptr) {
    return NVRTC_ERROR_INVALID_PROGRAM;
  }
  *prog = nullptr;
  return NVRTC_SUCCESS;
}

nvrtcResult nvrtcCompileProgram(nvrtcProgram prog, int numOptions, const char* const* options) {
  auto state = ProgramTable().Lookup(prog);
  if (state == nullptr) {
    return NVRTC_ERROR_INVALID_PROGRAM;
  }
  if (numOptions < 0 || (numOptions > 0 && options == nullptr)) {
    return NVRTC_ERROR_INVALID_OPTION;
  }

  std::lock_guard<std::mutex> lock(state->mutex);
  muda::compiler::TranslationOptions trans_opts;
  const char* env_fp64 = std::getenv("MUDA_FP64");
  if (env_fp64 != nullptr && env_fp64[0] != '\0') {
    trans_opts.fp64_policy = muda::compiler::ParseFp64Policy(env_fp64);
  }

  std::string macro_preamble;
  for (int i = 0; i < numOptions; ++i) {
    if (options[i] == nullptr) {
      return NVRTC_ERROR_INVALID_OPTION;
    }
    const std::string_view opt = options[i];
    if (opt.starts_with("--muda-fp64=")) {
      trans_opts.fp64_policy =
          muda::compiler::ParseFp64Policy(opt.substr(std::string_view("--muda-fp64=").size()));
      continue;
    }
    if (opt.starts_with("--fp64=")) {
      trans_opts.fp64_policy =
          muda::compiler::ParseFp64Policy(opt.substr(std::string_view("--fp64=").size()));
      continue;
    }
    if (opt.starts_with("-D") && opt.size() > 2) {
      const std::string_view def = opt.substr(2);
      const std::size_t eq = def.find('=');
      if (eq == std::string_view::npos) {
        macro_preamble += "#define " + std::string(def) + " 1\n";
      } else {
        macro_preamble += "#define " + std::string(def.substr(0, eq)) + " " +
                          std::string(def.substr(eq + 1)) + "\n";
      }
    }
  }

  const std::string full_source =
      macro_preamble + InlineVirtualHeaders(state->source, state->virtual_headers);
  const muda::compiler::TranslationResult tr =
      muda::compiler::TranslateCudaToMsl(full_source, trans_opts);
  state->compiled = true;

  if (!tr.ok()) {
    state->compile_succeeded = false;
    state->log = std::string(muda::compiler::DiagnosticId(tr.code)) + ": " +
                 std::string(muda::compiler::DiagnosticMessage(tr.code)) + "\n";
    return NVRTC_ERROR_COMPILATION;
  }

  state->compile_succeeded = true;
  state->log = "NVRTC compilation succeeded (" + std::to_string(tr.kernels.size()) + " kernels)\n";

  // Emit an NVRTC PTX envelope carrying the validated CUDA source so cuModuleLoadData
  // translates and compiles all kernels directly in-memory via Metal.
  std::string ptx = ".version 7.8\n.target sm_86\n.address_size 64\n";
  if (trans_opts.fp64_policy != muda::compiler::Fp64Policy::kOff) {
    ptx += "// MUDA_FP64_POLICY=" +
           std::string(muda::compiler::Fp64PolicyName(trans_opts.fp64_policy)) + "\n";
  }
  for (const auto& k : tr.kernels) {
    ptx += "// .entry " + k.name + "\n";
    for (auto& [expr, lowered] : state->lowered_names) {
      if (ExtractCleanIdentifier(expr) == k.name) {
        lowered = k.name;
      }
    }
  }
  ptx += "// MUDA_NVRTC_CUDA_SOURCE\n";
  ptx += full_source;
  state->emitted_ptx = std::move(ptx);
  return NVRTC_SUCCESS;
}

nvrtcResult nvrtcGetPTXSize(nvrtcProgram prog, size_t* ptxSizeRet) {
  if (ptxSizeRet == nullptr) {
    return NVRTC_ERROR_INVALID_INPUT;
  }
  auto state = ProgramTable().Lookup(prog);
  if (state == nullptr) {
    return NVRTC_ERROR_INVALID_PROGRAM;
  }
  std::lock_guard<std::mutex> lock(state->mutex);
  if (!state->compiled || !state->compile_succeeded) {
    return NVRTC_ERROR_INVALID_PROGRAM;
  }
  *ptxSizeRet = state->emitted_ptx.size() + 1;
  return NVRTC_SUCCESS;
}

nvrtcResult nvrtcGetPTX(nvrtcProgram prog, char* ptx) {
  if (ptx == nullptr) {
    return NVRTC_ERROR_INVALID_INPUT;
  }
  auto state = ProgramTable().Lookup(prog);
  if (state == nullptr) {
    return NVRTC_ERROR_INVALID_PROGRAM;
  }
  std::lock_guard<std::mutex> lock(state->mutex);
  if (!state->compiled || !state->compile_succeeded) {
    return NVRTC_ERROR_INVALID_PROGRAM;
  }
  std::memcpy(ptx, state->emitted_ptx.c_str(), state->emitted_ptx.size() + 1);
  return NVRTC_SUCCESS;
}

nvrtcResult nvrtcGetProgramLogSize(nvrtcProgram prog, size_t* logSizeRet) {
  if (logSizeRet == nullptr) {
    return NVRTC_ERROR_INVALID_INPUT;
  }
  auto state = ProgramTable().Lookup(prog);
  if (state == nullptr) {
    return NVRTC_ERROR_INVALID_PROGRAM;
  }
  std::lock_guard<std::mutex> lock(state->mutex);
  *logSizeRet = state->log.size() + 1;
  return NVRTC_SUCCESS;
}

nvrtcResult nvrtcGetProgramLog(nvrtcProgram prog, char* log) {
  if (log == nullptr) {
    return NVRTC_ERROR_INVALID_INPUT;
  }
  auto state = ProgramTable().Lookup(prog);
  if (state == nullptr) {
    return NVRTC_ERROR_INVALID_PROGRAM;
  }
  std::lock_guard<std::mutex> lock(state->mutex);
  std::memcpy(log, state->log.c_str(), state->log.size() + 1);
  return NVRTC_SUCCESS;
}

nvrtcResult nvrtcAddNameExpression(nvrtcProgram prog, const char* name_expression) {
  if (name_expression == nullptr || name_expression[0] == '\0') {
    return NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID;
  }
  auto state = ProgramTable().Lookup(prog);
  if (state == nullptr) {
    return NVRTC_ERROR_INVALID_PROGRAM;
  }
  std::lock_guard<std::mutex> lock(state->mutex);
  if (state->compiled) {
    return NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION;
  }
  state->lowered_names[name_expression] = ExtractCleanIdentifier(name_expression);
  return NVRTC_SUCCESS;
}

nvrtcResult nvrtcGetLoweredName(nvrtcProgram prog, const char* name_expression,
                                const char** lowered_name) {
  if (name_expression == nullptr || lowered_name == nullptr) {
    return NVRTC_ERROR_INVALID_INPUT;
  }
  auto state = ProgramTable().Lookup(prog);
  if (state == nullptr) {
    return NVRTC_ERROR_INVALID_PROGRAM;
  }
  std::lock_guard<std::mutex> lock(state->mutex);
  if (!state->compiled) {
    return NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION;
  }
  const auto it = state->lowered_names.find(name_expression);
  if (it == state->lowered_names.end()) {
    return NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID;
  }
  *lowered_name = it->second.c_str();
  return NVRTC_SUCCESS;
}

}  // extern "C"

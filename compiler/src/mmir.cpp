/**
 * @file mmir.cpp
 * @brief Implements MMIR verification and deterministic Metal Shading Language emission.
 */

#include "muda/compiler/mmir.hpp"

#include <cctype>
#include <string>
#include <string_view>

namespace muda::compiler::mmir {
namespace {

bool IsIdentifierStart(char value) noexcept {
  return std::isalpha(static_cast<unsigned char>(value)) != 0 || value == '_';
}

bool IsIdentifierChar(char value) noexcept {
  return std::isalnum(static_cast<unsigned char>(value)) != 0 || value == '_';
}

bool IsSpace(char value) noexcept { return std::isspace(static_cast<unsigned char>(value)) != 0; }

bool ContainsCallTo(std::string_view body, std::string_view fn_name) noexcept {
  for (std::size_t index = 0; index < body.size();) {
    if (!IsIdentifierStart(body[index])) {
      ++index;
      continue;
    }
    const std::size_t start = index;
    ++index;
    while (index < body.size() && IsIdentifierChar(body[index])) {
      ++index;
    }
    const std::string_view word = body.substr(start, index - start);
    if (word == fn_name) {
      std::size_t next = index;
      while (next < body.size() && IsSpace(body[next])) {
        ++next;
      }
      if (next < body.size() && body[next] == '(') {
        return true;
      }
    }
  }
  return false;
}

void EmitDeviceFunctionMsl(const DeviceFunction& fn, std::string* out) {
  *out += "inline ";
  if (fn.returns_void) {
    *out += "void ";
  } else {
    *out += MslTypeName(fn.return_type);
    out->push_back(' ');
  }
  *out += fn.name;
  *out += "(";
  for (std::size_t index = 0; index < fn.parameters.size(); ++index) {
    if (index > 0) {
      *out += ", ";
    }
    const KernelParameter& param = fn.parameters[index];
    if (param.is_pointer()) {
      *out += AddressSpaceName(param.address_space);
      out->push_back(' ');
      if (param.is_const()) {
        *out += "const ";
      }
      *out += param.metal_type();
      *out += "* ";
    } else {
      if (param.is_const()) {
        // Value parameter
      }
      *out += param.metal_type();
      out->push_back(' ');
    }
    *out += param.name;
  }
  *out += ")\n{\n  ";
  *out += fn.lowered_body;
  *out += "\n}\n\n";
}

void EmitKernelFunctionMsl(const KernelFunction& kernel, std::string* out) {
  const std::string& emitted_name = kernel.msl_name.empty() ? kernel.name : kernel.msl_name;
  *out += "kernel void ";
  *out += emitted_name;
  *out += "(\n";

  for (const KernelParameter& parameter : kernel.parameters) {
    *out += "    ";
    *out += AddressSpaceName(parameter.address_space);
    out->push_back(' ');
    if (parameter.is_pointer()) {
      if (parameter.is_const()) {
        *out += "const ";
      }
      *out += parameter.metal_type();
      *out += "* ";
    } else {
      *out += parameter.metal_type();
      *out += "& ";
    }
    *out += parameter.name;
    *out += " [[buffer(";
    *out += std::to_string(parameter.buffer_index);
    *out += ")]],\n";
  }

  for (const SharedAllocation& shared : kernel.shared_allocations) {
    if (shared.is_dynamic) {
      *out += "    threadgroup ";
      *out += MslTypeName(shared.element_type);
      *out += "* ";
      *out += shared.name;
      *out += " [[threadgroup(0)]],\n";
      break;
    }
  }

  *out += "    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],\n";
  *out += "    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],\n";
  *out += "    uint3 threads_per_threadgroup [[threads_per_threadgroup]],\n";
  *out += "    uint3 threadgroups_per_grid [[threadgroups_per_grid]])\n";
  *out += "{\n";

  for (const SharedAllocation& shared : kernel.shared_allocations) {
    if (!shared.is_dynamic) {
      *out += "  threadgroup ";
      *out += MslTypeName(shared.element_type);
      out->push_back(' ');
      *out += shared.name;
      out->push_back('[');
      *out += shared.count_expression;
      *out += "];\n";
    }
  }

  *out += "  ";
  *out += kernel.lowered_body;
  *out += "\n}\n";
}

}  // namespace

DiagnosticCode VerifyModule(const Module& module) noexcept {
  if (module.kernels.empty()) {
    return DiagnosticCode::kMissingGlobalKernel;
  }

  for (const DeviceFunction& fn : module.device_functions) {
    if (fn.name.empty()) {
      return DiagnosticCode::kMissingKernelName;
    }
    if (ContainsCallTo(fn.lowered_body, fn.name)) {
      return DiagnosticCode::kRecursionUnsupported;
    }
  }

  for (const KernelFunction& kernel : module.kernels) {
    if (kernel.name.empty()) {
      return DiagnosticCode::kMissingKernelName;
    }
    if (ContainsCallTo(kernel.lowered_body, kernel.name)) {
      return DiagnosticCode::kRecursionUnsupported;
    }
    std::size_t dynamic_shared_count = 0;
    for (const SharedAllocation& shared : kernel.shared_allocations) {
      if (shared.name.empty()) {
        return DiagnosticCode::kInvalidSharedMemoryDeclaration;
      }
      if (shared.is_dynamic) {
        ++dynamic_shared_count;
      } else if (shared.count_expression.empty()) {
        return DiagnosticCode::kInvalidSharedMemoryDeclaration;
      }
    }
    if (dynamic_shared_count > 1) {
      return DiagnosticCode::kInvalidSharedMemoryDeclaration;
    }
  }

  return DiagnosticCode::kOk;
}

std::string EmitMslModule(const Module& module) {
  std::string msl;
  msl.reserve(1024);
  msl += "#include <metal_stdlib>\n";
  msl += "using namespace metal;\n\n";

  for (const DeviceFunction& fn : module.device_functions) {
    EmitDeviceFunctionMsl(fn, &msl);
  }

  for (std::size_t index = 0; index < module.kernels.size(); ++index) {
    if (index > 0) {
      msl.push_back('\n');
    }
    EmitKernelFunctionMsl(module.kernels[index], &msl);
  }
  return msl;
}

std::string EmitMslKernel(const Module& module, const KernelFunction& kernel) {
  std::string msl;
  msl.reserve(1024);
  msl += "#include <metal_stdlib>\n";
  msl += "using namespace metal;\n\n";

  for (const DeviceFunction& fn : module.device_functions) {
    EmitDeviceFunctionMsl(fn, &msl);
  }
  EmitKernelFunctionMsl(kernel, &msl);
  return msl;
}

}  // namespace muda::compiler::mmir

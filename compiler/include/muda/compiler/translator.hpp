/**
 * @file translator.hpp
 * @brief Converts the supported CUDA kernel subset into Metal Shading Language.
 */

#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

#include "muda/compiler/types.hpp"

namespace muda::compiler {

struct KernelParameter {
  std::string name;
  DataType type = {ScalarType::kFloat32, VectorWidth::k1};
  ParameterBindingKind binding_kind = ParameterBindingKind::kConstantValue;
  AddressSpace address_space = AddressSpace::kConstant;
  AccessQualifier access = AccessQualifier::kReadOnly;
  std::size_t buffer_index = 0;

  [[nodiscard]] constexpr bool is_pointer() const noexcept {
    return binding_kind == ParameterBindingKind::kBufferPointer;
  }

  [[nodiscard]] constexpr bool is_const() const noexcept {
    return access == AccessQualifier::kReadOnly;
  }

  [[nodiscard]] constexpr std::string_view metal_type() const noexcept { return MslTypeName(type); }

  [[nodiscard]] constexpr std::size_t byte_size() const noexcept {
    return is_pointer() ? sizeof(void*) : ByteSize(type);
  }
};

struct TranslatedKernel {
  std::string name;
  std::vector<KernelParameter> parameters;
  std::string msl;
};

struct TranslationResult {
  DiagnosticCode code = DiagnosticCode::kMissingGlobalKernel;
  TranslatedKernel kernel;

  [[nodiscard]] constexpr bool ok() const noexcept { return code == DiagnosticCode::kOk; }

  [[nodiscard]] constexpr std::string_view diagnostic_id() const noexcept {
    return DiagnosticId(code);
  }

  [[nodiscard]] constexpr std::string_view error() const noexcept {
    return code == DiagnosticCode::kOk ? std::string_view{} : DiagnosticMessage(code);
  }
};

[[nodiscard]] TranslationResult TranslateCudaToMsl(std::string_view source);

}  // namespace muda::compiler

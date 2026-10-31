/**
 * @file type_table.hpp
 * @brief CUDA-to-Metal type and intrinsic lookup tables.
 */

#pragma once

#include <array>
#include <optional>
#include <string_view>

#include "muda/compiler/mmir.hpp"

namespace muda::compiler {

struct TypeMappingEntry {
  std::string_view cuda_spelling;
  DataType type;
};

struct BuiltinMappingEntry {
  std::string_view cuda_spelling;
  std::string_view msl_replacement;
};

extern const std::array<TypeMappingEntry, 45> kTypeTable;
extern const std::array<TypeMappingEntry, 4> kFloatTypeTable;
extern const std::array<BuiltinMappingEntry, 4> kBuiltinTable;
extern const std::array<BuiltinMappingEntry, 18> kMathIntrinsicTable;
extern const std::array<std::string_view, 18> kUnsupportedConstructs;

[[nodiscard]] std::optional<DataType> ResolveCudaType(std::string_view spelling) noexcept;
[[nodiscard]] std::string_view BuiltinReplacement(std::string_view word) noexcept;
[[nodiscard]] std::string_view MathIntrinsicReplacement(std::string_view word) noexcept;

}  // namespace muda::compiler


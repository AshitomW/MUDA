/**
 * @file type_inference.hpp
 * @brief Address space and type inference engine for PTX registers and memory operations.
 */

#pragma once

#include <string>
#include <string_view>
#include <unordered_map>

#include "muda/compiler/ptx/ast.hpp"
#include "muda/compiler/types.hpp"

namespace muda::compiler::ptx {

struct InferredType {
  DataType data_type = {ScalarType::kFloat32, VectorWidth::k1};
  AddressSpace address_space = AddressSpace::kThread;
  bool is_pointer = false;
};

class TypeInference final {
 public:
  TypeInference() = default;

  /**
   * @brief Analyzes a PTX function and infers data types and address spaces for registers.
   */
  [[nodiscard]] static TypeInference Analyze(const Function& fn);

  [[nodiscard]] const InferredType* Lookup(std::string_view reg_name) const noexcept;
  [[nodiscard]] AddressSpace GetAddressSpace(std::string_view reg_name) const noexcept;
  [[nodiscard]] DataType GetDataType(std::string_view reg_name) const noexcept;
  [[nodiscard]] bool IsPointer(std::string_view reg_name) const noexcept;

 private:
  void SeedDeclarations(const Function& fn);
  void PropagateInstructions(const Function& fn);

  std::unordered_map<std::string_view, InferredType> reg_types_;
};

}  // namespace muda::compiler::ptx

/**
 * @file mmir.hpp
 * @brief Typed MUDA Metal Intermediate Representation (MMIR) for device functions and kernels.
 */

#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "muda/compiler/translator.hpp"
#include "muda/compiler/types.hpp"

namespace muda::compiler::mmir {

struct SharedAllocation {
  std::string name;
  DataType element_type = {ScalarType::kFloat32, VectorWidth::k1};
  std::string count_expression;
  bool is_dynamic = false;
};

struct DeviceFunction {
  std::string name;
  DataType return_type = {ScalarType::kFloat32, VectorWidth::k1};
  bool returns_void = false;
  std::vector<KernelParameter> parameters;
  std::string lowered_body;
};

struct KernelFunction {
  std::string name;
  std::string msl_name;
  std::vector<KernelParameter> parameters;
  std::vector<SharedAllocation> shared_allocations;
  std::string lowered_body;

  [[nodiscard]] bool uses_dynamic_shared_memory() const noexcept {
    for (const SharedAllocation& alloc : shared_allocations) {
      if (alloc.is_dynamic) {
        return true;
      }
    }
    return false;
  }
};

struct Module {
  std::vector<DeviceFunction> device_functions;
  std::vector<KernelFunction> kernels;
};

/**
 * @brief Verifies structural and semantic invariants of an MMIR Module.
 */
[[nodiscard]] DiagnosticCode VerifyModule(const Module& module) noexcept;

/**
 * @brief Emits deterministic Metal Shading Language source for a verified MMIR Module.
 */
[[nodiscard]] std::string EmitMslModule(const Module& module);

/**
 * @brief Emits standalone Metal Shading Language source for a single KernelFunction (and any
 *        device functions in the module).
 */
[[nodiscard]] std::string EmitMslKernel(const Module& module, const KernelFunction& kernel);

}  // namespace muda::compiler::mmir

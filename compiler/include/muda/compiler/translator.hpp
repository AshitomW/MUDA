/**
 * @file translator.hpp
 * @brief Converts the supported CUDA kernel subset into Metal Shading Language.
 */

#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace muda::compiler {

struct KernelParameter {
  std::string name;
  std::string cuda_type;
  std::string metal_type;
  bool is_pointer = false;
  bool is_const = false;
  std::size_t buffer_index = 0;
};

struct TranslatedKernel {
  std::string name;
  std::vector<KernelParameter> parameters;
  std::string msl;
};

struct TranslationResult {
  bool ok = false;
  TranslatedKernel kernel;
  std::string error;
};

[[nodiscard]] TranslationResult TranslateCudaToMsl(std::string_view source);

}

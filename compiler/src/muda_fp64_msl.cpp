/**
 * @file muda_fp64_msl.cpp
 * @brief Locates and reads the MSL Double-Single 64-bit floating point emulation shader.
 */

#include "muda_fp64_msl.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace muda::compiler {
namespace {

std::string LoadMudaFp64MetalSource() {
  const char* env_path = std::getenv("MUDA_FP64_METAL_PATH");
  std::vector<std::filesystem::path> candidates;
  if (env_path != nullptr && env_path[0] != '\0') {
    candidates.emplace_back(env_path);
  }
#ifdef MUDA_DEFAULT_RUNTIME_INCLUDE_DIR
  candidates.emplace_back(std::filesystem::path(MUDA_DEFAULT_RUNTIME_INCLUDE_DIR) / "muda" /
                          "muda_fp64.metal");
#endif
  candidates.emplace_back("runtime/include/muda/muda_fp64.metal");
  candidates.emplace_back("../runtime/include/muda/muda_fp64.metal");
  candidates.emplace_back("../../runtime/include/muda/muda_fp64.metal");

  for (const auto& path : candidates) {
    std::ifstream file(path);
    if (file.is_open()) {
      std::stringstream ss;
      ss << file.rdbuf();
      return ss.str();
    }
  }
  return {};
}

}  // namespace

std::string_view GetMudaFp64MslPreamble(Fp64Policy policy) noexcept {
  if (policy == Fp64Policy::kIeee64) {
    static const std::string kIeee64Preamble =
        "#define MUDA_FP64_IEEE 1\n" + LoadMudaFp64MetalSource();
    return kIeee64Preamble;
  }
  static const std::string kFast48Preamble = LoadMudaFp64MetalSource();
  return kFast48Preamble;
}

}  // namespace muda::compiler

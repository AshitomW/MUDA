/**
 * @file toolchain.cpp
 * @brief Locates the Metal compiler and library tools.
 */

#include "muda/compiler/toolchain.hpp"

#include "process.hpp"

namespace muda::compiler {

bool MetalToolchainAvailable() {
  return detail::RunProcess({"xcrun", "-f", "metal"}, true) == 0 &&
         detail::RunProcess({"xcrun", "-f", "metallib"}, true) == 0;
}

}

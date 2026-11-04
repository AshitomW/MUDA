/**
 * @file module_manager.hpp
 * @brief In-memory Metal JIT compilation, PTX module loading, and function resolution.
 */

#pragma once

#import <Metal/Metal.h>

#include <cstddef>
#include <string_view>

#include "cuda.h"
#include "muda/runtime/driver_types.hpp"

namespace muda::runtime::cudart {
class LaunchEngine;
}

namespace muda::runtime::driver {

class ModuleManager final {
 public:
  ModuleManager() = default;
  ~ModuleManager() = default;

  [[nodiscard]] static CUresult LoadPtx(std::string_view ptx_source, CUmodule* out_module,
                                        id<MTLDevice> device,
                                        cudart::LaunchEngine& launcher) noexcept;

  [[nodiscard]] static CUresult LoadMetallibData(const void* data, std::size_t size,
                                                 CUmodule* out_module, id<MTLDevice> device,
                                                 cudart::LaunchEngine& launcher) noexcept;

  [[nodiscard]] static CUresult GetFunction(CUfunction* hfunc, CUmodule hmod,
                                            const char* name) noexcept;

  [[nodiscard]] static CUresult Unload(CUmodule hmod) noexcept;
};

}  // namespace muda::runtime::driver

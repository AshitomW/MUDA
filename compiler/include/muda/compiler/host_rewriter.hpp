/**
 * @file host_rewriter.hpp
 * @brief Lowers CUDA host source into standard C++20 with embedded metallib registration.
 */

#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <string_view>

#include "muda/compiler/translator.hpp"
#include "muda/compiler/types.hpp"

namespace muda::compiler {

struct HostRewriteResult {
  DiagnosticCode code = DiagnosticCode::kOk;
  std::string host_cpp;

  [[nodiscard]] constexpr bool ok() const noexcept { return code == DiagnosticCode::kOk; }
};

/**
 * @brief Rewrites CUDA source into standard C++20 host source that registers the embedded
 *        Metal library and lowers <<<grid, block, shared, stream>>> kernel launches to
 *        cudaLaunchKernel calls.
 */
[[nodiscard]] HostRewriteResult RewriteHostCudaToCpp(std::string_view cuda_source,
                                                     std::span<const TranslatedKernel> kernels,
                                                     std::span<const std::uint8_t> metallib_bytes);

}  // namespace muda::compiler

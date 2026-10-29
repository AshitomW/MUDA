/**
 * @file process.hpp
 * @brief Internal process execution helper.
 */

#pragma once

#include <cstdint>
#include <span>
#include <string_view>

namespace muda::compiler::detail {

enum class ProcessOutputMode : std::uint8_t {
  kInherit,
  kSilent,
};

struct ProcessResult {
  int exit_code = -1;

  [[nodiscard]] constexpr bool succeeded() const noexcept { return exit_code == 0; }
};

[[nodiscard]] ProcessResult RunProcess(std::span<const std::string_view> arguments,
                                       ProcessOutputMode output_mode = ProcessOutputMode::kInherit);

}  // namespace muda::compiler::detail

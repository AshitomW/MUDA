/**
 * @file process.hpp
 * @brief Internal process execution helper.
 */

#pragma once

#include <string>
#include <vector>

namespace muda::compiler::detail {

[[nodiscard]] int RunProcess(const std::vector<std::string>& arguments, bool silent = false);

}

/**
 * @file error_state.hpp
 * @brief Thread-local CUDA error state and error string lookup.
 */

#pragma once

#include "cuda_runtime.h"

namespace muda::runtime::cudart {

[[nodiscard]] cudaError_t RecordError(cudaError_t error) noexcept;
[[nodiscard]] cudaError_t ConsumeLastError() noexcept;
[[nodiscard]] cudaError_t PeekLastError() noexcept;

[[nodiscard]] const char* ErrorName(cudaError_t error) noexcept;
[[nodiscard]] const char* ErrorString(cudaError_t error) noexcept;

}  // namespace muda::runtime::cudart


/**
 * @file driver_internal.hpp
 * @brief Internal shared helpers and context state for CUDA Driver C API translation units.
 */

#pragma once

#include <atomic>

#include "cuda.h"
#include "cuda_runtime.h"
#include "muda/runtime/driver_types.hpp"

namespace muda::runtime::driver {

extern thread_local CUcontext g_current_ctx;
extern std::atomic<int> g_primary_ctx_refcount;
extern CUctx_st g_primary_ctx;

[[nodiscard]] CUresult MapCudaError(cudaError_t err) noexcept;
CUcontext EnsureContext() noexcept;

}  // namespace muda::runtime::driver

/**
 * @file cublas_context.hpp
 * @brief Internal cuBLAS handle state, handle table management, and scalar resolution helpers.
 */

#pragma once

#import <Metal/Metal.h>

#include <memory>
#include <mutex>
#include <string_view>

#include "cublas_v2.h"
#include "handle_table.hpp"

namespace muda::libraries::blas {

struct CublasContext {
  mutable std::mutex mutex;
  cudaStream_t stream = nullptr;
  cublasPointerMode_t pointer_mode = CUBLAS_POINTER_MODE_HOST;
  cublasMath_t math_mode = CUBLAS_DEFAULT_MATH;
  id<MTLBuffer> scratch_buffer = nil;
};

using CublasHandleTable =
    runtime::cudart::HandleTable<CublasContext, runtime::cudart::HandleKind::kCublasHandle>;

[[nodiscard]] CublasHandleTable& GetCublasHandleTable() noexcept;

[[nodiscard]] std::shared_ptr<CublasContext> LookupContext(cublasHandle_t handle) noexcept;

[[nodiscard]] id<MTLComputePipelineState> GetCublasPipeline(
    std::string_view function_name) noexcept;

[[nodiscard]] cublasStatus_t ResolveScalarFloat(const CublasContext& ctx, const float* ptr,
                                                float* out_value) noexcept;

[[nodiscard]] cublasStatus_t ResolveScalarDouble(const CublasContext& ctx, const double* ptr,
                                                 double* out_value) noexcept;

[[nodiscard]] cublasStatus_t ResolveScalarHalfAsFloat(const CublasContext& ctx, const __half* ptr,
                                                      float* out_value) noexcept;

[[nodiscard]] float HalfBitsToFloat(std::uint16_t bits) noexcept;
[[nodiscard]] std::uint16_t FloatToHalfBits(float value) noexcept;

}  // namespace muda::libraries::blas

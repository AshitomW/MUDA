/**
 * @file cublas_context.mm
 * @brief Implementation of cuBLAS context lifecycle, configuration, and scalar helpers.
 */

#include "cublas_context.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstring>

#include "muda/libraries/metal_executor.hpp"

namespace muda::libraries::blas {
namespace {

constexpr int kCublasVersion = 12000;
constexpr std::size_t kScratchBufferBytes = 4096;

}  // namespace

CublasHandleTable& GetCublasHandleTable() noexcept {
  static CublasHandleTable table;
  return table;
}

std::shared_ptr<CublasContext> LookupContext(cublasHandle_t handle) noexcept {
  return GetCublasHandleTable().Lookup(handle);
}

id<MTLComputePipelineState> GetCublasPipeline(std::string_view function_name) noexcept {
  return LibraryExecutor::Instance().GetOrLoadPipeline(
      MUDA_CUBLAS_METALLIB_PATH, MUDA_CUBLAS_METAL_SOURCE_PATH, function_name);
}

float HalfBitsToFloat(std::uint16_t bits) noexcept {
  __fp16 h;
  std::memcpy(&h, &bits, sizeof(bits));
  return static_cast<float>(h);
}

std::uint16_t FloatToHalfBits(float value) noexcept {
  const __fp16 h = static_cast<__fp16>(value);
  std::uint16_t bits = 0;
  std::memcpy(&bits, &h, sizeof(bits));
  return bits;
}

cublasStatus_t ResolveScalarFloat(const CublasContext& ctx, const float* ptr,
                                  float* out_value) noexcept {
  if (ptr == nullptr || out_value == nullptr) {
    return CUBLAS_STATUS_INVALID_VALUE;
  }
  cublasPointerMode_t mode = CUBLAS_POINTER_MODE_HOST;
  cudaStream_t stream = nullptr;
  {
    std::lock_guard<std::mutex> lock(ctx.mutex);
    mode = ctx.pointer_mode;
    stream = ctx.stream;
  }
  if (mode == CUBLAS_POINTER_MODE_HOST) {
    *out_value = *ptr;
    return CUBLAS_STATUS_SUCCESS;
  }

  ResolvedBuffer resolved;
  auto& exec = LibraryExecutor::Instance();
  if (!exec.ResolveDeviceBuffer(ptr, sizeof(float), &resolved)) {
    return CUBLAS_STATUS_INVALID_VALUE;
  }
  if (exec.SynchronizeStream(stream) != cudaSuccess) {
    return CUBLAS_STATUS_EXECUTION_FAILED;
  }
  const auto* base = static_cast<const std::uint8_t*>([resolved.buffer contents]);
  if (base == nullptr) {
    return CUBLAS_STATUS_MAPPING_ERROR;
  }
  std::memcpy(out_value, base + resolved.offset, sizeof(float));
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t ResolveScalarDouble(const CublasContext& ctx, const double* ptr,
                                   double* out_value) noexcept {
  if (ptr == nullptr || out_value == nullptr) {
    return CUBLAS_STATUS_INVALID_VALUE;
  }
  cublasPointerMode_t mode = CUBLAS_POINTER_MODE_HOST;
  cudaStream_t stream = nullptr;
  {
    std::lock_guard<std::mutex> lock(ctx.mutex);
    mode = ctx.pointer_mode;
    stream = ctx.stream;
  }
  if (mode == CUBLAS_POINTER_MODE_HOST) {
    *out_value = *ptr;
    return CUBLAS_STATUS_SUCCESS;
  }

  ResolvedBuffer resolved;
  auto& exec = LibraryExecutor::Instance();
  if (!exec.ResolveDeviceBuffer(ptr, sizeof(double), &resolved)) {
    return CUBLAS_STATUS_INVALID_VALUE;
  }
  if (exec.SynchronizeStream(stream) != cudaSuccess) {
    return CUBLAS_STATUS_EXECUTION_FAILED;
  }
  const auto* base = static_cast<const std::uint8_t*>([resolved.buffer contents]);
  if (base == nullptr) {
    return CUBLAS_STATUS_MAPPING_ERROR;
  }
  std::memcpy(out_value, base + resolved.offset, sizeof(double));
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t ResolveScalarHalfAsFloat(const CublasContext& ctx, const __half* ptr,
                                        float* out_value) noexcept {
  if (ptr == nullptr || out_value == nullptr) {
    return CUBLAS_STATUS_INVALID_VALUE;
  }
  cublasPointerMode_t mode = CUBLAS_POINTER_MODE_HOST;
  cudaStream_t stream = nullptr;
  {
    std::lock_guard<std::mutex> lock(ctx.mutex);
    mode = ctx.pointer_mode;
    stream = ctx.stream;
  }
  if (mode == CUBLAS_POINTER_MODE_HOST) {
    *out_value = HalfBitsToFloat(ptr->x);
    return CUBLAS_STATUS_SUCCESS;
  }

  ResolvedBuffer resolved;
  auto& exec = LibraryExecutor::Instance();
  if (!exec.ResolveDeviceBuffer(ptr, sizeof(__half), &resolved)) {
    return CUBLAS_STATUS_INVALID_VALUE;
  }
  if (exec.SynchronizeStream(stream) != cudaSuccess) {
    return CUBLAS_STATUS_EXECUTION_FAILED;
  }
  const auto* base = static_cast<const std::uint8_t*>([resolved.buffer contents]);
  if (base == nullptr) {
    return CUBLAS_STATUS_MAPPING_ERROR;
  }
  std::uint16_t raw = 0;
  std::memcpy(&raw, base + resolved.offset, sizeof(raw));
  *out_value = HalfBitsToFloat(raw);
  return CUBLAS_STATUS_SUCCESS;
}

}  // namespace muda::libraries::blas

using namespace muda::libraries;
using namespace muda::libraries::blas;

extern "C" {

cublasStatus_t cublasCreate_v2(cublasHandle_t* handle) {
  if (handle == nullptr) {
    return CUBLAS_STATUS_INVALID_VALUE;
  }
  auto& exec = LibraryExecutor::Instance();
  if (exec.EnsureInitialized() != cudaSuccess) {
    return CUBLAS_STATUS_NOT_INITIALIZED;
  }
  id<MTLDevice> device = exec.Device();
  if (device == nil) {
    return CUBLAS_STATUS_NOT_INITIALIZED;
  }
  @autoreleasepool {
    auto ctx = std::make_shared<CublasContext>();
    ctx->scratch_buffer = [device newBufferWithLength:kScratchBufferBytes
                                              options:MTLResourceStorageModeShared];
    if (ctx->scratch_buffer == nil) {
      return CUBLAS_STATUS_ALLOC_FAILED;
    }
    *handle = GetCublasHandleTable().Insert<cublasHandle_t>(std::move(ctx));
    return CUBLAS_STATUS_SUCCESS;
  }
}

cublasStatus_t cublasDestroy_v2(cublasHandle_t handle) {
  auto removed = GetCublasHandleTable().Remove(handle);
  if (removed == nullptr) {
    return CUBLAS_STATUS_NOT_INITIALIZED;
  }
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetVersion_v2(cublasHandle_t handle, int* version) {
  if (version == nullptr) {
    return CUBLAS_STATUS_INVALID_VALUE;
  }
  if (handle != nullptr && LookupContext(handle) == nullptr) {
    return CUBLAS_STATUS_NOT_INITIALIZED;
  }
  *version = kCublasVersion;
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetStream_v2(cublasHandle_t handle, cudaStream_t streamId) {
  auto ctx = LookupContext(handle);
  if (ctx == nullptr) {
    return CUBLAS_STATUS_NOT_INITIALIZED;
  }
  if (!LibraryExecutor::Instance().IsValidStream(streamId)) {
    return CUBLAS_STATUS_INVALID_VALUE;
  }
  std::lock_guard<std::mutex> lock(ctx->mutex);
  ctx->stream = streamId;
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetStream_v2(cublasHandle_t handle, cudaStream_t* streamId) {
  if (streamId == nullptr) {
    return CUBLAS_STATUS_INVALID_VALUE;
  }
  auto ctx = LookupContext(handle);
  if (ctx == nullptr) {
    return CUBLAS_STATUS_NOT_INITIALIZED;
  }
  std::lock_guard<std::mutex> lock(ctx->mutex);
  *streamId = ctx->stream;
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetPointerMode_v2(cublasHandle_t handle, cublasPointerMode_t mode) {
  auto ctx = LookupContext(handle);
  if (ctx == nullptr) {
    return CUBLAS_STATUS_NOT_INITIALIZED;
  }
  if (mode != CUBLAS_POINTER_MODE_HOST && mode != CUBLAS_POINTER_MODE_DEVICE) {
    return CUBLAS_STATUS_INVALID_VALUE;
  }
  std::lock_guard<std::mutex> lock(ctx->mutex);
  ctx->pointer_mode = mode;
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetPointerMode_v2(cublasHandle_t handle, cublasPointerMode_t* mode) {
  if (mode == nullptr) {
    return CUBLAS_STATUS_INVALID_VALUE;
  }
  auto ctx = LookupContext(handle);
  if (ctx == nullptr) {
    return CUBLAS_STATUS_NOT_INITIALIZED;
  }
  std::lock_guard<std::mutex> lock(ctx->mutex);
  *mode = ctx->pointer_mode;
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetMathMode(cublasHandle_t handle, cublasMath_t mode) {
  auto ctx = LookupContext(handle);
  if (ctx == nullptr) {
    return CUBLAS_STATUS_NOT_INITIALIZED;
  }
  if (mode < CUBLAS_DEFAULT_MATH || mode > CUBLAS_TF32_TENSOR_OP_MATH) {
    return CUBLAS_STATUS_INVALID_VALUE;
  }
  std::lock_guard<std::mutex> lock(ctx->mutex);
  ctx->math_mode = mode;
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetMathMode(cublasHandle_t handle, cublasMath_t* mode) {
  if (mode == nullptr) {
    return CUBLAS_STATUS_INVALID_VALUE;
  }
  auto ctx = LookupContext(handle);
  if (ctx == nullptr) {
    return CUBLAS_STATUS_NOT_INITIALIZED;
  }
  std::lock_guard<std::mutex> lock(ctx->mutex);
  *mode = ctx->math_mode;
  return CUBLAS_STATUS_SUCCESS;
}

const char* cublasGetStatusName(cublasStatus_t status) {
  switch (status) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED";
  }
  return "CUBLAS_STATUS_UNKNOWN";
}

const char* cublasGetStatusString(cublasStatus_t status) { return cublasGetStatusName(status); }

}  // extern "C"

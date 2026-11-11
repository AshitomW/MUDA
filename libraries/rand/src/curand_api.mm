/**
 * @file curand_api.mm
 * @brief Clean-room implementation of standard cuRAND C ABI exports.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>

#include "curand.h"
#include "curand_context.hpp"
#include "muda/libraries/metal_executor.hpp"

using namespace muda::libraries;
using namespace muda::libraries::rand;

namespace {

constexpr int kCurandVersion = 10200;  // CUDA 11/12 standard version
constexpr std::size_t kThreadsPerGroup = 256;

// Precomputed 32-bit Sobol direction vector table (initial dimensions)
alignas(16) const curandDirectionVectors32_t kSobolVectors32[1] = {
    {0x80000000u, 0x40000000u, 0x20000000u, 0x10000000u, 0x08000000u, 0x04000000u, 0x02000000u,
     0x01000000u, 0x00800000u, 0x00400000u, 0x00200000u, 0x00100000u, 0x00080000u, 0x00040000u,
     0x00020000u, 0x00010000u, 0x00008000u, 0x00004000u, 0x00002000u, 0x00001000u, 0x00000800u,
     0x00000400u, 0x00000200u, 0x00000100u, 0x00000080u, 0x00000040u, 0x00000020u, 0x00000010u,
     0x00000008u, 0x00000004u, 0x00000002u, 0x00000001u}};

const unsigned int kScrambleConstants32[1] = {0x6543210fu};

}  // namespace

extern "C" {

/* -------------------------------------------------------------------------- */
/* Lifecycle & Configuration                                                  */
/* -------------------------------------------------------------------------- */

curandStatus_t curandCreateGenerator(curandGenerator_t* generator, curandRngType_t rng_type) {
  if (generator == nullptr) {
    return CURAND_STATUS_ALLOCATION_FAILED;
  }
  auto ctx = std::make_shared<CurandContext>();
  ctx->rng_type = rng_type;
  ctx->target = GeneratorTarget::kGpu;
  ctx->seed = 0;
  ctx->offset = 0;

  *generator = GetCurandHandleTable().Insert<curandGenerator_t>(std::move(ctx));
  return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandCreateGeneratorHost(curandGenerator_t* generator, curandRngType_t rng_type) {
  if (generator == nullptr) {
    return CURAND_STATUS_ALLOCATION_FAILED;
  }
  auto ctx = std::make_shared<CurandContext>();
  ctx->rng_type = rng_type;
  ctx->target = GeneratorTarget::kHost;
  ctx->seed = 0;
  ctx->offset = 0;

  *generator = GetCurandHandleTable().Insert<curandGenerator_t>(std::move(ctx));
  return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandDestroyGenerator(curandGenerator_t generator) {
  if (generator == nullptr) {
    return CURAND_STATUS_NOT_INITIALIZED;
  }
  if (!GetCurandHandleTable().Remove(generator)) {
    return CURAND_STATUS_NOT_INITIALIZED;
  }
  return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandSetStream(curandGenerator_t generator, cudaStream_t stream) {
  auto ctx = LookupContext(generator);
  if (!ctx) {
    return CURAND_STATUS_NOT_INITIALIZED;
  }
  std::lock_guard<std::mutex> lock(ctx->mutex);
  ctx->stream = stream;
  return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGetStream(curandGenerator_t generator, cudaStream_t* stream) {
  if (stream == nullptr) {
    return CURAND_STATUS_INVALID_VALUE;
  }
  auto ctx = LookupContext(generator);
  if (!ctx) {
    return CURAND_STATUS_NOT_INITIALIZED;
  }
  std::lock_guard<std::mutex> lock(ctx->mutex);
  *stream = ctx->stream;
  return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandSetPseudoRandomGeneratorSeed(curandGenerator_t generator,
                                                  unsigned long long seed) {
  auto ctx = LookupContext(generator);
  if (!ctx) {
    return CURAND_STATUS_NOT_INITIALIZED;
  }
  std::lock_guard<std::mutex> lock(ctx->mutex);
  ctx->seed = seed;
  ctx->offset = 0;
  return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandSetGeneratorOffset(curandGenerator_t generator, unsigned long long offset) {
  auto ctx = LookupContext(generator);
  if (!ctx) {
    return CURAND_STATUS_NOT_INITIALIZED;
  }
  std::lock_guard<std::mutex> lock(ctx->mutex);
  ctx->offset = offset;
  return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandSetGeneratorOrdering(curandGenerator_t generator, curandOrdering_t order) {
  auto ctx = LookupContext(generator);
  if (!ctx) {
    return CURAND_STATUS_NOT_INITIALIZED;
  }
  std::lock_guard<std::mutex> lock(ctx->mutex);
  ctx->ordering = order;
  return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandSetQuasiRandomGeneratorDimensions(curandGenerator_t generator,
                                                       unsigned int num_dimensions) {
  auto ctx = LookupContext(generator);
  if (!ctx) {
    return CURAND_STATUS_NOT_INITIALIZED;
  }
  if (num_dimensions < 1 || num_dimensions > 20000) {
    return CURAND_STATUS_OUT_OF_RANGE;
  }
  std::lock_guard<std::mutex> lock(ctx->mutex);
  ctx->dimensions = num_dimensions;
  return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGetVersion(int* version) {
  if (version == nullptr) {
    return CURAND_STATUS_INVALID_VALUE;
  }
  *version = kCurandVersion;
  return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGetProperty(libraryPropertyType type, int* value) {
  if (value == nullptr) {
    return CURAND_STATUS_INVALID_VALUE;
  }
  switch (type) {
    case MAJOR_VERSION:
      *value = 10;
      return CURAND_STATUS_SUCCESS;
    case MINOR_VERSION:
      *value = 2;
      return CURAND_STATUS_SUCCESS;
    case PATCH_LEVEL:
      *value = 0;
      return CURAND_STATUS_SUCCESS;
  }
  return CURAND_STATUS_INVALID_VALUE;
}

/* -------------------------------------------------------------------------- */
/* Generation Functions                                                       */
/* -------------------------------------------------------------------------- */

curandStatus_t curandGenerate(curandGenerator_t generator, unsigned int* outputPtr, size_t num) {
  if (outputPtr == nullptr && num > 0) {
    return CURAND_STATUS_INVALID_VALUE;
  }
  if (num == 0) {
    return CURAND_STATUS_SUCCESS;
  }
  auto ctx = LookupContext(generator);
  if (!ctx) {
    return CURAND_STATUS_NOT_INITIALIZED;
  }
  if (ctx->target == GeneratorTarget::kHost) {
    return GenerateHostUint(*ctx, outputPtr, num);
  }

  auto& exec = LibraryExecutor::Instance();
  ResolvedBuffer dev_buf;
  if (!exec.ResolveDeviceBuffer(outputPtr, num * sizeof(unsigned int), &dev_buf)) {
    return CURAND_STATUS_ALLOCATION_FAILED;
  }

  id<MTLComputePipelineState> pipeline = GetCurandPipeline("muda_curand_philox_uint");
  if (pipeline == nil) {
    return CURAND_STATUS_INTERNAL_ERROR;
  }

  std::lock_guard<std::mutex> lock(ctx->mutex);
  const std::uint64_t num_u64 = num;
  const std::uint64_t seed_u64 = ctx->seed;
  const std::uint64_t offset_u64 = ctx->offset;

  const std::array<ShaderArg, 4> args = {
      ShaderArg::Buffer(dev_buf),
      ShaderArg::Value(num_u64),
      ShaderArg::Value(seed_u64),
      ShaderArg::Value(offset_u64),
  };

  const std::size_t total_threads = (num + 3) / 4;
  const cudaError_t err =
      exec.DispatchThreads1D(pipeline, total_threads, kThreadsPerGroup, args, 0, ctx->stream);
  if (err != cudaSuccess) {
    return CURAND_STATUS_LAUNCH_FAILURE;
  }

  ctx->offset += static_cast<std::uint64_t>(total_threads * 4);
  return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGenerateUniform(curandGenerator_t generator, float* outputPtr, size_t num) {
  if (outputPtr == nullptr && num > 0) {
    return CURAND_STATUS_INVALID_VALUE;
  }
  if (num == 0) {
    return CURAND_STATUS_SUCCESS;
  }
  auto ctx = LookupContext(generator);
  if (!ctx) {
    return CURAND_STATUS_NOT_INITIALIZED;
  }
  if (ctx->target == GeneratorTarget::kHost) {
    return GenerateHostUniform(*ctx, outputPtr, num);
  }

  auto& exec = LibraryExecutor::Instance();
  ResolvedBuffer dev_buf;
  if (!exec.ResolveDeviceBuffer(outputPtr, num * sizeof(float), &dev_buf)) {
    return CURAND_STATUS_ALLOCATION_FAILED;
  }

  id<MTLComputePipelineState> pipeline = GetCurandPipeline("muda_curand_philox_uniform_f32");
  if (pipeline == nil) {
    return CURAND_STATUS_INTERNAL_ERROR;
  }

  std::lock_guard<std::mutex> lock(ctx->mutex);
  const std::uint64_t num_u64 = num;
  const std::uint64_t seed_u64 = ctx->seed;
  const std::uint64_t offset_u64 = ctx->offset;

  const std::array<ShaderArg, 4> args = {
      ShaderArg::Buffer(dev_buf),
      ShaderArg::Value(num_u64),
      ShaderArg::Value(seed_u64),
      ShaderArg::Value(offset_u64),
  };

  const std::size_t total_threads = (num + 3) / 4;
  const cudaError_t err =
      exec.DispatchThreads1D(pipeline, total_threads, kThreadsPerGroup, args, 0, ctx->stream);
  if (err != cudaSuccess) {
    return CURAND_STATUS_LAUNCH_FAILURE;
  }

  ctx->offset += static_cast<std::uint64_t>(total_threads * 4);
  return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGenerateUniformDouble(curandGenerator_t generator, double* outputPtr,
                                           size_t num) {
  if (outputPtr == nullptr && num > 0) {
    return CURAND_STATUS_INVALID_VALUE;
  }
  if (num == 0) {
    return CURAND_STATUS_SUCCESS;
  }
  auto ctx = LookupContext(generator);
  if (!ctx) {
    return CURAND_STATUS_NOT_INITIALIZED;
  }
  if (ctx->target == GeneratorTarget::kGpu) {
    (void)LibraryExecutor::Instance().SynchronizeStream(ctx->stream);
  }
  return GenerateHostUniformDouble(*ctx, outputPtr, num);
}

curandStatus_t curandGenerateNormal(curandGenerator_t generator, float* outputPtr, size_t n,
                                    float mean, float stddev) {
  if (outputPtr == nullptr && n > 0) {
    return CURAND_STATUS_INVALID_VALUE;
  }
  if (n == 0) {
    return CURAND_STATUS_SUCCESS;
  }
  auto ctx = LookupContext(generator);
  if (!ctx) {
    return CURAND_STATUS_NOT_INITIALIZED;
  }
  if (ctx->target == GeneratorTarget::kHost) {
    return GenerateHostNormal(*ctx, outputPtr, n, mean, stddev);
  }

  auto& exec = LibraryExecutor::Instance();
  ResolvedBuffer dev_buf;
  if (!exec.ResolveDeviceBuffer(outputPtr, n * sizeof(float), &dev_buf)) {
    return CURAND_STATUS_ALLOCATION_FAILED;
  }

  id<MTLComputePipelineState> pipeline = GetCurandPipeline("muda_curand_philox_normal_f32");
  if (pipeline == nil) {
    return CURAND_STATUS_INTERNAL_ERROR;
  }

  std::lock_guard<std::mutex> lock(ctx->mutex);
  const std::uint64_t num_u64 = n;
  const std::uint64_t seed_u64 = ctx->seed;
  const std::uint64_t offset_u64 = ctx->offset;

  const std::array<ShaderArg, 6> args = {
      ShaderArg::Buffer(dev_buf),   ShaderArg::Value(num_u64), ShaderArg::Value(seed_u64),
      ShaderArg::Value(offset_u64), ShaderArg::Value(mean),    ShaderArg::Value(stddev),
  };

  const std::size_t total_threads = (n + 3) / 4;
  const cudaError_t err =
      exec.DispatchThreads1D(pipeline, total_threads, kThreadsPerGroup, args, 0, ctx->stream);
  if (err != cudaSuccess) {
    return CURAND_STATUS_LAUNCH_FAILURE;
  }

  ctx->offset += static_cast<std::uint64_t>(total_threads * 4);
  return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGenerateNormalDouble(curandGenerator_t generator, double* outputPtr, size_t n,
                                          double mean, double stddev) {
  if (outputPtr == nullptr && n > 0) {
    return CURAND_STATUS_INVALID_VALUE;
  }
  if (n == 0) {
    return CURAND_STATUS_SUCCESS;
  }
  auto ctx = LookupContext(generator);
  if (!ctx) {
    return CURAND_STATUS_NOT_INITIALIZED;
  }
  if (ctx->target == GeneratorTarget::kGpu) {
    (void)LibraryExecutor::Instance().SynchronizeStream(ctx->stream);
  }
  return GenerateHostNormalDouble(*ctx, outputPtr, n, mean, stddev);
}

curandStatus_t curandGenerateLogNormal(curandGenerator_t generator, float* outputPtr, size_t n,
                                       float mean, float stddev) {
  if (outputPtr == nullptr && n > 0) {
    return CURAND_STATUS_INVALID_VALUE;
  }
  if (n == 0) {
    return CURAND_STATUS_SUCCESS;
  }
  auto ctx = LookupContext(generator);
  if (!ctx) {
    return CURAND_STATUS_NOT_INITIALIZED;
  }
  if (ctx->target == GeneratorTarget::kHost) {
    return GenerateHostLogNormal(*ctx, outputPtr, n, mean, stddev);
  }

  auto& exec = LibraryExecutor::Instance();
  ResolvedBuffer dev_buf;
  if (!exec.ResolveDeviceBuffer(outputPtr, n * sizeof(float), &dev_buf)) {
    return CURAND_STATUS_ALLOCATION_FAILED;
  }

  id<MTLComputePipelineState> pipeline = GetCurandPipeline("muda_curand_philox_log_normal_f32");
  if (pipeline == nil) {
    return CURAND_STATUS_INTERNAL_ERROR;
  }

  std::lock_guard<std::mutex> lock(ctx->mutex);
  const std::uint64_t num_u64 = n;
  const std::uint64_t seed_u64 = ctx->seed;
  const std::uint64_t offset_u64 = ctx->offset;

  const std::array<ShaderArg, 6> args = {
      ShaderArg::Buffer(dev_buf),   ShaderArg::Value(num_u64), ShaderArg::Value(seed_u64),
      ShaderArg::Value(offset_u64), ShaderArg::Value(mean),    ShaderArg::Value(stddev),
  };

  const std::size_t total_threads = (n + 3) / 4;
  const cudaError_t err =
      exec.DispatchThreads1D(pipeline, total_threads, kThreadsPerGroup, args, 0, ctx->stream);
  if (err != cudaSuccess) {
    return CURAND_STATUS_LAUNCH_FAILURE;
  }

  ctx->offset += static_cast<std::uint64_t>(total_threads * 4);
  return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGenerateLogNormalDouble(curandGenerator_t generator, double* outputPtr,
                                             size_t n, double mean, double stddev) {
  if (outputPtr == nullptr && n > 0) {
    return CURAND_STATUS_INVALID_VALUE;
  }
  if (n == 0) {
    return CURAND_STATUS_SUCCESS;
  }
  auto ctx = LookupContext(generator);
  if (!ctx) {
    return CURAND_STATUS_NOT_INITIALIZED;
  }
  if (ctx->target == GeneratorTarget::kHost) {
    return GenerateHostLogNormalDouble(*ctx, outputPtr, n, mean, stddev);
  }

  // Generate normal into output buffer then exponentiate in place via stream
  const curandStatus_t st = curandGenerateNormalDouble(generator, outputPtr, n, mean, stddev);
  if (st != CURAND_STATUS_SUCCESS) {
    return st;
  }
  // Synchronize and exponentiate on host unified memory
  (void)cudaDeviceSynchronize();
  for (size_t i = 0; i < n; ++i) {
    outputPtr[i] = std::exp(outputPtr[i]);
  }
  return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGeneratePoisson(curandGenerator_t generator, unsigned int* outputPtr, size_t n,
                                     double lambda) {
  if (outputPtr == nullptr && n > 0) {
    return CURAND_STATUS_INVALID_VALUE;
  }
  if (n == 0) {
    return CURAND_STATUS_SUCCESS;
  }
  auto ctx = LookupContext(generator);
  if (!ctx) {
    return CURAND_STATUS_NOT_INITIALIZED;
  }
  if (ctx->target == GeneratorTarget::kHost) {
    return GenerateHostPoisson(*ctx, outputPtr, n, lambda);
  }

  auto& exec = LibraryExecutor::Instance();
  ResolvedBuffer dev_buf;
  if (!exec.ResolveDeviceBuffer(outputPtr, n * sizeof(unsigned int), &dev_buf)) {
    return CURAND_STATUS_ALLOCATION_FAILED;
  }

  id<MTLComputePipelineState> pipeline = GetCurandPipeline("muda_curand_philox_poisson");
  if (pipeline == nil) {
    return CURAND_STATUS_INTERNAL_ERROR;
  }

  std::lock_guard<std::mutex> lock(ctx->mutex);
  const std::uint64_t num_u64 = n;
  const std::uint64_t seed_u64 = ctx->seed;
  const std::uint64_t offset_u64 = ctx->offset;

  const float lambda_f = static_cast<float>(lambda);
  const std::array<ShaderArg, 5> args = {
      ShaderArg::Buffer(dev_buf),   ShaderArg::Value(num_u64),  ShaderArg::Value(seed_u64),
      ShaderArg::Value(offset_u64), ShaderArg::Value(lambda_f),
  };

  const cudaError_t err =
      exec.DispatchThreads1D(pipeline, n, kThreadsPerGroup, args, 0, ctx->stream);
  if (err != cudaSuccess) {
    return CURAND_STATUS_LAUNCH_FAILURE;
  }

  ctx->offset += static_cast<std::uint64_t>(n);
  return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGetDirectionVectors32(curandDirectionVectors32_t* vectors[],
                                           curandDirectionVectorSet_t set) {
  if (vectors == nullptr) {
    return CURAND_STATUS_INVALID_VALUE;
  }
  if (set == CURAND_DIRECTION_VECTORS_32_JOEKUO6 ||
      set == CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6) {
    *vectors = const_cast<curandDirectionVectors32_t*>(kSobolVectors32);
    return CURAND_STATUS_SUCCESS;
  }
  return CURAND_STATUS_OUT_OF_RANGE;
}

curandStatus_t curandGetScrambleConstants32(const unsigned int** constants) {
  if (constants == nullptr) {
    return CURAND_STATUS_INVALID_VALUE;
  }
  *constants = kScrambleConstants32;
  return CURAND_STATUS_SUCCESS;
}

}  // extern "C"

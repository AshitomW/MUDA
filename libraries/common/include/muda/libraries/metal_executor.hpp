/**
 * @file metal_executor.hpp
 * @brief Shared Metal compute pipeline loader, buffer resolver, and stream encoder for libraries.
 */

#pragma once

#import <Metal/Metal.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>

#include "cuda_runtime.h"

namespace muda::libraries {

/**
 * @brief Resolved Metal buffer and byte offset for a device or unified memory pointer.
 */
struct ResolvedBuffer {
  id<MTLBuffer> buffer = nil;
  std::size_t offset = 0;
  std::size_t available_bytes = 0;
};

/**
 * @brief Kernel argument binding descriptor (either a Metal buffer or inline constant bytes).
 */
struct ShaderArg {
  enum class Kind : std::uint8_t {
    kBuffer = 0,
    kBytes = 1,
  };

  Kind kind = Kind::kBuffer;
  id<MTLBuffer> buffer = nil;
  std::size_t offset = 0;
  const void* bytes = nullptr;
  std::size_t size = 0;

  [[nodiscard]] static ShaderArg Buffer(const ResolvedBuffer& resolved,
                                        std::size_t extra_offset = 0) noexcept {
    ShaderArg arg;
    arg.kind = Kind::kBuffer;
    arg.buffer = resolved.buffer;
    arg.offset = resolved.offset + extra_offset;
    return arg;
  }

  [[nodiscard]] static ShaderArg RawBuffer(id<MTLBuffer> buf, std::size_t byte_offset = 0) noexcept {
    ShaderArg arg;
    arg.kind = Kind::kBuffer;
    arg.buffer = buf;
    arg.offset = byte_offset;
    return arg;
  }

  template <typename T>
  [[nodiscard]] static ShaderArg Value(const T& val) noexcept {
    ShaderArg arg;
    arg.kind = Kind::kBytes;
    arg.bytes = &val;
    arg.size = sizeof(T);
    return arg;
  }
};

/**
 * @brief Singleton helper coordinating metallib loading and stream-ordered GPU execution.
 */
class LibraryExecutor {
 public:
  static LibraryExecutor& Instance() noexcept;

  [[nodiscard]] cudaError_t EnsureInitialized() noexcept;
  [[nodiscard]] id<MTLDevice> Device() const noexcept;
  [[nodiscard]] bool IsValidStream(cudaStream_t stream) const noexcept;

  [[nodiscard]] bool ResolveDeviceBuffer(const void* ptr, std::size_t min_bytes,
                                         ResolvedBuffer* out_resolved) const noexcept;

  [[nodiscard]] id<MTLComputePipelineState> GetOrLoadPipeline(
      std::string_view metallib_path, std::string_view metal_source_path,
      std::string_view function_name) noexcept;

  [[nodiscard]] cudaError_t DispatchThreadgroups(id<MTLComputePipelineState> pipeline,
                                                 MTLSize grid_size, MTLSize block_size,
                                                 std::span<const ShaderArg> args,
                                                 std::size_t shared_mem_bytes,
                                                 cudaStream_t stream) noexcept;

  [[nodiscard]] cudaError_t DispatchThreads1D(id<MTLComputePipelineState> pipeline,
                                              std::size_t total_threads,
                                              std::size_t threads_per_group,
                                              std::span<const ShaderArg> args,
                                              std::size_t shared_mem_bytes,
                                              cudaStream_t stream) noexcept;

  [[nodiscard]] cudaError_t EncodeCommandBuffer(
      cudaStream_t stream,
      const std::function<bool(id<MTLCommandBuffer>)>& encode_callback) noexcept;

  [[nodiscard]] cudaError_t SynchronizeStream(cudaStream_t stream) noexcept;

 private:
  LibraryExecutor() = default;

  mutable std::mutex mutex_;
  id<MTLDevice> cached_device_ = nil;
  std::unordered_map<std::string, id<MTLLibrary>> libraries_;
  std::unordered_map<std::string, id<MTLComputePipelineState>> pipelines_;
};

}  // namespace muda::libraries

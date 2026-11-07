/**
 * @file metal_executor.mm
 * @brief Implementation of shared Metal library loading and stream execution for MUDA libraries.
 */

#include "muda/libraries/metal_executor.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>

#include "runtime_context.hpp"

namespace muda::libraries {
namespace {

constexpr std::size_t kMaxBatchDispatches = 256;

[[nodiscard]] constexpr std::size_t AlignUp16(std::size_t value) noexcept {
  const std::size_t rem = value % 16u;
  return rem == 0 ? value : (value + (16u - rem));
}

}  // namespace

LibraryExecutor& LibraryExecutor::Instance() noexcept {
  static LibraryExecutor instance;
  return instance;
}

cudaError_t LibraryExecutor::EnsureInitialized() noexcept {
  return runtime::cudart::RuntimeContext::Instance().EnsureInitialized();
}

id<MTLDevice> LibraryExecutor::Device() const noexcept {
  return runtime::cudart::RuntimeContext::Instance().Device();
}

bool LibraryExecutor::IsValidStream(cudaStream_t stream) const noexcept {
  if (stream == nullptr) {
    return true;
  }
  return runtime::cudart::RuntimeContext::Instance().Streams().Lookup(stream) != nullptr;
}

bool LibraryExecutor::ResolveDeviceBuffer(const void* ptr, std::size_t min_bytes,
                                          ResolvedBuffer* out_resolved) const noexcept {
  if (ptr == nullptr || out_resolved == nullptr) {
    return false;
  }
  auto& ctx = runtime::cudart::RuntimeContext::Instance();
  runtime::cudart::ResolvedEndpoint endpoint;
  const auto addr = runtime::cudart::DeviceAddress::FromPointer(ptr);
  if (!ctx.Memory().ResolveEndpoint(addr, &endpoint)) {
    return false;
  }
  if (endpoint.buffer == nil || endpoint.offset > endpoint.size) {
    return false;
  }
  const std::size_t remaining = endpoint.size - endpoint.offset;
  if (min_bytes > remaining) {
    return false;
  }
  out_resolved->buffer = endpoint.buffer;
  out_resolved->offset = endpoint.offset;
  out_resolved->available_bytes = remaining;
  return true;
}

id<MTLComputePipelineState> LibraryExecutor::GetOrLoadPipeline(
    std::string_view metallib_path, std::string_view metal_source_path,
    std::string_view function_name) noexcept {
  if (EnsureInitialized() != cudaSuccess) {
    return nil;
  }
  id<MTLDevice> device = Device();
  if (device == nil) {
    return nil;
  }

  std::string pipe_key;
  pipe_key.reserve(metallib_path.size() + 2 + function_name.size());
  pipe_key.append(metallib_path);
  pipe_key.append("::");
  pipe_key.append(function_name);

  {
    std::shared_lock<std::shared_mutex> rlock(mutex_);
    if (cached_device_ == device) {
      auto pipe_it = pipelines_.find(pipe_key);
      if (pipe_it != pipelines_.end()) {
        return pipe_it->second;
      }
    }
  }

  const std::string lib_key(metallib_path);
  std::unique_lock<std::shared_mutex> lock(mutex_);
  if (cached_device_ != device) {
    libraries_.clear();
    pipelines_.clear();
    cached_device_ = device;
  }

  auto pipe_it = pipelines_.find(pipe_key);
  if (pipe_it != pipelines_.end()) {
    return pipe_it->second;
  }

  @autoreleasepool {
    id<MTLLibrary> library = nil;
    auto lib_it = libraries_.find(lib_key);
    if (lib_it != libraries_.end()) {
      library = lib_it->second;
    } else {
      NSString* mlib_str = [[NSString alloc] initWithBytes:metallib_path.data()
                                                    length:metallib_path.size()
                                                  encoding:NSUTF8StringEncoding];
      NSError* err = nil;
      if (mlib_str != nil && [[NSFileManager defaultManager] fileExistsAtPath:mlib_str]) {
        library = [device newLibraryWithURL:[NSURL fileURLWithPath:mlib_str] error:&err];
      }
      if (library == nil && !metal_source_path.empty()) {
        NSString* src_path = [[NSString alloc] initWithBytes:metal_source_path.data()
                                                      length:metal_source_path.size()
                                                    encoding:NSUTF8StringEncoding];
        NSString* src_contents = [NSString stringWithContentsOfFile:src_path
                                                           encoding:NSUTF8StringEncoding
                                                              error:&err];
        if (src_contents != nil) {
          MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
          opts.mathMode = MTLMathModeFast;
          library = [device newLibraryWithSource:src_contents options:opts error:&err];
        }
      }
      if (library == nil) {
        return nil;
      }
      libraries_[lib_key] = library;
    }

    NSString* fn_str = [[NSString alloc] initWithBytes:function_name.data()
                                                length:function_name.size()
                                              encoding:NSUTF8StringEncoding];
    if (fn_str == nil) {
      return nil;
    }
    id<MTLFunction> function = [library newFunctionWithName:fn_str];
    if (function == nil) {
      return nil;
    }
    NSError* err = nil;
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function
                                                                                 error:&err];
    if (pipeline == nil) {
      return nil;
    }
    pipelines_[pipe_key] = pipeline;
    return pipeline;
  }
}

cudaError_t LibraryExecutor::DispatchThreadgroups(id<MTLComputePipelineState> pipeline,
                                                  MTLSize grid_size, MTLSize block_size,
                                                  std::span<const ShaderArg> args,
                                                  std::size_t shared_mem_bytes,
                                                  cudaStream_t stream) noexcept {
  if (pipeline == nil || grid_size.width == 0 || grid_size.height == 0 || grid_size.depth == 0 ||
      block_size.width == 0 || block_size.height == 0 || block_size.depth == 0) {
    return cudaErrorInvalidConfiguration;
  }

  auto& ctx = runtime::cudart::RuntimeContext::Instance();
  const cudaError_t init_err = ctx.EnsureInitialized();
  if (init_err != cudaSuccess) {
    return init_err;
  }

  std::shared_ptr<runtime::cudart::StreamState> stream_state;
  id<MTLCommandQueue> target_queue = nil;
  const cudaError_t s_err =
      ctx.Streams().ResolveStreamQueue(stream, ctx.DefaultQueue(), &stream_state, &target_queue);
  if (s_err != cudaSuccess) {
    return s_err;
  }

  @autoreleasepool {
    auto& streams = ctx.Streams();
    std::unique_lock<std::mutex> lock =
        (stream_state != nullptr) ? std::unique_lock<std::mutex>(stream_state->mutex)
                                  : std::unique_lock<std::mutex>(streams.DefaultStreamMutex());

    __strong id<MTLCommandBuffer>& active_cmd = (stream_state != nullptr)
                                                    ? stream_state->active_command_buffer
                                                    : streams.DefaultActiveCommandBuffer();
    __strong id<MTLComputeCommandEncoder>& active_enc =
        (stream_state != nullptr) ? stream_state->active_encoder : streams.DefaultActiveEncoder();
    __strong id<MTLBlitCommandEncoder>& active_blit = (stream_state != nullptr)
                                                          ? stream_state->active_blit_encoder
                                                          : streams.DefaultActiveBlitEncoder();
    __strong id<MTLComputePipelineState>& active_pipe =
        (stream_state != nullptr) ? stream_state->active_pipeline : streams.DefaultActivePipeline();
    std::size_t& active_disp = (stream_state != nullptr) ? stream_state->active_dispatches
                                                         : streams.DefaultActiveDispatches();
    runtime::cudart::EncoderHazardState& enc_state =
        (stream_state != nullptr) ? stream_state->encoder_state : streams.DefaultEncoderState();

    if (active_blit != nil) {
      [active_blit endEncoding];
      active_blit = nil;
    }

    if (active_cmd == nil) {
      active_cmd = [target_queue commandBuffer];
      if (active_cmd == nil) {
        return cudaErrorLaunchFailure;
      }
      active_disp = 0;
    }

    if (active_enc == nil) {
      active_enc = [active_cmd computeCommandEncoder];
      if (active_enc == nil) {
        active_cmd = nil;
        return cudaErrorLaunchFailure;
      }
      active_pipe = nil;
      enc_state.Reset();
    }

    const bool pipeline_changed = (active_pipe != nil && active_pipe != pipeline);
    if (active_pipe != pipeline) {
      [active_enc setComputePipelineState:pipeline];
      active_pipe = pipeline;
    }

    bool needs_barrier = false;
    if (active_disp > 0) {
      for (std::size_t idx = 0; idx < args.size(); ++idx) {
        const ShaderArg& arg = args[idx];
        if (arg.kind == ShaderArg::Kind::kBuffer) {
          const void* buf_ptr = (__bridge const void*)arg.buffer;
          if (enc_state.HasHazard(buf_ptr, arg.offset, idx, pipeline_changed)) {
            needs_barrier = true;
            break;
          }
        }
      }
    }
    if (needs_barrier) {
      [active_enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
      enc_state.ClearHazards();
    }

    for (std::size_t idx = 0; idx < args.size(); ++idx) {
      const ShaderArg& arg = args[idx];
      if (arg.kind == ShaderArg::Kind::kBuffer) {
        const void* buf_ptr = (__bridge const void*)arg.buffer;
        if (idx < runtime::cudart::kMaxEncoderBufferSlots) {
          runtime::cudart::BoundBufferSlot& slot = enc_state.slots[idx];
          if (slot.valid && slot.buffer_ptr == buf_ptr) {
            if (slot.offset != arg.offset) {
              [active_enc setBufferOffset:arg.offset atIndex:idx];
              slot.offset = arg.offset;
            }
          } else {
            [active_enc setBuffer:arg.buffer offset:arg.offset atIndex:idx];
            slot.buffer_ptr = buf_ptr;
            slot.offset = arg.offset;
            slot.valid = true;
          }
        } else {
          [active_enc setBuffer:arg.buffer offset:arg.offset atIndex:idx];
        }
        enc_state.MarkDirty(buf_ptr, arg.offset, idx);
      } else {
        [active_enc setBytes:arg.bytes length:arg.size atIndex:idx];
        if (idx < runtime::cudart::kMaxEncoderBufferSlots) {
          enc_state.slots[idx].valid = false;
        }
      }
    }

    if (shared_mem_bytes > 0) {
      [active_enc setThreadgroupMemoryLength:AlignUp16(shared_mem_bytes) atIndex:0];
    }

    [active_enc dispatchThreadgroups:grid_size threadsPerThreadgroup:block_size];
    active_disp++;
    if (stream_state == nullptr) {
      streams.SetDefaultActiveDispatchesAtomic(active_disp);
    }

    if (active_disp >= kMaxBatchDispatches) {
      if (stream_state != nullptr) {
        streams.FlushStreamLocked(stream_state.get());
      } else {
        streams.FlushDefaultStreamLocked();
      }
    }
    return cudaSuccess;
  }
}

cudaError_t LibraryExecutor::DispatchThreads1D(
    id<MTLComputePipelineState> pipeline, std::size_t total_threads, std::size_t threads_per_group,
    std::span<const ShaderArg> args, std::size_t shared_mem_bytes, cudaStream_t stream) noexcept {
  if (pipeline == nil || total_threads == 0) {
    return cudaErrorInvalidConfiguration;
  }
  const std::size_t max_threads = pipeline.maxTotalThreadsPerThreadgroup;
  const std::size_t block_x =
      std::min(threads_per_group > 0 ? threads_per_group : 256u, max_threads);
  const std::size_t grid_x = (total_threads + block_x - 1) / block_x;
  return DispatchThreadgroups(pipeline, MTLSizeMake(grid_x, 1, 1), MTLSizeMake(block_x, 1, 1), args,
                              shared_mem_bytes, stream);
}

cudaError_t LibraryExecutor::EncodeCommandBuffer(
    cudaStream_t stream,
    const std::function<bool(id<MTLCommandBuffer>)>& encode_callback) noexcept {
  if (!encode_callback) {
    return cudaErrorInvalidValue;
  }
  auto& ctx = runtime::cudart::RuntimeContext::Instance();
  const cudaError_t init_err = ctx.EnsureInitialized();
  if (init_err != cudaSuccess) {
    return init_err;
  }

  std::shared_ptr<runtime::cudart::StreamState> stream_state;
  id<MTLCommandQueue> target_queue = nil;
  const cudaError_t s_err =
      ctx.Streams().ResolveStreamQueue(stream, ctx.DefaultQueue(), &stream_state, &target_queue);
  if (s_err != cudaSuccess) {
    return s_err;
  }

  @autoreleasepool {
    auto& streams = ctx.Streams();
    std::unique_lock<std::mutex> lock =
        (stream_state != nullptr) ? std::unique_lock<std::mutex>(stream_state->mutex)
                                  : std::unique_lock<std::mutex>(streams.DefaultStreamMutex());

    __strong id<MTLCommandBuffer>& active_cmd = (stream_state != nullptr)
                                                    ? stream_state->active_command_buffer
                                                    : streams.DefaultActiveCommandBuffer();
    std::size_t& active_disp = (stream_state != nullptr) ? stream_state->active_dispatches
                                                         : streams.DefaultActiveDispatches();

    streams.EndActiveEncodersLocked(stream_state.get());

    if (active_cmd == nil) {
      active_cmd = [target_queue commandBuffer];
      if (active_cmd == nil) {
        return cudaErrorLaunchFailure;
      }
      active_disp = 0;
    }

    if (!encode_callback(active_cmd)) {
      return cudaErrorLaunchFailure;
    }

    active_disp++;
    if (stream_state == nullptr) {
      streams.SetDefaultActiveDispatchesAtomic(active_disp);
    }

    if (active_disp >= kMaxBatchDispatches) {
      if (stream_state != nullptr) {
        streams.FlushStreamLocked(stream_state.get());
      } else {
        streams.FlushDefaultStreamLocked();
      }
    }
    return cudaSuccess;
  }
}

cudaError_t LibraryExecutor::SynchronizeStream(cudaStream_t stream) noexcept {
  auto& ctx = runtime::cudart::RuntimeContext::Instance();
  const cudaError_t init_err = ctx.EnsureInitialized();
  if (init_err != cudaSuccess) {
    return init_err;
  }
  return ctx.Streams().StreamSynchronize(stream, ctx.DefaultQueue());
}

}  // namespace muda::libraries

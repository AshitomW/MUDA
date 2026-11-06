/**
 * @file cufft_exec.mm
 * @brief Implementation of cuFFT C2C, R2C, and C2R execution pipelines on Apple Silicon GPUs.
 */

#include "cufft_plan.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <array>
#include <string_view>
#include <utility>

#include "muda/libraries/metal_executor.hpp"

namespace muda::libraries::fft {
namespace {

[[nodiscard]] id<MTLComputePipelineState> GetCufftPipeline(
    std::string_view function_name) noexcept {
  return LibraryExecutor::Instance().GetOrLoadPipeline(
      MUDA_CUFFT_METALLIB_PATH, MUDA_CUFFT_METAL_SOURCE_PATH, function_name);
}

[[nodiscard]] constexpr bool IsPowerOfTwo(std::uint32_t x) noexcept {
  return x > 1u && (x & (x - 1u)) == 0u;
}

[[nodiscard]] cufftResult TransformAxis(id<MTLBuffer>* io_current, id<MTLBuffer>* io_scratch,
                                        std::uint32_t total_elements, std::uint32_t axis_len,
                                        std::uint32_t inner_stride, float sign_dir,
                                        cudaStream_t stream) noexcept {
  if (axis_len <= 1u) {
    return CUFFT_SUCCESS;
  }

  auto& exec = LibraryExecutor::Instance();
  if (IsPowerOfTwo(axis_len)) {
    id<MTLComputePipelineState> stage_pipe = GetCufftPipeline("muda_cufft_stockham_stage");
    if (stage_pipe == nil) {
      return CUFFT_INTERNAL_ERROR;
    }
    for (std::uint32_t half_m = 1u; half_m < axis_len; half_m <<= 1u) {
      const std::array<ShaderArg, 7> args = {
          ShaderArg::RawBuffer(*io_current, 0),
          ShaderArg::RawBuffer(*io_scratch, 0),
          ShaderArg::Value(total_elements),
          ShaderArg::Value(axis_len),
          ShaderArg::Value(inner_stride),
          ShaderArg::Value(half_m),
          ShaderArg::Value(sign_dir),
      };
      if (exec.DispatchThreads1D(stage_pipe, total_elements, 256, args, 0, stream) != cudaSuccess) {
        return CUFFT_EXEC_FAILED;
      }
      std::swap(*io_current, *io_scratch);
    }
    return CUFFT_SUCCESS;
  }

  id<MTLComputePipelineState> dft_pipe = GetCufftPipeline("muda_cufft_axis_dft");
  if (dft_pipe == nil) {
    return CUFFT_INTERNAL_ERROR;
  }
  const std::array<ShaderArg, 6> args = {
      ShaderArg::RawBuffer(*io_current, 0), ShaderArg::RawBuffer(*io_scratch, 0),
      ShaderArg::Value(total_elements),     ShaderArg::Value(axis_len),
      ShaderArg::Value(inner_stride),       ShaderArg::Value(sign_dir),
  };
  if (exec.DispatchThreads1D(dft_pipe, total_elements, 256, args, 0, stream) != cudaSuccess) {
    return CUFFT_EXEC_FAILED;
  }
  std::swap(*io_current, *io_scratch);
  return CUFFT_SUCCESS;
}

}  // namespace
}  // namespace muda::libraries::fft

using namespace muda::libraries;
using namespace muda::libraries::fft;

extern "C" {

cufftResult cufftExecC2C(cufftHandle plan, cufftComplex* idata, cufftComplex* odata,
                         int direction) {
  auto state = LookupPlan(plan);
  if (state == nullptr || !state->configured) {
    return CUFFT_INVALID_PLAN;
  }
  if (idata == nullptr || odata == nullptr ||
      (direction != CUFFT_FORWARD && direction != CUFFT_INVERSE)) {
    return CUFFT_INVALID_VALUE;
  }

  FftPlanConfig cfg;
  cudaStream_t stream = nullptr;
  id<MTLBuffer> cur_buf = nil;
  id<MTLBuffer> scr_buf = nil;
  {
    std::lock_guard<std::mutex> lock(state->mutex);
    if (state->config.type != CUFFT_C2C) {
      return CUFFT_INVALID_TYPE;
    }
    cfg = state->config;
    stream = state->stream;
    cur_buf = state->workspace_a;
    scr_buf = state->workspace_b;
  }

  std::uint32_t elem_per_batch = 1u;
  for (int i = 0; i < cfg.rank; ++i) {
    elem_per_batch *= static_cast<std::uint32_t>(cfg.dims[static_cast<std::size_t>(i)]);
  }
  const auto batch_u = static_cast<std::uint32_t>(cfg.batch);
  const std::uint32_t total_elems = elem_per_batch * batch_u;

  const std::size_t in_span =
      static_cast<std::size_t>(cfg.batch - 1) * static_cast<std::size_t>(cfg.idist) +
      static_cast<std::size_t>(elem_per_batch - 1u) * static_cast<std::size_t>(cfg.istride) + 1u;
  const std::size_t out_span =
      static_cast<std::size_t>(cfg.batch - 1) * static_cast<std::size_t>(cfg.odist) +
      static_cast<std::size_t>(elem_per_batch - 1u) * static_cast<std::size_t>(cfg.ostride) + 1u;

  auto& exec = LibraryExecutor::Instance();
  ResolvedBuffer buf_in;
  ResolvedBuffer buf_out;
  if (!exec.ResolveDeviceBuffer(idata, in_span * sizeof(cufftComplex), &buf_in) ||
      !exec.ResolveDeviceBuffer(odata, out_span * sizeof(cufftComplex), &buf_out)) {
    return CUFFT_INVALID_VALUE;
  }

  id<MTLComputePipelineState> gather_pipe = GetCufftPipeline("muda_cufft_gather_c2c");
  id<MTLComputePipelineState> scatter_pipe = GetCufftPipeline("muda_cufft_scatter_c2c");
  if (gather_pipe == nil || scatter_pipe == nil) {
    return CUFFT_INTERNAL_ERROR;
  }

  const auto u_istride = static_cast<std::uint32_t>(cfg.istride);
  const auto u_idist = static_cast<std::uint32_t>(cfg.idist);
  const std::array<ShaderArg, 6> gather_args = {
      ShaderArg::Buffer(buf_in),        ShaderArg::RawBuffer(cur_buf, 0),
      ShaderArg::Value(elem_per_batch), ShaderArg::Value(batch_u),
      ShaderArg::Value(u_istride),      ShaderArg::Value(u_idist),
  };
  if (exec.DispatchThreads1D(gather_pipe, total_elems, 256, gather_args, 0, stream) !=
      cudaSuccess) {
    return CUFFT_EXEC_FAILED;
  }

  const float sign_dir = (direction == CUFFT_FORWARD) ? -1.0f : 1.0f;
  std::uint32_t inner_stride = 1u;
  for (int axis = cfg.rank - 1; axis >= 0; --axis) {
    const auto axis_len = static_cast<std::uint32_t>(cfg.dims[static_cast<std::size_t>(axis)]);
    const cufftResult tr =
        TransformAxis(&cur_buf, &scr_buf, total_elems, axis_len, inner_stride, sign_dir, stream);
    if (tr != CUFFT_SUCCESS) {
      return tr;
    }
    inner_stride *= axis_len;
  }

  const auto u_ostride = static_cast<std::uint32_t>(cfg.ostride);
  const auto u_odist = static_cast<std::uint32_t>(cfg.odist);
  const std::array<ShaderArg, 6> scatter_args = {
      ShaderArg::RawBuffer(cur_buf, 0), ShaderArg::Buffer(buf_out),
      ShaderArg::Value(elem_per_batch), ShaderArg::Value(batch_u),
      ShaderArg::Value(u_ostride),      ShaderArg::Value(u_odist),
  };
  if (exec.DispatchThreads1D(scatter_pipe, total_elems, 256, scatter_args, 0, stream) !=
      cudaSuccess) {
    return CUFFT_EXEC_FAILED;
  }
  return CUFFT_SUCCESS;
}

cufftResult cufftExecR2C(cufftHandle plan, cufftReal* idata, cufftComplex* odata) {
  auto state = LookupPlan(plan);
  if (state == nullptr || !state->configured) {
    return CUFFT_INVALID_PLAN;
  }
  if (idata == nullptr || odata == nullptr) {
    return CUFFT_INVALID_VALUE;
  }

  FftPlanConfig cfg;
  cudaStream_t stream = nullptr;
  id<MTLBuffer> cur_buf = nil;
  id<MTLBuffer> scr_buf = nil;
  {
    std::lock_guard<std::mutex> lock(state->mutex);
    if (state->config.type != CUFFT_R2C) {
      return CUFFT_INVALID_TYPE;
    }
    cfg = state->config;
    stream = state->stream;
    cur_buf = state->workspace_a;
    scr_buf = state->workspace_b;
  }

  std::uint32_t elem_per_batch = 1u;
  for (int i = 0; i < cfg.rank; ++i) {
    elem_per_batch *= static_cast<std::uint32_t>(cfg.dims[static_cast<std::size_t>(i)]);
  }
  const auto last_dim =
      static_cast<std::uint32_t>(cfg.dims[static_cast<std::size_t>(cfg.rank - 1)]);
  const std::uint32_t herm_dim = (last_dim / 2u) + 1u;
  const std::uint32_t prefix_count = elem_per_batch / last_dim;
  const std::uint32_t herm_per_batch = prefix_count * herm_dim;
  const auto batch_u = static_cast<std::uint32_t>(cfg.batch);
  const std::uint32_t total_elems = elem_per_batch * batch_u;
  const std::uint32_t total_herm = herm_per_batch * batch_u;

  const std::size_t in_span =
      static_cast<std::size_t>(cfg.batch - 1) * static_cast<std::size_t>(cfg.idist) +
      static_cast<std::size_t>(elem_per_batch - 1u) * static_cast<std::size_t>(cfg.istride) + 1u;
  const std::size_t out_span =
      static_cast<std::size_t>(cfg.batch - 1) * static_cast<std::size_t>(cfg.odist) +
      static_cast<std::size_t>(herm_per_batch - 1u) * static_cast<std::size_t>(cfg.ostride) + 1u;

  auto& exec = LibraryExecutor::Instance();
  ResolvedBuffer buf_in;
  ResolvedBuffer buf_out;
  if (!exec.ResolveDeviceBuffer(idata, in_span * sizeof(cufftReal), &buf_in) ||
      !exec.ResolveDeviceBuffer(odata, out_span * sizeof(cufftComplex), &buf_out)) {
    return CUFFT_INVALID_VALUE;
  }

  id<MTLComputePipelineState> gather_pipe = GetCufftPipeline("muda_cufft_gather_r2c");
  id<MTLComputePipelineState> scatter_pipe = GetCufftPipeline("muda_cufft_scatter_r2c_hermitian");
  if (gather_pipe == nil || scatter_pipe == nil) {
    return CUFFT_INTERNAL_ERROR;
  }

  const auto u_istride = static_cast<std::uint32_t>(cfg.istride);
  const auto u_idist = static_cast<std::uint32_t>(cfg.idist);
  const std::array<ShaderArg, 6> gather_args = {
      ShaderArg::Buffer(buf_in),        ShaderArg::RawBuffer(cur_buf, 0),
      ShaderArg::Value(elem_per_batch), ShaderArg::Value(batch_u),
      ShaderArg::Value(u_istride),      ShaderArg::Value(u_idist),
  };
  if (exec.DispatchThreads1D(gather_pipe, total_elems, 256, gather_args, 0, stream) !=
      cudaSuccess) {
    return CUFFT_EXEC_FAILED;
  }

  std::uint32_t inner_stride = 1u;
  for (int axis = cfg.rank - 1; axis >= 0; --axis) {
    const auto axis_len = static_cast<std::uint32_t>(cfg.dims[static_cast<std::size_t>(axis)]);
    const cufftResult tr =
        TransformAxis(&cur_buf, &scr_buf, total_elems, axis_len, inner_stride, -1.0f, stream);
    if (tr != CUFFT_SUCCESS) {
      return tr;
    }
    inner_stride *= axis_len;
  }

  const auto u_ostride = static_cast<std::uint32_t>(cfg.ostride);
  const auto u_odist = static_cast<std::uint32_t>(cfg.odist);
  const std::array<ShaderArg, 8> scatter_args = {
      ShaderArg::RawBuffer(cur_buf, 0), ShaderArg::Buffer(buf_out),
      ShaderArg::Value(prefix_count),   ShaderArg::Value(last_dim),
      ShaderArg::Value(herm_dim),       ShaderArg::Value(batch_u),
      ShaderArg::Value(u_ostride),      ShaderArg::Value(u_odist),
  };
  if (exec.DispatchThreads1D(scatter_pipe, total_herm, 256, scatter_args, 0, stream) !=
      cudaSuccess) {
    return CUFFT_EXEC_FAILED;
  }
  return CUFFT_SUCCESS;
}

cufftResult cufftExecC2R(cufftHandle plan, cufftComplex* idata, cufftReal* odata) {
  auto state = LookupPlan(plan);
  if (state == nullptr || !state->configured) {
    return CUFFT_INVALID_PLAN;
  }
  if (idata == nullptr || odata == nullptr) {
    return CUFFT_INVALID_VALUE;
  }

  FftPlanConfig cfg;
  cudaStream_t stream = nullptr;
  id<MTLBuffer> cur_buf = nil;
  id<MTLBuffer> scr_buf = nil;
  {
    std::lock_guard<std::mutex> lock(state->mutex);
    if (state->config.type != CUFFT_C2R) {
      return CUFFT_INVALID_TYPE;
    }
    cfg = state->config;
    stream = state->stream;
    cur_buf = state->workspace_a;
    scr_buf = state->workspace_b;
  }

  std::uint32_t elem_per_batch = 1u;
  for (int i = 0; i < cfg.rank; ++i) {
    elem_per_batch *= static_cast<std::uint32_t>(cfg.dims[static_cast<std::size_t>(i)]);
  }
  const auto last_dim =
      static_cast<std::uint32_t>(cfg.dims[static_cast<std::size_t>(cfg.rank - 1)]);
  const std::uint32_t herm_dim = (last_dim / 2u) + 1u;
  const std::uint32_t prefix_count = elem_per_batch / last_dim;
  const std::uint32_t herm_per_batch = prefix_count * herm_dim;
  const auto batch_u = static_cast<std::uint32_t>(cfg.batch);
  const std::uint32_t total_elems = elem_per_batch * batch_u;
  const std::uint32_t total_herm = herm_per_batch * batch_u;

  const std::size_t in_span =
      static_cast<std::size_t>(cfg.batch - 1) * static_cast<std::size_t>(cfg.idist) +
      static_cast<std::size_t>(herm_per_batch - 1u) * static_cast<std::size_t>(cfg.istride) + 1u;
  const std::size_t out_span =
      static_cast<std::size_t>(cfg.batch - 1) * static_cast<std::size_t>(cfg.odist) +
      static_cast<std::size_t>(elem_per_batch - 1u) * static_cast<std::size_t>(cfg.ostride) + 1u;

  auto& exec = LibraryExecutor::Instance();
  ResolvedBuffer buf_in;
  ResolvedBuffer buf_out;
  if (!exec.ResolveDeviceBuffer(idata, in_span * sizeof(cufftComplex), &buf_in) ||
      !exec.ResolveDeviceBuffer(odata, out_span * sizeof(cufftReal), &buf_out)) {
    return CUFFT_INVALID_VALUE;
  }

  id<MTLComputePipelineState> gather_pipe = GetCufftPipeline("muda_cufft_gather_c2r_hermitian");
  id<MTLComputePipelineState> expand_pipe =
      GetCufftPipeline("muda_cufft_expand_hermitian_last_axis");
  id<MTLComputePipelineState> scatter_pipe = GetCufftPipeline("muda_cufft_scatter_c2r_real");
  if (gather_pipe == nil || expand_pipe == nil || scatter_pipe == nil) {
    return CUFFT_INTERNAL_ERROR;
  }

  const auto u_istride = static_cast<std::uint32_t>(cfg.istride);
  const auto u_idist = static_cast<std::uint32_t>(cfg.idist);
  const std::array<ShaderArg, 6> gather_args = {
      ShaderArg::Buffer(buf_in),        ShaderArg::RawBuffer(cur_buf, 0),
      ShaderArg::Value(herm_per_batch), ShaderArg::Value(batch_u),
      ShaderArg::Value(u_istride),      ShaderArg::Value(u_idist),
  };
  if (exec.DispatchThreads1D(gather_pipe, total_herm, 256, gather_args, 0, stream) != cudaSuccess) {
    return CUFFT_EXEC_FAILED;
  }

  std::uint32_t inner_stride = herm_dim;
  for (int axis = cfg.rank - 2; axis >= 0; --axis) {
    const auto axis_len = static_cast<std::uint32_t>(cfg.dims[static_cast<std::size_t>(axis)]);
    const cufftResult tr =
        TransformAxis(&cur_buf, &scr_buf, total_herm, axis_len, inner_stride, 1.0f, stream);
    if (tr != CUFFT_SUCCESS) {
      return tr;
    }
    inner_stride *= axis_len;
  }

  const std::array<ShaderArg, 6> expand_args = {
      ShaderArg::RawBuffer(cur_buf, 0), ShaderArg::RawBuffer(scr_buf, 0),
      ShaderArg::Value(prefix_count),   ShaderArg::Value(last_dim),
      ShaderArg::Value(herm_dim),       ShaderArg::Value(batch_u),
  };
  if (exec.DispatchThreads1D(expand_pipe, total_elems, 256, expand_args, 0, stream) !=
      cudaSuccess) {
    return CUFFT_EXEC_FAILED;
  }
  std::swap(cur_buf, scr_buf);

  const cufftResult last_tr =
      TransformAxis(&cur_buf, &scr_buf, total_elems, last_dim, 1u, 1.0f, stream);
  if (last_tr != CUFFT_SUCCESS) {
    return last_tr;
  }

  const auto u_ostride = static_cast<std::uint32_t>(cfg.ostride);
  const auto u_odist = static_cast<std::uint32_t>(cfg.odist);
  const std::array<ShaderArg, 6> scatter_args = {
      ShaderArg::RawBuffer(cur_buf, 0), ShaderArg::Buffer(buf_out),
      ShaderArg::Value(elem_per_batch), ShaderArg::Value(batch_u),
      ShaderArg::Value(u_ostride),      ShaderArg::Value(u_odist),
  };
  if (exec.DispatchThreads1D(scatter_pipe, total_elems, 256, scatter_args, 0, stream) !=
      cudaSuccess) {
    return CUFFT_EXEC_FAILED;
  }
  return CUFFT_SUCCESS;
}

}  // extern "C"

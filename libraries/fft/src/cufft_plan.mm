/**
 * @file cufft_plan.mm
 * @brief Implementation of cuFFT plan creation, configuration, caching, and lifecycle.
 */

#include "cufft_plan.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "muda/libraries/metal_executor.hpp"

namespace muda::libraries::fft {
namespace {

struct CachedWorkspacePair {
  id<MTLBuffer> workspace_a = nil;
  id<MTLBuffer> workspace_b = nil;
  std::size_t work_size_bytes = 0;
};

[[nodiscard]] LruPlanCache<FftPlanConfig, CachedWorkspacePair, FftPlanConfigHash>&
GetFftPlanCache() noexcept {
  static LruPlanCache<FftPlanConfig, CachedWorkspacePair, FftPlanConfigHash> cache(128);
  return cache;
}

[[nodiscard]] constexpr bool IsSupportedFftType(cufftType type) noexcept {
  return type == CUFFT_C2C || type == CUFFT_R2C || type == CUFFT_C2R;
}

[[nodiscard]] constexpr bool IsRejectedDoubleFftType(cufftType type) noexcept {
  return type == CUFFT_Z2Z || type == CUFFT_D2Z || type == CUFFT_Z2D;
}

[[nodiscard]] cufftResult ConfigurePlanState(const std::shared_ptr<CufftPlanState>& state, int rank,
                                             const int* dims, const int* inembed, int istride,
                                             int idist, const int* onembed, int ostride, int odist,
                                             cufftType type, int batch,
                                             std::size_t* out_work_size) noexcept {
  if (state == nullptr || dims == nullptr || rank < 1 || rank > 3 || batch <= 0) {
    return CUFFT_INVALID_VALUE;
  }
  if (IsRejectedDoubleFftType(type)) {
    return CUFFT_NOT_SUPPORTED;
  }
  if (!IsSupportedFftType(type)) {
    return CUFFT_INVALID_TYPE;
  }

  std::array<int, 3> norm_dims{1, 1, 1};
  std::size_t logical_elems_per_batch = 1;
  for (int i = 0; i < rank; ++i) {
    if (dims[i] <= 0) {
      return CUFFT_INVALID_SIZE;
    }
    norm_dims[static_cast<std::size_t>(i)] = dims[i];
    logical_elems_per_batch *= static_cast<std::size_t>(dims[i]);
  }

  const int last_dim = norm_dims[static_cast<std::size_t>(rank - 1)];
  const int hermitian_last = (last_dim / 2) + 1;
  const std::size_t prefix_prod =
      logical_elems_per_batch / static_cast<std::size_t>(last_dim);
  const std::size_t hermitian_elems_per_batch =
      prefix_prod * static_cast<std::size_t>(hermitian_last);

  int eff_istride = (inembed != nullptr) ? istride : 1;
  int eff_ostride = (onembed != nullptr) ? ostride : 1;
  if (eff_istride <= 0 || eff_ostride <= 0) {
    return CUFFT_INVALID_VALUE;
  }

  int default_idist = static_cast<int>(logical_elems_per_batch);
  int default_odist = static_cast<int>(logical_elems_per_batch);
  if (type == CUFFT_R2C) {
    default_odist = static_cast<int>(hermitian_elems_per_batch);
  } else if (type == CUFFT_C2R) {
    default_idist = static_cast<int>(hermitian_elems_per_batch);
  }

  const int eff_idist = (inembed != nullptr && idist > 0) ? idist : default_idist;
  const int eff_odist = (onembed != nullptr && odist > 0) ? odist : default_odist;

  const FftPlanConfig cfg{
      .rank = rank,
      .dims = norm_dims,
      .batch = batch,
      .type = type,
      .istride = eff_istride,
      .idist = eff_idist,
      .ostride = eff_ostride,
      .odist = eff_odist,
  };

  auto& exec = LibraryExecutor::Instance();
  if (exec.EnsureInitialized() != cudaSuccess) {
    return CUFFT_SETUP_FAILED;
  }
  id<MTLDevice> device = exec.Device();
  if (device == nil) {
    return CUFFT_INVALID_DEVICE;
  }

  const std::size_t total_complex_elems =
      logical_elems_per_batch * static_cast<std::size_t>(batch);
  const std::size_t buf_bytes = total_complex_elems * sizeof(cufftComplex);

  @autoreleasepool {
    CachedWorkspacePair ws_pair;
    auto& cache = GetFftPlanCache();
    if (auto cached = cache.Lookup(cfg); cached.has_value()) {
      ws_pair = *cached;
    } else {
      ws_pair.workspace_a = [device newBufferWithLength:buf_bytes
                                                options:MTLResourceStorageModeShared];
      ws_pair.workspace_b = [device newBufferWithLength:buf_bytes
                                                options:MTLResourceStorageModeShared];
      if (ws_pair.workspace_a == nil || ws_pair.workspace_b == nil) {
        return CUFFT_ALLOC_FAILED;
      }
      ws_pair.work_size_bytes = buf_bytes * 2u;
      cache.Insert(cfg, ws_pair);
    }

    std::lock_guard<std::mutex> lock(state->mutex);
    state->configured = true;
    state->config = cfg;
    state->work_size_bytes = ws_pair.work_size_bytes;
    state->workspace_a = ws_pair.workspace_a;
    state->workspace_b = ws_pair.workspace_b;
    if (out_work_size != nullptr) {
      *out_work_size = ws_pair.work_size_bytes;
    }
    return CUFFT_SUCCESS;
  }
}

}  // namespace

CufftHandleTable& GetCufftHandleTable() noexcept {
  static CufftHandleTable table;
  return table;
}

std::shared_ptr<CufftPlanState> LookupPlan(cufftHandle handle) noexcept {
  return GetCufftHandleTable().Lookup(handle);
}

}  // namespace muda::libraries::fft

using namespace muda::libraries;
using namespace muda::libraries::fft;

extern "C" {

cufftResult cufftCreate(cufftHandle* handle) {
  if (handle == nullptr) {
    return CUFFT_INVALID_VALUE;
  }
  if (LibraryExecutor::Instance().EnsureInitialized() != cudaSuccess) {
    return CUFFT_SETUP_FAILED;
  }
  auto state = std::make_shared<CufftPlanState>();
  *handle = GetCufftHandleTable().Insert<cufftHandle>(std::move(state));
  return CUFFT_SUCCESS;
}

cufftResult cufftMakePlan1d(cufftHandle plan, int nx, cufftType type, int batch,
                            size_t* workSize) {
  auto state = LookupPlan(plan);
  if (state == nullptr) {
    return CUFFT_INVALID_PLAN;
  }
  const int dims[1] = {nx};
  return ConfigurePlanState(state, 1, dims, nullptr, 1, 0, nullptr, 1, 0, type, batch, workSize);
}

cufftResult cufftMakePlan2d(cufftHandle plan, int nx, int ny, cufftType type, size_t* workSize) {
  auto state = LookupPlan(plan);
  if (state == nullptr) {
    return CUFFT_INVALID_PLAN;
  }
  const int dims[2] = {nx, ny};
  return ConfigurePlanState(state, 2, dims, nullptr, 1, 0, nullptr, 1, 0, type, 1, workSize);
}

cufftResult cufftMakePlan3d(cufftHandle plan, int nx, int ny, int nz, cufftType type,
                            size_t* workSize) {
  auto state = LookupPlan(plan);
  if (state == nullptr) {
    return CUFFT_INVALID_PLAN;
  }
  const int dims[3] = {nx, ny, nz};
  return ConfigurePlanState(state, 3, dims, nullptr, 1, 0, nullptr, 1, 0, type, 1, workSize);
}

cufftResult cufftMakePlanMany(cufftHandle plan, int rank, int* n, int* inembed, int istride,
                              int idist, int* onembed, int ostride, int odist, cufftType type,
                              int batch, size_t* workSize) {
  auto state = LookupPlan(plan);
  if (state == nullptr) {
    return CUFFT_INVALID_PLAN;
  }
  return ConfigurePlanState(state, rank, n, inembed, istride, idist, onembed, ostride, odist, type,
                            batch, workSize);
}

cufftResult cufftPlan1d(cufftHandle* plan, int nx, cufftType type, int batch) {
  if (plan == nullptr) {
    return CUFFT_INVALID_VALUE;
  }
  cufftHandle created = nullptr;
  const cufftResult cr = cufftCreate(&created);
  if (cr != CUFFT_SUCCESS) {
    return cr;
  }
  const cufftResult mr = cufftMakePlan1d(created, nx, type, batch, nullptr);
  if (mr != CUFFT_SUCCESS) {
    (void)cufftDestroy(created);
    return mr;
  }
  *plan = created;
  return CUFFT_SUCCESS;
}

cufftResult cufftPlan2d(cufftHandle* plan, int nx, int ny, cufftType type) {
  if (plan == nullptr) {
    return CUFFT_INVALID_VALUE;
  }
  cufftHandle created = nullptr;
  const cufftResult cr = cufftCreate(&created);
  if (cr != CUFFT_SUCCESS) {
    return cr;
  }
  const cufftResult mr = cufftMakePlan2d(created, nx, ny, type, nullptr);
  if (mr != CUFFT_SUCCESS) {
    (void)cufftDestroy(created);
    return mr;
  }
  *plan = created;
  return CUFFT_SUCCESS;
}

cufftResult cufftPlan3d(cufftHandle* plan, int nx, int ny, int nz, cufftType type) {
  if (plan == nullptr) {
    return CUFFT_INVALID_VALUE;
  }
  cufftHandle created = nullptr;
  const cufftResult cr = cufftCreate(&created);
  if (cr != CUFFT_SUCCESS) {
    return cr;
  }
  const cufftResult mr = cufftMakePlan3d(created, nx, ny, nz, type, nullptr);
  if (mr != CUFFT_SUCCESS) {
    (void)cufftDestroy(created);
    return mr;
  }
  *plan = created;
  return CUFFT_SUCCESS;
}

cufftResult cufftPlanMany(cufftHandle* plan, int rank, int* n, int* inembed, int istride,
                          int idist, int* onembed, int ostride, int odist, cufftType type,
                          int batch) {
  if (plan == nullptr) {
    return CUFFT_INVALID_VALUE;
  }
  cufftHandle created = nullptr;
  const cufftResult cr = cufftCreate(&created);
  if (cr != CUFFT_SUCCESS) {
    return cr;
  }
  const cufftResult mr = cufftMakePlanMany(created, rank, n, inembed, istride, idist, onembed,
                                           ostride, odist, type, batch, nullptr);
  if (mr != CUFFT_SUCCESS) {
    (void)cufftDestroy(created);
    return mr;
  }
  *plan = created;
  return CUFFT_SUCCESS;
}

cufftResult cufftSetStream(cufftHandle plan, cudaStream_t stream) {
  auto state = LookupPlan(plan);
  if (state == nullptr) {
    return CUFFT_INVALID_PLAN;
  }
  if (!LibraryExecutor::Instance().IsValidStream(stream)) {
    return CUFFT_INVALID_VALUE;
  }
  std::lock_guard<std::mutex> lock(state->mutex);
  state->stream = stream;
  return CUFFT_SUCCESS;
}

cufftResult cufftGetSize(cufftHandle handle, size_t* workSize) {
  if (workSize == nullptr) {
    return CUFFT_INVALID_VALUE;
  }
  auto state = LookupPlan(handle);
  if (state == nullptr || !state->configured) {
    return CUFFT_INVALID_PLAN;
  }
  std::lock_guard<std::mutex> lock(state->mutex);
  *workSize = state->work_size_bytes;
  return CUFFT_SUCCESS;
}

cufftResult cufftDestroy(cufftHandle plan) {
  auto removed = GetCufftHandleTable().Remove(plan);
  if (removed == nullptr) {
    return CUFFT_INVALID_PLAN;
  }
  return CUFFT_SUCCESS;
}

}  // extern "C"


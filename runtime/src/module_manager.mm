/**
 * @file module_manager.mm
 * @brief Implementation of in-memory PTX translation, Metal JIT compilation, and function caching.
 */

#include "muda/runtime/module_manager.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cuda.h"
#include "launch_engine.hpp"
#include "muda/compiler/cubin/cubin_translator.hpp"
#include "muda/compiler/ptx/ptx_to_mmir.hpp"
#include "muda/compiler/translator.hpp"
#include "muda/runtime/driver_types.hpp"

namespace muda::runtime::driver {
namespace {

constexpr std::string_view kNvrtcCudaMarker = "// MUDA_NVRTC_CUDA_SOURCE";
constexpr std::size_t kMaxCachedJitModules = 64;

struct CachedKernelInfo {
  std::string name;
  std::string msl_name;
  id<MTLFunction> function = nil;
  id<MTLComputePipelineState> pipeline = nil;
  std::size_t static_shared_bytes = 0;
  bool uses_dynamic_shared_bytes = false;
  std::vector<mudaKernelParamDesc> params;
};

struct CachedJitModule {
  std::uint64_t registry_id = 0;
  std::uint8_t fp64_policy = 0;
  std::string payload;
  id<MTLLibrary> library = nil;
  std::string msl_source;
  std::vector<CachedKernelInfo> kernels;
};

struct JitCacheKey {
  std::uint64_t registry_id = 0;
  std::uint8_t fp64_policy = 0;
  std::uint64_t hash = 0;
  std::size_t size = 0;

  bool operator==(const JitCacheKey&) const noexcept = default;
};

struct JitCacheKeyHash {
  std::size_t operator()(const JitCacheKey& k) const noexcept {
    std::uint64_t h = k.hash ^ (k.registry_id * 0x9e3779b97f4a7c15ULL);
    h ^= (static_cast<std::uint64_t>(k.fp64_policy) << 32) ^ k.size;
    return static_cast<std::size_t>(h);
  }
};

[[nodiscard]] std::span<const std::uint8_t> InferBoundedCubinSpan(
    std::span<const std::uint8_t> data) noexcept {
  if (data.size() >= 16) {
    std::uint32_t magic = 0;
    std::memcpy(&magic, data.data(), 4);
    if (magic == 0xBA55ED50u) {
      std::uint16_t hdr_size = 0;
      std::uint64_t total_size = 0;
      std::memcpy(&hdr_size, data.data() + 6, 2);
      std::memcpy(&total_size, data.data() + 8, 8);
      const std::size_t exact = static_cast<std::size_t>(hdr_size) + static_cast<std::size_t>(total_size);
      if (exact >= 16 && exact <= data.size()) {
        return data.subspan(0, exact);
      }
    } else if (data.size() >= 64 && data[0] == 0x7F && data[1] == 'E' && data[2] == 'L' &&
               data[3] == 'F') {
      std::uint64_t shoff = 0;
      std::uint16_t shentsize = 0;
      std::uint16_t shnum = 0;
      std::memcpy(&shoff, data.data() + 40, 8);
      std::memcpy(&shentsize, data.data() + 58, 2);
      std::memcpy(&shnum, data.data() + 60, 2);
      const std::size_t exact =
          static_cast<std::size_t>(shoff) + static_cast<std::size_t>(shentsize) * shnum;
      if (exact >= 64 && exact <= data.size()) {
        return data.subspan(0, exact);
      }
    }
  }
  return {};
}

[[nodiscard]] std::uint64_t HashBytesFnv1a(std::span<const std::uint8_t> bytes) noexcept {
  std::uint64_t h = 14695981039346656037ULL;
  for (std::uint8_t b : bytes) {
    h ^= b;
    h *= 1099511628211ULL;
  }
  return h;
}

class JitModuleCache {
 public:
  [[nodiscard]] std::shared_ptr<const CachedJitModule> Lookup(
      std::uint64_t registry_id, muda::compiler::Fp64Policy policy,
      std::span<const std::uint8_t> payload) noexcept {
    const JitCacheKey key{
        .registry_id = registry_id,
        .fp64_policy = static_cast<std::uint8_t>(policy),
        .hash = HashBytesFnv1a(payload),
        .size = payload.size(),
    };
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = entries_.find(key);
    if (it == entries_.end()) {
      return nullptr;
    }
    const std::string_view payload_sv(reinterpret_cast<const char*>(payload.data()),
                                      payload.size());
    if (it->second->payload != payload_sv) {
      return nullptr;
    }
    return it->second;
  }

  void Insert(std::uint64_t registry_id, muda::compiler::Fp64Policy policy,
              std::span<const std::uint8_t> payload,
              std::shared_ptr<const CachedJitModule> entry) noexcept {
    const JitCacheKey key{
        .registry_id = registry_id,
        .fp64_policy = static_cast<std::uint8_t>(policy),
        .hash = HashBytesFnv1a(payload),
        .size = payload.size(),
    };
    std::lock_guard<std::mutex> lock(mutex_);
    if (entries_.contains(key)) {
      entries_[key] = std::move(entry);
      return;
    }
    if (fifo_.size() >= kMaxCachedJitModules) {
      entries_.erase(fifo_.front());
      fifo_.pop_front();
    }
    fifo_.push_back(key);
    entries_.emplace(key, std::move(entry));
  }

 private:
  std::mutex mutex_;
  std::unordered_map<JitCacheKey, std::shared_ptr<const CachedJitModule>, JitCacheKeyHash> entries_;
  std::deque<JitCacheKey> fifo_;
};

JitModuleCache& GetJitModuleCache() noexcept {
  static JitModuleCache cache;
  return cache;
}

CUresult InstantiateModuleFromCache(const CachedJitModule& cached, CUmodule* out_module,
                                    cudart::LaunchEngine& launcher) noexcept {
  auto mod = std::make_unique<CUmod_st>();
  mod->library = cached.library;
  mod->source = cached.msl_source;

  for (const CachedKernelInfo& info : cached.kernels) {
    auto cu_fn = std::make_unique<CUfunc_st>();
    cu_fn->name = info.name;
    cu_fn->msl_name = info.msl_name;
    cu_fn->function = info.function;
    cu_fn->pipeline = info.pipeline;
    cu_fn->static_shared_bytes = info.static_shared_bytes;
    cu_fn->uses_dynamic_shared_bytes = info.uses_dynamic_shared_bytes;
    cu_fn->module = mod.get();
    cu_fn->params = info.params;

    launcher.RegisterKernelPipeline(cu_fn.get(), cu_fn->params, info.pipeline);
    mod->function_map[info.name] = cu_fn.get();
    if (info.msl_name.starts_with("k_")) {
      mod->function_map[info.msl_name.substr(2)] = cu_fn.get();
    }
    mod->functions.push_back(std::move(cu_fn));
  }

  *out_module = mod.release();
  return CUDA_SUCCESS;
}

CUresult CompileAndRegisterModule(const muda::compiler::TranslationResult& tr, CUmodule* out_module,
                                  id<MTLDevice> device, cudart::LaunchEngine& launcher,
                                  const muda::compiler::TranslationOptions& trans_opts,
                                  std::span<const std::uint8_t> cache_payload) noexcept {
  @autoreleasepool {
    NSString* msl_ns = [NSString stringWithUTF8String:tr.msl.c_str()];
    if (msl_ns == nil) {
      return CUDA_ERROR_INVALID_IMAGE;
    }

    MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    options.fastMathEnabled =
        (trans_opts.fp64_policy == muda::compiler::Fp64Policy::kOff) ? YES : NO;
#pragma clang diagnostic pop

    NSError* error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:msl_ns options:options error:&error];
    if (library == nil) {
      if (error != nil) {
        NSLog(@"[MUDA Metal JIT Error] %@\nSource:\n%@", error.localizedDescription, msl_ns);
      }
      return CUDA_ERROR_INVALID_IMAGE;
    }

    auto cached = std::make_shared<CachedJitModule>();
    cached->registry_id = device.registryID;
    cached->fp64_policy = static_cast<std::uint8_t>(trans_opts.fp64_policy);
    cached->payload.assign(reinterpret_cast<const char*>(cache_payload.data()),
                           cache_payload.size());
    cached->library = library;
    cached->msl_source = tr.msl;

    for (const muda::compiler::TranslatedKernel& tk : tr.kernels) {
      NSString* msl_fn_ns = [NSString stringWithUTF8String:tk.msl_name.c_str()];
      id<MTLFunction> fn = [library newFunctionWithName:msl_fn_ns];
      if (fn == nil) {
        continue;
      }

      id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn
                                                                              error:&error];
      if (pso == nil) {
        continue;
      }

      CachedKernelInfo info;
      info.name = tk.name;
      info.msl_name = tk.msl_name;
      info.function = fn;
      info.pipeline = pso;
      info.static_shared_bytes = tk.static_shared_memory_bytes;
      info.uses_dynamic_shared_bytes = tk.uses_dynamic_shared_memory;
      for (const muda::compiler::KernelParameter& p : tk.parameters) {
        mudaKernelParamDesc desc;
        desc.kind = p.is_pointer() ? mudaKernelParamBuffer : mudaKernelParamBytes;
        desc.size = p.byte_size();
        info.params.push_back(desc);
      }
      cached->kernels.push_back(std::move(info));
    }

    if (!cache_payload.empty()) {
      GetJitModuleCache().Insert(device.registryID, trans_opts.fp64_policy, cache_payload, cached);
    }
    return InstantiateModuleFromCache(*cached, out_module, launcher);
  }
}

}  // namespace

CUresult ModuleManager::LoadPtx(std::string_view ptx_source, CUmodule* out_module,
                                id<MTLDevice> device, cudart::LaunchEngine& launcher) noexcept {
  if (ptx_source.empty() || out_module == nullptr || device == nil) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  muda::compiler::TranslationOptions trans_opts;
  const char* env_fp64 = std::getenv("MUDA_FP64");
  if (env_fp64 != nullptr && env_fp64[0] != '\0') {
    trans_opts.fp64_policy = muda::compiler::ParseFp64Policy(env_fp64);
  } else {
    constexpr std::string_view kFp64PolicyMarker = "// MUDA_FP64_POLICY=";
    const std::size_t policy_pos = ptx_source.find(kFp64PolicyMarker);
    if (policy_pos != std::string_view::npos) {
      const std::size_t start = policy_pos + kFp64PolicyMarker.size();
      const std::size_t end = ptx_source.find('\n', start);
      std::string_view policy_name = (end == std::string_view::npos)
                                         ? ptx_source.substr(start)
                                         : ptx_source.substr(start, end - start);
      while (!policy_name.empty() && (policy_name.back() == '\r' || policy_name.back() == ' ')) {
        policy_name.remove_suffix(1);
      }
      trans_opts.fp64_policy = muda::compiler::ParseFp64Policy(policy_name);
    }
  }

  const std::span<const std::uint8_t> ptx_bytes(
      reinterpret_cast<const std::uint8_t*>(ptx_source.data()), ptx_source.size());
  if (auto cached =
          GetJitModuleCache().Lookup(device.registryID, trans_opts.fp64_policy, ptx_bytes)) {
    return InstantiateModuleFromCache(*cached, out_module, launcher);
  }

  muda::compiler::TranslationResult tr;
  const std::size_t nvrtc_pos = ptx_source.find(kNvrtcCudaMarker);
  if (nvrtc_pos != std::string_view::npos) {
    std::string_view cuda_src = ptx_source.substr(nvrtc_pos + kNvrtcCudaMarker.size());
    while (!cuda_src.empty() && (cuda_src.front() == '\n' || cuda_src.front() == '\r')) {
      cuda_src.remove_prefix(1);
    }
    tr = muda::compiler::TranslateCudaToMsl(cuda_src, trans_opts);
  } else {
    tr = muda::compiler::ptx::TranslatePtxToMsl(ptx_source, trans_opts);
  }
  if (!tr.ok()) {
    return CUDA_ERROR_INVALID_PTX;
  }

  return CompileAndRegisterModule(tr, out_module, device, launcher, trans_opts, ptx_bytes);
}

CUresult ModuleManager::LoadCubin(std::span<const std::uint8_t> cubin_data, CUmodule* out_module,
                                  id<MTLDevice> device, cudart::LaunchEngine& launcher) noexcept {
  if (cubin_data.empty() || out_module == nullptr || device == nil) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  muda::compiler::TranslationOptions trans_opts;
  const char* env_fp64 = std::getenv("MUDA_FP64");
  if (env_fp64 != nullptr && env_fp64[0] != '\0') {
    trans_opts.fp64_policy = muda::compiler::ParseFp64Policy(env_fp64);
  }

  const std::span<const std::uint8_t> bounded_cubin = InferBoundedCubinSpan(cubin_data);
  if (!bounded_cubin.empty()) {
    if (auto cached = GetJitModuleCache().Lookup(device.registryID, trans_opts.fp64_policy,
                                                 bounded_cubin)) {
      return InstantiateModuleFromCache(*cached, out_module, launcher);
    }
  }

  muda::compiler::TranslationResult tr =
      muda::compiler::cubin::TranslateCubinOrFatbinToMsl(cubin_data, trans_opts);
  if (!tr.ok()) {
    return CUDA_ERROR_INVALID_IMAGE;
  }

  return CompileAndRegisterModule(tr, out_module, device, launcher, trans_opts, bounded_cubin);
}

CUresult ModuleManager::LoadMetallibData(const void* data, std::size_t size, CUmodule* out_module,
                                         id<MTLDevice> device,
                                         [[maybe_unused]] cudart::LaunchEngine& launcher) noexcept {
  if (data == nullptr || size == 0 || out_module == nullptr || device == nil) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  @autoreleasepool {
    dispatch_data_t ddata = dispatch_data_create(data, size, dispatch_get_main_queue(),
                                                 DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    if (ddata == nil) {
      return CUDA_ERROR_INVALID_IMAGE;
    }

    NSError* error = nil;
    id<MTLLibrary> library = [device newLibraryWithData:ddata error:&error];
    if (library == nil) {
      return CUDA_ERROR_INVALID_IMAGE;
    }

    auto mod = std::make_unique<CUmod_st>();
    mod->library = library;
    *out_module = mod.release();
    return CUDA_SUCCESS;
  }
}

CUresult ModuleManager::GetFunction(CUfunction* hfunc, CUmodule hmod, const char* name) noexcept {
  if (hfunc == nullptr || hmod == nullptr || name == nullptr || name[0] == '\0') {
    return CUDA_ERROR_INVALID_VALUE;
  }

  const auto it = hmod->function_map.find(name);
  if (it != hmod->function_map.end()) {
    *hfunc = it->second;
    return CUDA_SUCCESS;
  }

  for (const auto& [fn_name, func_ptr] : hmod->function_map) {
    if (fn_name.find(name) != std::string::npos ||
        func_ptr->msl_name.find(name) != std::string::npos) {
      *hfunc = func_ptr;
      return CUDA_SUCCESS;
    }
  }

  return CUDA_ERROR_NOT_FOUND;
}

CUresult ModuleManager::Unload(CUmodule hmod) noexcept {
  if (hmod == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  delete hmod;
  return CUDA_SUCCESS;
}

}  // namespace muda::runtime::driver

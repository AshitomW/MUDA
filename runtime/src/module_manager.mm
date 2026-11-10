/**
 * @file module_manager.mm
 * @brief Implementation of in-memory PTX translation, Metal JIT compilation, and function caching.
 */

#include "muda/runtime/module_manager.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "cuda.h"
#include "launch_engine.hpp"
#include "muda/compiler/ptx/ptx_to_mmir.hpp"
#include "muda/compiler/translator.hpp"
#include "muda/runtime/driver_types.hpp"

namespace muda::runtime::driver {
namespace {

constexpr std::string_view kNvrtcCudaMarker = "// MUDA_NVRTC_CUDA_SOURCE";

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

  @autoreleasepool {
    NSString* msl_ns = [NSString stringWithUTF8String:tr.msl.c_str()];
    if (msl_ns == nil) {
      return CUDA_ERROR_INVALID_PTX;
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

    auto mod = std::make_unique<CUmod_st>();
    mod->library = library;
    mod->source = tr.msl;

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

      auto cu_fn = std::make_unique<CUfunc_st>();
      cu_fn->name = tk.name;
      cu_fn->msl_name = tk.msl_name;
      cu_fn->function = fn;
      cu_fn->pipeline = pso;
      cu_fn->static_shared_bytes = tk.static_shared_memory_bytes;
      cu_fn->uses_dynamic_shared_bytes = tk.uses_dynamic_shared_memory;
      cu_fn->module = mod.get();

      for (const muda::compiler::KernelParameter& p : tk.parameters) {
        mudaKernelParamDesc desc;
        desc.kind = p.is_pointer() ? mudaKernelParamBuffer : mudaKernelParamBytes;
        desc.size = p.byte_size();
        cu_fn->params.push_back(desc);
      }

      launcher.RegisterKernelPipeline(cu_fn.get(), cu_fn->params, pso);
      mod->function_map[tk.name] = cu_fn.get();
      if (tk.msl_name.starts_with("k_")) {
        mod->function_map[tk.msl_name.substr(2)] = cu_fn.get();
      }
      mod->functions.push_back(std::move(cu_fn));
    }

    *out_module = mod.release();
    return CUDA_SUCCESS;
  }
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

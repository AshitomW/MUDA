/**
 * @file driver_types.hpp
 * @brief Internal handle definitions for CUDA Driver API contexts, modules, and functions.
 */

#pragma once

#import <Metal/Metal.h>

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "cuda.h"
#include "cuda_runtime.h"

struct CUctx_st {
  int device_ordinal = 0;
  unsigned int flags = 0;
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
};

struct CUfunc_st {
  std::string name;
  std::string msl_name;
  id<MTLFunction> function = nil;
  id<MTLComputePipelineState> pipeline = nil;
  std::vector<mudaKernelParamDesc> params;
  std::size_t static_shared_bytes = 0;
  bool uses_dynamic_shared_bytes = false;
  CUmodule module = nullptr;
};

struct CUmod_st {
  id<MTLLibrary> library = nil;
  std::string source;
  std::vector<std::unique_ptr<CUfunc_st>> functions;
  std::unordered_map<std::string, CUfunc_st*> function_map;
};


/**
 * @file translator_test.cpp
 * @brief Tests CUDA-to-MSL translation and diagnostic codes.
 */

#include "muda/compiler/translator.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

int main() {
  const std::filesystem::path source_path =
      std::filesystem::path(MUDA_SOURCE_DIR) / "samples/vector_add.cu";
  std::ifstream source_file(source_path, std::ios::binary);
  if (!source_file) {
    std::cerr << "cannot open " << source_path << "\n";
    return 1;
  }

  std::ostringstream source_buffer;
  source_buffer << source_file.rdbuf();
  const std::string source = source_buffer.str();

  const muda::compiler::TranslationResult result = muda::compiler::TranslateCudaToMsl(source);
  if (!result.ok()) {
    std::cerr << "translation failed [" << result.diagnostic_id() << "]: " << result.error()
              << "\n";
    return 1;
  }
  if (result.kernel.name != "vector_add") {
    std::cerr << "unexpected kernel name: " << result.kernel.name << "\n";
    return 1;
  }
  if (result.kernel.parameters.size() != 4) {
    std::cerr << "unexpected parameter count\n";
    return 1;
  }
  const auto& param0 = result.kernel.parameters[0];
  const auto& param2 = result.kernel.parameters[2];
  const auto& param3 = result.kernel.parameters[3];
  if (param0.name != "a" ||
      param0.type != muda::compiler::DataType{muda::compiler::ScalarType::kFloat32,
                                              muda::compiler::VectorWidth::k1} ||
      param0.address_space != muda::compiler::AddressSpace::kDevice ||
      param0.access != muda::compiler::AccessQualifier::kReadOnly) {
    std::cerr << "unexpected parameter[0] metadata\n";
    return 1;
  }
  if (param2.access != muda::compiler::AccessQualifier::kReadWrite) {
    std::cerr << "unexpected parameter[2] mutability\n";
    return 1;
  }
  if (param3.name != "n" ||
      param3.type != muda::compiler::DataType{muda::compiler::ScalarType::kInt32,
                                              muda::compiler::VectorWidth::k1} ||
      param3.address_space != muda::compiler::AddressSpace::kConstant || param3.byte_size() != 4) {
    std::cerr << "unexpected parameter[3] metadata\n";
    return 1;
  }

  const std::string& msl = result.kernel.msl;
  const char* required[] = {
      "kernel void vector_add(",
      "device const float* a [[buffer(0)]]",
      "device const float* b [[buffer(1)]]",
      "device float* out [[buffer(2)]]",
      "constant int& n [[buffer(3)]]",
      "threadgroup_position_in_grid.x",
      "threads_per_threadgroup.x",
      "thread_position_in_threadgroup.x",
  };
  for (const char* text : required) {
    if (msl.find(text) == std::string::npos) {
      std::cerr << "missing MSL fragment: " << text << "\n";
      return 1;
    }
  }

  const muda::compiler::TranslationResult invalid =
      muda::compiler::TranslateCudaToMsl("int main() { return 0; }");
  if (invalid.ok() || invalid.code != muda::compiler::DiagnosticCode::kMissingGlobalKernel) {
    std::cerr << "invalid input should fail with kMissingGlobalKernel\n";
    return 1;
  }

  const muda::compiler::TranslationResult unsupported =
      muda::compiler::TranslateCudaToMsl("__global__ void k(double* x) {}");
  if (unsupported.ok() ||
      unsupported.code != muda::compiler::DiagnosticCode::kUnsupportedParameterType) {
    std::cerr << "double parameter should fail with kUnsupportedParameterType\n";
    return 1;
  }

  const muda::compiler::TranslationResult vector_types =
      muda::compiler::TranslateCudaToMsl("__global__ void k(const float4* a, ulong4* b) {}");
  if (!vector_types.ok() || vector_types.kernel.parameters.size() != 2 ||
      vector_types.kernel.parameters[0].metal_type() != "float4" ||
      vector_types.kernel.parameters[1].metal_type() != "ulong4") {
    std::cerr << "float4 and ulong4 vector parameters should translate cleanly\n";
    return 1;
  }

  // Test __device__ helper function + static __shared__ + math intrinsics + multiple kernels
  const char* complex_source = R"CUDA(
__device__ float clamp_val(float x, float lo, float hi) {
  return fminf(fmaxf(fmaf(x, 1.0f, 0.0f), lo), hi);
}

__global__ void reduce_block(const float* in, float* out, int n) {
  __shared__ float tile[256];
  const int tid = threadIdx.x;
  const int idx = blockIdx.x * blockDim.x + tid;
  tile[tid] = (idx < n) ? clamp_val(in[idx], -10.0f, 10.0f) : 0.0f;
  __syncthreads();
  if (tid == 0) {
    out[blockIdx.x] = tile[0];
  }
}

__global__ void dyn_shared_kernel(const float* in, float* out, int n) {
  extern __shared__ float dyn_buf[];
  const int tid = threadIdx.x;
  dyn_buf[tid] = in[tid];
  __syncthreads();
  out[tid] = dyn_buf[tid];
}
)CUDA";

  const muda::compiler::TranslationResult complex_res =
      muda::compiler::TranslateCudaToMsl(complex_source);
  if (!complex_res.ok() || complex_res.kernels.size() != 2) {
    std::cerr << "complex multi-kernel translation failed [" << complex_res.diagnostic_id()
              << "]: " << complex_res.error() << "\n";
    return 1;
  }
  if (complex_res.msl.find("inline float clamp_val(") == std::string::npos ||
      complex_res.msl.find("threadgroup float tile[256];") == std::string::npos ||
      complex_res.msl.find("threadgroup float* dyn_buf [[threadgroup(0)]]") == std::string::npos ||
      complex_res.msl.find("min(max(fma(") == std::string::npos) {
    std::cerr << "complex MSL missing expected lowered constructs:\n" << complex_res.msl << "\n";
    return 1;
  }

  // Rejection tests for recursion, function pointers, and dynamic parallelism
  const muda::compiler::TranslationResult recursive_fn = muda::compiler::TranslateCudaToMsl(
      "__device__ int fact(int n) { return n <= 1 ? 1 : n * fact(n - 1); }\n"
      "__global__ void k(int* out) { out[0] = fact(5); }");
  if (recursive_fn.ok() ||
      recursive_fn.code != muda::compiler::DiagnosticCode::kRecursionUnsupported) {
    std::cerr << "recursive device function should fail with ALY-C-101\n";
    return 1;
  }

  const muda::compiler::TranslationResult fn_ptr =
      muda::compiler::TranslateCudaToMsl("__global__ void k(void (*fn)(int), int* out) {}");
  if (fn_ptr.ok() || fn_ptr.code != muda::compiler::DiagnosticCode::kFunctionPointerUnsupported) {
    std::cerr << "function pointer parameter should fail with ALY-C-102\n";
    return 1;
  }

  const muda::compiler::TranslationResult dyn_par = muda::compiler::TranslateCudaToMsl(
      "__global__ void child() {}\n"
      "__global__ void parent() { child<<<1, 1>>>(); }");
  if (dyn_par.ok() ||
      dyn_par.code != muda::compiler::DiagnosticCode::kDynamicParallelismUnsupported) {
    std::cerr << "dynamic parallelism should fail with ALY-C-106\n";
    return 1;
  }

  std::cout << "translated vector_add and verified language constructs\n";
  return 0;
}

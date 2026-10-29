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

  std::cout << "translated vector_add\n";
  return 0;
}

/**
 * @file translator.cpp
 * @brief Implements CUDA-to-MMIR extraction and MSL translation orchestration.
 */

#include "muda/compiler/translator.hpp"

#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "ast_extractor.hpp"
#include "muda/compiler/mmir.hpp"
#include "type_table.hpp"

namespace muda::compiler {
namespace {

TranslationResult Failure(DiagnosticCode code) {
  TranslationResult result;
  result.code = code;
  return result;
}

DiagnosticCode PopulateDeviceFunction(const RawFunctionSlice& slice, mmir::DeviceFunction& dev_fn) {
  dev_fn.name = slice.name;
  if (slice.return_type == "void") {
    dev_fn.returns_void = true;
  } else {
    const std::optional<DataType> ret_type = ResolveCudaType(slice.return_type);
    if (!ret_type.has_value()) {
      return DiagnosticCode::kUnsupportedParameterType;
    }
    dev_fn.return_type = *ret_type;
  }

  const DiagnosticCode param_status = ParseParameterList(slice.parameter_list, &dev_fn.parameters);
  if (param_status != DiagnosticCode::kOk) {
    return param_status;
  }

  return ExtractSharedMemoryAndLowerBody(slice.body, nullptr, &dev_fn.lowered_body);
}

DiagnosticCode PopulateKernelFunction(const RawFunctionSlice& slice,
                                      mmir::KernelFunction& kernel_fn) {
  kernel_fn.name = slice.name;
  kernel_fn.msl_name = slice.name;

  const DiagnosticCode param_status =
      ParseParameterList(slice.parameter_list, &kernel_fn.parameters);
  if (param_status != DiagnosticCode::kOk) {
    return param_status;
  }

  return ExtractSharedMemoryAndLowerBody(slice.body, &kernel_fn.shared_allocations,
                                         &kernel_fn.lowered_body);
}

}  // namespace

TranslationResult TranslateCudaToMsl(std::string_view source) {
  std::vector<RawFunctionSlice> slices;
  const DiagnosticCode extract_status = ExtractFunctionsFromSource(source, &slices);
  if (extract_status != DiagnosticCode::kOk) {
    return Failure(extract_status);
  }

  mmir::Module module;
  for (const RawFunctionSlice& slice : slices) {
    if (slice.kind == QualifierKind::kDevice) {
      mmir::DeviceFunction dev_fn;
      const DiagnosticCode status = PopulateDeviceFunction(slice, dev_fn);
      if (status != DiagnosticCode::kOk) {
        return Failure(status);
      }
      module.device_functions.push_back(std::move(dev_fn));
    } else {
      mmir::KernelFunction kernel_fn;
      const DiagnosticCode status = PopulateKernelFunction(slice, kernel_fn);
      if (status != DiagnosticCode::kOk) {
        return Failure(status);
      }
      module.kernels.push_back(std::move(kernel_fn));
    }
  }

  const DiagnosticCode verify_status = mmir::VerifyModule(module);
  if (verify_status != DiagnosticCode::kOk) {
    return Failure(verify_status);
  }

  TranslationResult result;
  result.code = DiagnosticCode::kOk;
  result.msl = mmir::EmitMslModule(module);

  result.kernels.reserve(module.kernels.size());
  for (const mmir::KernelFunction& kernel_fn : module.kernels) {
    TranslatedKernel tk;
    tk.name = kernel_fn.name;
    tk.msl_name = kernel_fn.msl_name;
    tk.parameters = kernel_fn.parameters;
    tk.uses_dynamic_shared_memory = kernel_fn.uses_dynamic_shared_memory();
    tk.msl = mmir::EmitMslKernel(module, kernel_fn);
    result.kernels.push_back(std::move(tk));
  }

  result.kernel = result.kernels.front();
  result.kernel.msl = result.msl;
  return result;
}

}  // namespace muda::compiler

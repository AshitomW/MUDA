/**
 * @file cubin_translator.cpp
 * @brief Unified translation entry point for NVIDIA Fatbin containers and ELF CUBIN binaries to MSL.
 */

#include "muda/compiler/cubin/cubin_translator.hpp"

#include <string>
#include <vector>

#include "muda/compiler/cubin/elf_reader.hpp"
#include "muda/compiler/cubin/fatbin.hpp"
#include "muda/compiler/cubin/sass_decoder.hpp"
#include "muda/compiler/cubin/sass_lifter.hpp"
#include "muda/compiler/ptx/ptx_to_mmir.hpp"

namespace muda::compiler::cubin {

bool IsFatbinary(std::span<const std::uint8_t> data) noexcept {
  return FatbinParser::IsFatbinary(data);
}

bool IsElfCubin(std::span<const std::uint8_t> data) noexcept {
  return ElfReader::IsCudaCubin(data);
}

TranslationResult TranslateCubinOrFatbinToMsl(std::span<const std::uint8_t> data,
                                             const TranslationOptions& options) {
  TranslationResult result;

  // 1. Check if the buffer is a Fatbinary container
  if (IsFatbinary(data)) {
    // Attempt to extract embedded PTX first (highest ROI, exact semantic fidelity)
    auto ptx_payload = FatbinParser::ExtractBestPtx(data);
    if (ptx_payload.has_value() && !ptx_payload->text.empty()) {
      return ptx::TranslatePtxToMsl(ptx_payload->text, options);
    }

    // Otherwise extract the best CUBIN from the fatbin
    auto cubin_payload = FatbinParser::ExtractBestCubin(data);
    if (cubin_payload.has_value() && !cubin_payload->raw_bytes.empty()) {
      return TranslateCubinOrFatbinToMsl(cubin_payload->raw_bytes, options);
    }

    result.code = DiagnosticCode::kMissingGlobalKernel;
    return result;
  }

  // 2. Check if the buffer is a direct ELF CUBIN
  if (!IsElfCubin(data)) {
    result.code = DiagnosticCode::kMissingGlobalKernel;
    return result;
  }

  std::vector<CubinKernel> kernels = ElfReader::ExtractKernels(data);
  if (kernels.empty()) {
    result.code = DiagnosticCode::kMissingGlobalKernel;
    return result;
  }

  std::string full_msl;
  for (const CubinKernel& k : kernels) {
    std::vector<SassInstruction> insts = SassDecoder::Decode(k.text_bytes, k.arch_sm);
    std::string msl = SassLifter::LiftToMsl(k, insts);
    full_msl += msl + "\n\n";

    TranslatedKernel tk;
    tk.name = k.name;
    tk.msl_name = "k_" + k.name;
    tk.static_shared_memory_bytes = k.static_shared_bytes;
    tk.uses_dynamic_shared_memory = (k.static_shared_bytes == 0);
    tk.msl = msl;

    for (const CubinKernelParam& p : k.params) {
      KernelParameter kp;
      kp.name = "p" + std::to_string(p.index);
      kp.buffer_index = p.index;
      if (p.is_pointer) {
        kp.binding_kind = ParameterBindingKind::kBufferPointer;
        kp.address_space = AddressSpace::kDevice;
        kp.type = DataType{ScalarType::kUInt8, VectorWidth::k1};
      } else {
        kp.binding_kind = ParameterBindingKind::kConstantValue;
        kp.address_space = AddressSpace::kConstant;
        kp.type = DataType{ScalarType::kUInt32, VectorWidth::k1};
      }
      tk.parameters.push_back(kp);
    }

    result.kernels.push_back(std::move(tk));
  }

  if (!result.kernels.empty()) {
    result.kernel = result.kernels.front();
    result.msl = full_msl;
    result.code = DiagnosticCode::kOk;
  }

  return result;
}

}  // namespace muda::compiler::cubin


/**
 * @file muda_wasm.cpp
 * @brief WebAssembly bridge exporting MUDA compiler translation via Emscripten embind.
 *
 * Implements isolated client-side translation of CUDA C++ and PTX assembly into
 * Metal Shading Language (MSL) with complete kernel and parameter reflection.
 */

#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

#include "muda/compiler/ptx/ptx_to_mmir.hpp"
#include "muda/compiler/translator.hpp"
#include "muda/compiler/types.hpp"

namespace muda::wasm {
namespace {

emscripten::val SerializeResult(const compiler::TranslationResult& result) {
  emscripten::val obj = emscripten::val::object();
  obj.set("ok", result.ok());
  obj.set("code", static_cast<int>(result.code));
  obj.set("diagnostic_id", std::string(result.diagnostic_id()));
  obj.set("error", std::string(result.error()));
  obj.set("msl", result.msl);

  emscripten::val kernels = emscripten::val::array();
  for (std::size_t i = 0; i < result.kernels.size(); ++i) {
    const auto& k = result.kernels[i];
    emscripten::val k_obj = emscripten::val::object();
    k_obj.set("name", k.name);
    k_obj.set("msl_name", k.msl_name);
    k_obj.set("static_shared_memory_bytes", static_cast<double>(k.static_shared_memory_bytes));
    k_obj.set("uses_dynamic_shared_memory", k.uses_dynamic_shared_memory);
    k_obj.set("msl", k.msl);

    emscripten::val params = emscripten::val::array();
    for (std::size_t j = 0; j < k.parameters.size(); ++j) {
      const auto& p = k.parameters[j];
      emscripten::val p_obj = emscripten::val::object();
      p_obj.set("name", p.name);
      p_obj.set("is_pointer", p.is_pointer());
      p_obj.set("is_const", p.is_const());
      p_obj.set("buffer_index", static_cast<double>(p.buffer_index));
      p_obj.set("metal_type", std::string(p.metal_type()));
      p_obj.set("byte_size", static_cast<double>(p.byte_size()));
      params.set(static_cast<unsigned int>(j), p_obj);
    }
    k_obj.set("parameters", params);
    kernels.set(static_cast<unsigned int>(i), k_obj);
  }
  obj.set("kernels", kernels);
  return obj;
}

}  // namespace

emscripten::val TranslateCuda(const std::string& source) {
  const compiler::TranslationResult res = compiler::TranslateCudaToMsl(source);
  return SerializeResult(res);
}

emscripten::val TranslatePtx(const std::string& ptx_source) {
  const compiler::TranslationResult res = compiler::ptx::TranslatePtxToMsl(ptx_source);
  return SerializeResult(res);
}

std::string GetVersion() {
  return "0.1.0";
}

}  // namespace muda::wasm

EMSCRIPTEN_BINDINGS(muda_wasm_module) {
  emscripten::function("translateCuda", &muda::wasm::TranslateCuda);
  emscripten::function("translatePtx", &muda::wasm::TranslatePtx);
  emscripten::function("getVersion", &muda::wasm::GetVersion);
}


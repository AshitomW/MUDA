#!/usr/bin/env bash
# ==============================================================================
# build_wasm.sh — Build MUDA Compiler WebAssembly Engine using Emscripten
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT_DIR="${REPO_ROOT}/web/dist"

mkdir -p "${OUTPUT_DIR}"

# Locate emcc
EMCC_CMD=""
if command -v emcc >/dev/null 2>&1; then
  EMCC_CMD="emcc"
elif [[ -n "${EMSDK:-}" && -x "${EMSDK}/upstream/emscripten/emcc" ]]; then
  EMCC_CMD="${EMSDK}/upstream/emscripten/emcc"
elif [[ -x "${HOME}/emsdk/upstream/emscripten/emcc" ]]; then
  EMCC_CMD="${HOME}/emsdk/upstream/emscripten/emcc"
fi

if [[ -z "${EMCC_CMD}" ]]; then
  echo "Error: Emscripten compiler (emcc) not found in PATH or EMSDK."
  echo "Please install Emscripten or activate emsdk environment:"
  echo "  source /path/to/emsdk/emsdk_env.sh"
  exit 1
fi

echo "==> Using Emscripten: $(${EMCC_CMD} --version | head -n 1)"
echo "==> Target output: ${OUTPUT_DIR}/muda_compiler.js (+ .wasm)"

SOURCES=(
  "${REPO_ROOT}/compiler/src/ast_extractor.cpp"
  "${REPO_ROOT}/compiler/src/translator.cpp"
  "${REPO_ROOT}/compiler/src/mmir.cpp"
  "${REPO_ROOT}/compiler/src/type_table.cpp"
  "${REPO_ROOT}/compiler/src/ptx/arena.cpp"
  "${REPO_ROOT}/compiler/src/ptx/cfg.cpp"
  "${REPO_ROOT}/compiler/src/ptx/lexer.cpp"
  "${REPO_ROOT}/compiler/src/ptx/parser.cpp"
  "${REPO_ROOT}/compiler/src/ptx/ptx_to_mmir.cpp"
  "${REPO_ROOT}/compiler/src/ptx/structurizer.cpp"
  "${REPO_ROOT}/compiler/src/ptx/tokens.cpp"
  "${REPO_ROOT}/compiler/src/ptx/type_inference.cpp"
  "${SCRIPT_DIR}/muda_wasm.cpp"
)

INCLUDES=(
  "-I${REPO_ROOT}/compiler/include"
  "-I${REPO_ROOT}/compiler/src"
  "-I${SCRIPT_DIR}"
)

FLAGS=(
  "-std=c++20"
  "-O3"
  "-flto"
  "-s" "WASM=1"
  "-s" "MODULARIZE=1"
  "-s" "EXPORT_NAME=createMudaModule"
  "-s" "ALLOW_MEMORY_GROWTH=1"
  "-s" "NO_FILESYSTEM=1"
  "-s" "ENVIRONMENT=web,worker,node"
  "-s" "EXPORTED_RUNTIME_METHODS=['ccall','cwrap']"
  "--bind"
)

echo "==> Compiling MUDA WebAssembly engine..."
"${EMCC_CMD}" "${FLAGS[@]}" "${INCLUDES[@]}" "${SOURCES[@]}" -o "${OUTPUT_DIR}/muda_compiler.js"

echo "==> Build successful!"
ls -lh "${OUTPUT_DIR}/muda_compiler.js" "${OUTPUT_DIR}/muda_compiler.wasm"


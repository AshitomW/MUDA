cmake_minimum_required(VERSION 3.20)
project(muda_wasm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")

add_executable(muda_compiler
  muda_wasm.cpp
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
)

target_include_directories(muda_compiler PRIVATE
  "${REPO_ROOT}/compiler/include"
  "${REPO_ROOT}/compiler/src"
  "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_compile_options(muda_compiler PRIVATE -O3 -flto)

target_link_options(muda_compiler PRIVATE
  -O3
  -flto
  -sWASM=1
  -sMODULARIZE=1
  -sEXPORT_NAME="createMudaModule"
  -sALLOW_MEMORY_GROWTH=1
  -sNO_FILESYSTEM=1
  -sENVIRONMENT="web,worker,node"
  --bind
)

set_target_properties(muda_compiler PROPERTIES
  OUTPUT_NAME "muda_compiler"
  SUFFIX ".js"
  RUNTIME_OUTPUT_DIRECTORY "${REPO_ROOT}/web/dist"
)


add_library(
  muda_compiler
  src/cache.cpp
  src/driver.cpp
  src/host_rewriter.cpp
  src/mmir.cpp
  src/process.cpp
  src/toolchain.cpp
  src/translator.cpp
)
add_library(muda::compiler ALIAS muda_compiler)

target_include_directories(
  muda_compiler
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${MUDA_GENERATED_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_definitions(
  muda_compiler
  PRIVATE
    MUDA_DEFAULT_RUNTIME_INCLUDE_DIR="${PROJECT_SOURCE_DIR}/runtime/include"
    MUDA_DEFAULT_RUNTIME_LIB_FILE="$<TARGET_FILE:muda_runtime>"
)

target_compile_features(muda_compiler PUBLIC cxx_std_20)
target_link_libraries(muda_compiler PRIVATE muda_warnings)

install(TARGETS muda_compiler ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

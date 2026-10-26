cmake_minimum_required(VERSION 3.28)

if(APPLE AND NOT DEFINED CMAKE_OSX_DEPLOYMENT_TARGET)
  set(CMAKE_OSX_DEPLOYMENT_TARGET "15.0" CACHE STRING "Minimum macOS deployment target")
endif()

project(
  MUDA
  VERSION 0.1.0
  DESCRIPTION "CUDA-to-Metal compatibility layer for Apple Silicon"
  LANGUAGES CXX
)

if(NOT APPLE)
  message(FATAL_ERROR "MUDA targets macOS")
endif()

enable_language(OBJCXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_OBJCXX_STANDARD 20)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(MUDA_BUILD_TESTS "Build MUDA tests" ON)
option(MUDA_BUILD_SAMPLES "Expose MUDA sample sources" ON)

include(GNUInstallDirs)

set(MUDA_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${MUDA_GENERATED_DIR}/muda")
configure_file(cmake/Version.hpp.in "${MUDA_GENERATED_DIR}/muda/version.hpp" @ONLY)

add_subdirectory(runtime)
add_subdirectory(compiler)
add_subdirectory(tools)

if(MUDA_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(MUDA_BUILD_SAMPLES)
  add_subdirectory(samples)
endif()

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
option(MUDA_BUILD_BENCHMARKS "Build MUDA benchmarks" ON)
option(MUDA_SANITIZE "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

add_library(muda_warnings INTERFACE)
target_compile_options(
  muda_warnings
  INTERFACE
    -Wall
    -Wextra
    -Wpedantic
    -Wshadow
    -Wconversion
    -Wsign-conversion
    -Wold-style-cast
    -Wcast-align
    -Wnon-virtual-dtor
    -Woverloaded-virtual
    -Wnull-dereference
    -Wdouble-promotion
    -Wformat=2
    -Wimplicit-fallthrough
    -Wunused
    -Wmissing-declarations
    -Wnewline-eof
    -Wthread-safety
    -Wunreachable-code
    -Wcomma
    "$<$<COMPILE_LANGUAGE:OBJCXX>:-Wobjc-missing-super-calls>"
    "$<$<COMPILE_LANGUAGE:OBJCXX>:-Wimplicit-retain-self>"
    "$<$<COMPILE_LANGUAGE:OBJCXX>:-Warc-repeated-use-of-weak>"
    "$<$<COMPILE_LANGUAGE:OBJCXX>:-Wnullable-to-nonnull-conversion>"
    "$<$<COMPILE_LANGUAGE:OBJCXX>:-Wdeprecated-implementations>"
)

if(MUDA_SANITIZE)
  target_compile_options(muda_warnings INTERFACE -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(muda_warnings INTERFACE -fsanitize=address,undefined)
endif()

include(GNUInstallDirs)

set(MUDA_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${MUDA_GENERATED_DIR}/muda")
configure_file(cmake/Version.hpp.in "${MUDA_GENERATED_DIR}/muda/version.hpp" @ONLY)

add_subdirectory(compiler)
add_subdirectory(runtime)
add_subdirectory(libraries)
add_subdirectory(tools)

# -----------------------------------------------------------------------------
# Build-tree CUDA_HOME Compatibility Layout (${CMAKE_BINARY_DIR}/cuda-home)
# -----------------------------------------------------------------------------
set(MUDA_CUDA_HOME_DIR "${CMAKE_CURRENT_BINARY_DIR}/cuda-home")
file(MAKE_DIRECTORY "${MUDA_CUDA_HOME_DIR}/bin")
file(MAKE_DIRECTORY "${MUDA_CUDA_HOME_DIR}/include")
file(MAKE_DIRECTORY "${MUDA_CUDA_HOME_DIR}/lib")
file(MAKE_DIRECTORY "${MUDA_CUDA_HOME_DIR}/lib64")

add_custom_target(
  muda_cuda_home ALL
  COMMAND ${CMAKE_COMMAND} -E copy_directory "${PROJECT_SOURCE_DIR}/runtime/include" "${MUDA_CUDA_HOME_DIR}/include"
  COMMAND ${CMAKE_COMMAND} -E copy_directory "${PROJECT_SOURCE_DIR}/libraries/blas/include" "${MUDA_CUDA_HOME_DIR}/include"
  COMMAND ${CMAKE_COMMAND} -E copy_directory "${PROJECT_SOURCE_DIR}/libraries/fft/include" "${MUDA_CUDA_HOME_DIR}/include"
  COMMAND ${CMAKE_COMMAND} -E copy_directory "${PROJECT_SOURCE_DIR}/libraries/dnn/include" "${MUDA_CUDA_HOME_DIR}/include"
  COMMAND ${CMAKE_COMMAND} -E copy_directory "${PROJECT_SOURCE_DIR}/libraries/sparse/include" "${MUDA_CUDA_HOME_DIR}/include"
  COMMAND ${CMAKE_COMMAND} -E copy_directory "${PROJECT_SOURCE_DIR}/libraries/nvrtc/include" "${MUDA_CUDA_HOME_DIR}/include"
  COMMAND ${CMAKE_COMMAND} -E copy_directory "${MUDA_GENERATED_DIR}" "${MUDA_CUDA_HOME_DIR}/include"
  COMMAND ${CMAKE_COMMAND} -E create_symlink "$<TARGET_FILE:mudac>" "${MUDA_CUDA_HOME_DIR}/bin/mudac"
  COMMAND ${CMAKE_COMMAND} -E create_symlink "$<TARGET_FILE:nvcc>" "${MUDA_CUDA_HOME_DIR}/bin/nvcc"
  COMMAND ${CMAKE_COMMAND} -E create_symlink "$<TARGET_FILE:muda>" "${MUDA_CUDA_HOME_DIR}/bin/muda"
  COMMAND ${CMAKE_COMMAND} -E create_symlink "$<TARGET_FILE:cuda>" "${MUDA_CUDA_HOME_DIR}/lib/libcuda.dylib"
  COMMAND ${CMAKE_COMMAND} -E create_symlink "$<TARGET_FILE:cudart>" "${MUDA_CUDA_HOME_DIR}/lib/libcudart.dylib"
  COMMAND ${CMAKE_COMMAND} -E create_symlink "$<TARGET_FILE:cublas>" "${MUDA_CUDA_HOME_DIR}/lib/libcublas.dylib"
  COMMAND ${CMAKE_COMMAND} -E create_symlink "$<TARGET_FILE:cufft>" "${MUDA_CUDA_HOME_DIR}/lib/libcufft.dylib"
  COMMAND ${CMAKE_COMMAND} -E create_symlink "$<TARGET_FILE:cudnn>" "${MUDA_CUDA_HOME_DIR}/lib/libcudnn.dylib"
  COMMAND ${CMAKE_COMMAND} -E create_symlink "$<TARGET_FILE:cusparse>" "${MUDA_CUDA_HOME_DIR}/lib/libcusparse.dylib"
  COMMAND ${CMAKE_COMMAND} -E create_symlink "$<TARGET_FILE:nvrtc>" "${MUDA_CUDA_HOME_DIR}/lib/libnvrtc.dylib"
  COMMAND ${CMAKE_COMMAND} -E create_symlink "$<TARGET_FILE:cuda>" "${MUDA_CUDA_HOME_DIR}/lib64/libcuda.dylib"
  COMMAND ${CMAKE_COMMAND} -E create_symlink "$<TARGET_FILE:cudart>" "${MUDA_CUDA_HOME_DIR}/lib64/libcudart.dylib"
  COMMAND ${CMAKE_COMMAND} -E create_symlink "$<TARGET_FILE:cublas>" "${MUDA_CUDA_HOME_DIR}/lib64/libcublas.dylib"
  COMMAND ${CMAKE_COMMAND} -E create_symlink "$<TARGET_FILE:cufft>" "${MUDA_CUDA_HOME_DIR}/lib64/libcufft.dylib"
  COMMAND ${CMAKE_COMMAND} -E create_symlink "$<TARGET_FILE:cudnn>" "${MUDA_CUDA_HOME_DIR}/lib64/libcudnn.dylib"
  COMMAND ${CMAKE_COMMAND} -E create_symlink "$<TARGET_FILE:cusparse>" "${MUDA_CUDA_HOME_DIR}/lib64/libcusparse.dylib"
  COMMAND ${CMAKE_COMMAND} -E create_symlink "$<TARGET_FILE:nvrtc>" "${MUDA_CUDA_HOME_DIR}/lib64/libnvrtc.dylib"
  DEPENDS mudac nvcc muda cuda cudart cublas cufft cudnn cusparse nvrtc
  COMMENT "Populating build-tree CUDA_HOME layout at ${MUDA_CUDA_HOME_DIR}"
  VERBATIM
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/MUDAConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/MUDAConfig.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/MUDA"
  PATH_VARS CMAKE_INSTALL_INCLUDEDIR CMAKE_INSTALL_LIBDIR CMAKE_INSTALL_BINDIR CMAKE_INSTALL_DATAROOTDIR
)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/MUDAConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/muda.pc.in"
  "${CMAKE_CURRENT_BINARY_DIR}/muda.pc"
  @ONLY
)

install(
  FILES
    "${CMAKE_CURRENT_BINARY_DIR}/MUDAConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/MUDAConfigVersion.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/MUDA"
)
install(
  FILES "${CMAKE_CURRENT_BINARY_DIR}/muda.pc"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig"
)
install(
  DIRECTORY "${MUDA_CUDA_HOME_DIR}/"
  DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/muda/cuda-home"
)

if(MUDA_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(MUDA_BUILD_SAMPLES)
  add_subdirectory(samples)
endif()

if(MUDA_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

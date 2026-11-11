function(muda_compile_metal_kernels TARGET_NAME METAL_SOURCE EXTRA_INCLUDE_DIR OUT_METALLIB_PATH)
  get_filename_component(SHADER_STEM "${METAL_SOURCE}" NAME_WE)
  set(AIR_FILE "${CMAKE_CURRENT_BINARY_DIR}/${SHADER_STEM}.air")
  set(METALLIB_FILE "${CMAKE_CURRENT_BINARY_DIR}/${SHADER_STEM}.metallib")

  execute_process(
    COMMAND xcrun -f metal
    RESULT_VARIABLE METAL_TOOLCHAIN_STATUS
    OUTPUT_QUIET ERROR_QUIET
  )

  if(METAL_TOOLCHAIN_STATUS EQUAL 0)
    add_custom_command(
      OUTPUT "${AIR_FILE}"
      COMMAND xcrun -sdk macosx metal -std=metal3.1 -O2 -I "${EXTRA_INCLUDE_DIR}" -c "${METAL_SOURCE}" -o "${AIR_FILE}"
      DEPENDS "${METAL_SOURCE}"
      COMMENT "Compiling Metal shader ${SHADER_STEM}.metal"
      VERBATIM
    )

    add_custom_command(
      OUTPUT "${METALLIB_FILE}"
      COMMAND xcrun -sdk macosx metallib "${AIR_FILE}" -o "${METALLIB_FILE}"
      DEPENDS "${AIR_FILE}"
      COMMENT "Linking Metal library ${SHADER_STEM}.metallib"
      VERBATIM
    )
  else()
    add_custom_command(
      OUTPUT "${METALLIB_FILE}"
      COMMAND ${CMAKE_COMMAND} -E touch "${METALLIB_FILE}"
      DEPENDS "${METAL_SOURCE}"
      COMMENT "Metal offline toolchain absent; runtime will compile ${SHADER_STEM}.metal via JIT"
      VERBATIM
    )
  endif()

  add_custom_target("${TARGET_NAME}_metallib" DEPENDS "${METALLIB_FILE}")
  add_dependencies("${TARGET_NAME}" "${TARGET_NAME}_metallib")
  set("${OUT_METALLIB_PATH}" "${METALLIB_FILE}" PARENT_SCOPE)
endfunction()

# -----------------------------------------------------------------------------
# Shared Library Execution & Plan Cache Infrastructure
# -----------------------------------------------------------------------------
add_library(
  muda_lib_common STATIC
  common/src/metal_executor.mm
)

target_include_directories(
  muda_lib_common
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/common/include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/runtime/include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/runtime/src>
    $<BUILD_INTERFACE:${MUDA_GENERATED_DIR}>
)

target_compile_features(muda_lib_common PUBLIC cxx_std_20)
target_compile_options(muda_lib_common PRIVATE "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>")
target_link_libraries(
  muda_lib_common
  PUBLIC
    cuda
  PRIVATE
    muda_warnings
    "-framework Foundation"
    "-framework Metal"
)

# -----------------------------------------------------------------------------
# cuBLAS Shim (libcublas.dylib)
# -----------------------------------------------------------------------------
add_library(
  cublas SHARED
  blas/src/cublas_context.mm
  blas/src/cublas_level1.mm
  blas/src/cublas_level2.mm
  blas/src/cublas_level3.mm
)
add_library(muda::cublas ALIAS cublas)

muda_compile_metal_kernels(
  cublas
  "${CMAKE_CURRENT_SOURCE_DIR}/blas/shaders/cublas_kernels.metal"
  "${CMAKE_CURRENT_SOURCE_DIR}/blas/src"
  CUBLAS_METALLIB_FILE
)

target_include_directories(
  cublas
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/blas/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/blas/src"
)

target_compile_definitions(
  cublas
  PRIVATE
    ACCELERATE_NEW_LAPACK
    MUDA_CUBLAS_METALLIB_PATH="${CUBLAS_METALLIB_FILE}"
    MUDA_CUBLAS_METAL_SOURCE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/blas/shaders/cublas_kernels.metal"
)

target_compile_features(cublas PUBLIC cxx_std_20)
target_compile_options(cublas PRIVATE "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>")
target_link_libraries(
  cublas
  PUBLIC
    cuda
  PRIVATE
    muda_lib_common
    muda_warnings
    "-framework Accelerate"
    "-framework Foundation"
    "-framework Metal"
    "-framework MetalPerformanceShaders"
)
set_target_properties(cublas PROPERTIES OUTPUT_NAME "cublas")

# -----------------------------------------------------------------------------
# cuFFT Shim (libcufft.dylib)
# -----------------------------------------------------------------------------
add_library(
  cufft SHARED
  fft/src/cufft_plan.mm
  fft/src/cufft_exec.mm
)
add_library(muda::cufft ALIAS cufft)

muda_compile_metal_kernels(
  cufft
  "${CMAKE_CURRENT_SOURCE_DIR}/fft/shaders/cufft_kernels.metal"
  "${CMAKE_CURRENT_SOURCE_DIR}/fft/src"
  CUFFT_METALLIB_FILE
)

target_include_directories(
  cufft
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/fft/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/fft/src"
)

target_compile_definitions(
  cufft
  PRIVATE
    MUDA_CUFFT_METALLIB_PATH="${CUFFT_METALLIB_FILE}"
    MUDA_CUFFT_METAL_SOURCE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/fft/shaders/cufft_kernels.metal"
)

target_compile_features(cufft PUBLIC cxx_std_20)
target_compile_options(cufft PRIVATE "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>")
target_link_libraries(
  cufft
  PUBLIC
    cuda
  PRIVATE
    muda_lib_common
    muda_warnings
    "-framework Foundation"
    "-framework Metal"
)
set_target_properties(cufft PROPERTIES OUTPUT_NAME "cufft")

# -----------------------------------------------------------------------------
# cuDNN Shim (libcudnn.dylib)
# -----------------------------------------------------------------------------
add_library(
  cudnn SHARED
  dnn/src/cudnn_descriptors.mm
  dnn/src/cudnn_ops.mm
)
add_library(muda::cudnn ALIAS cudnn)

muda_compile_metal_kernels(
  cudnn
  "${CMAKE_CURRENT_SOURCE_DIR}/dnn/shaders/cudnn_kernels.metal"
  "${CMAKE_CURRENT_SOURCE_DIR}/dnn/src"
  CUDNN_METALLIB_FILE
)

target_include_directories(
  cudnn
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/dnn/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/dnn/src"
)

target_compile_definitions(
  cudnn
  PRIVATE
    MUDA_CUDNN_METALLIB_PATH="${CUDNN_METALLIB_FILE}"
    MUDA_CUDNN_METAL_SOURCE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/dnn/shaders/cudnn_kernels.metal"
)

target_compile_features(cudnn PUBLIC cxx_std_20)
target_compile_options(cudnn PRIVATE "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>")
target_link_libraries(
  cudnn
  PUBLIC
    cuda
  PRIVATE
    muda_lib_common
    muda_warnings
    "-framework Foundation"
    "-framework Metal"
)
set_target_properties(cudnn PROPERTIES OUTPUT_NAME "cudnn")

# -----------------------------------------------------------------------------
# cuSPARSE Shim (libcusparse.dylib)
# -----------------------------------------------------------------------------
add_library(
  cusparse SHARED
  sparse/src/cusparse_descriptors.mm
  sparse/src/cusparse_ops.mm
)
add_library(muda::cusparse ALIAS cusparse)

muda_compile_metal_kernels(
  cusparse
  "${CMAKE_CURRENT_SOURCE_DIR}/sparse/shaders/cusparse_kernels.metal"
  "${CMAKE_CURRENT_SOURCE_DIR}/sparse/src"
  CUSPARSE_METALLIB_FILE
)

target_include_directories(
  cusparse
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/sparse/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/sparse/src"
)

target_compile_definitions(
  cusparse
  PRIVATE
    MUDA_CUSPARSE_METALLIB_PATH="${CUSPARSE_METALLIB_FILE}"
    MUDA_CUSPARSE_METAL_SOURCE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/sparse/shaders/cusparse_kernels.metal"
)

target_compile_features(cusparse PUBLIC cxx_std_20)
target_compile_options(cusparse PRIVATE "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>")
target_link_libraries(
  cusparse
  PUBLIC
    cuda
  PRIVATE
    muda_lib_common
    muda_warnings
    "-framework Foundation"
    "-framework Metal"
)
set_target_properties(cusparse PROPERTIES OUTPUT_NAME "cusparse")

# -----------------------------------------------------------------------------
# NVRTC Shim (libnvrtc.dylib)
# -----------------------------------------------------------------------------
add_library(
  nvrtc SHARED
  nvrtc/src/nvrtc.cpp
)
add_library(muda::nvrtc ALIAS nvrtc)

target_include_directories(
  nvrtc
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/nvrtc/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    "${PROJECT_SOURCE_DIR}/runtime/src"
)

target_compile_features(nvrtc PUBLIC cxx_std_20)
target_link_libraries(
  nvrtc
  PUBLIC
    cuda
  PRIVATE
    muda::compiler
    muda_warnings
)
set_target_properties(nvrtc PROPERTIES OUTPUT_NAME "nvrtc")

# -----------------------------------------------------------------------------
# cuRAND Shim (libcurand.dylib)
# -----------------------------------------------------------------------------
add_library(
  curand SHARED
  rand/src/curand_context.mm
  rand/src/curand_api.mm
)
add_library(muda::curand ALIAS curand)

muda_compile_metal_kernels(
  curand
  "${CMAKE_CURRENT_SOURCE_DIR}/rand/shaders/curand_kernels.metal"
  "${CMAKE_CURRENT_SOURCE_DIR}/rand/src"
  CURAND_METALLIB_FILE
)

target_include_directories(
  curand
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/rand/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/rand/src"
)

target_compile_definitions(
  curand
  PRIVATE
    MUDA_CURAND_METALLIB_PATH="${CURAND_METALLIB_FILE}"
    MUDA_CURAND_METAL_SOURCE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/rand/shaders/curand_kernels.metal"
)

target_compile_features(curand PUBLIC cxx_std_20)
target_compile_options(curand PRIVATE "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>")
target_link_libraries(
  curand
  PUBLIC
    cuda
  PRIVATE
    muda_lib_common
    muda_warnings
    "-framework Foundation"
    "-framework Metal"
)
set_target_properties(curand PROPERTIES OUTPUT_NAME "curand")

install(
  TARGETS cublas cufft cudnn cusparse nvrtc curand
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY blas/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY fft/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY dnn/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY sparse/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY nvrtc/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY rand/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

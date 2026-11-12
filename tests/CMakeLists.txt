add_executable(version_test version_test.cpp)
target_link_libraries(version_test PRIVATE muda::runtime muda_warnings)
add_test(NAME version_test COMMAND version_test)

add_executable(device_test device_test.cpp)
target_link_libraries(device_test PRIVATE muda::runtime muda_warnings)
add_test(NAME device_test COMMAND device_test)

add_executable(cuda_runtime_test cuda_runtime_test.cpp)
target_link_libraries(cuda_runtime_test PRIVATE muda::compiler muda::runtime muda_warnings)
target_compile_definitions(cuda_runtime_test PRIVATE MUDA_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
add_test(NAME cuda_runtime_test COMMAND cuda_runtime_test)

add_executable(translator_test translator_test.cpp)
target_link_libraries(translator_test PRIVATE muda::compiler muda_warnings)
target_compile_definitions(translator_test PRIVATE MUDA_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
add_test(NAME translator_test COMMAND translator_test)

add_executable(compiler_test compiler_test.cpp)
target_link_libraries(compiler_test PRIVATE muda::compiler muda::runtime muda_warnings)
target_compile_definitions(compiler_test PRIVATE MUDA_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
add_test(NAME compiler_test COMMAND compiler_test)

add_executable(vector_add_test vector_add_test.cpp)
target_link_libraries(vector_add_test PRIVATE muda::compiler muda::runtime muda_warnings)
target_compile_definitions(vector_add_test PRIVATE MUDA_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
add_test(NAME vector_add_test COMMAND vector_add_test)

add_executable(corpus_test corpus_test.cpp)
target_link_libraries(corpus_test PRIVATE muda::compiler muda::runtime muda_warnings)
target_compile_definitions(corpus_test PRIVATE MUDA_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
add_test(NAME corpus_test COMMAND corpus_test)

add_executable(warp_intrinsics_test warp_intrinsics_test.cpp)
target_link_libraries(warp_intrinsics_test PRIVATE muda::compiler muda::runtime muda_warnings)
target_compile_definitions(warp_intrinsics_test PRIVATE MUDA_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
add_test(NAME warp_intrinsics_test COMMAND warp_intrinsics_test)

add_executable(atomics_test atomics_test.cpp)
target_link_libraries(atomics_test PRIVATE muda::compiler muda::runtime muda_warnings)
target_compile_definitions(atomics_test PRIVATE MUDA_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
add_test(NAME atomics_test COMMAND atomics_test)

add_executable(half_vector_test half_vector_test.cpp)
target_link_libraries(half_vector_test PRIVATE muda::compiler muda::runtime muda_warnings)
target_compile_definitions(half_vector_test PRIVATE MUDA_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
add_test(NAME half_vector_test COMMAND half_vector_test)

add_executable(buffer_pool_test buffer_pool_test.cpp)
target_link_libraries(buffer_pool_test PRIVATE muda::runtime muda_warnings)
add_test(NAME buffer_pool_test COMMAND buffer_pool_test)

add_executable(compiler_cache_test compiler_cache_test.cpp)
target_link_libraries(compiler_cache_test PRIVATE muda::compiler muda::runtime muda_warnings)
add_test(NAME compiler_cache_test COMMAND compiler_cache_test)

add_executable(ptx_lexer_test ptx/ptx_lexer_test.cpp)
target_link_libraries(ptx_lexer_test PRIVATE muda::compiler muda_warnings)
add_test(NAME ptx_lexer_test COMMAND ptx_lexer_test)

add_executable(ptx_parser_test ptx/ptx_parser_test.cpp)
target_link_libraries(ptx_parser_test PRIVATE muda::compiler muda_warnings)
add_test(NAME ptx_parser_test COMMAND ptx_parser_test)

add_executable(ptx_to_mmir_test ptx/ptx_to_mmir_test.cpp)
target_link_libraries(ptx_to_mmir_test PRIVATE muda::compiler muda::runtime muda_warnings)
add_test(NAME ptx_to_mmir_test COMMAND ptx_to_mmir_test)

add_executable(driver_api_test driver_api_test.cpp)
target_link_libraries(driver_api_test PRIVATE cuda muda::runtime muda_warnings)
add_test(NAME driver_api_test COMMAND driver_api_test)

add_executable(numba_mock_test numba_mock_test.cpp)
target_link_libraries(numba_mock_test PRIVATE cuda muda::runtime muda_warnings)
add_test(NAME numba_mock_test COMMAND numba_mock_test)

add_executable(cublas_test cublas_test.cpp)
target_link_libraries(cublas_test PRIVATE muda::cublas cuda muda_warnings)
add_test(NAME cublas_test COMMAND cublas_test)

add_executable(cufft_test cufft_test.cpp)
target_link_libraries(cufft_test PRIVATE muda::cufft cuda muda_warnings)
add_test(NAME cufft_test COMMAND cufft_test)

add_executable(cudnn_test cudnn_test.cpp)
target_link_libraries(cudnn_test PRIVATE muda::cudnn cuda muda_warnings)
add_test(NAME cudnn_test COMMAND cudnn_test)

add_executable(cusparse_test cusparse_test.cpp)
target_link_libraries(cusparse_test PRIVATE muda::cusparse cuda muda_warnings)
add_test(NAME cusparse_test COMMAND cusparse_test)

add_executable(nvrtc_test nvrtc_test.cpp)
target_link_libraries(nvrtc_test PRIVATE muda::nvrtc muda::cudart cuda muda_warnings)
add_test(NAME nvrtc_test COMMAND nvrtc_test)

add_executable(curand_test curand_test.cpp)
target_link_libraries(curand_test PRIVATE muda::curand cuda muda_warnings)
add_test(NAME curand_test COMMAND curand_test)

add_executable(fp64_test fp64_test.cpp)
target_link_libraries(fp64_test PRIVATE muda::nvrtc muda::cudart cuda muda_warnings)
add_test(NAME fp64_test COMMAND fp64_test)

add_executable(header_hygiene_test header_hygiene_test.cpp)
target_link_libraries(
  header_hygiene_test
  PRIVATE
    muda::cudart
    muda::cublas
    muda::cufft
    muda::cudnn
    muda::cusparse
    muda::curand
    muda::nvrtc
    muda_warnings
)
target_include_directories(
  header_hygiene_test
  PRIVATE
    "${PROJECT_SOURCE_DIR}/compiler/include"
)
add_test(NAME header_hygiene_test COMMAND header_hygiene_test)

add_executable(cubin_test cubin_test.cpp)
target_link_libraries(cubin_test PRIVATE muda::compiler cuda muda_warnings)
add_test(NAME cubin_test COMMAND cubin_test)

set_tests_properties(compiler_test compiler_cache_test vector_add_test corpus_test warp_intrinsics_test atomics_test half_vector_test PROPERTIES SKIP_RETURN_CODE 77)

add_test(NAME mudac_version COMMAND mudac --version)
add_test(NAME nvcc_version COMMAND nvcc --version)
add_test(NAME muda_version COMMAND muda version)

add_executable(muda_cli_test muda_cli_test.cpp)
target_link_libraries(muda_cli_test PRIVATE muda::compiler muda::runtime muda_warnings)
target_compile_definitions(muda_cli_test PRIVATE MUDA_CLI_BINARY_PATH="$<TARGET_FILE:muda>")
add_dependencies(muda_cli_test muda mudac cuda)
add_test(NAME muda_cli_test COMMAND muda_cli_test)

find_program(UV_EXECUTABLE uv)
if(UV_EXECUTABLE)
  add_test(
    NAME python_driver_abi_test
    COMMAND ${UV_EXECUTABLE} run --with numpy python3 ${CMAKE_CURRENT_SOURCE_DIR}/python/test_cuda_driver_abi.py
  )
  set_tests_properties(
    python_driver_abi_test
    PROPERTIES
      ENVIRONMENT "CUDA_DRIVER_LIBRARY=$<TARGET_FILE:cuda>"
  )

  add_test(
    NAME python_libraries_abi_test
    COMMAND ${UV_EXECUTABLE} run --with numpy python3 ${CMAKE_CURRENT_SOURCE_DIR}/python/test_cuda_libraries_abi.py
  )
  set_tests_properties(
    python_libraries_abi_test
    PROPERTIES
      ENVIRONMENT "CUDA_DRIVER_LIBRARY=$<TARGET_FILE:cuda>;CUBLAS_LIBRARY=$<TARGET_FILE:cublas>;CUFFT_LIBRARY=$<TARGET_FILE:cufft>;CUDNN_LIBRARY=$<TARGET_FILE:cudnn>;CUSPARSE_LIBRARY=$<TARGET_FILE:cusparse>"
  )

  add_test(
    NAME python_numba_cuda_test
    COMMAND ${UV_EXECUTABLE} run --with numba --with numpy python3 ${CMAKE_CURRENT_SOURCE_DIR}/python/test_numba_cuda.py
  )
  set_tests_properties(
    python_numba_cuda_test
    PROPERTIES
      ENVIRONMENT "NUMBA_CUDA_DRIVER=$<TARGET_FILE:cuda>"
  )

  add_test(
    NAME python_cuda_library_test
    COMMAND ${UV_EXECUTABLE} run --with numba --with numpy python3 ${CMAKE_CURRENT_SOURCE_DIR}/python/test_python_cuda_library.py
  )
  set_tests_properties(
    python_cuda_library_test
    PROPERTIES
      ENVIRONMENT "NUMBA_CUDA_DRIVER=$<TARGET_FILE:cuda>"
  )

  add_test(
    NAME python_pytorch_extension_test
    COMMAND ${UV_EXECUTABLE} run --with torch --with numpy python3 ${CMAKE_CURRENT_SOURCE_DIR}/python/test_pytorch_cuda_extension.py
  )
  set_tests_properties(
    python_pytorch_extension_test
    PROPERTIES
      ENVIRONMENT "MUDA_CUDA_HOME=${PROJECT_BINARY_DIR}/cuda-home"
  )
endif()

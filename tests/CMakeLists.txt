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
target_link_libraries(compiler_test PRIVATE muda::compiler muda_warnings)
target_compile_definitions(compiler_test PRIVATE MUDA_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
add_test(NAME compiler_test COMMAND compiler_test)

add_executable(vector_add_test vector_add_test.cpp)
target_link_libraries(vector_add_test PRIVATE muda::compiler muda::runtime muda_warnings)
target_compile_definitions(vector_add_test PRIVATE MUDA_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
add_test(NAME vector_add_test COMMAND vector_add_test)

set_tests_properties(compiler_test vector_add_test PROPERTIES SKIP_RETURN_CODE 77)

add_test(NAME mudac_version COMMAND mudac --version)
add_test(NAME muda_version COMMAND muda version)

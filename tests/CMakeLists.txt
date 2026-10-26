add_executable(version_test version_test.cpp)
target_link_libraries(version_test PRIVATE muda::runtime)
target_compile_options(version_test PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME version_test COMMAND version_test)

add_executable(device_test device_test.cpp)
target_link_libraries(device_test PRIVATE muda::runtime)
target_compile_options(device_test PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME device_test COMMAND device_test)

add_executable(compiler_test compiler_test.cpp)
target_link_libraries(compiler_test PRIVATE muda::compiler)
target_compile_options(compiler_test PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME compiler_test COMMAND compiler_test)

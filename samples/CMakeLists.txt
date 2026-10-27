add_executable(muda_vector_add vector_add_runner.cpp)
target_link_libraries(muda_vector_add PRIVATE muda::runtime)
target_compile_options(muda_vector_add PRIVATE -Wall -Wextra -Wpedantic)

add_custom_target(muda_samples SOURCES vector_add.cu)

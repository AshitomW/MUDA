add_executable(runtime_bench runtime_bench.cpp)
target_link_libraries(runtime_bench PRIVATE muda::compiler muda::runtime muda_warnings)
target_compile_definitions(runtime_bench PRIVATE MUDA_SOURCE_DIR="${PROJECT_SOURCE_DIR}")

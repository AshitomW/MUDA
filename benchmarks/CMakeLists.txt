add_executable(runtime_bench
    runtime_bench.cpp
    benchmark_runner.cpp
    microbenchmarks.cpp
    throughput_benchmarks.cpp
    library_benchmarks.cpp
)
target_link_libraries(runtime_bench PRIVATE
    muda::cublas
    muda::cufft
    muda::cudnn
    muda::cusparse
    muda::nvrtc
    cuda
    muda::compiler
    muda_warnings
)
target_compile_definitions(runtime_bench PRIVATE MUDA_SOURCE_DIR="${PROJECT_SOURCE_DIR}")

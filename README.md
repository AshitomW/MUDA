# MUDA

MUDA is a CUDA-to-Metal compatibility layer for Apple Silicon. Its goal is to
let CUDA programs run on Macs that do not have an NVIDIA GPU.

It is not a machine-learning framework. It is aimed at the gap between
CUDA-only software and Apple's Metal ecosystem.

## Why it exists

Apple GPUs do not execute CUDA. CUDA code is written for NVIDIA's toolchain,
compiled to PTX/SASS, and expects CUDA runtime and driver libraries. Metal has a
different language, memory model, and API, so CUDA software cannot simply be
recompiled for it.

## Examples

- A PyTorch extension built with `torch.utils.cpp_extension` invokes `nvcc` and
  includes CUDA headers. On macOS, `nvcc` and the CUDA toolkit are unavailable,
  so the extension cannot be built.
- A custom CUDA kernel is compiled to NVIDIA PTX/SASS and loaded through the
  CUDA driver. Metal cannot load or execute that binary.
- Code using cuBLAS, cuDNN, NCCL, or other CUDA-specific APIs depends on NVIDIA
  libraries and kernels. Those APIs have no direct Metal equivalent.
- CUDA features such as dynamic parallelism, device-side function pointers, and
  some warp-level primitives have no complete Metal equivalent.

MUDA intends to translate what can be translated, provide compatible runtime
pieces where practical, and report clear errors for what cannot run.

## Status

Supported capabilities:
- **Compiler (`mudac`)**: End-to-end compilation of `.cu` sources to native macOS executables, object files (`-c`), Metal Shading Language (`-S`), or `.metallib` (`--emit-metallib`). Supports multi-kernel translation units, `__device__` helper functions, static and dynamic `__shared__` memory, `__syncthreads()`, FP32 math intrinsics, MMIR validation, content-addressed artifact caching (`~/.cache/muda/v1`), and deterministic `ALY-*` diagnostics.
- **Runtime (`libmuda_runtime`)**: Unified-memory allocation (`cudaMalloc`, `cudaMallocManaged`, `cudaMallocHost`, `cudaMallocPitch`), 1D/2D synchronous and asynchronous copies/memset, generation-validated streams and events (`MTLCommandQueue` / `MTLSharedEvent`), device property and pointer queries, and embedded `.metallib` kernel registration and launch (`<<<...>>>` and `cudaLaunchKernel`).
- **Verification**: 10-kernel numerical parity corpus and Apple Silicon GPU benchmark harness.

## Quick Start

```bash
# Configure, build, and run all tests
cmake --preset release
cmake --build --preset release
ctest --preset release --output-on-failure

# Compile and run a standalone .cu program
./build/release/tools/mudac samples/vector_add.cu -o /tmp/vector_add
/tmp/vector_add

# Run the GPU benchmark suite
./build/release/benchmarks/runtime_bench
```

See [docs/EXAMPLES.md](docs/EXAMPLES.md) for detailed usage, working CUDA examples, Metal Shading Language inspection, stream/event examples, and benchmark documentation.

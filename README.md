# MUDA: CUDA-to-Metal Compatibility Layer for Apple Silicon

MUDA is an experimental, clean-room CUDA-to-Metal translation engine and runtime compatibility layer that allows CUDA C++ kernels and PTX assembly to run directly on Apple Silicon GPUs without an NVIDIA graphics card.

It works by translating CUDA source code and PTX assembly into Metal Shading Language (MSL), managing memory and compute dispatches through Apple's Metal framework, and providing drop-in library shims for standard CUDA driver, runtime, and math APIs.

> [!NOTE]
> **Project Scope & Realistic Expectations**: MUDA is an independent, clean-room implementation of a focused, compute-oriented subset of CUDA. It is designed for developers who want to run custom compute kernels, machine learning operations, and workflows from tools like PyTorch and Numba on Apple Silicon.
> 
> It provides compilation for CUDA C++ source code (`.cu`), PTX virtual assembly (`.ptx`), and clean-room SASS disassembly and execution for modern NVIDIA CUBIN (`.cubin`) and Fatbinary (`.fatbin`) containers. Code using unsupported features (such as device-side dynamic parallelism, function pointers, or graphics interop) will require modification or will emit compile-time diagnostics.

---

## Why MUDA?

Apple Silicon Macs feature unified memory architectures and capable GPUs, but they speak Metal rather than CUDA:

1. **No Native CUDA Toolchain on macOS:** macOS lacks `nvcc`, the NVIDIA CUDA Driver, and runtime libraries (`libcudart`, `libcuda`).
2. **Architectural Differences:** Metal Shading Language (MSL) has distinct execution models, memory spaces, and threadgroup synchronization semantics. Notably, MSL disallows arbitrary `goto` jumps and labeled control flow.
3. **Dynamic JIT Workflows:** Many Python packages (such as Numba's `@cuda.jit`) generate PTX virtual assembly on the fly in memory and load it via the CUDA Driver API.

MUDA bridges these differences by translating CUDA C++ and PTX into MSL, providing compatible runtime shims, and mapping CUDA concepts cleanly onto Apple's Metal API.

---

## What Is Implemented Today

### 1. Compiler Toolchain (`mudac` & `nvcc`)
- **Compilation Frontends:** Compiles CUDA C++ (`.cu`) and PTX assembly (`.ptx`) into native macOS executables, shared libraries (`-shared` / `-dynamiclib`), object files (`-c`), Metal Shading Language (`-S`), or precompiled Metal libraries (`--emit-metallib`).
- **`nvcc` Compatibility Wrapper:** Translates standard `nvcc` flags (`-shared`, `-fPIC`, `-D`, `-U`, `-I`, `-L`, `-l`, `-Xcompiler`, `-Xlinker`, `-gencode`, `-arch`, `--expt-relaxed-constexpr`) so standard build tools (CMake, `setuptools`, PyTorch) can use MUDA as a drop-in compiler.
- **Muda Metal Intermediate Representation (MMIR):** Validates kernel signatures, parameter bindings, access qualifiers, and shared memory allocations before emitting MSL.
- **Content-Addressed Build Cache:** Avoids redundant recompilation by caching artifacts based on source hashes, macro states, and compiler flags.
- **Zero-Toolchain JIT Resiliency:** Allows compilation and execution on macOS developer environments with only Xcode Command Line Tools installed by embedding MSL source into compiled artifacts and dynamically creating pipeline states via `mudaRegisterKernelFromSource` when offline Metal bytecode tools (`xcrun metal` / `xcrun metallib`) are absent.
- **Standard `CUDA_HOME` Layout:** Builds a familiar directory structure (`bin/`, `include/`, `lib/`) containing `nvcc`, `mudac`, clean-room headers, and library symlinks.

### 2. PTX Translation Engine
- **Fast Table-Driven Parser:** Pure C++ arena-allocated lexer and shift-reduce parser supporting PTX 6.x and 7.x instruction sets.
- **Relooper CFG Structurizer:** Analyzes basic blocks and transforms arbitrary branch targets into structured Metal loops and `if`/`else` control flow, with an emulated state-machine dispatcher fallback for irreducible loops.
- **In-Memory JIT Compilation:** Translates PTX strings into Metal compute pipelines in-process via `cuModuleLoadData` in milliseconds, without spawning subprocesses or writing temporary files to disk.

### 3. CUBIN & Fatbinary Disassembly & Execution (`muda::compiler::cubin`)
- **Fatbinary Container Extraction (`FatbinParser`):** Automatically detects 32-bit (`0xBA55ED50`) and 64-bit (`0x1EE55A01`) NVIDIA Fatbin containers and extracts embedded PTX payloads directly for in-memory JIT execution.
- **ELF64 Container & Metadata Deserializer (`ElfReader`):** Clean-room ELF64 parser (`EM_CUDA = 190`) deserializing `.nv.info` attributes (`EIATTR_KPARAM_INFO`, `EIATTR_SMEM_SIZE`, `EIATTR_MAX_THREADS`) to reconstruct kernel argument bank offsets, byte sizes, and alignment.
- **128-bit SASS Machine Instruction Decoder (`SassDecoder`):** Decodes compute instructions for Turing (SM 7.5) and Ampere (SM 8.0/8.6), covering control flow (`EXIT`, `BRA`, `BAR.SYNC`), special registers (`S2R`), integer/address arithmetic (`IMAD`, `IADD`, `ISETP`), float arithmetic (`FADD`, `FMUL`, `FFMA`), and global/shared memory operations (`LDG.E`, `STG.E`, `LDS`, `STS`).
- **Register-Union State Lifter (`SassLifter`):** Lifts SASS into MSL using the Register-Union State Model (`union Reg32 { uint32_t u; int32_t i; float f; }; thread Reg32 R[256]; thread bool P[8];`), Constant Bank 0 parameter mapping, and unified 64-bit memory addressing.
- **Transparent Driver JIT Loading:** Routes `.cubin` and `.fatbin` binaries through `cuModuleLoadData` and `ModuleManager::LoadCubin` directly into Metal compute pipeline states.

### 4. CUDA Runtime API (`libcudart`)
- **Unified & Host Memory:** `cudaMalloc`, `cudaMallocManaged`, `cudaMallocHost`, `cudaFree`, `cudaFreeHost`, and `cudaMallocPitch`.
- **Segregated Buffer Pool:** Power-of-two size-classed allocator (256 B to 16 MB) that minimizes heap allocation overhead during iterative kernel loops.
- **Transfers & Fills:** Synchronous and asynchronous 1D and 2D memory copies and sets (`cudaMemcpy`, `cudaMemcpyAsync`, `cudaMemcpy2D`, `cudaMemcpy2DAsync`, `cudaMemset`, `cudaMemsetAsync`, `cudaMemset2D`).
- **Streams & Events:** Non-blocking and default streams mapped to `MTLCommandQueue`, with GPU timing events mapped to `MTLSharedEvent`.
- **Core Language & Device Primitives:**
  - **Thread Indexing:** `threadIdx`, `blockIdx`, `blockDim`, `gridDim`.
  - **Shared Memory:** `__shared__` and dynamic `extern __shared__` allocations.
  - **Synchronization:** `__syncthreads()`.
  - **Warp Intrinsics:** Shuffle (`__shfl_sync`, `__shfl_down_sync`, `__shfl_xor_sync`), vote (`__all_sync`, `__any_sync`, `__ballot_sync`), and bitwise operations (`__popc`, `__clz`, `__ffs`, `__brev`).
  - **Atomics:** 32-bit and 64-bit integer atomics (`atomicAdd`, `atomicSub`, `atomicExch`, `atomicMin`, `atomicMax`, `atomicAnd`, `atomicOr`, `atomicXor`) and 32-bit float CAS loops.
  - **Half-Precision & Vectors:** IEEE 754-2008 `half`, `half2`, and aligned vector types (`float2/3/4`, `int2/3/4`, `uint2/3/4`).
  - **Double Precision (FP64) Emulation:** Support for `double`, `double2/3/4`, double-precision shared memory, and math intrinsics (`fma`, `sqrt`, `rsqrt`, `abs`, `min`, `max`) via `--muda-fp64=fast48` (high-throughput Double-Single emulation in `runtime/include/muda/muda_fp64.metal`) or `--muda-fp64=ieee64` (bit-exact IEEE-754 64-bit integer bitfield emulation with subnormals and round-to-nearest-even).

### 5. CUDA Driver API (`libcuda.dylib`)
- **Clean-Room Driver Interface:** Declares and exports standard driver functions:
  - Device inspection (`cuInit`, `cuDeviceGet*`, `cuDeviceGetName`, `cuDeviceTotalMem`, `cuDeviceGetAttribute`, `cuDeviceGetUuid`).
  - Context lifecycle (`cuCtxCreate`, `cuCtxDestroy`, `cuCtxSetCurrent`, `cuCtxGetCurrent`, `cuCtxSynchronize`, `cuDevicePrimaryCtx*`).
  - Device memory management (`cuMemAlloc`, `cuMemAllocPitch`, `cuMemAllocManaged`, `cuMemFree`, `cuMemGetAddressRange`, `cuPointerGetAttribute`).
  - Strided & Async Copies (`cuMemcpyHtoD`, `cuMemcpyDtoH`, `cuMemcpyDtoD`, `cuMemcpy2D`, `cuMemcpyAsync`, `cuMemset*`).
  - In-memory dynamic module loading and launch (`cuModuleLoadData`, `cuModuleGetFunction`, `cuLaunchKernel`).
  - Occupancy queries (`cuOccupancyMaxActiveBlocksPerMultiprocessor`, `cuOccupancyMaxPotentialBlockSize`).
- **Driver ABI Compatibility:** Exports standard CUDA 4.0+ `_v2` symbol aliases, and provides stubs for profiler and IPC symbols to avoid runtime binding crashes in Python `ctypes`.

### 6. Acceleration Libraries & Runtime Compilation
- **cuBLAS (`libcublas.dylib`):**
  - Level-1: `Saxpy`, `Scopy`, `Sscal`, `Sswap`, `Sdot`, `Snrm2`, `Sasum`, `Isamax`, and double precision (`Daxpy`, `Dcopy`, `Dscal`, `Dswap`, `Ddot`, `Dnrm2`, `Dasum`, `Idamax`).
  - Level-2: `Sgemv`, `Dgemv`.
  - Level-3: `Sgemm`, `Hgemm`, `GemmEx`, `SgemmStridedBatched`, `GemmStridedBatchedEx`, `Sgeam`, and double precision (`Dgemm`, `DgemmStridedBatched`, `Dgeam`).
  - Uses Apple's `MPSMatrixMultiplication` for large matrix multiplications and Apple Silicon `simdgroup_float8x8` hardware matrix shaders for smaller or batched shapes; accelerates double-precision routines directly on unified device memory via Apple Accelerate.
- **cuFFT (`libcufft.dylib`):**
  - 1D, 2D, 3D, and batched `cufftPlanMany` transforms (`C2C`, `R2C`, `C2R`).
  - Single-pass shared memory kernels for common power-of-two sizes alongside general multi-pass fallbacks, backed by an LRU plan cache.
- **cuDNN (`libcudnn.dylib`):**
  - 4D tensor descriptors (`NCHW`, `NHWC`).
  - 2D Convolutions (forward, backward-data, backward-filter, backward-bias) with implicit-GEMM SIMD acceleration.
  - Activations (ReLU, Sigmoid, Tanh, Clipped ReLU, ELU, Swish) with vectorized data paths.
  - 2D Pooling (Max, Average), Softmax, and Batch Normalization (inference and training forward).
- **cuSPARSE (`libcusparse.dylib`):**
  - CSR sparse matrix descriptors (0-based and 1-based indexing).
  - Sparse matrix-vector (`cusparseSpMV`, FP32/FP16) and sparse matrix-matrix (`cusparseSpMM`, FP32) multiplication.
- **cuRAND (`libcurand.dylib`):**
  - High-throughput pseudorandom number generation via vectorized 128-bit Philox4x32-10 GPU compute shaders.
  - Uniform float `(0, 1]`, Box-Muller Gaussian normal, log-normal, and Poisson distributions.
  - Stream binding, seed and offset determinism, and Apple Silicon unified-memory CPU fallback for double-precision generations (`curandGenerateUniformDouble`, `curandGenerateNormalDouble`).
- **NVRTC (`libnvrtc.dylib`):**
  - Runtime compilation of CUDA C++ strings directly into Metal pipelines via `nvrtcCreateProgram`, `nvrtcCompileProgram`, and `nvrtcGetPTX`.

### 7. Python, Numba & PyTorch Integration
- **Numba CUDA Support:** Verified with Numba's CUDA driver interface via `NUMBA_CUDA_DRIVER`. Supports elementwise vector math, shared-memory reductions with `bar.sync`, and 2D tiled matrix multiplications.
- **PyTorch Extension Bridge (`python/muda`):** Allows custom `.cu` kernel extensions to be compiled with `nvcc` and called directly from PyTorch, with support for `torch.autograd.Function` and `torch.Tensor` operations via `libcublas`.

### 8. Diagnostics & Environment Tooling (`muda`)
- `muda doctor [--json] [--smoke-test]`: Inspects the macOS environment, Metal GPU limits, toolchain availability, compiler cache, and runs an end-to-end GPU compute smoke test.
- `muda path <target>`: Prints absolute paths to `cuda-home`, `libcuda.dylib`, `libcudart.dylib`, `include`, `cache`, or `mudac`.
- `muda cache <info|clear|verify>`: Inspects, clears, or validates the compiler artifact cache.

### 9. Interactive WebAssembly Playground
- The core C++ translation engine compiles to WebAssembly (`web/wasm/`), powering an offline in-browser playground (`web/`).
- Allows developers to paste CUDA C++ or PTX code and view the generated Metal Shading Language in real time, inspect reflected kernel buffer slots and threadgroup allocations, and share code snippets via URL.

---

## Realistic Boundaries & Limitations

To keep expectations grounded, here is an honest summary of what MUDA does not do or where Apple Silicon differs from NVIDIA hardware:

1. **Precompiled SASS / `.cubin` & Fatbinary Scope:** MUDA provides clean-room ELF64 CUBIN and Fatbinary parsing, with 128-bit SASS disassembly and execution for modern NVIDIA architectures (Turing SM 7.5, Ampere SM 8.0/8.6) covering compute instructions, control flow, register-union state lifting, and memory operations. Precompiled proprietary blobs relying on architecture-specific graphics pipelines, hardware video decoding, or undocumented instruction sets are not supported.
2. **Unsupported Device Constructs:**
   - **Dynamic Parallelism:** Metal does not support launching kernels from inside a running GPU kernel (`<<<...>>>` device launches).
   - **Device Function Pointers & Virtual Methods:** Indirect device-side calls and C++ virtual tables on the device are unsupported.
   - **Device Recursion:** Kernels cannot be recursive.
3. **Control Flow Differences:**
   - Metal Shading Language strictly forbids arbitrary `goto` statements and labeled jumps.
   - Natural loops (`while`, `for`) and conditional branches (`if`/`else`) are structurized cleanly. Complex irreducible branch graphs fall back to an emulated state-machine dispatcher, which can impact performance.
4. **FP64 (Double Precision) Hardware Limitations & Emulation:**
   - Apple Silicon GPUs possess zero native 64-bit floating-point ALU hardware.
   - MUDA implements normative Decision D8: by default (`--muda-fp64=off`), kernels containing `double` are rejected with diagnostic `ALY-C-401` to prevent silent precision loss.
   - Passing `--muda-fp64=fast48` (or `MUDA_FP64=fast48`) activates high-throughput Double-Single (DS) software emulation via two 32-bit floats with Knuth error correction (`runtime/include/muda/muda_fp64.metal`), providing ~10^-14 relative precision.
   - Passing `--muda-fp64=ieee64` (or `MUDA_FP64=ieee64`) activates bit-exact IEEE-754 software emulation using 64-bit integer bitfield arithmetic, providing exact IEEE-754 semantics including subnormals, infinities, NaNs, and round-to-nearest-even.
   - In cuBLAS, double-precision routines (`cublasD*`) execute with full native 64-bit hardware precision directly on unified device memory via Apple Accelerate BLAS.
5. **Driver API Scope:**
   - The Driver API implementation focuses on compute, memory management, streams, events, and module loading.
   - OpenGL/Vulkan interop, multi-GPU peer-to-peer memory transfers, and NVLink APIs are not supported.

---

## Getting Started

### Prerequisites

* **Hardware:** Any Apple Silicon Mac (M1, M2, M3, M4 or Pro/Max/Ultra).
* **Operating System:** macOS 14.0 (Sonoma) or newer.
* **Build Tools:**
  * Xcode Command Line Tools (`xcode-select --install`).
  * CMake 3.28 or newer.
  * Ninja build system (`brew install ninja`).
  * Python 3.10+ (for tests and Python bindings; `brew install uv` recommended).

---

### Building from Source

Clone the repository and build using CMake presets:

```bash
# Clone the repository
git clone https://github.com/AshitomW/MUDA.git
cd MUDA

# Configure Release build with Ninja
cmake --preset release

# Build compiler, runtime, libraries, tools, and benchmarks
cmake --build --preset release

# Run the test suite (35 automated tests)
ctest --preset release --output-on-failure
```

Alternatively, you can build with standard CMake commands:

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja -C build
ctest --test-dir build --output-on-failure
```

---

### Verifying Your Setup

Run the built-in diagnostic tool to check your GPU, toolchain, and run a quick compute smoke test:

```bash
./build/tools/muda doctor --smoke-test
```

Example output:
```text
muda version: 0.1.0
host os: Darwin (arm64)
device: Apple M1
  backend: metal
  unified memory: yes
  working set: 5726633984 bytes
  max buffer length: 4294967296 bytes
  max threads per threadgroup: 1024
  max threadgroup memory: 32768 bytes
metal toolchain: available
cache directory: ~/.cache/muda/v1 (writable)
libcuda.dylib: .../build/runtime/libcuda.dylib
gpu smoke test: passed
```

---

### Environment Setup

To use MUDA with existing build systems or Python packages, configure your environment to point to MUDA's `cuda-home` layout:

```bash
# Point CUDA_HOME to MUDA's build or install directory
export CUDA_HOME="$(pwd)/build/cuda-home"
export PATH="$CUDA_HOME/bin:$PATH"
export DYLD_LIBRARY_PATH="$CUDA_HOME/lib:$DYLD_LIBRARY_PATH"
```

---

### Compiling Your First Kernel

Given a simple CUDA file `vector_add.cu`:

```cpp
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    const int n = 1024;
    size_t bytes = n * sizeof(float);

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    vectorAdd<<<(n + 255) / 256, 256>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    printf("Kernel executed successfully on Apple Silicon!\n");
    return 0;
}
```

You can compile and run it directly:

```bash
# Compile to a native macOS binary
mudac vector_add.cu -o vector_add

# Run on your Apple Silicon GPU
./vector_add

# Or inspect the generated Metal Shading Language
mudac vector_add.cu -S -o vector_add.metal
```

---

## Python & Framework Usage

### Numba Integration

You can instruct Numba to use MUDA's `libcuda.dylib` by setting the `NUMBA_CUDA_DRIVER` environment variable:

```bash
export NUMBA_CUDA_DRIVER="$(muda path libcuda.dylib)"
python3 -c "from numba import cuda; print('Discovered GPU:', cuda.gpus[0].name)"
```

Numba `@cuda.jit` functions will compile PTX in memory, which MUDA JIT-compiles directly into Metal compute pipelines on the fly.

### PyTorch Custom CUDA Extensions

MUDA includes a PyTorch bridge in `python/muda` to help compile custom CUDA C++ extensions with `nvcc`:

```python
import os
import torch
from muda.pytorch_bridge import configure_environment, compile_cuda_extension

# Configure environment paths (CUDA_HOME, DYLD_LIBRARY_PATH)
configure_environment()

# Compile a custom CUDA extension into a dylib
ext = compile_cuda_extension(
    name="custom_ops",
    sources=["custom_kernel.cu"],
    extra_cflags=["-O3"]
)

# Call compiled kernels from PyTorch autograd functions or tensors
```

---

## Verification & Testing

MUDA includes 35 automated test suites executed via CTest, covering:

| Category | Tests | Description |
|---|---|---|
| **Language & Primitives** | `vector_add_test`, `warp_intrinsics_test`, `atomics_test`, `half_vector_test` | Shuffle, ballot, 32/64-bit atomics, float CAS, FP16 vectors |
| **Numerical Parity & Emulation** | `corpus_test`, `fp64_test` | 30 numerical parity kernels comparing Metal output against CPU reference results, plus double-precision FP64 emulation (Fast48 & IEEE-754 bitfields) |
| **Compiler, PTX & SASS** | `compiler_test`, `ptx_lexer_test`, `ptx_parser_test`, `ptx_to_mmir_test`, `cubin_test`, `compiler_cache_test` | AST rewriting, PTX shift-reduce parsing, CFG relooper, ELF64/Fatbinary parsing, 128-bit SASS disassembly & state-union lifting, and content cache |
| **Runtime & Driver** | `cuda_runtime_test`, `driver_api_test`, `buffer_pool_test`, `numba_mock_test` | Unified memory, streams, events, buffer pool reuse, and in-memory module loading |
| **Acceleration Libraries** | `cublas_test`, `cufft_test`, `cudnn_test`, `cusparse_test`, `curand_test`, `nvrtc_test` | cuBLAS GEMM, cuFFT 1D/2D transforms, cuDNN convolutions/activations, cuSPARSE SpMV/SpMM, cuRAND Philox RNG, and NVRTC compilation |
| **Integration & Tooling** | `header_hygiene_test`, `muda_cli_test`, `python_*_test` | CLI commands, Python ctypes ABI, Numba CUDA execution, and PyTorch extensions |

To run the full suite:

```bash
ctest --test-dir build --output-on-failure
```

---

## Performance Benchmarks

A dedicated benchmark suite is available under `benchmarks/runtime_bench` to measure micro-latencies, memory transfer bandwidth, and math library throughput:

```bash
# Run all benchmark suites
./build/benchmarks/runtime_bench

# Or run specific benchmark categories
./build/benchmarks/runtime_bench --micro        # CPU enqueue latency, dispatch overhead, event timing
./build/benchmarks/runtime_bench --throughput   # Streaming memory bandwidth, GEMM compute throughput
./build/benchmarks/runtime_bench --libraries    # cuBLAS, cuFFT, cuDNN, cuSPARSE, and cuRAND throughput
```

---

## Contributing

Bug reports, test cases, and pull requests are welcome. When submitting changes:

1. Ensure code compiles cleanly with `-Wall -Wextra -Wpedantic -Wshadow -Wconversion` without warnings.
2. Format code with `clang-format` before opening a pull request.
3. Verify that all 35 test suites pass:
   ```bash
   ctest --test-dir build --output-on-failure
   ```

---

## License

MUDA is licensed under the **Apache License 2.0 with LLVM Exceptions**. See the [LICENSE](LICENSE) file for full license text.

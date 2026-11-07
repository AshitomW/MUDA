# MUDA: CUDA-to-Metal Compatibility Layer for Apple Silicon

MUDA is an experimental, clean-room CUDA-to-Metal translation engine and runtime compatibility layer designed to run CUDA programs and PTX assembly directly on Apple Silicon GPUs without an NVIDIA GPU.

> [!NOTE]
> **Project Scope & Maturity**: MUDA is an active engineering project implementing a focused, verified subset of CUDA and PTX. It is **not** a 100% complete drop-in replacement for the full NVIDIA CUDA Toolkit or precompiled binary SASS. Existing CUDA code relying on unsupported features (e.g. device-side dynamic parallelism, function pointers, or proprietary hardware libraries) will require adjustment or emit explicit rejection diagnostics.

---

## Why It Exists

Apple Silicon GPUs do not natively execute CUDA code, PTX, or NVIDIA SASS machine instructions. CUDA applications are tailored to NVIDIA's compilation pipeline, memory architecture, and driver model:
- **Missing Toolchain on macOS:** macOS lacks `nvcc`, the NVIDIA CUDA Toolkit, and `libcuda.so` / `libcudart.so`.
- **Architectural Disparities:** Metal Shading Language (MSL) has different memory models, no support for `goto` or labeled jump targets, and distinct threadgroup execution semantics.
- **Dynamic Workflows:** Frameworks like Python Numba (`@cuda.jit`) generate PTX in-memory at runtime and load it dynamically through the CUDA Driver API.

MUDA bridges this gap by translating source and PTX into Metal, providing runtime shims for `libcudart` and `libcuda`, and rejecting unsupported patterns with clear diagnostics.

---

## Current Status & Capabilities

```
+-----------------------------------------------------------------------------------+
|                                 User Applications                                 |
|          CUDA C++ (.cu) Sources           |        Numba / Python @cuda.jit       |
+-------------------------------------------+---------------------------------------+
                     |                                          |
                     v                                          v
+-------------------------------------------+---------------------------------------+
|              mudac Compiler               |            libcuda.dylib              |
|   AST Rewriter -> MMIR -> MSL / Metallib  |    cuInit, cuCtx*, cuMem*, cuLaunch*  |
+-------------------------------------------+---------------------------------------+
                     |                                          |
                     +--------------------+---------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                                  MUDA Core Engine                                 |
|  - PTX Translation (Arena, Table-Driven Parser, CFG Structurizer, In-Memory JIT)  |
|  - Runtime Subsystems (BufferPool, MemoryManager, StreamManager, LaunchEngine)    |
|  - Metal Backend (MTLDevice, MTLCommandQueue, MTLSharedEvent, Metal Shading Lang) |
+-----------------------------------------------------------------------------------+
```

### 1. What Is Working Today

#### Compiler (`mudac` & `nvcc`)
- **Compilation Targets:** Compiles `.cu` and `.ptx` files into native macOS executables, shared libraries (`-shared` / `-dynamiclib`), object files (`-c`), Metal Shading Language (`-S`), or `.metallib` (`--emit-metallib`).
- **`nvcc` Compatibility Wrapper:** Supports `-shared`, `-fPIC`, `-D`, `-U`, `-Xcompiler`, `-Xlinker`, `-gencode`, `-arch`, `--expt-relaxed-constexpr`, and `--muda-print-metadata`.
- **Intermediate Representation (MMIR):** Validates kernel signatures, parameter bindings, access qualifiers, and shared memory allocations.
- **Content-Addressed Cache v2:** Eliminates redundant recompilations via content hashing, dependency closure tracking, and fast-math flags (`~/.cache/muda/v1`), inspectable and verifiable via `muda cache <info|clear|verify>`.
- **Diagnostics (`ALY-*`):** Generates deterministic diagnostic errors and source maps when unsupported constructs are detected.

#### Acceleration Libraries & Runtime Compilation (`libcublas`, `libcufft`, `libcudnn`, `libcusparse`, `libnvrtc`)
- **cuBLAS (`libcublas.dylib`):** Level-1 (`Saxpy`, `Scopy`, `Sscal`, `Sswap`, `Sdot`, `Snrm2`, `Sasum`, `Isamax`), Level-2 (`Sgemv`), and Level-3 (`Sgemm`, `Hgemm`, `GemmEx`, `SgemmStridedBatched`, `Sgeam`) backed by Apple `MPSMatrixMultiplication` and `simdgroup_float8x8` shaders.
- **cuFFT (`libcufft.dylib`):** 1D, 2D, 3D, and batched `cufftPlanMany` (`C2C`, `R2C`, `C2R`) with Stockham Radix-2 and general-radix Metal kernels.
- **cuDNN (`libcudnn.dylib`):** 2D Convolutions (forward, backward-data, backward-filter, backward-bias), Activations (ReLU, Sigmoid, Tanh, ELU, Swish), Pooling (Max, Avg), Softmax, and Batch Normalization (inference & training).
- **cuSPARSE (`libcusparse.dylib`):** CSR sparse matrix descriptors, `cusparseSpMV`, and `cusparseSpMM`.
- **NVRTC (`libnvrtc.dylib`):** Runtime compilation of CUDA C++ strings (`nvrtcCreateProgram`, `nvrtcCompileProgram`, `nvrtcGetPTX`, `nvrtcGetLoweredName`) directly into in-memory Metal pipelines via `cuModuleLoadData`.
- **PyTorch Custom CUDA Extension Bridge (`python/muda`):** Compiles custom `.cu` kernels into `.dylib` extensions and integrates with `torch.autograd.Function` and `torch.Tensor` on Apple Silicon.

#### CUDA Runtime API (`libcudart` / `muda_runtime`)
- **Unified Memory:** `cudaMalloc`, `cudaMallocManaged`, `cudaMallocHost`, and `cudaMallocPitch`.
- **Segregated Buffer Pool:** Power-of-two size-classed allocator (256B to 16MB) to minimize heap allocations and launch overhead.
- **Transfers & Strides:** 1D and 2D synchronous and asynchronous memory copies and sets (`cudaMemcpy`, `cudaMemcpyAsync`, `cudaMemset`, `cudaMemset2D`).
- **Streams & Events:** Non-blocking and default streams mapped to `MTLCommandQueue`, and GPU-timed events mapped to `MTLSharedEvent`.
- **Language Coverage (30-Kernel Parity Corpus):**
  - **Shared Memory & Synchronization:** `__shared__`, `extern __shared__`, and `__syncthreads()`.
  - **Warp Intrinsics:** Shuffle (`__shfl_sync`, `__shfl_down_sync`, `__shfl_xor_sync`), vote (`__all_sync`, `__any_sync`, `__ballot_sync`), and bitwise operations (`__popc`, `__clz`, `__ffs`, `__brev`).
  - **Atomics:** 32-bit and 64-bit integer atomics (`atomicAdd`, `atomicSub`, `atomicExch`, `atomicMin`, `atomicMax`, `atomicAnd`, `atomicOr`, `atomicXor`) and 32-bit float CAS loops.
  - **Half-Precision & Vectors:** `half`, `half2`, `float2/3/4`, `int2/3/4`, and `uint2/3/4` aligned math and constructors.

#### PTX Translation Engine (`compiler/ptx`)
- **Bump-Pointer Memory Arena:** `ptx::Arena` provides $O(1)$ memory allocation and single-pass resets for compiler translation units.
- **Zero-Copy Lexer & Shift-Reduce Parser:** Non-recursive, table-driven parser that constructs a strongly-typed AST from PTX 6.x/7.x inputs.
- **CFG & Structurizer:** Computes basic blocks, register data types, and address-space tracking (`device` vs `threadgroup`). The Relooper engine transforms arbitrary branches into structured MSL loops and guard clauses, falling back to a deterministic state-machine dispatcher for irreducible loops.

#### CUDA Driver API (`libcuda.dylib` & `cuda.h`)
- **Clean-Room C Interface:** Declares and exports standard driver types and functions:
  - Initialization & Device queries (`cuInit`, `cuDeviceGet*`, `cuDeviceTotalMem`, `cuDeviceGetAttribute`).
  - Context lifecycle (`cuCtxCreate`, `cuCtxDestroy`, `cuCtxSetCurrent`, `cuCtxGetCurrent`, `cuCtxSynchronize`).
  - Device and host memory management (`cuMemAlloc`, `cuMemAllocPitch`, `cuMemFree`, `cuMemcpy*`, `cuMemset*`).
  - Streams & Events (`cuStream*`, `cuEvent*`).
  - Dynamic in-memory kernel launch (`cuLaunchKernel`).
- **In-Memory Metal JIT:** `cuModuleLoadData` compiles PTX strings in-process using `-[MTLDevice newLibraryWithSource:options:error:]` without spawning subprocesses or touching the disk, reducing module load latency from ~300 ms to < 15 ms.
- **Numba Integration:** Validated against simulated Numba `@cuda.jit` workflows for elementwise vector addition, shared-memory tree reduction with `bar.sync`, and 2D tiled matrix multiplication.

---

### 2. What Is In Progress & Remaining Roadmap

The project is currently working through remaining Gate B and Gate C deliverables:
- **CUDA Textures v1:** 2D texture memory (`cudaArray`), texture objects (`cudaTextureObject_t`), nearest/linear filtering, and clamp/wrap addressing modes.
- **Device-Side Diagnostics & Printf:** Emulating device `printf(...)` and `assert(...)` via ring-buffer logging.
- **PyTorch Extension Bridge:** Drop-in `nvcc` wrapper script enabling `torch.utils.cpp_extension` to build CUDA C++ modules on macOS without modifying Python code.
- **Phase 4 Library Shims:** Acceleration shims for cuBLAS (SGEMM/HGEMM via Apple MPS / `simdgroup_matrix`), cuFFT, and a subset of cuDNN operations.
- **Extended PTX Instructions:** Widening opcode coverage for specialized vector packing, bit-manipulation, and surface/texture PTX instructions.

---

### 3. Known Limitations (What May Not Work)

> [!WARNING]
> Please review these architectural boundaries before attempting to compile arbitrary CUDA codebases.

1. **No Precompiled Binary (SASS) Execution:** MUDA does **not** decompile or run NVIDIA binary files (`.cubin`, ELF objects). Programs must be compiled from CUDA C++ source (`.cu`) or PTX assembly (`.ptx`).
2. **Unsupported Device Constructs:**
   - **Dynamic Parallelism:** Device-side kernel launches (`<<<...>>>` from inside a kernel) are unsupported by Metal.
   - **Device Function Pointers & Virtual Methods:** Indirect device function calls are rejected.
   - **Device-Side Recursion:** Kernel functions cannot be recursive.
3. **Metal Control Flow Constraints:**
   - Metal Shading Language strictly forbids `goto` statements and labeled jumps.
   - While natural loops (`while`, `for`) and conditionals (`if`/`else`) are cleanly structurized, complex irreducible branching graphs fall back to state-machine dispatchers, which may carry a performance penalty.
4. **Precision and Hardware Differences:**
   - **FP64 (Double Precision):** Apple Silicon GPUs lack native hardware FP64 execution units. Double precision is mapped to single precision (FP32) or emulated in accordance with the project's precision policy.
   - **Warp Size:** Apple Silicon executes threads in 32-wide SIMD groups (matching NVIDIA warp size), but memory coalescing, cache line dimensions, and threadgroup synchronization characteristics differ from NVIDIA architectures.
5. **Driver ABI Subsets:**
   - Only the Driver API subset required for compute and Python/Numba execution is currently implemented. CUDA-OpenGL/Vulkan interop and NVLink peer-to-peer memory APIs are stubs or return `CUDA_ERROR_NOT_SUPPORTED`.

---

## Getting Started

### Prerequisites

* **Hardware:** Apple Silicon Mac (M1, M2, M3, M4 or Pro/Max/Ultra variants).
* **Operating System:** macOS 14.0 (Sonoma) or macOS 15.0+ (Sequoia).
* **Toolchain:**
  * Xcode Command Line Tools (`xcode-select --install`).
  * CMake 3.28 or newer.
  * Ninja build system (`brew install ninja`).

---

### Building from Source

Clone the repository and configure using CMake Presets:

```bash
# Clone the repository
git clone https://github.com/AshitomW/cuda-metal.git
cd cuda-metal

# Configure Release build with Ninja
cmake --preset release

# Build compiler, runtime, libraries, tools, and test suites
cmake --build --preset release

# Run the complete test suite (19 test targets)
ctest --preset release --output-on-failure
```

You can also build manually with standard CMake arguments:

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja -C build
ctest --test-dir build --output-on-failure
```

---

### Installing MUDA

To install headers, the `mudac` compiler driver, and libraries into your system or local prefix:

```bash
# Install to /usr/local (or specify -DCMAKE_INSTALL_PREFIX=/path)
sudo cmake --install build
```

This installs:
- **Headers:** `/usr/local/include/cuda.h`, `/usr/local/include/cuda_runtime.h`, and `/usr/local/include/muda/`
- **Compiler:** `/usr/local/bin/mudac` and `/usr/local/bin/muda`
- **Libraries:**
  - `/usr/local/lib/libcuda.dylib` (CUDA Driver API)
  - `/usr/local/lib/libmuda_runtime.a` (CUDA Runtime API)
  - `/usr/local/lib/libmuda_compiler.a` (Translation & PTX engine)

---

### Linking and Environment Setup

To allow dynamic linkers or Python (`ctypes`, Numba) to locate `libcuda.dylib` on macOS, export the library directory:

```bash
export DYLD_LIBRARY_PATH="/usr/local/lib:$DYLD_LIBRARY_PATH"
# Or if running directly from the build directory:
export DYLD_LIBRARY_PATH="$(pwd)/build/runtime:$DYLD_LIBRARY_PATH"
```

#### Compiling Standalone CUDA Sources
```bash
# Compile and link a CUDA .cu source directly to an executable
mudac samples/vector_add.cu -o vector_add_demo
./vector_add_demo

# Emit Metal Shading Language for inspection
mudac samples/vector_add.cu -S -o vector_add.metal

# Compile PTX directly
mudac kernel.ptx -o kernel_runner
```

#### Running the Performance Benchmarks
```bash
# Execute microbenchmarks and GPU throughput suite
./build/benchmarks/runtime_bench
```

---

## Verification & Test Coverage

MUDA maintains a zero-warning compiler policy under strict Clang flags (`-Wall -Wextra -Wpedantic -Wshadow -Wconversion`). The test suite currently includes:

| Test Group | Test Binary | Coverage |
|---|---|---|
| **Driver API** | `driver_api_test` | Device discovery, context stack, memory alloc/copy, in-memory PTX compilation, and kernel launch |
| **Numba Simulation** | `numba_mock_test` | Elementwise `vectorAdd`, shared memory tree reduction (`bar.sync`), and 2D tiled matrix multiply |
| **PTX Engine** | `ptx_lexer_test`, `ptx_parser_test`, `ptx_to_mmir_test` | Tokens, shift-reduce AST generation, CFG, and end-to-end PTX execution on GPU |
| **Numerical Parity** | `corpus_test`, `vector_add_test` | 30 numerical parity kernels comparing Metal output against CPU reference results |
| **Language Coverage** | `warp_intrinsics_test`, `atomics_test`, `half_vector_test` | Shuffle, ballot, 32/64-bit atomics, float CAS, and half-precision vectors |
| **Runtime Infrastructure** | `cuda_runtime_test`, `buffer_pool_test`, `compiler_cache_test` | Streams, events, pitch memory, memory pool reuse, and content-addressed cache |

---

## Contributing & Development

Contributions, bug reports, and kernel test cases are welcome. When submitting changes:
1. Ensure all code compiles cleanly with `-Wall -Wextra -Wpedantic -Wshadow -Wconversion` without warnings.
2. Run `clang-format` across changed files before committing.
3. Ensure all 19 test suites pass: `ctest --test-dir build --output-on-failure`.
4. Follow the engineering standards outlined in [`AGENTS.md`](AGENTS.md) and [`docs/ROADMAP.md`](docs/ROADMAP.md).

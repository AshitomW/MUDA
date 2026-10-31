# MUDA Usage, Examples & Benchmarks Guide

This guide covers building MUDA, compiling `.cu` programs with `mudac`, inspected generated Metal Shading Language (MSL), writing multi-kernel and shared-memory programs, using streams and events, and running the Apple Silicon GPU benchmark suite.

---

## Table of Contents

1. [Building & Verifying the Environment](#1-building--verifying-the-environment)
2. [`mudac` Compiler CLI Reference](#2-mudac-compiler-cli-reference)
3. [Example 1: Standalone `vector_add.cu` (`<<<...>>>` Launch)](#3-example-1-standalone-vector_addcu--launch)
4. [Example 2: Inspecting Translated Metal Shading Language (`-S`)](#4-example-2-inspecting-translated-metal-shading-language--s)
5. [Example 3: `__device__` Helpers, Math Intrinsics & Shared Memory Reduction](#5-example-3-__device__-helpers-math-intrinsics--shared-memory-reduction)
6. [Example 4: Tiled Matrix Multiplication (`__shared__` + `__syncthreads`)](#6-example-4-tiled-matrix-multiplication-__shared__--__syncthreads)
7. [Example 5: Streams, Events & Asynchronous Copies](#7-example-5-streams-events--asynchronous-copies)
8. [Example 6: 2D Pitched Memory (`cudaMallocPitch` & `cudaMemcpy2D`)](#8-example-6-2d-pitched-memory-cudamallocpitch--cudamemcpy2d)
9. [Running the Benchmark Suite (`runtime_bench`)](#9-running-the-benchmark-suite-runtime_bench)

---

## 1. Building & Verifying the Environment

Configure and build using the CMake presets in [`CMakePresets.json`](../CMakePresets.json):

```bash
# Debug build + test suite
cmake --preset debug
cmake --build --preset debug
ctest --preset debug --output-on-failure

# Release build + test suite
cmake --preset release
cmake --build --preset release
ctest --preset release --output-on-failure
```

Verify that your Apple Silicon GPU and Xcode Metal Toolchain (`xcrun --sdk macosx metal`) are detected using the `muda` CLI:

```bash
./build/release/tools/muda doctor
```

Example output:

```text
device: Apple M1
  backend: metal
  working set: 5726625792 bytes
  max threads per threadgroup: 1024
metal toolchain: available
```

---

## 2. `mudac` Compiler CLI Reference

`mudac` compiles `.cu` files into native macOS executables, host object files (`.o`), Metal Shading Language (`.metal`), or compiled Metal libraries (`.metallib`).

```text
Usage: mudac [options] <file.cu>

Options:
  -o <path>                 Write output to <path>
  -c                        Compile to host object file (.o) with embedded metallib
  -S                        Emit translated Metal Shading Language (.metal) source
  --emit-metallib           Emit standalone Metal library (.metallib)
  -I <dir>, -I<dir>         Add header include directory
  -L <dir>, -L<dir>         Add library search directory
  -l <lib>, -l<lib>         Link against library <lib>
  -k, --keep-intermediates  Keep generated .metal, .air, and .host.cpp files
  --muda-keep-msl           Keep generated .metal and intermediate files
  --muda-save-temps         Keep all intermediate compilation artifacts
  --muda-cache=<dir>        Use <dir> for the compiler artifact cache
  --muda-no-cache           Disable the compiler artifact cache
  -h, --help                Show help
  --version                 Show version
```

### Common Workflows

| Goal | Command |
|---|---|
| Compile `.cu` directly to a macOS executable | `./build/release/tools/mudac samples/vector_add.cu -o vector_add` |
| Compile `.cu` to a host `.o` object file | `./build/release/tools/mudac -c samples/vector_add.cu -o vector_add.o` |
| Emit translated `.metal` source only | `./build/release/tools/mudac -S samples/vector_add.cu -o vector_add.metal` |
| Emit `.metallib` binary only | `./build/release/tools/mudac --emit-metallib samples/vector_add.cu -o vector_add.metallib` |
| Keep intermediate `.metal`, `.air`, and `.host.cpp` | `./build/release/tools/mudac --muda-save-temps samples/vector_add.cu -o vector_add` |

---

## 3. Example 1: Standalone `vector_add.cu` (`<<<...>>>` Launch)

The included [`samples/vector_add.cu`](../samples/vector_add.cu) uses standard CUDA runtime API calls and `<<<grid, block>>>` launch syntax:

```cuda
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

__global__ void vector_add(const float* a, const float* b, float* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = a[i] + b[i];
  }
}

int main() {
  constexpr int kCount = 1024;
  constexpr std::size_t kBytes = kCount * sizeof(float);

  std::vector<float> a(kCount, 1.0f);
  std::vector<float> b(kCount, 2.0f);
  std::vector<float> out(kCount, 0.0f);

  float *d_a = nullptr, *d_b = nullptr, *d_out = nullptr;
  cudaMalloc(reinterpret_cast<void**>(&d_a), kBytes);
  cudaMalloc(reinterpret_cast<void**>(&d_b), kBytes);
  cudaMalloc(reinterpret_cast<void**>(&d_out), kBytes);

  cudaMemcpy(d_a, a.data(), kBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b.data(), kBytes, cudaMemcpyHostToDevice);

  vector_add<<<4, 256>>>(d_a, d_b, d_out, kCount);

  cudaMemcpy(out.data(), d_out, kBytes, cudaMemcpyDeviceToHost);
  std::printf("%.6f\n", static_cast<double>(out[0]));

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_out);
  return 0;
}
```

Compile and run with `mudac`:

```bash
./build/release/tools/mudac samples/vector_add.cu -o /tmp/vector_add
/tmp/vector_add
```

Output:

```text
3.000000
```

---

## 4. Example 2: Inspecting Translated Metal Shading Language (`-S`)

To inspect how `mudac` lowers CUDA kernels, address spaces (`device`, `constant`, `threadgroup`), and thread indexing into Metal Shading Language:

```bash
./build/release/tools/mudac -S samples/vector_add.cu -o /tmp/vector_add.metal
cat /tmp/vector_add.metal
```

Generated MSL:

```metal
#include <metal_stdlib>
using namespace metal;

kernel void vector_add(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant int& n [[buffer(3)]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 blockDim [[threads_per_threadgroup]],
    uint3 gridDim [[threadgroups_per_grid]]) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = a[i] + b[i];
  }
}
```

---

## 5. Example 3: `__device__` Helpers, Math Intrinsics & Shared Memory Reduction

`mudac` supports `__device__` functions, FP32 math intrinsics (`fmaf`, `fminf`, `fmaxf`, `sqrtf`, `rsqrtf`, `fabsf`, `expf`, `logf`, `sinf`, `cosf`, etc.), static `__shared__` arrays, dynamic `extern __shared__` memory, and `__syncthreads()`.

```cuda
// Save as /tmp/dot_reduce.cu
#include <cuda_runtime.h>
#include <cstdio>

__device__ float pair_sum(float a, float b) {
  return a + b;
}

__global__ void dot_reduce(const float* a, const float* b, float* block_sums, int n) {
  __shared__ float sdata[64];
  const int tid = threadIdx.x;
  const int idx = blockIdx.x * blockDim.x + tid;

  sdata[tid] = (idx < n) ? fmaf(a[idx], b[idx], 0.0f) : 0.0f;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sdata[tid] = pair_sum(sdata[tid], sdata[tid + stride]);
    }
    __syncthreads();
  }

  if (tid == 0) {
    block_sums[blockIdx.x] = sdata[0];
  }
}

int main() {
  constexpr int kN = 256;
  constexpr int kBlock = 64;
  constexpr int kGrid = kN / kBlock;

  float *a = nullptr, *b = nullptr, *sums = nullptr;
  cudaMallocManaged(reinterpret_cast<void**>(&a), kN * sizeof(float));
  cudaMallocManaged(reinterpret_cast<void**>(&b), kN * sizeof(float));
  cudaMallocManaged(reinterpret_cast<void**>(&sums), kGrid * sizeof(float));

  for (int i = 0; i < kN; ++i) {
    a[i] = 1.0f;
    b[i] = 2.0f;
  }

  dot_reduce<<<kGrid, kBlock>>>(a, b, sums, kN);
  cudaDeviceSynchronize();

  float total = 0.0f;
  for (int i = 0; i < kGrid; ++i) {
    total += sums[i];
  }
  std::printf("Dot product: %.1f (expected 512.0)\n", static_cast<double>(total));

  cudaFree(sums);
  cudaFree(b);
  cudaFree(a);
  return 0;
}
```

Compile and run:

```bash
./build/release/tools/mudac /tmp/dot_reduce.cu -o /tmp/dot_reduce && /tmp/dot_reduce
```

---

## 6. Example 4: Tiled Matrix Multiplication (`__shared__` + `__syncthreads`)

2D/3D grids and threadblocks (`dim3`), shared-memory tiles, and barrier synchronization work out of the box:

```cuda
// Save as /tmp/gemm_tiled.cu
#include <cuda_runtime.h>
#include <cstdio>

__global__ void gemm_tiled(const float* a, const float* b, float* c, int n) {
  __shared__ float tile_a[64];
  __shared__ float tile_b[64];
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int row = blockIdx.y * 8 + ty;
  const int col = blockIdx.x * 8 + tx;

  float acc = 0.0f;
  const int tiles = (n + 7) / 8;
  for (int t = 0; t < tiles; ++t) {
    const int a_col = t * 8 + tx;
    const int b_row = t * 8 + ty;
    tile_a[ty * 8 + tx] = (row < n && a_col < n) ? a[row * n + a_col] : 0.0f;
    tile_b[ty * 8 + tx] = (b_row < n && col < n) ? b[b_row * n + col] : 0.0f;
    __syncthreads();

    for (int k = 0; k < 8; ++k) {
      acc = fmaf(tile_a[ty * 8 + k], tile_b[k * 8 + tx], acc);
    }
    __syncthreads();
  }

  if (row < n && col < n) {
    c[row * n + col] = acc;
  }
}

int main() {
  constexpr int kN = 16;
  constexpr std::size_t kBytes = kN * kN * sizeof(float);

  float *a = nullptr, *b = nullptr, *c = nullptr;
  cudaMallocManaged(reinterpret_cast<void**>(&a), kBytes);
  cudaMallocManaged(reinterpret_cast<void**>(&b), kBytes);
  cudaMallocManaged(reinterpret_cast<void**>(&c), kBytes);

  for (int i = 0; i < kN * kN; ++i) {
    a[i] = 1.0f;
    b[i] = 2.0f;
  }

  const dim3 block(8, 8);
  const dim3 grid(kN / 8, kN / 8);
  gemm_tiled<<<grid, block>>>(a, b, c, kN);
  cudaDeviceSynchronize();

  std::printf("C[0,0] = %.1f (expected %.1f)\n",
              static_cast<double>(c[0]),
              static_cast<double>(kN * 2.0f));

  cudaFree(c);
  cudaFree(b);
  cudaFree(a);
  return 0;
}
```

---

## 7. Example 5: Streams, Events & Asynchronous Copies

MUDA maps each `cudaStream_t` to an ordered `MTLCommandQueue` and each `cudaEvent_t` to an `MTLSharedEvent` with GPU completion timestamps:

```cuda
// Save as /tmp/streams_events.cu
#include <cuda_runtime.h>
#include <cstdio>

__global__ void scale_kernel(float* data, float factor, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    data[i] *= factor;
  }
}

int main() {
  constexpr int kN = 4096;
  constexpr std::size_t kBytes = kN * sizeof(float);

  cudaStream_t stream = nullptr;
  cudaEvent_t start = nullptr, stop = nullptr;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  float *d_src = nullptr, *d_dst = nullptr;
  cudaMalloc(reinterpret_cast<void**>(&d_src), kBytes);
  cudaMalloc(reinterpret_cast<void**>(&d_dst), kBytes);

  // Initialize source on host and copy asynchronously
  float* h_pinned = nullptr;
  cudaMallocHost(reinterpret_cast<void**>(&h_pinned), kBytes);
  for (int i = 0; i < kN; ++i) {
    h_pinned[i] = 3.0f;
  }

  cudaEventRecord(start, stream);
  cudaMemcpyAsync(d_src, h_pinned, kBytes, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_dst, d_src, kBytes, cudaMemcpyDeviceToDevice, stream);
  scale_kernel<<<16, 256, 0, stream>>>(d_dst, 4.0f, kN);
  cudaMemcpyAsync(h_pinned, d_dst, kBytes, cudaMemcpyDeviceToHost, stream);
  cudaEventRecord(stop, stream);

  cudaEventSynchronize(stop);

  float elapsed_ms = 0.0f;
  cudaEventElapsedTime(&elapsed_ms, start, stop);
  std::printf("h_pinned[0] = %.1f (expected 12.0), elapsed = %.3f ms\n",
              static_cast<double>(h_pinned[0]),
              static_cast<double>(elapsed_ms));

  cudaFreeHost(h_pinned);
  cudaFree(d_dst);
  cudaFree(d_src);
  cudaEventDestroy(stop);
  cudaEventDestroy(start);
  cudaStreamDestroy(stream);
  return 0;
}
```

---

## 8. Example 6: 2D Pitched Memory (`cudaMallocPitch` & `cudaMemcpy2D`)

Use `cudaMallocPitch`, `cudaMemset2D`, and `cudaMemcpy2D` for 2D row-aligned allocations:

```cpp
constexpr std::size_t kWidthBytes = 64 * sizeof(float);
constexpr std::size_t kHeight = 32;

void* d_pitched = nullptr;
std::size_t pitch = 0;
cudaMallocPitch(&d_pitched, &pitch, kWidthBytes, kHeight);
cudaMemset2D(d_pitched, pitch, 0, kWidthBytes, kHeight);

std::vector<float> host_matrix(64 * kHeight, 7.5f);
cudaMemcpy2D(d_pitched, pitch, host_matrix.data(), kWidthBytes,
             kWidthBytes, kHeight, cudaMemcpyHostToDevice);

cudaFree(d_pitched);
```

---

## 9. Running the Benchmark Suite (`runtime_bench`)

The benchmark target [`benchmarks/runtime_bench.cpp`](../benchmarks/runtime_bench.cpp) measures four core GPU workloads on Apple Silicon:

1. **Empty Kernel Launch Latency**: Asynchronous command-buffer enqueue latency vs. synchronized end-to-end launch latency (`cudaDeviceSynchronize`).
2. **Copy Bandwidth (64 MiB)**: Host-to-Device (`H2D`), Device-to-Device (`D2D`), and Device-to-Host (`D2H`) throughput.
3. **`vector_add` Throughput**: 16M `float` elements (64 MiB per buffer, 192 MiB total memory traffic per launch) timed via `cudaEventElapsedTime`.
4. **Tiled FP32 GEMM Baseline**: `512 x 512` matrix multiplication using `16 x 16` `threadgroup` shared-memory tiles and `fmaf`.

### Build & Run

```bash
cmake --build --preset release
./build/release/benchmarks/runtime_bench
```

### Sample Output (Apple M1)

```text
=== MUDA Benchmark Suite ===
Device: Apple M1 (Unified Memory: Yes)

[1] Empty Kernel Launch Latency (500 iterations):
    Async enqueue latency : 28.98 us / launch
    Synchronized latency  : 32.58 us / launch

[2] Copy Bandwidth (64 MiB, 25 iterations):
    H2D Bandwidth : 19.42 GB/s
    D2D Bandwidth : 19.04 GB/s
    D2H Bandwidth : 21.41 GB/s

[3] vector_add Throughput (N = 16777216, 20 iterations):
    Avg kernel time     : 4.071 ms
    Effective bandwidth : 49.46 GB/s

[4] Tiled GEMM Baseline (512x512 FP32, 20 iterations):
    Avg kernel time : 0.901 ms
    Throughput      : 297.81 GFLOP/s
```

### Running the 10-Kernel Numerical Parity Suite

[`tests/corpus_test.cpp`](../tests/corpus_test.cpp) compiles and validates 10 kernels against CPU reference implementations (`vector_add`, `saxpy`, `scale_clamp`, `dot_reduce`, `transpose_naive`, `transpose_tiled`, `gemm_naive`, `gemm_tiled`, `stencil_1d` with dynamic shared memory, and `row_sum`):

```bash
./build/release/tests/corpus_test
```

Output:

```text
corpus_test: all 10 kernels passed numerical parity
```


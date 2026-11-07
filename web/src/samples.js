/**
 * @file samples.js
 * @brief Curated collection of CUDA C++ and PTX kernels for the MUDA playground.
 */

export const SAMPLES = [
  {
    id: "vector_add",
    name: "Vector Add (Basic)",
    category: "CUDA C++",
    mode: "cuda",
    description: "Standard 1D parallel elementwise addition across device memory pointers.",
    code: `#include <cuda_runtime.h>

__global__ void vector_add(const float* a, const float* b, float* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = a[i] + b[i];
  }
}
`
  },
  {
    id: "saxpy",
    name: "SAXPY (Scalar-Vector FMA)",
    category: "CUDA C++",
    mode: "cuda",
    description: "Single-precision scalar multiplication and vector addition with fmaf intrinsic.",
    code: `#include <cuda_runtime.h>

__global__ void saxpy(float alpha, const float* x, const float* y, float* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = fmaf(alpha, x[i], y[i]);
  }
}
`
  },
  {
    id: "tiled_gemm",
    name: "2D Tiled GEMM (Shared Memory)",
    category: "CUDA C++",
    mode: "cuda",
    description: "Matrix multiplication with 16x16 shared memory cache tiles and barrier synchronization.",
    code: `#include <cuda_runtime.h>

#define TILE_DIM 16

__global__ void tiled_gemm(const float* A, const float* B, float* C, int N) {
  __shared__ float sA[TILE_DIM][TILE_DIM];
  __shared__ float sB[TILE_DIM][TILE_DIM];

  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  const int row = by * TILE_DIM + ty;
  const int col = bx * TILE_DIM + tx;

  float acc = 0.0f;
  const int num_tiles = (N + TILE_DIM - 1) / TILE_DIM;

  for (int t = 0; t < num_tiles; ++t) {
    const int a_col = t * TILE_DIM + tx;
    const int b_row = t * TILE_DIM + ty;

    sA[ty][tx] = (row < N && a_col < N) ? A[row * N + a_col] : 0.0f;
    sB[ty][tx] = (b_row < N && col < N) ? B[b_row * N + col] : 0.0f;
    __syncthreads();

    for (int k = 0; k < TILE_DIM; ++k) {
      acc = fmaf(sA[ty][k], sB[k][tx], acc);
    }
    __syncthreads();
  }

  if (row < N && col < N) {
    C[row * N + col] = acc;
  }
}
`
  },
  {
    id: "matrix_transpose",
    name: "Matrix Transpose (Padding)",
    category: "CUDA C++",
    mode: "cuda",
    description: "Bank-conflict-free 32x32 matrix transpose using padded threadgroup shared memory.",
    code: `#include <cuda_runtime.h>

#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void matrix_transpose_coalesced(const float* idata, float* odata, int width, int height) {
  __shared__ float tile[TILE_DIM][TILE_DIM + 1];

  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
    if (x < width && (y + j) < height) {
      tile[threadIdx.y + j][threadIdx.x] = idata[(y + j) * width + x];
    }
  }
  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;
  y = blockIdx.x * TILE_DIM + threadIdx.y;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
    if (x < height && (y + j) < width) {
      odata[(y + j) * height + x] = tile[threadIdx.x][threadIdx.y + j];
    }
  }
}
`
  },
  {
    id: "dot_reduce",
    name: "Dot Product Reduction",
    category: "CUDA C++",
    mode: "cuda",
    description: "Parallel reduction tree using dynamic tree folding in threadgroup memory.",
    code: `#include <cuda_runtime.h>

__global__ void dot_reduce(const float* a, const float* b, float* block_sums, int n) {
  __shared__ float sdata[64];
  const int tid = threadIdx.x;
  const int idx = blockIdx.x * blockDim.x + tid;

  sdata[tid] = (idx < n) ? (a[idx] * b[idx]) : 0.0f;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    block_sums[blockIdx.x] = sdata[0];
  }
}
`
  },
  {
    id: "warp_shuffle",
    name: "Warp Shuffle Reduction",
    category: "CUDA C++",
    mode: "cuda",
    description: "Fast SIMD-group register reduction using __shfl_down_sync mapped to simd_shuffle_down.",
    code: `#include <cuda_runtime.h>

__inline__ __device__ float warp_reduce_sum(float val) {
  for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

__global__ void device_warp_reduction(const float* in, float* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  float sum = (i < n) ? in[i] : 0.0f;
  sum = warp_reduce_sum(sum);

  if ((threadIdx.x & 31) == 0) {
    out[blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32)] = sum;
  }
}
`
  },
  {
    id: "atomic_histogram",
    name: "Atomic Histogram",
    category: "CUDA C++",
    mode: "cuda",
    description: "Atomic counter updates using atomicAdd mapped to Metal atomic_fetch_add_explicit.",
    code: `#include <cuda_runtime.h>

__global__ void histogram_binning(const int* data, int* bins, int n, int num_bins) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    const int val = data[i];
    if (val >= 0 && val < num_bins) {
      atomicAdd(&bins[val], 1);
    }
  }
}
`
  },
  {
    id: "half_vector",
    name: "Half-Precision (FP16) Vector Math",
    category: "CUDA C++",
    mode: "cuda",
    description: "IEEE 754 half-precision arithmetic mapped directly to Metal native half scalar/vectors.",
    code: `#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void half_vector_fma(const half* a, const half* b, const half* c, half* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = a[i] * b[i] + c[i];
  }
}
`
  },
  {
    id: "ptx_vector_add",
    name: "PTX Assembly Vector Add",
    category: "PTX 6.5 Assembly",
    mode: "ptx",
    description: "Direct PTX virtual assembly translated via bump-pointer arena, lexer, and CFG structurizer.",
    code: `.version 6.5
.target sm_70
.address_size 64

.visible .entry ptx_vector_add(
    .param .u64 param_a,
    .param .u64 param_b,
    .param .u64 param_c,
    .param .u32 param_n
)
{
    .reg .pred  %p<2>;
    .reg .b32   %r<5>;
    .reg .b64   %rd<9>;
    .reg .f32   %f<4>;

    ld.param.u64    %rd1, [param_a];
    ld.param.u64    %rd2, [param_b];
    ld.param.u64    %rd3, [param_c];
    ld.param.u32    %r1, [param_n];

    mov.u32         %r2, %ctaid.x;
    mov.u32         %r3, %ntid.x;
    mov.u32         %r4, %tid.x;
    mad.lo.s32      %r0, %r2, %r3, %r4;

    setp.ge.s32     %p1, %r0, %r1;
    @%p1 bra        $L_exit;

    cvta.to.global.u64 %rd4, %rd1;
    mul.wide.s32    %rd5, %r0, 4;
    add.s64         %rd6, %rd4, %rd5;
    ld.global.f32   %f1, [%rd6];

    cvta.to.global.u64 %rd7, %rd2;
    add.s64         %rd8, %rd7, %rd5;
    ld.global.f32   %f2, [%rd8];

    add.f32         %f3, %f1, %f2;

    cvta.to.global.u64 %rd9, %rd3;
    add.s64         %rd9, %rd9, %rd5;
    st.global.f32   [%rd9], %f3;

$L_exit:
    ret;
}
`
  }
];


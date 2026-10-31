/**
 * @file tiled_gemm_pipeline.cu
 * @brief Multi-kernel CUDA sample demonstrating tiled shared-memory GEMM, __device__ helpers,
 *        math intrinsics, streams, and GPU event timing.
 */

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

__device__ float relu_clamp(float x, float max_val) {
  return fminf(fmaxf(x, 0.0f), max_val);
}

__global__ void gemm_tiled(const float* a, const float* b, float* c, int n) {
  __shared__ float tile_a[64];
  __shared__ float tile_b[64];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int row = blockIdx.y * 8 + ty;
  const int col = blockIdx.x * 8 + tx;

  float acc = 0.0f;
  const int num_tiles = (n + 7) / 8;

  for (int t = 0; t < num_tiles; ++t) {
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

__global__ void bias_relu_clamp(float* c, const float* bias, float max_val, int total) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total) {
    c[idx] = relu_clamp(c[idx] + bias[idx], max_val);
  }
}

int main() {
  constexpr int kN = 32;
  constexpr int kElements = kN * kN;
  constexpr std::size_t kBytes = static_cast<std::size_t>(kElements) * sizeof(float);

  std::vector<float> h_a(kElements);
  std::vector<float> h_b(kElements);
  std::vector<float> h_bias(kElements, -2.5f);
  std::vector<float> h_out(kElements, 0.0f);

  for (int r = 0; r < kN; ++r) {
    for (int c = 0; c < kN; ++c) {
      h_a[r * kN + c] = static_cast<float>((r + c) % 5) * 0.5f;
      h_b[r * kN + c] = (r == c) ? 2.0f : 0.25f;
    }
  }

  float* d_a = nullptr;
  float* d_b = nullptr;
  float* d_bias = nullptr;
  float* d_c = nullptr;
  if (cudaMalloc(reinterpret_cast<void**>(&d_a), kBytes) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&d_b), kBytes) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&d_bias), kBytes) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&d_c), kBytes) != cudaSuccess) {
    std::fprintf(stderr, "cudaMalloc failed\n");
    return 1;
  }

  cudaStream_t stream = nullptr;
  cudaEvent_t ev_start = nullptr;
  cudaEvent_t ev_stop = nullptr;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  cudaEventCreate(&ev_start);
  cudaEventCreate(&ev_stop);

  cudaEventRecord(ev_start, stream);
  cudaMemcpyAsync(d_a, h_a.data(), kBytes, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_b, h_b.data(), kBytes, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_bias, h_bias.data(), kBytes, cudaMemcpyHostToDevice, stream);

  const dim3 block_2d(8, 8);
  const dim3 grid_2d(kN / 8, kN / 8);
  gemm_tiled<<<grid_2d, block_2d, 0, stream>>>(d_a, d_b, d_c, kN);

  constexpr float kClampMax = 25.0f;
  bias_relu_clamp<<<(kElements + 127) / 128, 128, 0, stream>>>(
      d_c, d_bias, kClampMax, kElements);

  cudaMemcpyAsync(h_out.data(), d_c, kBytes, cudaMemcpyDeviceToHost, stream);
  cudaEventRecord(ev_stop, stream);
  cudaEventSynchronize(ev_stop);

  float elapsed_ms = 0.0f;
  cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop);

  // Verify against CPU reference
  for (int r = 0; r < kN; ++r) {
    for (int c = 0; c < kN; ++c) {
      float acc = 0.0f;
      for (int k = 0; k < kN; ++k) {
        acc = std::fma(h_a[r * kN + k], h_b[k * kN + c], acc);
      }
      const float expected = std::fmin(std::fmax(acc + h_bias[r * kN + c], 0.0f), kClampMax);
      const float actual = h_out[r * kN + c];
      if (std::fabs(actual - expected) > 1e-4f) {
        std::fprintf(stderr, "mismatch at (%d, %d): got %.6f, expected %.6f\n",
                     r, c, static_cast<double>(actual), static_cast<double>(expected));
        return 1;
      }
    }
  }

  std::printf("tiled_gemm_pipeline: PASS (%dx%d matrix, out[0]=%.2f, out[last]=%.2f, %.3f ms)\n",
              kN, kN, static_cast<double>(h_out.front()), static_cast<double>(h_out.back()),
              static_cast<double>(elapsed_ms));

  cudaEventDestroy(ev_stop);
  cudaEventDestroy(ev_start);
  cudaStreamDestroy(stream);
  cudaFree(d_c);
  cudaFree(d_bias);
  cudaFree(d_b);
  cudaFree(d_a);
  return 0;
}

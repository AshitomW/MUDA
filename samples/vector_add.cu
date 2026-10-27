/**
 * @file vector_add.cu
 * @brief Sample CUDA vector-add kernel used by the prototype.
 */

#include <cuda_runtime.h>

#include <cstddef>
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
  constexpr int kThreads = 256;
  constexpr int kBlocks = (kCount + kThreads - 1) / kThreads;
  constexpr std::size_t kBytes = kCount * sizeof(float);

  std::vector<float> a(kCount, 1.0f);
  std::vector<float> b(kCount, 2.0f);
  std::vector<float> out(kCount, 0.0f);

  float* device_a = nullptr;
  float* device_b = nullptr;
  float* device_out = nullptr;
  if (cudaMalloc(&device_a, kBytes) != cudaSuccess) return 1;
  if (cudaMalloc(&device_b, kBytes) != cudaSuccess) return 1;
  if (cudaMalloc(&device_out, kBytes) != cudaSuccess) return 1;

  if (cudaMemcpy(device_a, a.data(), kBytes, cudaMemcpyHostToDevice) != cudaSuccess) return 1;
  if (cudaMemcpy(device_b, b.data(), kBytes, cudaMemcpyHostToDevice) != cudaSuccess) return 1;

  vector_add<<<kBlocks, kThreads>>>(device_a, device_b, device_out, kCount);

  if (cudaMemcpy(out.data(), device_out, kBytes, cudaMemcpyDeviceToHost) != cudaSuccess) return 1;

  cudaFree(device_a);
  cudaFree(device_b);
  cudaFree(device_out);

  std::printf("%s\n", out[0] == 3.0f ? "ok" : "failed");
  return out[0] == 3.0f ? 0 : 1;
}

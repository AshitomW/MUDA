/**
 * @file vector_add.cu
 * @brief Sample CUDA vector-add kernel used by the prototype.
 */

#include <cuda_runtime.h>

#include <cmath>
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
  constexpr std::size_t kCount = 1024;
  constexpr unsigned int kThreads = 256;
  constexpr unsigned int kBlocks = static_cast<unsigned int>((kCount + kThreads - 1) / kThreads);
  constexpr std::size_t kBytes = kCount * sizeof(float);
  constexpr float kExpectedSum = 3.0f;
  constexpr float kTolerance = 1e-5f;

  std::vector<float> a(kCount, 1.0f);
  std::vector<float> b(kCount, 2.0f);
  std::vector<float> out(kCount, 0.0f);

  float* device_a = nullptr;
  float* device_b = nullptr;
  float* device_out = nullptr;
  if (cudaMalloc(reinterpret_cast<void**>(&device_a), kBytes) != cudaSuccess) {
    return 1;
  }
  if (cudaMalloc(reinterpret_cast<void**>(&device_b), kBytes) != cudaSuccess) {
    return 1;
  }
  if (cudaMalloc(reinterpret_cast<void**>(&device_out), kBytes) != cudaSuccess) {
    return 1;
  }

  if (cudaMemcpy(device_a, a.data(), kBytes, cudaMemcpyHostToDevice) != cudaSuccess) {
    return 1;
  }
  if (cudaMemcpy(device_b, b.data(), kBytes, cudaMemcpyHostToDevice) != cudaSuccess) {
    return 1;
  }

  vector_add<<<kBlocks, kThreads>>>(device_a, device_b, device_out, static_cast<int>(kCount));

  if (cudaMemcpy(out.data(), device_out, kBytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
    return 1;
  }

  cudaFree(device_a);
  cudaFree(device_b);
  cudaFree(device_out);

  const bool is_ok = std::fabs(out[0] - kExpectedSum) <= kTolerance;
  std::printf("%s\n", is_ok ? "ok" : "failed");
  return is_ok ? 0 : 1;
}

/**
 * @file curand_test.cpp
 * @brief Comprehensive verification test suite for the MUDA cuRAND shim (libcurand.dylib).
 */

#include "curand.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>

#include "cuda_runtime.h"

namespace {

bool Expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    return false;
  }
  return true;
}

bool ExpectClose(float actual, float expected, float tol, const char* message) {
  if (std::fabs(actual - expected) > tol) {
    std::cerr << "FAIL: " << message << " (actual=" << actual << ", expected=" << expected
              << ", diff=" << std::fabs(actual - expected) << ")\n";
    return false;
  }
  return true;
}

bool ExpectCloseDouble(double actual, double expected, double tol, const char* message) {
  if (std::fabs(actual - expected) > tol) {
    std::cerr << "FAIL: " << message << " (actual=" << actual << ", expected=" << expected
              << ", diff=" << std::fabs(actual - expected) << ")\n";
    return false;
  }
  return true;
}

bool TestGeneratorLifecycle() {
  std::cout << "Testing cuRAND generator lifecycle and property queries...\n";

  curandGenerator_t gen = nullptr;
  if (!Expect(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) == CURAND_STATUS_SUCCESS,
              "curandCreateGenerator GPU should succeed")) {
    return false;
  }
  if (!Expect(gen != nullptr, "generator handle must be non-null")) {
    return false;
  }

  int version = 0;
  if (!Expect(curandGetVersion(&version) == CURAND_STATUS_SUCCESS && version >= 10000,
              "curandGetVersion should report valid version")) {
    return false;
  }

  int major = 0;
  if (!Expect(curandGetProperty(MAJOR_VERSION, &major) == CURAND_STATUS_SUCCESS && major >= 10,
              "curandGetProperty MAJOR_VERSION should succeed")) {
    return false;
  }

  cudaStream_t stream = nullptr;
  if (!Expect(cudaStreamCreate(&stream) == cudaSuccess, "cudaStreamCreate should succeed")) {
    return false;
  }
  if (!Expect(curandSetStream(gen, stream) == CURAND_STATUS_SUCCESS,
              "curandSetStream should succeed")) {
    return false;
  }
  cudaStream_t stream_out = nullptr;
  if (!Expect(curandGetStream(gen, &stream_out) == CURAND_STATUS_SUCCESS && stream_out == stream,
              "curandGetStream should return bound stream")) {
    return false;
  }

  if (!Expect(curandSetPseudoRandomGeneratorSeed(gen, 1337ull) == CURAND_STATUS_SUCCESS,
              "curandSetPseudoRandomGeneratorSeed should succeed")) {
    return false;
  }
  if (!Expect(curandSetGeneratorOffset(gen, 1024ull) == CURAND_STATUS_SUCCESS,
              "curandSetGeneratorOffset should succeed")) {
    return false;
  }
  if (!Expect(
          curandSetGeneratorOrdering(gen, CURAND_ORDERING_PSEUDO_DEFAULT) == CURAND_STATUS_SUCCESS,
          "curandSetGeneratorOrdering should succeed")) {
    return false;
  }

  if (!Expect(curandDestroyGenerator(gen) == CURAND_STATUS_SUCCESS,
              "curandDestroyGenerator should succeed")) {
    return false;
  }
  (void)cudaStreamDestroy(stream);

  // Host generator lifecycle
  curandGenerator_t host_gen = nullptr;
  if (!Expect(curandCreateGeneratorHost(&host_gen, CURAND_RNG_PSEUDO_PHILOX4_32_10) ==
                  CURAND_STATUS_SUCCESS,
              "curandCreateGeneratorHost should succeed")) {
    return false;
  }
  if (!Expect(curandDestroyGenerator(host_gen) == CURAND_STATUS_SUCCESS,
              "curandDestroyGenerator host should succeed")) {
    return false;
  }

  return true;
}

bool TestHostGeneration() {
  std::cout << "Testing cuRAND host generation (Philox, Uniform, Normal, Poisson)...\n";

  curandGenerator_t host_gen = nullptr;
  if (!Expect(
          curandCreateGeneratorHost(&host_gen, CURAND_RNG_PSEUDO_DEFAULT) == CURAND_STATUS_SUCCESS,
          "curandCreateGeneratorHost failed")) {
    return false;
  }
  (void)curandSetPseudoRandomGeneratorSeed(host_gen, 42ull);

  // 1. Uniform float in (0, 1]
  constexpr std::size_t kNumSamples = 100000;
  std::vector<float> uniform_f(kNumSamples, 0.0f);
  if (!Expect(
          curandGenerateUniform(host_gen, uniform_f.data(), kNumSamples) == CURAND_STATUS_SUCCESS,
          "curandGenerateUniform host failed")) {
    return false;
  }

  double sum_u = 0.0;
  double sum_u_sq = 0.0;
  for (float val : uniform_f) {
    if (!Expect(val > 0.0f && val <= 1.0f, "Uniform sample must be in (0, 1]")) {
      return false;
    }
    sum_u += static_cast<double>(val);
    sum_u_sq += static_cast<double>(val) * static_cast<double>(val);
  }
  const double mean_u = sum_u / static_cast<double>(kNumSamples);
  const double var_u = (sum_u_sq / static_cast<double>(kNumSamples)) - (mean_u * mean_u);

  // Theoretical: mean = 0.5, variance = 1/12 ≈ 0.08333
  if (!ExpectClose(static_cast<float>(mean_u), 0.5f, 0.01f, "Host uniform mean should be ~0.5")) {
    return false;
  }
  if (!ExpectClose(static_cast<float>(var_u), 0.08333f, 0.005f,
                   "Host uniform variance should be ~0.0833")) {
    return false;
  }

  // 2. Normal float (mean = 10.0, stddev = 3.0)
  std::vector<float> normal_f(kNumSamples, 0.0f);
  if (!Expect(curandGenerateNormal(host_gen, normal_f.data(), kNumSamples, 10.0f, 3.0f) ==
                  CURAND_STATUS_SUCCESS,
              "curandGenerateNormal host failed")) {
    return false;
  }

  double sum_n = 0.0;
  double sum_n_sq = 0.0;
  for (float val : normal_f) {
    sum_n += static_cast<double>(val);
    sum_n_sq += static_cast<double>(val) * static_cast<double>(val);
  }
  const double mean_n = sum_n / static_cast<double>(kNumSamples);
  const double var_n = (sum_n_sq / static_cast<double>(kNumSamples)) - (mean_n * mean_n);

  if (!ExpectClose(static_cast<float>(mean_n), 10.0f, 0.05f, "Host normal mean should be ~10.0")) {
    return false;
  }
  if (!ExpectClose(static_cast<float>(var_n), 9.0f, 0.25f, "Host normal variance should be ~9.0")) {
    return false;
  }

  // 3. Poisson (lambda = 15.0)
  std::vector<unsigned int> poisson(kNumSamples, 0u);
  if (!Expect(curandGeneratePoisson(host_gen, poisson.data(), kNumSamples, 15.0) ==
                  CURAND_STATUS_SUCCESS,
              "curandGeneratePoisson host failed")) {
    return false;
  }

  double sum_p = 0.0;
  for (unsigned int val : poisson) {
    sum_p += static_cast<double>(val);
  }
  const double mean_p = sum_p / static_cast<double>(kNumSamples);
  if (!ExpectCloseDouble(mean_p, 15.0, 0.15, "Host Poisson mean should be ~15.0")) {
    return false;
  }

  (void)curandDestroyGenerator(host_gen);
  return true;
}

bool TestGpuGeneration() {
  std::cout << "Testing cuRAND GPU generation on Apple Silicon...\n";

  curandGenerator_t gpu_gen = nullptr;
  if (!Expect(curandCreateGenerator(&gpu_gen, CURAND_RNG_PSEUDO_DEFAULT) == CURAND_STATUS_SUCCESS,
              "curandCreateGenerator GPU failed")) {
    return false;
  }
  (void)curandSetPseudoRandomGeneratorSeed(gpu_gen, 9999ull);

  constexpr std::size_t kNumSamples = 65536;

  // 1. GPU Uniform float
  float* dev_uniform = nullptr;
  if (!Expect(cudaMalloc(reinterpret_cast<void**>(&dev_uniform), kNumSamples * sizeof(float)) ==
                  cudaSuccess,
              "cudaMalloc dev_uniform failed")) {
    return false;
  }

  if (!Expect(curandGenerateUniform(gpu_gen, dev_uniform, kNumSamples) == CURAND_STATUS_SUCCESS,
              "curandGenerateUniform GPU failed")) {
    return false;
  }
  (void)cudaDeviceSynchronize();

  std::vector<float> host_uniform(kNumSamples, 0.0f);
  if (!Expect(cudaMemcpy(host_uniform.data(), dev_uniform, kNumSamples * sizeof(float),
                         cudaMemcpyDeviceToHost) == cudaSuccess,
              "cudaMemcpy uniform to host failed")) {
    return false;
  }

  double sum_u = 0.0;
  for (float val : host_uniform) {
    if (!Expect(val > 0.0f && val <= 1.0f, "GPU Uniform sample must be in (0, 1]")) {
      return false;
    }
    sum_u += static_cast<double>(val);
  }
  const double mean_u = sum_u / static_cast<double>(kNumSamples);
  if (!ExpectClose(static_cast<float>(mean_u), 0.5f, 0.015f, "GPU uniform mean should be ~0.5")) {
    return false;
  }
  (void)cudaFree(dev_uniform);

  // 2. GPU Normal float (mean = 0.0, stddev = 1.0)
  float* dev_normal = nullptr;
  if (!Expect(cudaMalloc(reinterpret_cast<void**>(&dev_normal), kNumSamples * sizeof(float)) ==
                  cudaSuccess,
              "cudaMalloc dev_normal failed")) {
    return false;
  }

  if (!Expect(curandGenerateNormal(gpu_gen, dev_normal, kNumSamples, 0.0f, 1.0f) ==
                  CURAND_STATUS_SUCCESS,
              "curandGenerateNormal GPU failed")) {
    return false;
  }
  (void)cudaDeviceSynchronize();

  std::vector<float> host_normal(kNumSamples, 0.0f);
  if (!Expect(cudaMemcpy(host_normal.data(), dev_normal, kNumSamples * sizeof(float),
                         cudaMemcpyDeviceToHost) == cudaSuccess,
              "cudaMemcpy normal to host failed")) {
    return false;
  }

  double sum_n = 0.0;
  double sum_n_sq = 0.0;
  for (float val : host_normal) {
    sum_n += static_cast<double>(val);
    sum_n_sq += static_cast<double>(val) * static_cast<double>(val);
  }
  const double mean_n = sum_n / static_cast<double>(kNumSamples);
  const double var_n = (sum_n_sq / static_cast<double>(kNumSamples)) - (mean_n * mean_n);

  if (!ExpectClose(static_cast<float>(mean_n), 0.0f, 0.03f, "GPU normal mean should be ~0.0")) {
    return false;
  }
  if (!ExpectClose(static_cast<float>(var_n), 1.0f, 0.05f, "GPU normal variance should be ~1.0")) {
    return false;
  }
  (void)cudaFree(dev_normal);

  // 3. GPU Uniform double
  double* dev_double = nullptr;
  if (!Expect(cudaMalloc(reinterpret_cast<void**>(&dev_double), kNumSamples * sizeof(double)) ==
                  cudaSuccess,
              "cudaMalloc dev_double failed")) {
    return false;
  }
  if (!Expect(
          curandGenerateUniformDouble(gpu_gen, dev_double, kNumSamples) == CURAND_STATUS_SUCCESS,
          "curandGenerateUniformDouble GPU failed")) {
    return false;
  }
  (void)cudaDeviceSynchronize();

  std::vector<double> host_double(kNumSamples, 0.0);
  (void)cudaMemcpy(host_double.data(), dev_double, kNumSamples * sizeof(double),
                   cudaMemcpyDeviceToHost);
  for (double val : host_double) {
    if (!Expect(val > 0.0 && val <= 1.0, "GPU Uniform double must be in (0, 1]")) {
      return false;
    }
  }
  (void)cudaFree(dev_double);

  (void)curandDestroyGenerator(gpu_gen);
  return true;
}

bool TestDeterminismAndOffset() {
  std::cout << "Testing cuRAND seed determinism and stream offset reproducibility...\n";

  curandGenerator_t gen1 = nullptr;
  curandGenerator_t gen2 = nullptr;
  (void)curandCreateGeneratorHost(&gen1, CURAND_RNG_PSEUDO_DEFAULT);
  (void)curandCreateGeneratorHost(&gen2, CURAND_RNG_PSEUDO_DEFAULT);

  (void)curandSetPseudoRandomGeneratorSeed(gen1, 777ull);
  (void)curandSetPseudoRandomGeneratorSeed(gen2, 777ull);

  constexpr std::size_t kN = 1024;
  std::vector<unsigned int> out1(kN, 0u);
  std::vector<unsigned int> out2(kN, 0u);

  (void)curandGenerate(gen1, out1.data(), kN);
  (void)curandGenerate(gen2, out2.data(), kN);

  for (std::size_t i = 0; i < kN; ++i) {
    if (!Expect(out1[i] == out2[i],
                "Identically seeded generators must produce identical outputs")) {
      return false;
    }
  }

  // Next batch from gen1 vs gen2 with explicit offset
  std::vector<unsigned int> next1(kN, 0u);
  (void)curandGenerate(gen1, next1.data(), kN);

  curandGenerator_t gen3 = nullptr;
  (void)curandCreateGeneratorHost(&gen3, CURAND_RNG_PSEUDO_DEFAULT);
  (void)curandSetPseudoRandomGeneratorSeed(gen3, 777ull);
  (void)curandSetGeneratorOffset(gen3, 1024ull);

  std::vector<unsigned int> next3(kN, 0u);
  (void)curandGenerate(gen3, next3.data(), kN);

  for (std::size_t i = 0; i < kN; ++i) {
    if (!Expect(next1[i] == next3[i], "Generator offset must yield identical future sequence")) {
      return false;
    }
  }

  (void)curandDestroyGenerator(gen1);
  (void)curandDestroyGenerator(gen2);
  (void)curandDestroyGenerator(gen3);
  return true;
}

bool TestQuasiRandomSobol() {
  std::cout << "Testing cuRAND Quasi-random Sobol direction vector queries...\n";

  curandDirectionVectors32_t* vectors = nullptr;
  if (!Expect(curandGetDirectionVectors32(&vectors, CURAND_DIRECTION_VECTORS_32_JOEKUO6) ==
                  CURAND_STATUS_SUCCESS,
              "curandGetDirectionVectors32 should succeed")) {
    return false;
  }
  if (!Expect(vectors != nullptr, "Sobol direction vectors must be non-null")) {
    return false;
  }
  if (!Expect((*vectors)[0] == 0x80000000u, "First Sobol direction vector component must match")) {
    return false;
  }

  const unsigned int* scramble = nullptr;
  if (!Expect(
          curandGetScrambleConstants32(&scramble) == CURAND_STATUS_SUCCESS && scramble != nullptr,
          "curandGetScrambleConstants32 should succeed")) {
    return false;
  }
  return true;
}

}  // namespace

int main() {
  std::cout << "=== Running MUDA cuRAND Verification Suite ===\n";

  if (!TestGeneratorLifecycle()) {
    return 1;
  }
  if (!TestHostGeneration()) {
    return 1;
  }
  if (!TestGpuGeneration()) {
    return 1;
  }
  if (!TestDeterminismAndOffset()) {
    return 1;
  }
  if (!TestQuasiRandomSobol()) {
    return 1;
  }

  std::cout << "=== All cuRAND Verification Tests Passed Successfully! ===\n";
  return 0;
}

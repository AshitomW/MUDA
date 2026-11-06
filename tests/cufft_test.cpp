/**
 * @file cufft_test.cpp
 * @brief Comprehensive verification test suite for the MUDA cuFFT shim (libcufft.dylib).
 */

#include "cufft.h"

#include <cmath>
#include <iostream>
#include <numbers>
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

bool Test1DC2CRoundTrip(int n) {
  std::cout << "Testing cuFFT 1D C2C forward + inverse round-trip (N=" << n << ")...\n";
  cufftHandle plan = 0;
  if (!Expect(cufftPlan1d(&plan, n, CUFFT_C2C, 1) == CUFFT_SUCCESS, "cufftPlan1d")) {
    return false;
  }

  std::vector<cufftComplex> h_in(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    const float phase =
        2.0f * std::numbers::pi_v<float> * static_cast<float>(i) / static_cast<float>(n);
    h_in[static_cast<std::size_t>(i)] = {std::cos(3.0f * phase) + 0.5f * std::sin(phase),
                                         0.25f * std::cos(2.0f * phase)};
  }

  cufftComplex* d_data = nullptr;
  cudaMalloc(reinterpret_cast<void**>(&d_data), h_in.size() * sizeof(cufftComplex));
  cudaMemcpy(d_data, h_in.data(), h_in.size() * sizeof(cufftComplex), cudaMemcpyHostToDevice);

  if (!Expect(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD) == CUFFT_SUCCESS,
              "cufftExecC2C FORWARD") ||
      !Expect(cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE) == CUFFT_SUCCESS,
              "cufftExecC2C INVERSE")) {
    return false;
  }

  std::vector<cufftComplex> h_out(static_cast<std::size_t>(n));
  cudaMemcpy(h_out.data(), d_data, h_out.size() * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

  const float scale = 1.0f / static_cast<float>(n);
  for (int i = 0; i < n; ++i) {
    const auto idx = static_cast<std::size_t>(i);
    if (!ExpectClose(h_out[idx].x * scale, h_in[idx].x, 1e-3f, "1D C2C round-trip real") ||
        !ExpectClose(h_out[idx].y * scale, h_in[idx].y, 1e-3f, "1D C2C round-trip imag")) {
      return false;
    }
  }

  cudaFree(d_data);
  return Expect(cufftDestroy(plan) == CUFFT_SUCCESS, "cufftDestroy");
}

bool Test1DR2CC2RRoundTrip() {
  std::cout << "Testing cuFFT 1D R2C + C2R round-trip (N=128)...\n";
  constexpr int kN = 128;
  constexpr int kComplexOut = kN / 2 + 1;

  cufftHandle plan_r2c = 0;
  cufftHandle plan_c2r = 0;
  if (!Expect(cufftPlan1d(&plan_r2c, kN, CUFFT_R2C, 1) == CUFFT_SUCCESS, "cufftPlan1d R2C") ||
      !Expect(cufftPlan1d(&plan_c2r, kN, CUFFT_C2R, 1) == CUFFT_SUCCESS, "cufftPlan1d C2R")) {
    return false;
  }

  std::vector<cufftReal> h_in(kN);
  for (int i = 0; i < kN; ++i) {
    const float t = static_cast<float>(i) / static_cast<float>(kN);
    h_in[static_cast<std::size_t>(i)] =
        1.5f + std::sin(2.0f * std::numbers::pi_v<float> * 4.0f * t) -
        0.5f * std::cos(2.0f * std::numbers::pi_v<float> * 7.0f * t);
  }

  cufftReal* d_real = nullptr;
  cufftComplex* d_freq = nullptr;
  cufftReal* d_reconstructed = nullptr;
  cudaMalloc(reinterpret_cast<void**>(&d_real), kN * sizeof(cufftReal));
  cudaMalloc(reinterpret_cast<void**>(&d_freq), kComplexOut * sizeof(cufftComplex));
  cudaMalloc(reinterpret_cast<void**>(&d_reconstructed), kN * sizeof(cufftReal));
  cudaMemcpy(d_real, h_in.data(), kN * sizeof(cufftReal), cudaMemcpyHostToDevice);

  if (!Expect(cufftExecR2C(plan_r2c, d_real, d_freq) == CUFFT_SUCCESS, "cufftExecR2C") ||
      !Expect(cufftExecC2R(plan_c2r, d_freq, d_reconstructed) == CUFFT_SUCCESS, "cufftExecC2R")) {
    return false;
  }

  std::vector<cufftReal> h_out(kN);
  cudaMemcpy(h_out.data(), d_reconstructed, kN * sizeof(cufftReal), cudaMemcpyDeviceToHost);
  for (int i = 0; i < kN; ++i) {
    const auto idx = static_cast<std::size_t>(i);
    if (!ExpectClose(h_out[idx] / static_cast<float>(kN), h_in[idx], 1e-3f,
                     "1D R2C->C2R round-trip")) {
      return false;
    }
  }

  cudaFree(d_real);
  cudaFree(d_freq);
  cudaFree(d_reconstructed);
  cufftDestroy(plan_r2c);
  cufftDestroy(plan_c2r);
  return true;
}

bool Test2DAnd3DC2C() {
  std::cout << "Testing cuFFT 2D (16x32) and 3D (8x8x8) C2C transforms...\n";
  // 2D impulse at (0, 0) -> flat spectrum of all 1s
  cufftHandle plan2d = 0;
  if (!Expect(cufftPlan2d(&plan2d, 16, 32, CUFFT_C2C) == CUFFT_SUCCESS, "cufftPlan2d")) {
    return false;
  }
  constexpr int kTotal2D = 16 * 32;
  std::vector<cufftComplex> h_2d(kTotal2D, {0.0f, 0.0f});
  h_2d[0] = {1.0f, 0.0f};

  cufftComplex* d_2d = nullptr;
  cudaMalloc(reinterpret_cast<void**>(&d_2d), kTotal2D * sizeof(cufftComplex));
  cudaMemcpy(d_2d, h_2d.data(), kTotal2D * sizeof(cufftComplex), cudaMemcpyHostToDevice);

  if (!Expect(cufftExecC2C(plan2d, d_2d, d_2d, CUFFT_FORWARD) == CUFFT_SUCCESS,
              "2D cufftExecC2C FORWARD")) {
    return false;
  }
  cudaMemcpy(h_2d.data(), d_2d, kTotal2D * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
  for (const auto& c : h_2d) {
    if (!ExpectClose(c.x, 1.0f, 1e-4f, "2D impulse real == 1") ||
        !ExpectClose(c.y, 0.0f, 1e-4f, "2D impulse imag == 0")) {
      return false;
    }
  }
  cudaFree(d_2d);
  cufftDestroy(plan2d);

  // 3D C2C round-trip (8x8x8)
  cufftHandle plan3d = 0;
  if (!Expect(cufftPlan3d(&plan3d, 8, 8, 8, CUFFT_C2C) == CUFFT_SUCCESS, "cufftPlan3d")) {
    return false;
  }
  constexpr int kTotal3D = 8 * 8 * 8;
  std::vector<cufftComplex> h_3d(kTotal3D);
  for (int i = 0; i < kTotal3D; ++i) {
    h_3d[static_cast<std::size_t>(i)] = {static_cast<float>((i % 7) - 3) * 0.2f,
                                         static_cast<float>((i % 5) - 2) * 0.1f};
  }
  cufftComplex* d_3d = nullptr;
  cudaMalloc(reinterpret_cast<void**>(&d_3d), kTotal3D * sizeof(cufftComplex));
  cudaMemcpy(d_3d, h_3d.data(), kTotal3D * sizeof(cufftComplex), cudaMemcpyHostToDevice);

  if (!Expect(cufftExecC2C(plan3d, d_3d, d_3d, CUFFT_FORWARD) == CUFFT_SUCCESS,
              "3D cufftExecC2C FORWARD") ||
      !Expect(cufftExecC2C(plan3d, d_3d, d_3d, CUFFT_INVERSE) == CUFFT_SUCCESS,
              "3D cufftExecC2C INVERSE")) {
    return false;
  }

  std::vector<cufftComplex> h_3d_out(kTotal3D);
  cudaMemcpy(h_3d_out.data(), d_3d, kTotal3D * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
  for (int i = 0; i < kTotal3D; ++i) {
    const auto idx = static_cast<std::size_t>(i);
    if (!ExpectClose(h_3d_out[idx].x / static_cast<float>(kTotal3D), h_3d[idx].x, 1e-3f,
                     "3D C2C round-trip real") ||
        !ExpectClose(h_3d_out[idx].y / static_cast<float>(kTotal3D), h_3d[idx].y, 1e-3f,
                     "3D C2C round-trip imag")) {
      return false;
    }
  }

  cudaFree(d_3d);
  cufftDestroy(plan3d);
  return true;
}

bool TestPlanManyBatched() {
  std::cout << "Testing cuFFT cufftPlanMany batched 1D C2C...\n";
  int n[1] = {64};
  cufftHandle plan = 0;
  constexpr int kBatch = 4;
  if (!Expect(cufftPlanMany(&plan, 1, n, nullptr, 1, 64, nullptr, 1, 64, CUFFT_C2C, kBatch) ==
                  CUFFT_SUCCESS,
              "cufftPlanMany")) {
    return false;
  }

  std::size_t work_size = 0;
  if (!Expect(cufftGetSize(plan, &work_size) == CUFFT_SUCCESS && work_size > 0, "cufftGetSize")) {
    return false;
  }

  std::vector<cufftComplex> h_data(static_cast<std::size_t>(64 * kBatch), {0.0f, 0.0f});
  for (int b = 0; b < kBatch; ++b) {
    h_data[static_cast<std::size_t>(b * 64)] = {static_cast<float>(b + 1), 0.0f};
  }

  cufftComplex* d_data = nullptr;
  cudaMalloc(reinterpret_cast<void**>(&d_data), h_data.size() * sizeof(cufftComplex));
  cudaMemcpy(d_data, h_data.data(), h_data.size() * sizeof(cufftComplex), cudaMemcpyHostToDevice);

  if (!Expect(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD) == CUFFT_SUCCESS,
              "cufftPlanMany ExecC2C")) {
    return false;
  }

  cudaMemcpy(h_data.data(), d_data, h_data.size() * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
  for (int b = 0; b < kBatch; ++b) {
    for (int i = 0; i < 64; ++i) {
      if (!ExpectClose(h_data[static_cast<std::size_t>(b * 64 + i)].x, static_cast<float>(b + 1),
                       1e-4f, "Batched impulse amplitude")) {
        return false;
      }
    }
  }

  cudaFree(d_data);
  cufftDestroy(plan);
  return true;
}

}  // namespace

int main() {
  // Power-of-two (256 -> Stockham Radix-2) and non-power-of-two (60 -> General Radix)
  if (!Test1DC2CRoundTrip(256) || !Test1DC2CRoundTrip(60) || !Test1DR2CC2RRoundTrip() ||
      !Test2DAnd3DC2C() || !TestPlanManyBatched()) {
    return 1;
  }
  std::cout << "ALL CUFFT TESTS PASSED!\n";
  return 0;
}


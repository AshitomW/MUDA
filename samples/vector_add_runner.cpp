/**
 * @file vector_add_runner.cpp
 * @brief Runs the translated vector-add kernel and verifies results.
 */

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <span>
#include <vector>

#include "muda/runtime/buffer.hpp"
#include "muda/runtime/device.hpp"
#include "muda/runtime/kernel.hpp"

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: muda_vector_add <vector_add.metallib>\n");
    return 2;
  }

  constexpr std::size_t kCount = 1024;
  constexpr std::size_t kThreads = 256;
  constexpr std::size_t kBytes = kCount * sizeof(float);
  constexpr float kExpectedSum = 3.0f;
  constexpr float kTolerance = 1e-5f;

  std::vector<float> a(kCount, 1.0f);
  std::vector<float> b(kCount, 2.0f);
  std::vector<float> out(kCount, 0.0f);

  try {
    const muda::runtime::Device device = muda::runtime::Device::Default();
    muda::runtime::Buffer buffer_a = device.Allocate(kBytes);
    muda::runtime::Buffer buffer_b = device.Allocate(kBytes);
    muda::runtime::Buffer buffer_out = device.Allocate(kBytes);

    buffer_a.Write(a.data(), kBytes);
    buffer_b.Write(b.data(), kBytes);

    const muda::runtime::Kernel kernel = muda::runtime::Kernel::Load(device, argv[1], "vector_add");

    const int count = static_cast<int>(kCount);
    const std::vector<muda::runtime::KernelArgument> arguments = {
        muda::runtime::KernelArgument::FromBuffer(0, buffer_a),
        muda::runtime::KernelArgument::FromBuffer(1, buffer_b),
        muda::runtime::KernelArgument::FromBuffer(2, buffer_out),
        muda::runtime::KernelArgument::FromBytes(3, &count, sizeof(count)),
    };

    kernel.Dispatch(std::span<const muda::runtime::KernelArgument>(arguments), {kCount, kThreads});
    buffer_out.Read(out.data(), kBytes);
  } catch (const std::exception& error) {
    std::fprintf(stderr, "muda_vector_add: %s\n", error.what());
    return 1;
  }

  for (std::size_t index = 0; index < kCount; ++index) {
    if (std::fabs(out[index] - kExpectedSum) > kTolerance) {
      std::fprintf(stderr, "mismatch at %zu: %.1f\n", index, static_cast<double>(out[index]));
      return 1;
    }
  }

  std::printf("ok: %zu elements, out[0]=%.1f, out[%zu]=%.1f\n", kCount, static_cast<double>(out[0]),
              kCount - 1, static_cast<double>(out[kCount - 1]));
  return 0;
}

/**
 * @file vector_add_test.cpp
 * @brief Tests the end-to-end vector-add prototype.
 */

#include <cstddef>
#include <exception>
#include <filesystem>
#include <iostream>
#include <span>
#include <string>
#include <vector>

#include "muda/compiler/driver.hpp"
#include "muda/compiler/toolchain.hpp"
#include "muda/runtime/buffer.hpp"
#include "muda/runtime/device.hpp"
#include "muda/runtime/kernel.hpp"
#include "test_utils.hpp"

int main() {
  if (!muda::compiler::MetalToolchainAvailable()) {
    std::cout << "skip: Metal toolchain unavailable\n";
    return 77;
  }

  const muda::testing::ScopedTempDir temp_dir("muda_vector_add_test");
  if (!temp_dir.valid()) {
    std::cerr << "cannot create temp directory\n";
    return 1;
  }

  try {
    muda::compiler::CompileOptions options;
    options.input = std::filesystem::path(MUDA_SOURCE_DIR) / "samples/vector_add.cu";
    options.output = temp_dir.path() / "vector_add.metallib";

    const muda::compiler::CompileResult compiled = muda::compiler::Compile(options);
    if (compiled.status != muda::compiler::Status::kOk) {
      std::cerr << "compile failed: " << compiled.message() << "\n";
      return 1;
    }

    constexpr std::size_t kCount = 1024;
    constexpr std::size_t kThreads = 256;
    constexpr std::size_t kBytes = kCount * sizeof(float);

    const muda::runtime::Device device = muda::runtime::Device::Default();
    muda::runtime::Buffer buffer_a = device.Allocate(kBytes);
    muda::runtime::Buffer buffer_b = device.Allocate(kBytes);
    muda::runtime::Buffer buffer_out = device.Allocate(kBytes);

    std::vector<float> a(kCount, 1.0f);
    std::vector<float> b(kCount, 2.0f);
    std::vector<float> out(kCount, 0.0f);
    buffer_a.Write(a.data(), kBytes);
    buffer_b.Write(b.data(), kBytes);

    const muda::runtime::Kernel kernel =
        muda::runtime::Kernel::Load(device, compiled.metallib_path, compiled.kernel_name);

    const int count = static_cast<int>(kCount);
    const std::vector<muda::runtime::KernelArgument> arguments = {
        muda::runtime::KernelArgument::FromBuffer(0, buffer_a),
        muda::runtime::KernelArgument::FromBuffer(1, buffer_b),
        muda::runtime::KernelArgument::FromBuffer(2, buffer_out),
        muda::runtime::KernelArgument::FromBytes(3, &count, sizeof(count)),
    };

    kernel.Dispatch(std::span<const muda::runtime::KernelArgument>(arguments), {kCount, kThreads});
    buffer_out.Read(out.data(), kBytes);

    for (std::size_t index = 0; index < kCount; ++index) {
      if (!muda::testing::AreClose(out[index], 3.0f)) {
        std::cerr << "mismatch at " << index << ": " << out[index] << "\n";
        return 1;
      }
    }
  } catch (const std::exception& exception) {
    std::cerr << "runtime failed: " << exception.what() << "\n";
    return 1;
  }

  std::cout << "executed vector_add\n";
  return 0;
}

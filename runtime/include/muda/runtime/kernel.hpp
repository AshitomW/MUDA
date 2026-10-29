/**
 * @file kernel.hpp
 * @brief Loads Metal compute pipelines and dispatches kernel arguments.
 */

#pragma once

#include <cstddef>
#include <filesystem>
#include <memory>
#include <span>
#include <string>

#include "muda/runtime/buffer.hpp"

namespace muda::runtime {

class Device;
struct KernelStorage;

struct KernelArgument {
  std::size_t index = 0;
  const Buffer* buffer = nullptr;
  const void* bytes = nullptr;
  std::size_t size = 0;

  [[nodiscard]] static constexpr KernelArgument FromBuffer(std::size_t index,
                                                           const Buffer& buffer) noexcept;
  [[nodiscard]] static constexpr KernelArgument FromBytes(std::size_t index, const void* data,
                                                          std::size_t size) noexcept;
};

struct DispatchSize {
  std::size_t threads = 1;
  std::size_t threads_per_threadgroup = 1;
};

class Kernel {
 public:
  Kernel();
  ~Kernel();

  Kernel(Kernel&&) noexcept;
  Kernel& operator=(Kernel&&) noexcept;
  Kernel(const Kernel&) = delete;
  Kernel& operator=(const Kernel&) = delete;

  [[nodiscard]] static Kernel Load(const Device& device, const std::filesystem::path& metallib,
                                   std::string name);

  void Dispatch(std::span<const KernelArgument> arguments, DispatchSize size) const;

 private:
  explicit Kernel(std::unique_ptr<KernelStorage> storage);

  std::unique_ptr<KernelStorage> storage_;
};

constexpr KernelArgument KernelArgument::FromBuffer(std::size_t index,
                                                    const Buffer& buffer) noexcept {
  return {index, &buffer, nullptr, 0};
}

constexpr KernelArgument KernelArgument::FromBytes(std::size_t index, const void* data,
                                                   std::size_t size) noexcept {
  return {index, nullptr, data, size};
}

}  // namespace muda::runtime

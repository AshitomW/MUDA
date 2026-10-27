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

  [[nodiscard]] static KernelArgument FromBuffer(std::size_t index, const Buffer& buffer);
  [[nodiscard]] static KernelArgument FromBytes(std::size_t index, const void* data,
                                                std::size_t size);
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
  explicit Kernel(std::shared_ptr<KernelStorage> storage);

  std::shared_ptr<KernelStorage> storage_;
};

inline KernelArgument KernelArgument::FromBuffer(std::size_t index, const Buffer& buffer) {
  return {index, &buffer, nullptr, 0};
}

inline KernelArgument KernelArgument::FromBytes(std::size_t index, const void* data,
                                                std::size_t size) {
  return {index, nullptr, data, size};
}

}

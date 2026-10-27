/**
 * @file buffer.hpp
 * @brief Owns a Metal buffer and provides host read and write access.
 */

#pragma once

#include <cstddef>
#include <memory>

namespace muda::runtime {

struct BufferStorage;
class Device;
class Kernel;

class Buffer {
 public:
  Buffer();
  ~Buffer();

  Buffer(Buffer&&) noexcept;
  Buffer& operator=(Buffer&&) noexcept;
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;

  [[nodiscard]] bool valid() const;
  [[nodiscard]] std::size_t size() const;

  void Write(const void* data, std::size_t size);
  void Read(void* data, std::size_t size) const;

 private:
  explicit Buffer(std::shared_ptr<BufferStorage> storage);

  std::shared_ptr<BufferStorage> storage_;
  friend class Device;
  friend class Kernel;
};

}

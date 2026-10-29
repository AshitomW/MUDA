/**
 * @file buffer.mm
 * @brief Implements Metal buffer access and validation.
 */

#include "muda/runtime/buffer.hpp"

#include <cstring>
#include <stdexcept>
#include <utility>

#include "storage.hpp"

namespace muda::runtime {

Buffer::Buffer() = default;
Buffer::~Buffer() = default;
Buffer::Buffer(Buffer&&) noexcept = default;
Buffer& Buffer::operator=(Buffer&&) noexcept = default;

Buffer::Buffer(std::unique_ptr<BufferStorage> storage) : storage_(std::move(storage)) {}

bool Buffer::valid() const noexcept { return storage_ != nullptr && storage_->buffer != nil; }

std::size_t Buffer::size() const noexcept { return storage_ != nullptr ? storage_->size : 0; }

void Buffer::Write(const void* data, std::size_t size) {
  if (!valid()) {
    throw std::runtime_error("invalid Metal buffer");
  }
  if (size > storage_->size) {
    throw std::runtime_error("buffer write exceeds allocation");
  }
  if (size > 0 && data == nullptr) {
    throw std::runtime_error("buffer write source is null");
  }
  if (size > 0) {
    std::memcpy(storage_->buffer.contents, data, size);
  }
}

void Buffer::Read(void* data, std::size_t size) const {
  if (!valid()) {
    throw std::runtime_error("invalid Metal buffer");
  }
  if (size > storage_->size) {
    throw std::runtime_error("buffer read exceeds allocation");
  }
  if (size > 0 && data == nullptr) {
    throw std::runtime_error("buffer read destination is null");
  }
  if (size > 0) {
    std::memcpy(data, storage_->buffer.contents, size);
  }
}

}  // namespace muda::runtime

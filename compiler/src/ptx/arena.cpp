/**
 * @file arena.cpp
 * @brief Implements monotonic bump-pointer arena allocator.
 */

#include "muda/compiler/ptx/arena.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

namespace muda::compiler::ptx {

Arena::Arena(std::size_t chunk_size) noexcept
    : chunk_size_(std::max<std::size_t>(chunk_size, 4096)) {}

Arena::~Arena() = default;

Arena::Arena(Arena&& other) noexcept
    : chunk_size_(other.chunk_size_),
      current_chunk_index_(other.current_chunk_index_),
      total_allocated_bytes_(other.total_allocated_bytes_),
      chunks_(std::move(other.chunks_)) {
  other.chunk_size_ = kDefaultChunkSize;
  other.current_chunk_index_ = 0;
  other.total_allocated_bytes_ = 0;
}

Arena& Arena::operator=(Arena&& other) noexcept {
  if (this != &other) {
    chunk_size_ = other.chunk_size_;
    current_chunk_index_ = other.current_chunk_index_;
    total_allocated_bytes_ = other.total_allocated_bytes_;
    chunks_ = std::move(other.chunks_);

    other.chunk_size_ = kDefaultChunkSize;
    other.current_chunk_index_ = 0;
    other.total_allocated_bytes_ = 0;
  }
  return *this;
}

void* Arena::Allocate(std::size_t bytes, std::size_t alignment) {
  if (bytes == 0) {
    return nullptr;
  }

  assert((alignment & (alignment - 1)) == 0 && "Alignment must be a power of two");
  const std::size_t effective_align = std::max<std::size_t>(alignment, alignof(std::byte));

  if (chunks_.empty() || current_chunk_index_ >= chunks_.size()) {
    Grow(bytes + effective_align);
  }

  Chunk* chunk = &chunks_[current_chunk_index_];
  const auto current_addr = reinterpret_cast<std::uintptr_t>(chunk->data.get() + chunk->offset);
  const std::size_t padding =
      (effective_align - (current_addr % effective_align)) % effective_align;

  if (chunk->offset + padding + bytes <= chunk->capacity) {
    chunk->offset += padding;
    void* result = chunk->data.get() + chunk->offset;
    chunk->offset += bytes;
    total_allocated_bytes_ += bytes + padding;
    return result;
  }

  // Current chunk doesn't fit, grow to a new chunk
  Grow(bytes + effective_align);
  chunk = &chunks_[current_chunk_index_];
  const auto new_addr = reinterpret_cast<std::uintptr_t>(chunk->data.get() + chunk->offset);
  const std::size_t new_padding =
      (effective_align - (new_addr % effective_align)) % effective_align;

  chunk->offset += new_padding;
  void* result = chunk->data.get() + chunk->offset;
  chunk->offset += bytes;
  total_allocated_bytes_ += bytes + new_padding;
  return result;
}

void Arena::Grow(std::size_t min_bytes) {
  // If we already have preallocated chunks from a previous Reset(), reuse them
  if (current_chunk_index_ + 1 < chunks_.size() && chunks_[current_chunk_index_].offset > 0) {
    ++current_chunk_index_;
    if (chunks_[current_chunk_index_].capacity >= min_bytes) {
      chunks_[current_chunk_index_].offset = 0;
      return;
    }
  }

  const std::size_t new_capacity = std::max(chunk_size_, min_bytes);
  Chunk new_chunk;
  new_chunk.data = std::make_unique<std::byte[]>(new_capacity);
  new_chunk.capacity = new_capacity;
  new_chunk.offset = 0;

  chunks_.push_back(std::move(new_chunk));
  current_chunk_index_ = chunks_.size() - 1;
}

void Arena::Reset() noexcept {
  for (Chunk& chunk : chunks_) {
    chunk.offset = 0;
  }
  current_chunk_index_ = 0;
  total_allocated_bytes_ = 0;
}

}  // namespace muda::compiler::ptx

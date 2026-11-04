/**
 * @file arena.hpp
 * @brief High-speed monotonic bump-pointer arena allocator for PTX AST and translation data.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <utility>
#include <vector>

namespace muda::compiler::ptx {

/**
 * @brief Fast monotonic arena allocator for AST nodes, tokens, and temporary compile objects.
 *
 * Allocates contiguous blocks of memory in chunks and advances a bump pointer.
 * Objects are destroyed and memory is freed en masse when the Arena is destroyed or Reset().
 */
class Arena final {
 public:
  static constexpr std::size_t kDefaultChunkSize = 64 * 1024;  // 64 KiB chunks

  explicit Arena(std::size_t chunk_size = kDefaultChunkSize) noexcept;
  ~Arena();

  Arena(const Arena&) = delete;
  Arena& operator=(const Arena&) = delete;

  Arena(Arena&& other) noexcept;
  Arena& operator=(Arena&& other) noexcept;

  /**
   * @brief Allocates raw bytes with the specified alignment from the arena.
   */
  [[nodiscard]] void* Allocate(std::size_t bytes,
                               std::size_t alignment = alignof(std::max_align_t));

  /**
   * @brief Constructs an object of type T within the arena.
   */
  template <typename T, typename... Args>
  [[nodiscard]] T* Create(Args&&... args) {
    void* memory = Allocate(sizeof(T), alignof(T));
    return ::new (memory) T(std::forward<Args>(args)...);
  }

  /**
   * @brief Allocates an array of uninitialized POD / trivial objects of type T.
   */
  template <typename T>
  [[nodiscard]] T* AllocateArray(std::size_t count) {
    return static_cast<T*>(Allocate(sizeof(T) * count, alignof(T)));
  }

  /**
   * @brief Resets the bump pointer, reusing allocated memory chunks without deallocation.
   */
  void Reset() noexcept;

  /**
   * @brief Total number of bytes allocated across all chunks.
   */
  [[nodiscard]] std::size_t TotalAllocatedBytes() const noexcept { return total_allocated_bytes_; }

  /**
   * @brief Total number of chunks currently held.
   */
  [[nodiscard]] std::size_t ChunkCount() const noexcept { return chunks_.size(); }

 private:
  struct Chunk {
    std::unique_ptr<std::byte[]> data;
    std::size_t capacity = 0;
    std::size_t offset = 0;
  };

  void Grow(std::size_t min_bytes);

  std::size_t chunk_size_;
  std::size_t current_chunk_index_ = 0;
  std::size_t total_allocated_bytes_ = 0;
  std::vector<Chunk> chunks_;
};

}  // namespace muda::compiler::ptx

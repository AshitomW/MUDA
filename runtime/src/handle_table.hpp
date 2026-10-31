/**
 * @file handle_table.hpp
 * @brief Validated generation- and type-tagged opaque handle table for CUDA resources.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <shared_mutex>
#include <vector>

namespace muda::runtime::cudart {

enum class HandleKind : std::uint8_t {
  kStream = 1,
  kEvent = 2,
};

/**
 * @brief Encodes and decodes 64-bit validated resource tokens cast to opaque C handles.
 *
 * Layout:
 *   bits [0..19]  : 20-bit slot index (up to 1,048,576 slots)
 *   bits [20..27] : 8-bit HandleKind tag
 *   bits [28..63] : 36-bit non-zero generation counter
 */
struct HandleToken {
  std::uint32_t index = 0;
  HandleKind kind = HandleKind::kStream;
  std::uint64_t generation = 0;

  static constexpr std::uint64_t kIndexBits = 20;
  static constexpr std::uint64_t kKindBits = 8;
  static constexpr std::uint64_t kIndexMask = (1ull << kIndexBits) - 1ull;
  static constexpr std::uint64_t kKindMask = (1ull << kKindBits) - 1ull;

  [[nodiscard]] constexpr std::uintptr_t Encode() const noexcept {
    return static_cast<std::uintptr_t>((generation << (kIndexBits + kKindBits)) |
                                       (static_cast<std::uint64_t>(kind) << kIndexBits) |
                                       (static_cast<std::uint64_t>(index) & kIndexMask));
  }

  template <typename HandleT>
  [[nodiscard]] HandleT ToOpaqueHandle() const noexcept {
    return reinterpret_cast<HandleT>(Encode());
  }

  template <typename HandleT>
  [[nodiscard]] static HandleToken Decode(HandleT raw_handle) noexcept {
    const auto raw = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(raw_handle));
    HandleToken token;
    token.index = static_cast<std::uint32_t>(raw & kIndexMask);
    token.kind = static_cast<HandleKind>((raw >> kIndexBits) & kKindMask);
    token.generation = raw >> (kIndexBits + kKindBits);
    return token;
  }
};

template <typename T, HandleKind KindTag>
class HandleTable {
 public:
  HandleTable() = default;

  template <typename HandleT>
  [[nodiscard]] HandleT Insert(std::shared_ptr<T> resource) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    std::uint32_t slot_index = 0;
    if (!free_list_.empty()) {
      slot_index = free_list_.back();
      free_list_.pop_back();
      Slot& slot = slots_[slot_index];
      slot.resource = std::move(resource);
      return HandleToken{slot_index, KindTag, slot.generation}.template ToOpaqueHandle<HandleT>();
    }

    slot_index = static_cast<std::uint32_t>(slots_.size());
    slots_.push_back(Slot{
        .generation = 1,
        .resource = std::move(resource),
    });
    return HandleToken{slot_index, KindTag, 1}.template ToOpaqueHandle<HandleT>();
  }

  template <typename HandleT>
  [[nodiscard]] std::shared_ptr<T> Lookup(HandleT raw_handle) const noexcept {
    if (raw_handle == nullptr) {
      return nullptr;
    }
    const HandleToken token = HandleToken::Decode(raw_handle);
    if (token.kind != KindTag || token.generation == 0) {
      return nullptr;
    }

    std::shared_lock<std::shared_mutex> lock(mutex_);
    if (token.index >= slots_.size()) {
      return nullptr;
    }
    const Slot& slot = slots_[token.index];
    if (slot.generation != token.generation || slot.resource == nullptr) {
      return nullptr;
    }
    return slot.resource;
  }

  template <typename HandleT>
  [[nodiscard]] std::shared_ptr<T> Remove(HandleT raw_handle) {
    if (raw_handle == nullptr) {
      return nullptr;
    }
    const HandleToken token = HandleToken::Decode(raw_handle);
    if (token.kind != KindTag || token.generation == 0) {
      return nullptr;
    }

    std::unique_lock<std::shared_mutex> lock(mutex_);
    if (token.index >= slots_.size()) {
      return nullptr;
    }
    Slot& slot = slots_[token.index];
    if (slot.generation != token.generation || slot.resource == nullptr) {
      return nullptr;
    }
    std::shared_ptr<T> removed = std::move(slot.resource);
    ++slot.generation;
    if (slot.generation == 0) {
      slot.generation = 1;
    }
    free_list_.push_back(token.index);
    return removed;
  }

  [[nodiscard]] std::vector<std::shared_ptr<T>> SnapshotAll() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    std::vector<std::shared_ptr<T>> active;
    active.reserve(slots_.size());
    for (const Slot& slot : slots_) {
      if (slot.resource != nullptr) {
        active.push_back(slot.resource);
      }
    }
    return active;
  }

  [[nodiscard]] std::vector<std::shared_ptr<T>> DrainAll() {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    std::vector<std::shared_ptr<T>> drained;
    drained.reserve(slots_.size());
    for (std::uint32_t index = 0; index < slots_.size(); ++index) {
      Slot& slot = slots_[index];
      if (slot.resource != nullptr) {
        drained.push_back(std::move(slot.resource));
        ++slot.generation;
        if (slot.generation == 0) {
          slot.generation = 1;
        }
        free_list_.push_back(index);
      }
    }
    return drained;
  }

 private:
  struct Slot {
    std::uint64_t generation = 1;
    std::shared_ptr<T> resource;
  };

  mutable std::shared_mutex mutex_;
  std::vector<Slot> slots_;
  std::vector<std::uint32_t> free_list_;
};

}  // namespace muda::runtime::cudart

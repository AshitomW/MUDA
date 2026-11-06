/**
 * @file plan_cache.hpp
 * @brief Thread-safe bounded LRU plan cache and deterministic hash combiners for MUDA libraries.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <list>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <utility>

namespace muda::libraries {

/**
 * @brief Deterministic 64-bit FNV-1a / Boost-style hash combiner.
 */
template <typename T>
constexpr void HashCombine(std::size_t& seed, const T& value) noexcept {
  const std::size_t hashed = std::hash<T>{}(value);
  seed ^= hashed + 0x9e3779b97f4a7c15ull + (seed << 6u) + (seed >> 2u);
}

/**
 * @brief Thread-safe bounded Least-Recently-Used (LRU) cache for compiled library plans.
 */
template <typename Key, typename Value, typename Hash = std::hash<Key>>
class LruPlanCache {
 public:
  static constexpr std::size_t kDefaultCapacity = 256;

  explicit LruPlanCache(std::size_t capacity = kDefaultCapacity) noexcept
      : capacity_(capacity > 0 ? capacity : 1) {}

  ~LruPlanCache() = default;
  LruPlanCache(const LruPlanCache&) = delete;
  LruPlanCache& operator=(const LruPlanCache&) = delete;

  [[nodiscard]] std::optional<Value> Lookup(const Key& key) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = index_.find(key);
    if (it == index_.end()) {
      return std::nullopt;
    }
    items_.splice(items_.begin(), items_, it->second);
    return it->second->second;
  }

  void Insert(const Key& key, Value value) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = index_.find(key);
    if (it != index_.end()) {
      it->second->second = std::move(value);
      items_.splice(items_.begin(), items_, it->second);
      return;
    }
    if (items_.size() >= capacity_) {
      const Key& evicted_key = items_.back().first;
      index_.erase(evicted_key);
      items_.pop_back();
    }
    items_.emplace_front(key, std::move(value));
    index_[items_.front().first] = items_.begin();
  }

  void Clear() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    index_.clear();
    items_.clear();
  }

  [[nodiscard]] std::size_t Size() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return items_.size();
  }

 private:
  using ListType = std::list<std::pair<Key, Value>>;

  std::size_t capacity_;
  mutable std::mutex mutex_;
  ListType items_;
  std::unordered_map<Key, typename ListType::iterator, Hash> index_;
};

}  // namespace muda::libraries


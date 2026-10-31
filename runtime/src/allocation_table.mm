/**
 * @file allocation_table.mm
 * @brief Tracks Metal buffer allocations and validates device pointer ranges.
 */

#include "allocation_table.hpp"

#include <array>
#include <utility>

namespace muda::runtime::cudart {
namespace {

struct MruEntry {
  std::uint64_t epoch = 0;
  DeviceAddress base{0};
  std::size_t capacity = 0;
  const AllocationRecord* record = nullptr;
};

constexpr std::size_t kMruSize = 4;
thread_local std::array<MruEntry, kMruSize> tl_mru{};
thread_local std::size_t tl_mru_cursor = 0;

}  // namespace

AllocationTable::AllocationTable(AllocationTable&& other) noexcept
    : records_(std::move(other.records_)), epoch_(other.epoch_.load(std::memory_order_relaxed)) {
  other.epoch_.fetch_add(1, std::memory_order_relaxed);
}

AllocationTable& AllocationTable::operator=(AllocationTable&& other) noexcept {
  if (this != &other) {
    records_ = std::move(other.records_);
    epoch_.store(other.epoch_.load(std::memory_order_relaxed), std::memory_order_relaxed);
    other.epoch_.fetch_add(1, std::memory_order_relaxed);
  }
  return *this;
}

void AllocationTable::Insert(AllocationRecord record) {
  const DeviceAddress key = record.base_address;
  records_[key] = std::move(record);
  epoch_.fetch_add(1, std::memory_order_release);
}

std::optional<AllocationRecord> AllocationTable::Remove(DeviceAddress base_address) {
  auto it = records_.find(base_address);
  if (it == records_.end()) {
    return std::nullopt;
  }
  AllocationRecord removed = std::move(it->second);
  records_.erase(it);
  epoch_.fetch_add(1, std::memory_order_release);
  return removed;
}

const AllocationRecord* AllocationTable::FindContaining(DeviceAddress address) const noexcept {
  const std::uint64_t cur_epoch = epoch_.load(std::memory_order_acquire);
  for (const MruEntry& entry : tl_mru) {
    if (entry.epoch == cur_epoch && entry.record != nullptr && address.value >= entry.base.value &&
        (address.value - entry.base.value) < entry.capacity) {
      return entry.record;
    }
  }

  if (records_.empty()) {
    return nullptr;
  }
  auto it = records_.upper_bound(address);
  if (it == records_.begin()) {
    return nullptr;
  }
  --it;
  const AllocationRecord& record = it->second;
  if (address.value >= record.base_address.value &&
      (address.value - record.base_address.value) < record.capacity) {
    tl_mru[tl_mru_cursor].epoch = cur_epoch;
    tl_mru[tl_mru_cursor].base = record.base_address;
    tl_mru[tl_mru_cursor].capacity = record.capacity;
    tl_mru[tl_mru_cursor].record = &record;
    tl_mru_cursor = (tl_mru_cursor + 1) % kMruSize;
    return &record;
  }
  return nullptr;
}

bool AllocationTable::IsRangeValid(const AllocationRecord* record, DeviceAddress address,
                                   std::size_t count) noexcept {
  if (record == nullptr || address.value < record->base_address.value) {
    return false;
  }
  const std::size_t offset = static_cast<std::size_t>(address.value - record->base_address.value);
  if (offset > record->size) {
    return false;
  }
  if (offset == record->size && count == 0) {
    return true;
  }
  return count <= (record->size - offset);
}

bool AllocationTable::IsOffsetValid(const AllocationRecord* record,
                                    DeviceAddress address) noexcept {
  if (record == nullptr || address.value < record->base_address.value) {
    return false;
  }
  return (address.value - record->base_address.value) < record->size;
}

std::size_t AllocationTable::TotalAllocatedBytes() const noexcept {
  std::size_t total = 0;
  for (const auto& [addr, record] : records_) {
    total += record.capacity;
  }
  return total;
}

void AllocationTable::Clear() {
  records_.clear();
  epoch_.fetch_add(1, std::memory_order_release);
}

}  // namespace muda::runtime::cudart

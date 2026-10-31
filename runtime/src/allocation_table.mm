/**
 * @file allocation_table.mm
 * @brief Tracks Metal buffer allocations and validates device pointer ranges.
 */

#include "allocation_table.hpp"

#include <utility>

namespace muda::runtime::cudart {

void AllocationTable::Insert(AllocationRecord record) {
  const DeviceAddress key = record.base_address;
  records_[key] = std::move(record);
}

std::optional<AllocationRecord> AllocationTable::Remove(DeviceAddress base_address) {
  auto it = records_.find(base_address);
  if (it == records_.end()) {
    return std::nullopt;
  }
  AllocationRecord removed = std::move(it->second);
  records_.erase(it);
  return removed;
}

const AllocationRecord* AllocationTable::FindContaining(DeviceAddress address) const noexcept {
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

void AllocationTable::Clear() { records_.clear(); }

}  // namespace muda::runtime::cudart

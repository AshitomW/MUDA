/**
 * @file fatbin.cpp
 * @brief Zero-overhead parser and PTX/CUBIN extractor for NVIDIA Fatbin containers.
 */

#include "muda/compiler/cubin/fatbin.hpp"

#include <algorithm>
#include <cstring>
#include <string_view>

namespace muda::compiler::cubin {
namespace {

struct FatbinHeader64 {
  std::uint32_t magic;
  std::uint16_t version;
  std::uint16_t header_size;
  std::uint64_t total_size;
};

struct FatbinHeader32 {
  std::uint32_t magic;
  std::uint16_t version;
  std::uint16_t header_size;
  std::uint32_t total_size;
};

[[nodiscard]] std::uint32_t ReadU32(const std::uint8_t* ptr) noexcept {
  std::uint32_t val = 0;
  std::memcpy(&val, ptr, sizeof(val));
  return val;
}

[[nodiscard]] std::uint16_t ReadU16(const std::uint8_t* ptr) noexcept {
  std::uint16_t val = 0;
  std::memcpy(&val, ptr, sizeof(val));
  return val;
}

[[nodiscard]] std::uint64_t ReadU64(const std::uint8_t* ptr) noexcept {
  std::uint64_t val = 0;
  std::memcpy(&val, ptr, sizeof(val));
  return val;
}

}  // namespace

bool FatbinParser::IsFatbinary(std::span<const std::uint8_t> data) noexcept {
  if (data.size() < 8) {
    return false;
  }
  const std::uint32_t magic = ReadU32(data.data());
  return magic == kFatbinMagic64 || magic == kFatbinMagic32 || magic == kFatbinMagicText;
}

std::vector<FatbinPayload> FatbinParser::Parse(std::span<const std::uint8_t> data) {
  std::vector<FatbinPayload> payloads;
  if (!IsFatbinary(data)) {
    return payloads;
  }

  const std::uint32_t magic = ReadU32(data.data());
  std::size_t offset = 0;

  if (magic == kFatbinMagic64) {
    if (data.size() < sizeof(FatbinHeader64)) {
      return payloads;
    }
    const std::uint16_t hdr_size = ReadU16(data.data() + 6);
    offset = (hdr_size >= sizeof(FatbinHeader64)) ? hdr_size : sizeof(FatbinHeader64);
    const std::uint64_t total_size = ReadU64(data.data() + 8);
    const std::size_t limit = (total_size > 0 && total_size <= data.size())
                                  ? static_cast<std::size_t>(total_size)
                                  : data.size();

    while (offset + 16 <= limit) {
      // Entry record in 64-bit fatbinary
      const std::uint16_t entry_kind = ReadU16(data.data() + offset);
      const std::uint16_t entry_hdr_size = ReadU16(data.data() + offset + 2);
      if (entry_hdr_size < 16 || offset + entry_hdr_size > limit) {
        break;
      }

      const std::uint64_t payload_size = ReadU64(data.data() + offset + 8);
      if (offset + entry_hdr_size + payload_size > limit) {
        break;
      }

      FatbinPayload payload;
      // Flags: 0x0001 = PTX text, 0x0002 = ELF CUBIN
      if ((entry_kind & 0x0001) != 0) {
        payload.kind = FatbinPayloadKind::kPtx;
      } else if ((entry_kind & 0x0002) != 0 ||
                 (payload_size >= 4 && data[offset + entry_hdr_size] == 0x7F &&
                  data[offset + entry_hdr_size + 1] == 'E' &&
                  data[offset + entry_hdr_size + 2] == 'L' &&
                  data[offset + entry_hdr_size + 3] == 'F')) {
        payload.kind = FatbinPayloadKind::kCubin;
      }

      const std::uint8_t* payload_ptr = data.data() + offset + entry_hdr_size;
      payload.raw_bytes = std::span<const std::uint8_t>(payload_ptr, static_cast<std::size_t>(payload_size));

      if (payload.kind == FatbinPayloadKind::kPtx) {
        payload.text = std::string_view(reinterpret_cast<const char*>(payload_ptr),
                                        static_cast<std::size_t>(payload_size));
        // Strip trailing null bytes
        while (!payload.text.empty() && payload.text.back() == '\0') {
          payload.text.remove_suffix(1);
        }
      }

      // Check architecture tag in entry header if available
      if (entry_hdr_size >= 24) {
        payload.arch_sm = ReadU32(data.data() + offset + 16);
      }

      payloads.push_back(std::move(payload));
      offset += entry_hdr_size + static_cast<std::size_t>(payload_size);
    }
  } else if (magic == kFatbinMagic32) {
    if (data.size() < sizeof(FatbinHeader32)) {
      return payloads;
    }
    const std::uint16_t hdr_size = ReadU16(data.data() + 6);
    offset = (hdr_size >= sizeof(FatbinHeader32)) ? hdr_size : sizeof(FatbinHeader32);
    const std::uint32_t total_size = ReadU32(data.data() + 8);
    const std::size_t limit = (total_size > 0 && total_size <= data.size()) ? total_size : data.size();

    while (offset + 12 <= limit) {
      const std::uint16_t entry_kind = ReadU16(data.data() + offset);
      const std::uint16_t entry_hdr_size = ReadU16(data.data() + offset + 2);
      if (entry_hdr_size < 12 || offset + entry_hdr_size > limit) {
        break;
      }
      const std::uint32_t payload_size = ReadU32(data.data() + offset + 4);
      if (offset + entry_hdr_size + payload_size > limit) {
        break;
      }

      FatbinPayload payload;
      if ((entry_kind & 0x0001) != 0) {
        payload.kind = FatbinPayloadKind::kPtx;
      } else {
        payload.kind = FatbinPayloadKind::kCubin;
      }

      const std::uint8_t* payload_ptr = data.data() + offset + entry_hdr_size;
      payload.raw_bytes = std::span<const std::uint8_t>(payload_ptr, payload_size);
      if (payload.kind == FatbinPayloadKind::kPtx) {
        payload.text = std::string_view(reinterpret_cast<const char*>(payload_ptr), payload_size);
        while (!payload.text.empty() && payload.text.back() == '\0') {
          payload.text.remove_suffix(1);
        }
      }

      payloads.push_back(std::move(payload));
      offset += entry_hdr_size + payload_size;
    }
  }

  // Resilient heuristic scanner fallback:
  // If structured records were unreadable or non-standard, scan data for embedded PTX or ELF markers
  if (payloads.empty()) {
    std::string_view text(reinterpret_cast<const char*>(data.data()), data.size());
    std::size_t ptx_pos = text.find(".version");
    while (ptx_pos != std::string_view::npos) {
      // Confirm this looks like a PTX file
      if (text.find(".target", ptx_pos) != std::string_view::npos) {
        std::size_t end_pos = text.find('\0', ptx_pos);
        if (end_pos == std::string_view::npos) {
          end_pos = text.size();
        }
        FatbinPayload p;
        p.kind = FatbinPayloadKind::kPtx;
        p.text = text.substr(ptx_pos, end_pos - ptx_pos);
        p.raw_bytes = data.subspan(ptx_pos, end_pos - ptx_pos);
        payloads.push_back(std::move(p));
        break;
      }
      ptx_pos = text.find(".version", ptx_pos + 8);
    }
  }

  return payloads;
}

std::optional<FatbinPayload> FatbinParser::ExtractBestPtx(std::span<const std::uint8_t> data) {
  std::vector<FatbinPayload> all = Parse(data);
  std::optional<FatbinPayload> best;

  for (auto& item : all) {
    if (item.kind == FatbinPayloadKind::kPtx && !item.text.empty()) {
      if (!best.has_value() || item.arch_sm > best->arch_sm) {
        best = std::move(item);
      }
    }
  }
  return best;
}

std::optional<FatbinPayload> FatbinParser::ExtractBestCubin(std::span<const std::uint8_t> data) {
  std::vector<FatbinPayload> all = Parse(data);
  std::optional<FatbinPayload> best;

  for (auto& item : all) {
    if (item.kind == FatbinPayloadKind::kCubin && !item.raw_bytes.empty()) {
      if (!best.has_value() || item.arch_sm > best->arch_sm) {
        best = std::move(item);
      }
    }
  }
  return best;
}

}  // namespace muda::compiler::cubin


/**
 * @file fatbin.hpp
 * @brief Zero-overhead parser and PTX/CUBIN extractor for NVIDIA Fatbin containers.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace muda::compiler::cubin {

/**
 * @brief Standard NVIDIA Fatbin magic numbers.
 */
inline constexpr std::uint32_t kFatbinMagic32 = 0xBA55ED50;  // 32-bit container
inline constexpr std::uint32_t kFatbinMagic64 = 0x1EE55A01;  // 64-bit container
inline constexpr std::uint32_t kFatbinMagicText = 0x46415442;  // "FATB"

enum class FatbinPayloadKind : std::uint8_t {
  kPtx,
  kCubin,
  kUnknown,
};

struct FatbinPayload {
  FatbinPayloadKind kind = FatbinPayloadKind::kUnknown;
  std::uint32_t arch_sm = 0;      // e.g. 70, 75, 80, 86, 90
  std::string target_name;        // e.g. "compute_70", "sm_80"
  std::span<const std::uint8_t> raw_bytes;
  std::string_view text;          // for PTX payloads
};

class FatbinParser final {
 public:
  /**
   * @brief Checks if a buffer begins with a recognized NVIDIA Fatbinary magic header.
   */
  [[nodiscard]] static bool IsFatbinary(std::span<const std::uint8_t> data) noexcept;

  /**
   * @brief Parses a fatbinary container and enumerates all candidate payloads.
   */
  [[nodiscard]] static std::vector<FatbinPayload> Parse(std::span<const std::uint8_t> data);

  /**
   * @brief Extracts the highest-capability PTX payload available in the container, if any.
   */
  [[nodiscard]] static std::optional<FatbinPayload> ExtractBestPtx(
      std::span<const std::uint8_t> data);

  /**
   * @brief Extracts the best matching CUBIN binary payload in the container, if any.
   */
  [[nodiscard]] static std::optional<FatbinPayload> ExtractBestCubin(
      std::span<const std::uint8_t> data);
};

}  // namespace muda::compiler::cubin


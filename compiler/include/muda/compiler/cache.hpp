/**
 * @file cache.hpp
 * @brief Content-addressed artifact cache for compiled Metal libraries.
 */

#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "muda/compiler/types.hpp"

namespace muda::compiler {

struct CacheKeyInput {
  std::string_view msl_source;
  std::string_view muda_version;
  std::string_view toolchain_version;
  MetalStandard metal_standard = MetalStandard::kMetal32;
  OptimizationLevel optimization = OptimizationLevel::kO3;
  std::string_view dependencies_hash;
};

struct CacheStats {
  std::size_t entry_count = 0;
  std::uint64_t total_bytes = 0;
};

struct CacheVerifyReport {
  std::size_t valid_entries = 0;
  std::size_t corrupted_evicted = 0;
};

class ArtifactCache {
 public:
  explicit ArtifactCache(std::filesystem::path root_dir);

  [[nodiscard]] static std::filesystem::path DefaultCacheDirectory();
  [[nodiscard]] static std::string ComputeKey(const CacheKeyInput& input) noexcept;
  [[nodiscard]] static std::string ComputeDependencyHash(
      const std::filesystem::path& source_file,
      const std::vector<std::filesystem::path>& include_dirs);

  [[nodiscard]] std::optional<std::vector<std::uint8_t>> Lookup(std::string_view key) const;
  bool Store(std::string_view key, std::span<const std::uint8_t> metallib_bytes) const;
  bool CopyCachedMetallibTo(std::string_view key, const std::filesystem::path& destination) const;

  [[nodiscard]] CacheStats Inspect() const noexcept;
  bool Clear() const noexcept;
  [[nodiscard]] CacheVerifyReport Verify() const noexcept;

  [[nodiscard]] const std::filesystem::path& root_dir() const noexcept { return root_dir_; }

 private:
  [[nodiscard]] std::filesystem::path EntryPath(std::string_view key) const;

  std::filesystem::path root_dir_;
};

}  // namespace muda::compiler

/**
 * @file cache.cpp
 * @brief Implements the content-addressed compiler artifact cache.
 */

#include "muda/compiler/cache.hpp"

#include <unistd.h>

#include <array>
#include <atomic>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <system_error>
#include <unordered_set>
#include <utility>

namespace muda::compiler {
namespace {

std::atomic<std::uint64_t> g_cache_temp_counter{0};

constexpr std::uint64_t FnV1a64(std::string_view bytes, std::uint64_t seed) noexcept {
  constexpr std::uint64_t kPrime = 1099511628211ull;
  std::uint64_t hash = seed;
  for (const char ch : bytes) {
    hash ^= static_cast<std::uint64_t>(static_cast<unsigned char>(ch));
    hash *= kPrime;
  }
  return hash;
}

std::string ToHex64(std::uint64_t value) {
  constexpr std::array<char, 16> kHexDigits = {'0', '1', '2', '3', '4', '5', '6', '7',
                                               '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'};
  std::string out(16, '0');
  for (int i = 15; i >= 0; --i) {
    out[static_cast<std::size_t>(i)] = kHexDigits[value & 0xFull];
    value >>= 4u;
  }
  return out;
}

void ScanIncludesRecursively(const std::filesystem::path& file_path,
                             const std::vector<std::filesystem::path>& include_dirs,
                             std::unordered_set<std::string>& visited, std::uint64_t* inout_hash) {
  std::ifstream stream(file_path);
  if (!stream) {
    return;
  }

  std::string line;
  while (std::getline(stream, line)) {
    std::string_view sv = line;
    while (!sv.empty() && (sv.front() == ' ' || sv.front() == '\t')) {
      sv.remove_prefix(1);
    }
    if (!sv.starts_with('#')) {
      continue;
    }
    sv.remove_prefix(1);
    while (!sv.empty() && (sv.front() == ' ' || sv.front() == '\t')) {
      sv.remove_prefix(1);
    }
    if (!sv.starts_with("include")) {
      continue;
    }
    sv.remove_prefix(7);
    while (!sv.empty() && (sv.front() == ' ' || sv.front() == '\t')) {
      sv.remove_prefix(1);
    }
    if (sv.empty()) {
      continue;
    }
    char close_delim = 0;
    if (sv.front() == '"') {
      close_delim = '"';
    } else if (sv.front() == '<') {
      close_delim = '>';
    } else {
      continue;
    }
    sv.remove_prefix(1);
    const auto close_pos = sv.find(close_delim);
    if (close_pos == std::string_view::npos) {
      continue;
    }
    const std::filesystem::path header_name = std::string(sv.substr(0, close_pos));

    std::filesystem::path resolved;
    if (close_delim == '"') {
      const std::filesystem::path local = file_path.parent_path() / header_name;
      std::error_code ec;
      if (std::filesystem::exists(local, ec) && !std::filesystem::is_directory(local, ec)) {
        resolved = std::filesystem::canonical(local, ec);
      }
    }
    if (resolved.empty()) {
      for (const auto& inc_dir : include_dirs) {
        const std::filesystem::path candidate = inc_dir / header_name;
        std::error_code ec;
        if (std::filesystem::exists(candidate, ec) &&
            !std::filesystem::is_directory(candidate, ec)) {
          resolved = std::filesystem::canonical(candidate, ec);
          break;
        }
      }
    }

    if (!resolved.empty()) {
      const std::string resolved_str = resolved.string();
      if (visited.insert(resolved_str).second) {
        std::ifstream header_stream(resolved, std::ios::binary);
        if (header_stream) {
          const std::string content((std::istreambuf_iterator<char>(header_stream)),
                                    std::istreambuf_iterator<char>());
          *inout_hash = FnV1a64(resolved_str, *inout_hash);
          *inout_hash = FnV1a64(":", *inout_hash);
          *inout_hash = FnV1a64(content, *inout_hash);
          *inout_hash = FnV1a64(";", *inout_hash);
        }
        ScanIncludesRecursively(resolved, include_dirs, visited, inout_hash);
      }
    }
  }
}

}  // namespace

ArtifactCache::ArtifactCache(std::filesystem::path root_dir) : root_dir_(std::move(root_dir)) {}

std::filesystem::path ArtifactCache::DefaultCacheDirectory() {
  if (const char* env_dir = std::getenv("MUDA_CACHE_DIR");
      env_dir != nullptr && env_dir[0] != '\0') {
    return std::filesystem::path(env_dir);
  }
  return std::filesystem::temp_directory_path() / "muda_artifact_cache";
}

std::string ArtifactCache::ComputeKey(const CacheKeyInput& input) noexcept {
  constexpr std::uint64_t kBasis1 = 14695981039346656037ull;
  constexpr std::uint64_t kBasis2 = 0x9e3779b97f4a7c15ull;

  const std::string_view opt_flag = OptimizationLevelFlag(input.optimization);
  const std::string_view std_flag = MetalStandardFlag(input.metal_standard);

  std::uint64_t h1 = FnV1a64(input.muda_version, kBasis1);
  h1 = FnV1a64("|", h1);
  h1 = FnV1a64(input.toolchain_version, h1);
  h1 = FnV1a64("|", h1);
  h1 = FnV1a64(std_flag, h1);
  h1 = FnV1a64("|", h1);
  h1 = FnV1a64(opt_flag, h1);
  h1 = FnV1a64("|", h1);
  h1 = FnV1a64(input.dependencies_hash, h1);
  h1 = FnV1a64("|", h1);
  h1 = FnV1a64(input.msl_source, h1);

  std::uint64_t h2 = FnV1a64(input.msl_source, kBasis2);
  h2 = FnV1a64("|", h2);
  h2 = FnV1a64(input.dependencies_hash, h2);
  h2 = FnV1a64("|", h2);
  h2 = FnV1a64(std_flag, h2);
  h2 = FnV1a64("|", h2);
  h2 = FnV1a64(opt_flag, h2);
  h2 = FnV1a64("|", h2);
  h2 = FnV1a64(input.toolchain_version, h2);
  h2 = FnV1a64("|", h2);
  h2 = FnV1a64(input.muda_version, h2);

  return ToHex64(h1) + ToHex64(h2);
}

std::string ArtifactCache::ComputeDependencyHash(
    const std::filesystem::path& source_file,
    const std::vector<std::filesystem::path>& include_dirs) {
  if (source_file.empty()) {
    return "";
  }
  std::unordered_set<std::string> visited;
  constexpr std::uint64_t kBasis = 14695981039346656037ull;
  std::uint64_t hash = kBasis;
  ScanIncludesRecursively(source_file, include_dirs, visited, &hash);
  if (visited.empty()) {
    return "";
  }
  return ToHex64(hash);
}

std::filesystem::path ArtifactCache::EntryPath(std::string_view key) const {
  return root_dir_ / std::string(key) / "artifact.metallib";
}

std::optional<std::vector<std::uint8_t>> ArtifactCache::Lookup(std::string_view key) const {
  if (root_dir_.empty() || key.empty()) {
    return std::nullopt;
  }
  const std::filesystem::path path = EntryPath(key);
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    return std::nullopt;
  }
  const std::streamoff size = file.tellg();
  if (size <= 4) {
    return std::nullopt;
  }
  file.seekg(0, std::ios::beg);
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
  file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(size));
  if (!file) {
    return std::nullopt;
  }
  // Verify Metal library magic header ("MTLB") to reject corrupted cache files.
  if (bytes[0] != 'M' || bytes[1] != 'T' || bytes[2] != 'L' || bytes[3] != 'B') {
    std::error_code ec;
    std::filesystem::remove(path, ec);
    return std::nullopt;
  }
  return bytes;
}

bool ArtifactCache::Store(std::string_view key,
                          std::span<const std::uint8_t> metallib_bytes) const {
  if (root_dir_.empty() || key.empty() || metallib_bytes.size() <= 4) {
    return false;
  }
  const std::filesystem::path entry_file = EntryPath(key);
  const std::filesystem::path entry_dir = entry_file.parent_path();

  std::error_code ec;
  std::filesystem::create_directories(entry_dir, ec);
  if (ec) {
    return false;
  }

  const std::uint64_t temp_id = g_cache_temp_counter.fetch_add(1, std::memory_order_relaxed);
  const std::filesystem::path temp_file =
      entry_dir / ("artifact.tmp." + std::to_string(getpid()) + "." + std::to_string(temp_id));

  {
    std::ofstream out(temp_file, std::ios::binary);
    if (!out) {
      return false;
    }
    out.write(reinterpret_cast<const char*>(metallib_bytes.data()),
              static_cast<std::streamsize>(metallib_bytes.size()));
    if (!out) {
      std::filesystem::remove(temp_file, ec);
      return false;
    }
  }

  std::filesystem::rename(temp_file, entry_file, ec);
  if (ec) {
    std::filesystem::remove(temp_file, ec);
    return false;
  }
  return true;
}

bool ArtifactCache::CopyCachedMetallibTo(std::string_view key,
                                         const std::filesystem::path& destination) const {
  const std::optional<std::vector<std::uint8_t>> cached = Lookup(key);
  if (!cached.has_value()) {
    return false;
  }
  std::ofstream out(destination, std::ios::binary);
  if (!out) {
    return false;
  }
  out.write(reinterpret_cast<const char*>(cached->data()),
            static_cast<std::streamsize>(cached->size()));
  return static_cast<bool>(out);
}

CacheStats ArtifactCache::Inspect() const noexcept {
  CacheStats stats;
  if (root_dir_.empty()) {
    return stats;
  }
  std::error_code ec;
  if (!std::filesystem::exists(root_dir_, ec) || !std::filesystem::is_directory(root_dir_, ec)) {
    return stats;
  }
  for (const auto& entry : std::filesystem::directory_iterator(root_dir_, ec)) {
    if (!entry.is_directory(ec)) {
      continue;
    }
    const std::filesystem::path artifact_file = entry.path() / "artifact.metallib";
    if (std::filesystem::is_regular_file(artifact_file, ec)) {
      const std::uintmax_t file_size = std::filesystem::file_size(artifact_file, ec);
      if (!ec) {
        ++stats.entry_count;
        stats.total_bytes += static_cast<std::uint64_t>(file_size);
      }
    }
  }
  return stats;
}

bool ArtifactCache::Clear() const noexcept {
  if (root_dir_.empty()) {
    return false;
  }
  std::error_code ec;
  if (!std::filesystem::exists(root_dir_, ec)) {
    return true;
  }
  std::filesystem::remove_all(root_dir_, ec);
  if (ec) {
    return false;
  }
  std::filesystem::create_directories(root_dir_, ec);
  return !ec;
}

CacheVerifyReport ArtifactCache::Verify() const noexcept {
  CacheVerifyReport report;
  if (root_dir_.empty()) {
    return report;
  }
  std::error_code ec;
  if (!std::filesystem::exists(root_dir_, ec) || !std::filesystem::is_directory(root_dir_, ec)) {
    return report;
  }
  for (const auto& entry : std::filesystem::directory_iterator(root_dir_, ec)) {
    if (!entry.is_directory(ec)) {
      continue;
    }
    const std::filesystem::path artifact_file = entry.path() / "artifact.metallib";
    if (!std::filesystem::is_regular_file(artifact_file, ec)) {
      std::filesystem::remove_all(entry.path(), ec);
      ++report.corrupted_evicted;
      continue;
    }
    std::ifstream file(artifact_file, std::ios::binary | std::ios::ate);
    const std::streamoff size = file ? static_cast<std::streamoff>(file.tellg()) : 0;
    bool valid = (size > 4);
    if (valid) {
      file.seekg(0, std::ios::beg);
      std::array<char, 4> magic{};
      file.read(magic.data(), 4);
      valid = static_cast<bool>(file) && magic[0] == 'M' && magic[1] == 'T' && magic[2] == 'L' &&
              magic[3] == 'B';
    }
    file.close();
    if (valid) {
      ++report.valid_entries;
    } else {
      std::filesystem::remove_all(entry.path(), ec);
      ++report.corrupted_evicted;
    }
  }
  return report;
}

}  // namespace muda::compiler

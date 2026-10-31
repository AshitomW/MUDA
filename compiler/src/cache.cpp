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
#include <system_error>
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

  std::uint64_t h1 = FnV1a64(input.muda_version, kBasis1);
  h1 = FnV1a64("|", h1);
  h1 = FnV1a64(input.msl_standard, h1);
  h1 = FnV1a64("|", h1);
  h1 = FnV1a64(input.msl_source, h1);

  std::uint64_t h2 = FnV1a64(input.msl_source, kBasis2);
  h2 = FnV1a64("|", h2);
  h2 = FnV1a64(input.muda_version, h2);

  return ToHex64(h1) + ToHex64(h2);
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

}  // namespace muda::compiler

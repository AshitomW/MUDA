/**
 * @file test_utils.hpp
 * @brief Shared RAII temporary directory and floating-point comparison helpers for tests.
 */

#pragma once

#include <unistd.h>

#include <cmath>
#include <filesystem>
#include <string>
#include <string_view>
#include <system_error>

namespace muda::testing {

inline constexpr float kDefaultFloatTolerance = 1e-5f;

[[nodiscard]] inline bool AreClose(float actual, float expected,
                                   float tolerance = kDefaultFloatTolerance) noexcept {
  return std::fabs(actual - expected) <= tolerance;
}

class ScopedTempDir {
 public:
  explicit ScopedTempDir(std::string_view prefix)
      : path_(std::filesystem::temp_directory_path() /
              (std::string(prefix) + "_" + std::to_string(getpid()))) {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
    std::filesystem::create_directories(path_, error);
    valid_ = !error;
  }

  ~ScopedTempDir() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }

  ScopedTempDir(const ScopedTempDir&) = delete;
  ScopedTempDir& operator=(const ScopedTempDir&) = delete;

  [[nodiscard]] bool valid() const noexcept { return valid_; }
  [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

 private:
  std::filesystem::path path_;
  bool valid_ = false;
};

}  // namespace muda::testing

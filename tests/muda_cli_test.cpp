/**
 * @file muda_cli_test.cpp
 * @brief Integration tests for `muda doctor`, `muda path`, and `muda cache` CLI commands.
 */

#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>

#include "muda/compiler/cache.hpp"
#include "test_utils.hpp"

namespace {

bool Expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    return false;
  }
  return true;
}

struct CommandOutput {
  int exit_code = -1;
  std::string stdout_text;
};

CommandOutput RunCommandCapture(const std::string& cmd) {
  CommandOutput result;
  FILE* const pipe = popen(cmd.c_str(), "r");
  if (pipe == nullptr) {
    return result;
  }
  std::array<char, 512> buffer{};
  while (std::fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
    result.stdout_text += buffer.data();
  }
  const int status = pclose(pipe);
  if (WIFEXITED(status)) {
    result.exit_code = WEXITSTATUS(status);
  } else {
    result.exit_code = status;
  }
  return result;
}

bool TestDoctorCommands(const std::string& muda_bin) {
  const CommandOutput text_out = RunCommandCapture("\"" + muda_bin + "\" doctor --smoke-test");
  if (!Expect(text_out.exit_code == 0, "muda doctor --smoke-test should succeed")) {
    return false;
  }
  if (!Expect(text_out.stdout_text.find("metal toolchain: available") != std::string::npos,
              "doctor output should report metal toolchain available")) {
    return false;
  }
  if (!Expect(text_out.stdout_text.find("gpu smoke test: passed") != std::string::npos,
              "doctor output should report gpu smoke test passed")) {
    return false;
  }

  const CommandOutput json_out =
      RunCommandCapture("\"" + muda_bin + "\" doctor --json --smoke-test");
  if (!Expect(json_out.exit_code == 0, "muda doctor --json --smoke-test should succeed")) {
    return false;
  }
  if (!Expect(json_out.stdout_text.find("\"healthy\": true") != std::string::npos,
              "doctor JSON output should report healthy: true")) {
    return false;
  }
  if (!Expect(json_out.stdout_text.find("\"smoke_test\": true") != std::string::npos,
              "doctor JSON output should report smoke_test: true")) {
    return false;
  }
  if (!Expect(json_out.stdout_text.find("\"devices\": [") != std::string::npos,
              "doctor JSON output should include devices array")) {
    return false;
  }
  return true;
}

bool TestPathCommands(const std::string& muda_bin) {
  constexpr std::array<std::string_view, 5> kTargets = {"mudac", "libcuda.dylib", "libcudart.dylib",
                                                        "include", "cache"};

  for (const std::string_view target : kTargets) {
    const CommandOutput out = RunCommandCapture("\"" + muda_bin + "\" path " + std::string(target));
    if (!Expect(out.exit_code == 0, "muda path target should exit with 0")) {
      return false;
    }
    std::string line = out.stdout_text;
    while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) {
      line.pop_back();
    }
    if (!Expect(!line.empty() && std::filesystem::exists(line),
                "muda path output must be an existing filesystem path")) {
      return false;
    }
  }
  return true;
}

bool TestCacheSubcommands(const std::string& muda_bin) {
  muda::testing::ScopedTempDir temp_cache("muda_cli_cache_test");
  if (!Expect(temp_cache.valid(), "Failed to create temporary cache directory")) {
    return false;
  }

  const muda::compiler::ArtifactCache cache(temp_cache.path());
  const std::array<std::uint8_t, 8> valid_metallib = {'M', 'T', 'L', 'B', 1, 2, 3, 4};
  if (!Expect(cache.Store("valid_entry_01", valid_metallib), "Failed to store valid cache entry")) {
    return false;
  }

  // Create a corrupted cache entry
  const std::filesystem::path bad_dir = temp_cache.path() / "corrupt_entry_02";
  std::filesystem::create_directories(bad_dir);
  {
    std::ofstream bad_file(bad_dir / "artifact.metallib", std::ios::binary);
    bad_file << "BAD!";
  }

  const std::string env_prefix = "MUDA_CACHE_DIR=\"" + temp_cache.path().string() + "\" ";
  const CommandOutput info_out = RunCommandCapture(env_prefix + "\"" + muda_bin + "\" cache info");
  if (!Expect(
          info_out.exit_code == 0 && info_out.stdout_text.find("entries: 2") != std::string::npos,
          "muda cache info should report 2 entries before verification")) {
    return false;
  }

  const CommandOutput verify_out =
      RunCommandCapture(env_prefix + "\"" + muda_bin + "\" cache verify");
  if (!Expect(verify_out.exit_code == 0 &&
                  verify_out.stdout_text.find("1 valid, 1 corrupted evicted") != std::string::npos,
              "muda cache verify should evict corrupted entry and retain valid entry")) {
    return false;
  }

  const CommandOutput clear_out =
      RunCommandCapture(env_prefix + "\"" + muda_bin + "\" cache clear");
  if (!Expect(clear_out.exit_code == 0, "muda cache clear should succeed")) {
    return false;
  }

  const muda::compiler::CacheStats after_clear = cache.Inspect();
  if (!Expect(after_clear.entry_count == 0, "Cache must have 0 entries after clear")) {
    return false;
  }

  return true;
}

}  // namespace

int main() {
#ifndef MUDA_CLI_BINARY_PATH
  std::cerr << "MUDA_CLI_BINARY_PATH not defined\n";
  return 1;
#else
  const std::string muda_bin = MUDA_CLI_BINARY_PATH;
  if (!TestDoctorCommands(muda_bin)) {
    return 1;
  }
  if (!TestPathCommands(muda_bin)) {
    return 1;
  }
  if (!TestCacheSubcommands(muda_bin)) {
    return 1;
  }
  std::cout << "All muda CLI tests passed!\n";
  return 0;
#endif
}

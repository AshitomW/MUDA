/**
 * @file main.cpp
 * @brief Implements the MUDA environment diagnostics, path resolution, and cache management CLI.
 */

#include <mach-o/dyld.h>
#include <sys/utsname.h>
#include <unistd.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include "cuda_runtime.h"
#include "muda/compiler/cache.hpp"
#include "muda/compiler/toolchain.hpp"
#include "muda/runtime/device.hpp"
#include "muda/version.hpp"

namespace {

void PrintUsage(std::ostream& out) {
  out << "muda " << muda::kVersion << "\n"
      << "Usage: muda <command> [options]\n"
      << "\n"
      << "Commands:\n"
      << "  doctor [--json] [--smoke-test]   Inspect the Metal environment and toolchain\n"
      << "  path <target>                    Print absolute path for integration targets\n"
      << "                                   (cuda-home, libcuda.dylib, libcudart.dylib,\n"
      << "                                    include, cache, mudac)\n"
      << "  cache <info|clear|verify>        Manage the compiler artifact cache\n"
      << "  version                          Show version and target metadata\n";
}

std::string EscapeJsonString(std::string_view input) {
  std::string out;
  out.reserve(input.size() + 4);
  for (const char ch : input) {
    switch (ch) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        out.push_back(ch);
        break;
    }
  }
  return out;
}

std::filesystem::path GetExecutableDirectory() {
  std::uint32_t buf_size = 0;
  (void)_NSGetExecutablePath(nullptr, &buf_size);
  if (buf_size == 0) {
    return std::filesystem::current_path();
  }
  std::vector<char> buffer(buf_size, '\0');
  if (_NSGetExecutablePath(buffer.data(), &buf_size) != 0) {
    return std::filesystem::current_path();
  }
  std::error_code ec;
  const std::filesystem::path canonical_path = std::filesystem::weakly_canonical(buffer.data(), ec);
  if (ec) {
    return std::filesystem::path(buffer.data()).parent_path();
  }
  return canonical_path.parent_path();
}

std::optional<std::filesystem::path> FindFirstExisting(
    const std::vector<std::filesystem::path>& candidates) {
  std::error_code ec;
  for (const auto& candidate : candidates) {
    if (!candidate.empty() && std::filesystem::exists(candidate, ec)) {
      return std::filesystem::weakly_canonical(candidate, ec);
    }
  }
  return std::nullopt;
}

std::optional<std::filesystem::path> ResolveTargetPath(std::string_view target) {
  const std::filesystem::path exe_dir = GetExecutableDirectory();
  const std::filesystem::path prefix_dir = exe_dir.parent_path();

#ifdef MUDA_BUILD_ROOT_DIR
  const std::filesystem::path build_root(MUDA_BUILD_ROOT_DIR);
#else
  const std::filesystem::path build_root = prefix_dir;
#endif

#ifdef MUDA_SOURCE_ROOT_DIR
  const std::filesystem::path source_root(MUDA_SOURCE_ROOT_DIR);
#else
  const std::filesystem::path source_root = prefix_dir;
#endif

  if (target == "cache") {
    const std::filesystem::path cache_dir = muda::compiler::ArtifactCache::DefaultCacheDirectory();
    std::error_code ec;
    std::filesystem::create_directories(cache_dir, ec);
    return std::filesystem::weakly_canonical(cache_dir, ec);
  }
  if (target == "mudac") {
    return FindFirstExisting({
        exe_dir / "mudac",
        build_root / "tools" / "mudac",
    });
  }
  if (target == "cuda-home") {
    return FindFirstExisting({
        prefix_dir / "share" / "muda" / "cuda-home",
        build_root / "cuda-home",
        prefix_dir,
    });
  }
  if (target == "libcuda.dylib") {
    return FindFirstExisting({
        prefix_dir / "lib" / "libcuda.dylib",
        build_root / "runtime" / "libcuda.dylib",
        build_root / "cuda-home" / "lib" / "libcuda.dylib",
    });
  }
  if (target == "libcudart.dylib") {
    return FindFirstExisting({
        prefix_dir / "lib" / "libcudart.dylib",
        build_root / "runtime" / "libcudart.dylib",
        build_root / "cuda-home" / "lib" / "libcudart.dylib",
        prefix_dir / "lib" / "libcuda.dylib",
        build_root / "runtime" / "libcuda.dylib",
    });
  }
  if (target == "include") {
    return FindFirstExisting({
        prefix_dir / "include",
        build_root / "cuda-home" / "include",
        source_root / "runtime" / "include",
    });
  }
  return std::nullopt;
}

bool CheckCacheWritable(const std::filesystem::path& cache_dir) noexcept {
  std::error_code ec;
  std::filesystem::create_directories(cache_dir, ec);
  if (ec) {
    return false;
  }
  const std::filesystem::path probe = cache_dir / (".muda_probe_" + std::to_string(getpid()));
  std::ofstream out(probe, std::ios::binary);
  if (!out) {
    return false;
  }
  out << "ok";
  out.close();
  std::filesystem::remove(probe, ec);
  return true;
}

bool RunRuntimeSmokeTest() noexcept {
  float* dev_ptr = nullptr;
  if (cudaMalloc(reinterpret_cast<void**>(&dev_ptr), 4 * sizeof(float)) != cudaSuccess) {
    return false;
  }
  const std::array<float, 4> host_in = {1.0f, 2.0f, 3.0f, 4.0f};
  std::array<float, 4> host_out = {0.0f, 0.0f, 0.0f, 0.0f};
  const bool ok = (cudaMemcpy(dev_ptr, host_in.data(), sizeof(host_in), cudaMemcpyHostToDevice) ==
                   cudaSuccess) &&
                  (cudaMemcpy(host_out.data(), dev_ptr, sizeof(host_out), cudaMemcpyDeviceToHost) ==
                   cudaSuccess) &&
                  (cudaDeviceSynchronize() == cudaSuccess) && (host_out == host_in);
  (void)cudaFree(dev_ptr);
  return ok;
}

int DoctorCommand(bool json_output, bool run_smoke_test) {
  struct utsname sys_info{};
  const bool has_uname = (uname(&sys_info) == 0);
  const std::string os_sysname = has_uname ? sys_info.sysname : "Darwin";
  const std::string os_release = has_uname ? sys_info.release : "unknown";
  const std::string os_machine = has_uname ? sys_info.machine : "arm64";

  const std::vector<muda::runtime::DeviceInfo> devices = muda::runtime::EnumerateDevices();
  const bool offline_toolchain_available = muda::compiler::MetalToolchainAvailable();
  const bool toolchain_available = offline_toolchain_available || !devices.empty();
  const std::string toolchain_version = offline_toolchain_available
                                            ? muda::compiler::GetMetalToolchainVersion()
                                            : (!devices.empty() ? "runtime_jit" : "missing");

  const std::filesystem::path cache_dir = muda::compiler::ArtifactCache::DefaultCacheDirectory();
  const bool cache_writable = CheckCacheWritable(cache_dir);
  const auto libcuda_path = ResolveTargetPath("libcuda.dylib");
  const auto include_path = ResolveTargetPath("include");
  const bool smoke_ok = run_smoke_test ? RunRuntimeSmokeTest() : true;

  const bool healthy = !devices.empty() && toolchain_available && cache_writable && smoke_ok;

  if (json_output) {
    std::cout << "{\n"
              << "  \"version\": \"" << EscapeJsonString(muda::kVersion) << "\",\n"
              << "  \"healthy\": " << (healthy ? "true" : "false") << ",\n"
              << "  \"os\": {\n"
              << "    \"sysname\": \"" << EscapeJsonString(os_sysname) << "\",\n"
              << "    \"release\": \"" << EscapeJsonString(os_release) << "\",\n"
              << "    \"machine\": \"" << EscapeJsonString(os_machine) << "\"\n"
              << "  },\n"
              << "  \"metal_toolchain\": {\n"
              << "    \"available\": " << (toolchain_available ? "true" : "false") << ",\n"
              << "    \"version\": \"" << EscapeJsonString(toolchain_version) << "\"\n"
              << "  },\n"
              << "  \"cache\": {\n"
              << "    \"path\": \"" << EscapeJsonString(cache_dir.string()) << "\",\n"
              << "    \"writable\": " << (cache_writable ? "true" : "false") << "\n"
              << "  },\n"
              << "  \"paths\": {\n"
              << "    \"libcuda\": \""
              << EscapeJsonString(libcuda_path ? libcuda_path->string() : "") << "\",\n"
              << "    \"include\": \""
              << EscapeJsonString(include_path ? include_path->string() : "") << "\"\n"
              << "  },\n";
    if (run_smoke_test) {
      std::cout << "  \"smoke_test\": " << (smoke_ok ? "true" : "false") << ",\n";
    }
    std::cout << "  \"devices\": [\n";
    for (std::size_t i = 0; i < devices.size(); ++i) {
      const auto& d = devices[i];
      std::cout << "    {\n"
                << "      \"name\": \"" << EscapeJsonString(d.name) << "\",\n"
                << "      \"backend\": \""
                << EscapeJsonString(muda::runtime::BackendName(d.backend)) << "\",\n"
                << "      \"unified_memory\": " << (d.unified_memory ? "true" : "false") << ",\n"
                << "      \"recommended_working_set_bytes\": " << d.recommended_working_set_bytes
                << ",\n"
                << "      \"max_buffer_length\": " << d.max_buffer_length << ",\n"
                << "      \"max_threads_per_threadgroup\": " << d.max_threads_per_threadgroup
                << ",\n"
                << "      \"max_threadgroup_memory_bytes\": " << d.max_threadgroup_memory_bytes
                << "\n"
                << "    }" << (i + 1 < devices.size() ? "," : "") << "\n";
    }
    std::cout << "  ]\n"
              << "}\n";
    return healthy ? 0 : 1;
  }

  std::cout << "muda version: " << muda::kVersion << "\n"
            << "host os: " << os_sysname << " " << os_release << " (" << os_machine << ")\n";
  for (const muda::runtime::DeviceInfo& device : devices) {
    std::cout << "device: " << device.name << "\n"
              << "  backend: " << muda::runtime::BackendName(device.backend) << "\n"
              << "  unified memory: " << (device.unified_memory ? "yes" : "no") << "\n"
              << "  working set: " << device.recommended_working_set_bytes << " bytes\n"
              << "  max buffer length: " << device.max_buffer_length << " bytes\n"
              << "  max threads per threadgroup: " << device.max_threads_per_threadgroup << "\n"
              << "  max threadgroup memory: " << device.max_threadgroup_memory_bytes << " bytes\n";
  }
  std::cout << "metal toolchain: " << (toolchain_available ? "available" : "missing") << " ("
            << toolchain_version << ")\n"
            << "cache directory: " << cache_dir.string() << " ("
            << (cache_writable ? "writable" : "not writable") << ")\n";
  if (libcuda_path.has_value()) {
    std::cout << "libcuda.dylib: " << libcuda_path->string() << "\n";
  }
  if (run_smoke_test) {
    std::cout << "gpu smoke test: " << (smoke_ok ? "passed" : "failed") << "\n";
  }

  if (devices.empty()) {
    std::cerr << "muda: no Metal devices found\n";
    return 1;
  }
  return healthy ? 0 : 1;
}

int PathCommand(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "Usage: muda path <cuda-home|libcuda.dylib|libcudart.dylib|include|cache|mudac>\n";
    return 2;
  }
  const std::string_view target = argv[2];
  const auto resolved = ResolveTargetPath(target);
  if (!resolved.has_value()) {
    std::cerr << "muda: unknown or missing path target: " << target << "\n";
    return 1;
  }
  std::cout << resolved->string() << "\n";
  return 0;
}

int CacheCommand(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "Usage: muda cache <info|clear|verify>\n";
    return 2;
  }
  const std::string_view subcommand = argv[2];
  const muda::compiler::ArtifactCache cache(muda::compiler::ArtifactCache::DefaultCacheDirectory());

  if (subcommand == "info") {
    const muda::compiler::CacheStats stats = cache.Inspect();
    std::cout << "cache directory: " << cache.root_dir().string() << "\n"
              << "entries: " << stats.entry_count << "\n"
              << "size: " << stats.total_bytes << " bytes\n";
    return 0;
  }
  if (subcommand == "clear") {
    if (!cache.Clear()) {
      std::cerr << "muda: failed to clear cache at " << cache.root_dir().string() << "\n";
      return 1;
    }
    std::cout << "cleared cache at " << cache.root_dir().string() << "\n";
    return 0;
  }
  if (subcommand == "verify") {
    const muda::compiler::CacheVerifyReport report = cache.Verify();
    std::cout << "verified cache at " << cache.root_dir().string() << ": " << report.valid_entries
              << " valid, " << report.corrupted_evicted << " corrupted evicted\n";
    return 0;
  }

  std::cerr << "muda: unknown cache subcommand: " << subcommand << "\n";
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    PrintUsage(std::cerr);
    return 2;
  }

  const std::string_view command = argv[1];
  if (command == "doctor") {
    bool json_output = false;
    bool smoke_test = false;
    for (int i = 2; i < argc; ++i) {
      const std::string_view arg = argv[i];
      if (arg == "--json") {
        json_output = true;
      } else if (arg == "--smoke-test") {
        smoke_test = true;
      } else {
        std::cerr << "muda doctor: unknown option: " << arg << "\n";
        return 2;
      }
    }
    return DoctorCommand(json_output, smoke_test);
  }
  if (command == "path") {
    return PathCommand(argc, argv);
  }
  if (command == "cache") {
    return CacheCommand(argc, argv);
  }
  if (command == "version" || command == "--version") {
    if (argc != 2) {
      PrintUsage(std::cerr);
      return 2;
    }
    std::cout << "MUDA " << muda::kVersion << " (CUDA 12.2 compatibility, Apple Silicon Metal)\n";
    return 0;
  }
  if (command == "help" || command == "-h" || command == "--help") {
    PrintUsage(std::cout);
    return 0;
  }

  std::cerr << "muda: unknown command: " << command << "\n";
  PrintUsage(std::cerr);
  return 2;
}

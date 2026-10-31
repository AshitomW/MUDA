/**
 * @file driver.cpp
 * @brief Orchestrates translation, Metal compilation, host lowering, and linking.
 */

#include "muda/compiler/driver.hpp"

#include <unistd.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "muda/compiler/cache.hpp"
#include "muda/compiler/host_rewriter.hpp"
#include "muda/compiler/toolchain.hpp"
#include "muda/compiler/translator.hpp"
#include "muda/version.hpp"
#include "process.hpp"

namespace muda::compiler {
namespace {

std::atomic<std::uint64_t> g_temp_counter{0};

CompileResult Failure(DiagnosticCode code) noexcept {
  CompileResult result;
  result.status = Status::kError;
  result.code = code;
  return result;
}

void RemoveFile(const std::filesystem::path& path) noexcept {
  if (path.empty()) {
    return;
  }
  std::error_code error;
  std::filesystem::remove(path, error);
}

class ScopedIntermediateCleaner {
 public:
  ScopedIntermediateCleaner(std::vector<std::filesystem::path> paths, IntermediatePolicy policy)
      : paths_(std::move(paths)), policy_(policy) {}

  ~ScopedIntermediateCleaner() {
    if (policy_ == IntermediatePolicy::kRemove) {
      for (const std::filesystem::path& path : paths_) {
        RemoveFile(path);
      }
    }
  }

  ScopedIntermediateCleaner(const ScopedIntermediateCleaner&) = delete;
  ScopedIntermediateCleaner& operator=(const ScopedIntermediateCleaner&) = delete;

 private:
  std::vector<std::filesystem::path> paths_;
  IntermediatePolicy policy_;
};

std::filesystem::path ResolveBaseDirectory(const std::filesystem::path& output,
                                           IntermediatePolicy policy) {
  if (policy == IntermediatePolicy::kKeep) {
    return output.has_parent_path() ? output.parent_path() : std::filesystem::current_path();
  }
  return std::filesystem::temp_directory_path();
}

std::string MakeIntermediateStem(const std::filesystem::path& output, IntermediatePolicy policy) {
  if (policy == IntermediatePolicy::kKeep) {
    return output.stem().string();
  }
  const std::uint64_t id = g_temp_counter.fetch_add(1, std::memory_order_relaxed);
  return output.stem().string() + "_" + std::to_string(getpid()) + "_" + std::to_string(id);
}

OutputKind ResolveOutputKind(const CompileOptions& options) {
  if (options.output_kind != OutputKind::kAuto) {
    return options.output_kind;
  }
  if (options.output.empty()) {
    return OutputKind::kMetallib;
  }
  const std::string ext = options.output.extension().string();
  if (ext == ".metallib") {
    return OutputKind::kMetallib;
  }
  if (ext == ".metal") {
    return OutputKind::kMslSource;
  }
  if (ext == ".o" || ext == ".obj") {
    return OutputKind::kObject;
  }
  return OutputKind::kExecutable;
}

std::filesystem::path ResolveOutputPath(const CompileOptions& options, OutputKind kind) {
  if (!options.output.empty()) {
    return options.output;
  }
  std::filesystem::path output = options.input;
  switch (kind) {
    case OutputKind::kAuto:
    case OutputKind::kMetallib:
      output.replace_extension(".metallib");
      break;
    case OutputKind::kMslSource:
      output.replace_extension(".metal");
      break;
    case OutputKind::kObject:
      output.replace_extension(".o");
      break;
    case OutputKind::kExecutable:
      output = options.input.stem();
      break;
  }
  return output;
}

bool WriteTextFile(const std::filesystem::path& path, std::string_view contents) {
  std::ofstream file(path, std::ios::binary);
  if (!file) {
    return false;
  }
  file.write(contents.data(), static_cast<std::streamsize>(contents.size()));
  return static_cast<bool>(file);
}

bool ReadBinaryFile(const std::filesystem::path& path, std::vector<std::uint8_t>* out_bytes) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    return false;
  }
  const std::streamoff size = file.tellg();
  if (size <= 0) {
    return false;
  }
  file.seekg(0, std::ios::beg);
  out_bytes->resize(static_cast<std::size_t>(size));
  file.read(reinterpret_cast<char*>(out_bytes->data()), static_cast<std::streamsize>(size));
  return static_cast<bool>(file);
}

bool RunProcessOwned(const std::vector<std::string>& args) {
  std::vector<std::string_view> views;
  views.reserve(args.size());
  for (const std::string& arg : args) {
    views.emplace_back(arg);
  }
  return detail::RunProcess(views).succeeded();
}

}  // namespace

CompileResult Compile(const CompileOptions& options) {
  if (options.input.empty()) {
    return Failure(DiagnosticCode::kNoInputFile);
  }

  const OutputKind output_kind = ResolveOutputKind(options);
  if (output_kind != OutputKind::kMslSource && !MetalToolchainAvailable()) {
    return Failure(DiagnosticCode::kToolchainNotFound);
  }

  std::ifstream input(options.input, std::ios::binary);
  if (!input) {
    return Failure(DiagnosticCode::kCannotOpenInput);
  }

  std::ostringstream source_buffer;
  source_buffer << input.rdbuf();
  const std::string source = source_buffer.str();

  TranslationResult translation = TranslateCudaToMsl(source);
  if (!translation.ok()) {
    return Failure(translation.code);
  }

  std::filesystem::path output = ResolveOutputPath(options, output_kind);
  std::error_code fs_error;
  if (output.has_parent_path() && !output.parent_path().empty()) {
    std::filesystem::create_directories(output.parent_path(), fs_error);
    if (fs_error) {
      return Failure(DiagnosticCode::kCannotCreateOutputDir);
    }
  }

  const std::string& msl_source =
      !translation.msl.empty() ? translation.msl : translation.kernel.msl;

  if (output_kind == OutputKind::kMslSource) {
    if (!WriteTextFile(output, msl_source)) {
      return Failure(DiagnosticCode::kCannotWriteMetalSource);
    }
    CompileResult result;
    result.status = Status::kOk;
    result.code = DiagnosticCode::kOk;
    result.output_path = output;
    result.kernel_name = std::move(translation.kernel.name);
    return result;
  }

  const std::filesystem::path base_dir = ResolveBaseDirectory(output, options.intermediates);
  const std::string stem = MakeIntermediateStem(output, options.intermediates);
  const std::filesystem::path metal_path = base_dir / (stem + ".metal");
  const std::filesystem::path air_path = base_dir / (stem + ".air");
  const std::filesystem::path metallib_path =
      output_kind == OutputKind::kMetallib ? output : (base_dir / (stem + ".metallib"));
  const std::filesystem::path host_cpp_path = base_dir / (stem + ".host.cpp");

  std::vector<std::filesystem::path> temp_files = {metal_path, air_path};
  if (output_kind != OutputKind::kMetallib) {
    temp_files.push_back(metallib_path);
    temp_files.push_back(host_cpp_path);
  }
  const ScopedIntermediateCleaner cleaner(std::move(temp_files), options.intermediates);

  if (!WriteTextFile(metal_path, msl_source)) {
    return Failure(DiagnosticCode::kCannotWriteMetalSource);
  }

  const char* env_disable_cache = std::getenv("MUDA_DISABLE_CACHE");
  const bool cache_enabled = !options.disable_cache &&
                             options.intermediates == IntermediatePolicy::kRemove &&
                             (env_disable_cache == nullptr || env_disable_cache[0] == '\0' ||
                              std::string_view(env_disable_cache) == "0");

  const std::filesystem::path cache_root =
      !options.cache_dir.empty() ? options.cache_dir : ArtifactCache::DefaultCacheDirectory();
  const ArtifactCache cache(cache_root);
  const std::string cache_key =
      cache_enabled ? ArtifactCache::ComputeKey({msl_source, muda::kVersion, "metal3.2"})
                    : std::string{};

  bool cache_hit = false;
  if (cache_enabled && cache.CopyCachedMetallibTo(cache_key, metallib_path)) {
    cache_hit = true;
  } else {
    const std::string metal_path_str = metal_path.string();
    const std::string air_path_str = air_path.string();
    const std::string metallib_path_str = metallib_path.string();

    const std::array<std::string_view, 9> compile_args = {
        "xcrun", "--sdk",        "macosx", "metal",     "-std=metal3.2",
        "-c",    metal_path_str, "-o",     air_path_str};
    if (!detail::RunProcess(compile_args).succeeded()) {
      return Failure(DiagnosticCode::kMetalCompilationFailed);
    }

    const std::array<std::string_view, 7> link_args = {
        "xcrun", "--sdk", "macosx", "metallib", air_path_str, "-o", metallib_path_str};
    if (!detail::RunProcess(link_args).succeeded()) {
      RemoveFile(metallib_path);
      return Failure(DiagnosticCode::kMetallibLinkFailed);
    }

    if (cache_enabled) {
      std::vector<std::uint8_t> compiled_bytes;
      if (ReadBinaryFile(metallib_path, &compiled_bytes)) {
        (void)cache.Store(cache_key, compiled_bytes);
      }
    }
  }

  if (output_kind == OutputKind::kMetallib) {
    CompileResult result;
    result.status = Status::kOk;
    result.code = DiagnosticCode::kOk;
    result.output_path = metallib_path;
    result.metallib_path = metallib_path;
    result.kernel_name = std::move(translation.kernel.name);
    result.cache_hit = cache_hit;
    return result;
  }

  std::vector<std::uint8_t> metallib_bytes;
  if (!ReadBinaryFile(metallib_path, &metallib_bytes)) {
    return Failure(DiagnosticCode::kMetallibLinkFailed);
  }

  std::vector<TranslatedKernel> kernels_list = translation.kernels;
  if (kernels_list.empty()) {
    kernels_list.push_back(translation.kernel);
  }

  const HostRewriteResult host_rewritten =
      RewriteHostCudaToCpp(source, kernels_list, metallib_bytes);
  if (!host_rewritten.ok()) {
    return Failure(host_rewritten.code);
  }

  if (!WriteTextFile(host_cpp_path, host_rewritten.host_cpp)) {
    return Failure(DiagnosticCode::kCannotWriteMetalSource);
  }

  std::vector<std::string> host_args = {"xcrun", "clang++", "-std=c++20", "-O2"};
#if defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(undefined_behavior_sanitizer)
  host_args.emplace_back("-fsanitize=address,undefined");
#endif
#endif
#ifdef MUDA_DEFAULT_RUNTIME_INCLUDE_DIR
  host_args.emplace_back("-I");
  host_args.emplace_back(MUDA_DEFAULT_RUNTIME_INCLUDE_DIR);
#endif
  if (options.input.has_parent_path()) {
    host_args.emplace_back("-I");
    host_args.emplace_back(options.input.parent_path().string());
  }
  for (const std::filesystem::path& inc : options.include_dirs) {
    host_args.emplace_back("-I");
    host_args.emplace_back(inc.string());
  }

  if (output_kind == OutputKind::kObject) {
    host_args.emplace_back("-c");
    host_args.emplace_back(host_cpp_path.string());
    host_args.emplace_back("-o");
    host_args.emplace_back(output.string());
    if (!RunProcessOwned(host_args)) {
      RemoveFile(output);
      return Failure(DiagnosticCode::kHostCompilationFailed);
    }
  } else {
    host_args.emplace_back(host_cpp_path.string());
#ifdef MUDA_DEFAULT_RUNTIME_LIB_FILE
    host_args.emplace_back(MUDA_DEFAULT_RUNTIME_LIB_FILE);
#endif
    for (const std::filesystem::path& lib_dir : options.library_dirs) {
      host_args.emplace_back("-L" + lib_dir.string());
    }
    for (const std::string& lib : options.libraries) {
      host_args.emplace_back("-l" + lib);
    }
    host_args.emplace_back("-framework");
    host_args.emplace_back("Foundation");
    host_args.emplace_back("-framework");
    host_args.emplace_back("Metal");
    host_args.emplace_back("-o");
    host_args.emplace_back(output.string());
    if (!RunProcessOwned(host_args)) {
      RemoveFile(output);
      return Failure(DiagnosticCode::kHostLinkFailed);
    }
  }

  CompileResult result;
  result.status = Status::kOk;
  result.code = DiagnosticCode::kOk;
  result.output_path = output;
  result.metallib_path = (options.intermediates == IntermediatePolicy::kKeep)
                             ? metallib_path
                             : std::filesystem::path{};
  result.kernel_name = std::move(translation.kernel.name);
  result.cache_hit = cache_hit;
  return result;
}

}  // namespace muda::compiler

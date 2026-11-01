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

CompileResult Success(std::filesystem::path output_path,
                      std::filesystem::path metallib_path,
                      std::string kernel_name,
                      bool cache_hit = false) {
  CompileResult result;
  result.status = Status::kOk;
  result.code = DiagnosticCode::kOk;
  result.output_path = std::move(output_path);
  result.metallib_path = std::move(metallib_path);
  result.kernel_name = std::move(kernel_name);
  result.cache_hit = cache_hit;
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

bool ReadSourceFile(const std::filesystem::path& path, std::string* out_source) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return false;
  }
  std::ostringstream source_buffer;
  source_buffer << input.rdbuf();
  *out_source = source_buffer.str();
  return true;
}

bool EnsureOutputDirectoryExists(const std::filesystem::path& output) {
  if (!output.has_parent_path() || output.parent_path().empty()) {
    return true;
  }
  std::error_code fs_error;
  std::filesystem::create_directories(output.parent_path(), fs_error);
  return !fs_error;
}

std::string_view SelectMslSource(const TranslationResult& translation) noexcept {
  if (!translation.msl.empty()) {
    return translation.msl;
  }
  return translation.kernel.msl;
}

struct IntermediateArtifacts {
  std::filesystem::path metal_path;
  std::filesystem::path air_path;
  std::filesystem::path metallib_path;
  std::filesystem::path host_cpp_path;
};

IntermediateArtifacts MakeIntermediateArtifacts(const std::filesystem::path& output,
                                                OutputKind output_kind,
                                                IntermediatePolicy policy) {
  const std::filesystem::path base_dir = ResolveBaseDirectory(output, policy);
  const std::string stem = MakeIntermediateStem(output, policy);
  IntermediateArtifacts artifacts;
  artifacts.metal_path = base_dir / (stem + ".metal");
  artifacts.air_path = base_dir / (stem + ".air");
  artifacts.metallib_path =
      (output_kind == OutputKind::kMetallib) ? output : (base_dir / (stem + ".metallib"));
  artifacts.host_cpp_path = base_dir / (stem + ".host.cpp");
  return artifacts;
}

ScopedIntermediateCleaner MakeIntermediateCleaner(const IntermediateArtifacts& artifacts,
                                                  OutputKind output_kind,
                                                  IntermediatePolicy policy) {
  std::vector<std::filesystem::path> temp_files = {artifacts.metal_path, artifacts.air_path};
  if (output_kind != OutputKind::kMetallib) {
    temp_files.push_back(artifacts.metallib_path);
    temp_files.push_back(artifacts.host_cpp_path);
  }
  return {std::move(temp_files), policy};
}

bool IsCacheEnabled(const CompileOptions& options) noexcept {
  if (options.disable_cache || options.intermediates != IntermediatePolicy::kRemove) {
    return false;
  }
  const char* env_disable = std::getenv("MUDA_DISABLE_CACHE");
  if (env_disable == nullptr || env_disable[0] == '\0') {
    return true;
  }
  return std::string_view(env_disable) == "0";
}

bool CompileMetalToAir(const std::filesystem::path& metal_path,
                       const std::filesystem::path& air_path,
                       const CompileOptions& options) {
  const std::string metal_path_str = metal_path.string();
  const std::string air_path_str = air_path.string();
  const std::string_view std_flag = MetalStandardFlag(options.metal_standard);
  const std::string_view opt_flag = OptimizationLevelFlag(options.optimization);
  const std::array<std::string_view, 10> compile_args = {
      "xcrun",  "--sdk", "macosx",       "metal", std_flag,
      opt_flag, "-c",    metal_path_str, "-o",    air_path_str};
  return detail::RunProcess(compile_args).succeeded();
}

bool LinkAirToMetallib(const std::filesystem::path& air_path,
                       const std::filesystem::path& metallib_path) {
  const std::string air_path_str = air_path.string();
  const std::string metallib_path_str = metallib_path.string();
  const std::array<std::string_view, 7> link_args = {
      "xcrun", "--sdk", "macosx", "metallib", air_path_str, "-o", metallib_path_str};
  return detail::RunProcess(link_args).succeeded();
}

struct BuildMetallibResult {
  DiagnosticCode code{DiagnosticCode::kOk};
  bool cache_hit{false};

  [[nodiscard]] bool ok() const noexcept { return code == DiagnosticCode::kOk; }
};

BuildMetallibResult BuildOrLoadMetallib(const IntermediateArtifacts& artifacts,
                                       std::string_view msl_source,
                                       const CompileOptions& options) {
  const bool cache_enabled = IsCacheEnabled(options);
  const std::filesystem::path cache_root =
      !options.cache_dir.empty() ? options.cache_dir : ArtifactCache::DefaultCacheDirectory();
  const ArtifactCache cache(cache_root);

  std::string cache_key;
  if (cache_enabled) {
    const CacheKeyInput key_input{
        .msl_source = msl_source,
        .muda_version = muda::kVersion,
        .metal_standard = options.metal_standard,
        .optimization = options.optimization,
    };
    cache_key = ArtifactCache::ComputeKey(key_input);
    if (cache.CopyCachedMetallibTo(cache_key, artifacts.metallib_path)) {
      return BuildMetallibResult{.code = DiagnosticCode::kOk, .cache_hit = true};
    }
  }

  if (!CompileMetalToAir(artifacts.metal_path, artifacts.air_path, options)) {
    return BuildMetallibResult{.code = DiagnosticCode::kMetalCompilationFailed,
                               .cache_hit = false};
  }

  if (!LinkAirToMetallib(artifacts.air_path, artifacts.metallib_path)) {
    RemoveFile(artifacts.metallib_path);
    return BuildMetallibResult{.code = DiagnosticCode::kMetallibLinkFailed,
                               .cache_hit = false};
  }

  if (cache_enabled) {
    std::vector<std::uint8_t> compiled_bytes;
    if (ReadBinaryFile(artifacts.metallib_path, &compiled_bytes)) {
      (void)cache.Store(cache_key, compiled_bytes);
    }
  }

  return BuildMetallibResult{.code = DiagnosticCode::kOk, .cache_hit = false};
}

DiagnosticCode PrepareHostSource(const std::string& source,
                                 const TranslationResult& translation,
                                 const IntermediateArtifacts& artifacts) {
  std::vector<std::uint8_t> metallib_bytes;
  if (!ReadBinaryFile(artifacts.metallib_path, &metallib_bytes)) {
    return DiagnosticCode::kMetallibLinkFailed;
  }

  std::vector<TranslatedKernel> kernels_list = translation.kernels;
  if (kernels_list.empty()) {
    kernels_list.push_back(translation.kernel);
  }

  const HostRewriteResult host_rewritten =
      RewriteHostCudaToCpp(source, kernels_list, metallib_bytes);
  if (!host_rewritten.ok()) {
    return host_rewritten.code;
  }

  if (!WriteTextFile(artifacts.host_cpp_path, host_rewritten.host_cpp)) {
    return DiagnosticCode::kCannotWriteMetalSource;
  }

  return DiagnosticCode::kOk;
}

std::vector<std::string> BuildHostCompilerArgs(const CompileOptions& options,
                                               OutputKind output_kind,
                                               const IntermediateArtifacts& artifacts,
                                               const std::filesystem::path& output) {
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
    host_args.emplace_back(artifacts.host_cpp_path.string());
    host_args.emplace_back("-o");
    host_args.emplace_back(output.string());
  } else {
    host_args.emplace_back(artifacts.host_cpp_path.string());
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
  }
  return host_args;
}

DiagnosticCode CompileHostCode(const CompileOptions& options,
                               OutputKind output_kind,
                               const IntermediateArtifacts& artifacts,
                               const std::filesystem::path& output) {
  const std::vector<std::string> host_args =
      BuildHostCompilerArgs(options, output_kind, artifacts, output);
  if (!RunProcessOwned(host_args)) {
    RemoveFile(output);
    return (output_kind == OutputKind::kObject) ? DiagnosticCode::kHostCompilationFailed
                                                : DiagnosticCode::kHostLinkFailed;
  }
  return DiagnosticCode::kOk;
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

  std::string source;
  if (!ReadSourceFile(options.input, &source)) {
    return Failure(DiagnosticCode::kCannotOpenInput);
  }

  TranslationResult translation = TranslateCudaToMsl(source);
  if (!translation.ok()) {
    return Failure(translation.code);
  }

  const std::filesystem::path output = ResolveOutputPath(options, output_kind);
  if (!EnsureOutputDirectoryExists(output)) {
    return Failure(DiagnosticCode::kCannotCreateOutputDir);
  }

  const std::string_view msl_source = SelectMslSource(translation);

  if (output_kind == OutputKind::kMslSource) {
    if (!WriteTextFile(output, msl_source)) {
      return Failure(DiagnosticCode::kCannotWriteMetalSource);
    }
    return Success(output, {}, std::move(translation.kernel.name));
  }

  const IntermediateArtifacts artifacts =
      MakeIntermediateArtifacts(output, output_kind, options.intermediates);
  const ScopedIntermediateCleaner cleaner =
      MakeIntermediateCleaner(artifacts, output_kind, options.intermediates);

  if (!WriteTextFile(artifacts.metal_path, msl_source)) {
    return Failure(DiagnosticCode::kCannotWriteMetalSource);
  }

  const BuildMetallibResult metallib_result =
      BuildOrLoadMetallib(artifacts, msl_source, options);
  if (!metallib_result.ok()) {
    return Failure(metallib_result.code);
  }

  if (output_kind == OutputKind::kMetallib) {
    return Success(artifacts.metallib_path, artifacts.metallib_path,
                   std::move(translation.kernel.name), metallib_result.cache_hit);
  }

  const DiagnosticCode host_prep_code =
      PrepareHostSource(source, translation, artifacts);
  if (host_prep_code != DiagnosticCode::kOk) {
    return Failure(host_prep_code);
  }

  const DiagnosticCode host_compile_code =
      CompileHostCode(options, output_kind, artifacts, output);
  if (host_compile_code != DiagnosticCode::kOk) {
    return Failure(host_compile_code);
  }

  const std::filesystem::path metallib_output =
      (options.intermediates == IntermediatePolicy::kKeep) ? artifacts.metallib_path
                                                           : std::filesystem::path{};
  return Success(output, metallib_output, std::move(translation.kernel.name),
                 metallib_result.cache_hit);
}

}  // namespace muda::compiler

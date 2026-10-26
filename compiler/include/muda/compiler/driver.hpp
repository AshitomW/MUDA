#pragma once

#include <filesystem>
#include <string>

namespace muda::compiler {

enum class Status {
  kOk,
  kNotImplemented,
  kError,
};

struct CompileOptions {
  std::filesystem::path input;
  std::filesystem::path output;
};

struct CompileResult {
  Status status = Status::kNotImplemented;
  std::string message;
};

[[nodiscard]] CompileResult Compile(const CompileOptions& options);

}

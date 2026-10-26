#include "muda/compiler/driver.hpp"

#include <iostream>

int main() {
  muda::compiler::CompileOptions options;
  options.input = "vector_add.cu";

  const muda::compiler::CompileResult result = muda::compiler::Compile(options);
  if (result.status != muda::compiler::Status::kNotImplemented) {
    std::cerr << "unexpected compiler status\n";
    return 1;
  }
  if (result.message.empty()) {
    std::cerr << "empty compiler message\n";
    return 1;
  }
  return 0;
}

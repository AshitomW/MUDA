#include "muda/compiler/driver.hpp"

namespace muda::compiler {

CompileResult Compile(const CompileOptions& options) {
  (void)options;
  return {Status::kNotImplemented, "compiler pipeline is not implemented yet"};
}

}

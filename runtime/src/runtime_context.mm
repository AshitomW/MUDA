/**
 * @file runtime_context.mm
 * @brief Singleton instance lifecycle for the MUDA RuntimeContext.
 */

#include "runtime_context.hpp"

#include <cstddef>
#include <mutex>
#include <new>

namespace muda::runtime::cudart {

RuntimeContext& RuntimeContext::Instance() noexcept {
  static std::once_flag init_flag;
  alignas(RuntimeContext) static std::byte storage[sizeof(RuntimeContext)];
  static RuntimeContext* instance = nullptr;
  std::call_once(init_flag, [&]() { instance = new (storage) RuntimeContext(); });
  return *instance;
}

}  // namespace muda::runtime::cudart


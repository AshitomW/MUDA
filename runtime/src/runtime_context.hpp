/**
 * @file runtime_context.hpp
 * @brief Global singleton coordinating runtime managers (device, memory, streams, events,
 * launches).
 */

#pragma once

#import <Metal/Metal.h>

#include "cuda_runtime.h"
#include "device_manager.hpp"
#include "error_state.hpp"
#include "event_manager.hpp"
#include "launch_engine.hpp"
#include "memory_manager.hpp"
#include "stream_manager.hpp"
#include "transfer_manager.hpp"

namespace muda::runtime::cudart {

class RuntimeContext {
 public:
  static RuntimeContext& Instance() noexcept;

  cudaError_t EnsureInitialized() noexcept { return devices_.EnsureInitialized(); }
  [[nodiscard]] id<MTLDevice> Device() const noexcept { return devices_.Device(); }
  [[nodiscard]] id<MTLCommandQueue> DefaultQueue() const noexcept {
    return devices_.DefaultQueue();
  }

  [[nodiscard]] DeviceManager& Devices() noexcept { return devices_; }
  [[nodiscard]] MemoryManager& Memory() noexcept { return memory_; }
  [[nodiscard]] StreamManager& Streams() noexcept { return streams_; }
  [[nodiscard]] EventManager& Events() noexcept { return events_; }
  [[nodiscard]] TransferManager& Transfers() noexcept { return transfers_; }
  [[nodiscard]] LaunchEngine& Launcher() noexcept { return launcher_; }

  cudaError_t Reset() noexcept {
    streams_.Reset();
    events_.Reset();
    memory_.Reset();
    launcher_.Clear();
    devices_.Reset();
    return cudaSuccess;
  }

 private:
  RuntimeContext() = default;

  DeviceManager devices_;
  MemoryManager memory_;
  StreamManager streams_;
  EventManager events_;
  TransferManager transfers_;
  LaunchEngine launcher_;
};

}  // namespace muda::runtime::cudart

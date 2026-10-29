/**
 * @file main.cpp
 * @brief Implements the MUDA environment and version CLI.
 */

#include <iostream>
#include <string_view>
#include <vector>

#include "muda/compiler/toolchain.hpp"
#include "muda/runtime/device.hpp"
#include "muda/version.hpp"

namespace {

void PrintUsage(std::ostream& out) {
  out << "muda " << muda::kVersion << "\n"
      << "Usage: muda <command>\n"
      << "\n"
      << "Commands:\n"
      << "  doctor     Inspect the Metal environment\n"
      << "  version    Show version\n";
}

int Doctor() {
  const std::vector<muda::runtime::DeviceInfo> devices = muda::runtime::EnumerateDevices();
  const bool toolchain = muda::compiler::MetalToolchainAvailable();

  for (const muda::runtime::DeviceInfo& device : devices) {
    std::cout << "device: " << device.name << "\n"
              << "  backend: " << muda::runtime::BackendName(device.backend) << "\n"
              << "  working set: " << device.recommended_working_set_bytes << " bytes\n"
              << "  max threads per threadgroup: " << device.max_threads_per_threadgroup << "\n";
  }
  std::cout << "metal toolchain: " << (toolchain ? "available" : "missing") << "\n";

  if (devices.empty()) {
    std::cerr << "muda: no Metal devices found\n";
    return 1;
  }
  return toolchain ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    PrintUsage(std::cerr);
    return 2;
  }

  const std::string_view command = argv[1];
  if (command == "doctor") {
    return Doctor();
  }
  if (command == "version" || command == "--version") {
    std::cout << "MUDA " << muda::kVersion << "\n";
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

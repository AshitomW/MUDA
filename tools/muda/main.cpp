#include "muda/runtime/device.hpp"
#include "muda/version.hpp"

#include <iostream>
#include <string_view>
#include <vector>

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
  if (devices.empty()) {
    std::cerr << "muda: no Metal devices found\n";
    return 1;
  }

  for (const muda::runtime::DeviceInfo& device : devices) {
    std::cout << "device: " << device.name << "\n"
              << "  backend: " << device.backend << "\n"
              << "  working set: " << device.recommended_working_set_bytes << " bytes\n"
              << "  max threads per threadgroup: " << device.max_threads_per_threadgroup << "\n";
  }

  return 0;
}

}

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

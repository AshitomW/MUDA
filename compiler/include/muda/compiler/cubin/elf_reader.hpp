/**
 * @file elf_reader.hpp
 * @brief Clean-room ELF64 reader and .nv.info attribute extractor for NVIDIA CUBIN binaries.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace muda::compiler::cubin {

inline constexpr std::uint16_t kElfMachineCuda = 190;  // EM_CUDA

struct CubinKernelParam {
  std::size_t index = 0;
  std::size_t offset = 0;  // Offset in constant memory bank 0 (c[0x0][offset])
  std::size_t size = 4;    // Size in bytes (4 for int/float, 8 for ptr/double)
  bool is_pointer = false;
};

struct CubinKernel {
  std::string name;
  std::uint32_t arch_sm = 0;  // e.g. 70, 75, 80, 86, 90
  std::span<const std::uint8_t> text_bytes;  // Raw SASS machine instructions
  std::vector<CubinKernelParam> params;
  std::size_t static_shared_bytes = 0;
  std::size_t max_threads_per_block = 1024;
  std::size_t register_count = 32;
};

class ElfReader final {
 public:
  /**
   * @brief Checks if buffer begins with standard ELF magic (0x7f 'E' 'L' 'F').
   */
  [[nodiscard]] static bool IsElf(std::span<const std::uint8_t> data) noexcept;

  /**
   * @brief Checks if buffer is a 64-bit CUDA ELF binary (e_machine == 190).
   */
  [[nodiscard]] static bool IsCudaCubin(std::span<const std::uint8_t> data) noexcept;

  /**
   * @brief Extracts the CUDA compute capability architecture (SM) from the ELF header flags.
   */
  [[nodiscard]] static std::uint32_t ExtractSmArch(std::span<const std::uint8_t> data) noexcept;

  /**
   * @brief Parses the ELF container, locates all kernels, reads .nv.info attributes, and extracts
   * SASS machine instructions and parameter layouts.
   */
  [[nodiscard]] static std::vector<CubinKernel> ExtractKernels(std::span<const std::uint8_t> data);
};

}  // namespace muda::compiler::cubin


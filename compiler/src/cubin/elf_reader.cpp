/**
 * @file elf_reader.cpp
 * @brief Clean-room ELF64 reader and .nv.info attribute extractor for NVIDIA CUBIN binaries.
 */

#include "muda/compiler/cubin/elf_reader.hpp"

#include <algorithm>
#include <cstring>
#include <unordered_map>

namespace muda::compiler::cubin {
namespace {

#pragma pack(push, 1)
struct Elf64Header {
  std::uint8_t ident[16];
  std::uint16_t type;
  std::uint16_t machine;
  std::uint32_t version;
  std::uint64_t entry;
  std::uint64_t phoff;
  std::uint64_t shoff;
  std::uint32_t flags;
  std::uint16_t ehsize;
  std::uint16_t phentsize;
  std::uint16_t phnum;
  std::uint16_t shentsize;
  std::uint16_t shnum;
  std::uint16_t shstrndx;
};

struct Elf64SectionHeader {
  std::uint32_t name_offset;
  std::uint32_t type;
  std::uint64_t flags;
  std::uint64_t addr;
  std::uint64_t offset;
  std::uint64_t size;
  std::uint32_t link;
  std::uint32_t info;
  std::uint64_t addralign;
  std::uint64_t entsize;
};

struct Elf64Symbol {
  std::uint32_t name_offset;
  std::uint8_t info;
  std::uint8_t other;
  std::uint16_t shndx;
  std::uint64_t value;
  std::uint64_t size;
};
#pragma pack(pop)

// Attribute Tag Constants in .nv.info
inline constexpr std::uint16_t kEiattrSharedSize16 = 0x0104;
inline constexpr std::uint16_t kEiattrMaxregCount16 = 0x0204;
inline constexpr std::uint16_t kEiattrMaxThreads16 = 0x0504;
inline constexpr std::uint16_t kEiattrKparamInfo16 = 0x1704;

inline constexpr std::uint8_t kEiattrSharedSize8 = 0x01;
inline constexpr std::uint8_t kEiattrMaxregCount8 = 0x02;
inline constexpr std::uint8_t kEiattrKparamInfo8 = 0x17;

struct RawSection {
  std::string_view name;
  std::uint32_t type = 0;
  std::uint64_t offset = 0;
  std::uint64_t size = 0;
  std::uint32_t link = 0;
  std::uint32_t info = 0;
  std::span<const std::uint8_t> data;
};

void ParseNvInfoAttributes(std::span<const std::uint8_t> data, CubinKernel* kernel) {
  if (data.size() < 4 || kernel == nullptr) {
    return;
  }

  std::size_t offset = 0;
  while (offset + 4 <= data.size()) {
    std::uint16_t tag = 0;
    std::memcpy(&tag, data.data() + offset, 2);
    std::uint16_t attr_size = 0;
    std::memcpy(&attr_size, data.data() + offset + 2, 2);

    offset += 4;
    if (offset + attr_size > data.size()) {
      break;
    }

    const std::uint8_t* payload = data.data() + offset;

    // Check 16-bit or 8-bit tag matches
    if (tag == kEiattrKparamInfo16 || (tag & 0xFF) == kEiattrKparamInfo8) {
      if (attr_size >= 12) {
        std::uint32_t param_idx = 0;
        std::uint32_t param_offset = 0;
        std::uint32_t param_size = 0;
        std::memcpy(&param_idx, payload, 4);
        std::memcpy(&param_offset, payload + 4, 4);
        std::memcpy(&param_size, payload + 8, 4);

        CubinKernelParam param;
        param.index = param_idx;
        param.offset = param_offset;
        param.size = (param_size > 0) ? param_size : 4;
        param.is_pointer = (param.size == 8);

        // Deduplicate or insert
        auto it = std::find_if(kernel->params.begin(), kernel->params.end(),
                               [param_idx](const CubinKernelParam& p) { return p.index == param_idx; });
        if (it != kernel->params.end()) {
          *it = param;
        } else {
          kernel->params.push_back(param);
        }
      }
    } else if (tag == kEiattrSharedSize16 || (tag & 0xFF) == kEiattrSharedSize8) {
      if (attr_size >= 4) {
        std::uint32_t smem = 0;
        std::memcpy(&smem, payload, 4);
        kernel->static_shared_bytes = smem;
      }
    } else if (tag == kEiattrMaxregCount16 || (tag & 0xFF) == kEiattrMaxregCount8) {
      if (attr_size >= 4) {
        std::uint32_t regs = 0;
        std::memcpy(&regs, payload, 4);
        kernel->register_count = regs;
      }
    } else if (tag == kEiattrMaxThreads16) {
      if (attr_size >= 4) {
        std::uint32_t threads = 0;
        std::memcpy(&threads, payload, 4);
        kernel->max_threads_per_block = threads;
      }
    }

    offset += attr_size;
  }

  // Sort parameters by index
  std::sort(kernel->params.begin(), kernel->params.end(),
            [](const CubinKernelParam& a, const CubinKernelParam& b) { return a.index < b.index; });
}

}  // namespace

bool ElfReader::IsElf(std::span<const std::uint8_t> data) noexcept {
  return data.size() >= 4 && data[0] == 0x7F && data[1] == 'E' && data[2] == 'L' && data[3] == 'F';
}

bool ElfReader::IsCudaCubin(std::span<const std::uint8_t> data) noexcept {
  if (data.size() < sizeof(Elf64Header) || !IsElf(data)) {
    return false;
  }
  const auto* header = reinterpret_cast<const Elf64Header*>(data.data());
  return header->ident[4] == 2 /* ELFCLASS64 */ && header->machine == kElfMachineCuda;
}

std::uint32_t ElfReader::ExtractSmArch(std::span<const std::uint8_t> data) noexcept {
  if (!IsCudaCubin(data)) {
    return 0;
  }
  const auto* header = reinterpret_cast<const Elf64Header*>(data.data());
  return header->flags & 0xFF;
}

std::vector<CubinKernel> ElfReader::ExtractKernels(std::span<const std::uint8_t> data) {
  std::vector<CubinKernel> kernels;
  if (!IsCudaCubin(data)) {
    return kernels;
  }

  const auto* header = reinterpret_cast<const Elf64Header*>(data.data());
  const std::uint32_t arch_sm = header->flags & 0xFF;

  if (header->shoff == 0 || header->shnum == 0 || header->shentsize < sizeof(Elf64SectionHeader)) {
    return kernels;
  }

  const std::size_t sh_table_end = header->shoff + static_cast<std::size_t>(header->shnum) * header->shentsize;
  if (sh_table_end > data.size()) {
    return kernels;
  }

  // Parse Section Header String Table
  if (header->shstrndx >= header->shnum) {
    return kernels;
  }

  const auto* strtab_hdr = reinterpret_cast<const Elf64SectionHeader*>(
      data.data() + header->shoff + static_cast<std::size_t>(header->shstrndx) * header->shentsize);
  if (strtab_hdr->offset + strtab_hdr->size > data.size()) {
    return kernels;
  }

  const char* shstrtab = reinterpret_cast<const char*>(data.data() + strtab_hdr->offset);
  const std::size_t shstrtab_size = static_cast<std::size_t>(strtab_hdr->size);

  std::vector<RawSection> sections;
  sections.reserve(header->shnum);

  for (std::size_t i = 0; i < header->shnum; ++i) {
    const auto* shdr = reinterpret_cast<const Elf64SectionHeader*>(
        data.data() + header->shoff + i * header->shentsize);

    RawSection sec;
    sec.type = shdr->type;
    sec.offset = shdr->offset;
    sec.size = shdr->size;
    sec.link = shdr->link;
    sec.info = shdr->info;

    if (shdr->name_offset < shstrtab_size) {
      sec.name = std::string_view(shstrtab + shdr->name_offset);
    }
    if (shdr->offset + shdr->size <= data.size()) {
      sec.data = data.subspan(static_cast<std::size_t>(shdr->offset), static_cast<std::size_t>(shdr->size));
    }
    sections.push_back(sec);
  }

  // Map .nv.info sections by kernel suffix (e.g. .nv.info._Z9vectorAddPKfS0_Pfi)
  std::unordered_map<std::string_view, std::span<const std::uint8_t>> nv_info_map;
  std::span<const std::uint8_t> global_nv_info;

  for (const RawSection& sec : sections) {
    if (sec.name == ".nv.info") {
      global_nv_info = sec.data;
    } else if (sec.name.starts_with(".nv.info.")) {
      std::string_view k_name = sec.name.substr(9);
      nv_info_map[k_name] = sec.data;
    }
  }

  // Find all .text sections (.text.<kernel_name> or .text)
  for (const RawSection& sec : sections) {
    if (sec.name.starts_with(".text.") && !sec.data.empty()) {
      std::string_view k_name = sec.name.substr(6);
      CubinKernel k;
      k.name = std::string(k_name);
      k.arch_sm = arch_sm;
      k.text_bytes = sec.data;

      // Parse kernel-specific .nv.info
      auto it = nv_info_map.find(k_name);
      if (it != nv_info_map.end()) {
        ParseNvInfoAttributes(it->second, &k);
      }
      if (!global_nv_info.empty()) {
        ParseNvInfoAttributes(global_nv_info, &k);
      }

      kernels.push_back(std::move(k));
    }
  }

  // If no .text.<name> sections were found, check symbol table for STT_FUNC symbols
  if (kernels.empty()) {
    for (const RawSection& sec : sections) {
      if (sec.type == 2 /* SHT_SYMTAB */ && sec.link < sections.size()) {
        const RawSection& strtab_sec = sections[sec.link];
        const char* sym_strtab = reinterpret_cast<const char*>(strtab_sec.data.data());
        const std::size_t sym_count = sec.size / sizeof(Elf64Symbol);

        for (std::size_t s = 0; s < sym_count; ++s) {
          const auto* sym = reinterpret_cast<const Elf64Symbol*>(sec.data.data() + s * sizeof(Elf64Symbol));
          const std::uint8_t sym_type = sym->info & 0x0F;
          if (sym_type == 2 /* STT_FUNC */ && sym->shndx < sections.size() && sym->size > 0) {
            std::string_view fn_name;
            if (sym->name_offset < strtab_sec.size) {
              fn_name = std::string_view(sym_strtab + sym->name_offset);
            }
            const RawSection& target_sec = sections[sym->shndx];
            if (sym->value + sym->size <= target_sec.size) {
              CubinKernel k;
              k.name = std::string(fn_name);
              k.arch_sm = arch_sm;
              k.text_bytes = target_sec.data.subspan(static_cast<std::size_t>(sym->value),
                                                     static_cast<std::size_t>(sym->size));

              auto it = nv_info_map.find(fn_name);
              if (it != nv_info_map.end()) {
                ParseNvInfoAttributes(it->second, &k);
              }
              if (!global_nv_info.empty()) {
                ParseNvInfoAttributes(global_nv_info, &k);
              }

              kernels.push_back(std::move(k));
            }
          }
        }
      }
    }
  }

  return kernels;
}

}  // namespace muda::compiler::cubin

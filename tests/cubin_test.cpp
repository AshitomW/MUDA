/**
 * @file cubin_test.cpp
 * @brief Verification test suite for Fatbin extraction, ELF64 CUBIN parsing, SASS decoding, and execution.
 */

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string_view>
#include <vector>

#include "cuda.h"
#include "muda/compiler/cubin/cubin_translator.hpp"
#include "muda/compiler/cubin/elf_reader.hpp"
#include "muda/compiler/cubin/fatbin.hpp"
#include "muda/compiler/cubin/sass_decoder.hpp"
#include "muda/compiler/cubin/sass_lifter.hpp"

namespace {

bool Expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    return false;
  }
  return true;
}

bool ExpectCuSuccess(CUresult res, const char* message) {
  if (res != CUDA_SUCCESS) {
    const char* err_name = "UNKNOWN";
    cuGetErrorName(res, &err_name);
    std::cerr << "FAIL: " << message << " (error: " << err_name << " / " << res << ")\n";
    return false;
  }
  return true;
}

constexpr std::string_view kSamplePtx = R"PTX(
.version 7.0
.target sm_75
.address_size 64

.visible .entry vec_add_ptx(
    .param .u64 param_a,
    .param .u64 param_b,
    .param .u64 param_c,
    .param .u32 param_n
)
{
    .reg .pred  %p<2>;
    .reg .b32   %r<5>;
    .reg .b64   %rd<9>;
    .reg .f32   %f<4>;

    ld.param.u64    %rd1, [param_a];
    ld.param.u64    %rd2, [param_b];
    ld.param.u64    %rd3, [param_c];
    ld.param.u32    %r1, [param_n];

    mov.u32         %r2, %ctaid.x;
    mov.u32         %r3, %ntid.x;
    mov.u32         %r4, %tid.x;
    mad.lo.s32      %r0, %r2, %r3, %r4;

    setp.ge.s32     %p1, %r0, %r1;
    @%p1 bra        L_DONE;

    mul.wide.s32    %rd4, %r0, 4;
    add.s64         %rd5, %rd1, %rd4;
    add.s64         %rd6, %rd2, %rd4;
    add.s64         %rd7, %rd3, %rd4;

    ld.global.f32   %f1, [%rd5];
    ld.global.f32   %f2, [%rd6];
    add.f32         %f3, %f1, %f2;
    st.global.f32   [%rd7], %f3;

L_DONE:
    ret;
}
)PTX";

// Helper to construct a synthetic 64-bit Fatbinary carrying PTX text
std::vector<std::uint8_t> CreateSyntheticFatbin64(std::string_view ptx_text, std::uint32_t arch_sm = 75) {
  std::vector<std::uint8_t> buffer;
  const std::uint16_t hdr_size = 16;
  const std::uint16_t entry_hdr_size = 24;
  const std::uint64_t payload_size = ptx_text.size() + 1;
  const std::uint64_t total_size = hdr_size + entry_hdr_size + payload_size;

  buffer.resize(total_size, 0);

  // 1. Fatbin Header (16 bytes)
  const std::uint32_t magic = muda::compiler::cubin::kFatbinMagic64;
  const std::uint16_t version = 1;
  std::memcpy(buffer.data(), &magic, 4);
  std::memcpy(buffer.data() + 4, &version, 2);
  std::memcpy(buffer.data() + 6, &hdr_size, 2);
  std::memcpy(buffer.data() + 8, &total_size, 8);

  // 2. Entry Record Header (24 bytes)
  const std::uint16_t entry_kind = 0x0001;  // PTX
  std::memcpy(buffer.data() + hdr_size, &entry_kind, 2);
  std::memcpy(buffer.data() + hdr_size + 2, &entry_hdr_size, 2);
  std::memcpy(buffer.data() + hdr_size + 8, &payload_size, 8);
  std::memcpy(buffer.data() + hdr_size + 16, &arch_sm, 4);

  // 3. Payload
  std::memcpy(buffer.data() + hdr_size + entry_hdr_size, ptx_text.data(), ptx_text.size());
  buffer.back() = '\0';

  return buffer;
}

// Helper to construct a synthetic ELF64 CUBIN with a .nv.info parameter section and SASS instructions
std::vector<std::uint8_t> CreateSyntheticCubinElf(const std::vector<std::uint8_t>& sass_bytecode,
                                                  std::uint32_t arch_sm = 80) {
  // We build a minimal conforming ELF64 binary with:
  // - Ehdr (64 bytes)
  // - .shstrtab (section names)
  // - .nv.info (parameter metadata)
  // - .text.k_vector_add (SASS bytecode)
  // - Section Header Table (4 headers: NULL, .shstrtab, .nv.info, .text.k_vector_add)

  std::vector<std::uint8_t> buffer;
  const std::uint16_t ehdr_size = 64;
  const std::uint16_t shent_size = 64;
  const std::uint16_t shnum = 4;

  constexpr char kShstrtabData[] = "\0.shstrtab\0.nv.info\0.text.k_vector_add\0";
  const std::size_t shstrtab_offset = ehdr_size;
  const std::size_t shstrtab_size = sizeof(kShstrtabData);

  // Construct .nv.info attribute stream:
  // Param 0: offset 0x160, size 8 (ptr)
  // Param 1: offset 0x168, size 8 (ptr)
  // Param 2: offset 0x170, size 8 (ptr)
  // Param 3: offset 0x178, size 4 (scalar int)
  std::vector<std::uint8_t> nv_info;
  auto add_kparam = [&](std::uint32_t idx, std::uint32_t offset, std::uint32_t size) {
    const std::uint16_t tag = 0x1704;
    const std::uint16_t attr_size = 12;
    const std::size_t cur = nv_info.size();
    nv_info.resize(cur + 4 + attr_size, 0);
    std::memcpy(nv_info.data() + cur, &tag, 2);
    std::memcpy(nv_info.data() + cur + 2, &attr_size, 2);
    std::memcpy(nv_info.data() + cur + 4, &idx, 4);
    std::memcpy(nv_info.data() + cur + 8, &offset, 4);
    std::memcpy(nv_info.data() + cur + 12, &size, 4);
  };
  add_kparam(0, 0x160, 8);
  add_kparam(1, 0x168, 8);
  add_kparam(2, 0x170, 8);
  add_kparam(3, 0x178, 4);

  const std::size_t nv_info_offset = shstrtab_offset + shstrtab_size;
  const std::size_t nv_info_size = nv_info.size();

  const std::size_t text_offset = nv_info_offset + nv_info_size;
  const std::size_t text_size = sass_bytecode.size();

  const std::size_t shoff = text_offset + text_size;
  const std::size_t total_size = shoff + shnum * shent_size;

  buffer.resize(total_size, 0);

  // 1. ELF Header (64 bytes)
  buffer[0] = 0x7F;
  buffer[1] = 'E';
  buffer[2] = 'L';
  buffer[3] = 'F';
  buffer[4] = 2;  // ELFCLASS64
  buffer[5] = 1;  // ELFDATA2LSB
  buffer[6] = 1;  // EV_CURRENT
  buffer[7] = 0x33;  // CUDA OSABI

  const std::uint16_t etype = 2;  // ET_EXEC
  const std::uint16_t emachine = muda::compiler::cubin::kElfMachineCuda;
  const std::uint32_t eversion = 1;
  const std::uint32_t eflags = arch_sm;
  const std::uint64_t eshoff = shoff;
  const std::uint16_t eshstrndx = 1;

  std::memcpy(buffer.data() + 16, &etype, 2);
  std::memcpy(buffer.data() + 18, &emachine, 2);
  std::memcpy(buffer.data() + 20, &eversion, 4);
  std::memcpy(buffer.data() + 40, &eshoff, 8);
  std::memcpy(buffer.data() + 48, &eflags, 4);
  std::memcpy(buffer.data() + 52, &ehdr_size, 2);
  std::memcpy(buffer.data() + 58, &shent_size, 2);
  std::memcpy(buffer.data() + 60, &shnum, 2);
  std::memcpy(buffer.data() + 62, &eshstrndx, 2);

  // 2. Sections Payload
  std::memcpy(buffer.data() + shstrtab_offset, kShstrtabData, shstrtab_size);
  std::memcpy(buffer.data() + nv_info_offset, nv_info.data(), nv_info_size);
  if (!sass_bytecode.empty()) {
    std::memcpy(buffer.data() + text_offset, sass_bytecode.data(), text_size);
  }

  // 3. Section Header Table
  auto write_shdr = [&](std::size_t idx, std::uint32_t name_off, std::uint32_t type,
                        std::uint64_t flags, std::uint64_t offset, std::uint64_t size) {
    const std::size_t base = shoff + idx * shent_size;
    std::memcpy(buffer.data() + base, &name_off, 4);
    std::memcpy(buffer.data() + base + 4, &type, 4);
    std::memcpy(buffer.data() + base + 8, &flags, 8);
    std::memcpy(buffer.data() + base + 24, &offset, 8);
    std::memcpy(buffer.data() + base + 32, &size, 8);
  };

  // Header 0: NULL
  write_shdr(0, 0, 0, 0, 0, 0);
  // Header 1: .shstrtab
  write_shdr(1, 1, 3 /* SHT_STRTAB */, 0, shstrtab_offset, shstrtab_size);
  // Header 2: .nv.info
  write_shdr(2, 11, 1 /* SHT_PROGBITS */, 0, nv_info_offset, nv_info_size);
  // Header 3: .text.k_vector_add
  write_shdr(3, 20, 1 /* SHT_PROGBITS */, 6 /* SHF_ALLOC | SHF_EXECINSTR */, text_offset, text_size);

  return buffer;
}

bool TestFatbinParser() {
  std::cout << "[Test 1] Testing Fatbin container parser and PTX extraction...\n";
  std::vector<std::uint8_t> fatbin = CreateSyntheticFatbin64(kSamplePtx, 75);

  if (!Expect(muda::compiler::cubin::FatbinParser::IsFatbinary(fatbin), "Fatbinary magic validation")) {
    return false;
  }

  auto ptx_opt = muda::compiler::cubin::FatbinParser::ExtractBestPtx(fatbin);
  if (!Expect(ptx_opt.has_value(), "ExtractBestPtx should find payload")) {
    return false;
  }
  if (!Expect(ptx_opt->kind == muda::compiler::cubin::FatbinPayloadKind::kPtx, "Payload kind is PTX")) {
    return false;
  }
  if (!Expect(ptx_opt->arch_sm == 75, "Target architecture is SM 75")) {
    return false;
  }
  if (!Expect(ptx_opt->text.find("vec_add_ptx") != std::string_view::npos, "PTX contains vec_add_ptx")) {
    return false;
  }

  std::cout << "  Passed Fatbin parser tests.\n";
  return true;
}

bool TestElfReader() {
  std::cout << "[Test 2] Testing ELF64 CUBIN container and .nv.info parameter parsing...\n";
  std::vector<std::uint8_t> mock_sass(64, 0);
  std::vector<std::uint8_t> elf_cubin = CreateSyntheticCubinElf(mock_sass, 80);

  if (!Expect(muda::compiler::cubin::ElfReader::IsElf(elf_cubin), "IsElf should be true")) {
    return false;
  }
  if (!Expect(muda::compiler::cubin::ElfReader::IsCudaCubin(elf_cubin), "IsCudaCubin should be true")) {
    return false;
  }
  if (!Expect(muda::compiler::cubin::ElfReader::ExtractSmArch(elf_cubin) == 80, "Arch is SM 80")) {
    return false;
  }

  auto kernels = muda::compiler::cubin::ElfReader::ExtractKernels(elf_cubin);
  if (!Expect(kernels.size() == 1, "Exactly one kernel extracted")) {
    return false;
  }
  const auto& k = kernels[0];
  if (!Expect(k.name == "k_vector_add", "Kernel name matches")) {
    return false;
  }
  if (!Expect(k.params.size() == 4, "Four parameters parsed from .nv.info")) {
    return false;
  }
  if (!Expect(k.params[0].is_pointer && k.params[0].offset == 0x160, "Param 0 is pointer at 0x160")) {
    return false;
  }
  if (!Expect(k.params[1].is_pointer && k.params[1].offset == 0x168, "Param 1 is pointer at 0x168")) {
    return false;
  }
  if (!Expect(k.params[2].is_pointer && k.params[2].offset == 0x170, "Param 2 is pointer at 0x170")) {
    return false;
  }
  if (!Expect(!k.params[3].is_pointer && k.params[3].offset == 0x178, "Param 3 is scalar at 0x178")) {
    return false;
  }

  std::cout << "  Passed ELF64 CUBIN reader tests.\n";
  return true;
}

bool TestSassDecoderAndLifter() {
  std::cout << "[Test 3] Testing SASS instruction decoding and MSL lifting...\n";

  // Create a minimal synthetic SASS sequence:
  // 1. S2R R0, SR_TID.X; (16 bytes)
  // 2. FADD R4, R2, R3; (16 bytes)
  // 3. EXIT; (16 bytes)
  std::vector<std::uint8_t> bytecode(48, 0);

  // Inst 0: S2R R0, SR_TID.X (op_major = 0xF19, rd = 0, sreg = 0x21)
  {
    uint64_t low = 0xF19 | (7ULL << 12) | (0ULL << 16) | (0x21ULL << 40);
    std::memcpy(bytecode.data(), &low, 8);
  }
  // Inst 1: FADD R4, R2, R3 (op_major = 0x221, rd = 4, ra = 2, rb = 3)
  {
    uint64_t low = 0x221 | (7ULL << 12) | (4ULL << 16) | (2ULL << 24) | (3ULL << 32);
    std::memcpy(bytecode.data() + 16, &low, 8);
  }
  // Inst 2: EXIT (op_major = 0x94D)
  {
    uint64_t low = 0x94D | (7ULL << 12);
    std::memcpy(bytecode.data() + 32, &low, 8);
  }

  auto insts = muda::compiler::cubin::SassDecoder::Decode(bytecode, 75);
  if (!Expect(insts.size() == 3, "Decoded 3 instructions")) {
    return false;
  }
  if (!Expect(insts[0].opcode == muda::compiler::cubin::SassOpcode::kS2r, "Inst 0 is S2R")) {
    return false;
  }
  if (!Expect(insts[1].opcode == muda::compiler::cubin::SassOpcode::kFadd, "Inst 1 is FADD")) {
    return false;
  }
  if (!Expect(insts[2].opcode == muda::compiler::cubin::SassOpcode::kExit, "Inst 2 is EXIT")) {
    return false;
  }

  // Lift to MSL
  muda::compiler::cubin::CubinKernel k;
  k.name = "test_kernel";
  k.arch_sm = 75;
  k.params = {{0, 0x160, 8, true}, {1, 0x168, 4, false}};

  std::string msl = muda::compiler::cubin::SassLifter::LiftToMsl(k, insts);
  if (!Expect(msl.find("union Reg32") != std::string::npos, "MSL contains Reg32 union")) {
    return false;
  }
  if (!Expect(msl.find("R[0].u = tid.x;") != std::string::npos, "MSL contains tid.x mapping")) {
    return false;
  }
  if (!Expect(msl.find("R[4].f = R[2].f + R[3].f;") != std::string::npos, "MSL contains FADD lifting")) {
    return false;
  }

  std::cout << "  Passed SASS decoder and lifter tests.\n";
  return true;
}

bool TestEndToEndFatbinExecution() {
  std::cout << "[Test 4] Testing end-to-end Fatbinary execution on GPU via cuModuleLoadData...\n";

  if (!ExpectCuSuccess(cuInit(0), "cuInit")) {
    return false;
  }

  std::vector<std::uint8_t> fatbin = CreateSyntheticFatbin64(kSamplePtx, 75);

  CUmodule mod = nullptr;
  if (!ExpectCuSuccess(cuModuleLoadData(&mod, fatbin.data()), "cuModuleLoadData with Fatbinary")) {
    return false;
  }
  if (!Expect(mod != nullptr, "Loaded module is non-null")) {
    return false;
  }

  CUfunction func = nullptr;
  if (!ExpectCuSuccess(cuModuleGetFunction(&func, mod, "vec_add_ptx"), "cuModuleGetFunction")) {
    cuModuleUnload(mod);
    return false;
  }

  // Allocate host and device memory for vector addition
  constexpr std::size_t kNumElements = 1024;
  constexpr std::size_t kBytes = kNumElements * sizeof(float);

  std::vector<float> h_a(kNumElements);
  std::vector<float> h_b(kNumElements);
  std::vector<float> h_c(kNumElements, 0.0f);

  for (std::size_t i = 0; i < kNumElements; ++i) {
    h_a[i] = static_cast<float>(i) * 1.5f;
    h_b[i] = static_cast<float>(i) * 2.5f;
  }

  CUdeviceptr d_a = 0;
  CUdeviceptr d_b = 0;
  CUdeviceptr d_c = 0;

  if (!ExpectCuSuccess(cuMemAlloc(&d_a, kBytes), "cuMemAlloc d_a")) return false;
  if (!ExpectCuSuccess(cuMemAlloc(&d_b, kBytes), "cuMemAlloc d_b")) return false;
  if (!ExpectCuSuccess(cuMemAlloc(&d_c, kBytes), "cuMemAlloc d_c")) return false;

  if (!ExpectCuSuccess(cuMemcpyHtoD(d_a, h_a.data(), kBytes), "cuMemcpyHtoD d_a")) return false;
  if (!ExpectCuSuccess(cuMemcpyHtoD(d_b, h_b.data(), kBytes), "cuMemcpyHtoD d_b")) return false;

  std::uint32_t n = static_cast<std::uint32_t>(kNumElements);
  void* args[] = {&d_a, &d_b, &d_c, &n};

  const unsigned int block_dim_x = 256;
  const unsigned int grid_dim_x =
      static_cast<unsigned int>((kNumElements + block_dim_x - 1) / block_dim_x);

  if (!ExpectCuSuccess(
          cuLaunchKernel(func, grid_dim_x, 1, 1, block_dim_x, 1, 1, 0, nullptr, args, nullptr),
          "cuLaunchKernel")) {
    return false;
  }

  if (!ExpectCuSuccess(cuCtxSynchronize(), "cuCtxSynchronize")) {
    return false;
  }

  if (!ExpectCuSuccess(cuMemcpyDtoH(h_c.data(), d_c, kBytes), "cuMemcpyDtoH d_c")) {
    return false;
  }

  // Validate results against CPU reference
  int errors = 0;
  for (std::size_t i = 0; i < kNumElements; ++i) {
    const float expected = h_a[i] + h_b[i];
    if (std::abs(h_c[i] - expected) > 1e-4f) {
      if (errors < 5) {
        std::cerr << "Mismatch at [" << i << "]: got " << h_c[i] << ", expected " << expected << "\n";
      }
      errors++;
    }
  }

  if (!Expect(errors == 0, "All vector addition results match CPU golden model")) {
    return false;
  }

  cuMemFree(d_a);
  cuMemFree(d_b);
  cuMemFree(d_c);
  cuModuleUnload(mod);

  std::cout << "  Passed end-to-end Fatbinary GPU execution test with 1024 elements.\n";
  return true;
}

bool TestEndToEndCubinDisassemblyExecution() {
  std::cout << "[Test 5] Testing end-to-end bare ELF CUBIN SASS disassembly and GPU execution...\n";

  // Build bytecode for vector addition using SASS machine instructions:
  // 1. S2R R0, SR_TID.X;            (tid.x -> R0)
  // 2. S2R R1, SR_CTAID.X;          (tgid.x -> R1)
  // 3. IMAD R2, RZ, RZ, c[0][0x160]; (ptr A low -> R2)
  // 4. IMAD R3, RZ, RZ, c[0][0x164]; (ptr A high -> R3)
  // 5. IMAD R4, RZ, RZ, c[0][0x168]; (ptr B low -> R4)
  // 6. IMAD R5, RZ, RZ, c[0][0x16c]; (ptr B high -> R5)
  // 7. IMAD R6, RZ, RZ, c[0][0x170]; (ptr C low -> R6)
  // 8. IMAD R7, RZ, RZ, c[0][0x174]; (ptr C high -> R7)
  // 9. IMAD R8, R0, 4, RZ;          (offset = tid.x * 4 -> R8)
  // 10. IADD R2, R2, R8;            (ptr A += offset)
  // 11. IADD R4, R4, R8;            (ptr B += offset)
  // 12. IADD R6, R6, R8;            (ptr C += offset)
  // 13. LDG.E R10, [R2.64];         (load A[i] -> R10)
  // 14. LDG.E R11, [R4.64];         (load B[i] -> R11)
  // 15. FADD R12, R10, R11;         (R12 = A[i] + B[i])
  // 16. STG.E [R6.64], R12;         (store R12 -> C[i])
  // 17. EXIT;
  std::vector<std::uint8_t> bytecode;
  auto append_inst = [&](uint64_t low, uint64_t high = 0) {
    const std::size_t cur = bytecode.size();
    bytecode.resize(cur + 16, 0);
    std::memcpy(bytecode.data() + cur, &low, 8);
    std::memcpy(bytecode.data() + cur + 8, &high, 8);
  };

  // 1. S2R R0, SR_TID.X
  append_inst(0xF19 | (7ULL << 12) | (0ULL << 16) | (0x21ULL << 40));
  // 2. S2R R1, SR_CTAID.X
  append_inst(0xF19 | (7ULL << 12) | (1ULL << 16) | (0x25ULL << 40));

  auto make_cbank_imad = [](uint32_t rd, uint32_t offset) -> uint64_t {
    const uint64_t cbank_code = (1ULL << 54) | (static_cast<uint64_t>(offset >> 2) << 40);
    return (7ULL << 12) | (static_cast<uint64_t>(rd) << 16) | (255ULL << 24) | cbank_code;
  };

  // 3..8. Load pointers from constant bank 0
  append_inst(make_cbank_imad(2, 0x160));
  append_inst(make_cbank_imad(3, 0x164));
  append_inst(make_cbank_imad(4, 0x168));
  append_inst(make_cbank_imad(5, 0x16C));
  append_inst(make_cbank_imad(6, 0x170));
  append_inst(make_cbank_imad(7, 0x174));

  // 9. IMAD R8, R0, 4, RZ (rd=8, ra=0, is_imm=1, imm=4, rc=255)
  append_inst((7ULL << 12) | (8ULL << 16) | (0ULL << 24) | (1ULL << 53) | (4ULL << 32) | (255ULL << 56));

  // 10. IADD R2, R2, R8 (rd=2, ra=2, rb=8)
  append_inst((7ULL << 12) | (2ULL << 16) | (2ULL << 24) | (8ULL << 32), (0x210ULL << 16));
  // 11. IADD R4, R4, R8 (rd=4, ra=4, rb=8)
  append_inst((7ULL << 12) | (4ULL << 16) | (4ULL << 24) | (8ULL << 32), (0x210ULL << 16));
  // 12. IADD R6, R6, R8 (rd=6, ra=6, rb=8)
  append_inst((7ULL << 12) | (6ULL << 16) | (6ULL << 24) | (8ULL << 32), (0x210ULL << 16));

  // 13. LDG.E R10, [R2.64] (rd=10, ra=2)
  append_inst((7ULL << 12) | (10ULL << 16) | (2ULL << 24), (0x381ULL << 16));
  // 14. LDG.E R11, [R4.64] (rd=11, ra=4)
  append_inst((7ULL << 12) | (11ULL << 16) | (4ULL << 24), (0x381ULL << 16));

  // 15. FADD R12, R10, R11 (rd=12, ra=10, rb=11)
  append_inst((7ULL << 12) | (12ULL << 16) | (10ULL << 24) | (11ULL << 32), (0x221ULL << 16));

  // 16. STG.E [R6.64], R12 (rd=12, ra=6)
  append_inst((7ULL << 12) | (12ULL << 16) | (6ULL << 24), (0x386ULL << 16));

  // 17. EXIT
  append_inst(7ULL << 12, (0x94DULL << 16));

  // Wrap bytecode into a valid ELF64 CUBIN container
  std::vector<std::uint8_t> cubin = CreateSyntheticCubinElf(bytecode, 80);

  CUmodule mod = nullptr;
  if (!ExpectCuSuccess(cuModuleLoadData(&mod, cubin.data()), "cuModuleLoadData with bare ELF CUBIN")) {
    return false;
  }
  if (!Expect(mod != nullptr, "Loaded CUBIN module is non-null")) {
    return false;
  }
  CUfunction func = nullptr;
  if (!ExpectCuSuccess(cuModuleGetFunction(&func, mod, "k_vector_add"), "cuModuleGetFunction")) {
    cuModuleUnload(mod);
    return false;
  }

  constexpr std::size_t kCount = 256;
  constexpr std::size_t kBytes = kCount * sizeof(float);

  std::vector<float> h_a(kCount);
  std::vector<float> h_b(kCount);
  std::vector<float> h_c(kCount, 0.0f);

  for (std::size_t i = 0; i < kCount; ++i) {
    h_a[i] = static_cast<float>(i) * 3.0f;
    h_b[i] = static_cast<float>(i) * 5.0f;
  }

  CUdeviceptr d_a = 0;
  CUdeviceptr d_b = 0;
  CUdeviceptr d_c = 0;

  if (!ExpectCuSuccess(cuMemAlloc(&d_a, kBytes), "cuMemAlloc d_a")) return false;
  if (!ExpectCuSuccess(cuMemAlloc(&d_b, kBytes), "cuMemAlloc d_b")) return false;
  if (!ExpectCuSuccess(cuMemAlloc(&d_c, kBytes), "cuMemAlloc d_c")) return false;

  if (!ExpectCuSuccess(cuMemcpyHtoD(d_a, h_a.data(), kBytes), "cuMemcpyHtoD d_a")) return false;
  if (!ExpectCuSuccess(cuMemcpyHtoD(d_b, h_b.data(), kBytes), "cuMemcpyHtoD d_b")) return false;

  std::uint32_t n = static_cast<std::uint32_t>(kCount);
  void* args[] = {&d_a, &d_b, &d_c, &n};

  if (!ExpectCuSuccess(
          cuLaunchKernel(func, 1, 1, 1, static_cast<unsigned int>(kCount), 1, 1, 0, nullptr, args, nullptr),
          "cuLaunchKernel")) {
    return false;
  }

  if (!ExpectCuSuccess(cuCtxSynchronize(), "cuCtxSynchronize")) {
    return false;
  }

  if (!ExpectCuSuccess(cuMemcpyDtoH(h_c.data(), d_c, kBytes), "cuMemcpyDtoH d_c")) {
    return false;
  }

  // Validate results: C[i] == A[i] + B[i] == 3*i + 5*i == 8*i
  int errors = 0;
  for (std::size_t i = 0; i < kCount; ++i) {
    const float expected = h_a[i] + h_b[i];
    if (std::abs(h_c[i] - expected) > 1e-4f) {
      if (errors < 5) {
        std::cerr << "CUBIN SASS Mismatch at [" << i << "]: got " << h_c[i] << ", expected " << expected << "\n";
      }
      errors++;
    }
  }

  if (!Expect(errors == 0, "All CUBIN SASS execution results match CPU reference")) {
    return false;
  }

  cuMemFree(d_a);
  cuMemFree(d_b);
  cuMemFree(d_c);
  cuModuleUnload(mod);

  std::cout << "  Passed end-to-end bare ELF CUBIN SASS GPU execution test with 256 elements.\n";
  return true;
}

}  // namespace

int main() {
  std::cout << "=== MUDA CUBIN & Fatbin Disassembly Execution Test Suite ===\n";
  if (!TestFatbinParser()) return 1;
  if (!TestElfReader()) return 1;
  if (!TestSassDecoderAndLifter()) return 1;
  if (!TestEndToEndFatbinExecution()) return 1;
  if (!TestEndToEndCubinDisassemblyExecution()) return 1;

  std::cout << "\nALL 5 CUBIN & FATBIN TEST SUITES PASSED SUCCESSFULLY!\n";
  return 0;
}

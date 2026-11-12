/**
 * @file sass_types.hpp
 * @brief Structured representation of NVIDIA SASS (Streaming ASSembly) instructions.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace muda::compiler::cubin {

enum class SassOpcode : std::uint16_t {
  kUnknown,
  // Control flow
  kBra,
  kJmp,
  kRet,
  kExit,
  kNop,
  kBar,
  // Special registers & moves
  kS2r,
  kMov,
  kLdc,
  // Integer arithmetic & logic
  kImad,
  kIadd,
  kIadd3,
  kIsub,
  kImul,
  kLea,
  kLop3,
  kShf,
  kShl,
  kShr,
  kIsetp,
  // Floating-point arithmetic
  kFadd,
  kFsub,
  kFmul,
  kFfma,
  kFsetp,
  kMufu,
  // Memory
  kLdg,
  kStg,
  kLds,
  kSts,
};

enum class SassSpecialReg : std::uint8_t {
  kTidX,
  kTidY,
  kTidZ,
  kCtaIdX,
  kCtaIdY,
  kCtaIdZ,
  kLaneId,
  kUnknown,
};

enum class SassOperandKind : std::uint8_t {
  kReg,         // R0..R254, RZ = 255
  kConstBank,   // c[bank][offset]
  kImm32,       // 32-bit immediate
  kSpecialReg,  // SR_TID.X, etc.
  kAddress,     // [R_base.64 + offset]
  kPredicate,   // P0..P6, PT = 7
};

enum class SassTypeModifier : std::uint8_t {
  kU32,
  kS32,
  kU64,
  kS64,
  kF32,
  kF64,
  kF16,
  kF16x2,
  kNone,
};

enum class SassCompareOp : std::uint8_t {
  kEq,
  kNe,
  kLt,
  kLe,
  kGt,
  kGe,
};

struct SassOperand {
  SassOperandKind kind = SassOperandKind::kReg;
  std::uint32_t reg_idx = 255;  // 255 is RZ (Register Zero)
  std::uint32_t cbank_idx = 0;
  std::uint32_t cbank_offset = 0;
  std::int64_t imm_val = 0;
  SassSpecialReg special_reg = SassSpecialReg::kUnknown;
  bool is_64bit = false;
  std::int32_t addr_offset = 0;

  [[nodiscard]] static SassOperand MakeReg(std::uint32_t reg, bool is_64 = false) noexcept {
    SassOperand op;
    op.kind = SassOperandKind::kReg;
    op.reg_idx = reg;
    op.is_64bit = is_64;
    return op;
  }

  [[nodiscard]] static SassOperand MakeConstBank(std::uint32_t bank, std::uint32_t offset) noexcept {
    SassOperand op;
    op.kind = SassOperandKind::kConstBank;
    op.cbank_idx = bank;
    op.cbank_offset = offset;
    return op;
  }

  [[nodiscard]] static SassOperand MakeImm(std::int64_t val) noexcept {
    SassOperand op;
    op.kind = SassOperandKind::kImm32;
    op.imm_val = val;
    return op;
  }

  [[nodiscard]] static SassOperand MakeSpecial(SassSpecialReg sr) noexcept {
    SassOperand op;
    op.kind = SassOperandKind::kSpecialReg;
    op.special_reg = sr;
    return op;
  }

  [[nodiscard]] static SassOperand MakeAddress(std::uint32_t base_reg, std::int32_t offset = 0,
                                               bool is_64 = true) noexcept {
    SassOperand op;
    op.kind = SassOperandKind::kAddress;
    op.reg_idx = base_reg;
    op.addr_offset = offset;
    op.is_64bit = is_64;
    return op;
  }
};

struct SassInstruction {
  std::uint64_t pc = 0;
  SassOpcode opcode = SassOpcode::kUnknown;
  std::uint8_t predicate_idx = 7;  // 7 is PT (Always True)
  bool predicate_negated = false;
  SassTypeModifier type_mod = SassTypeModifier::kNone;
  SassCompareOp cmp_op = SassCompareOp::kEq;
  std::vector<SassOperand> dsts;
  std::vector<SassOperand> srcs;
  std::int64_t branch_target_pc = -1;  // For BRA instructions
};

}  // namespace muda::compiler::cubin


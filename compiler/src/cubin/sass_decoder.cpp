/**
 * @file sass_decoder.cpp
 * @brief Clean-room 128-bit SASS instruction decoder for NVIDIA Turing (SM 7.5) and Ampere (SM 8.0).
 */

#include "muda/compiler/cubin/sass_decoder.hpp"

#include <cstring>

namespace muda::compiler::cubin {
namespace {

struct Word128 {
  std::uint64_t low = 0;
  std::uint64_t high = 0;

  [[nodiscard]] std::uint64_t Extract(std::size_t start, std::size_t end) const noexcept {
    if (start > 127 || end > 127 || start > end) {
      return 0;
    }
    const std::size_t count = end - start + 1;
    const std::uint64_t mask = (count >= 64) ? ~0ULL : ((1ULL << count) - 1ULL);

    if (end < 64) {
      return (low >> start) & mask;
    }
    if (start >= 64) {
      return (high >> (start - 64)) & mask;
    }
    // Crosses 64-bit boundary
    const std::size_t low_count = 64 - start;
    const std::size_t high_count = end - 63;
    const std::uint64_t low_part = (low >> start);
    const std::uint64_t high_part = (high & ((1ULL << high_count) - 1ULL)) << low_count;
    return (low_part | high_part) & mask;
  }
};

SassSpecialReg DecodeSpecialReg(std::uint32_t val) noexcept {
  switch (val) {
    case 0x21:
    case 0x00:
      return SassSpecialReg::kTidX;
    case 0x22:
    case 0x01:
      return SassSpecialReg::kTidY;
    case 0x23:
    case 0x02:
      return SassSpecialReg::kTidZ;
    case 0x25:
    case 0x04:
      return SassSpecialReg::kCtaIdX;
    case 0x26:
    case 0x05:
      return SassSpecialReg::kCtaIdY;
    case 0x27:
    case 0x06:
      return SassSpecialReg::kCtaIdZ;
    case 0x38:
    case 0x07:
      return SassSpecialReg::kLaneId;
    default:
      break;
  }
  return SassSpecialReg::kUnknown;
}

}  // namespace

SassInstruction SassDecoder::DecodeInstruction(std::span<const std::uint8_t> word_bytes,
                                               std::uint64_t pc,
                                               [[maybe_unused]] std::uint32_t arch_sm) {
  SassInstruction inst;
  inst.pc = pc;
  if (word_bytes.size() < 16) {
    return inst;
  }

  Word128 w;
  std::memcpy(&w.low, word_bytes.data(), 8);
  std::memcpy(&w.high, word_bytes.data() + 8, 8);

  // Predicate: bits 12..14 (index), bit 15 (negation)
  inst.predicate_idx = static_cast<std::uint8_t>(w.Extract(12, 14));
  inst.predicate_negated = (w.Extract(15, 15) != 0);

  // Common register fields
  const std::uint32_t rd = static_cast<std::uint32_t>(w.Extract(16, 23));
  const std::uint32_t ra = static_cast<std::uint32_t>(w.Extract(24, 31));
  const std::uint32_t rb = static_cast<std::uint32_t>(w.Extract(32, 39));

  // Opcode discriminators (Turing SM 7.5 / Ampere SM 8.0)
  const std::uint32_t op_major = static_cast<std::uint32_t>(w.Extract(0, 11));
  const std::uint32_t op_high = static_cast<std::uint32_t>(w.Extract(80, 95));

  // Constant memory bank references
  const bool has_cbank = (w.Extract(54, 55) != 0);
  const std::uint32_t cbank_idx = static_cast<std::uint32_t>(w.Extract(56, 59));
  const std::uint32_t cbank_offset = static_cast<std::uint32_t>(w.Extract(40, 53)) << 2;

  // Immediate values (flagged by bit 53)
  const bool is_imm = (w.Extract(53, 53) != 0);
  const std::int32_t imm_val = static_cast<std::int32_t>(w.Extract(32, 52));
  const std::uint32_t rc = is_imm ? static_cast<std::uint32_t>(w.Extract(56, 63))
                                  : static_cast<std::uint32_t>(w.Extract(40, 47));

  // Classification based on op_high / op_major patterns:
  // 1. EXIT
  if ((op_high & 0x0FFF) == 0x94D || op_major == 0x94D || (w.Extract(0, 15) == 0x0005)) {
    inst.opcode = SassOpcode::kExit;
    return inst;
  }

  // 2. BRA (Branch)
  if ((op_high & 0x0FFF) == 0x947 || op_major == 0x947 || (w.Extract(0, 8) == 0x47)) {
    inst.opcode = SassOpcode::kBra;
    const std::int64_t rel_offset = static_cast<std::int64_t>(static_cast<std::int32_t>(w.Extract(32, 63)));
    inst.branch_target_pc = static_cast<std::int64_t>(pc) + rel_offset + 16;
    inst.srcs.push_back(SassOperand::MakeImm(inst.branch_target_pc));
    return inst;
  }

  // 3. BAR (Threadgroup Barrier)
  if ((op_high & 0x0FFF) == 0xB1D || op_major == 0xB1D) {
    inst.opcode = SassOpcode::kBar;
    return inst;
  }

  // 4. S2R (Special Register to Register)
  if ((op_high & 0x0FFF) == 0xF19 || (op_high & 0x0FFF) == 0xF1A || op_major == 0xF19) {
    inst.opcode = SassOpcode::kS2r;
    inst.dsts.push_back(SassOperand::MakeReg(rd));
    const std::uint32_t sreg_code = static_cast<std::uint32_t>(w.Extract(40, 47));
    inst.srcs.push_back(SassOperand::MakeSpecial(DecodeSpecialReg(sreg_code)));
    return inst;
  }

  // 5. Memory Loads / Stores (LDG.E / STG.E / LDS / STS)
  // Global Store STG.E
  if ((op_high & 0x0FFF) == 0x386 || (op_high & 0x0FFF) == 0x387 || op_major == 0x386) {
    inst.opcode = SassOpcode::kStg;
    inst.type_mod = SassTypeModifier::kF32;
    const std::int32_t offset = static_cast<std::int32_t>(w.Extract(40, 63));
    inst.dsts.push_back(SassOperand::MakeAddress(ra, offset, true));
    inst.srcs.push_back(SassOperand::MakeReg(rd));
    return inst;
  }

  // Global Load LDG.E
  if ((op_high & 0x0FFF) == 0x381 || (op_high & 0x0FFF) == 0x382 || op_major == 0x381) {
    inst.opcode = SassOpcode::kLdg;
    inst.type_mod = SassTypeModifier::kF32;
    inst.dsts.push_back(SassOperand::MakeReg(rd));
    const std::int32_t offset = static_cast<std::int32_t>(w.Extract(40, 63));
    inst.srcs.push_back(SassOperand::MakeAddress(ra, offset, true));
    return inst;
  }

  // Shared Store STS
  if ((op_high & 0x0FFF) == 0x388 || op_major == 0x388) {
    inst.opcode = SassOpcode::kSts;
    const std::int32_t offset = static_cast<std::int32_t>(w.Extract(40, 63));
    inst.dsts.push_back(SassOperand::MakeAddress(ra, offset, false));
    inst.srcs.push_back(SassOperand::MakeReg(rd));
    return inst;
  }

  // Shared Load LDS
  if ((op_high & 0x0FFF) == 0x384 || op_major == 0x384) {
    inst.opcode = SassOpcode::kLds;
    inst.dsts.push_back(SassOperand::MakeReg(rd));
    const std::int32_t offset = static_cast<std::int32_t>(w.Extract(40, 63));
    inst.srcs.push_back(SassOperand::MakeAddress(ra, offset, false));
    return inst;
  }

  // 6. Floating Point Operations (FFMA, FADD, FMUL)
  // FFMA: Rd = Ra * Rb + Rc
  if ((op_high & 0x0FFF) == 0x223 || op_major == 0x223) {
    inst.opcode = SassOpcode::kFfma;
    inst.type_mod = SassTypeModifier::kF32;
    inst.dsts.push_back(SassOperand::MakeReg(rd));
    inst.srcs.push_back(SassOperand::MakeReg(ra));
    if (has_cbank) {
      inst.srcs.push_back(SassOperand::MakeConstBank(cbank_idx, cbank_offset));
    } else if (is_imm) {
      inst.srcs.push_back(SassOperand::MakeImm(imm_val));
    } else {
      inst.srcs.push_back(SassOperand::MakeReg(rb));
    }
    inst.srcs.push_back(SassOperand::MakeReg(rc));
    return inst;
  }

  // FADD: Rd = Ra + Rb
  if ((op_high & 0x0FFF) == 0x221 || op_major == 0x221) {
    inst.opcode = SassOpcode::kFadd;
    inst.type_mod = SassTypeModifier::kF32;
    inst.dsts.push_back(SassOperand::MakeReg(rd));
    inst.srcs.push_back(SassOperand::MakeReg(ra));
    if (has_cbank) {
      inst.srcs.push_back(SassOperand::MakeConstBank(cbank_idx, cbank_offset));
    } else {
      inst.srcs.push_back(SassOperand::MakeReg(rb));
    }
    return inst;
  }

  // FMUL: Rd = Ra * Rb
  if ((op_high & 0x0FFF) == 0x220 || op_major == 0x220) {
    inst.opcode = SassOpcode::kFmul;
    inst.type_mod = SassTypeModifier::kF32;
    inst.dsts.push_back(SassOperand::MakeReg(rd));
    inst.srcs.push_back(SassOperand::MakeReg(ra));
    if (has_cbank) {
      inst.srcs.push_back(SassOperand::MakeConstBank(cbank_idx, cbank_offset));
    } else {
      inst.srcs.push_back(SassOperand::MakeReg(rb));
    }
    return inst;
  }

  // 7. Comparison & Predicate: ISETP / FSETP
  if ((op_high & 0x0FFF) == 0x00C || (op_high & 0x0FFF) == 0x00D || op_major == 0x00C) {
    inst.opcode = SassOpcode::kIsetp;
    inst.type_mod = SassTypeModifier::kU32;
    const std::uint32_t pred_dst = static_cast<std::uint32_t>(w.Extract(16, 18));
    inst.dsts.push_back(SassOperand::MakeReg(pred_dst));
    inst.srcs.push_back(SassOperand::MakeReg(ra));
    if (has_cbank) {
      inst.srcs.push_back(SassOperand::MakeConstBank(cbank_idx, cbank_offset));
    } else if (is_imm) {
      inst.srcs.push_back(SassOperand::MakeImm(imm_val));
    } else {
      inst.srcs.push_back(SassOperand::MakeReg(rb));
    }
    // Condition code (bits 48..51)
    const std::uint32_t cond = static_cast<std::uint32_t>(w.Extract(48, 51));
    switch (cond) {
      case 0x1:
        inst.cmp_op = SassCompareOp::kLt;
        break;
      case 0x2:
        inst.cmp_op = SassCompareOp::kEq;
        break;
      case 0x3:
        inst.cmp_op = SassCompareOp::kLe;
        break;
      case 0x4:
        inst.cmp_op = SassCompareOp::kGt;
        break;
      case 0x5:
        inst.cmp_op = SassCompareOp::kNe;
        break;
      case 0x6:
      default:
        inst.cmp_op = SassCompareOp::kGe;
        break;
    }
    return inst;
  }

  // 8. Integer Arithmetic & Address Math (IADD / IMAD / IADD3 / LEA)
  if ((op_high & 0x0FFF) == 0x210 || op_major == 0x210) {
    inst.opcode = SassOpcode::kIadd;
    inst.type_mod = SassTypeModifier::kU32;
    inst.dsts.push_back(SassOperand::MakeReg(rd));
    inst.srcs.push_back(SassOperand::MakeReg(ra));
    if (has_cbank) {
      inst.srcs.push_back(SassOperand::MakeConstBank(cbank_idx, cbank_offset));
    } else if (is_imm) {
      inst.srcs.push_back(SassOperand::MakeImm(imm_val));
    } else {
      inst.srcs.push_back(SassOperand::MakeReg(rb));
    }
    return inst;
  }

  // IMAD: Rd = Ra * Rb + Rc (or IMAD.MOV if Ra=RZ and Rc=RZ)
  inst.opcode = SassOpcode::kImad;
  inst.type_mod = SassTypeModifier::kU32;
  inst.dsts.push_back(SassOperand::MakeReg(rd));
  if (has_cbank) {
    if (ra == 255) {
      inst.srcs.push_back(SassOperand::MakeConstBank(cbank_idx, cbank_offset));
    } else {
      inst.srcs.push_back(SassOperand::MakeReg(ra));
      inst.srcs.push_back(SassOperand::MakeConstBank(cbank_idx, cbank_offset));
    }
  } else {
    inst.srcs.push_back(SassOperand::MakeReg(ra));
    if (is_imm) {
      inst.srcs.push_back(SassOperand::MakeImm(imm_val));
    } else {
      inst.srcs.push_back(SassOperand::MakeReg(rb));
    }
    if (rc != 255) {
      inst.srcs.push_back(SassOperand::MakeReg(rc));
    }
  }

  return inst;
}

std::vector<SassInstruction> SassDecoder::Decode(std::span<const std::uint8_t> text_bytes,
                                                 std::uint32_t arch_sm) {
  std::vector<SassInstruction> instructions;
  if (text_bytes.empty()) {
    return instructions;
  }

  const std::size_t count = text_bytes.size() / 16;
  instructions.reserve(count);

  for (std::size_t i = 0; i < count; ++i) {
    const std::uint64_t pc = i * 16;
    std::span<const std::uint8_t> word(text_bytes.data() + pc, 16);
    instructions.push_back(DecodeInstruction(word, pc, arch_sm));
  }

  return instructions;
}

}  // namespace muda::compiler::cubin

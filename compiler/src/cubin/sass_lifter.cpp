/**
 * @file sass_lifter.cpp
 * @brief Lifts decoded SASS instructions and CUBIN metadata into Metal Shading Language (MSL).
 */

#include "muda/compiler/cubin/sass_lifter.hpp"

#include <algorithm>
#include <sstream>
#include <unordered_set>

namespace muda::compiler::cubin {
namespace {

std::string SanitizeIdentifier(std::string_view name) {
  std::string s;
  for (char c : name) {
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_') {
      s.push_back(c);
    } else {
      s.push_back('_');
    }
  }
  return s;
}

std::string FormatOperandU32(const SassOperand& op) {
  switch (op.kind) {
    case SassOperandKind::kReg:
      if (op.reg_idx == 255) {
        return "0u";
      }
      return "R[" + std::to_string(op.reg_idx) + "].u";
    case SassOperandKind::kConstBank:
      return "read_c0(" + std::to_string(op.cbank_offset) + ")";
    case SassOperandKind::kImm32:
      return std::to_string(op.imm_val) + "u";
    case SassOperandKind::kSpecialReg:
      switch (op.special_reg) {
        case SassSpecialReg::kTidX:
          return "tid.x";
        case SassSpecialReg::kTidY:
          return "tid.y";
        case SassSpecialReg::kTidZ:
          return "tid.z";
        case SassSpecialReg::kCtaIdX:
          return "tgid.x";
        case SassSpecialReg::kCtaIdY:
          return "tgid.y";
        case SassSpecialReg::kCtaIdZ:
          return "tgid.z";
        case SassSpecialReg::kLaneId:
          return "lane_id";
        default:
          return "0u";
      }
    default:
      return "0u";
  }
}

std::string FormatOperandF32(const SassOperand& op) {
  switch (op.kind) {
    case SassOperandKind::kReg:
      if (op.reg_idx == 255) {
        return "0.0f";
      }
      return "R[" + std::to_string(op.reg_idx) + "].f";
    case SassOperandKind::kConstBank:
      return "as_type<float>(read_c0(" + std::to_string(op.cbank_offset) + "))";
    case SassOperandKind::kImm32:
      return "as_type<float>(" + std::to_string(op.imm_val) + "u)";
    default:
      return "0.0f";
  }
}

std::string FormatCmpOp(SassCompareOp op) {
  switch (op) {
    case SassCompareOp::kEq:
      return "==";
    case SassCompareOp::kNe:
      return "!=";
    case SassCompareOp::kLt:
      return "<";
    case SassCompareOp::kLe:
      return "<=";
    case SassCompareOp::kGt:
      return ">";
    case SassCompareOp::kGe:
      return ">=";
  }
  return "==";
}

void EmitInstruction(const SassInstruction& inst, std::ostringstream& ss, bool in_switch,
                     std::string_view indent) {
  ss << indent;

  // Predicate guard
  const bool has_pred = (inst.predicate_idx != 7);
  if (has_pred) {
    ss << "if (" << (inst.predicate_negated ? "!" : "") << "P[" << std::to_string(inst.predicate_idx) << "]) { ";
  }

  switch (inst.opcode) {
    case SassOpcode::kExit:
      if (in_switch) {
        ss << "pc = 0xFFFFFFFF; break;";
      } else {
        ss << "return;";
      }
      break;

    case SassOpcode::kBra:
      if (inst.branch_target_pc >= 0) {
        ss << "pc = " << std::to_string(inst.branch_target_pc) << "; break;";
      }
      break;

    case SassOpcode::kBar:
      ss << "threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);";
      break;

    case SassOpcode::kS2r:
      if (!inst.dsts.empty() && !inst.srcs.empty() && inst.dsts[0].reg_idx != 255) {
        ss << "R[" << std::to_string(inst.dsts[0].reg_idx) << "].u = " << FormatOperandU32(inst.srcs[0]) << ";";
      }
      break;

    case SassOpcode::kImad:
      if (!inst.dsts.empty() && inst.dsts[0].reg_idx != 255) {
        const std::string dst = "R[" + std::to_string(inst.dsts[0].reg_idx) + "].u";
        if (inst.srcs.size() == 1) {
          ss << dst << " = " << FormatOperandU32(inst.srcs[0]) << ";";
        } else {
          const std::string s0 = (inst.srcs.size() > 0) ? FormatOperandU32(inst.srcs[0]) : "0u";
          const std::string s1 = (inst.srcs.size() > 1) ? FormatOperandU32(inst.srcs[1]) : "0u";
          const std::string s2 = (inst.srcs.size() > 2) ? FormatOperandU32(inst.srcs[2]) : "0u";

          if (s0 == "0u" && s2 == "0u") {
            // IMAD.MOV
            ss << dst << " = " << s1 << ";";
          } else if (s2 == "0u") {
            ss << dst << " = " << s0 << " * " << s1 << ";";
          } else if (s1 == "1u" && s0 == "0u") {
            ss << dst << " = " << s2 << ";";
          } else if (s0 == "1u") {
            ss << dst << " = " << s1 << " + " << s2 << ";";
          } else {
            ss << dst << " = " << s0 << " * " << s1 << " + " << s2 << ";";
          }
        }
      }
      break;

    case SassOpcode::kIadd:
      if (!inst.dsts.empty() && inst.dsts[0].reg_idx != 255 && inst.srcs.size() >= 2) {
        ss << "R[" << std::to_string(inst.dsts[0].reg_idx) << "].u = " << FormatOperandU32(inst.srcs[0])
           << " + " << FormatOperandU32(inst.srcs[1]) << ";";
      }
      break;

    case SassOpcode::kIadd3:
      if (!inst.dsts.empty() && inst.dsts[0].reg_idx != 255 && inst.srcs.size() >= 3) {
        ss << "R[" << std::to_string(inst.dsts[0].reg_idx) << "].u = " << FormatOperandU32(inst.srcs[0])
           << " + " << FormatOperandU32(inst.srcs[1]) << " + " << FormatOperandU32(inst.srcs[2]) << ";";
      }
      break;

    case SassOpcode::kIsub:
      if (!inst.dsts.empty() && inst.dsts[0].reg_idx != 255 && inst.srcs.size() >= 2) {
        ss << "R[" << std::to_string(inst.dsts[0].reg_idx) << "].u = " << FormatOperandU32(inst.srcs[0])
           << " - " << FormatOperandU32(inst.srcs[1]) << ";";
      }
      break;

    case SassOpcode::kFadd:
      if (!inst.dsts.empty() && inst.dsts[0].reg_idx != 255 && inst.srcs.size() >= 2) {
        ss << "R[" << std::to_string(inst.dsts[0].reg_idx) << "].f = " << FormatOperandF32(inst.srcs[0])
           << " + " << FormatOperandF32(inst.srcs[1]) << ";";
      }
      break;

    case SassOpcode::kFsub:
      if (!inst.dsts.empty() && inst.dsts[0].reg_idx != 255 && inst.srcs.size() >= 2) {
        ss << "R[" << std::to_string(inst.dsts[0].reg_idx) << "].f = " << FormatOperandF32(inst.srcs[0])
           << " - " << FormatOperandF32(inst.srcs[1]) << ";";
      }
      break;

    case SassOpcode::kFmul:
      if (!inst.dsts.empty() && inst.dsts[0].reg_idx != 255 && inst.srcs.size() >= 2) {
        ss << "R[" << std::to_string(inst.dsts[0].reg_idx) << "].f = " << FormatOperandF32(inst.srcs[0])
           << " * " << FormatOperandF32(inst.srcs[1]) << ";";
      }
      break;

    case SassOpcode::kFfma:
      if (!inst.dsts.empty() && inst.dsts[0].reg_idx != 255 && inst.srcs.size() >= 3) {
        ss << "R[" << std::to_string(inst.dsts[0].reg_idx) << "].f = fma("
           << FormatOperandF32(inst.srcs[0]) << ", " << FormatOperandF32(inst.srcs[1]) << ", "
           << FormatOperandF32(inst.srcs[2]) << ");";
      }
      break;

    case SassOpcode::kIsetp:
      if (!inst.dsts.empty() && inst.srcs.size() >= 2) {
        ss << "P[" << std::to_string(inst.dsts[0].reg_idx) << "] = (" << FormatOperandU32(inst.srcs[0])
           << " " << FormatCmpOp(inst.cmp_op) << " " << FormatOperandU32(inst.srcs[1]) << ");";
      }
      break;

    case SassOpcode::kLdg:
      if (!inst.dsts.empty() && !inst.srcs.empty() && inst.dsts[0].reg_idx != 255) {
        const std::uint32_t ra = inst.srcs[0].reg_idx;
        const std::int32_t off = inst.srcs[0].addr_offset;
        ss << "{ uint64_t ea = ((static_cast<uint64_t>(R[" << (ra + 1) << "].u) << 32) | R[" << ra << "].u) + "
           << off << "; R[" << inst.dsts[0].reg_idx << "].f = *reinterpret_cast<device const float*>(ea); }";
      }
      break;

    case SassOpcode::kStg:
      if (!inst.dsts.empty() && !inst.srcs.empty()) {
        const std::uint32_t ra = inst.dsts[0].reg_idx;
        const std::int32_t off = inst.dsts[0].addr_offset;
        ss << "{ uint64_t ea = ((static_cast<uint64_t>(R[" << (ra + 1) << "].u) << 32) | R[" << ra << "].u) + "
           << off << "; *reinterpret_cast<device float*>(ea) = " << FormatOperandF32(inst.srcs[0]) << "; }";
      }
      break;

    case SassOpcode::kLds:
      if (!inst.dsts.empty() && !inst.srcs.empty() && inst.dsts[0].reg_idx != 255) {
        const std::uint32_t ra = inst.srcs[0].reg_idx;
        const std::int32_t off = inst.srcs[0].addr_offset;
        ss << "R[" << inst.dsts[0].reg_idx << "].f = *reinterpret_cast<threadgroup const float*>(smem + R["
           << ra << "].u + " << off << ");";
      }
      break;

    case SassOpcode::kSts:
      if (!inst.dsts.empty() && !inst.srcs.empty()) {
        const std::uint32_t ra = inst.dsts[0].reg_idx;
        const std::int32_t off = inst.dsts[0].addr_offset;
        ss << "*reinterpret_cast<threadgroup float*>(smem + R[" << ra << "].u + " << off
           << ") = " << FormatOperandF32(inst.srcs[0]) << ";";
      }
      break;

    default:
      ss << "/* unsupported opcode */";
      break;
  }

  if (has_pred) {
    ss << " }";
  }
  ss << "\n";
}

}  // namespace

std::string SassLifter::LiftToMsl(const CubinKernel& kernel,
                                  const std::vector<SassInstruction>& instructions) {
  std::ostringstream ss;
  ss << "#include <metal_stdlib>\n";
  ss << "using namespace metal;\n\n";

  ss << "union Reg32 {\n";
  ss << "  uint32_t u;\n";
  ss << "  int32_t  i;\n";
  ss << "  float    f;\n";
  ss << "};\n\n";

  const std::string msl_func_name = "k_" + SanitizeIdentifier(kernel.name);
  ss << "[[kernel]] void " << msl_func_name << "(\n";

  // Parameter bindings
  for (std::size_t i = 0; i < kernel.params.size(); ++i) {
    const CubinKernelParam& p = kernel.params[i];
    if (p.is_pointer) {
      ss << "    device uint8_t* p" << i << " [[buffer(" << i << ")]],\n";
    } else {
      ss << "    constant uint32_t& p" << i << " [[buffer(" << i << ")]],\n";
    }
  }

  // Built-in Metal attributes
  ss << "    threadgroup uint8_t* smem [[threadgroup(0)]],\n";
  ss << "    uint3 tid [[thread_position_in_threadgroup]],\n";
  ss << "    uint3 tgid [[threadgroup_position_in_grid]],\n";
  ss << "    uint3 tptg [[threads_per_threadgroup]],\n";
  ss << "    uint lane_id [[thread_index_in_simdgroup]])\n";
  ss << "{\n";

  // Register and predicate state initialization
  ss << "  thread Reg32 R[256];\n";
  ss << "  for (int i = 0; i < 256; ++i) { R[i].u = 0; }\n";
  ss << "  thread bool P[8] = {false, false, false, false, false, false, false, true};\n";
  ss << "  R[255].u = 0; // RZ is hardwired to zero\n\n";

  // Helper lambda to read from Constant Memory Bank 0 (c[0x0][offset])
  ss << "  auto read_c0 = [&](uint32_t offset) -> uint32_t {\n";
  for (std::size_t i = 0; i < kernel.params.size(); ++i) {
    const CubinKernelParam& p = kernel.params[i];
    if (p.is_pointer) {
      ss << "    if (offset == " << p.offset << ") return static_cast<uint32_t>(reinterpret_cast<uint64_t>(p" << i
         << "));\n";
      ss << "    if (offset == " << (p.offset + 4) << ") return static_cast<uint32_t>(reinterpret_cast<uint64_t>(p"
         << i << ") >> 32);\n";
    } else {
      ss << "    if (offset == " << p.offset << ") return *reinterpret_cast<constant const uint32_t*>(&p" << i
         << ");\n";
    }
  }
  ss << "    return 0u;\n";
  ss << "  };\n\n";

  // Check if any BRA instructions exist
  bool has_branches = false;
  std::unordered_set<uint64_t> branch_targets;
  for (const SassInstruction& inst : instructions) {
    if (inst.opcode == SassOpcode::kBra && inst.branch_target_pc >= 0) {
      has_branches = true;
      branch_targets.insert(static_cast<uint64_t>(inst.branch_target_pc));
    }
  }

  if (!has_branches) {
    // Pure sequential execution
    for (const SassInstruction& inst : instructions) {
      EmitInstruction(inst, ss, false, "  ");
    }
  } else {
    // General State Machine execution loop for control flow graphs
    ss << "  uint32_t pc = 0;\n";
    ss << "  while (pc != 0xFFFFFFFF) {\n";
    ss << "    switch (pc) {\n";

    for (std::size_t i = 0; i < instructions.size(); ++i) {
      const SassInstruction& inst = instructions[i];
      ss << "      case " << inst.pc << ": {\n";
      EmitInstruction(inst, ss, true, "        ");
      if (inst.opcode != SassOpcode::kBra && inst.opcode != SassOpcode::kExit) {
        const uint64_t next_pc = (i + 1 < instructions.size()) ? instructions[i + 1].pc : 0xFFFFFFFF;
        ss << "        pc = " << next_pc << ";\n";
        ss << "        break;\n";
      }
      ss << "      }\n";
    }

    ss << "      default:\n";
    ss << "        return;\n";
    ss << "    }\n";
    ss << "  }\n";
  }

  ss << "}\n";
  return ss.str();
}

}  // namespace muda::compiler::cubin

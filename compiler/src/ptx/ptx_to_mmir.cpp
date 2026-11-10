/**
 * @file ptx_to_mmir.cpp
 * @brief Implementation of PTX lowering to MMIR Module and MSL emission.
 */

#include "muda/compiler/ptx/ptx_to_mmir.hpp"

#include <cstddef>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

#include "muda/compiler/mmir.hpp"
#include "muda/compiler/ptx/arena.hpp"
#include "muda/compiler/ptx/ast.hpp"
#include "muda/compiler/ptx/cfg.hpp"
#include "muda/compiler/ptx/lexer.hpp"
#include "muda/compiler/ptx/parser.hpp"
#include "muda/compiler/ptx/structurizer.hpp"
#include "muda/compiler/ptx/type_inference.hpp"
#include "muda/compiler/translator.hpp"
#include "muda/compiler/types.hpp"

namespace muda::compiler::ptx {
namespace {

std::string SanitizeName(std::string_view name) {
  std::string s;
  for (char c : name) {
    if (c == '%' || c == '$') {
      // omit special prefixes for clean C++ variable names
    } else if (c == '.') {
      s.push_back('_');
    } else {
      s.push_back(c);
    }
  }
  return s;
}

std::string TranslateSpecialRegister(std::string_view reg) {
  if (reg == "%tid.x")
    return "tid.x";
  if (reg == "%tid.y")
    return "tid.y";
  if (reg == "%tid.z")
    return "tid.z";
  if (reg == "%ntid.x")
    return "tptg.x";
  if (reg == "%ntid.y")
    return "tptg.y";
  if (reg == "%ntid.z")
    return "tptg.z";
  if (reg == "%ctaid.x")
    return "tgid.x";
  if (reg == "%ctaid.y")
    return "tgid.y";
  if (reg == "%ctaid.z")
    return "tgid.z";
  if (reg == "%nctaid.x")
    return "tpg.x";
  if (reg == "%nctaid.y")
    return "tpg.y";
  if (reg == "%nctaid.z")
    return "tpg.z";
  return SanitizeName(reg);
}

std::string TranslateOperand(const Operand& op) {
  switch (op.kind) {
    case OperandKind::kRegister:
      return SanitizeName(op.name);
    case OperandKind::kSpecialRegister:
      return TranslateSpecialRegister(op.name);
    case OperandKind::kImmediateInt:
      return std::to_string(op.int_val);
    case OperandKind::kImmediateFloat:
      return std::to_string(op.float_val) + "f";
    case OperandKind::kAddress: {
      std::string res = SanitizeName(op.name);
      if (op.offset != 0) {
        res += " + " + std::to_string(op.offset);
      }
      return res;
    }
    case OperandKind::kLabel:
      return SanitizeName(op.name);
    case OperandKind::kVectorList:
      return "/* vector */";
  }
  return "0";
}

std::string GetModifierMslType(const Instruction& inst, const TypeInference& types,
                               std::string_view reg_name) {
  for (std::string_view mod : inst.modifiers) {
    if (mod == ".f32")
      return "float";
    if (mod == ".f64")
      return "muda_fp64";
    if (mod == ".s32")
      return "int";
    if (mod == ".u32")
      return "uint";
    if (mod == ".s64")
      return "int64_t";
    if (mod == ".u64")
      return "uint64_t";
    if (mod == ".s16")
      return "short";
    if (mod == ".u16")
      return "ushort";
    if (mod == ".s8" || mod == ".b8")
      return "int8_t";
    if (mod == ".u8")
      return "uint8_t";
    if (mod == ".f16")
      return "half";
    if (mod == ".pred")
      return "bool";
  }
  if (inst.type == TypeKind::kF32)
    return "float";
  if (inst.type == TypeKind::kF64)
    return "muda_fp64";
  if (inst.type == TypeKind::kS32)
    return "int";
  if (inst.type == TypeKind::kU32)
    return "uint";
  if (inst.type == TypeKind::kS64)
    return "int64_t";
  if (inst.type == TypeKind::kU64)
    return "uint64_t";
  if (inst.type == TypeKind::kF16)
    return "half";
  if (inst.type == TypeKind::kS16)
    return "short";
  if (inst.type == TypeKind::kU16)
    return "ushort";
  if (inst.type == TypeKind::kS8 || inst.type == TypeKind::kB8)
    return "int8_t";
  if (inst.type == TypeKind::kU8)
    return "uint8_t";
  if (inst.type == TypeKind::kPred)
    return "bool";

  const DataType dt = types.GetDataType(reg_name);
  return std::string(MslTypeName(dt));
}

std::string GetAddressSpaceKeyword(const Instruction& inst, const TypeInference& types,
                                   std::string_view addr_reg) {
  for (std::string_view mod : inst.modifiers) {
    if (mod == ".shared")
      return "threadgroup";
    if (mod == ".global")
      return "device";
    if (mod == ".const")
      return "constant";
  }
  const AddressSpace sp = types.GetAddressSpace(addr_reg);
  if (sp == AddressSpace::kThreadgroup)
    return "threadgroup";
  if (sp == AddressSpace::kConstant)
    return "constant";
  return "device";
}

void LowerInstruction(const Instruction& inst, const TypeInference& types, std::string* body) {
  if (inst.operands.empty()) {
    if (inst.opcode == TokenKind::kOpRet) {
      *body += "return;\n  ";
    } else if (inst.opcode == TokenKind::kOpBar) {
      *body += "threadgroup_barrier(mem_flags::mem_threadgroup);\n  ";
    }
    return;
  }

  // Branch instruction
  if (inst.opcode == TokenKind::kOpBra) {
    if (inst.predicate.has_value()) {
      const auto& p = inst.predicate.value();
      *body += "if (" + std::string(p.negated ? "!" : "") + SanitizeName(p.reg_name) +
               ") { return; }\n  ";
    } else {
      *body += "return;\n  ";
    }
    return;
  }

  const std::string dest = TranslateOperand(inst.operands[0]);

  // Handle ld (ld.param, ld.global, ld.shared)
  if (inst.opcode == TokenKind::kOpLd) {
    bool is_param = false;
    for (std::string_view mod : inst.modifiers) {
      if (mod == ".param")
        is_param = true;
    }
    if (is_param && inst.operands.size() >= 2) {
      const std::string src = TranslateOperand(inst.operands[1]);
      if (types.IsPointer(inst.operands[1].name)) {
        *body += dest + " = reinterpret_cast<uint64_t>(" + src + ");\n  ";
      } else {
        const std::string type_name = GetModifierMslType(inst, types, inst.operands[0].name);
        *body += dest + " = static_cast<" + type_name + ">(" + src + ");\n  ";
      }
      return;
    }

    if (inst.operands.size() >= 2 && inst.operands[1].kind == OperandKind::kAddress) {
      const std::string addr = TranslateOperand(inst.operands[1]);
      const std::string type_name = GetModifierMslType(inst, types, inst.operands[0].name);
      const std::string space = GetAddressSpaceKeyword(inst, types, inst.operands[1].name);
      *body += dest + " = (*reinterpret_cast<" + space + " const " + type_name + "*>(" + addr +
               "));\n  ";
      return;
    }
  }

  // Handle st (st.global, st.shared)
  if (inst.opcode == TokenKind::kOpSt && inst.operands.size() >= 2) {
    const std::string addr = TranslateOperand(inst.operands[0]);
    const std::string val = TranslateOperand(inst.operands[1]);
    const std::string type_name = GetModifierMslType(inst, types, inst.operands[1].name);
    const std::string space = GetAddressSpaceKeyword(inst, types, inst.operands[0].name);
    *body +=
        "(*reinterpret_cast<" + space + " " + type_name + "*>(" + addr + ")) = " + val + ";\n  ";
    return;
  }

  // Handle cvta (convert address)
  if (inst.opcode == TokenKind::kOpCvta && inst.operands.size() >= 2) {
    const std::string src = TranslateOperand(inst.operands[1]);
    *body += dest + " = reinterpret_cast<uint64_t>(" + src + ");\n  ";
    return;
  }

  // Handle cvt (convert data type)
  if (inst.opcode == TokenKind::kOpCvt && inst.operands.size() >= 2) {
    const std::string src = TranslateOperand(inst.operands[1]);
    const std::string type_name = GetModifierMslType(inst, types, inst.operands[0].name);
    *body += dest + " = static_cast<" + type_name + ">(" + src + ");\n  ";
    return;
  }

  // Handle selp (predicate select: selp dest, src1, src2, pred)
  if (inst.opcode == TokenKind::kOpSelp && inst.operands.size() >= 4) {
    const std::string src1 = TranslateOperand(inst.operands[1]);
    const std::string src2 = TranslateOperand(inst.operands[2]);
    const std::string pred = TranslateOperand(inst.operands[3]);
    *body += dest + " = (" + pred + " ? " + src1 + " : " + src2 + ");\n  ";
    return;
  }

  // Handle setp (e.g. setp.ge.s32 %p1, %r0, %r1)
  if (inst.opcode == TokenKind::kOpSetp && inst.operands.size() >= 3) {
    const std::string op1 = TranslateOperand(inst.operands[1]);
    const std::string op2 = TranslateOperand(inst.operands[2]);
    std::string cmp = "==";
    for (std::string_view mod : inst.modifiers) {
      if (mod == ".ge")
        cmp = ">=";
      else if (mod == ".gt")
        cmp = ">";
      else if (mod == ".le")
        cmp = "<=";
      else if (mod == ".lt")
        cmp = "<";
      else if (mod == ".ne")
        cmp = "!=";
      else if (mod == ".eq")
        cmp = "==";
    }
    *body += dest + " = (" + op1 + " " + cmp + " " + op2 + ");\n  ";
    return;
  }

  // Handle fma
  if (inst.opcode == TokenKind::kOpFma && inst.operands.size() >= 4) {
    const std::string a = TranslateOperand(inst.operands[1]);
    const std::string b = TranslateOperand(inst.operands[2]);
    const std::string c = TranslateOperand(inst.operands[3]);
    *body += dest + " = fma(" + a + ", " + b + ", " + c + ");\n  ";
    return;
  }

  // Handle mad (e.g. mad.lo.s32 %r0, %r2, %r3, %r4)
  if (inst.opcode == TokenKind::kOpMad && inst.operands.size() >= 4) {
    const std::string a = TranslateOperand(inst.operands[1]);
    const std::string b = TranslateOperand(inst.operands[2]);
    const std::string c = TranslateOperand(inst.operands[3]);
    *body += dest + " = (" + a + " * " + b + ") + " + c + ";\n  ";
    return;
  }

  // Handle rem
  if (inst.opcode == TokenKind::kOpRem && inst.operands.size() >= 3) {
    const std::string a = TranslateOperand(inst.operands[1]);
    const std::string b = TranslateOperand(inst.operands[2]);
    *body += dest + " = " + a + " % " + b + ";\n  ";
    return;
  }

  // Handle min / max
  if ((inst.opcode == TokenKind::kOpMin || inst.opcode == TokenKind::kOpMax) &&
      inst.operands.size() >= 3) {
    const std::string a = TranslateOperand(inst.operands[1]);
    const std::string b = TranslateOperand(inst.operands[2]);
    const std::string fn_name = (inst.opcode == TokenKind::kOpMin) ? "min" : "max";
    *body += dest + " = " + fn_name + "(" + a + ", " + b + ");\n  ";
    return;
  }

  // Handle neg / abs
  if ((inst.opcode == TokenKind::kOpNeg || inst.opcode == TokenKind::kOpAbs) &&
      inst.operands.size() >= 2) {
    const std::string src = TranslateOperand(inst.operands[1]);
    if (inst.opcode == TokenKind::kOpNeg) {
      *body += dest + " = -" + src + ";\n  ";
    } else {
      *body += dest + " = abs(" + src + ");\n  ";
    }
    return;
  }

  // Handle bitwise ops: and, or, xor, not, shl, shr
  if (inst.opcode == TokenKind::kOpNot && inst.operands.size() >= 2) {
    const std::string src = TranslateOperand(inst.operands[1]);
    *body += dest + " = ~" + src + ";\n  ";
    return;
  }
  if ((inst.opcode == TokenKind::kOpAnd || inst.opcode == TokenKind::kOpOr ||
       inst.opcode == TokenKind::kOpXor || inst.opcode == TokenKind::kOpShl ||
       inst.opcode == TokenKind::kOpShr) &&
      inst.operands.size() >= 3) {
    const std::string op1 = TranslateOperand(inst.operands[1]);
    const std::string op2 = TranslateOperand(inst.operands[2]);
    std::string sym = "&";
    if (inst.opcode == TokenKind::kOpOr)
      sym = "|";
    else if (inst.opcode == TokenKind::kOpXor)
      sym = "^";
    else if (inst.opcode == TokenKind::kOpShl)
      sym = "<<";
    else if (inst.opcode == TokenKind::kOpShr)
      sym = ">>";
    *body += dest + " = " + op1 + " " + sym + " " + op2 + ";\n  ";
    return;
  }

  // Handle binary ops (add, sub, mul, div)
  if (inst.operands.size() >= 3) {
    const std::string op1 = TranslateOperand(inst.operands[1]);
    const std::string op2 = TranslateOperand(inst.operands[2]);

    bool is_wide = false;
    for (std::string_view mod : inst.modifiers) {
      if (mod == ".wide")
        is_wide = true;
    }
    if (inst.opcode == TokenKind::kOpMul && is_wide) {
      *body += dest + " = (static_cast<int64_t>(" + op1 + ")) * (" + op2 + ");\n  ";
      return;
    }

    std::string sym = "+";
    if (inst.opcode == TokenKind::kOpSub)
      sym = "-";
    else if (inst.opcode == TokenKind::kOpMul)
      sym = "*";
    else if (inst.opcode == TokenKind::kOpDiv)
      sym = "/";
    *body += dest + " = " + op1 + " " + sym + " " + op2 + ";\n  ";
    return;
  }

  // Handle mov
  if (inst.opcode == TokenKind::kOpMov && inst.operands.size() >= 2) {
    const std::string src = TranslateOperand(inst.operands[1]);
    if (types.IsPointer(inst.operands[1].name) &&
        (inst.operands[0].name.starts_with("%rd") || inst.operands[0].name.starts_with("rd"))) {
      *body += dest + " = reinterpret_cast<uint64_t>(" + src + ");\n  ";
    } else {
      *body += dest + " = " + src + ";\n  ";
    }
    return;
  }
}

void DeclareRegisters(const Function& fn, const TypeInference& types, std::string* body) {
  std::unordered_set<std::string> declared;

  for (const Statement& stmt : fn.statements) {
    if (stmt.kind != StatementKind::kInstruction)
      continue;
    for (const Operand& op : stmt.instruction.operands) {
      if (op.kind == OperandKind::kRegister) {
        const std::string name = SanitizeName(op.name);
        if (declared.insert(name).second) {
          const DataType dt = types.GetDataType(op.name);
          *body += std::string(MslTypeName(dt)) + " " + name + " = 0;\n  ";
        }
      }
    }
    if (stmt.instruction.predicate.has_value()) {
      const std::string pred_name = SanitizeName(stmt.instruction.predicate->reg_name);
      if (declared.insert(pred_name).second) {
        *body += "bool " + pred_name + " = false;\n  ";
      }
    }
  }
}

}  // namespace

mmir::Module LowerPtxToMmir(const Program& program, const TranslationOptions& options) {
  mmir::Module module;
  module.fp64_policy = options.fp64_policy;

  for (const Function& fn : program.functions) {
    const TypeInference types = TypeInference::Analyze(fn);
    const ControlFlowGraph cfg = ControlFlowGraph::Build(fn);
    const StructuredFunction structured = Structurizer::Structurize(fn, cfg);

    mmir::KernelFunction kernel_fn;
    kernel_fn.name = std::string(fn.name);
    kernel_fn.msl_name = "k_" + std::string(fn.name);

    std::size_t buffer_idx = 0;
    for (const VariableDecl& param : fn.params) {
      KernelParameter kp;
      kp.name = std::string(param.name);
      kp.type = types.GetDataType(param.name);
      kp.buffer_index = buffer_idx++;

      if (types.IsPointer(param.name) || param.type == TypeKind::kU64 ||
          param.type == TypeKind::kB64) {
        kp.binding_kind = ParameterBindingKind::kBufferPointer;
        kp.address_space = AddressSpace::kDevice;
        kp.access = AccessQualifier::kReadWrite;
      } else {
        kp.binding_kind = ParameterBindingKind::kConstantValue;
        kp.address_space = AddressSpace::kConstant;
        kp.access = AccessQualifier::kReadOnly;
      }

      kernel_fn.parameters.push_back(std::move(kp));
    }

    // Shared allocations
    for (const VariableDecl& decl : fn.declarations) {
      if (decl.space == StateSpace::kShared) {
        mmir::SharedAllocation sa;
        sa.name = std::string(decl.name);
        sa.element_type = types.GetDataType(decl.name);
        sa.count_expression = std::to_string(decl.count);
        sa.is_dynamic = false;
        kernel_fn.shared_allocations.push_back(std::move(sa));
      }
    }

    // Lower body
    std::string body;
    body += "[[maybe_unused]] const auto tid = thread_position_in_threadgroup;\n  ";
    body += "[[maybe_unused]] const auto tgid = threadgroup_position_in_grid;\n  ";
    body += "[[maybe_unused]] const auto tptg = threads_per_threadgroup;\n  ";
    body += "[[maybe_unused]] const auto tpg = threadgroups_per_grid;\n  ";
    DeclareRegisters(fn, types, &body);

    for (const Statement& stmt : fn.statements) {
      if (stmt.kind == StatementKind::kInstruction) {
        LowerInstruction(stmt.instruction, types, &body);
      }
    }

    kernel_fn.lowered_body = std::move(body);
    module.kernels.push_back(std::move(kernel_fn));
  }

  return module;
}

TranslationResult TranslatePtxToMsl(std::string_view ptx_source,
                                    const TranslationOptions& options) {
  Arena arena;
  Lexer lexer(ptx_source);
  Parser parser(lexer, arena);

  const ParseResult pr = parser.ParseProgram();
  if (!pr.ok()) {
    TranslationResult failure;
    failure.code = DiagnosticCode::kMalformedKernelDeclaration;
    return failure;
  }

  mmir::Module module = LowerPtxToMmir(pr.program, options);
  const DiagnosticCode verify_code = mmir::VerifyModule(module);
  if (verify_code != DiagnosticCode::kOk) {
    TranslationResult failure;
    failure.code = verify_code;
    return failure;
  }

  TranslationResult result;
  result.code = DiagnosticCode::kOk;
  result.msl = mmir::EmitMslModule(module);

  for (const mmir::KernelFunction& kf : module.kernels) {
    TranslatedKernel tk;
    tk.name = kf.name;
    tk.msl_name = kf.msl_name;
    tk.parameters = kf.parameters;
    tk.msl = mmir::EmitMslKernel(module, kf);
    result.kernels.push_back(std::move(tk));
  }
  if (!result.kernels.empty()) {
    result.kernel = result.kernels[0];
  }

  return result;
}

}  // namespace muda::compiler::ptx

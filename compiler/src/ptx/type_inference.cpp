/**
 * @file type_inference.cpp
 * @brief Implementation of PTX type and address space inference.
 */

#include "muda/compiler/ptx/type_inference.hpp"

#include <string_view>
#include <unordered_map>

#include "muda/compiler/ptx/ast.hpp"
#include "muda/compiler/types.hpp"

namespace muda::compiler::ptx {
namespace {

DataType ConvertPtxType(TypeKind kind) noexcept {
  switch (kind) {
    case TypeKind::kPred:
      return DataType{ScalarType::kBool, VectorWidth::k1};
    case TypeKind::kB8:
    case TypeKind::kS8:
      return DataType{ScalarType::kInt8, VectorWidth::k1};
    case TypeKind::kU8:
      return DataType{ScalarType::kUInt8, VectorWidth::k1};
    case TypeKind::kB16:
    case TypeKind::kS16:
      return DataType{ScalarType::kInt16, VectorWidth::k1};
    case TypeKind::kU16:
      return DataType{ScalarType::kUInt16, VectorWidth::k1};
    case TypeKind::kB32:
    case TypeKind::kS32:
      return DataType{ScalarType::kInt32, VectorWidth::k1};
    case TypeKind::kU32:
      return DataType{ScalarType::kUInt32, VectorWidth::k1};
    case TypeKind::kB64:
    case TypeKind::kS64:
      return DataType{ScalarType::kInt64, VectorWidth::k1};
    case TypeKind::kU64:
      return DataType{ScalarType::kUInt64, VectorWidth::k1};
    case TypeKind::kF16:
      return DataType{ScalarType::kFloat16, VectorWidth::k1};
    case TypeKind::kF16x2:
      return DataType{ScalarType::kFloat16, VectorWidth::k2};
    case TypeKind::kF32:
      return DataType{ScalarType::kFloat32, VectorWidth::k1};
    case TypeKind::kF64:
      return DataType{ScalarType::kFloat64, VectorWidth::k1};
  }
  return DataType{ScalarType::kFloat32, VectorWidth::k1};
}

}  // namespace

const InferredType* TypeInference::Lookup(std::string_view reg_name) const noexcept {
  const auto it = reg_types_.find(reg_name);
  return (it != reg_types_.end()) ? &it->second : nullptr;
}

AddressSpace TypeInference::GetAddressSpace(std::string_view reg_name) const noexcept {
  const InferredType* info = Lookup(reg_name);
  return info ? info->address_space : AddressSpace::kThread;
}

DataType TypeInference::GetDataType(std::string_view reg_name) const noexcept {
  const InferredType* info = Lookup(reg_name);
  if (info) {
    return info->data_type;
  }
  // Standard PTX register naming conventions (raw %r or sanitized r)
  if (reg_name.starts_with("%p") || reg_name.starts_with("p")) {
    return DataType{ScalarType::kBool, VectorWidth::k1};
  }
  if (reg_name.starts_with("%rd") || reg_name.starts_with("rd")) {
    return DataType{ScalarType::kUInt64, VectorWidth::k1};
  }
  if (reg_name.starts_with("%r") || reg_name.starts_with("r")) {
    return DataType{ScalarType::kInt32, VectorWidth::k1};
  }
  if (reg_name.starts_with("%fd") || reg_name.starts_with("fd")) {
    return DataType{ScalarType::kFloat64, VectorWidth::k1};
  }
  if (reg_name.starts_with("%f") || reg_name.starts_with("f")) {
    return DataType{ScalarType::kFloat32, VectorWidth::k1};
  }
  return DataType{ScalarType::kFloat32, VectorWidth::k1};
}

bool TypeInference::IsPointer(std::string_view reg_name) const noexcept {
  const InferredType* info = Lookup(reg_name);
  return info ? info->is_pointer : false;
}

void TypeInference::SeedDeclarations(const Function& fn) {
  // Parameters
  for (const VariableDecl& param : fn.params) {
    InferredType info;
    info.data_type = ConvertPtxType(param.type);
    if (param.type == TypeKind::kU64 || param.type == TypeKind::kB64) {
      info.is_pointer = true;
      info.address_space = AddressSpace::kDevice;
    }
    reg_types_[param.name] = info;
  }

  // Local declarations (register arrays and shared memory)
  for (const VariableDecl& decl : fn.declarations) {
    InferredType info;
    info.data_type = ConvertPtxType(decl.type);
    if (decl.space == StateSpace::kShared) {
      info.is_pointer = true;
      info.address_space = AddressSpace::kThreadgroup;
    } else {
      info.address_space = AddressSpace::kThread;
    }
    reg_types_[decl.name] = info;
  }
}

void TypeInference::PropagateInstructions(const Function& fn) {
  // Fixed-point propagation pass over instructions
  bool changed = true;
  std::size_t passes = 0;

  while (changed && passes < 10) {
    changed = false;
    ++passes;

    for (const Statement& stmt : fn.statements) {
      if (stmt.kind != StatementKind::kInstruction) {
        continue;
      }

      const Instruction& inst = stmt.instruction;

      // Handle parameter loads: ld.param.u64 %rd1, [param_a]
      if (inst.opcode == TokenKind::kOpLd && !inst.operands.empty()) {
        bool is_param = false;
        for (std::string_view mod : inst.modifiers) {
          if (mod == ".param")
            is_param = true;
        }
        const std::string_view dest_reg = inst.operands[0].name;
        if (is_param && inst.operands.size() >= 2 &&
            inst.operands[1].kind == OperandKind::kAddress) {
          const std::string_view src_param = inst.operands[1].name;
          const InferredType* src_info = Lookup(src_param);
          if (src_info && src_info->is_pointer) {
            InferredType& dest_info = reg_types_[dest_reg];
            if (!dest_info.is_pointer || dest_info.address_space != src_info->address_space) {
              dest_info.is_pointer = true;
              dest_info.address_space = src_info->address_space;
              dest_info.data_type = src_info->data_type;
              changed = true;
            }
          }
        } else if (!is_param) {
          // Regular load (ld.global / ld.shared): result has type inst.type and is not a pointer
          InferredType& dest_info = reg_types_[dest_reg];
          const DataType dt = ConvertPtxType(inst.type);
          if (dest_info.data_type.scalar != dt.scalar || dest_info.is_pointer) {
            dest_info.data_type = dt;
            dest_info.is_pointer = false;
            dest_info.address_space = AddressSpace::kThread;
            changed = true;
          }
        }
      }

      // Handle address space conversions: cvta.to.global / cvta.to.shared
      if (inst.opcode == TokenKind::kOpCvta && !inst.operands.empty()) {
        const std::string_view dest_reg = inst.operands[0].name;
        InferredType& dest_info = reg_types_[dest_reg];
        dest_info.is_pointer = true;
        dest_info.data_type = DataType{ScalarType::kUInt64, VectorWidth::k1};

        for (std::string_view mod : inst.modifiers) {
          if (mod == ".global") {
            dest_info.address_space = AddressSpace::kDevice;
          } else if (mod == ".shared") {
            dest_info.address_space = AddressSpace::kThreadgroup;
          } else if (mod == ".const") {
            dest_info.address_space = AddressSpace::kConstant;
          }
        }
        changed = true;
      }

      // Handle pointer arithmetic: add.s64 %rd5, %rd1, %rd4
      if ((inst.opcode == TokenKind::kOpAdd || inst.opcode == TokenKind::kOpSub) &&
          inst.operands.size() >= 3) {
        const std::string_view dest_reg = inst.operands[0].name;
        const std::string_view src_reg1 = inst.operands[1].name;
        const std::string_view src_reg2 = inst.operands[2].name;

        const InferredType* info1 = Lookup(src_reg1);
        const InferredType* info2 = Lookup(src_reg2);

        const InferredType* ptr_info = (info1 && info1->is_pointer)   ? info1
                                       : (info2 && info2->is_pointer) ? info2
                                                                      : nullptr;

        if (ptr_info) {
          InferredType& dest_info = reg_types_[dest_reg];
          if (!dest_info.is_pointer || dest_info.address_space != ptr_info->address_space) {
            dest_info.is_pointer = true;
            dest_info.address_space = ptr_info->address_space;
            dest_info.data_type = ptr_info->data_type;
            changed = true;
          }
        }
      }

      // Direct register-to-register moves: mov %dest, %src
      if (inst.opcode == TokenKind::kOpMov && inst.operands.size() >= 2) {
        const std::string_view dest_reg = inst.operands[0].name;
        const std::string_view src_reg = inst.operands[1].name;
        const InferredType* src_info = Lookup(src_reg);

        if (src_info && src_info->is_pointer) {
          InferredType& dest_info = reg_types_[dest_reg];
          if (!dest_info.is_pointer || dest_info.address_space != src_info->address_space) {
            dest_info = *src_info;
            if (dest_reg.starts_with("%rd") || dest_reg.starts_with("rd")) {
              dest_info.data_type = DataType{ScalarType::kUInt64, VectorWidth::k1};
            }
            changed = true;
          }
        }
      }
    }
  }
}

TypeInference TypeInference::Analyze(const Function& fn) {
  TypeInference engine;
  engine.SeedDeclarations(fn);
  engine.PropagateInstructions(fn);
  return engine;
}

}  // namespace muda::compiler::ptx

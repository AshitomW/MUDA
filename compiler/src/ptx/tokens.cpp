/**
 * @file tokens.cpp
 * @brief Token classifications and stringification for PTX tokens.
 */

#include "muda/compiler/ptx/tokens.hpp"

#include <string_view>

namespace muda::compiler::ptx {

std::string_view TokenKindName(TokenKind kind) noexcept {
  switch (kind) {
    case TokenKind::kUnknown:
      return "unknown";
    case TokenKind::kEndOfFile:
      return "end-of-file";

    case TokenKind::kDirectiveVersion:
      return ".version";
    case TokenKind::kDirectiveTarget:
      return ".target";
    case TokenKind::kDirectiveAddressSize:
      return ".address_size";
    case TokenKind::kDirectiveVisible:
      return ".visible";
    case TokenKind::kDirectiveEntry:
      return ".entry";
    case TokenKind::kDirectiveFunc:
      return ".func";
    case TokenKind::kDirectiveParam:
      return ".param";
    case TokenKind::kDirectiveReg:
      return ".reg";
    case TokenKind::kDirectiveShared:
      return ".shared";
    case TokenKind::kDirectiveGlobal:
      return ".global";
    case TokenKind::kDirectiveConst:
      return ".const";
    case TokenKind::kDirectiveLocal:
      return ".local";
    case TokenKind::kDirectiveAlign:
      return ".align";
    case TokenKind::kDirectiveWeak:
      return ".weak";
    case TokenKind::kDirectiveExtern:
      return ".extern";
    case TokenKind::kDirectiveFile:
      return ".file";
    case TokenKind::kDirectiveLoc:
      return ".loc";

    case TokenKind::kTypePred:
      return ".pred";
    case TokenKind::kTypeB8:
      return ".b8";
    case TokenKind::kTypeB16:
      return ".b16";
    case TokenKind::kTypeB32:
      return ".b32";
    case TokenKind::kTypeB64:
      return ".b64";
    case TokenKind::kTypeU8:
      return ".u8";
    case TokenKind::kTypeU16:
      return ".u16";
    case TokenKind::kTypeU32:
      return ".u32";
    case TokenKind::kTypeU64:
      return ".u64";
    case TokenKind::kTypeS8:
      return ".s8";
    case TokenKind::kTypeS16:
      return ".s16";
    case TokenKind::kTypeS32:
      return ".s32";
    case TokenKind::kTypeS64:
      return ".s64";
    case TokenKind::kTypeF16:
      return ".f16";
    case TokenKind::kTypeF16x2:
      return ".f16x2";
    case TokenKind::kTypeF32:
      return ".f32";
    case TokenKind::kTypeF64:
      return ".f64";

    case TokenKind::kOpMov:
      return "mov";
    case TokenKind::kOpLd:
      return "ld";
    case TokenKind::kOpSt:
      return "st";
    case TokenKind::kOpCvta:
      return "cvta";
    case TokenKind::kOpCvt:
      return "cvt";
    case TokenKind::kOpSelp:
      return "selp";
    case TokenKind::kOpAdd:
      return "add";
    case TokenKind::kOpSub:
      return "sub";
    case TokenKind::kOpMul:
      return "mul";
    case TokenKind::kOpMad:
      return "mad";
    case TokenKind::kOpFma:
      return "fma";
    case TokenKind::kOpDiv:
      return "div";
    case TokenKind::kOpRem:
      return "rem";
    case TokenKind::kOpNeg:
      return "neg";
    case TokenKind::kOpAbs:
      return "abs";
    case TokenKind::kOpMin:
      return "min";
    case TokenKind::kOpMax:
      return "max";
    case TokenKind::kOpRsqrt:
      return "rsqrt";
    case TokenKind::kOpSqrt:
      return "sqrt";
    case TokenKind::kOpSin:
      return "sin";
    case TokenKind::kOpCos:
      return "cos";
    case TokenKind::kOpRcp:
      return "rcp";
    case TokenKind::kOpEx2:
      return "ex2";
    case TokenKind::kOpLg2:
      return "lg2";
    case TokenKind::kOpAnd:
      return "and";
    case TokenKind::kOpOr:
      return "or";
    case TokenKind::kOpXor:
      return "xor";
    case TokenKind::kOpNot:
      return "not";
    case TokenKind::kOpShl:
      return "shl";
    case TokenKind::kOpShr:
      return "shr";
    case TokenKind::kOpCnot:
      return "cnot";
    case TokenKind::kOpSetp:
      return "setp";
    case TokenKind::kOpBra:
      return "bra";
    case TokenKind::kOpRet:
      return "ret";
    case TokenKind::kOpCall:
      return "call";
    case TokenKind::kOpExit:
      return "exit";
    case TokenKind::kOpBar:
      return "bar";
    case TokenKind::kOpAtom:
      return "atom";
    case TokenKind::kOpRed:
      return "red";
    case TokenKind::kOpVote:
      return "vote";
    case TokenKind::kOpShfl:
      return "shfl";

    case TokenKind::kIdentifier:
      return "identifier";
    case TokenKind::kRegister:
      return "register";
    case TokenKind::kIntegerLiteral:
      return "integer-literal";
    case TokenKind::kFloatLiteral:
      return "float-literal";
    case TokenKind::kStringLiteral:
      return "string-literal";

    case TokenKind::kSpecialTid:
      return "%tid";
    case TokenKind::kSpecialNtid:
      return "%ntid";
    case TokenKind::kSpecialCtaid:
      return "%ctaid";
    case TokenKind::kSpecialNctaid:
      return "%nctaid";
    case TokenKind::kSpecialLaneId:
      return "%laneid";
    case TokenKind::kSpecialWarpId:
      return "%warpid";
    case TokenKind::kSpecialClock:
      return "%clock";
    case TokenKind::kSpecialClock64:
      return "%clock64";

    case TokenKind::kSemicolon:
      return ";";
    case TokenKind::kComma:
      return ",";
    case TokenKind::kColon:
      return ":";
    case TokenKind::kLeftBrace:
      return "{";
    case TokenKind::kRightBrace:
      return "}";
    case TokenKind::kLeftParen:
      return "(";
    case TokenKind::kRightParen:
      return ")";
    case TokenKind::kLeftBracket:
      return "[";
    case TokenKind::kRightBracket:
      return "]";
    case TokenKind::kLeftAngle:
      return "<";
    case TokenKind::kRightAngle:
      return ">";
    case TokenKind::kEquals:
      return "=";
    case TokenKind::kAt:
      return "@";
    case TokenKind::kExclamation:
      return "!";
    case TokenKind::kDot:
      return ".";
    case TokenKind::kPlus:
      return "+";
    case TokenKind::kMinus:
      return "-";
  }
  return "unknown";
}

bool IsDirective(TokenKind kind) noexcept {
  return kind >= TokenKind::kDirectiveVersion && kind <= TokenKind::kDirectiveLoc;
}

bool IsType(TokenKind kind) noexcept {
  return kind >= TokenKind::kTypePred && kind <= TokenKind::kTypeF64;
}

bool IsInstruction(TokenKind kind) noexcept {
  return kind >= TokenKind::kOpMov && kind <= TokenKind::kOpShfl;
}

bool IsSpecialRegister(TokenKind kind) noexcept {
  return kind >= TokenKind::kSpecialTid && kind <= TokenKind::kSpecialClock64;
}

}  // namespace muda::compiler::ptx

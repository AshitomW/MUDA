/**
 * @file parser.cpp
 * @brief Implementation of table-driven LALR(1) shift-reduce parser for PTX.
 */

#include "muda/compiler/ptx/parser.hpp"

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "muda/compiler/ptx/ast.hpp"
#include "muda/compiler/ptx/lexer.hpp"
#include "muda/compiler/ptx/tokens.hpp"

namespace muda::compiler::ptx {
namespace {

TypeKind TokenToType(TokenKind kind) noexcept {
  switch (kind) {
    case TokenKind::kTypePred:
      return TypeKind::kPred;
    case TokenKind::kTypeB8:
      return TypeKind::kB8;
    case TokenKind::kTypeB16:
      return TypeKind::kB16;
    case TokenKind::kTypeB32:
      return TypeKind::kB32;
    case TokenKind::kTypeB64:
      return TypeKind::kB64;
    case TokenKind::kTypeU8:
      return TypeKind::kU8;
    case TokenKind::kTypeU16:
      return TypeKind::kU16;
    case TokenKind::kTypeU32:
      return TypeKind::kU32;
    case TokenKind::kTypeU64:
      return TypeKind::kU64;
    case TokenKind::kTypeS8:
      return TypeKind::kS8;
    case TokenKind::kTypeS16:
      return TypeKind::kS16;
    case TokenKind::kTypeS32:
      return TypeKind::kS32;
    case TokenKind::kTypeS64:
      return TypeKind::kS64;
    case TokenKind::kTypeF16:
      return TypeKind::kF16;
    case TokenKind::kTypeF16x2:
      return TypeKind::kF16x2;
    case TokenKind::kTypeF32:
      return TypeKind::kF32;
    case TokenKind::kTypeF64:
      return TypeKind::kF64;
    default:
      return TypeKind::kB32;
  }
}

StateSpace TokenToSpace(TokenKind kind) noexcept {
  switch (kind) {
    case TokenKind::kDirectiveReg:
      return StateSpace::kReg;
    case TokenKind::kDirectiveParam:
      return StateSpace::kParam;
    case TokenKind::kDirectiveShared:
      return StateSpace::kShared;
    case TokenKind::kDirectiveGlobal:
      return StateSpace::kGlobal;
    case TokenKind::kDirectiveConst:
      return StateSpace::kConst;
    case TokenKind::kDirectiveLocal:
      return StateSpace::kLocal;
    default:
      return StateSpace::kGeneric;
  }
}

}  // namespace

Parser::Parser(Lexer& lexer, Arena& arena) noexcept : lexer_(lexer), arena_(arena) {}

Parser::Action Parser::QueryAction(StateId state, const Token& lookahead) const noexcept {
  const TokenKind kind = lookahead.kind;
  switch (state) {
    case 0:  // Top-level state
      if (kind == TokenKind::kDirectiveVersion)
        return Action{ActionKind::kShift, 1};
      if (kind == TokenKind::kDirectiveTarget)
        return Action{ActionKind::kShift, 2};
      if (kind == TokenKind::kDirectiveAddressSize)
        return Action{ActionKind::kShift, 3};
      if (kind == TokenKind::kDirectiveVisible || kind == TokenKind::kDirectiveWeak) {
        return Action{ActionKind::kShift, 4};
      }
      if (kind == TokenKind::kDirectiveEntry)
        return Action{ActionKind::kShift, 5};
      if (kind == TokenKind::kDirectiveFunc)
        return Action{ActionKind::kShift, 6};
      if (kind == TokenKind::kDirectiveGlobal || kind == TokenKind::kDirectiveConst ||
          kind == TokenKind::kDirectiveShared) {
        return Action{ActionKind::kShift, 7};
      }
      if (kind == TokenKind::kEndOfFile)
        return Action{ActionKind::kAccept, 0};
      return Action{ActionKind::kError, 0};

    case 1:  // Saw .version
      if (kind == TokenKind::kFloatLiteral || kind == TokenKind::kIntegerLiteral) {
        return Action{ActionKind::kShift, 11};
      }
      return Action{ActionKind::kError, 0};

    case 11:  // Saw .version <val>
      return Action{ActionKind::kReduce, 1};

    case 2:  // Saw .target
      if (kind == TokenKind::kIdentifier)
        return Action{ActionKind::kShift, 12};
      return Action{ActionKind::kError, 0};

    case 12:  // Saw .target <val>
      return Action{ActionKind::kReduce, 2};

    case 3:  // Saw .address_size
      if (kind == TokenKind::kIntegerLiteral)
        return Action{ActionKind::kShift, 13};
      return Action{ActionKind::kError, 0};

    case 13:  // Saw .address_size <val>
      return Action{ActionKind::kReduce, 3};

    case 4:  // Saw .visible / .weak
      if (kind == TokenKind::kDirectiveEntry)
        return Action{ActionKind::kShift, 5};
      if (kind == TokenKind::kDirectiveFunc)
        return Action{ActionKind::kShift, 6};
      if (kind == TokenKind::kDirectiveGlobal || kind == TokenKind::kDirectiveConst ||
          kind == TokenKind::kDirectiveShared) {
        return Action{ActionKind::kShift, 7};
      }
      return Action{ActionKind::kError, 0};

    case 5:  // Saw .entry
    case 6:  // Saw .func
      if (kind == TokenKind::kIdentifier)
        return Action{ActionKind::kShift, 20};
      return Action{ActionKind::kError, 0};

    case 7:  // Inside global variable declaration
      if (kind == TokenKind::kSemicolon)
        return Action{ActionKind::kShift, 17};
      if (kind != TokenKind::kEndOfFile)
        return Action{ActionKind::kShift, 7};
      return Action{ActionKind::kError, 0};

    case 17:  // Semicolon of global variable declaration
      return Action{ActionKind::kReduce, 4};

    case 20:  // Saw function name
      if (kind == TokenKind::kLeftParen)
        return Action{ActionKind::kShift, 21};
      if (kind == TokenKind::kLeftBrace)
        return Action{ActionKind::kShift, 30};
      return Action{ActionKind::kError, 0};

    case 21:  // Inside parameter list
      if (kind == TokenKind::kDirectiveParam)
        return Action{ActionKind::kShift, 22};
      if (kind == TokenKind::kRightParen)
        return Action{ActionKind::kShift, 29};
      if (kind == TokenKind::kComma)
        return Action{ActionKind::kShift, 21};
      return Action{ActionKind::kError, 0};

    case 22:  // Saw .param in parameter list
      if (kind == TokenKind::kDirectiveAlign)
        return Action{ActionKind::kShift, 28};
      if (IsType(kind))
        return Action{ActionKind::kShift, 23};
      return Action{ActionKind::kError, 0};

    case 28:  // Saw .align in param
      if (kind == TokenKind::kIntegerLiteral)
        return Action{ActionKind::kShift, 22};
      return Action{ActionKind::kError, 0};

    case 23:  // Saw param type
      if (kind == TokenKind::kIdentifier || kind == TokenKind::kRegister) {
        return Action{ActionKind::kShift, 24};
      }
      return Action{ActionKind::kError, 0};

    case 24:  // Saw param name
      if (kind == TokenKind::kLeftBracket)
        return Action{ActionKind::kShift, 25};
      if (kind == TokenKind::kComma)
        return Action{ActionKind::kShift, 21};
      if (kind == TokenKind::kRightParen)
        return Action{ActionKind::kShift, 29};
      return Action{ActionKind::kError, 0};

    case 25:  // Saw '[' in param array
      if (kind == TokenKind::kIntegerLiteral)
        return Action{ActionKind::kShift, 26};
      if (kind == TokenKind::kRightBracket)
        return Action{ActionKind::kShift, 24};
      return Action{ActionKind::kError, 0};

    case 26:  // Saw count in param array
      if (kind == TokenKind::kRightBracket)
        return Action{ActionKind::kShift, 24};
      return Action{ActionKind::kError, 0};

    case 29:  // Closed parameter list ')'
      if (kind == TokenKind::kLeftBrace)
        return Action{ActionKind::kShift, 30};
      return Action{ActionKind::kError, 0};

    case 30:  // Inside function body '{'
      if (kind == TokenKind::kDirectiveReg || kind == TokenKind::kDirectiveShared ||
          kind == TokenKind::kDirectiveLocal) {
        return Action{ActionKind::kShift, 31};
      }
      if (kind == TokenKind::kAt)
        return Action{ActionKind::kShift, 33};
      if (IsInstruction(kind))
        return Action{ActionKind::kShift, 60};
      if (kind == TokenKind::kIdentifier || kind == TokenKind::kRegister) {
        return Action{ActionKind::kShift, 32};
      }
      if (kind == TokenKind::kRightBrace)
        return Action{ActionKind::kShift, 40};
      return Action{ActionKind::kError, 0};

    case 31:  // Inside local variable declaration (saw .reg / .shared / .local)
      if (kind == TokenKind::kDirectiveAlign)
        return Action{ActionKind::kShift, 45};
      if (IsType(kind))
        return Action{ActionKind::kShift, 46};
      return Action{ActionKind::kError, 0};

    case 45:  // Saw .align in local decl
      if (kind == TokenKind::kIntegerLiteral)
        return Action{ActionKind::kShift, 31};
      return Action{ActionKind::kError, 0};

    case 46:  // Saw local decl type (.b32, .pred, etc.)
      if (kind == TokenKind::kIdentifier || kind == TokenKind::kRegister) {
        return Action{ActionKind::kShift, 47};
      }
      return Action{ActionKind::kError, 0};

    case 47:  // Saw local decl variable/register name
      if (kind == TokenKind::kLeftAngle)
        return Action{ActionKind::kShift, 48};
      if (kind == TokenKind::kLeftBracket)
        return Action{ActionKind::kShift, 49};
      if (kind == TokenKind::kComma)
        return Action{ActionKind::kShift, 46};
      if (kind == TokenKind::kSemicolon)
        return Action{ActionKind::kShift, 35};
      return Action{ActionKind::kError, 0};

    case 48:  // Saw '<' in register range (%r<16>)
      if (kind == TokenKind::kIntegerLiteral)
        return Action{ActionKind::kShift, 52};
      return Action{ActionKind::kError, 0};

    case 52:  // Saw count in register range
      if (kind == TokenKind::kRightAngle)
        return Action{ActionKind::kShift, 47};
      return Action{ActionKind::kError, 0};

    case 49:  // Saw '[' in shared array (s_cache[256])
      if (kind == TokenKind::kIntegerLiteral)
        return Action{ActionKind::kShift, 53};
      return Action{ActionKind::kError, 0};

    case 53:  // Saw count in shared array
      if (kind == TokenKind::kRightBracket)
        return Action{ActionKind::kShift, 47};
      return Action{ActionKind::kError, 0};

    case 35:  // Semicolon of local variable declaration
      return Action{ActionKind::kReduce, 6};

    case 32:  // Label identifier
      if (kind == TokenKind::kColon)
        return Action{ActionKind::kShift, 36};
      return Action{ActionKind::kError, 0};

    case 36:  // Saw ':' of label
      return Action{ActionKind::kReduce, 7};

    case 33:  // Saw '@' predicate
      if (kind == TokenKind::kExclamation)
        return Action{ActionKind::kShift, 38};
      if (kind == TokenKind::kRegister || kind == TokenKind::kIdentifier) {
        return Action{ActionKind::kShift, 39};
      }
      return Action{ActionKind::kError, 0};

    case 38:  // Saw '!' in predicate
      if (kind == TokenKind::kRegister || kind == TokenKind::kIdentifier) {
        return Action{ActionKind::kShift, 39};
      }
      return Action{ActionKind::kError, 0};

    case 39:  // Saw predicate register name
      if (IsInstruction(kind))
        return Action{ActionKind::kShift, 60};
      return Action{ActionKind::kError, 0};

    case 60:  // Opcode of instruction (saw mov, add, ld, st, etc.)
      if (IsType(kind) || kind == TokenKind::kDirectiveParam ||
          kind == TokenKind::kDirectiveGlobal || kind == TokenKind::kDirectiveShared ||
          lookahead.text.starts_with(".")) {
        return Action{ActionKind::kShift, 61};
      }
      if (kind == TokenKind::kRegister || IsSpecialRegister(kind) ||
          kind == TokenKind::kIdentifier || kind == TokenKind::kIntegerLiteral ||
          kind == TokenKind::kFloatLiteral) {
        return Action{ActionKind::kShift, 62};
      }
      if (kind == TokenKind::kLeftBracket)
        return Action{ActionKind::kShift, 63};
      if (kind == TokenKind::kSemicolon)
        return Action{ActionKind::kShift, 37};
      return Action{ActionKind::kError, 0};

    case 61:  // Instruction modifier or type
      if (IsType(kind) || kind == TokenKind::kDirectiveParam ||
          kind == TokenKind::kDirectiveGlobal || kind == TokenKind::kDirectiveShared ||
          lookahead.text.starts_with(".")) {
        return Action{ActionKind::kShift, 61};
      }
      if (kind == TokenKind::kRegister || IsSpecialRegister(kind) ||
          kind == TokenKind::kIdentifier || kind == TokenKind::kIntegerLiteral ||
          kind == TokenKind::kFloatLiteral) {
        return Action{ActionKind::kShift, 62};
      }
      if (kind == TokenKind::kLeftBracket)
        return Action{ActionKind::kShift, 63};
      if (kind == TokenKind::kSemicolon)
        return Action{ActionKind::kShift, 37};
      return Action{ActionKind::kError, 0};

    case 62:  // Saw instruction operand
      if (kind == TokenKind::kComma)
        return Action{ActionKind::kShift, 60};
      if (kind == TokenKind::kSemicolon)
        return Action{ActionKind::kShift, 37};
      return Action{ActionKind::kError, 0};

    case 63:  // Saw '[' in address operand
      if (kind == TokenKind::kRegister || kind == TokenKind::kIdentifier) {
        return Action{ActionKind::kShift, 64};
      }
      return Action{ActionKind::kError, 0};

    case 64:  // Saw base register in address operand
      if (kind == TokenKind::kPlus)
        return Action{ActionKind::kShift, 65};
      if (kind == TokenKind::kMinus)
        return Action{ActionKind::kShift, 66};
      if (kind == TokenKind::kRightBracket)
        return Action{ActionKind::kShift, 62};
      return Action{ActionKind::kError, 0};

    case 65:  // Saw '+' in address offset
    case 66:  // Saw '-' in address offset
      if (kind == TokenKind::kIntegerLiteral)
        return Action{ActionKind::kShift, 67};
      return Action{ActionKind::kError, 0};

    case 67:  // Saw offset integer literal in address
      if (kind == TokenKind::kRightBracket)
        return Action{ActionKind::kShift, 62};
      return Action{ActionKind::kError, 0};

    case 37:  // Semicolon of instruction
      return Action{ActionKind::kReduce, 8};

    case 40:  // Closed function body '}'
      return Action{ActionKind::kReduce, 9};

    default:
      return Action{ActionKind::kError, 0};
  }
}

Parser::StateId Parser::QueryGoto(StateId state, TokenKind non_terminal) const noexcept {
  (void)non_terminal;
  if (state <= 6) {
    return 0;  // return to top-level state
  }
  return 30;  // return to function body state
}

void Parser::ExecuteReduce(std::uint16_t rule_id, Program* program, Function* current_function) {
  switch (rule_id) {
    case 1: {  // .version <val>
      if (symbol_stack_.size() >= 2) {
        const std::string_view ver = symbol_stack_[symbol_stack_.size() - 1].token.text;
        const auto dot = ver.find('.');
        if (dot != std::string_view::npos) {
          (void)std::from_chars(ver.data(), ver.data() + dot, program->version_major);
          (void)std::from_chars(ver.data() + dot + 1, ver.data() + ver.size(),
                                program->version_minor);
        }
      }
      break;
    }
    case 2: {  // .target <id>
      if (symbol_stack_.size() >= 2) {
        program->target = symbol_stack_[symbol_stack_.size() - 1].token.text;
      }
      break;
    }
    case 3: {  // .address_size <val>
      if (symbol_stack_.size() >= 2) {
        const std::string_view sz = symbol_stack_[symbol_stack_.size() - 1].token.text;
        (void)std::from_chars(sz.data(), sz.data() + sz.size(), program->address_size);
      }
      break;
    }
    case 7: {  // label statement ($L1:)
      if (symbol_stack_.size() >= 2) {
        const std::string_view label = symbol_stack_[symbol_stack_.size() - 2].token.text;
        current_function->statements.push_back(Statement::MakeLabel(label));
      }
      break;
    }
    case 9: {  // function definition
      program->functions.push_back(std::move(*current_function));
      *current_function = Function{};
      break;
    }
    default:
      break;
  }
}

ParseResult Parser::ParseProgram() {
  ParseResult result;
  state_stack_.clear();
  symbol_stack_.clear();

  state_stack_.push_back(0);
  Token lookahead = lexer_.NextToken();

  Function current_function;
  bool in_function = false;

  VariableDecl current_param{};
  VariableDecl current_decl{};
  Instruction current_inst{};
  std::optional<Predicate> current_pred{};
  std::string_view address_base{};
  std::int64_t address_sign = 1;

  while (true) {
    const StateId current_state = state_stack_.back();
    const Action action = QueryAction(current_state, lookahead);

    if (action.kind == ActionKind::kShift) {
      SemanticSymbol sym;
      sym.token = lookahead;

      // Track function entry point
      if (lookahead.kind == TokenKind::kDirectiveEntry ||
          lookahead.kind == TokenKind::kDirectiveFunc) {
        current_function = Function{};
        current_function.is_entry = (lookahead.kind == TokenKind::kDirectiveEntry);
        in_function = true;
      } else if (in_function && current_state == 5 && lookahead.kind == TokenKind::kIdentifier) {
        current_function.name = lookahead.text;
      } else if (in_function && current_state == 6 && lookahead.kind == TokenKind::kIdentifier) {
        current_function.name = lookahead.text;
      }

      // Track parameter declarations
      if (current_state == 21 && lookahead.kind == TokenKind::kDirectiveParam) {
        current_param = VariableDecl{};
        current_param.space = StateSpace::kParam;
      } else if (current_state == 22 && IsType(lookahead.kind)) {
        current_param.type = TokenToType(lookahead.kind);
      } else if (current_state == 23) {
        current_param.name = lookahead.text;
        current_function.params.push_back(current_param);
      }

      // Track local declarations (.reg / .shared / .local)
      if (current_state == 30 && (lookahead.kind == TokenKind::kDirectiveReg ||
                                  lookahead.kind == TokenKind::kDirectiveShared ||
                                  lookahead.kind == TokenKind::kDirectiveLocal)) {
        current_decl = VariableDecl{};
        current_decl.space = TokenToSpace(lookahead.kind);
      } else if (current_state == 31 && lookahead.kind == TokenKind::kDirectiveAlign) {
        // Will see align int next
      } else if (current_state == 45 && lookahead.kind == TokenKind::kIntegerLiteral) {
        (void)std::from_chars(lookahead.text.data(), lookahead.text.data() + lookahead.text.size(),
                              current_decl.alignment);
      } else if (current_state == 31 && IsType(lookahead.kind)) {
        current_decl.type = TokenToType(lookahead.kind);
      } else if (current_state == 46) {
        current_decl.name = lookahead.text;
      } else if (current_state == 48 && lookahead.kind == TokenKind::kIntegerLiteral) {
        (void)std::from_chars(lookahead.text.data(), lookahead.text.data() + lookahead.text.size(),
                              current_decl.count);
      } else if (current_state == 49 && lookahead.kind == TokenKind::kIntegerLiteral) {
        current_decl.is_array = true;
        (void)std::from_chars(lookahead.text.data(), lookahead.text.data() + lookahead.text.size(),
                              current_decl.count);
      }

      // Track predicates
      if (current_state == 30 && lookahead.kind == TokenKind::kAt) {
        current_pred = Predicate{};
        current_pred->negated = false;
      } else if (current_state == 33 && lookahead.kind == TokenKind::kExclamation) {
        if (current_pred.has_value()) {
          current_pred->negated = true;
        }
      } else if ((current_state == 33 || current_state == 38) &&
                 (lookahead.kind == TokenKind::kRegister ||
                  lookahead.kind == TokenKind::kIdentifier)) {
        if (current_pred.has_value()) {
          current_pred->reg_name = lookahead.text;
        }
      }

      // Track instruction opcodes and modifiers
      if (IsInstruction(lookahead.kind) && (current_state == 30 || current_state == 39)) {
        current_inst = Instruction{};
        current_inst.opcode = lookahead.kind;
        current_inst.location = lookahead.location;
        current_inst.predicate = current_pred;
        current_pred.reset();
      } else if (current_state == 60 || current_state == 61) {
        if (IsType(lookahead.kind)) {
          current_inst.type = TokenToType(lookahead.kind);
        } else if (lookahead.text.starts_with(".")) {
          current_inst.modifiers.push_back(lookahead.text);
        }
      }

      // Track operands
      if ((current_state == 60 || current_state == 61) && !lookahead.text.starts_with(".") &&
          !IsType(lookahead.kind) && !IsDirective(lookahead.kind) &&
          !lookahead.text.starts_with(".") && !IsType(lookahead.kind) &&
          !IsDirective(lookahead.kind) &&
          (lookahead.kind == TokenKind::kRegister || IsSpecialRegister(lookahead.kind) ||
           lookahead.kind == TokenKind::kIdentifier ||
           lookahead.kind == TokenKind::kIntegerLiteral ||
           lookahead.kind == TokenKind::kFloatLiteral)) {
        if (lookahead.kind == TokenKind::kRegister) {
          current_inst.operands.push_back(Operand::MakeRegister(lookahead.text));
        } else if (IsSpecialRegister(lookahead.kind)) {
          current_inst.operands.push_back(Operand::MakeSpecialRegister(lookahead.text));
        } else if (lookahead.kind == TokenKind::kIntegerLiteral) {
          std::int64_t val = 0;
          if (lookahead.text.starts_with("0x") || lookahead.text.starts_with("0X")) {
            val = std::strtoll(lookahead.text.data(), nullptr, 16);
          } else {
            (void)std::from_chars(lookahead.text.data(),
                                  lookahead.text.data() + lookahead.text.size(), val);
          }
          current_inst.operands.push_back(Operand::MakeImmediateInt(val));
        } else if (lookahead.kind == TokenKind::kFloatLiteral) {
          const double val = std::strtod(lookahead.text.data(), nullptr);
          current_inst.operands.push_back(Operand::MakeImmediateFloat(val));
        } else if (lookahead.kind == TokenKind::kIdentifier) {
          current_inst.operands.push_back(Operand::MakeLabel(lookahead.text));
        }
      }

      // Track address operands [base + offset]
      if (current_state == 63) {
        address_base = lookahead.text;
        address_sign = 1;
      } else if (current_state == 64 && lookahead.kind == TokenKind::kMinus) {
        address_sign = -1;
      } else if (current_state == 64 && lookahead.kind == TokenKind::kRightBracket) {
        current_inst.operands.push_back(Operand::MakeAddress(address_base, 0));
      } else if ((current_state == 65 || current_state == 66) &&
                 lookahead.kind == TokenKind::kIntegerLiteral) {
        std::int64_t off = 0;
        (void)std::from_chars(lookahead.text.data(), lookahead.text.data() + lookahead.text.size(),
                              off);
        current_inst.operands.push_back(Operand::MakeAddress(address_base, off * address_sign));
      }

      symbol_stack_.push_back(std::move(sym));
      state_stack_.push_back(action.target);
      lookahead = lexer_.NextToken();
    } else if (action.kind == ActionKind::kReduce) {
      if (action.target == 6) {  // local decl
        current_function.declarations.push_back(current_decl);
      } else if (action.target == 8) {  // instruction
        current_function.statements.push_back(Statement::MakeInstruction(std::move(current_inst)));
      }

      ExecuteReduce(action.target, &result.program, &current_function);

      // Pop state stack based on rule reduction
      if (action.target == 9) {
        state_stack_.clear();
        state_stack_.push_back(0);
        symbol_stack_.clear();
      } else {
        std::size_t pop_count = 1;
        if (action.target == 1 || action.target == 2 || action.target == 3 || action.target == 7) {
          pop_count = 2;
        }

        for (std::size_t i = 0; i < pop_count && !state_stack_.empty(); ++i) {
          state_stack_.pop_back();
          if (!symbol_stack_.empty()) {
            symbol_stack_.pop_back();
          }
        }

        const StateId goto_state = QueryGoto(state_stack_.back(), TokenKind::kUnknown);
        state_stack_.push_back(goto_state);
      }
    } else if (action.kind == ActionKind::kAccept) {
      result.success = true;
      return result;
    } else {
      result.success = false;
      result.error_message =
          "Syntax error at unexpected token: " + std::string(TokenKindName(lookahead.kind)) +
          " ('" + std::string(lookahead.text) + "')";
      result.error_location = lookahead.location;
      return result;
    }
  }
}

}  // namespace muda::compiler::ptx

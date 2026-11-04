/**
 * @file ptx_parser_test.cpp
 * @brief Unit tests for PTX recursive-descent parser and AST construction.
 */

#include <iostream>
#include <string_view>

#include "muda/compiler/ptx/arena.hpp"
#include "muda/compiler/ptx/ast.hpp"
#include "muda/compiler/ptx/cfg.hpp"
#include "muda/compiler/ptx/lexer.hpp"
#include "muda/compiler/ptx/parser.hpp"

namespace {

bool Expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    return false;
  }
  return true;
}

bool TestControlFlowGraph() {
  constexpr std::string_view kPtxSource = R"PTX(
.version 6.5
.target sm_70
.address_size 64

.visible .entry branching_kernel(.param .u32 n)
{
    .reg .b32 %r<5>;
    .reg .pred %p<2>;

    mov.u32 %r0, %tid.x;
    setp.ge.s32 %p1, %r0, 100;
    @%p1 bra $L_exit;

    add.s32 %r1, %r0, 1;

$L_exit:
    ret;
}
)PTX";

  muda::compiler::ptx::Arena arena;
  muda::compiler::ptx::Lexer lexer(kPtxSource);
  muda::compiler::ptx::Parser parser(lexer, arena);

  const muda::compiler::ptx::ParseResult result = parser.ParseProgram();
  if (!Expect(result.ok(), result.error_message.c_str()))
    return false;

  const muda::compiler::ptx::Function& fn = result.program.functions[0];
  const muda::compiler::ptx::ControlFlowGraph cfg =
      muda::compiler::ptx::ControlFlowGraph::Build(fn);

  if (!Expect(cfg.BlockCount() >= 2, "CFG should contain at least 2 basic blocks"))
    return false;
  const muda::compiler::ptx::BasicBlock& entry = cfg.GetBlock(0);
  if (!Expect(entry.successors.size() == 2,
              "Entry block with conditional branch should have 2 successors"))
    return false;

  return true;
}

bool TestParseVectorAddKernel() {
  constexpr std::string_view kPtxSource = R"PTX(
.version 6.5
.target sm_70
.address_size 64

.visible .entry vector_add(
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
    @%p1 bra        $L__BB0_2;

    mul.wide.s32    %rd4, %r0, 4;
    add.s64         %rd5, %rd1, %rd4;
    add.s64         %rd6, %rd2, %rd4;
    ld.global.f32   %f1, [%rd5];
    ld.global.f32   %f2, [%rd6];
    add.f32         %f3, %f1, %f2;

    add.s64         %rd7, %rd3, %rd4;
    st.global.f32   [%rd7], %f3;

$L__BB0_2:
    ret;
}
)PTX";

  muda::compiler::ptx::Arena arena;
  muda::compiler::ptx::Lexer lexer(kPtxSource);
  muda::compiler::ptx::Parser parser(lexer, arena);

  const muda::compiler::ptx::ParseResult result = parser.ParseProgram();
  if (!Expect(result.ok(), result.error_message.c_str()))
    return false;

  const muda::compiler::ptx::Program& prog = result.program;
  if (!Expect(prog.version_major == 6 && prog.version_minor == 5, "Version should be 6.5"))
    return false;
  if (!Expect(prog.target == "sm_70", "Target should be sm_70"))
    return false;
  if (!Expect(prog.address_size == 64, "Address size should be 64"))
    return false;

  if (!Expect(prog.functions.size() == 1, "Should contain exactly 1 function"))
    return false;
  const muda::compiler::ptx::Function& fn = prog.functions[0];
  if (!Expect(fn.name == "vector_add", "Function name should be vector_add"))
    return false;
  if (!Expect(fn.is_entry, "Function should be marked as entry (.entry)"))
    return false;
  if (!Expect(fn.params.size() == 4, "Function should have 4 parameters"))
    return false;
  if (!Expect(fn.params[0].name == "param_a" &&
                  fn.params[0].type == muda::compiler::ptx::TypeKind::kU64,
              "Param 0 should be param_a (u64)"))
    return false;
  if (!Expect(fn.params[3].name == "param_n" &&
                  fn.params[3].type == muda::compiler::ptx::TypeKind::kU32,
              "Param 3 should be param_n (u32)"))
    return false;

  // Check register declarations
  if (!Expect(fn.declarations.size() == 4, "Should have 4 register declarations"))
    return false;
  if (!Expect(fn.declarations[0].space == muda::compiler::ptx::StateSpace::kReg &&
                  fn.declarations[0].count == 2,
              "First decl should be %p<2>"))
    return false;

  // Check statements (instructions and label)
  if (!Expect(!fn.statements.empty(), "Statements should not be empty"))
    return false;

  // Check the label $L__BB0_2
  bool found_label = false;
  for (const auto& stmt : fn.statements) {
    if (stmt.kind == muda::compiler::ptx::StatementKind::kLabel && stmt.label_name == "$L__BB0_2") {
      found_label = true;
      break;
    }
  }
  if (!Expect(found_label, "Should have found $L__BB0_2 label in statements"))
    return false;

  return true;
}

bool TestParseSharedMemoryAndBarriers() {
  constexpr std::string_view kPtxSource = R"PTX(
.version 7.0
.target sm_75
.address_size 64

.visible .entry shared_reduce(
    .param .u64 in_data,
    .param .u64 out_data
)
{
    .shared .align 4 .f32 s_cache[256];
    .reg .b32 %r<4>;
    .reg .f32 %f<2>;

    mov.u32 %r1, %tid.x;
    bar.sync 0;
    ret;
}
)PTX";

  muda::compiler::ptx::Arena arena;
  muda::compiler::ptx::Lexer lexer(kPtxSource);
  muda::compiler::ptx::Parser parser(lexer, arena);

  const muda::compiler::ptx::ParseResult result = parser.ParseProgram();
  if (!Expect(result.ok(), result.error_message.c_str()))
    return false;

  const muda::compiler::ptx::Function& fn = result.program.functions[0];
  if (!Expect(fn.declarations.size() == 3, "Should have shared memory and register declarations"))
    return false;
  if (!Expect(fn.declarations[0].space == muda::compiler::ptx::StateSpace::kShared &&
                  fn.declarations[0].alignment == 4 && fn.declarations[0].count == 256,
              "Should have shared array s_cache[256] with align 4"))
    return false;

  return true;
}

}  // namespace

int main() {
  if (!TestParseVectorAddKernel())
    return 1;
  if (!TestParseSharedMemoryAndBarriers())
    return 1;
  if (!TestControlFlowGraph())
    return 1;

  std::cout << "ptx_parser_test: all tests passed!\n";
  return 0;
}

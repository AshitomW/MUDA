/**
 * @file type_table.cpp
 * @brief CUDA-to-Metal type and intrinsic lookup tables.
 */

#include "type_table.hpp"

namespace muda::compiler {

const std::array<TypeMappingEntry, 45> kTypeTable = {{
    {"bool", {ScalarType::kBool, VectorWidth::k1}},
    {"char", {ScalarType::kInt8, VectorWidth::k1}},
    {"signed char", {ScalarType::kInt8, VectorWidth::k1}},
    {"int8_t", {ScalarType::kInt8, VectorWidth::k1}},
    {"uchar", {ScalarType::kUInt8, VectorWidth::k1}},
    {"unsigned char", {ScalarType::kUInt8, VectorWidth::k1}},
    {"uint8_t", {ScalarType::kUInt8, VectorWidth::k1}},
    {"short", {ScalarType::kInt16, VectorWidth::k1}},
    {"int16_t", {ScalarType::kInt16, VectorWidth::k1}},
    {"ushort", {ScalarType::kUInt16, VectorWidth::k1}},
    {"unsigned short", {ScalarType::kUInt16, VectorWidth::k1}},
    {"uint16_t", {ScalarType::kUInt16, VectorWidth::k1}},
    {"int", {ScalarType::kInt32, VectorWidth::k1}},
    {"int32_t", {ScalarType::kInt32, VectorWidth::k1}},
    {"int2", {ScalarType::kInt32, VectorWidth::k2}},
    {"int3", {ScalarType::kInt32, VectorWidth::k3}},
    {"int4", {ScalarType::kInt32, VectorWidth::k4}},
    {"uint", {ScalarType::kUInt32, VectorWidth::k1}},
    {"unsigned", {ScalarType::kUInt32, VectorWidth::k1}},
    {"unsigned int", {ScalarType::kUInt32, VectorWidth::k1}},
    {"uint32_t", {ScalarType::kUInt32, VectorWidth::k1}},
    {"uint2", {ScalarType::kUInt32, VectorWidth::k2}},
    {"uint3", {ScalarType::kUInt32, VectorWidth::k3}},
    {"uint4", {ScalarType::kUInt32, VectorWidth::k4}},
    {"long", {ScalarType::kInt64, VectorWidth::k1}},
    {"long long", {ScalarType::kInt64, VectorWidth::k1}},
    {"int64_t", {ScalarType::kInt64, VectorWidth::k1}},
    {"ptrdiff_t", {ScalarType::kInt64, VectorWidth::k1}},
    {"long2", {ScalarType::kInt64, VectorWidth::k2}},
    {"long3", {ScalarType::kInt64, VectorWidth::k3}},
    {"long4", {ScalarType::kInt64, VectorWidth::k4}},
    {"ulong", {ScalarType::kUInt64, VectorWidth::k1}},
    {"unsigned long", {ScalarType::kUInt64, VectorWidth::k1}},
    {"unsigned long long", {ScalarType::kUInt64, VectorWidth::k1}},
    {"uint64_t", {ScalarType::kUInt64, VectorWidth::k1}},
    {"size_t", {ScalarType::kUInt64, VectorWidth::k1}},
    {"ulong2", {ScalarType::kUInt64, VectorWidth::k2}},
    {"ulong3", {ScalarType::kUInt64, VectorWidth::k3}},
    {"ulong4", {ScalarType::kUInt64, VectorWidth::k4}},
    {"half", {ScalarType::kFloat16, VectorWidth::k1}},
    {"__half", {ScalarType::kFloat16, VectorWidth::k1}},
    {"half2", {ScalarType::kFloat16, VectorWidth::k2}},
    {"__half2", {ScalarType::kFloat16, VectorWidth::k2}},
    {"half3", {ScalarType::kFloat16, VectorWidth::k3}},
    {"half4", {ScalarType::kFloat16, VectorWidth::k4}},
}};

const std::array<TypeMappingEntry, 4> kFloatTypeTable = {{
    {"float", {ScalarType::kFloat32, VectorWidth::k1}},
    {"float2", {ScalarType::kFloat32, VectorWidth::k2}},
    {"float3", {ScalarType::kFloat32, VectorWidth::k3}},
    {"float4", {ScalarType::kFloat32, VectorWidth::k4}},
}};

const std::array<BuiltinMappingEntry, 4> kBuiltinTable = {{
    {"blockIdx", "threadgroup_position_in_grid"},
    {"blockDim", "threads_per_threadgroup"},
    {"threadIdx", "thread_position_in_threadgroup"},
    {"gridDim", "threadgroups_per_grid"},
}};

const std::array<BuiltinMappingEntry, 18> kMathIntrinsicTable = {{
    {"fmaf", "fma"},
    {"__fmaf_rn", "fma"},
    {"sqrtf", "sqrt"},
    {"__fsqrt_rn", "sqrt"},
    {"rsqrtf", "rsqrt"},
    {"fabsf", "abs"},
    {"fminf", "min"},
    {"fmaxf", "max"},
    {"expf", "exp"},
    {"__expf", "fast::exp"},
    {"logf", "log"},
    {"__logf", "fast::log"},
    {"sinf", "sin"},
    {"__sinf", "fast::sin"},
    {"cosf", "cos"},
    {"__cosf", "fast::cos"},
    {"tanf", "tan"},
    {"powf", "pow"},
}};

const std::array<std::string_view, 18> kUnsupportedConstructs = {{
    "__constant__",
    "__managed__",
    "cooperative_groups",
    "atomicAdd",
    "atomicSub",
    "atomicCAS",
    "atomicExch",
    "atomicMin",
    "atomicMax",
    "atomicInc",
    "atomicDec",
    "atomicAnd",
    "atomicOr",
    "atomicXor",
    "__shfl_sync",
    "__shfl_up_sync",
    "__shfl_down_sync",
    "printf",
}};

std::optional<DataType> ResolveCudaType(std::string_view spelling) noexcept {
  for (const TypeMappingEntry& entry : kTypeTable) {
    if (entry.cuda_spelling == spelling) {
      return entry.type;
    }
  }
  for (const TypeMappingEntry& entry : kFloatTypeTable) {
    if (entry.cuda_spelling == spelling) {
      return entry.type;
    }
  }
  return std::nullopt;
}

std::string_view BuiltinReplacement(std::string_view word) noexcept {
  for (const BuiltinMappingEntry& entry : kBuiltinTable) {
    if (entry.cuda_spelling == word) {
      return entry.msl_replacement;
    }
  }
  return {};
}

std::string_view MathIntrinsicReplacement(std::string_view word) noexcept {
  for (const BuiltinMappingEntry& entry : kMathIntrinsicTable) {
    if (entry.cuda_spelling == word) {
      return entry.msl_replacement;
    }
  }
  return {};
}

}  // namespace muda::compiler


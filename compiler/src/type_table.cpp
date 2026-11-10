/**
 * @file type_table.cpp
 * @brief CUDA-to-Metal type and intrinsic lookup tables.
 */

#include "type_table.hpp"

namespace muda::compiler {

const std::array<TypeMappingEntry, 49> kTypeTable = {{
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
    {"double", {ScalarType::kFloat64, VectorWidth::k1}},
    {"double2", {ScalarType::kFloat64, VectorWidth::k2}},
    {"double3", {ScalarType::kFloat64, VectorWidth::k3}},
    {"double4", {ScalarType::kFloat64, VectorWidth::k4}},
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

const std::array<BuiltinMappingEntry, 46> kMathIntrinsicTable = {{
    {"fmaf", "fma"},
    {"__fmaf_rn", "fma"},
    {"fma", "muda_fp64_fma"},
    {"__fma_rn", "muda_fp64_fma"},
    {"sqrtf", "sqrt"},
    {"__fsqrt_rn", "sqrt"},
    {"sqrt", "muda_fp64_sqrt"},
    {"rsqrtf", "rsqrt"},
    {"rsqrt", "muda_fp64_rsqrt"},
    {"fabsf", "abs"},
    {"fabs", "muda_fp64_abs"},
    {"fminf", "min"},
    {"fmin", "muda_fp64_min"},
    {"fmaxf", "max"},
    {"fmax", "muda_fp64_max"},
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
    {"__popc", "popcount"},
    {"__popcll", "popcount"},
    {"__clz", "clz"},
    {"__clzll", "clz"},
    {"__brev", "reverse_bits"},
    {"__brevll", "reverse_bits"},
    {"make_float2", "float2"},
    {"make_float3", "float3"},
    {"make_float4", "float4"},
    {"make_double2", "muda_fp64_2"},
    {"make_double3", "muda_fp64_3"},
    {"make_double4", "muda_fp64_4"},
    {"make_int2", "int2"},
    {"make_int3", "int3"},
    {"make_int4", "int4"},
    {"make_uint2", "uint2"},
    {"make_uint3", "uint3"},
    {"make_uint4", "uint4"},
    {"make_half2", "half2"},
    {"__hfma", "fma"},
    {"__hfma2", "fma"},
}};

const std::array<std::string_view, 4> kUnsupportedConstructs = {{
    "__constant__",
    "__managed__",
    "cooperative_groups",
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

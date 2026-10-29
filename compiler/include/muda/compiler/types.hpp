/**
 * @file types.hpp
 * @brief CUDA-to-MSL data types, address spaces, and diagnostic codes.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace muda::compiler {

/**
 * @brief Scalar element types supported by the compiler.
 */
enum class ScalarType : std::uint8_t {
  kBool,
  kInt8,
  kUInt8,
  kInt16,
  kUInt16,
  kInt32,
  kUInt32,
  kInt64,
  kUInt64,
  kFloat16,
  kFloat32,
};

/**
 * @brief Vector lane counts supported by CUDA and MSL vector types.
 */
enum class VectorWidth : std::uint8_t {
  k1 = 1,
  k2 = 2,
  k3 = 3,
  k4 = 4,
};

/**
 * @brief Concrete scalar or vector data type.
 */
struct DataType {
  ScalarType scalar = ScalarType::kFloat32;
  VectorWidth width = VectorWidth::k1;

  constexpr bool operator==(const DataType&) const noexcept = default;
};

/**
 * @brief Memory address spaces in Metal Shading Language.
 */
enum class AddressSpace : std::uint8_t {
  kDevice,
  kConstant,
  kThreadgroup,
  kThread,
};

/**
 * @brief Memory access mutability qualifier.
 */
enum class AccessQualifier : std::uint8_t {
  kReadWrite,
  kReadOnly,
};

/**
 * @brief Kernel parameter binding category.
 */
enum class ParameterBindingKind : std::uint8_t {
  kBufferPointer,
  kConstantValue,
};

/**
 * @brief Intermediate file retention policy during compilation.
 */
enum class IntermediatePolicy : std::uint8_t {
  kRemove,
  kKeep,
};

/**
 * @brief Stable compiler and driver diagnostic codes (ALY-C* / ALY-M* / ALY-T*).
 */
enum class DiagnosticCode : std::uint16_t {
  kOk = 0,
  kMissingGlobalKernel,
  kMultipleKernelsUnsupported,
  kMalformedKernelDeclaration,
  kUnterminatedParameterList,
  kMissingKernelName,
  kNonVoidKernelReturn,
  kMissingKernelBody,
  kUnterminatedKernelBody,
  kUnsupportedConstruct,
  kEmptyKernelParameter,
  kMissingParameterName,
  kUnsupportedParameterType,
  kNoInputFile,
  kToolchainNotFound,
  kCannotOpenInput,
  kCannotCreateOutputDir,
  kCannotWriteMetalSource,
  kMetalCompilationFailed,
  kMetallibLinkFailed,
};

[[nodiscard]] constexpr std::size_t ScalarByteSize(ScalarType scalar) noexcept {
  switch (scalar) {
    case ScalarType::kBool:
    case ScalarType::kInt8:
    case ScalarType::kUInt8:
      return 1;
    case ScalarType::kInt16:
    case ScalarType::kUInt16:
    case ScalarType::kFloat16:
      return 2;
    case ScalarType::kInt32:
    case ScalarType::kUInt32:
    case ScalarType::kFloat32:
      return 4;
    case ScalarType::kInt64:
    case ScalarType::kUInt64:
      return 8;
  }
  return 0;
}

[[nodiscard]] constexpr std::size_t ByteSize(DataType type) noexcept {
  return ScalarByteSize(type.scalar) * static_cast<std::size_t>(type.width);
}

[[nodiscard]] constexpr std::string_view MslTypeName(DataType type) noexcept {
  switch (type.scalar) {
    case ScalarType::kBool:
      return type.width == VectorWidth::k1 ? "bool" : "";
    case ScalarType::kInt8:
      return type.width == VectorWidth::k1 ? "char" : "";
    case ScalarType::kUInt8:
      return type.width == VectorWidth::k1 ? "uchar" : "";
    case ScalarType::kInt16:
      return type.width == VectorWidth::k1 ? "short" : "";
    case ScalarType::kUInt16:
      return type.width == VectorWidth::k1 ? "ushort" : "";
    case ScalarType::kInt32:
      switch (type.width) {
        case VectorWidth::k1:
          return "int";
        case VectorWidth::k2:
          return "int2";
        case VectorWidth::k3:
          return "int3";
        case VectorWidth::k4:
          return "int4";
      }
      return "";
    case ScalarType::kUInt32:
      switch (type.width) {
        case VectorWidth::k1:
          return "uint";
        case VectorWidth::k2:
          return "uint2";
        case VectorWidth::k3:
          return "uint3";
        case VectorWidth::k4:
          return "uint4";
      }
      return "";
    case ScalarType::kInt64:
      switch (type.width) {
        case VectorWidth::k1:
          return "long";
        case VectorWidth::k2:
          return "long2";
        case VectorWidth::k3:
          return "long3";
        case VectorWidth::k4:
          return "long4";
      }
      return "";
    case ScalarType::kUInt64:
      switch (type.width) {
        case VectorWidth::k1:
          return "ulong";
        case VectorWidth::k2:
          return "ulong2";
        case VectorWidth::k3:
          return "ulong3";
        case VectorWidth::k4:
          return "ulong4";
      }
      return "";
    case ScalarType::kFloat16:
      switch (type.width) {
        case VectorWidth::k1:
          return "half";
        case VectorWidth::k2:
          return "half2";
        case VectorWidth::k3:
          return "half3";
        case VectorWidth::k4:
          return "half4";
      }
      return "";
    case ScalarType::kFloat32:
      switch (type.width) {
        case VectorWidth::k1:
          return "float";
        case VectorWidth::k2:
          return "float2";
        case VectorWidth::k3:
          return "float3";
        case VectorWidth::k4:
          return "float4";
      }
      return "";
  }
  return "";
}

[[nodiscard]] constexpr std::string_view AddressSpaceName(AddressSpace space) noexcept {
  switch (space) {
    case AddressSpace::kDevice:
      return "device";
    case AddressSpace::kConstant:
      return "constant";
    case AddressSpace::kThreadgroup:
      return "threadgroup";
    case AddressSpace::kThread:
      return "thread";
  }
  return "";
}

[[nodiscard]] constexpr std::string_view DiagnosticId(DiagnosticCode code) noexcept {
  switch (code) {
    case DiagnosticCode::kOk:
      return "ALY-0000";
    case DiagnosticCode::kMissingGlobalKernel:
      return "ALY-C001";
    case DiagnosticCode::kMultipleKernelsUnsupported:
      return "ALY-C002";
    case DiagnosticCode::kMalformedKernelDeclaration:
      return "ALY-C003";
    case DiagnosticCode::kUnterminatedParameterList:
      return "ALY-C004";
    case DiagnosticCode::kMissingKernelName:
      return "ALY-C005";
    case DiagnosticCode::kNonVoidKernelReturn:
      return "ALY-C006";
    case DiagnosticCode::kMissingKernelBody:
      return "ALY-C007";
    case DiagnosticCode::kUnterminatedKernelBody:
      return "ALY-C008";
    case DiagnosticCode::kUnsupportedConstruct:
      return "ALY-C009";
    case DiagnosticCode::kEmptyKernelParameter:
      return "ALY-C010";
    case DiagnosticCode::kMissingParameterName:
      return "ALY-C011";
    case DiagnosticCode::kUnsupportedParameterType:
      return "ALY-C012";
    case DiagnosticCode::kNoInputFile:
      return "ALY-T001";
    case DiagnosticCode::kToolchainNotFound:
      return "ALY-T002";
    case DiagnosticCode::kCannotOpenInput:
      return "ALY-T003";
    case DiagnosticCode::kCannotCreateOutputDir:
      return "ALY-T004";
    case DiagnosticCode::kCannotWriteMetalSource:
      return "ALY-T005";
    case DiagnosticCode::kMetalCompilationFailed:
      return "ALY-M001";
    case DiagnosticCode::kMetallibLinkFailed:
      return "ALY-M002";
  }
  return "ALY-X000";
}

[[nodiscard]] constexpr std::string_view DiagnosticMessage(DiagnosticCode code) noexcept {
  switch (code) {
    case DiagnosticCode::kOk:
      return "ok";
    case DiagnosticCode::kMissingGlobalKernel:
      return "no __global__ kernel found";
    case DiagnosticCode::kMultipleKernelsUnsupported:
      return "only one kernel is supported";
    case DiagnosticCode::kMalformedKernelDeclaration:
      return "malformed kernel declaration";
    case DiagnosticCode::kUnterminatedParameterList:
      return "unterminated kernel parameter list";
    case DiagnosticCode::kMissingKernelName:
      return "missing kernel name";
    case DiagnosticCode::kNonVoidKernelReturn:
      return "kernel return type must be void";
    case DiagnosticCode::kMissingKernelBody:
      return "missing kernel body";
    case DiagnosticCode::kUnterminatedKernelBody:
      return "unterminated kernel body";
    case DiagnosticCode::kUnsupportedConstruct:
      return "unsupported kernel construct";
    case DiagnosticCode::kEmptyKernelParameter:
      return "empty kernel parameter";
    case DiagnosticCode::kMissingParameterName:
      return "missing kernel parameter name";
    case DiagnosticCode::kUnsupportedParameterType:
      return "unsupported kernel parameter type";
    case DiagnosticCode::kNoInputFile:
      return "no input file";
    case DiagnosticCode::kToolchainNotFound:
      return "Metal toolchain not found; install Xcode and run "
             "xcodebuild -downloadComponent metalToolchain";
    case DiagnosticCode::kCannotOpenInput:
      return "cannot open input file";
    case DiagnosticCode::kCannotCreateOutputDir:
      return "cannot create output directory";
    case DiagnosticCode::kCannotWriteMetalSource:
      return "cannot write generated Metal source";
    case DiagnosticCode::kMetalCompilationFailed:
      return "Metal compilation failed";
    case DiagnosticCode::kMetallibLinkFailed:
      return "Metal library link failed";
  }
  return "unknown compiler error";
}

}  // namespace muda::compiler

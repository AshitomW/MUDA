/**
 * @file nvrtc.h
 * @brief Clean-room NVIDIA Runtime Compilation (NVRTC) C API subset for MUDA.
 */

#pragma once

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief NVRTC status codes.
 */
enum nvrtcResult : std::uint8_t {
  NVRTC_SUCCESS = 0,
  NVRTC_ERROR_OUT_OF_MEMORY = 1,
  NVRTC_ERROR_PROGRAM_CREATION_FAILURE = 2,
  NVRTC_ERROR_INVALID_INPUT = 3,
  NVRTC_ERROR_INVALID_PROGRAM = 4,
  NVRTC_ERROR_INVALID_OPTION = 5,
  NVRTC_ERROR_COMPILATION = 6,
  NVRTC_ERROR_BUILTIN_OPERATION_FAILURE = 7,
  NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION = 8,
  NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION = 9,
  NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID = 10,
  NVRTC_ERROR_INTERNAL_ERROR = 11
};

/**
 * @brief Opaque NVRTC program handle.
 */
using nvrtcProgram = struct _nvrtcProgram*;

/**
 * @brief Returns a human-readable description string for an nvrtcResult status code.
 */
const char* nvrtcGetErrorString(nvrtcResult result);

/**
 * @brief Queries the NVRTC major and minor version numbers.
 */
nvrtcResult nvrtcVersion(int* major, int* minor);

/**
 * @brief Creates an NVRTC compilation program from an in-memory CUDA C++ source string.
 */
nvrtcResult nvrtcCreateProgram(nvrtcProgram* prog, const char* src, const char* name,
                               int numHeaders, const char* const* headers,
                               const char* const* includeNames);

/**
 * @brief Destroys an NVRTC program handle and releases its resources.
 */
nvrtcResult nvrtcDestroyProgram(nvrtcProgram* prog);

/**
 * @brief Compiles the NVRTC program with the given compiler options.
 */
nvrtcResult nvrtcCompileProgram(nvrtcProgram prog, int numOptions, const char* const* options);

/**
 * @brief Retrieves the byte size (including trailing NUL) of the PTX emitted by compilation.
 */
nvrtcResult nvrtcGetPTXSize(nvrtcProgram prog, size_t* ptxSizeRet);

/**
 * @brief Copies the emitted PTX string (including trailing NUL) into the caller's buffer.
 */
nvrtcResult nvrtcGetPTX(nvrtcProgram prog, char* ptx);

/**
 * @brief Retrieves the byte size (including trailing NUL) of the compilation diagnostic log.
 */
nvrtcResult nvrtcGetProgramLogSize(nvrtcProgram prog, size_t* logSizeRet);

/**
 * @brief Copies the compilation diagnostic log into the caller's buffer.
 */
nvrtcResult nvrtcGetProgramLog(nvrtcProgram prog, char* log);

/**
 * @brief Registers a kernel name expression before compilation for lowered name lookup.
 */
nvrtcResult nvrtcAddNameExpression(nvrtcProgram prog, const char* name_expression);

/**
 * @brief Retrieves the lowered symbol name for a registered name expression after compilation.
 */
nvrtcResult nvrtcGetLoweredName(nvrtcProgram prog, const char* name_expression,
                                const char** lowered_name);

#ifdef __cplusplus
}  // extern "C"
#endif

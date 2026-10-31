/**
 * @file error_state.cpp
 * @brief Thread-local CUDA error state and error string lookup.
 */

#include "error_state.hpp"

namespace muda::runtime::cudart {
namespace {

thread_local cudaError_t g_last_error = cudaSuccess;

}  // namespace

cudaError_t RecordError(cudaError_t error) noexcept {
  if (error != cudaSuccess) {
    g_last_error = error;
  }
  return error;
}

cudaError_t ConsumeLastError() noexcept {
  const cudaError_t error = g_last_error;
  g_last_error = cudaSuccess;
  return error;
}

cudaError_t PeekLastError() noexcept { return g_last_error; }

const char* ErrorName(cudaError_t error) noexcept {
  switch (error) {
    case cudaSuccess:
      return "cudaSuccess";
    case cudaErrorInvalidValue:
      return "cudaErrorInvalidValue";
    case cudaErrorMemoryAllocation:
      return "cudaErrorMemoryAllocation";
    case cudaErrorInitializationError:
      return "cudaErrorInitializationError";
    case cudaErrorInvalidConfiguration:
      return "cudaErrorInvalidConfiguration";
    case cudaErrorInvalidDevice:
      return "cudaErrorInvalidDevice";
    case cudaErrorInvalidPitchValue:
      return "cudaErrorInvalidPitchValue";
    case cudaErrorSymbolNotFound:
      return "cudaErrorSymbolNotFound";
    case cudaErrorInvalidDevicePointer:
      return "cudaErrorInvalidDevicePointer";
    case cudaErrorInvalidMemcpyDirection:
      return "cudaErrorInvalidMemcpyDirection";
    case cudaErrorInvalidDeviceFunction:
      return "cudaErrorInvalidDeviceFunction";
    case cudaErrorNoDevice:
      return "cudaErrorNoDevice";
    case cudaErrorInvalidResourceHandle:
      return "cudaErrorInvalidResourceHandle";
    case cudaErrorNotReady:
      return "cudaErrorNotReady";
    case cudaErrorIllegalAddress:
      return "cudaErrorIllegalAddress";
    case cudaErrorLaunchOutOfResources:
      return "cudaErrorLaunchOutOfResources";
    case cudaErrorLaunchFailure:
      return "cudaErrorLaunchFailure";
    case cudaErrorNotSupported:
      return "cudaErrorNotSupported";
    case cudaErrorUnsupportedLimit:
      return "cudaErrorUnsupportedLimit";
    case cudaErrorInternalError:
      return "cudaErrorInternalError";
    case cudaErrorUnknown:
      return "cudaErrorUnknown";
  }
  return "cudaErrorUnknown";
}

const char* ErrorString(cudaError_t error) noexcept {
  switch (error) {
    case cudaSuccess:
      return "no error";
    case cudaErrorInvalidValue:
      return "invalid argument";
    case cudaErrorMemoryAllocation:
      return "out of memory";
    case cudaErrorInitializationError:
      return "initialization error";
    case cudaErrorInvalidConfiguration:
      return "invalid configuration argument";
    case cudaErrorInvalidDevice:
      return "invalid device ordinal";
    case cudaErrorInvalidPitchValue:
      return "invalid pitch argument";
    case cudaErrorSymbolNotFound:
      return "named symbol not found";
    case cudaErrorInvalidDevicePointer:
      return "invalid device pointer";
    case cudaErrorInvalidMemcpyDirection:
      return "invalid memcpy direction";
    case cudaErrorInvalidDeviceFunction:
      return "invalid device function";
    case cudaErrorNoDevice:
      return "no CUDA-capable Metal device is detected";
    case cudaErrorInvalidResourceHandle:
      return "invalid resource handle";
    case cudaErrorNotReady:
      return "device not ready";
    case cudaErrorIllegalAddress:
      return "an illegal memory access was encountered";
    case cudaErrorLaunchOutOfResources:
      return "too many resources requested for launch";
    case cudaErrorLaunchFailure:
      return "unspecified launch failure";
    case cudaErrorNotSupported:
      return "operation not supported";
    case cudaErrorUnsupportedLimit:
      return "unsupported limit";
    case cudaErrorInternalError:
      return "internal error";
    case cudaErrorUnknown:
      return "unknown error";
  }
  return "unknown error";
}

}  // namespace muda::runtime::cudart

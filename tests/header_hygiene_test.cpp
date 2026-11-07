/**
 * @file header_hygiene_test.cpp
 * @brief Standalone pure C++20 translation unit verifying all public C and C++ headers compile
 *        cleanly without Objective-C/Metal leakage or missing includes.
 */

#include "cublas.h"
#include "cublas_v2.h"
#include "cuda.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "cudnn.h"
#include "cufft.h"
#include "cusparse.h"
#include "device_launch_parameters.h"
#include "muda/compiler/cache.hpp"
#include "muda/compiler/driver.hpp"
#include "muda/compiler/host_rewriter.hpp"
#include "muda/compiler/mmir.hpp"
#include "muda/compiler/toolchain.hpp"
#include "muda/compiler/translator.hpp"
#include "muda/compiler/types.hpp"
#include "muda/runtime/buffer.hpp"
#include "muda/runtime/device.hpp"
#include "muda/runtime/kernel.hpp"
#include "muda/version.hpp"
#include "nvrtc.h"

int main() {
  int major = 0;
  int minor = 0;
  if (nvrtcVersion(&major, &minor) != NVRTC_SUCCESS) {
    return 1;
  }
  if (major != 12) {
    return 2;
  }
  if (muda::kVersion.empty()) {
    return 3;
  }
  return 0;
}

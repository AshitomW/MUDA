# MUDA

MUDA is a CUDA-to-Metal compatibility layer for Apple Silicon. Its goal is to
let CUDA programs run on Macs that do not have an NVIDIA GPU.

It is not a machine-learning framework. It is aimed at the gap between
CUDA-only software and Apple's Metal ecosystem.

## Why it exists

Apple GPUs do not execute CUDA. CUDA code is written for NVIDIA's toolchain,
compiled to PTX/SASS, and expects CUDA runtime and driver libraries. Metal has a
different language, memory model, and API, so CUDA software cannot simply be
recompiled for it.

## Examples

- A PyTorch extension built with `torch.utils.cpp_extension` invokes `nvcc` and
  includes CUDA headers. On macOS, `nvcc` and the CUDA toolkit are unavailable,
  so the extension cannot be built.
- A custom CUDA kernel is compiled to NVIDIA PTX/SASS and loaded through the
  CUDA driver. Metal cannot load or execute that binary.
- Code using cuBLAS, cuDNN, NCCL, or other CUDA-specific APIs depends on NVIDIA
  libraries and kernels. Those APIs have no direct Metal equivalent.
- CUDA features such as dynamic parallelism, device-side function pointers, and
  some warp-level primitives have no complete Metal equivalent.

MUDA intends to translate what can be translated, provide compatible runtime
pieces where practical, and report clear errors for what cannot run.

## Status

Early prototype. MUDA can translate and run a simple vector-add kernel on Apple
Silicon. It is not generally usable yet.

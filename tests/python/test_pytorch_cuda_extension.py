#!/usr/bin/env python3
"""
End-to-end Python + PyTorch integration test for MUDA on Apple Silicon.

Validates:
1. MUDA's `CUDA_HOME` layout (`bin/nvcc`, `bin/mudac`, `lib/libcudart.dylib`, `lib/libnvrtc.dylib`).
2. Compiling a custom multi-kernel CUDA C++ (`.cu`) extension (Fused Swish/SiLU + Residual Forward &
   Backward kernels) into a `.dylib` via MUDA's `nvcc` and wrapping it in a `torch.autograd.Function`.
3. Running a PyTorch `nn.Module` forward + backward pass (`loss.backward()`) on Apple Silicon GPU
   through the compiled CUDA kernels and verifying exact numerical parity against native PyTorch.
4. Executing matrix multiplication on `torch.Tensor` via MUDA's `libcublas.dylib` (`cublasSgemm_v2`).
"""

from __future__ import annotations

import ctypes
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import muda  # noqa: E402

CUDA_EXTENSION_SOURCE = r"""
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_silu_residual_fwd_kernel(const float* x,
                                               const float* residual,
                                               float* out,
                                               float beta,
                                               int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float val = x[i];
    float sig = 1.0f / (1.0f + expf(-beta * val));
    out[i] = val * sig + residual[i];
  }
}

__global__ void fused_silu_residual_bwd_kernel(const float* grad_out,
                                               const float* x,
                                               float* grad_x,
                                               float* grad_residual,
                                               float beta,
                                               int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float val = x[i];
    float g = grad_out[i];
    float sig = 1.0f / (1.0f + expf(-beta * val));
    float dsilu = sig * (1.0f + beta * val * (1.0f - sig));
    grad_x[i] = g * dsilu;
    grad_residual[i] = g;
  }
}

extern "C" int launch_fused_silu_residual_fwd(const float* h_x,
                                              const float* h_res,
                                              float* h_out,
                                              float beta,
                                              int n) {
  float* d_x = nullptr;
  float* d_res = nullptr;
  float* d_out = nullptr;
  size_t bytes = static_cast<size_t>(n) * sizeof(float);

  if (cudaMalloc(reinterpret_cast<void**>(&d_x), bytes) != cudaSuccess) return 1;
  if (cudaMalloc(reinterpret_cast<void**>(&d_res), bytes) != cudaSuccess) return 2;
  if (cudaMalloc(reinterpret_cast<void**>(&d_out), bytes) != cudaSuccess) return 3;

  cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_res, h_res, bytes, cudaMemcpyHostToDevice);

  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  fused_silu_residual_fwd_kernel<<<blocks, threads>>>(d_x, d_res, d_out, beta, n);
  cudaDeviceSynchronize();

  cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

  cudaFree(d_out);
  cudaFree(d_res);
  cudaFree(d_x);
  return 0;
}

extern "C" int launch_fused_silu_residual_bwd(const float* h_grad_out,
                                              const float* h_x,
                                              float* h_grad_x,
                                              float* h_grad_res,
                                              float beta,
                                              int n) {
  float* d_grad_out = nullptr;
  float* d_x = nullptr;
  float* d_grad_x = nullptr;
  float* d_grad_res = nullptr;
  size_t bytes = static_cast<size_t>(n) * sizeof(float);

  if (cudaMalloc(reinterpret_cast<void**>(&d_grad_out), bytes) != cudaSuccess) return 1;
  if (cudaMalloc(reinterpret_cast<void**>(&d_x), bytes) != cudaSuccess) return 2;
  if (cudaMalloc(reinterpret_cast<void**>(&d_grad_x), bytes) != cudaSuccess) return 3;
  if (cudaMalloc(reinterpret_cast<void**>(&d_grad_res), bytes) != cudaSuccess) return 4;

  cudaMemcpy(d_grad_out, h_grad_out, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);

  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  fused_silu_residual_bwd_kernel<<<blocks, threads>>>(
      d_grad_out, d_x, d_grad_x, d_grad_res, beta, n);
  cudaDeviceSynchronize();

  cudaMemcpy(h_grad_x, d_grad_x, bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_grad_res, d_grad_res, bytes, cudaMemcpyDeviceToHost);

  cudaFree(d_grad_res);
  cudaFree(d_grad_x);
  cudaFree(d_x);
  cudaFree(d_grad_out);
  return 0;
}
"""


def test_cuda_home_and_nvcc_toolkit() -> None:
    env = muda.configure_environment()
    home = Path(env["CUDA_HOME"])
    assert (home / "bin" / "nvcc").exists(), "Missing nvcc in CUDA_HOME/bin"
    assert (home / "bin" / "mudac").exists(), "Missing mudac in CUDA_HOME/bin"
    assert (home / "lib" / "libcuda.dylib").exists(), "Missing libcuda.dylib"
    assert (home / "lib" / "libcudart.dylib").exists(), "Missing libcudart.dylib"
    assert (home / "lib" / "libnvrtc.dylib").exists(), "Missing libnvrtc.dylib"
    assert (home / "include" / "cuda_runtime.h").exists(), "Missing cuda_runtime.h"

    ver = subprocess.run(
        [str(home / "bin" / "nvcc"), "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "mudac" in ver.stdout.lower()


def test_pytorch_custom_cuda_autograd_extension() -> None:
    ext = muda.compile_cuda_extension("fused_silu_residual", CUDA_EXTENSION_SOURCE)

    ext.launch_fused_silu_residual_fwd.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_float,
        ctypes.c_int,
    ]
    ext.launch_fused_silu_residual_fwd.restype = ctypes.c_int

    ext.launch_fused_silu_residual_bwd.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_float,
        ctypes.c_int,
    ]
    ext.launch_fused_silu_residual_bwd.restype = ctypes.c_int

    class MudaFusedSiLUResidualFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x: torch.Tensor, residual: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
            x_c = x.detach().to(dtype=torch.float32, device="cpu").contiguous()
            r_c = residual.detach().to(dtype=torch.float32, device="cpu").contiguous()
            out = torch.empty_like(x_c)
            n = x_c.numel()

            rc = ext.launch_fused_silu_residual_fwd(
                ctypes.c_void_p(x_c.data_ptr()),
                ctypes.c_void_p(r_c.data_ptr()),
                ctypes.c_void_p(out.data_ptr()),
                ctypes.c_float(beta),
                ctypes.c_int(n),
            )
            if rc != 0:
                raise RuntimeError(f"launch_fused_silu_residual_fwd failed with code {rc}")

            ctx.save_for_backward(x_c)
            ctx.beta = beta
            ctx.orig_device = x.device
            return out.to(device=x.device)

        @staticmethod
        def backward(ctx, grad_output: torch.Tensor):
            (x_c,) = ctx.saved_tensors
            g_c = grad_output.detach().to(dtype=torch.float32, device="cpu").contiguous()
            grad_x = torch.empty_like(x_c)
            grad_res = torch.empty_like(x_c)
            n = x_c.numel()

            rc = ext.launch_fused_silu_residual_bwd(
                ctypes.c_void_p(g_c.data_ptr()),
                ctypes.c_void_p(x_c.data_ptr()),
                ctypes.c_void_p(grad_x.data_ptr()),
                ctypes.c_void_p(grad_res.data_ptr()),
                ctypes.c_float(ctx.beta),
                ctypes.c_int(n),
            )
            if rc != 0:
                raise RuntimeError(f"launch_fused_silu_residual_bwd failed with code {rc}")

            return (
                grad_x.to(device=ctx.orig_device),
                grad_res.to(device=ctx.orig_device),
                None,
            )

    torch.manual_seed(42)
    shape = (16, 64)
    beta = 1.25

    x_muda = torch.randn(shape, dtype=torch.float32, requires_grad=True)
    res_muda = torch.randn(shape, dtype=torch.float32, requires_grad=True)
    target = torch.randn(shape, dtype=torch.float32)

    x_ref = x_muda.detach().clone().requires_grad_(True)
    res_ref = res_muda.detach().clone().requires_grad_(True)

    # Forward pass through MUDA-compiled CUDA kernel vs PyTorch reference
    out_muda = MudaFusedSiLUResidualFunction.apply(x_muda, res_muda, beta)
    out_ref = x_ref * torch.sigmoid(beta * x_ref) + res_ref

    assert torch.allclose(out_muda, out_ref, atol=1e-5, rtol=1e-5), (
        f"Forward max diff: {(out_muda - out_ref).abs().max().item()}"
    )

    # Backward pass through MUDA-compiled CUDA kernel vs PyTorch autograd
    loss_muda = F.mse_loss(out_muda, target)
    loss_ref = F.mse_loss(out_ref, target)
    loss_muda.backward()
    loss_ref.backward()

    assert x_muda.grad is not None and x_ref.grad is not None
    assert res_muda.grad is not None and res_ref.grad is not None
    assert torch.allclose(x_muda.grad, x_ref.grad, atol=1e-5, rtol=1e-5), (
        f"Backward grad_x max diff: {(x_muda.grad - x_ref.grad).abs().max().item()}"
    )
    assert torch.allclose(res_muda.grad, res_ref.grad, atol=1e-5, rtol=1e-5), (
        f"Backward grad_res max diff: {(res_muda.grad - res_ref.grad).abs().max().item()}"
    )


def test_pytorch_muda_cublas_matmul() -> None:
    torch.manual_seed(123)
    a = torch.randn((32, 48), dtype=torch.float32)
    b = torch.randn((48, 24), dtype=torch.float32)

    c_muda = muda.muda_cublas_matmul(a, b)
    c_ref = a @ b

    assert torch.allclose(c_muda, c_ref, atol=1e-4, rtol=1e-4), (
        f"cuBLAS matmul max diff: {(c_muda - c_ref).abs().max().item()}"
    )


def main() -> int:
    test_cuda_home_and_nvcc_toolkit()
    test_pytorch_custom_cuda_autograd_extension()
    test_pytorch_muda_cublas_matmul()
    print("MUDA + PyTorch custom CUDA extension & cuBLAS tests passed!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


"""PyTorch & ctypes bridge for compiling and executing CUDA C++ (.cu) extensions on Apple Silicon via MUDA."""

from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union


def cuda_home() -> Path:
    """Resolve the active MUDA CUDA_HOME directory."""
    env_home = os.environ.get("MUDA_CUDA_HOME") or os.environ.get("CUDA_HOME")
    if env_home and Path(env_home).is_dir():
        return Path(env_home).resolve()

    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        repo_root / "build" / "release" / "cuda-home",
        repo_root / "build" / "debug" / "cuda-home",
        repo_root / "build" / "asan" / "cuda-home",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not locate MUDA cuda-home directory. Set MUDA_CUDA_HOME or build the muda_cuda_home target."
    )


def configure_environment() -> Dict[str, str]:
    """Configure process environment variables (CUDA_HOME, PATH, DYLD_LIBRARY_PATH) for MUDA."""
    home = cuda_home()
    bin_dir = str(home / "bin")
    lib_dir = str(home / "lib")

    os.environ["CUDA_HOME"] = str(home)
    os.environ["CUDA_PATH"] = str(home)
    os.environ["PATH"] = f"{bin_dir}:{os.environ.get('PATH', '')}"
    os.environ["DYLD_LIBRARY_PATH"] = f"{lib_dir}:{os.environ.get('DYLD_LIBRARY_PATH', '')}"

    return {
        "CUDA_HOME": str(home),
        "CUDA_PATH": str(home),
        "BIN_DIR": bin_dir,
        "LIB_DIR": lib_dir,
        "INCLUDE_DIR": str(home / "include"),
    }


class MudaRuntimeBindings:
    """Thin ctypes wrapper around MUDA's libcudart.dylib and libcublas.dylib."""

    cudaMemcpyHostToDevice = 1
    cudaMemcpyDeviceToHost = 2
    CUBLAS_OP_N = 0
    CUBLAS_OP_T = 1

    def __init__(self, home: Optional[Path] = None) -> None:
        self.home = home or cuda_home()
        lib_dir = self.home / "lib"

        # Load libcuda.dylib first with RTLD_GLOBAL so RuntimeContext singleton is shared.
        self.libcuda = ctypes.CDLL(str(lib_dir / "libcuda.dylib"), mode=ctypes.RTLD_GLOBAL)
        self.libcudart = ctypes.CDLL(str(lib_dir / "libcudart.dylib"), mode=ctypes.RTLD_GLOBAL)
        self.libcublas = ctypes.CDLL(str(lib_dir / "libcublas.dylib"), mode=ctypes.RTLD_GLOBAL)

        self.libcudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.libcudart.cudaMalloc.restype = ctypes.c_int

        self.libcudart.cudaFree.argtypes = [ctypes.c_void_p]
        self.libcudart.cudaFree.restype = ctypes.c_int

        self.libcudart.cudaMemcpy.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        self.libcudart.cudaMemcpy.restype = ctypes.c_int

        self.libcudart.cudaDeviceSynchronize.argtypes = []
        self.libcudart.cudaDeviceSynchronize.restype = ctypes.c_int

        self.libcublas.cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.libcublas.cublasCreate_v2.restype = ctypes.c_int

        self.libcublas.cublasDestroy_v2.argtypes = [ctypes.c_void_p]
        self.libcublas.cublasDestroy_v2.restype = ctypes.c_int

        self.libcublas.cublasSgemm_v2.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        self.libcublas.cublasSgemm_v2.restype = ctypes.c_int

    def malloc(self, nbytes: int) -> ctypes.c_void_p:
        ptr = ctypes.c_void_p()
        err = self.libcudart.cudaMalloc(ctypes.byref(ptr), nbytes)
        if err != 0:
            raise RuntimeError(f"cudaMalloc({nbytes}) failed with error {err}")
        return ptr

    def free(self, ptr: ctypes.c_void_p) -> None:
        if ptr and ptr.value:
            err = self.libcudart.cudaFree(ptr)
            if err != 0:
                raise RuntimeError(f"cudaFree failed with error {err}")

    def copy_to_device(self, dst_dev: ctypes.c_void_p, src_host_ptr: int, nbytes: int) -> None:
        err = self.libcudart.cudaMemcpy(
            dst_dev,
            ctypes.c_void_p(src_host_ptr),
            nbytes,
            self.cudaMemcpyHostToDevice,
        )
        if err != 0:
            raise RuntimeError(f"cudaMemcpy(H2D) failed with error {err}")

    def copy_to_host(self, dst_host_ptr: int, src_dev: ctypes.c_void_p, nbytes: int) -> None:
        err = self.libcudart.cudaMemcpy(
            ctypes.c_void_p(dst_host_ptr),
            src_dev,
            nbytes,
            self.cudaMemcpyDeviceToHost,
        )
        if err != 0:
            raise RuntimeError(f"cudaMemcpy(D2H) failed with error {err}")

    def synchronize(self) -> None:
        err = self.libcudart.cudaDeviceSynchronize()
        if err != 0:
            raise RuntimeError(f"cudaDeviceSynchronize failed with error {err}")


class MudaCudaExtension:
    """Represents a compiled `.cu` shared library extension loaded into Python/PyTorch."""

    def __init__(self, name: str, dylib_path: Path, runtime: MudaRuntimeBindings) -> None:
        self.name = name
        self.dylib_path = dylib_path
        self.runtime = runtime
        self.lib = ctypes.CDLL(str(dylib_path), mode=ctypes.RTLD_GLOBAL)

    def __getattr__(self, item: str):
        return getattr(self.lib, item)


def compile_cuda_extension(
    name: str,
    cu_source: Union[str, Path],
    extra_cflags: Optional[Sequence[str]] = None,
    build_dir: Optional[Union[str, Path]] = None,
) -> MudaCudaExtension:
    """
    Compile a CUDA `.cu` source file or inline CUDA C++ source string into a macOS `.dylib`
    using MUDA's `nvcc`/`mudac` compiler driver and load it for use with PyTorch tensors.
    """
    configure_environment()
    home = cuda_home()
    runtime = MudaRuntimeBindings(home)
    nvcc_bin = home / "bin" / "nvcc"
    if not nvcc_bin.exists():
        nvcc_bin = home / "bin" / "mudac"

    if build_dir is None:
        digest = hashlib.sha256(str(cu_source).encode("utf-8")).hexdigest()[:12]
        out_dir = Path(tempfile.gettempdir()) / f"muda_torch_ext_{name}_{digest}"
    else:
        out_dir = Path(build_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(cu_source, Path) or (
        isinstance(cu_source, str) and "\n" not in cu_source and Path(cu_source).exists()
    ):
        src_path = Path(cu_source).resolve()
    else:
        src_path = out_dir / f"{name}.cu"
        src_path.write_text(str(cu_source), encoding="utf-8")

    dylib_path = out_dir / f"lib{name}.dylib"
    cmd: List[str] = [
        str(nvcc_bin),
        "-shared",
        "-fPIC",
        "-O2",
        f"-I{home / 'include'}",
        str(src_path),
        "-o",
        str(dylib_path),
    ]
    if extra_cflags:
        cmd.extend(extra_cflags)

    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"MUDA nvcc failed to compile CUDA extension '{name}' (exit {proc.returncode}):\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )

    return MudaCudaExtension(name=name, dylib_path=dylib_path, runtime=runtime)


def muda_cublas_matmul(a_tensor, b_tensor):
    """
    Execute matrix multiplication C = A @ B on 2D float32 PyTorch tensors using MUDA's
    Apple Silicon Metal-backed cuBLAS (`cublasSgemm_v2`).
    """
    import torch

    if a_tensor.ndim != 2 or b_tensor.ndim != 2:
        raise ValueError("muda_cublas_matmul expects 2D tensors")
    if a_tensor.shape[1] != b_tensor.shape[0]:
        raise ValueError(f"Incompatible shapes: {a_tensor.shape} and {b_tensor.shape}")

    a_contig = a_tensor.detach().to(dtype=torch.float32, device="cpu").contiguous()
    b_contig = b_tensor.detach().to(dtype=torch.float32, device="cpu").contiguous()
    m, k = a_contig.shape
    _, n = b_contig.shape
    c_contig = torch.empty((m, n), dtype=torch.float32, device="cpu")

    rt = MudaRuntimeBindings()
    bytes_a = a_contig.numel() * 4
    bytes_b = b_contig.numel() * 4
    bytes_c = c_contig.numel() * 4

    d_a = rt.malloc(bytes_a)
    d_b = rt.malloc(bytes_b)
    d_c = rt.malloc(bytes_c)
    handle = ctypes.c_void_p()
    try:
        rt.copy_to_device(d_a, a_contig.data_ptr(), bytes_a)
        rt.copy_to_device(d_b, b_contig.data_ptr(), bytes_b)

        status = rt.libcublas.cublasCreate_v2(ctypes.byref(handle))
        if status != 0:
            raise RuntimeError(f"cublasCreate_v2 failed with status {status}")

        alpha = ctypes.c_float(1.0)
        beta = ctypes.c_float(0.0)
        # Row-major C(m, n) = A(m, k) @ B(k, n) <=> Column-major C^T(n, m) = B^T(n, k) @ A^T(k, m)
        status = rt.libcublas.cublasSgemm_v2(
            handle,
            rt.CUBLAS_OP_N,
            rt.CUBLAS_OP_N,
            n,
            m,
            k,
            ctypes.byref(alpha),
            d_b,
            n,
            d_a,
            k,
            ctypes.byref(beta),
            d_c,
            n,
        )
        if status != 0:
            raise RuntimeError(f"cublasSgemm_v2 failed with status {status}")

        rt.synchronize()
        rt.copy_to_host(c_contig.data_ptr(), d_c, bytes_c)
    finally:
        if handle and handle.value:
            rt.libcublas.cublasDestroy_v2(handle)
        rt.free(d_c)
        rt.free(d_b)
        rt.free(d_a)

    return c_contig.to(device=a_tensor.device)


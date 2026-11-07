"""MUDA: CUDA-to-Metal compatibility layer & PyTorch bridge for Apple Silicon."""

from .pytorch_bridge import (
    MudaCudaExtension,
    compile_cuda_extension,
    configure_environment,
    cuda_home,
    muda_cublas_matmul,
)

__version__ = "0.1.0"

__all__ = [
    "MudaCudaExtension",
    "compile_cuda_extension",
    "configure_environment",
    "cuda_home",
    "muda_cublas_matmul",
]


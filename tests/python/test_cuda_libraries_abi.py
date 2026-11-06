#!/usr/bin/env python3
"""
test_cuda_libraries_abi.py
End-to-end verification of the MUDA CUDA Acceleration Libraries C-ABI
(libcublas.dylib, libcufft.dylib, libcudnn.dylib, libcusparse.dylib) via Python ctypes.
"""

import ctypes
from ctypes import byref, c_double, c_float, c_int, c_int64, c_size_t, c_ulonglong, c_void_p, POINTER, Structure
import os
import sys
import numpy as np

CWD = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CWD, "..", ".."))

LIBCUDA_PATH = os.environ.get("CUDA_DRIVER_LIBRARY", os.path.join(REPO_ROOT, "build", "runtime", "libcuda.dylib"))
LIBCUBLAS_PATH = os.environ.get("CUBLAS_LIBRARY", os.path.join(REPO_ROOT, "build", "libraries", "libcublas.dylib"))
LIBCUFFT_PATH = os.environ.get("CUFFT_LIBRARY", os.path.join(REPO_ROOT, "build", "libraries", "libcufft.dylib"))
LIBCUDNN_PATH = os.environ.get("CUDNN_LIBRARY", os.path.join(REPO_ROOT, "build", "libraries", "libcudnn.dylib"))
LIBCUSPARSE_PATH = os.environ.get("CUSPARSE_LIBRARY", os.path.join(REPO_ROOT, "build", "libraries", "libcusparse.dylib"))

for p in (LIBCUDA_PATH, LIBCUBLAS_PATH, LIBCUFFT_PATH, LIBCUDNN_PATH, LIBCUSPARSE_PATH):
    if not os.path.exists(p):
        print(f"Error: Library not found at {p}")
        sys.exit(1)

cuda = ctypes.CDLL(LIBCUDA_PATH)
cublas = ctypes.CDLL(LIBCUBLAS_PATH)
cufft = ctypes.CDLL(LIBCUFFT_PATH)
cudnn = ctypes.CDLL(LIBCUDNN_PATH)
cusparse = ctypes.CDLL(LIBCUSPARSE_PATH)

CUdeviceptr = c_ulonglong

cuda.cuInit.argtypes = [ctypes.c_uint]
cuda.cuInit.restype = c_int
cuda.cuMemAlloc.argtypes = [POINTER(CUdeviceptr), c_size_t]
cuda.cuMemAlloc.restype = c_int
cuda.cuMemFree.argtypes = [CUdeviceptr]
cuda.cuMemFree.restype = c_int
cuda.cuMemcpyHtoD.argtypes = [CUdeviceptr, c_void_p, c_size_t]
cuda.cuMemcpyHtoD.restype = c_int
cuda.cuMemcpyDtoH.argtypes = [c_void_p, CUdeviceptr, c_size_t]
cuda.cuMemcpyDtoH.restype = c_int


def check(status, msg):
    if status != 0:
        raise RuntimeError(f"ABI error ({status}) at: {msg}")


def test_cublas():
    print("[TEST] Testing libcublas.dylib via ctypes...")
    cublas.cublasCreate_v2.argtypes = [POINTER(c_void_p)]
    cublas.cublasCreate_v2.restype = c_int
    cublas.cublasDestroy_v2.argtypes = [c_void_p]
    cublas.cublasDestroy_v2.restype = c_int
    cublas.cublasSgemm_v2.argtypes = [
        c_void_p, c_int, c_int, c_int, c_int, c_int,
        POINTER(c_float), c_void_p, c_int, c_void_p, c_int,
        POINTER(c_float), c_void_p, c_int
    ]
    cublas.cublasSgemm_v2.restype = c_int

    handle = c_void_p()
    check(cublas.cublasCreate_v2(byref(handle)), "cublasCreate_v2")

    n = 64
    h_a = np.ones((n, n), dtype=np.float32, order="F") * 0.5
    h_b = np.ones((n, n), dtype=np.float32, order="F") * 2.0
    h_c = np.zeros((n, n), dtype=np.float32, order="F")

    d_a, d_b, d_c = CUdeviceptr(), CUdeviceptr(), CUdeviceptr()
    check(cuda.cuMemAlloc(byref(d_a), h_a.nbytes), "cuMemAlloc a")
    check(cuda.cuMemAlloc(byref(d_b), h_b.nbytes), "cuMemAlloc b")
    check(cuda.cuMemAlloc(byref(d_c), h_c.nbytes), "cuMemAlloc c")
    check(cuda.cuMemcpyHtoD(d_a, h_a.ctypes.data, h_a.nbytes), "HtoD a")
    check(cuda.cuMemcpyHtoD(d_b, h_b.ctypes.data, h_b.nbytes), "HtoD b")

    alpha = c_float(1.0)
    beta = c_float(0.0)
    check(
        cublas.cublasSgemm_v2(
            handle, 0, 0, n, n, n,
            byref(alpha), c_void_p(d_a.value), n,
            c_void_p(d_b.value), n,
            byref(beta), c_void_p(d_c.value), n
        ),
        "cublasSgemm_v2"
    )
    check(cuda.cuMemcpyDtoH(h_c.ctypes.data, d_c, h_c.nbytes), "DtoH c")
    assert np.allclose(h_c, n * 1.0, atol=1e-4)

    cuda.cuMemFree(d_a)
    cuda.cuMemFree(d_b)
    cuda.cuMemFree(d_c)
    check(cublas.cublasDestroy_v2(handle), "cublasDestroy_v2")
    print("  [PASS] libcublas.dylib verified.")


def test_cufft():
    print("[TEST] Testing libcufft.dylib via ctypes...")
    cufft.cufftPlan1d.argtypes = [POINTER(c_int), c_int, c_int, c_int]
    cufft.cufftPlan1d.restype = c_int
    cufft.cufftExecC2C.argtypes = [c_int, c_void_p, c_void_p, c_int]
    cufft.cufftExecC2C.restype = c_int
    cufft.cufftDestroy.argtypes = [c_int]
    cufft.cufftDestroy.restype = c_int

    n = 128
    plan = c_int(0)
    check(cufft.cufftPlan1d(byref(plan), n, 0x29, 1), "cufftPlan1d")

    h_sig = np.exp(2j * np.pi * 3 * np.arange(n) / n).astype(np.complex64)
    h_out = np.zeros_like(h_sig)
    d_sig = CUdeviceptr()
    check(cuda.cuMemAlloc(byref(d_sig), h_sig.nbytes), "cuMemAlloc fft")
    check(cuda.cuMemcpyHtoD(d_sig, h_sig.ctypes.data, h_sig.nbytes), "HtoD fft")

    check(cufft.cufftExecC2C(plan, c_void_p(d_sig.value), c_void_p(d_sig.value), -1), "cufftExecC2C fwd")
    check(cufft.cufftExecC2C(plan, c_void_p(d_sig.value), c_void_p(d_sig.value), 1), "cufftExecC2C inv")
    check(cuda.cuMemcpyDtoH(h_out.ctypes.data, d_sig, h_out.nbytes), "DtoH fft")
    assert np.allclose(h_out / n, h_sig, atol=1e-3)

    cuda.cuMemFree(d_sig)
    check(cufft.cufftDestroy(plan), "cufftDestroy")
    print("  [PASS] libcufft.dylib verified.")


def test_cudnn():
    print("[TEST] Testing libcudnn.dylib via ctypes...")
    cudnn.cudnnCreate.argtypes = [POINTER(c_void_p)]
    cudnn.cudnnCreate.restype = c_int
    cudnn.cudnnDestroy.argtypes = [c_void_p]
    cudnn.cudnnDestroy.restype = c_int
    cudnn.cudnnCreateTensorDescriptor.argtypes = [POINTER(c_void_p)]
    cudnn.cudnnCreateTensorDescriptor.restype = c_int
    cudnn.cudnnSetTensor4dDescriptor.argtypes = [c_void_p, c_int, c_int, c_int, c_int, c_int, c_int]
    cudnn.cudnnSetTensor4dDescriptor.restype = c_int
    cudnn.cudnnDestroyTensorDescriptor.argtypes = [c_void_p]
    cudnn.cudnnDestroyTensorDescriptor.restype = c_int
    cudnn.cudnnCreateActivationDescriptor.argtypes = [POINTER(c_void_p)]
    cudnn.cudnnCreateActivationDescriptor.restype = c_int
    cudnn.cudnnSetActivationDescriptor.argtypes = [c_void_p, c_int, c_int, c_double]
    cudnn.cudnnSetActivationDescriptor.restype = c_int
    cudnn.cudnnDestroyActivationDescriptor.argtypes = [c_void_p]
    cudnn.cudnnDestroyActivationDescriptor.restype = c_int
    cudnn.cudnnActivationForward.argtypes = [
        c_void_p, c_void_p, POINTER(c_float), c_void_p, c_void_p,
        POINTER(c_float), c_void_p, c_void_p
    ]
    cudnn.cudnnActivationForward.restype = c_int

    handle = c_void_p()
    t_desc = c_void_p()
    act_desc = c_void_p()
    check(cudnn.cudnnCreate(byref(handle)), "cudnnCreate")
    check(cudnn.cudnnCreateTensorDescriptor(byref(t_desc)), "cudnnCreateTensorDescriptor")
    check(cudnn.cudnnSetTensor4dDescriptor(t_desc, 0, 0, 1, 2, 2, 2), "cudnnSetTensor4dDescriptor")
    check(cudnn.cudnnCreateActivationDescriptor(byref(act_desc)), "cudnnCreateActivationDescriptor")
    check(cudnn.cudnnSetActivationDescriptor(act_desc, 1, 0, 0.0), "cudnnSetActivationDescriptor")

    h_x = np.array([-3.0, -1.0, 0.0, 1.0, 2.0, -0.5, 4.0, -2.0], dtype=np.float32)
    h_y = np.zeros_like(h_x)
    d_x, d_y = CUdeviceptr(), CUdeviceptr()
    check(cuda.cuMemAlloc(byref(d_x), h_x.nbytes), "cuMemAlloc x")
    check(cuda.cuMemAlloc(byref(d_y), h_y.nbytes), "cuMemAlloc y")
    check(cuda.cuMemcpyHtoD(d_x, h_x.ctypes.data, h_x.nbytes), "HtoD x")

    alpha = c_float(1.0)
    beta = c_float(0.0)
    check(
        cudnn.cudnnActivationForward(
            handle, act_desc, byref(alpha), t_desc, c_void_p(d_x.value),
            byref(beta), t_desc, c_void_p(d_y.value)
        ),
        "cudnnActivationForward"
    )
    check(cuda.cuMemcpyDtoH(h_y.ctypes.data, d_y, h_y.nbytes), "DtoH y")
    assert np.allclose(h_y, np.maximum(h_x, 0.0), atol=1e-5)

    cuda.cuMemFree(d_x)
    cuda.cuMemFree(d_y)
    cudnn.cudnnDestroyActivationDescriptor(act_desc)
    cudnn.cudnnDestroyTensorDescriptor(t_desc)
    cudnn.cudnnDestroy(handle)
    print("  [PASS] libcudnn.dylib verified.")


def test_cusparse():
    print("[TEST] Testing libcusparse.dylib via ctypes...")
    cusparse.cusparseCreate.argtypes = [POINTER(c_void_p)]
    cusparse.cusparseCreate.restype = c_int
    cusparse.cusparseDestroy.argtypes = [c_void_p]
    cusparse.cusparseDestroy.restype = c_int
    cusparse.cusparseCreateCsr.argtypes = [
        POINTER(c_void_p), c_int64, c_int64, c_int64,
        c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int
    ]
    cusparse.cusparseCreateCsr.restype = c_int
    cusparse.cusparseCreateDnVec.argtypes = [POINTER(c_void_p), c_int64, c_void_p, c_int]
    cusparse.cusparseCreateDnVec.restype = c_int
    cusparse.cusparseSpMV.argtypes = [
        c_void_p, c_int, POINTER(c_float), c_void_p, c_void_p,
        POINTER(c_float), c_void_p, c_int, c_int, c_void_p
    ]
    cusparse.cusparseSpMV.restype = c_int
    cusparse.cusparseDestroySpMat.argtypes = [c_void_p]
    cusparse.cusparseDestroySpMat.restype = c_int
    cusparse.cusparseDestroyDnVec.argtypes = [c_void_p]
    cusparse.cusparseDestroyDnVec.restype = c_int

    handle = c_void_p()
    check(cusparse.cusparseCreate(byref(handle)), "cusparseCreate")

    h_offsets = np.array([0, 1, 2], dtype=np.int32)
    h_cols = np.array([0, 1], dtype=np.int32)
    h_vals = np.array([3.0, 5.0], dtype=np.float32)
    h_x = np.array([2.0, 4.0], dtype=np.float32)
    h_y = np.zeros(2, dtype=np.float32)

    d_off, d_col, d_val, d_x, d_y = CUdeviceptr(), CUdeviceptr(), CUdeviceptr(), CUdeviceptr(), CUdeviceptr()
    cuda.cuMemAlloc(byref(d_off), h_offsets.nbytes)
    cuda.cuMemAlloc(byref(d_col), h_cols.nbytes)
    cuda.cuMemAlloc(byref(d_val), h_vals.nbytes)
    cuda.cuMemAlloc(byref(d_x), h_x.nbytes)
    cuda.cuMemAlloc(byref(d_y), h_y.nbytes)
    cuda.cuMemcpyHtoD(d_off, h_offsets.ctypes.data, h_offsets.nbytes)
    cuda.cuMemcpyHtoD(d_col, h_cols.ctypes.data, h_cols.nbytes)
    cuda.cuMemcpyHtoD(d_val, h_vals.ctypes.data, h_vals.nbytes)
    cuda.cuMemcpyHtoD(d_x, h_x.ctypes.data, h_x.nbytes)
    cuda.cuMemcpyHtoD(d_y, h_y.ctypes.data, h_y.nbytes)

    mat_a, vec_x, vec_y = c_void_p(), c_void_p(), c_void_p()
    check(
        cusparse.cusparseCreateCsr(
            byref(mat_a), 2, 2, 2,
            c_void_p(d_off.value), c_void_p(d_col.value), c_void_p(d_val.value),
            2, 2, 0, 0
        ),
        "cusparseCreateCsr"
    )
    check(cusparse.cusparseCreateDnVec(byref(vec_x), 2, c_void_p(d_x.value), 0), "cusparseCreateDnVec x")
    check(cusparse.cusparseCreateDnVec(byref(vec_y), 2, c_void_p(d_y.value), 0), "cusparseCreateDnVec y")

    alpha = c_float(1.0)
    beta = c_float(0.0)
    check(
        cusparse.cusparseSpMV(
            handle, 0, byref(alpha), mat_a, vec_x,
            byref(beta), vec_y, 0, 0, None
        ),
        "cusparseSpMV"
    )
    cuda.cuMemcpyDtoH(h_y.ctypes.data, d_y, h_y.nbytes)
    assert np.allclose(h_y, np.array([6.0, 20.0], dtype=np.float32))

    cusparse.cusparseDestroyDnVec(vec_x)
    cusparse.cusparseDestroyDnVec(vec_y)
    cusparse.cusparseDestroySpMat(mat_a)
    cuda.cuMemFree(d_off)
    cuda.cuMemFree(d_col)
    cuda.cuMemFree(d_val)
    cuda.cuMemFree(d_x)
    cuda.cuMemFree(d_y)
    cusparse.cusparseDestroy(handle)
    print("  [PASS] libcusparse.dylib verified.")


def main():
    check(cuda.cuInit(0), "cuInit")
    test_cublas()
    test_cufft()
    test_cudnn()
    test_cusparse()
    print("\nALL CUDA ACCELERATION LIBRARIES C-ABI TESTS PASSED!\n")


if __name__ == "__main__":
    main()

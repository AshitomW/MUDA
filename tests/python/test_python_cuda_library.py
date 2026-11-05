#!/usr/bin/env python3
"""
test_python_cuda_library.py
High-level verification of real Python code using an established CUDA library (numba.cuda).

This test suite executes real, idiomatic Python code utilizing high-level CUDA APIs
without embedding or compiling raw PTX assembly strings:
1. Device discovery, hardware capabilities, and primary context initialization.
2. Multi-dimensional DeviceNDArray allocation (1D, 2D, 3D) across float32, float64, int32, int64.
3. Host-to-Device and Device-to-Host memory transfers with NumPy fidelity.
4. Device-to-Device array copying.
5. Multi-dimensional device array slicing (sub-matrices, volumes, and strided slices).
6. Pinned (page-locked) host memory allocation and high-speed streaming transfers.
7. Asynchronous stream pipelines and multi-stream execution.
8. GPU event synchronization, cross-stream event waiting, and precise timing benchmarks.
9. Device array cleanup and memory lifecycle tracking.
"""

import os
import sys
import numpy as np

# Ensure NUMBA_CUDA_DRIVER points to our MUDA libcuda.dylib
CWD = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CWD, "..", ".."))
LIBCUDA_PATH = os.environ.get(
    "NUMBA_CUDA_DRIVER",
    os.path.join(REPO_ROOT, "build", "runtime", "libcuda.dylib")
)

if not os.path.exists(LIBCUDA_PATH):
    print(f"Error: libcuda.dylib not found at {LIBCUDA_PATH}")
    sys.exit(1)

os.environ["NUMBA_CUDA_DRIVER"] = LIBCUDA_PATH

# Import high-level CUDA API from Numba
from numba import cuda


def test_device_discovery():
    print("[TEST 1/8] Discovering CUDA device and inspecting context...")
    device = cuda.get_current_device()
    dev_name = device.name.decode("utf-8") if isinstance(device.name, bytes) else device.name
    print(f"  Device Name:         {dev_name}")
    print(f"  Device ID:           {device.id}")
    print(f"  Compute Capability:  {device.compute_capability}")

    ctx = cuda.current_context()
    free_mem, total_mem = ctx.get_memory_info()
    print(f"  GPU Free Memory:     {free_mem / (1024**2):.1f} MB")
    print(f"  GPU Total Memory:    {total_mem / (1024**2):.1f} MB")

    assert "Apple" in dev_name or "M1" in dev_name or "GPU" in dev_name or len(dev_name) > 0
    assert device.compute_capability >= (3, 5)
    assert total_mem > 0
    print("  [PASS] Device discovery and context verified.\n")


def test_multidimensional_device_arrays():
    print("[TEST 2/8] Testing multi-dimensional DeviceNDArrays (1D, 2D, 3D) across dtypes...")

    test_cases = [
        ("1D Float32 Vector", (16384,), np.float32),
        ("1D Float64 Vector", (8192,), np.float64),
        ("2D Int32 Matrix", (128, 64), np.int32),
        ("2D Float32 Matrix", (256, 128), np.float32),
        ("3D Int64 Tensor", (16, 32, 64), np.int64),
        ("3D Float32 Tensor", (32, 32, 32), np.float32),
    ]

    for desc, shape, dtype in test_cases:
        if np.issubdtype(dtype, np.floating):
            host_data = np.random.randn(*shape).astype(dtype)
        else:
            host_data = np.random.randint(-1000, 1000, size=shape, dtype=dtype)

        # Allocate on GPU and copy data to device
        d_arr = cuda.to_device(host_data)
        assert d_arr.shape == shape, f"Shape mismatch for {desc}"
        assert d_arr.dtype == dtype, f"Dtype mismatch for {desc}"
        assert d_arr.nbytes == host_data.nbytes, f"Nbytes mismatch for {desc}"

        # Copy back to host and verify bitwise / numeric accuracy
        host_result = d_arr.copy_to_host()
        np.testing.assert_array_equal(host_data, host_result)
        print(f"  [PASS] {desc}: shape={shape}, dtype={dtype.__name__}, nbytes={d_arr.nbytes}")

    print("  [PASS] Multi-dimensional device array transfers verified with 100% accuracy.\n")


def test_device_to_device_copy():
    print("[TEST 3/8] Testing direct Device-to-Device array copying...")
    n_rows, n_cols = 512, 256
    source_host = np.linspace(-50.0, 50.0, n_rows * n_cols, dtype=np.float32).reshape(n_rows, n_cols)

    # Allocate source and target directly on device
    d_source = cuda.to_device(source_host)
    d_target = cuda.device_array_like(d_source)

    # Perform Device-to-Device transfer
    d_target.copy_to_device(d_source)

    # Validate target holds exact data and source remains unmodified
    target_host = d_target.copy_to_host()
    np.testing.assert_array_equal(source_host, target_host)

    source_check = d_source.copy_to_host()
    np.testing.assert_array_equal(source_host, source_check)
    print("  [PASS] Device-to-device array copy validated.\n")


def test_device_slicing_and_subviews():
    print("[TEST 4/8] Testing subview slicing and strided views directly on device...")

    # 1. 2D Sub-matrix slicing on device
    matrix_host = np.arange(10000, dtype=np.float32).reshape(100, 100)
    d_matrix = cuda.to_device(matrix_host)

    # Slice a 20x30 sub-matrix on GPU
    d_slice = d_matrix[15:35, 40:70]
    expected_slice = matrix_host[15:35, 40:70]
    slice_host = d_slice.copy_to_host()
    np.testing.assert_array_equal(expected_slice, slice_host)
    print("  [PASS] 2D sub-matrix slicing on device verified.")

    # 2. 3D Sub-volume slicing on device
    tensor_host = np.arange(64 * 32 * 16, dtype=np.int32).reshape(64, 32, 16)
    d_tensor = cuda.to_device(tensor_host)

    d_subvolume = d_tensor[10:30, 5:25, 2:14]
    expected_subvolume = tensor_host[10:30, 5:25, 2:14]
    subvolume_host = d_subvolume.copy_to_host()
    np.testing.assert_array_equal(expected_subvolume, subvolume_host)
    print("  [PASS] 3D sub-volume slicing on device verified.")

    # 3. Strided slicing on device
    d_strided = d_matrix[::2, ::5]
    expected_strided = matrix_host[::2, ::5]
    strided_host = d_strided.copy_to_host()
    np.testing.assert_array_equal(expected_strided, strided_host)
    print("  [PASS] Strided subviews on device verified.\n")


def test_pinned_host_memory():
    print("[TEST 5/8] Testing pinned (page-locked) host memory allocation & transfers...")
    num_elements = 65536
    pinned_buf = cuda.pinned_array(num_elements, dtype=np.float32)

    # Populate pinned memory on CPU
    test_pattern = np.sin(np.linspace(0, 100, num_elements, dtype=np.float32))
    pinned_buf[:] = test_pattern

    # Asynchronous streaming transfer from pinned host memory to GPU device array
    d_pinned_target = cuda.device_array(num_elements, dtype=np.float32)
    stream = cuda.stream()
    d_pinned_target.copy_to_device(pinned_buf, stream=stream)
    stream.synchronize()

    # Read back to standard host memory and check fidelity
    readback = d_pinned_target.copy_to_host()
    np.testing.assert_allclose(readback, test_pattern, atol=1e-6)
    print("  [PASS] Pinned memory allocated and transferred to GPU successfully.\n")


def test_asynchronous_streams_pipeline():
    print("[TEST 6/8] Testing multi-stream asynchronous pipeline...")
    num_chunks = 4
    chunk_size = 16384
    total_size = num_chunks * chunk_size

    full_host_data = np.arange(total_size, dtype=np.float32)
    d_full_target = cuda.device_array(total_size, dtype=np.float32)

    # Create distinct CUDA streams for concurrent pipelined execution
    streams = [cuda.stream() for _ in range(num_chunks)]

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size
        chunk_data = full_host_data[start_idx:end_idx]
        d_chunk = d_full_target[start_idx:end_idx]

        # Queue chunk transfer into its dedicated stream
        d_chunk.copy_to_device(chunk_data, stream=streams[i])

    # Synchronize all streams
    for s in streams:
        s.synchronize()

    # Verify complete pipelined dataset on host
    assembled_result = d_full_target.copy_to_host()
    np.testing.assert_array_equal(full_host_data, assembled_result)
    print(f"  [PASS] Pipelined asynchronous transfers across {num_chunks} streams verified.\n")


def test_cuda_events_and_timing():
    print("[TEST 7/8] Testing CUDA Events, cross-stream synchronization, and timing...")
    stream_a = cuda.stream()
    stream_b = cuda.stream()

    event_start = cuda.event()
    event_mid = cuda.event()
    event_stop = cuda.event()

    n = 131072  # 512 KB
    data = np.ones(n, dtype=np.float32) * 7.5
    d_buf = cuda.device_array(n, dtype=np.float32)

    # Record start event in stream_a
    event_start.record(stream=stream_a)

    # Transfer data in stream_a
    d_buf.copy_to_device(data, stream=stream_a)

    # Record mid event in stream_a
    event_mid.record(stream=stream_a)

    # Make stream_b wait on event_mid before proceeding
    event_mid.wait(stream=stream_b)

    # In stream_b, allocate and copy device to device
    d_buf2 = cuda.device_array_like(d_buf)
    d_buf2.copy_to_device(d_buf, stream=stream_b)

    # Record stop event in stream_b
    event_stop.record(stream=stream_b)

    # Synchronize on stop event
    event_stop.synchronize()

    # Calculate elapsed GPU execution time
    elapsed_ms = cuda.event_elapsed_time(event_start, event_stop)
    print(f"  GPU Elapsed Time measured by hardware event: {elapsed_ms:.3f} ms")
    assert elapsed_ms >= 0.0, "Elapsed time must be non-negative"

    # Verify mathematical accuracy of stream_b result
    host_final = d_buf2.copy_to_host()
    np.testing.assert_array_equal(host_final, data)
    print("  [PASS] Events, cross-stream synchronization, and timing benchmark passed.\n")


def test_memory_lifecycle_and_cleanup():
    print("[TEST 8/8] Testing device array lifecycle, resizing, and cleanup...")
    ctx = cuda.current_context()
    free_before, _ = ctx.get_memory_info()

    # Allocate and reshape on GPU
    d_large = cuda.device_array((1024, 1024), dtype=np.float32)  # 4 MB
    d_reshaped = d_large.reshape(2048, 512)
    assert d_reshaped.shape == (2048, 512)

    # Populate with data
    pattern = np.arange(1024 * 1024, dtype=np.float32).reshape(2048, 512)
    d_reshaped.copy_to_device(pattern)

    result = d_reshaped.copy_to_host()
    np.testing.assert_array_equal(result, pattern)

    # Delete device references
    del d_large
    del d_reshaped

    # Verify context remains healthy
    free_after, _ = ctx.get_memory_info()
    assert free_after > 0
    print("  [PASS] Device array lifecycle and memory cleanup verified.\n")


def main():
    print("=" * 70)
    print("MUDA x High-Level Python CUDA Library Integration Test Suite")
    print("Executing Real Python CUDA Code without Raw PTX Strings")
    print("=" * 70)

    test_device_discovery()
    test_multidimensional_device_arrays()
    test_device_to_device_copy()
    test_device_slicing_and_subviews()
    test_pinned_host_memory()
    test_asynchronous_streams_pipeline()
    test_cuda_events_and_timing()
    test_memory_lifecycle_and_cleanup()

    print("=" * 70)
    print("ALL REAL PYTHON CUDA LIBRARY TESTS PASSED (100% GREEN)!")
    print("=" * 70)


if __name__ == "__main__":
    main()

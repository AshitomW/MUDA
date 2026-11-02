#include <cuda_runtime.h>

__device__ float affine_clamp(float x, float scale, float bias, float lo, float hi) {
  return fminf(fmaxf(fmaf(x, scale, bias), lo), hi);
}

__device__ float accumulate_pair(float a, float b) {
  return a + b;
}

// 1. vector_add
__global__ void k1_vector_add(const float* a, const float* b, float* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = a[i] + b[i];
  }
}

// 2. saxpy
__global__ void k2_saxpy(float alpha, const float* x, const float* y, float* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = fmaf(alpha, x[i], y[i]);
  }
}

// 3. scale_clamp
__global__ void k3_scale_clamp(const float* x, float* out, float scale, float bias, float lo,
                               float hi, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = affine_clamp(x[i], scale, bias, lo, hi);
  }
}

// 4. dot_reduce
__global__ void k4_dot_reduce(const float* a, const float* b, float* block_sums, int n) {
  __shared__ float sdata[64];
  const int tid = threadIdx.x;
  const int idx = blockIdx.x * blockDim.x + tid;
  sdata[tid] = (idx < n) ? (a[idx] * b[idx]) : 0.0f;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sdata[tid] = accumulate_pair(sdata[tid], sdata[tid + stride]);
    }
    __syncthreads();
  }

  if (tid == 0) {
    block_sums[blockIdx.x] = sdata[0];
  }
}

// 5. transpose_naive
__global__ void k5_transpose_naive(const float* in, float* out, int width, int height) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (col < width && row < height) {
    out[col * height + row] = in[row * width + col];
  }
}

// 6. transpose_tiled
__global__ void k6_transpose_tiled(const float* in, float* out, int width, int height) {
  __shared__ float tile[64];
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int col = blockIdx.x * 8 + tx;
  const int row = blockIdx.y * 8 + ty;
  if (col < width && row < height) {
    tile[ty * 8 + tx] = in[row * width + col];
  }
  __syncthreads();

  const int out_col = blockIdx.y * 8 + tx;
  const int out_row = blockIdx.x * 8 + ty;
  if (out_col < height && out_row < width) {
    out[out_row * height + out_col] = tile[tx * 8 + ty];
  }
}

// 7. gemm_naive
__global__ void k7_gemm_naive(const float* a, const float* b, float* c, int m, int n, int k) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < m && col < n) {
    float acc = 0.0f;
    for (int p = 0; p < k; ++p) {
      acc = fmaf(a[row * k + p], b[p * n + col], acc);
    }
    c[row * n + col] = acc;
  }
}

// 8. gemm_tiled
__global__ void k8_gemm_tiled(const float* a, const float* b, float* c, int n) {
  __shared__ float tile_a[64];
  __shared__ float tile_b[64];
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int row = blockIdx.y * 8 + ty;
  const int col = blockIdx.x * 8 + tx;

  float acc = 0.0f;
  const int num_tiles = (n + 7) / 8;
  for (int t = 0; t < num_tiles; ++t) {
    const int a_col = t * 8 + tx;
    const int b_row = t * 8 + ty;
    tile_a[ty * 8 + tx] = (row < n && a_col < n) ? a[row * n + a_col] : 0.0f;
    tile_b[ty * 8 + tx] = (b_row < n && col < n) ? b[b_row * n + col] : 0.0f;
    __syncthreads();

    for (int p = 0; p < 8; ++p) {
      acc = fmaf(tile_a[ty * 8 + p], tile_b[p * 8 + tx], acc);
    }
    __syncthreads();
  }

  if (row < n && col < n) {
    c[row * n + col] = acc;
  }
}

// 9. stencil_1d
__global__ void k9_stencil_1d(const float* in, float* out, int n) {
  extern __shared__ float halo[];
  const int tid = threadIdx.x;
  const int gid = blockIdx.x * blockDim.x + tid;
  const int bdim = blockDim.x;

  if (gid < n) {
    halo[tid + 1] = in[gid];
    if (tid == 0) {
      halo[0] = (gid > 0) ? in[gid - 1] : in[0];
    }
    if (tid == bdim - 1 || gid == n - 1) {
      const int right_idx = (gid + 1 < n) ? (gid + 1) : (n - 1);
      halo[tid + 2] = in[right_idx];
    }
  }
  __syncthreads();

  if (gid < n) {
    out[gid] = 0.25f * halo[tid] + 0.5f * halo[tid + 1] + 0.25f * halo[tid + 2];
  }
}

// 10. row_sum
__global__ void k10_row_sum(const float* mat, float* row_sums, int rows, int cols) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r < rows) {
    float sum = 0.0f;
    for (int c = 0; c < cols; ++c) {
      sum = accumulate_pair(sum, mat[r * cols + c]);
    }
    row_sums[r] = sum;
  }
}

// 11. warp_shuffle_reduce
__global__ void k11_warp_shuffle_reduce(const float* in, float* out, int n) {
  const int tid = threadIdx.x;
  const int gid = blockIdx.x * 32 + tid;
  float val = (gid < n) ? in[gid] : 0.0f;

  for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }

  if (tid == 0) {
    out[blockIdx.x] = val;
  }
}

// 12. warp_prefix_scan
__global__ void k12_warp_prefix_scan(const float* in, float* out, int n) {
  const int tid = threadIdx.x;
  const int gid = blockIdx.x * 32 + tid;
  float val = (gid < n) ? in[gid] : 0.0f;

  for (int offset = 1; offset < 32; offset *= 2) {
    const float nval = __shfl_up_sync(0xffffffff, val, offset);
    if (tid >= offset) {
      val += nval;
    }
  }

  if (gid < n) {
    out[gid] = val;
  }
}

// 13. warp_broadcast
__global__ void k13_warp_broadcast(const float* in, float* out, int n) {
  const int tid = threadIdx.x;
  const int gid = blockIdx.x * 32 + tid;
  float val = (gid < n) ? in[gid] : 0.0f;
  const float bcast = __shfl_sync(0xffffffff, val, 0);
  if (gid < n) {
    out[gid] = bcast;
  }
}

// 14. warp_vote_all
__global__ void k14_warp_vote_all(const float* in, int* out_all, int* out_any, int n) {
  const int tid = threadIdx.x;
  const int gid = blockIdx.x * 32 + tid;
  const int pred = (gid < n && in[gid] > 0.0f) ? 1 : 0;
  const int all_val = __all_sync(0xffffffff, pred);
  const int any_val = __any_sync(0xffffffff, pred);
  if (tid == 0) {
    out_all[blockIdx.x] = all_val;
    out_any[blockIdx.x] = any_val;
  }
}

// 15. warp_activemask_ballot
__global__ void k15_warp_activemask_ballot(const int* in, int* out_popc, int n) {
  const int tid = threadIdx.x;
  const int gid = blockIdx.x * 32 + tid;
  const int pred = (gid < n && in[gid] != 0) ? 1 : 0;
  const unsigned int ballot = __ballot_sync(0xffffffff, pred);
  if (tid == 0) {
    out_popc[blockIdx.x] = __popc(ballot);
  }
}

// 16. bit_reverse_brev
__global__ void k16_bit_reverse_brev(const unsigned int* in, unsigned int* out, int n) {
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < n) {
    out[gid] = __brev(in[gid]);
  }
}

// 17. leading_zeros_clz
__global__ void k17_leading_zeros_clz(const unsigned int* in, int* out, int n) {
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < n) {
    out[gid] = __clz(in[gid]);
  }
}

// 18. atomic_histogram
__global__ void k18_atomic_histogram(const int* data, int* hist, int n) {
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < n) {
    const int bin = data[gid];
    if (bin >= 0 && bin < 16) {
      atomicAdd(&hist[bin], 1);
    }
  }
}

// 19. atomic_min_max
__global__ void k19_atomic_min_max(const int* data, int* out_min, int* out_max, int n) {
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < n) {
    atomicMin(out_min, data[gid]);
    atomicMax(out_max, data[gid]);
  }
}

// 20. atomic_float_add
__global__ void k20_atomic_float_add(const float* in, float* total, int n) {
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < n) {
    atomicAdd(total, in[gid]);
  }
}

// 21. atomic_cas_counter
__global__ void k21_atomic_cas_counter(int* queue_head, int* assigned_slots, int n) {
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < n) {
    int old_val = *queue_head;
    while (true) {
      int prev = atomicCAS(queue_head, old_val, old_val + 1);
      if (prev == old_val) {
        assigned_slots[gid] = old_val;
        break;
      }
      old_val = prev;
    }
  }
}

// 22. half_vector_add
__global__ void k22_half_vector_add(const float* a, const float* b, float* out, int n) {
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < n) {
    __half ha = __float2half(a[gid]);
    __half hb = __float2half(b[gid]);
    __half hc = __hadd(ha, hb);
    out[gid] = __half2float(hc);
  }
}

// 23. half2_simd_saxpy
__global__ void k23_half2_simd_saxpy(const float* a, const float* b, float* out, float alpha, int n) {
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid * 2 + 1 < n) {
    half2 ha = __floats2half2_rn(a[gid * 2], a[gid * 2 + 1]);
    half2 hb = __floats2half2_rn(b[gid * 2], b[gid * 2 + 1]);
    half2 halpha = __float2half2_rn(alpha);
    half2 hres = __hfma2(halpha, ha, hb);
    out[gid * 2] = __half2float(hres.x);
    out[gid * 2 + 1] = __half2float(hres.y);
  }
}

// 24. vector4_unpack_math
__global__ void k24_vector4_unpack_math(const float* in, float* out, int n) {
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid * 4 + 3 < n) {
    float4 v = make_float4(in[gid * 4], in[gid * 4 + 1], in[gid * 4 + 2], in[gid * 4 + 3]);
    float4 res = make_float4(v.x * 2.0f, v.y + 1.0f, v.z - 0.5f, v.w * 0.25f);
    out[gid * 4] = res.x;
    out[gid * 4 + 1] = res.y;
    out[gid * 4 + 2] = res.z;
    out[gid * 4 + 3] = res.w;
  }
}

// 25. int4_vector_transform
__global__ void k25_int4_vector_transform(const int* in, int* out, int n) {
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid * 4 + 3 < n) {
    int4 iv = make_int4(in[gid * 4], in[gid * 4 + 1], in[gid * 4 + 2], in[gid * 4 + 3]);
    int4 ires = make_int4(iv.x + 10, iv.y * 2, iv.z ^ 0xff, iv.w - 5);
    out[gid * 4] = ires.x;
    out[gid * 4 + 1] = ires.y;
    out[gid * 4 + 2] = ires.z;
    out[gid * 4 + 3] = ires.w;
  }
}

// 26. relu_activation
__global__ void k26_relu_activation(const float* in, float* out, int n) {
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < n) {
    out[gid] = (in[gid] > 0.0f) ? in[gid] : 0.0f;
  }
}

// 27. leaky_relu
__global__ void k27_leaky_relu(const float* in, float* out, float alpha, int n) {
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < n) {
    out[gid] = (in[gid] > 0.0f) ? in[gid] : (in[gid] * alpha);
  }
}

// 28. softmax_warp
__global__ void k28_softmax_warp(const float* in, float* out, int n) {
  const int tid = threadIdx.x;
  const int row = blockIdx.x;
  const int idx = row * 32 + tid;
  float val = (idx < n) ? in[idx] : -1e20f;

  float max_val = val;
  for (int offset = 16; offset > 0; offset /= 2) {
    const float other = __shfl_down_sync(0xffffffff, max_val, offset);
    if (other > max_val) {
      max_val = other;
    }
  }
  max_val = __shfl_sync(0xffffffff, max_val, 0);

  float exp_val = (idx < n) ? expf(val - max_val) : 0.0f;
  float sum_exp = exp_val;
  for (int offset = 16; offset > 0; offset /= 2) {
    sum_exp += __shfl_down_sync(0xffffffff, sum_exp, offset);
  }
  sum_exp = __shfl_sync(0xffffffff, sum_exp, 0);

  if (idx < n) {
    out[idx] = (sum_exp > 0.0f) ? (exp_val / sum_exp) : 0.0f;
  }
}

// 29. layer_norm
__global__ void k29_layer_norm(const float* in, float* out, float gamma, float beta, float eps, int n) {
  const int tid = threadIdx.x;
  const int row = blockIdx.x;
  const int idx = row * 32 + tid;
  const float val = (idx < n) ? in[idx] : 0.0f;

  float sum = val;
  for (int offset = 16; offset > 0; offset /= 2) {
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  }
  const float mean = __shfl_sync(0xffffffff, sum, 0) / 32.0f;

  const float diff = (idx < n) ? (val - mean) : 0.0f;
  float sq_sum = diff * diff;
  for (int offset = 16; offset > 0; offset /= 2) {
    sq_sum += __shfl_down_sync(0xffffffff, sq_sum, offset);
  }
  const float variance = __shfl_sync(0xffffffff, sq_sum, 0) / 32.0f;
  const float inv_std = rsqrtf(variance + eps);

  if (idx < n) {
    out[idx] = (val - mean) * inv_std * gamma + beta;
  }
}

// 30. spmv_csr
__global__ void k30_spmv_csr(const int* row_offsets, const int* col_indices, const float* values,
                             const float* x, float* y, int num_rows) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < num_rows) {
    const int start = row_offsets[row];
    const int end = row_offsets[row + 1];
    float sum = 0.0f;
    for (int j = start; j < end; ++j) {
      sum += values[j] * x[col_indices[j]];
    }
    y[row] = sum;
  }
}

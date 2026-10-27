// strata_round_elementwise.cuh - Strata's decode round's copies through mapped memory, its doorbell, its embedding
// gather and stream broadcast, and its experts' q8_1 quantizer, ported literally (strata-port/decode).
//
// Ported from Strata (https://github.com/Niko1221/Strata, commit 296b59d2), MIT License, Copyright (c) 2026 Niko1221
// and the Strata contributors (licenses/Strata-MIT.txt), from src/kernels/cuda/elementwise.cu
// (copy_from_mapped_kernel, copy_rows_from_mapped_kernel, copy_i32_from_mapped_kernel, doorbell_publish_res_kernel),
// verify_kernels.cu (broadcast_streams_kernel, wait_flag_ge_kernel, fetch_blobs_kernel, rebase_ptrs_kernel),
// s2_expert_grouped.cu (add_hits_kernel), ple.cu (history_advance_kernel), shared_expert.cu (sigmoid_scale_rows_kernel and its float4 form), iq_kernels.cu (embed_rows_kernel over dq_q8_0, q8_1_store and
// quantize_q8_1_kernel, which Strata adapted from llama.cpp's convert.cu and quantize.cu, MIT, Copyright (c) 2023-2026
// The ggml authors) and include/strata/kernels/dp4a.hpp (strata_spin_pause).
//
// Strata compiles these files WITHOUT fast math, as this module is compiled. Only the Q8_0 form of embed_rows'
// dequantizer is here: Flash-Next's token embedding is Q8_0. The host sides are translated in
// internal/cuda/executor/launch_strata_round_elementwise.go.

namespace strata_round {

__device__ __forceinline__ void strata_spin_pause() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700
    // sm_6x: the loop spins.  Every call site is a single-thread doorbell wait, so nothing else is delayed.
#else
    __nanosleep(100);
#endif
}

// q8_1_finite.hpp - #606: the q8_1 activation block's scale and sum, kept finite.
constexpr float kHalfMax = 65504.0f;
__device__ __forceinline__ float q8_1_finite(const float v) {
    return fabsf(v) > kHalfMax ? copysignf(kHalfMax, v) : v;
}
__device__ __forceinline__ int8_t q8_1_quant(const float xi, const float d, const float amax) {
    if (amax == 0.0f) return 0;
    const float q = roundf(xi / d);
    return (int8_t) (q > 127.0f ? 127.0f : q < -127.0f ? -127.0f : q);
}
__device__ __forceinline__ half2 q8_1_ds(const float d, const float sum) {
    return make_half2(__float2half(q8_1_finite(d)), __float2half(q8_1_finite(sum)));
}

struct block_q8_0 {
    half d;
    int8_t qs[32];
};
struct block_q8_1 {
    half2 ds;
    int8_t qs[32];
};
constexpr int QK_K = 256;
constexpr int QK8_0 = 32;

__device__ __forceinline__ void q8_1_store(const float xi, block_q8_1* __restrict__ y, const long long i) {
    float amax = fabsf(xi), sum = xi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        sum += __shfl_xor_sync(0xffffffffu, sum, o);
    }
    const float d = q8_1_finite(amax / 127.0f);   // #606: q8_1_finite.hpp - the same bits for every finite block
    const int8_t q = q8_1_quant(xi, d, amax);
    const long long ib = i / 32, iqs = i % 32;
    y[ib].qs[iqs] = q;
    if (iqs == 0) y[ib].ds = q8_1_ds(d, sum);
}

__device__ void dq_q8_0(const void* vx, int64_t ibs, float* yy, int tid) {
    const block_q8_0* x = (const block_q8_0*) vx + ibs * (QK_K / QK8_0);
    const int ib = tid % 8, il = tid / 8;
    const float d = __half2float(x[ib].d);
    float* y = yy + 32 * ib + 8 * il;
    for (int j = 0; j < 8; ++j) y[j] = (float) x[ib].qs[8 * il + j] * d;
}

}  // namespace strata_round

extern "C" __global__ void strata_copy_from_mapped(float4* __restrict__ dst, const volatile float4* src, long long n4) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += (int64_t) gridDim.x * blockDim.x) {
        const float4 v = const_cast<const float4*>(src)[i];
        dst[i] = v;
    }
}

// the CPU rows of a verify window, skipping the rows the GPU plan computes itself (the
// pool writes +0.0 into those, so this writes +0.0 too): block = row, the plan's hit rows `dst[0, *count)`.
extern "C" __global__ void strata_copy_rows_from_mapped(float4* __restrict__ dst, const volatile float4* src,
                                                        long long row4, const int32_t* __restrict__ hit_rows,
                                                        const int32_t* __restrict__ count,
                                                        const volatile uint32_t* wait_flag, uint32_t wait_value) {
    const int row = blockIdx.x;
    __shared__ int hit;
    if (threadIdx.x == 0) {
        if (wait_flag != nullptr) {   // 143-amdk: the wait_flag_ge launch folded in
            while (*wait_flag < wait_value) strata_round::strata_spin_pause();
            __threadfence_system();
        }
        int h = 0;
        const int c = *count;
        for (int i = 0; i < c; ++i) h |= hit_rows[i] == row;
        hit = h;
    }
    __syncthreads();
    float4* d = dst + (int64_t) row * row4;
    if (hit) {
        for (int64_t i = threadIdx.x; i < row4; i += blockDim.x) d[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    } else {
        const volatile float4* sr = src + (int64_t) row * row4;
        for (int64_t i = threadIdx.x; i < row4; i += blockDim.x) d[i] = const_cast<const float4*>(sr)[i];
    }
}

extern "C" __global__ void strata_copy_i32_from_mapped(int32_t* __restrict__ dst, const volatile int32_t* src, int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) dst[i] = src[i];
}

extern "C" __global__ void strata_doorbell_publish_res(const float* __restrict__ x, const int32_t* __restrict__ ids,
                                                       const int32_t* __restrict__ d_res, int n_expert, int n, int k,
                                                       float* x_out, int32_t* ids_out, uint32_t* seq) {
    int any_miss = 0;
    if ((int) threadIdx.x < k) {
        const int32_t id = ids[threadIdx.x];
        ids_out[threadIdx.x] = id;
        if (d_res == nullptr || id < 0 || id >= n_expert || d_res[id] < 0) any_miss = 1;
    }
    if (__syncthreads_or(any_miss)) {
        for (int i = threadIdx.x; i < n; i += blockDim.x) x_out[i] = x[i];
        __threadfence_system();
    } else if ((int) threadIdx.x < k) {
        __threadfence_system();
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        *(volatile uint32_t*) seq = *(volatile uint32_t*) seq + 1u;
    }
}

extern "C" __global__ void strata_broadcast_streams(const float* __restrict__ x, float* __restrict__ R, long long n, int hc) {
    const int t = blockIdx.y;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * hc) return;
    R[(size_t) t * n * hc + i] = x[(size_t) t * n + i % n];
}

// embed_rows_kernel for a Q8_0 table (dq_dispatch's case 8)
extern "C" __global__ void strata_embed_rows_q8_0(const uint8_t* __restrict__ table, unsigned long long row_bytes,
                                                  const int32_t* __restrict__ tokens, long long n_embd,
                                                  float* __restrict__ y) {
    const int t = blockIdx.y;
    const int64_t b = blockIdx.x;
    const uint8_t* row = table + (size_t) tokens[t] * row_bytes;
    strata_round::dq_q8_0(row, b, y + (size_t) t * n_embd + b * strata_round::QK_K, threadIdx.x);
}

namespace strata_round {

// verify_kernels.cu's multi-block argmax: kArgMaxBlocks blocks a row at most, kArgThreads threads a block.
constexpr int kArgMaxBlocks = 128;

// verify_kernels.hpp: the #879 guard's slices a row.
constexpr int kNonfiniteBlocks = 16;

// the larger value, on equality the lower index: sampler_greedy_kernel's order
__device__ __forceinline__ void arg_take(float ov, int oi, float& bv, int& bi) {
    if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
}

// a block's (value, index) pick, in thread 0; `none` is the index of no candidate
__device__ __forceinline__ void arg_block(float& bv, int& bi, int none) {
    __shared__ float sv[32];
    __shared__ int si[32];
    for (int off = 16; off > 0; off >>= 1) {
        const float ov = __shfl_down_sync(0xffffffffu, bv, off);
        const int oi = __shfl_down_sync(0xffffffffu, bi, off);
        arg_take(ov, oi, bv, bi);
    }
    const int warp = (int) (threadIdx.x >> 5), lane = (int) (threadIdx.x & 31);
    if (lane == 0) { sv[warp] = bv; si[warp] = bi; }
    __syncthreads();
    if (warp == 0) {
        const int nw = (int) (blockDim.x >> 5);
        bv = lane < nw ? sv[lane] : __int_as_float(0xff800000);
        bi = lane < nw ? si[lane] : none;
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_down_sync(0xffffffffu, bv, off);
            const int oi = __shfl_down_sync(0xffffffffu, bi, off);
            arg_take(ov, oi, bv, bi);
        }
    }
}

// A row's scratch, whatever the launch's row count (one scratch serves windows of every size): its counter, then
// the blocks' values and indices.
struct ArgRow {
    unsigned counter;
    unsigned pad[3];
    float v[kArgMaxBlocks];
    int i[kArgMaxBlocks];
};

}  // namespace strata_round

// grid (blocks, rows): block b scans [b * per_block, +per_block) of its row
extern "C" __global__ void strata_argmax_rows(const float* __restrict__ logits, int n, int per_block,
                                              strata_round::ArgRow* __restrict__ rows, int32_t* out) {
    using namespace strata_round;
    const int row = (int) blockIdx.y, b = (int) blockIdx.x, nb = (int) gridDim.x;
    unsigned* counter = &rows[row].counter;
    float* pv = rows[row].v;
    int* pi = rows[row].i;
    const float* l = logits + (size_t) row * n;
    const int lo = b * per_block, hi = min(n, lo + per_block);
    float bv = __int_as_float(0xff800000);   // -inf
    int bi = n;
    for (int v = lo + (int) threadIdx.x; v < hi; v += (int) blockDim.x) {
        const float s = l[v];
        if (s > bv) { bv = s; bi = v; }
    }
    arg_block(bv, bi, n);
    __shared__ bool last;
    if (threadIdx.x == 0) {
        pv[b] = bv;
        pi[b] = bi;
        __threadfence();
        last = atomicAdd(counter, 1u) == (unsigned) (nb - 1);
    }
    __syncthreads();
    if (!last) return;
    __threadfence();
    bv = __int_as_float(0xff800000);
    bi = n;
    if ((int) threadIdx.x < nb) {
        bv = ((volatile const float*) pv)[threadIdx.x];
        bi = ((volatile const int*) pi)[threadIdx.x];
    }
    arg_block(bv, bi, n);
    if (threadIdx.x == 0) {
        out[row] = bi < n ? bi : 0;   // no value above -inf: 0, as the one-block kernel
        *counter = 0;
    }
}

// grid (kNonfiniteBlocks, rows): block b of row t scans [b * per_block, +per_block) and writes its own word.  The
// test is on the bits (exponent all ones), so --use_fast_math cannot fold it away.
extern "C" __global__ void strata_logits_nonfinite(const float* __restrict__ logits, int n, int per_block, uint32_t* flags) {
    const int row = (int) blockIdx.y, b = (int) blockIdx.x;
    const float* l = logits + (size_t) row * n;
    const int lo = b * per_block, hi = min(n, lo + per_block);
    int bad = 0;
    for (int v = lo + (int) threadIdx.x; v < hi; v += (int) blockDim.x)
        bad |= (__float_as_uint(l[v]) & 0x7f800000u) == 0x7f800000u;
    bad = __syncthreads_or(bad);
    if (threadIdx.x == 0) flags[row * strata_round::kNonfiniteBlocks + b] = bad ? 1u : 0u;
}

// ple.cu's history_advance_kernel: each channel's nine-row history shifted up one, the new normalized row last.
extern "C" __global__ void strata_ple_history_advance(float* __restrict__ history, const float* __restrict__ normalized) {
    constexpr int NG_HC_DIM = 10240, NG_HIST = 9;
    const int channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= NG_HC_DIM) return;
    float* column = history + (size_t) channel * NG_HIST;
    for (int row = 0; row + 1 < NG_HIST; ++row) column[row] = column[row + 1];
    column[NG_HIST - 1] = normalized[channel];
}

// shared_expert.cu's sigmoid_scale_rows_kernel and its float4 form: each token's n outputs times sigmoid(g[t]).
extern "C" __global__ void strata_sigmoid_scale_rows(float* __restrict__ out, const float* __restrict__ g, int n) {
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float gt = __fdividef(1.0f, 1.0f + __expf(-g[t]));
        out[(size_t) t * n + i] *= gt;
    }
}

extern "C" __global__ void strata_sigmoid_scale_rows_vec4(float4* __restrict__ out4, const float* __restrict__ g, int n4) {
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4) {
        const float gt = __fdividef(1.0f, 1.0f + __expf(-__ldg(g + t)));
        float4 v = out4[(size_t) t * n4 + i];
        v.x *= gt;
        v.y *= gt;
        v.z *= gt;
        v.w *= gt;
        out4[(size_t) t * n4 + i] = v;
    }
}

extern "C" __global__ void strata_quantize_q8_1(const float* __restrict__ x, strata_round::block_q8_1* __restrict__ y,
                                                long long n) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    strata_round::q8_1_store(x[i], y, i);
}

// verify_kernels.cu's wait_flag_ge_kernel: one thread spins until the host's flag reaches value.
extern "C" __global__ void strata_wait_flag_ge(const volatile uint32_t* flag, uint32_t value) {
    while (*flag < value) strata_round::strata_spin_pause();
    __threadfence_system();
}

// verify_kernels.cu's fetch_blobs_kernel: the first *n blobs, per 16-byte words each, from src[k] to dst back to back.
extern "C" __global__ void strata_fetch_blobs(const unsigned long long* __restrict__ src, const int32_t* __restrict__ n,
                                              uint4* __restrict__ dst, long long per) {
    const long long total = (long long) *n * per;
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (long long) gridDim.x * blockDim.x) {
        const long long k = i / per, off = i - k * per;
        dst[i] = ((const uint4*) src[k])[off];
    }
}

// verify_kernels.cu's rebase_ptrs_kernel: the first *n pointers to the blobs fetch_blobs packed at base.
extern "C" __global__ void strata_rebase_ptrs(unsigned long long* ptr, const int32_t* n, unsigned long long base,
                                              long long bytes) {
    const int k = threadIdx.x;
    if (k < *n) ptr[k] = base + (unsigned long long) k * (unsigned long long) bytes;
}

// s2_expert_grouped.cu's add_hits_kernel (moe_hit_add): the rows dst[0, *count) of parts gain hit_out's.
extern "C" __global__ void strata_add_hits(float* __restrict__ parts, const float* __restrict__ hit_out,
                                           const int32_t* __restrict__ dst, const int32_t* __restrict__ count,
                                           int n_embd) {
    const int h = blockIdx.y;
    if (h >= *count) return;
    const size_t row = (size_t) dst[h] * (size_t) n_embd;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_embd; i += gridDim.x * blockDim.x)
        parts[row + i] += hit_out[row + i];
}

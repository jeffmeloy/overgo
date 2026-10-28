// strata_native_mmvq.cuh - Strata's native MMVQ projections and q8_1 activation quantizers, ported literally into the
// operation module for its decode round (strata-port/decode).
//
// Ported from Strata (https://github.com/Niko1221/Strata, commit 296b59d2), MIT License, Copyright (c) 2026 Niko1221
// and the Strata contributors (licenses/Strata-MIT.txt), from src/kernels/cuda/native_mmvq.cu, q8_1_il.cuh,
// include/strata/kernels/q8_1_finite.hpp and pdl.hpp, which Strata transcribed from llama.cpp's vecdotq.cuh and
// mmvq.cu (ggml/src/ggml-cuda):
//
// MIT License
// Copyright (c) 2023-2026 The ggml authors
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// Strata compiles native_mmvq.cu WITH --use_fast_math, as the operation module is compiled. The device code is
// Strata's, in its namespace; each __global__ template is a __device__ body here, launched through an extern "C"
// entry a template instance the round takes (the entry carries the template's launch bounds), and the host side's
// dispatch (native_mmvq, small_mmvq, launch_multi_n, native_q6_k_mmvq, native_mmvq_il) is translated in
// internal/cuda/executor/launch_strata_native_mmvq.go. The formats are the ones Flash-Next's dense projections and head
// store, Q8_0 and Q6_K, and Q4_K and Q5_K beside Q6_K for every K-quant product of one to eight columns
// (strata-port/widen-windows). Programmatic dependent launch is opt-in in Strata (STRATA_DF_PDL), so pdl_wait and pdl_trigger
// keep their bodies, which compile to nothing below sm_90, and STRATA_PDL_RESTRICT is empty, as on sm_70 and newer.

#include <cstddef>
#include <cstdint>

namespace strata_native {

constexpr int QK = 256;
constexpr int Q8K = 32;
constexpr int WARPS = 4;
constexpr int WARP = 32;
constexpr int QUANT_THREADS = 256;
constexpr int MAX_NCOLS = 8;
// The K-quants' (Q4_K, Q5_K) ints of codes a block (QI) and blocks a thread takes an iteration (VDR).
constexpr int QI = 32;
constexpr int VDR = 2;

#define STRATA_PDL_RESTRICT
#define STRATA_DP4A(a, b, c) __dp4a((a), (b), (c))

// pdl.hpp: the weights a kernel can load before pdl_wait() are loaded there (CUDA, sm_70 and newer)
constexpr bool kPdlPrefetch = true;

__device__ __forceinline__ void pdl_wait() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

__device__ __forceinline__ void pdl_trigger() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.launch_dependents;" :::);
#endif
}

struct Q81Block {
    half2 ds;
    int8_t qs[32];
};
struct Q6KBlock {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    half d;
};
struct Q80Block {
    half d;
    int8_t qs[32];
};
struct Q5KBlock {
    half2 dm;
    uint8_t scales[12];
    uint8_t qh[32];
    uint8_t qs[128];
};
struct Q4KBlock {
    half2 dm;
    uint8_t scales[12];
    uint8_t qs[128];
};
static_assert(sizeof(Q81Block) == 36 && alignof(Q81Block) == 4);
static_assert(sizeof(Q5KBlock) == 176 && alignof(Q5KBlock) == 4);
static_assert(offsetof(Q5KBlock, scales) == 4 && offsetof(Q5KBlock, qh) == 16 &&
              offsetof(Q5KBlock, qs) == 48 && offsetof(Q81Block, qs) == 4);
static_assert(sizeof(Q4KBlock) == 144 && alignof(Q4KBlock) == 4 &&
              offsetof(Q4KBlock, scales) == 4 && offsetof(Q4KBlock, qs) == 16);
static_assert(sizeof(Q6KBlock) == 210 && alignof(Q6KBlock) == 2 &&
              offsetof(Q6KBlock, qh) == 128 && offsetof(Q6KBlock, scales) == 192 &&
              offsetof(Q6KBlock, d) == 208);
static_assert(sizeof(Q80Block) == 34 && alignof(Q80Block) == 2 && offsetof(Q80Block, qs) == 2);

// q8_1_finite.hpp - #606: the q8_1 activation block's scale and sum, kept finite.
constexpr float kHalfMax = 65504.0f;

__device__ __forceinline__ float q8_1_finite(const float v) {
    return fabsf(v) > kHalfMax ? copysignf(kHalfMax, v) : v;
}

/// The block's quantized value of `xi` for the (finite) scale `d`; `amax` = 0 is an all-zero block.
__device__ __forceinline__ int8_t q8_1_quant(const float xi, const float d, const float amax) {
    if (amax == 0.0f) return 0;
    const float q = roundf(xi / d);
    return (int8_t) (q > 127.0f ? 127.0f : q < -127.0f ? -127.0f : q);
}

__device__ __forceinline__ half2 q8_1_ds(const float d, const float sum) {
    return make_half2(__float2half(q8_1_finite(d)), __float2half(q8_1_finite(sum)));
}

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int offset = WARP / 2; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset, WARP);
    }
    return x;
}

__device__ __forceinline__ float warp_max(float x) {
#pragma unroll
    for (int offset = WARP / 2; offset > 0; offset >>= 1) {
        x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, offset, WARP));
    }
    return x;
}

__device__ __forceinline__ void native_quantize_q8_1_kernel(const float* x_, Q81Block* y_, int n_in) {
    const float* STRATA_PDL_RESTRICT x = x_;   // __restrict__ below sm_70 only (pdl.hpp, #1469)
    Q81Block* STRATA_PDL_RESTRICT y = y_;
    pdl_trigger();   // PDL (pdl.hpp): the projection after this one may start loading its weights
    pdl_wait();
    const int i = int(blockIdx.x) * QUANT_THREADS + int(threadIdx.x);
    if (i >= n_in) return; // n_in is a multiple of 32: only whole warps return.
    const float xi = x[i];
    const float amax = warp_max(fabsf(xi));
    const float sum = warp_sum(xi);
    const float d = q8_1_finite(amax / 127.0f);   // #606: q8_1_finite.hpp - the same bits for every finite block
    const int8_t q = q8_1_quant(xi, d, amax);
    y[i / Q8K].qs[i % Q8K] = q;
    if (i % Q8K == 0) y[i / Q8K].ds = q8_1_ds(d, sum);
}

__device__ __forceinline__ void native_swiglu_quantize_q8_1_kernel(const float* __restrict__ gate,
                                                                   const float* __restrict__ up,
                                                                   Q81Block* __restrict__ y, int n_in) {
    const int i = int(blockIdx.x) * QUANT_THREADS + int(threadIdx.x);
    if (i >= n_in) return; // n_in is a multiple of 32: only whole warps return.
    const float gi = gate[i];
    const float xi = __fmul_rn(__fdividef(gi, __fadd_rn(1.0f, __expf(-gi))), up[i]);
    const float amax = warp_max(fabsf(xi));
    const float sum = warp_sum(xi);
    const float d = q8_1_finite(amax / 127.0f);   // #606, as native_quantize_q8_1_kernel: finite blocks bit for bit
    const int8_t q = q8_1_quant(xi, d, amax);
    y[i / Q8K].qs[i % Q8K] = q;
    if (i % Q8K == 0) y[i / Q8K].ds = q8_1_ds(d, sum);
}

// Q3_K's 110-byte stride gives alternate blocks only two-byte alignment.
// Preserve the pinned helper's pair of 16-bit loads and little-endian combine.
__device__ __forceinline__ int load_int_b2(const void* ptr, int i32) {
    const auto* x = static_cast<const uint16_t*>(ptr);
    int value = x[2 * i32] << 0;
    value |= x[2 * i32 + 1] << 16;
    return value;
}

// Exact pinned vec_dot_q6_K_q8_1: keep signed per-16-element scales,
// signed-byte subtraction, DP4A order, and the float accumulation sequence.
__device__ __forceinline__ float q6_q8_dot_impl(int vl, int vh, const int* __restrict__ u,
                                              const int8_t* __restrict__ scales,
                                              float d, const float* __restrict__ d8) {
    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int sc = scales[4 * i];
        const int vil = (vl >> (4 * i)) & 0x0f0f0f0f;
        const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
        const int vi = __vsubss4(vil | vih, 0x20202020);
        sumf += d8[i] * (STRATA_DP4A(vi, u[i], 0) * sc);
    }
    return d * sumf;
}

__device__ __forceinline__ float q6_q8_dot(const Q6KBlock* __restrict__ w,
                                          const Q81Block* __restrict__ x, int iqs) {
    const int bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
    const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
    const int vh_shift = 2 * ((iqs % 16) / 8);
    const int vl = load_int_b2(w->ql, iqs);
    const int vh = load_int_b2(w->qh, 8 * (iqs / 16) + iqs % 8) >> vh_shift;
    int u[2];
    float d8[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        u[i] = reinterpret_cast<const int*>(x[bq8_offset + 2 * i].qs)[iqs % 8];
        d8[i] = __low2float(x[bq8_offset + 2 * i].ds);
    }
    return q6_q8_dot_impl(vl, vh, u, w->scales + scale_offset, w->d, d8);
}

// Q6_K generic MMVQ: QK=256, QI=32, VDR=1, four blocks per iteration.
template<bool SmallK>
__device__ __forceinline__ void native_q6_k_mmvq_kernel(const Q6KBlock* __restrict__ w,
                                                        const Q81Block* __restrict__ x,
                                                        float* __restrict__ y, int n_in, int n_out) {
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = WARPS * WARP / 32;
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int row0 = ROWS * int(blockIdx.x);
    const int blocks_per_row = n_in / 256;
    float tmp[ROWS] = {};
    for (int kbx = tid / 32; kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kby = kbx * 8;
        const int kqs = tid % 32;
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += q6_q8_dot(w + block, x + kby, kqs);
            }
        }
    }
    __shared__ float partial[WARPS - 1][ROWS][WARP];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) partial[threadIdx.y - 1][i][threadIdx.x] = tmp[i];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l) tmp[i] += partial[l][i][threadIdx.x];
        tmp[i] = warp_sum(tmp[i]);
        if (threadIdx.x == i && row0 + i < n_out) y[row0 + i] = tmp[i];
    }
}

// Exact pinned vec_dot_q5_K_q8_1_impl_vmmq expression and integer dot order.
__device__ __forceinline__ float q5_q8_dot_impl(
    const int* __restrict__ vl, const int* __restrict__ vh, const int* __restrict__ u,
    const uint8_t* __restrict__ sc, const uint8_t* __restrict__ m, const half2& dm5,
    const float* __restrict__ d8) {
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int vl0i = (vl[0] >> (4 * i)) & 0x0f0f0f0f;
        const int vl1i = (vl[1] >> (4 * i)) & 0x0f0f0f0f;
        const int vh0i = ((vh[0] >> i) << 4) & 0x10101010;
        const int vh1i = ((vh[1] >> i) << 4) & 0x10101010;
        const int v0i = vl0i | vh0i;
        const int v1i = vl1i | vh1i;
        const int dot1 = STRATA_DP4A(v0i, u[2 * i], STRATA_DP4A(v1i, u[2 * i + 1], 0));
        const int dot2 = STRATA_DP4A(0x01010101, u[2 * i], STRATA_DP4A(0x01010101, u[2 * i + 1], 0));
        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }
    const float2 dm5f = __half22float2(dm5);
    return dm5f.x * sumf_d - dm5f.y * sumf_m;
}

// The 6-bit scale and min pair of a K-quant block's sub-block pair j (get_scale_min_k4 for two sub-blocks at once).
__device__ __forceinline__ void k_scale_min_pair(const uint8_t* __restrict__ packed, int j, uint16_t aux[2]) {
    const uint16_t* scales = reinterpret_cast<const uint16_t*>(packed);
    const int jm = j & 1;
    const uint32_t s0 = scales[jm];
    const uint32_t s2 = scales[jm + 2];
    const uint32_t s4 = scales[jm + 4];
    const uint32_t hi = uint32_t(-int32_t(j >= 2));
    aux[0] = uint16_t(((s0 & 0x3f3f) & ~hi) |
                      ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
    aux[1] = uint16_t(((s2 & 0x3f3f) & ~hi) |
                      ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
}

// A K-quant dot's activation side: the two q8_1 blocks' codes at the thread's ints and their scales.
__device__ __forceinline__ void k_q8_codes(const Q81Block* __restrict__ bq8, int bq8_offset, int iqs, int u[4],
                                           float d8[2]) {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const Q81Block* bq8i = bq8 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);
        const int* q8 = reinterpret_cast<const int*>(bq8i->qs) + ((iqs / 2) % 4);
        u[2 * i] = q8[0];
        u[2 * i + 1] = q8[4];
    }
}

__device__ __forceinline__ float q5_q8_dot(const Q5KBlock* __restrict__ bq5,
                                          const Q81Block* __restrict__ bq8, int iqs) {
    int vl[2];
    int vh[2];
    int u[4];
    float d8[2];
    const int bq8_offset = 2 * ((iqs / 2) / 4);
    const int* ql = reinterpret_cast<const int*>(bq5->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    const int* qh = reinterpret_cast<const int*>(bq5->qh + 4 * ((iqs / 2) % 4));
    vl[0] = ql[0];
    vl[1] = ql[4];
    vh[0] = qh[0] >> bq8_offset;
    vh[1] = qh[4] >> bq8_offset;
    uint16_t aux[2];
    k_scale_min_pair(bq5->scales, bq8_offset / 2, aux);
    const uint8_t* sc = reinterpret_cast<const uint8_t*>(aux);
    k_q8_codes(bq8, bq8_offset, iqs, u, d8);
    return q5_q8_dot_impl(vl, vh, u, sc, sc + 2, bq5->dm, d8);
}

// Exact pinned vec_dot_q4_K_q8_1_impl_vmmq expression and integer dot order.
__device__ __forceinline__ float q4_q8_dot_impl(
    const int* __restrict__ v, const int* __restrict__ u,
    const uint8_t* __restrict__ sc, const uint8_t* __restrict__ m, const half2& dm4,
    const float* __restrict__ d8) {
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int v0i = (v[0] >> (4 * i)) & 0x0f0f0f0f;
        const int v1i = (v[1] >> (4 * i)) & 0x0f0f0f0f;
        const int dot1 = STRATA_DP4A(v1i, u[2 * i + 1], STRATA_DP4A(v0i, u[2 * i], 0));
        const int dot2 = STRATA_DP4A(0x01010101, u[2 * i + 1], STRATA_DP4A(0x01010101, u[2 * i], 0));
        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }
    const float2 dm4f = __half22float2(dm4);
    return dm4f.x * sumf_d - dm4f.y * sumf_m;
}

__device__ __forceinline__ float q4_q8_dot(const Q4KBlock* __restrict__ bq4,
                                          const Q81Block* __restrict__ bq8, int iqs) {
    int v[2];
    int u[4];
    float d8[2];
    const int bq8_offset = 2 * ((iqs / 2) / 4);
    const int* ql = reinterpret_cast<const int*>(bq4->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    v[0] = ql[0];
    v[1] = ql[4];
    uint16_t aux[2];
    k_scale_min_pair(bq4->scales, bq8_offset / 2, aux);
    const uint8_t* sc = reinterpret_cast<const uint8_t*>(aux);
    k_q8_codes(bq8, bq8_offset, iqs, u, d8);
    return q4_q8_dot_impl(v, u, sc, sc + 2, bq4->dm, d8);
}

// The K-quants' generic ncols=1 MMVQ (native_q5_k_mmvq_kernel, native_q4_k_mmvq_kernel): 4 warps and 1 row (or 4 rows
// for small K), eight weight blocks per K iteration, warp-ascending shared sum, then XOR tree.
template<typename Block, float (*Dot)(const Block*, const Q81Block*, int), bool SmallK>
__device__ __forceinline__ void native_k_mmvq_kernel(const Block* __restrict__ w, const Q81Block* __restrict__ x,
                                                     float* __restrict__ y, int n_in, int n_out) {
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = VDR * WARPS * WARP / QI;
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int row0 = ROWS * int(blockIdx.x);
    const int blocks_per_row = n_in / QK;
    float tmp[ROWS] = {};
    for (int kbx = tid / (QI / VDR); kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kby = kbx * (QK / Q8K);
        const int kqs = VDR * (tid % (QI / VDR));
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            // The source assumes allocator padding for partial row groups. This
            // guard preserves every valid row's math without an out-of-bounds read.
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += Dot(w + block, x + kby, kqs);
            }
        }
    }
    __shared__ float partial[WARPS - 1][ROWS][WARP];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) partial[threadIdx.y - 1][i][threadIdx.x] = tmp[i];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l) tmp[i] += partial[l][i][threadIdx.x];
        tmp[i] = warp_sum(tmp[i]);
        if (threadIdx.x == i && row0 + i < n_out) y[row0 + i] = tmp[i];
    }
}

__device__ __forceinline__ float small_q8_dot(const Q80Block* __restrict__ w,
                                             const Q81Block* __restrict__ x, int iqs) {
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int v = load_int_b2(w->qs, iqs + i);
        const int u = reinterpret_cast<const int*>(x->qs)[iqs + i];
        sumi = STRATA_DP4A(v, u, sumi);
    }
    const float d0 = w->d;
    const float d1 = __low2float(x->ds);
    return d0 * d1 * float(sumi);
}

// QI=4 for Q4_0/Q5_0/IQ4_NL and QI=8 for Q8_0. With VDR=2 this preserves
// the pinned 64/32-block iteration and 2048/1024-element small-K thresholds.
template<typename Weight, int Qi, bool SmallK>
__device__ __forceinline__ void native_small_mmvq_kernel(const Weight* __restrict__ w,
                                                         const Q81Block* __restrict__ x,
                                                         float* __restrict__ y, int n_in, int n_out) {
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = 2 * WARPS * WARP / Qi;
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int row0 = ROWS * int(blockIdx.x);
    const int blocks_per_row = n_in / 32;
    float tmp[ROWS] = {};
    for (int kbx = tid / (Qi / 2); kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kqs = 2 * (tid % (Qi / 2));
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += small_q8_dot(w + block, x + kbx, kqs);
            }
        }
    }
    __shared__ float partial[WARPS - 1][ROWS][WARP];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) partial[threadIdx.y - 1][i][threadIdx.x] = tmp[i];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l) tmp[i] += partial[l][i][threadIdx.x];
        tmp[i] = warp_sum(tmp[i]);
        if (threadIdx.x == i && row0 + i < n_out) y[row0 + i] = tmp[i];
    }
}

// Each format splits its dot product into `load` (everything that depends only on the weight block: codes,
// scales) and `apply` (the activation-dependent part), so the multi-column kernel decodes a weight block once.
struct Q6KTraits {
    using Block = Q6KBlock;
    static constexpr int DIV = 256, T = 32, KBY = 8, BPI = WARPS * WARP / 32;
    __device__ static int kqs(int tid) { return tid % 32; }
    struct W { int vl, vh; float d; const int8_t* scales; int bq8_offset; };
    __device__ static W load(const Block* __restrict__ w, int iqs) {
        W r;
        r.bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
        const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
        const int vh_shift = 2 * ((iqs % 16) / 8);
        r.vl = load_int_b2(w->ql, iqs);
        r.vh = load_int_b2(w->qh, 8 * (iqs / 16) + iqs % 8) >> vh_shift;
        r.scales = w->scales + scale_offset;
        r.d = w->d;
        return r;
    }
    __device__ static float apply(const W& r, const Q81Block* __restrict__ x, int iqs) {
        int u[2];
        float d8[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            u[i] = reinterpret_cast<const int*>(x[r.bq8_offset + 2 * i].qs)[iqs % 8];
            d8[i] = __low2float(x[r.bq8_offset + 2 * i].ds);
        }
        return q6_q8_dot_impl(r.vl, r.vh, u, r.scales, r.d, d8);
    }
};

// Q5_K's and Q4_K's split: `load` reads the codes and unpacks the scale and min pair, `apply` calls the same impl as
// the ncols = 1 dot with the same values.
struct Q5KTraits {
    using Block = Q5KBlock;
    static constexpr int DIV = QK, T = QI / VDR, KBY = QK / Q8K, BPI = VDR * WARPS * WARP / QI;
    __device__ static int kqs(int tid) { return VDR * (tid % (QI / VDR)); }
    struct W { int vl[2], vh[2]; uint16_t aux[2]; half2 dm; int bq8_offset; };
    __device__ static W load(const Block* __restrict__ bq5, int iqs) {
        W r;
        r.bq8_offset = 2 * ((iqs / 2) / 4);
        const int* ql = reinterpret_cast<const int*>(bq5->qs + 16 * r.bq8_offset + 4 * ((iqs / 2) % 4));
        const int* qh = reinterpret_cast<const int*>(bq5->qh + 4 * ((iqs / 2) % 4));
        r.vl[0] = ql[0];
        r.vl[1] = ql[4];
        r.vh[0] = qh[0] >> r.bq8_offset;
        r.vh[1] = qh[4] >> r.bq8_offset;
        k_scale_min_pair(bq5->scales, r.bq8_offset / 2, r.aux);
        r.dm = bq5->dm;
        return r;
    }
    __device__ static float apply(const W& r, const Q81Block* __restrict__ bq8, int iqs) {
        int u[4];
        float d8[2];
        k_q8_codes(bq8, r.bq8_offset, iqs, u, d8);
        const uint8_t* sc = reinterpret_cast<const uint8_t*>(r.aux);
        return q5_q8_dot_impl(r.vl, r.vh, u, sc, sc + 2, r.dm, d8);
    }
};
struct Q4KTraits {
    using Block = Q4KBlock;
    static constexpr int DIV = QK, T = QI / VDR, KBY = QK / Q8K, BPI = VDR * WARPS * WARP / QI;
    __device__ static int kqs(int tid) { return VDR * (tid % (QI / VDR)); }
    struct W { int v[2]; uint16_t aux[2]; half2 dm; int bq8_offset; };
    __device__ static W load(const Block* __restrict__ bq4, int iqs) {
        W r;
        r.bq8_offset = 2 * ((iqs / 2) / 4);
        const int* ql = reinterpret_cast<const int*>(bq4->qs + 16 * r.bq8_offset + 4 * ((iqs / 2) % 4));
        r.v[0] = ql[0];
        r.v[1] = ql[4];
        k_scale_min_pair(bq4->scales, r.bq8_offset / 2, r.aux);
        r.dm = bq4->dm;
        return r;
    }
    __device__ static float apply(const W& r, const Q81Block* __restrict__ bq8, int iqs) {
        int u[4];
        float d8[2];
        k_q8_codes(bq8, r.bq8_offset, iqs, u, d8);
        const uint8_t* sc = reinterpret_cast<const uint8_t*>(r.aux);
        return q4_q8_dot_impl(r.v, u, sc, sc + 2, r.dm, d8);
    }
};

template<typename Weight, int Qi>
struct SmallTraits;

template<>
struct SmallTraits<Q80Block, 8> {
    using Block = Q80Block;
    static constexpr int DIV = 32, T = 4, KBY = 1, BPI = 2 * WARPS * WARP / 8;
    __device__ static int kqs(int tid) { return 2 * (tid % 4); }
    struct W { int v[2]; float d0; };
    __device__ static W load(const Block* __restrict__ w, int iqs) {
        W r;
#pragma unroll
        for (int i = 0; i < 2; ++i) r.v[i] = load_int_b2(w->qs, iqs + i);
        r.d0 = w->d;
        return r;
    }
    __device__ static float apply(const W& r, const Q81Block* __restrict__ x, int iqs) {
        int sumi = 0;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int u = reinterpret_cast<const int*>(x->qs)[iqs + i];
            sumi = STRATA_DP4A(r.v[i], u, sumi);
        }
        const float d1 = __low2float(x->ds);
        return r.d0 * d1 * float(sumi);
    }
};

// NW warps per block and ROWS rows per block. The EXACT layout (NW = 4, ROWS = 1, or 4 for small K) is the
// ncols = 1 layout and keeps every column bitwise equal to a single-column call.
template<typename F, int NCOLS, int NW, int ROWS>
__device__ __forceinline__ void native_mmvq_multi_kernel(const typename F::Block* __restrict__ w, const Q81Block* x_,
                                                         float* y_, int n_in, int n_out) {
    const Q81Block* STRATA_PDL_RESTRICT x = x_;   // __restrict__ below sm_70 only (pdl.hpp, #1469)
    float* STRATA_PDL_RESTRICT y = y_;
    constexpr int BPI = F::BPI * NW / WARPS;           // blocks per iteration scale with the warp count
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int bxi = int(blockIdx.x);
    const int row0 = ROWS * bxi;
    const int blocks_per_row = n_in / F::DIV;
    const int x_stride = n_in / Q8K;                   // Q8_1 blocks per activation column
    float tmp[NCOLS][ROWS] = {};
    int kbx = tid / F::T;
    if constexpr (kPdlPrefetch) {
        // PDL (pdl.hpp): the first block's weights are loaded before waiting for the activations, then that block
        // is applied exactly as the loop below would apply it (the same rows, columns and order)
        typename F::W w0[ROWS];
        const int kqs = F::kqs(tid);
        if (kbx < blocks_per_row) {
#pragma unroll
            for (int i = 0; i < ROWS; ++i)
                if (row0 + i < n_out) w0[i] = F::load(w + std::size_t(row0 + i) * blocks_per_row + kbx, kqs);
        }
        pdl_trigger();
        pdl_wait();
        if (kbx < blocks_per_row) {
            const int kby = kbx * F::KBY;
#pragma unroll
            for (int i = 0; i < ROWS; ++i) {
                if (row0 + i < n_out) {
#pragma unroll
                    for (int j = 0; j < NCOLS; ++j)
                        tmp[j][i] += F::apply(w0[i], x + std::size_t(j) * x_stride + kby, kqs);
                }
            }
        }
        kbx += BPI;
    }
    for (; kbx < blocks_per_row; kbx += BPI) {
        const int kby = kbx * F::KBY;
        const int kqs = F::kqs(tid);
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                const typename F::W wv = F::load(w + block, kqs);      // once per (row, block)
#pragma unroll
                for (int j = 0; j < NCOLS; ++j)                         // then per column
                    tmp[j][i] += F::apply(wv, x + std::size_t(j) * x_stride + kby, kqs);
            }
        }
    }
    __shared__ float partial[NW - 1 > 0 ? NW - 1 : 1][NCOLS][ROWS][WARP];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < NCOLS; ++j)
#pragma unroll
            for (int i = 0; i < ROWS; ++i) partial[threadIdx.y - 1][j][i][threadIdx.x] = tmp[j][i];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int j = 0; j < NCOLS; ++j) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
#pragma unroll
            for (int l = 0; l < NW - 1; ++l) tmp[j][i] += partial[l][j][i][threadIdx.x];
            tmp[j][i] = warp_sum(tmp[j][i]);
            if (threadIdx.x == i && row0 + i < n_out) y[std::size_t(j) * n_out + row0 + i] = tmp[j][i];
        }
    }
}

// q8_1_il.cuh / native_mmvq.hpp: the interleaved copy of 2..8 q8_1 activation columns as the multi-column dot products
// read it.
constexpr int native_q8_1_il_cp(int ncols) { return ncols <= 2 ? 2 : ncols <= 4 ? 4 : 8; }

// NC columns of the block-major (PM false) or position-major copy: u(b, p, o) = int p of q8_1 block b of every column,
// scales(b, o) = their scales
template <int NC, bool PM> struct Q81Il {
    static constexpr int CP = native_q8_1_il_cp(NC);
    const int* __restrict__ q;
    const float* __restrict__ d;
    int nb;   // blocks a column
    __device__ __forceinline__ void u(int b, int p, int (&o)[NC]) const {
        const int* s = q + std::size_t(PM ? p * nb + b : b * 8 + p) * CP;
        if constexpr (CP == 2) {
            const int2 a = *reinterpret_cast<const int2*>(s);
            o[0] = a.x;
            o[1] = a.y;
        } else {
            const int4 a = *reinterpret_cast<const int4*>(s);
            o[0] = a.x;
            if constexpr (NC > 1) o[1] = a.y;
            if constexpr (NC > 2) o[2] = a.z;
            if constexpr (NC > 3) o[3] = a.w;
            if constexpr (NC > 4) {
                const int4 e = *reinterpret_cast<const int4*>(s + 4);
                o[4] = e.x;
                if constexpr (NC > 5) o[5] = e.y;
                if constexpr (NC > 6) o[6] = e.z;
                if constexpr (NC > 7) o[7] = e.w;
            }
        }
    }
    __device__ __forceinline__ void scales(int b, float (&o)[NC]) const {
        const float* s = d + std::size_t(b) * CP;
        if constexpr (CP == 2) {
            const float2 a = *reinterpret_cast<const float2*>(s);
            o[0] = a.x;
            o[1] = a.y;
        } else {
            const float4 a = *reinterpret_cast<const float4*>(s);
            o[0] = a.x;
            if constexpr (NC > 1) o[1] = a.y;
            if constexpr (NC > 2) o[2] = a.z;
            if constexpr (NC > 3) o[3] = a.w;
            if constexpr (NC > 4) {
                const float4 e = *reinterpret_cast<const float4*>(s + 4);
                o[4] = e.x;
                if constexpr (NC > 5) o[5] = e.y;
                if constexpr (NC > 6) o[6] = e.z;
                if constexpr (NC > 7) o[7] = e.w;
            }
        }
    }
};

// One column's plain q8_1 blocks (36 bytes: the half2 scale and sum, then 32 codes) behind Q81Il's interface
struct Q81One {
    const uint8_t* __restrict__ x;
    __device__ __forceinline__ void u(int b, int p, int (&o)[1]) const {
        o[0] = reinterpret_cast<const int*>(x + std::size_t(b) * 36 + 4)[p];
    }
    __device__ __forceinline__ void scales(int b, float (&o)[1]) const {
        o[0] = __low2float(*reinterpret_cast<const half2*>(x + std::size_t(b) * 36));
    }
};

// The activations a kernel for NC columns reads: one column's q8_1 blocks, 2+ columns' interleaved copy (nb blocks a
// column)
template <int NC, bool PM>
__device__ __forceinline__ auto q8_1_cols(const void* xq, const float* xd, int nb) {
    if constexpr (NC == 1) return Q81One{static_cast<const uint8_t*>(xq)};
    else return Q81Il<NC, PM>{static_cast<const int*>(xq), xd, nb};
}

// ============================ ncols = 2..4 from interleaved activations (fork F4, Eddoursul) ============================
// `native_quantize_q8_1_il` also writes the columns interleaved (native_mmvq.hpp), so one load reads the same int of
// every column. A warp takes R rows and walks their blocks in the chunks of the exact layout above - its lane l's chunk
// m is what thread 32 (m % 4) + l does in iteration m / 4 - with a sum per virtual warp, added in warp order before the
// xor tree; each weight block is decoded once (the traits' `load`) and every column takes the single-column dot's own
// expression (the same *_q8_dot_impl calls), so each value is bitwise the multi-column kernel's.
struct IlQ6K {
    using F = Q6KTraits;
    static constexpr bool PM = false;
    template <int NC, class X>
    __device__ static void apply(const F::W& r, const X& x, int kby, int iqs, float (&out)[NC]) {
        int u[2][NC];
        float d8[2][NC];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            x.u(kby + r.bq8_offset + 2 * i, iqs % 8, u[i]);
            x.scales(kby + r.bq8_offset + 2 * i, d8[i]);
        }
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            const int uc[2] = {u[0][c], u[1][c]};
            const float dc[2] = {d8[0][c], d8[1][c]};
            out[c] = q6_q8_dot_impl(r.vl, r.vh, uc, r.scales, r.d, dc);
        }
    }
};

template <typename I, int NC, int R>
__device__ __forceinline__ void native_mmvq_il_kernel(const typename I::F::Block* __restrict__ w,
                                                      const int* __restrict__ xq, const float* __restrict__ xd,
                                                      float* __restrict__ y, int n_in, int n_out) {
    using F = typename I::F;
    constexpr int SUB = WARP / F::T;   // weight blocks a chunk covers
    const int lane = int(threadIdx.x) & (WARP - 1);
    const auto x = q8_1_cols<NC, I::PM>(xq, xd, n_in / Q8K);
    const int bpr = n_in / F::DIV;
    const int kqs = F::kqs(lane);
    const int nm = (bpr + SUB - 1) / SUB;
    // A grid-stride loop over the row groups, though the grid gives each warp one: without it ptxas gives several
    // instances fewer registers (the fork measured Q6_K's at 3 columns 10-50% slower).
    for (int grp = int(blockIdx.x) * 4 + (int(threadIdx.x) >> 5); grp * R < n_out; grp += int(gridDim.x) * 4) {
        const int row0 = grp * R;
        float acc[4][R][NC];
#pragma unroll
        for (int v = 0; v < 4; ++v)
#pragma unroll
            for (int i = 0; i < R; ++i)
#pragma unroll
                for (int c = 0; c < NC; ++c) acc[v][i][c] = 0.0f;
        for (int m0 = 0; m0 < nm; m0 += 4) {
#pragma unroll
            for (int v = 0; v < 4; ++v) {
                const int kbx = SUB * (m0 + v) + lane / F::T;
                if (kbx < bpr) {
#pragma unroll
                    for (int i = 0; i < R; ++i) {
                        const int row = min(row0 + i, n_out - 1);   // a partial last group recomputes its last row
                        const typename F::W wv = F::load(w + std::size_t(row) * bpr + kbx, kqs);
                        float o[NC];
                        I::template apply<NC>(wv, x, kbx * F::KBY, kqs, o);
#pragma unroll
                        for (int c = 0; c < NC; ++c) acc[v][i][c] += o[c];
                    }
                }
            }
        }
#pragma unroll
        for (int i = 0; i < R; ++i)
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                float s = acc[0][i][c];
                s += acc[1][i][c];
                s += acc[2][i][c];
                s += acc[3][i][c];
                s = warp_sum(s);
                if (lane == 0 && row0 + i < n_out) y[std::size_t(c) * n_out + row0 + i] = s;
            }
    }
}

// The interleaved copy (native_mmvq.hpp) of columns already quantized into plain q8_1 blocks, whoever wrote them (the
// quantizer, a fused norm): the same bytes, one int of a block a thread.
__device__ __forceinline__ void native_q8_1_interleave_kernel(const Q81Block* __restrict__ y, int* __restrict__ bm,
                                                              int* __restrict__ pm, float* __restrict__ dl, int nb,
                                                              int n_total, int cp) {
    const int i = int(blockIdx.x) * 256 + int(threadIdx.x);
    if (i >= n_total) return;
    const int c = i / (nb * 8), r = i - c * nb * 8, b = r / 8, p = r % 8;
    const Q81Block& blk = y[std::size_t(c) * nb + b];
    const int q = reinterpret_cast<const int*>(blk.qs)[p];
    bm[(std::size_t(b) * 8 + p) * cp + c] = q;
    pm[(std::size_t(p) * nb + b) * cp + c] = q;
    if (p == 0) dl[std::size_t(b) * cp + c] = __low2float(blk.ds);
}

}  // namespace strata_native

// ---- The entries: a template instance each the round launches, with its template's launch bounds.

extern "C" __global__ void __launch_bounds__(strata_native::QUANT_THREADS, 1) strata_native_quantize_q8_1(
        const float * x, strata_native::Q81Block * y, int n_in) {
    strata_native::native_quantize_q8_1_kernel(x, y, n_in);
}

extern "C" __global__ void __launch_bounds__(strata_native::QUANT_THREADS, 1) strata_native_swiglu_quantize_q8_1(
        const float * __restrict__ gate, const float * __restrict__ up, strata_native::Q81Block * __restrict__ y,
        int n_in) {
    strata_native::native_swiglu_quantize_q8_1_kernel(gate, up, y, n_in);
}

#define STRATA_SMALL_MMVQ(NAME, WEIGHT, QI, SMALLK)                                                                     \
    extern "C" __global__ void __launch_bounds__(strata_native::WARPS * strata_native::WARP, 1) NAME(                  \
            const strata_native::WEIGHT * __restrict__ w, const strata_native::Q81Block * __restrict__ x,             \
            float * __restrict__ y, int n_in, int n_out) {                                                             \
        strata_native::native_small_mmvq_kernel<strata_native::WEIGHT, QI, SMALLK>(w, x, y, n_in, n_out);              \
    }
STRATA_SMALL_MMVQ(strata_native_small_mmvq_q8_0, Q80Block, 8, false)
STRATA_SMALL_MMVQ(strata_native_small_mmvq_q8_0_small_k, Q80Block, 8, true)
#undef STRATA_SMALL_MMVQ

#define STRATA_Q6_K_MMVQ(NAME, SMALLK)                                                                                  \
    extern "C" __global__ void __launch_bounds__(strata_native::WARPS * strata_native::WARP, 1) NAME(                  \
            const strata_native::Q6KBlock * __restrict__ w, const strata_native::Q81Block * __restrict__ x,           \
            float * __restrict__ y, int n_in, int n_out) {                                                             \
        strata_native::native_q6_k_mmvq_kernel<SMALLK>(w, x, y, n_in, n_out);                                          \
    }
STRATA_Q6_K_MMVQ(strata_native_q6_k_mmvq, false)
STRATA_Q6_K_MMVQ(strata_native_q6_k_mmvq_small_k, true)
#undef STRATA_Q6_K_MMVQ

#define STRATA_K_MMVQ(NAME, BLOCK, DOT, SMALLK)                                                                         \
    extern "C" __global__ void __launch_bounds__(strata_native::WARPS * strata_native::WARP, 1) NAME(                  \
            const strata_native::BLOCK * __restrict__ w, const strata_native::Q81Block * __restrict__ x,              \
            float * __restrict__ y, int n_in, int n_out) {                                                             \
        strata_native::native_k_mmvq_kernel<strata_native::BLOCK, strata_native::DOT, SMALLK>(w, x, y, n_in, n_out);  \
    }
STRATA_K_MMVQ(strata_native_q5_k_mmvq, Q5KBlock, q5_q8_dot, false)
STRATA_K_MMVQ(strata_native_q5_k_mmvq_small_k, Q5KBlock, q5_q8_dot, true)
STRATA_K_MMVQ(strata_native_q4_k_mmvq, Q4KBlock, q4_q8_dot, false)
STRATA_K_MMVQ(strata_native_q4_k_mmvq_small_k, Q4KBlock, q4_q8_dot, true)
#undef STRATA_K_MMVQ

// launch_multi_n's exact layout: four warps, two rows a block (one row a block only where measured faster on a gfx12 or
// sm_86 card, neither of which this is).
#define STRATA_MMVQ_MULTI(NAME, TRAITS, NCOLS)                                                                          \
    extern "C" __global__ void __launch_bounds__(strata_native::WARPS * strata_native::WARP, 4) NAME(                  \
            const strata_native::TRAITS::Block * __restrict__ w, const strata_native::Q81Block * x, float * y,        \
            int n_in, int n_out) {                                                                                     \
        strata_native::native_mmvq_multi_kernel<strata_native::TRAITS, NCOLS, strata_native::WARPS, 2>(w, x, y, n_in,   \
                                                                                                       n_out);         \
    }
#define STRATA_Q8_0_TRAITS SmallTraits<strata_native::Q80Block, 8>
STRATA_MMVQ_MULTI(strata_native_mmvq_multi_q8_0_c2, STRATA_Q8_0_TRAITS, 2)
STRATA_MMVQ_MULTI(strata_native_mmvq_multi_q8_0_c3, STRATA_Q8_0_TRAITS, 3)
STRATA_MMVQ_MULTI(strata_native_mmvq_multi_q8_0_c4, STRATA_Q8_0_TRAITS, 4)
STRATA_MMVQ_MULTI(strata_native_mmvq_multi_q8_0_c5, STRATA_Q8_0_TRAITS, 5)
STRATA_MMVQ_MULTI(strata_native_mmvq_multi_q8_0_c6, STRATA_Q8_0_TRAITS, 6)
STRATA_MMVQ_MULTI(strata_native_mmvq_multi_q8_0_c7, STRATA_Q8_0_TRAITS, 7)
STRATA_MMVQ_MULTI(strata_native_mmvq_multi_q8_0_c8, STRATA_Q8_0_TRAITS, 8)
STRATA_MMVQ_MULTI(strata_native_mmvq_multi_q6_k_c2, Q6KTraits, 2)
STRATA_MMVQ_MULTI(strata_native_mmvq_multi_q6_k_c3, Q6KTraits, 3)
STRATA_MMVQ_MULTI(strata_native_mmvq_multi_q6_k_c4, Q6KTraits, 4)
STRATA_MMVQ_MULTI(strata_native_mmvq_multi_q6_k_c5, Q6KTraits, 5)
STRATA_MMVQ_MULTI(strata_native_mmvq_multi_q6_k_c6, Q6KTraits, 6)
STRATA_MMVQ_MULTI(strata_native_mmvq_multi_q6_k_c7, Q6KTraits, 7)
STRATA_MMVQ_MULTI(strata_native_mmvq_multi_q6_k_c8, Q6KTraits, 8)
STRATA_MMVQ_MULTI(strata_native_mmvq_multi_q5_k_c2, Q5KTraits, 2)
STRATA_MMVQ_MULTI(strata_native_mmvq_multi_q5_k_c3, Q5KTraits, 3)
STRATA_MMVQ_MULTI(strata_native_mmvq_multi_q5_k_c4, Q5KTraits, 4)
STRATA_MMVQ_MULTI(strata_native_mmvq_multi_q5_k_c5, Q5KTraits, 5)
STRATA_MMVQ_MULTI(strata_native_mmvq_multi_q5_k_c6, Q5KTraits, 6)
STRATA_MMVQ_MULTI(strata_native_mmvq_multi_q5_k_c7, Q5KTraits, 7)
STRATA_MMVQ_MULTI(strata_native_mmvq_multi_q5_k_c8, Q5KTraits, 8)
STRATA_MMVQ_MULTI(strata_native_mmvq_multi_q4_k_c2, Q4KTraits, 2)
STRATA_MMVQ_MULTI(strata_native_mmvq_multi_q4_k_c3, Q4KTraits, 3)
STRATA_MMVQ_MULTI(strata_native_mmvq_multi_q4_k_c4, Q4KTraits, 4)
STRATA_MMVQ_MULTI(strata_native_mmvq_multi_q4_k_c5, Q4KTraits, 5)
STRATA_MMVQ_MULTI(strata_native_mmvq_multi_q4_k_c6, Q4KTraits, 6)
STRATA_MMVQ_MULTI(strata_native_mmvq_multi_q4_k_c7, Q4KTraits, 7)
STRATA_MMVQ_MULTI(strata_native_mmvq_multi_q4_k_c8, Q4KTraits, 8)
#undef STRATA_Q8_0_TRAITS
#undef STRATA_MMVQ_MULTI

// launch_il's instances for Q6_K: one column (the plain q8_1 blocks) and two to four interleaved, at each rows count the
// table names (1, 2 or 4 rows a warp).
#define STRATA_MMVQ_IL(NAME, NC, R)                                                                                     \
    extern "C" __global__ void __launch_bounds__(4 * strata_native::WARP) NAME(                                        \
            const strata_native::Q6KBlock * __restrict__ w, const int * __restrict__ xq,                              \
            const float * __restrict__ xd, float * __restrict__ y, int n_in, int n_out) {                             \
        strata_native::native_mmvq_il_kernel<strata_native::IlQ6K, NC, R>(w, xq, xd, y, n_in, n_out);                  \
    }
STRATA_MMVQ_IL(strata_native_mmvq_il_q6_k_c2_r1, 2, 1)
STRATA_MMVQ_IL(strata_native_mmvq_il_q6_k_c2_r2, 2, 2)
STRATA_MMVQ_IL(strata_native_mmvq_il_q6_k_c2_r4, 2, 4)
STRATA_MMVQ_IL(strata_native_mmvq_il_q6_k_c3_r1, 3, 1)
STRATA_MMVQ_IL(strata_native_mmvq_il_q6_k_c3_r2, 3, 2)
STRATA_MMVQ_IL(strata_native_mmvq_il_q6_k_c3_r4, 3, 4)
STRATA_MMVQ_IL(strata_native_mmvq_il_q6_k_c4_r1, 4, 1)
STRATA_MMVQ_IL(strata_native_mmvq_il_q6_k_c4_r2, 4, 2)
STRATA_MMVQ_IL(strata_native_mmvq_il_q6_k_c4_r4, 4, 4)
#undef STRATA_MMVQ_IL

extern "C" __global__ void strata_native_q8_1_interleave(const strata_native::Q81Block * __restrict__ y,
                                                         int * __restrict__ bm, int * __restrict__ pm,
                                                         float * __restrict__ dl, int nb, int n_total, int cp) {
    strata_native::native_q8_1_interleave_kernel(y, bm, pm, dl, nb, n_total, cp);
}

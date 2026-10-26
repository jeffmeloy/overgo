// strata_qsa.cu - Strata's QSA attention layer kernels, ported literally.
//
// Ported from Strata (https://github.com/Niko1221/Strata, commit 296b59d2), MIT License,
// Copyright (c) 2026 Niko1221 and the Strata contributors; the license text is licenses/Strata-MIT.txt.
// Sources: include/strata/kernels/f16_bits.hpp, include/strata/kernels/qsa.hpp, include/strata/kernels/kv_q8.hpp,
// src/kernels/cuda/qsa.cu, src/kernels/cuda/kv_q8.cu, src/kernels/cuda/qsa_select.cu,
// src/kernels/cuda/qsa_decode_attn.cu, src/kernels/cuda/qsa_prompt_attn.cu, src/kernels/cuda/kv_stream.cu, and
// src/prefill/kernels.cu (kv_append_kernel).
//
// This module compiles WITHOUT -use_fast_math, as Strata compiles these files: its parity checks are bitwise
// (codes, scales, gathered fp16), and fast math would turn the quantizer's division approximate. Each kernel
// keeps Strata's body; the ABI differs only where the driver API needs flat arguments: Strata's KvHostPools
// struct arrives as its pointers, and a template instantiation is an extern "C" entry of its own. The host launch
// geometry (grids, blocks) is Strata's, reproduced in internal/cuda/executor/launch_strata_qsa.go.
//
// Layouts (qsa.hpp): K/V pools `[page][kv_head][page_size][head_dim]`, head_dim fastest, behind a page table
// (logical page -> physical page, -1 for a page not resident); int8 codes in that layout with one fp16 scale a
// 64-value group, scales `[page][kv_head][page_size][head_dim/64]`. A host copy, when present, holds the identity
// layout. Every per-token count is read from the device step state, so every argument list is constant across
// tokens and a launch sequence captures.

#include <cuda_fp16.h>
#include <float.h>
#include <stdint.h>
#include <string.h>

// ---- f16_bits.hpp: round-to-nearest-even f32 -> fp16 as bits, and the exact reverse. Not __float2half, for the
// reason f16_bits.hpp records (wrong bits, not a rounding difference).

__device__ __forceinline__ uint16_t strata_f16_from_f32(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t rawexp = (x >> 23) & 0xFFu;
    int exp = (int) rawexp - 127 + 15;
    uint32_t man = x & 0x7FFFFFu;
    if (rawexp == 0xFFu) return (uint16_t) (sign | 0x7C00u | (man ? 0x200u : 0u));
    if (exp >= 31) return (uint16_t) (sign | 0x7C00u);
    if (exp <= 0) {
        if (exp < -10) return (uint16_t) sign;
        man |= 0x800000u;
        const uint32_t sh = (uint32_t) (14 - exp);
        uint32_t h = (man >> sh) & 0x3FFu;
        const uint32_t rem = man & ((1u << sh) - 1u);
        if (rem > (1u << (sh - 1)) || (rem == (1u << (sh - 1)) && (h & 1u))) ++h;
        return (uint16_t) (sign | h);
    }
    uint16_t h = (uint16_t) (sign | ((uint32_t) exp << 10) | (man >> 13));
    const uint32_t rem = man & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) ++h;
    return h;
}

__device__ __forceinline__ float strata_f32_from_f16(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
    const uint32_t ex = (h >> 10) & 0x1Fu, man = h & 0x3FFu;
    uint32_t out;
    if (ex == 0) {
        if (man == 0) {
            out = sign;
        } else {
            uint32_t m = man;
            int s = 0;
            while ((m & 0x400u) == 0) { m <<= 1; ++s; }
            out = sign | ((uint32_t) (113 - s) << 23) | ((m & 0x3FFu) << 13);
        }
    } else if (ex == 31) {
        out = sign | 0x7F800000u | (man << 13);
    } else {
        out = sign | ((ex - 15 + 127) << 23) | (man << 13);
    }
    float f;
    memcpy(&f, &out, 4);
    return f;
}

// ---- qsa.hpp's step state: (kStepCount,) int32 in device memory, written once a token from the position:
// pos, n_kv = pos + 1, n_bid = n_kv / idx_block, width = min(n_kv, idx_top_k + idx_block - 1).
#define STRATA_STEP_POS 0
#define STRATA_STEP_NKV 1
#define STRATA_STEP_NBID 2
#define STRATA_STEP_WIDTH 3

// kv_q8.hpp: one fp16 scale a 64-value group of a cell's head. Provisional for the load-costs step: Strata's
// constant, tuned on its author's cards.
#define STRATA_KV_Q8_GROUP 64

// ---- qsa.cu 1. kv_append: one token's K and V into the paged FP16 pools at step[pos]; the host copy always, the
// VRAM page only if its block is resident. Launch: grid ceil(kv_heads * head_dim / 128), 128 threads.
extern "C" __global__ void strata_kv_append_f16(
        uint16_t * __restrict__ k_pool, uint16_t * __restrict__ v_pool,
        const int32_t * __restrict__ table, const int32_t * __restrict__ step,
        const float * __restrict__ kcur, const float * __restrict__ vcur,
        int kv_heads, int head_dim, int page_size,
        uint16_t * host_k_pool, uint16_t * host_v_pool) {
    const long long pos = (long long) __ldg(step + STRATA_STEP_POS);
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= kv_heads * head_dim) return;
    const int h = i / head_dim, d = i - h * head_dim;
    const long long page = (long long) table[pos / page_size];
    if (page >= 0) {
        const long long row = (page * kv_heads + h) * page_size + (pos % page_size);
        k_pool[row * head_dim + d] = strata_f16_from_f32(kcur[i]);
        v_pool[row * head_dim + d] = strata_f16_from_f32(vcur[i]);
    }
    if (host_k_pool != nullptr) {
        const long long row = ((pos / page_size) * kv_heads + h) * page_size + (pos % page_size);
        host_k_pool[row * head_dim + d] = strata_f16_from_f32(kcur[i]);
        host_v_pool[row * head_dim + d] = strata_f16_from_f32(vcur[i]);
    }
}

// ---- qsa.cu 5. kv_gather: step[width] selected cells through the page table into a contiguous fp16 scratch
// `[id][kv_head][head_dim]`, four halfs a thread. Launch: grid ceil(max_ids * kv_heads * head_dim / 4 / 256), 256.
extern "C" __global__ void strata_kv_gather_f16(
        const uint16_t * __restrict__ k_pool, const uint16_t * __restrict__ v_pool,
        const int32_t * __restrict__ table, const int32_t * __restrict__ ids,
        const int32_t * __restrict__ step, int kv_heads, int head_dim, int page_size,
        uint16_t * __restrict__ k_scratch, uint16_t * __restrict__ v_scratch) {
    const long long n_ids = (long long) __ldg(step + STRATA_STEP_WIDTH);
    const int per = head_dim / 4;
    const long long total = n_ids * kv_heads * per;
    const long long i = blockIdx.x * (long long) blockDim.x + threadIdx.x;
    if (i >= total) return;
    const long long id = i / (kv_heads * (long long) per);
    const int rem = (int) (i % (kv_heads * (long long) per));
    const int h = rem / per, q = rem - h * per;
    const int cell = ids[id];
    const long long page = (long long) table[cell / page_size];
    const long long src = ((page * kv_heads + h) * page_size + (cell % page_size)) * (long long) per + q;
    const long long dst = (id * kv_heads + h) * (long long) per + q;
    reinterpret_cast<uint2 *>(k_scratch)[dst] = reinterpret_cast<const uint2 *>(k_pool)[src];
    reinterpret_cast<uint2 *>(v_scratch)[dst] = reinterpret_cast<const uint2 *>(v_pool)[src];
}

// ---- kv_q8.cu kv_append_q8_kernel: one block = one 64-value group of one KV head of K (z = 0) or V (1), 64
// threads, one value each; scale = fp16(max|x| / 127), codes against the STORED scale, clamped to +-127. MULTI:
// z = 2 * token + (K / V), token j at step + j * step_stride and its rows at kcur / vcur + j * cur_stride.
// Launch: grid (kv_heads, head_dim / 64, 2) or (kv_heads, head_dim / 64, 2 * tokens), 64 threads.
template <bool MULTI>
__device__ __forceinline__ void strata_kv_append_q8_body(
        int8_t * __restrict__ k_q, int8_t * __restrict__ v_q,
        uint16_t * __restrict__ k_scale, uint16_t * __restrict__ v_scale,
        const int32_t * __restrict__ table, const int32_t * __restrict__ step,
        const float * __restrict__ kcur, const float * __restrict__ vcur,
        int kv_heads, int head_dim, int page_size,
        int8_t * host_k_q, int8_t * host_v_q, uint16_t * host_k_scale, uint16_t * host_v_scale,
        int step_stride, int cur_stride) {
    if (MULTI) {
        const int j = blockIdx.z >> 1;
        step += (size_t) j * step_stride; kcur += (size_t) j * cur_stride; vcur += (size_t) j * cur_stride;
    }
    const long long pos = (long long) __ldg(step + STRATA_STEP_POS);
    const int h = blockIdx.x, g = blockIdx.y, t = threadIdx.x;
    const bool is_v = MULTI ? (blockIdx.z & 1) == 1 : blockIdx.z == 1;
    const int groups = head_dim / STRATA_KV_Q8_GROUP;
    const float x = (is_v ? vcur : kcur)[h * head_dim + g * STRATA_KV_Q8_GROUP + t];
    float a = fabsf(x);
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    __shared__ float warp_max[2];
    if ((t & 31) == 0) warp_max[t >> 5] = a;
    __syncthreads();
    const float amax = fmaxf(warp_max[0], warp_max[1]);
    const uint16_t sbits = strata_f16_from_f32(amax / 127.0f);
    const float sf = strata_f32_from_f16(sbits);
    int q = 0;
    if (sf > 0.0f) {
        q = __float2int_rn(x / sf);
        q = q < -127 ? -127 : (q > 127 ? 127 : q);
    }
    const long long page = (long long) table[pos / page_size];
    if (page >= 0) {
        const long long row = (page * kv_heads + h) * page_size + (pos % page_size);
        (is_v ? v_q : k_q)[row * head_dim + g * STRATA_KV_Q8_GROUP + t] = (int8_t) q;
        if (t == 0) (is_v ? v_scale : k_scale)[row * groups + g] = sbits;
    }
    if (host_k_q != nullptr) {
        const long long row = ((pos / page_size) * kv_heads + h) * page_size + (pos % page_size);
        (is_v ? host_v_q : host_k_q)[row * head_dim + g * STRATA_KV_Q8_GROUP + t] = (int8_t) q;
        if (t == 0) (is_v ? host_v_scale : host_k_scale)[row * groups + g] = sbits;
    }
}

extern "C" __global__ void strata_kv_append_q8(
        int8_t * __restrict__ k_q, int8_t * __restrict__ v_q,
        uint16_t * __restrict__ k_scale, uint16_t * __restrict__ v_scale,
        const int32_t * __restrict__ table, const int32_t * __restrict__ step,
        const float * __restrict__ kcur, const float * __restrict__ vcur,
        int kv_heads, int head_dim, int page_size,
        int8_t * host_k_q, int8_t * host_v_q, uint16_t * host_k_scale, uint16_t * host_v_scale) {
    strata_kv_append_q8_body<false>(k_q, v_q, k_scale, v_scale, table, step, kcur, vcur, kv_heads, head_dim,
                                    page_size, host_k_q, host_v_q, host_k_scale, host_v_scale, 0, 0);
}

extern "C" __global__ void strata_kv_append_q8_steps(
        int8_t * __restrict__ k_q, int8_t * __restrict__ v_q,
        uint16_t * __restrict__ k_scale, uint16_t * __restrict__ v_scale,
        const int32_t * __restrict__ table, const int32_t * __restrict__ step,
        const float * __restrict__ kcur, const float * __restrict__ vcur,
        int kv_heads, int head_dim, int page_size,
        int8_t * host_k_q, int8_t * host_v_q, uint16_t * host_k_scale, uint16_t * host_v_scale,
        int step_stride, int cur_stride) {
    strata_kv_append_q8_body<true>(k_q, v_q, k_scale, v_scale, table, step, kcur, vcur, kv_heads, head_dim,
                                   page_size, host_k_q, host_v_q, host_k_scale, host_v_scale, step_stride,
                                   cur_stride);
}

// ---- kv_q8.cu kv_gather_q8_kernel: step[width] selected cells dequantized into the fp16 scratch the attention
// reads, four values a thread. Launch: grid ceil(max_ids * kv_heads * head_dim / 4 / 256), 256 threads.
extern "C" __global__ void strata_kv_gather_q8(
        const int8_t * __restrict__ k_q, const int8_t * __restrict__ v_q,
        const uint16_t * __restrict__ k_scale, const uint16_t * __restrict__ v_scale,
        const int32_t * __restrict__ table, const int32_t * __restrict__ ids,
        const int32_t * __restrict__ step, int kv_heads, int head_dim, int page_size,
        uint16_t * __restrict__ k_scratch, uint16_t * __restrict__ v_scratch) {
    const long long n_ids = (long long) __ldg(step + STRATA_STEP_WIDTH);
    const int per = head_dim / 4;
    const long long total = n_ids * kv_heads * per;
    const long long i = blockIdx.x * (long long) blockDim.x + threadIdx.x;
    if (i >= total) return;
    const long long id = i / (kv_heads * (long long) per);
    const int rem = (int) (i % (kv_heads * (long long) per));
    const int h = rem / per, q4 = rem - h * per;
    const int cell = ids[id];
    const long long page = (long long) table[cell / page_size];
    const long long row = (page * kv_heads + h) * page_size + (cell % page_size);
    const int d = q4 * 4;
    const int groups = head_dim / STRATA_KV_Q8_GROUP;
    const float ks = strata_f32_from_f16(k_scale[row * groups + d / STRATA_KV_Q8_GROUP]);
    const float vs = strata_f32_from_f16(v_scale[row * groups + d / STRATA_KV_Q8_GROUP]);
    const char4 kc = reinterpret_cast<const char4 *>(k_q + row * head_dim)[q4];
    const char4 vc = reinterpret_cast<const char4 *>(v_q + row * head_dim)[q4];
    ushort4 ko, vo;
    ko.x = strata_f16_from_f32((float) kc.x * ks); ko.y = strata_f16_from_f32((float) kc.y * ks);
    ko.z = strata_f16_from_f32((float) kc.z * ks); ko.w = strata_f16_from_f32((float) kc.w * ks);
    vo.x = strata_f16_from_f32((float) vc.x * vs); vo.y = strata_f16_from_f32((float) vc.y * vs);
    vo.z = strata_f16_from_f32((float) vc.z * vs); vo.w = strata_f16_from_f32((float) vc.w * vs);
    const long long dst = (id * kv_heads + h) * (long long) per + q4;
    reinterpret_cast<ushort4 *>(k_scratch)[dst] = ko;
    reinterpret_cast<ushort4 *>(v_scratch)[dst] = vo;
}

// ---- rope.hpp rope_neox_pair: `i` pairs with `i + n_rot/2`; one place for the pairing and the sign.
__device__ __forceinline__ void strata_rope_neox_pair(float a, float b, float c, float s, float & oa, float & ob) {
    oa = a * c - b * s;
    ob = a * s + b * c;
}

// ---- mrope.hpp mrope_pos: the table row of position `pos` for pair `pair`, from a multi-axis table when set.
__device__ __forceinline__ int strata_mrope_pos(const int32_t * tab, int pos, int pair) {
    return tab ? __ldg(tab + (size_t) pos * 3 + pair % 3) : pos;
}

// ---- qsa.cu 2. indexer_key_append: the raw indexer key into the tail, and on a block completion the block's
// pooled key, rms_norm(mean(raw)) * w, rotated IN PLACE at the block's FIRST cell; the spare slot is
// rms_norm(raw[0]) and moves to b + 1. One thread an indexer dim; the pooling arithmetic in double with
// __dadd_rn/__dmul_rn/__ddiv_rn/__dsqrt_rn in a fixed order, as qsa.cu's decision 4 records. Launch: 1 block,
// idx_dim threads, idx_dim doubles of shared memory.
extern "C" __global__ void strata_indexer_key_append(
        const float * __restrict__ raw, const int32_t * __restrict__ pos_dev,
        int pos_base, const float * __restrict__ w_k_norm, float eps,
        float * __restrict__ tail, float * __restrict__ dead,
        float * __restrict__ pooled, int32_t * __restrict__ block_pos,
        int idx_dim, int r, int n_rot, const float * __restrict__ cos_tab,
        const float * __restrict__ sin_tab, const int32_t * __restrict__ mtab) {
    extern __shared__ double s_mean[];
    const int d = threadIdx.x;
    const int pos = (int) __ldg(pos_dev);
    const int slot = pos % r;
    if (slot < r - 1) tail[(size_t) slot * idx_dim + d] = raw[d];
    if (pos == 0) {
        double p = (double) raw[d];
        double ss = 0.0;
        for (int i = 0; i < idx_dim; ++i) {
            const double v = (double) raw[i];
            ss = __dadd_rn(ss, __dmul_rn(v, v));
        }
        const double inv = __ddiv_rn(1.0, __dsqrt_rn(ss / (double) idx_dim + (double) eps));
        float y = (float) (p * inv * (double) w_k_norm[d]);
        if (d < n_rot) y *= cos_tab[d % (n_rot / 2)];
        dead[d] = y;
        pooled[d] = y;
    }
    if (slot != r - 1) return;
    __syncthreads();
    double m = 0.0;
    for (int j = 0; j < r - 1; ++j) m = __dadd_rn(m, (double) tail[(size_t) j * idx_dim + d]);
    m = __dadd_rn(m, (double) raw[d]);
    m = m / (double) r;
    s_mean[d] = m;
    __syncthreads();
    double ss = 0.0;
    for (int i = 0; i < idx_dim; ++i) ss = __dadd_rn(ss, __dmul_rn(s_mean[i], s_mean[i]));
    const double inv = __ddiv_rn(1.0, __dsqrt_rn(ss / (double) idx_dim + (double) eps));
    const int b = pos / r;
    pooled[(size_t) b * idx_dim + d] = (float) (m * inv * (double) w_k_norm[d]);
    pooled[(size_t) (b + 1) * idx_dim + d] = dead[d];
    if (d == 0) *block_pos = (int32_t) (pos_base + b * r);
    __syncthreads();
    const int half = n_rot / 2;
    if (d < half) {
        float * row = pooled + (size_t) b * idx_dim;
        const size_t toff = (size_t) strata_mrope_pos(mtab, pos_base + b * r, d) * half;
        strata_rope_neox_pair(row[d], row[half + d], cos_tab[toff + d], sin_tab[toff + d], row[d], row[half + d]);
    }
}

// ---- qsa.cu qsa_gate_apply_f32: attn * sigmoid(gate), the gate the SECOND half of each head's 2 * head_dim block
// of q_full, in double, without the fp16 round (the form a K-quant output projection takes). Launch: grid
// ceil(n_head * head_dim / 256), 256 threads.
extern "C" __global__ void strata_gate_apply_f32(
        const float * __restrict__ attn, const float * __restrict__ q_full,
        int n_head, int head_dim, float * __restrict__ out) {
    const long long i = blockIdx.x * (long long) blockDim.x + threadIdx.x;
    if (i >= (long long) n_head * head_dim) return;
    const int h = (int) (i / head_dim), d = (int) (i % head_dim);
    const double g = (double) q_full[((size_t) h * 2 * head_dim) + head_dim + d];
    const double sig = 1.0 / (1.0 + exp(-g));
    out[i] = (float) ((double) attn[i] * sig);
}

// ================= qsa_select.cu: the indexer's block scores and the block top-k =================
//
// Compiled for qsa_select's geometry (core/geometry.hpp): a 128-wide indexer key, 4 indexer heads, 4-cell blocks.
// qsa_block_scores: one warp a (query, block), FP32; relu per indexer head, summed; block n_bid (the incomplete
// tail) scores the `dead` key and gets +1e9 when it has cells. qsa_block_topk: one block a query; a 4-pass radix
// select over the query's n_bid + 1 blocks, each weighted by its cell count, then the cells emitted ascending with
// ties to the lowest index. Each query carries its own step record (pos, n_kv, n_bid, width).
#define STRATA_IDX_DIM 128
#define STRATA_IDX_HEADS 4
#define STRATA_IDX_BLOCK 4
#define STRATA_SCORE_WARPS 8
#define STRATA_TOPK_T 256
#define STRATA_STEP_COUNT 4

__device__ __forceinline__ uint32_t strata_order_key(float s) {
    const float v = s + 0.0f;
    if (!(v == v)) return 0u;
    const uint32_t b = __float_as_uint(v);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

// block_scores_kernel. Launch: grid (ceil(reach / 8), nq), 256 threads.
extern "C" __global__ void __launch_bounds__(STRATA_SCORE_WARPS * 32) strata_block_scores(
        const float * __restrict__ pooled, const float * __restrict__ dead, const float * __restrict__ q_idx,
        const int32_t * __restrict__ steps, int64_t max_blocks, float * __restrict__ out) {
    const int64_t qi = blockIdx.y;
    const int32_t * st = steps + qi * STRATA_STEP_COUNT;
    const int64_t n_kv = st[STRATA_STEP_NKV], n_bid = st[STRATA_STEP_NBID];
    const int64_t b = (int64_t) blockIdx.x * STRATA_SCORE_WARPS + (threadIdx.x >> 5);
    if (b > n_bid || b >= max_blocks) return;
    const int lane = threadIdx.x & 31;
    const float * key = (b == n_bid) ? dead : pooled + b * STRATA_IDX_DIM;
    const float4 k4 = *reinterpret_cast<const float4 *>(key + lane * 4);
    const float * q = q_idx + qi * STRATA_IDX_HEADS * STRATA_IDX_DIM + lane * 4;
    float score = 0.0f;
#pragma unroll
    for (int h = 0; h < STRATA_IDX_HEADS; ++h) {
        const float4 q4 = *reinterpret_cast<const float4 *>(q + h * STRATA_IDX_DIM);
        float d = k4.x * q4.x + k4.y * q4.y + k4.z * q4.z + k4.w * q4.w;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
        score += d > 0.0f ? d : 0.0f;
    }
    if (lane == 0) {
        if (b == n_bid && n_kv % STRATA_IDX_BLOCK != 0) score += 1e9f;
        out[qi * max_blocks + b] = score;
    }
}

// block_scores_multi_kernel: every key block read ONCE for all of a call's queries (at most 8), a fixed grid
// striding over the blocks, per (block, query) block_scores_kernel's arithmetic in its order. early_exit (Strata's
// bool, an int here): a CTA past the batch's largest n_bid returns before staging the queries. Launch: 256 blocks,
// 256 threads.
#define STRATA_SCORES_MQ 8
extern "C" __global__ void __launch_bounds__(STRATA_SCORE_WARPS * 32) strata_block_scores_multi(
        const float * __restrict__ pooled, const float * __restrict__ dead, const float * __restrict__ q_idx,
        const int32_t * __restrict__ steps, int nq, int64_t max_blocks, float * __restrict__ out, int early_exit) {
    __shared__ __align__(16) float qs[STRATA_SCORES_MQ * STRATA_IDX_HEADS * STRATA_IDX_DIM];
    __shared__ int64_t s_nkv[STRATA_SCORES_MQ], s_nbid[STRATA_SCORES_MQ];
    if (!early_exit)
        for (int i = threadIdx.x; i < nq * STRATA_IDX_HEADS * STRATA_IDX_DIM; i += blockDim.x) qs[i] = q_idx[i];
    if (threadIdx.x < nq) {
        s_nkv[threadIdx.x] = steps[threadIdx.x * STRATA_STEP_COUNT + STRATA_STEP_NKV];
        s_nbid[threadIdx.x] = steps[threadIdx.x * STRATA_STEP_COUNT + STRATA_STEP_NBID];
    }
    __syncthreads();
    int64_t top = 0;
    for (int q = 0; q < nq; ++q) top = s_nbid[q] > top ? s_nbid[q] : top;
    if (early_exit) {
        if ((int64_t) blockIdx.x * STRATA_SCORE_WARPS > top) return;
        for (int i = threadIdx.x; i < nq * STRATA_IDX_HEADS * STRATA_IDX_DIM; i += blockDim.x) qs[i] = q_idx[i];
        __syncthreads();
    }
    const int lane = threadIdx.x & 31;
    const int64_t wstride = (int64_t) gridDim.x * STRATA_SCORE_WARPS;
    for (int64_t b = (int64_t) blockIdx.x * STRATA_SCORE_WARPS + (threadIdx.x >> 5); b <= top && b < max_blocks; b += wstride) {
        const float4 kp = *reinterpret_cast<const float4 *>(pooled + b * STRATA_IDX_DIM + lane * 4);
        const float4 kd = *reinterpret_cast<const float4 *>(dead + lane * 4);
        for (int qi = 0; qi < nq; ++qi) {
            const int64_t n_bid = s_nbid[qi];
            if (b > n_bid) continue;
            const float4 k4 = (b == n_bid) ? kd : kp;
            const float * q = qs + qi * STRATA_IDX_HEADS * STRATA_IDX_DIM + lane * 4;
            float score = 0.0f;
#pragma unroll
            for (int h = 0; h < STRATA_IDX_HEADS; ++h) {
                const float4 q4 = *reinterpret_cast<const float4 *>(q + h * STRATA_IDX_DIM);
                float d = k4.x * q4.x + k4.y * q4.y + k4.z * q4.z + k4.w * q4.w;
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
                score += d > 0.0f ? d : 0.0f;
            }
            if (lane == 0) {
                if (b == n_bid && s_nkv[qi] % STRATA_IDX_BLOCK != 0) score += 1e9f;
                out[qi * max_blocks + b] = score;
            }
        }
    }
}

// The prompt path's block scores on tensor cores: rows (query, indexer head) x columns (blocks) as one GEMM, K = 128,
// relu per head summed. 3xTF32 (each operand split into a TF32 hi and lo part, hi*hi + hi*lo + lo*hi) keeps FP32-level
// accuracy in another summation order than the warp kernel (not bitwise). Blocks < n_bid only; the tail block n_bid is
// the warp kernel's arithmetic (strata_block_scores_tail). Only the sm_80 path: the module targets compute_89.
#define STRATA_TC_QT 16
#define STRATA_TC_NB 32
#define STRATA_TC_ITER 4
#define STRATA_TC_QS (STRATA_IDX_HEADS * STRATA_IDX_DIM + 4)
#define STRATA_TC_KS (STRATA_IDX_DIM + 4)

__device__ __forceinline__ uint32_t strata_tf32_hi(float x) {
    uint32_t r;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x));
    return r;
}
__device__ __forceinline__ void strata_mma_tf32(float * c, const uint32_t * a, const uint32_t * b) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// block_scores_tc_kernel. Launch: grid (ceil(reach / 128), ceil(nq / 16)), 128 threads,
// (16 * 516 + 32 * 132) floats of dynamic shared memory (49,920 bytes, past the 48 KB default).
extern "C" __global__ void __launch_bounds__(128) strata_block_scores_tc(
        const float * __restrict__ pooled, const float * __restrict__ q_idx, const int32_t * __restrict__ steps,
        int64_t nq, int64_t max_blocks, int64_t reach, float * __restrict__ out) {
    extern __shared__ __align__(16) float sm[];
    float * sQ = sm;
    float * sK = sm + STRATA_TC_QT * STRATA_TC_QS;
    __shared__ int s_nbid[STRATA_TC_QT];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5, gid = lane >> 2, tig = lane & 3;
    const int64_t q0 = (int64_t) blockIdx.y * STRATA_TC_QT;
    for (int i = t; i < STRATA_TC_QT * STRATA_IDX_HEADS * STRATA_IDX_DIM / 4; i += 128) {
        const int r = i / (STRATA_IDX_HEADS * STRATA_IDX_DIM / 4), c = i % (STRATA_IDX_HEADS * STRATA_IDX_DIM / 4);
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (q0 + r < nq) v = reinterpret_cast<const float4 *>(q_idx + (q0 + r) * STRATA_IDX_HEADS * STRATA_IDX_DIM)[c];
        *reinterpret_cast<float4 *>(sQ + r * STRATA_TC_QS + c * 4) = v;
    }
    if (t < STRATA_TC_QT) s_nbid[t] = q0 + t < nq ? steps[(q0 + t) * STRATA_STEP_COUNT + STRATA_STEP_NBID] : 0;
    int lo_nbid = 0x7fffffff, hi_nbid = 0;
    __syncthreads();
    for (int i = 0; i < STRATA_TC_QT; ++i) {
        if (q0 + i >= nq) break;
        lo_nbid = min(lo_nbid, s_nbid[i]);
        hi_nbid = max(hi_nbid, s_nbid[i]);
    }
    for (int it = 0; it < STRATA_TC_ITER; ++it) {
        const int64_t b0 = ((int64_t) blockIdx.x * STRATA_TC_ITER + it) * STRATA_TC_NB;
        if (b0 >= reach || b0 >= hi_nbid) break;
        __syncthreads();
        for (int i = t; i < STRATA_TC_NB * STRATA_IDX_DIM / 4; i += 128) {
            const int r = i / (STRATA_IDX_DIM / 4), c = i % (STRATA_IDX_DIM / 4);
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (b0 + r < hi_nbid) v = reinterpret_cast<const float4 *>(pooled + (b0 + r) * STRATA_IDX_DIM)[c];
            *reinterpret_cast<float4 *>(sK + r * STRATA_TC_KS + c * 4) = v;
        }
        __syncthreads();
        float acc[STRATA_IDX_HEADS][4];
#pragma unroll
        for (int h = 0; h < STRATA_IDX_HEADS; ++h) acc[h][0] = acc[h][1] = acc[h][2] = acc[h][3] = 0.f;
        const float * kr = sK + (warp * 8 + gid) * STRATA_TC_KS;
#pragma unroll 4
        for (int k0 = 0; k0 < STRATA_IDX_DIM; k0 += 8) {
            const float kx0 = kr[k0 + tig], kx1 = kr[k0 + tig + 4];
            uint32_t bh[2], bl[2];
            bh[0] = strata_tf32_hi(kx0);
            bh[1] = strata_tf32_hi(kx1);
            bl[0] = strata_tf32_hi(kx0 - __uint_as_float(bh[0]));
            bl[1] = strata_tf32_hi(kx1 - __uint_as_float(bh[1]));
#pragma unroll
            for (int h = 0; h < STRATA_IDX_HEADS; ++h) {
                const float * qa = sQ + h * STRATA_IDX_DIM + k0 + tig;
                const float x0 = qa[gid * STRATA_TC_QS], x1 = qa[(gid + 8) * STRATA_TC_QS];
                const float x2 = qa[gid * STRATA_TC_QS + 4], x3 = qa[(gid + 8) * STRATA_TC_QS + 4];
                uint32_t ah[4], al[4];
                ah[0] = strata_tf32_hi(x0); ah[1] = strata_tf32_hi(x1); ah[2] = strata_tf32_hi(x2); ah[3] = strata_tf32_hi(x3);
                al[0] = strata_tf32_hi(x0 - __uint_as_float(ah[0]));
                al[1] = strata_tf32_hi(x1 - __uint_as_float(ah[1]));
                al[2] = strata_tf32_hi(x2 - __uint_as_float(ah[2]));
                al[3] = strata_tf32_hi(x3 - __uint_as_float(ah[3]));
                strata_mma_tf32(acc[h], al, bh);
                strata_mma_tf32(acc[h], ah, bl);
                strata_mma_tf32(acc[h], ah, bh);
            }
        }
        const int64_t bc = b0 + warp * 8 + 2 * tig;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int qr = gid + half * 8;
            const int64_t qi = q0 + qr;
            if (qi >= nq) continue;
            const int nb = s_nbid[qr];
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int64_t b = bc + j;
                if (b >= nb || b >= max_blocks) continue;
                float score = 0.0f;
#pragma unroll
                for (int h = 0; h < STRATA_IDX_HEADS; ++h) {
                    const float d = acc[h][half * 2 + j];
                    score += d > 0.0f ? d : 0.0f;
                }
                out[qi * max_blocks + b] = score;
            }
        }
    }
}

// block_scores_tail_kernel: each query's tail block n_bid from the spare key, +1e9 when it holds cells, in the warp
// kernel's arithmetic. Launch: nq blocks, 32 threads.
extern "C" __global__ void __launch_bounds__(32) strata_block_scores_tail(
        const float * __restrict__ dead, const float * __restrict__ q_idx, const int32_t * __restrict__ steps,
        int64_t max_blocks, float * __restrict__ out) {
    const int64_t qi = blockIdx.x;
    const int32_t * st = steps + qi * STRATA_STEP_COUNT;
    const int64_t n_kv = st[STRATA_STEP_NKV], n_bid = st[STRATA_STEP_NBID];
    if (n_bid >= max_blocks) return;
    const int lane = threadIdx.x & 31;
    const float4 k4 = *reinterpret_cast<const float4 *>(dead + lane * 4);
    const float * q = q_idx + qi * STRATA_IDX_HEADS * STRATA_IDX_DIM + lane * 4;
    float score = 0.0f;
#pragma unroll
    for (int h = 0; h < STRATA_IDX_HEADS; ++h) {
        const float4 q4 = *reinterpret_cast<const float4 *>(q + h * STRATA_IDX_DIM);
        float d = k4.x * q4.x + k4.y * q4.y + k4.z * q4.z + k4.w * q4.w;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
        score += d > 0.0f ? d : 0.0f;
    }
    if (lane == 0) {
        if (n_kv % STRATA_IDX_BLOCK != 0) score += 1e9f;
        out[qi * max_blocks + n_bid] = score;
    }
}

// block_topk_kernel, the original (256 threads, one shared histogram, keys read from memory on every radix pass):
// the reference the dispatched kernels must equal. Launch: nq blocks, 256 threads.
extern "C" __global__ void __launch_bounds__(STRATA_TOPK_T) strata_block_topk_ref(
        const float * __restrict__ scores, const int32_t * __restrict__ steps, int64_t max_blocks,
        int64_t cap, int32_t * __restrict__ ids) {
    __shared__ int hist[256];
    __shared__ int s_a[STRATA_TOPK_T], s_b[STRATA_TOPK_T];
    __shared__ int s_digit, s_above;
    const int64_t qi = blockIdx.x;
    const int32_t * st = steps + qi * STRATA_STEP_COUNT;
    const int64_t n_kv = st[STRATA_STEP_NKV], n_bid = st[STRATA_STEP_NBID], width = st[STRATA_STEP_WIDTH];
    int32_t * out = ids + qi * cap;
    const int t = threadIdx.x;
    if (n_kv <= width) {
        for (int64_t j = t; j < n_kv; j += STRATA_TOPK_T) out[j] = (int32_t) j;
        return;
    }
    const float * sc = scores + qi * max_blocks;
    const int64_t nb = n_bid + 1;
    const int64_t per = (nb + STRATA_TOPK_T - 1) / STRATA_TOPK_T;
    const int64_t b0 = (int64_t) t * per, b1 = (b0 + per < nb) ? b0 + per : nb;
    auto weight = [&](int64_t b) -> int { return b < n_bid ? STRATA_IDX_BLOCK : (int) (n_kv - n_bid * STRATA_IDX_BLOCK); };
    uint32_t prefix = 0;
    int above = 0;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = t; i < 256; i += STRATA_TOPK_T) hist[i] = 0;
        __syncthreads();
        const uint32_t hi_mask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
        for (int64_t b = b0; b < b1; ++b) {
            const int w = weight(b);
            if (w == 0) continue;
            const uint32_t k = strata_order_key(sc[b]);
            if ((k & hi_mask) == (prefix & hi_mask)) atomicAdd(&hist[(k >> shift) & 255], w);
        }
        __syncthreads();
        if (t == 0) {
            int cum = above, d = 255;
            for (; d > 0; --d) {
                if (cum + hist[d] >= width) break;
                cum += hist[d];
            }
            s_digit = d;
            s_above = cum;
        }
        __syncthreads();
        prefix |= (uint32_t) s_digit << shift;
        above = s_above;
        __syncthreads();
    }
    const uint32_t thr = prefix;
    const int64_t eq_budget = width - above;
    int gt = 0, eq = 0;
    for (int64_t b = b0; b < b1; ++b) {
        const int w = weight(b);
        if (w == 0) continue;
        const uint32_t k = strata_order_key(sc[b]);
        if (k > thr) gt += w;
        else if (k == thr) eq += w;
    }
    s_a[t] = gt;
    s_b[t] = eq;
    __syncthreads();
    if (t == 0) {
        int ag = 0, ae = 0;
        for (int i = 0; i < STRATA_TOPK_T; ++i) {
            const int g = s_a[i], e = s_b[i];
            s_a[i] = ag; s_b[i] = ae;
            ag += g; ae += e;
        }
    }
    __syncthreads();
    const int64_t eq_before = s_b[t];
    int64_t my_eq = eq_budget - eq_before;
    if (my_eq < 0) my_eq = 0;
    if (my_eq > eq) my_eq = eq;
    const int sel = gt + (int) my_eq;
    __syncthreads();
    s_a[t] = sel;
    __syncthreads();
    if (t == 0) {
        int a = 0;
        for (int i = 0; i < STRATA_TOPK_T; ++i) { const int c = s_a[i]; s_a[i] = a; a += c; }
    }
    __syncthreads();
    int64_t wpos = s_a[t];
    int64_t eq_left = my_eq;
    for (int64_t b = b0; b < b1; ++b) {
        const int w = weight(b);
        if (w == 0) continue;
        const uint32_t k = strata_order_key(sc[b]);
        if (k > thr) {
            for (int c = 0; c < w; ++c) out[wpos++] = (int32_t) (b * STRATA_IDX_BLOCK + c);
        } else if (k == thr) {
            for (int c = 0; c < w && eq_left > 0; ++c, --eq_left) out[wpos++] = (int32_t) (b * STRATA_IDX_BLOCK + c);
        }
    }
}

// The register top-k: 1,024 threads hold up to TK_PER consecutive blocks' keys in registers (contexts up to
// 4 * 1024 * 33 cells), per-warp histograms, block-wide scans; block_topk_kernel's rule, identical ids. TK_PER is
// Strata's CUDA value (NVIDIA's 64 registers a thread at 1,024 threads), provisional for the load-costs step.
#define STRATA_TK_T 1024
#define STRATA_TK_PER 33

__device__ __forceinline__ int strata_block_excl_scan(int v, int * s_warp, int & total) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int x = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o) x += y;
    }
    if (lane == 31) s_warp[warp] = x;
    __syncthreads();
    if (warp == 0) {
        int w = s_warp[lane];
        int z = w;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, z, o);
            if (lane >= o) z += y;
        }
        s_warp[lane] = z - w;
        if (lane == 31) s_warp[32] = z;
    }
    __syncthreads();
    const int r = s_warp[warp] + x - v;
    total = s_warp[32];
    __syncthreads();
    return r;
}

// block_topk_reg_kernel<TK_PER>. Launch: nq blocks, 1,024 threads.
extern "C" __global__ void __launch_bounds__(STRATA_TK_T) strata_block_topk_reg(
        const float * __restrict__ scores, const int32_t * __restrict__ steps, int64_t max_blocks,
        int64_t cap, int32_t * __restrict__ ids) {
    __shared__ int hist[STRATA_TK_T / 32][256];
    __shared__ int s_warp[33];
    __shared__ int s_digit, s_above;
    const int64_t qi = blockIdx.x;
    const int32_t * st = steps + qi * STRATA_STEP_COUNT;
    const int64_t n_kv = st[STRATA_STEP_NKV], n_bid = st[STRATA_STEP_NBID], width = st[STRATA_STEP_WIDTH];
    int32_t * out = ids + qi * cap;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    if (n_kv <= width) {
        for (int64_t j = t; j < n_kv; j += STRATA_TK_T) out[j] = (int32_t) j;
        return;
    }
    const float * sc = scores + qi * max_blocks;
    const int64_t nb = n_bid + 1;
    const int64_t per = (nb + STRATA_TK_T - 1) / STRATA_TK_T;
    const int64_t b0 = (int64_t) t * per, b1 = (b0 + per < nb) ? b0 + per : nb;
    uint32_t key[STRATA_TK_PER];
#pragma unroll
    for (int j = 0; j < STRATA_TK_PER; ++j) key[j] = (b0 + j < b1) ? strata_order_key(sc[b0 + j]) : 0u;
    auto weight = [&](int64_t b) -> int { return b < n_bid ? STRATA_IDX_BLOCK : (int) (n_kv - n_bid * STRATA_IDX_BLOCK); };
    uint32_t prefix = 0;
    int above = 0;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = lane; i < 256; i += 32) hist[warp][i] = 0;
        __syncwarp();
        const uint32_t hi_mask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
#pragma unroll
        for (int j = 0; j < STRATA_TK_PER; ++j) {
            const int64_t b = b0 + j;
            if (b >= b1) break;
            const int w = weight(b);
            if (w == 0) continue;
            if ((key[j] & hi_mask) == (prefix & hi_mask)) atomicAdd(&hist[warp][(key[j] >> shift) & 255], w);
        }
        __syncthreads();
        if (t < 256) {
            int s = 0;
            for (int w2 = 0; w2 < STRATA_TK_T / 32; ++w2) s += hist[w2][t];
            hist[0][t] = s;
        }
        __syncthreads();
        if (t == 0) {
            int cum = above, d = 255;
            for (; d > 0; --d) {
                if (cum + hist[0][d] >= width) break;
                cum += hist[0][d];
            }
            s_digit = d;
            s_above = cum;
        }
        __syncthreads();
        prefix |= (uint32_t) s_digit << shift;
        above = s_above;
        __syncthreads();
    }
    const uint32_t thr = prefix;
    const int64_t eq_budget = width - above;
    int gt = 0, eq = 0;
#pragma unroll
    for (int j = 0; j < STRATA_TK_PER; ++j) {
        const int64_t b = b0 + j;
        if (b >= b1) break;
        const int w = weight(b);
        if (w == 0) continue;
        if (key[j] > thr) gt += w;
        else if (key[j] == thr) eq += w;
    }
    int tot;
    const int eq_before = strata_block_excl_scan(eq, s_warp, tot);
    int64_t my_eq = eq_budget - eq_before;
    if (my_eq < 0) my_eq = 0;
    if (my_eq > eq) my_eq = eq;
    const int sel = gt + (int) my_eq;
    int64_t wpos = strata_block_excl_scan(sel, s_warp, tot);
    int64_t eq_left = my_eq;
#pragma unroll
    for (int j = 0; j < STRATA_TK_PER; ++j) {
        const int64_t b = b0 + j;
        if (b >= b1) break;
        const int w = weight(b);
        if (w == 0) continue;
        if (key[j] > thr) {
            for (int c = 0; c < w; ++c) out[wpos++] = (int32_t) (b * STRATA_IDX_BLOCK + c);
        } else if (key[j] == thr) {
            for (int c = 0; c < w && eq_left > 0; ++c, --eq_left) out[wpos++] = (int32_t) (b * STRATA_IDX_BLOCK + c);
        }
    }
}

// block_topk_wide_kernel: a query with more blocks than the register kernel holds; its threads, per-warp
// histograms and warp scans, each key read from memory again on every pass; identical ids. Launch: nq blocks,
// 1,024 threads.
extern "C" __global__ void __launch_bounds__(STRATA_TK_T) strata_block_topk_wide(
        const float * __restrict__ scores, const int32_t * __restrict__ steps, int64_t max_blocks,
        int64_t cap, int32_t * __restrict__ ids) {
    __shared__ int hist[STRATA_TK_T / 32][256];
    __shared__ int s_gt[STRATA_TK_T / 32], s_eq[STRATA_TK_T / 32];
    __shared__ int s_digit, s_above;
    const int64_t qi = blockIdx.x;
    const int32_t * st = steps + qi * STRATA_STEP_COUNT;
    const int64_t n_kv = st[STRATA_STEP_NKV], n_bid = st[STRATA_STEP_NBID], width = st[STRATA_STEP_WIDTH];
    int32_t * out = ids + qi * cap;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    if (n_kv <= width) {
        for (int64_t j = t; j < n_kv; j += STRATA_TK_T) out[j] = (int32_t) j;
        return;
    }
    const float * sc = scores + qi * max_blocks;
    const int64_t nb = n_bid + 1;
    const int64_t per = (nb + STRATA_TK_T - 1) / STRATA_TK_T;
    auto weight = [&](int64_t b) -> int { return b < n_bid ? STRATA_IDX_BLOCK : (int) (n_kv - n_bid * STRATA_IDX_BLOCK); };
    uint32_t prefix = 0;
    int above = 0;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = lane; i < 256; i += 32) hist[warp][i] = 0;
        __syncwarp();
        const uint32_t hi_mask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
        for (int64_t b = t; b < nb; b += STRATA_TK_T) {
            const int w = weight(b);
            if (w == 0) continue;
            const uint32_t k = strata_order_key(sc[b]);
            if ((k & hi_mask) == (prefix & hi_mask)) atomicAdd(&hist[warp][(k >> shift) & 255], w);
        }
        __syncthreads();
        if (t < 256) {
            int sum = 0;
            for (int w2 = 0; w2 < STRATA_TK_T / 32; ++w2) sum += hist[w2][t];
            hist[0][t] = sum;
        }
        __syncthreads();
        if (t == 0) {
            int cum = above, d = 255;
            for (; d > 0; --d) {
                if (cum + hist[0][d] >= width) break;
                cum += hist[0][d];
            }
            s_digit = d;
            s_above = cum;
        }
        __syncthreads();
        prefix |= (uint32_t) s_digit << shift;
        above = s_above;
        __syncthreads();
    }
    const uint32_t thr = prefix;
    const int eq_budget = (int) (width - above);
    const int64_t w0 = (int64_t) warp * 32 * per;
    int gt = 0, eq = 0;
    for (int64_t b = w0 + lane; b < nb && b < w0 + 32 * per; b += 32) {
        const int w = weight(b);
        if (w == 0) continue;
        const uint32_t k = strata_order_key(sc[b]);
        if (k > thr) gt += w;
        else if (k == thr) eq += w;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        gt += __shfl_xor_sync(0xffffffffu, gt, o);
        eq += __shfl_xor_sync(0xffffffffu, eq, o);
    }
    if (lane == 0) { s_gt[warp] = gt; s_eq[warp] = eq; }
    __syncthreads();
    if (t == 0) {
        int eb = 0, sb = 0;
        for (int w2 = 0; w2 < STRATA_TK_T / 32; ++w2) {
            const int g2 = s_gt[w2], e2 = s_eq[w2];
            const int take = eq_budget - eb < 0 ? 0 : (eq_budget - eb > e2 ? e2 : eq_budget - eb);
            s_gt[w2] = sb;
            s_eq[w2] = eb;
            sb += g2 + take;
            eb += e2;
        }
    }
    __syncthreads();
    int run_sel = s_gt[warp], run_eq = s_eq[warp];
    for (int64_t r0 = w0; r0 < nb && r0 < w0 + 32 * per; r0 += 32) {
        const int64_t b = r0 + lane;
        const int w = b < nb ? weight(b) : 0;
        const uint32_t k = w ? strata_order_key(sc[b]) : 0u;
        const int my_gt = (w && k > thr) ? w : 0, my_eq = (w && k == thr) ? w : 0;
        if (__ballot_sync(0xffffffffu, (my_gt | my_eq) != 0) == 0u) continue;
        int pe = my_eq;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, pe, o);
            if (lane >= o) pe += y;
        }
        const int left = eq_budget - (run_eq + pe - my_eq);
        const int take = left < 0 ? 0 : (left > my_eq ? my_eq : left);
        const int my_sel = my_gt + take;
        int ps = my_sel;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, ps, o);
            if (lane >= o) ps += y;
        }
        int32_t * dst = out + run_sel + ps - my_sel;
        for (int c = 0; c < my_sel; ++c) dst[c] = (int32_t) (b * STRATA_IDX_BLOCK + c);
        run_eq += __shfl_sync(0xffffffffu, pe, 31);
        run_sel += __shfl_sync(0xffffffffu, ps, 31);
    }
}

// ================= qsa_decode_attn.cu: split-K decode attention over the selected cells =================
//
// The pools read directly through the page table and the selection ids (no gather copy), in chunks of 64 cells a
// block, all 12 query heads that share a KV head served from one read of the chunk: grid (ceil(cap / 64),
// n_head_kv, n_q) scores q.k * scale for 12 heads x 64 cells, takes the chunk's max m and sum l, and accumulates
// p . V into partials; grid (n_head, n_q) merges the chunks with the usual log-sum-exp rescale. Compiled for
// core/geometry.hpp: a 256-wide head and 12 query heads a KV head. This is attn_chunk_kernel's default branch
// (warp scoring), the one the CUDA dispatch runs from sm_75 on, for fp16 pools (KV_MODE 0) and int8 (1).
#define STRATA_HD 256
#define STRATA_G 12
#define STRATA_CHUNK 64
#define STRATA_ATTN_THREADS 256
#define STRATA_ATTN_WARPS (STRATA_ATTN_THREADS / 32)

__device__ __forceinline__ float strata_warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float strata_warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// 8 consecutive values of one cell's key or value row, dimensions [d0, d0 + 8): load8_f16 and load8_q8.
template <int KV_MODE>
__device__ __forceinline__ void strata_load8(
        const uint16_t * k_pool, const uint16_t * v_pool, const int8_t * k_q, const int8_t * v_q,
        const uint16_t * k_scale, const uint16_t * v_scale, bool value, long long row, int d0, float * out) {
    if (KV_MODE == 0) {
        const uint16_t * base = (value ? v_pool : k_pool) + row * STRATA_HD + d0;
        const uint4 raw = *reinterpret_cast<const uint4 *>(base);
        const __half2 * h2 = reinterpret_cast<const __half2 *>(&raw);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 f = __half22float2(h2[j]);
            out[2 * j] = f.x;
            out[2 * j + 1] = f.y;
        }
    } else {
        const int8_t * codes = (value ? v_q : k_q) + row * STRATA_HD + d0;
        const uint16_t sbits = (value ? v_scale : k_scale)[row * (STRATA_HD / STRATA_KV_Q8_GROUP) + d0 / STRATA_KV_Q8_GROUP];
        const float sc = __half2float(__ushort_as_half(sbits));
        const uint2 raw = *reinterpret_cast<const uint2 *>(codes);
        const int8_t * c = reinterpret_cast<const int8_t *>(&raw);
#pragma unroll
        for (int j = 0; j < 8; ++j) out[j] = (float) c[j] * sc;
    }
}

// attn_chunk_kernel<KV_MODE> with Strata's QsaAttnPools arriving as its pointers. Single-query calls pass cap 0 and
// stride 0 (blockIdx.z is 0). Launch: grid (ceil(cap / 64), n_head_kv, n_q), 256 threads.
template <int KV_MODE>
__device__ __forceinline__ void strata_attn_chunk_body(
        const float * __restrict__ q, const uint16_t * k_pool, const uint16_t * v_pool, const int8_t * k_q,
        const int8_t * v_q, const uint16_t * k_scale, const uint16_t * v_scale, const int32_t * page_table,
        const int32_t * __restrict__ ids, const int32_t * __restrict__ step, int n_kv_heads, int page_size,
        float scale, float * __restrict__ part_acc, float * __restrict__ part_m, float * __restrict__ part_l,
        int n_chunks, int cap, long long scratch_stride) {
    q += (size_t) blockIdx.z * (size_t) (n_kv_heads * STRATA_G) * STRATA_HD;
    ids += (size_t) blockIdx.z * (size_t) cap;
    step += (size_t) blockIdx.z * STRATA_STEP_COUNT;
    part_acc += (size_t) blockIdx.z * (size_t) scratch_stride;
    part_m += (size_t) blockIdx.z * (size_t) scratch_stride;
    part_l += (size_t) blockIdx.z * (size_t) scratch_stride;
    __shared__ __align__(16) float sq[STRATA_G][STRATA_HD];
    __shared__ float sp[STRATA_G][STRATA_CHUNK];
    __shared__ long long srow[STRATA_CHUNK];
    const int n_ids = __ldg(step + STRATA_STEP_WIDTH);
    const int chunk = blockIdx.x, kvh = blockIdx.y;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int c0 = chunk * STRATA_CHUNK;
    const int n_here = min(STRATA_CHUNK, n_ids - c0);
    const int slot = kvh * n_chunks + chunk;
    if (n_here <= 0) {
        if (t < STRATA_G) { part_m[slot * STRATA_G + t] = -FLT_MAX; part_l[slot * STRATA_G + t] = 0.0f; }
        return;
    }
    for (int i = t; i < STRATA_G * STRATA_HD; i += STRATA_ATTN_THREADS) sq[i / STRATA_HD][i % STRATA_HD] = q[(size_t) (kvh * STRATA_G) * STRATA_HD + i];
    if (t < STRATA_CHUNK) {
        long long r = -1;
        if (t < n_here) {
            const int cell = ids[c0 + t];
            const long long page = (long long) page_table[cell / page_size];
            // a block the K/V streaming could not make resident keeps page -1: its cells are masked
            if (page >= 0) r = (page * n_kv_heads + kvh) * page_size + (cell % page_size);
        }
        srow[t] = r;
    }
    __syncthreads();
    for (int c = warp; c < STRATA_CHUNK; c += STRATA_ATTN_WARPS) {
        if (c >= n_here || srow[c] < 0) {
            if (lane < STRATA_G) sp[lane][c] = -FLT_MAX;
            continue;
        }
        float k8[8];
        strata_load8<KV_MODE>(k_pool, v_pool, k_q, v_q, k_scale, v_scale, false, srow[c], lane * 8, k8);
#pragma unroll
        for (int h = 0; h < STRATA_G; ++h) {
            const float4 qa = *reinterpret_cast<const float4 *>(&sq[h][lane * 8]);
            const float4 qb = *reinterpret_cast<const float4 *>(&sq[h][lane * 8 + 4]);
            float s = k8[0] * qa.x + k8[1] * qa.y + k8[2] * qa.z + k8[3] * qa.w +
                      k8[4] * qb.x + k8[5] * qb.y + k8[6] * qb.z + k8[7] * qb.w;
            s = strata_warp_sum(s);
            if (lane == 0) sp[h][c] = s * scale;
        }
    }
    __syncthreads();
    for (int h = warp; h < STRATA_G; h += STRATA_ATTN_WARPS) {
        const float a = sp[h][lane], b = sp[h][lane + 32];
        const float m = strata_warp_max(fmaxf(a, b));
        const float ea = (lane < n_here && srow[lane] >= 0) ? __expf(a - m) : 0.0f;
        const float eb = (lane + 32 < n_here && srow[lane + 32] >= 0) ? __expf(b - m) : 0.0f;
        sp[h][lane] = ea;
        sp[h][lane + 32] = eb;
        const float l = strata_warp_sum(ea + eb);
        if (lane == 0) { part_m[slot * STRATA_G + h] = m; part_l[slot * STRATA_G + h] = l; }
    }
    __syncthreads();
    float acc[STRATA_G];
#pragma unroll
    for (int h = 0; h < STRATA_G; ++h) acc[h] = 0.0f;
    for (int c = 0; c < n_here; ++c) {
        if (srow[c] < 0) continue;
        float v;
        if (KV_MODE == 0) {
            v = __half2float(__ushort_as_half(v_pool[srow[c] * STRATA_HD + t]));
        } else {
            const float sc = __half2float(__ushort_as_half(v_scale[srow[c] * (STRATA_HD / STRATA_KV_Q8_GROUP) + t / STRATA_KV_Q8_GROUP]));
            v = (float) v_q[srow[c] * STRATA_HD + t] * sc;
        }
#pragma unroll
        for (int h = 0; h < STRATA_G; ++h) acc[h] = fmaf(sp[h][c], v, acc[h]);
    }
#pragma unroll
    for (int h = 0; h < STRATA_G; ++h) part_acc[((size_t) slot * STRATA_G + h) * STRATA_HD + t] = acc[h];
}

extern "C" __global__ void __launch_bounds__(STRATA_ATTN_THREADS) strata_attn_chunk_f16(
        const float * __restrict__ q, const uint16_t * k_pool, const uint16_t * v_pool,
        const int32_t * page_table, const int32_t * __restrict__ ids, const int32_t * __restrict__ step,
        int n_kv_heads, int page_size, float scale, float * __restrict__ part_acc, float * __restrict__ part_m,
        float * __restrict__ part_l, int n_chunks, int cap, long long scratch_stride) {
    strata_attn_chunk_body<0>(q, k_pool, v_pool, nullptr, nullptr, nullptr, nullptr, page_table, ids, step, n_kv_heads,
                              page_size, scale, part_acc, part_m, part_l, n_chunks, cap, scratch_stride);
}

extern "C" __global__ void __launch_bounds__(STRATA_ATTN_THREADS) strata_attn_chunk_q8(
        const float * __restrict__ q, const int8_t * k_q, const int8_t * v_q, const uint16_t * k_scale,
        const uint16_t * v_scale, const int32_t * page_table, const int32_t * __restrict__ ids,
        const int32_t * __restrict__ step, int n_kv_heads, int page_size, float scale,
        float * __restrict__ part_acc, float * __restrict__ part_m, float * __restrict__ part_l, int n_chunks,
        int cap, long long scratch_stride) {
    strata_attn_chunk_body<1>(q, nullptr, nullptr, k_q, v_q, k_scale, v_scale, page_table, ids, step, n_kv_heads,
                              page_size, scale, part_acc, part_m, part_l, n_chunks, cap, scratch_stride);
}

// attn_merge_kernel: the chunks of each query head merged with the log-sum-exp rescale, one thread a dimension.
// Launch: grid (n_head, n_q), 256 threads.
extern "C" __global__ void __launch_bounds__(STRATA_HD) strata_attn_merge(
        const float * __restrict__ part_acc, const float * __restrict__ part_m, const float * __restrict__ part_l,
        int n_chunks, float * __restrict__ attn, long long scratch_stride) {
    part_acc += (size_t) blockIdx.y * (size_t) scratch_stride;
    part_m += (size_t) blockIdx.y * (size_t) scratch_stride;
    part_l += (size_t) blockIdx.y * (size_t) scratch_stride;
    attn += (size_t) blockIdx.y * (size_t) gridDim.x * STRATA_HD;
    const int h = blockIdx.x;
    const int kvh = h / STRATA_G, hl = h % STRATA_G;
    const int d = threadIdx.x;
    float M = -FLT_MAX;
    for (int c = 0; c < n_chunks; ++c) M = fmaxf(M, part_m[(kvh * n_chunks + c) * STRATA_G + hl]);
    float L = 0.0f, acc = 0.0f;
    for (int c = 0; c < n_chunks; ++c) {
        const int slot = kvh * n_chunks + c;
        const float m = part_m[slot * STRATA_G + hl];
        if (m == -FLT_MAX) continue;
        const float w = __expf(m - M);
        L = fmaf(part_l[slot * STRATA_G + hl], w, L);
        acc = fmaf(part_acc[((size_t) slot * STRATA_G + hl) * STRATA_HD + d], w, acc);
    }
    attn[(size_t) h * STRATA_HD + d] = L > 0.0f ? acc / L : 0.0f;
}

// ================= qsa_prompt_attn.cu: the prompt path's QSA attention on tensor cores =================
//
// The same per-query selection and pools as the decode kernel; one block a (query, KV head) walks all its selected
// cells in 32-cell chunks with an online softmax (no split-K scratch, no merge); q.k and p.v are m16n8k16 FP16 MMAs
// with FP32 accumulation over the 12 query heads of the KV head (+4 pad rows); int8 codes enter exactly in FP16
// and their per-64 scales in FP32; q and p are split into FP16 hi + lo parts (two MMAs each), keeping ~22 bits. Not
// bitwise the decode kernel; qsa_prompt_attn_parity bounds the difference. This module targets compute_89, so only
// the sm_80 paths are here (Strata's Turing, Volta and AMD forms are not).
#define STRATA_PA_THREADS 128

__device__ __forceinline__ void strata_mma16816(float * c, const uint32_t * a, const uint32_t * b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Two int8 codes (low byte first) as an exact half2: 1024 + (c + 128) built in the mantissa, minus 1152.
__device__ __forceinline__ uint32_t strata_i8x2_to_h2(uint32_t x) {
    uint32_t y = ((x & 0xffu) | ((x & 0xff00u) << 8)) ^ 0x00800080u;
    y |= 0x64006400u;
    __half2 h = *reinterpret_cast<__half2 *>(&y);
    h = __hsub2(h, __halves2half2(__float2half(1152.f), __float2half(1152.f)));
    return *reinterpret_cast<uint32_t *>(&h);
}

__device__ __forceinline__ uint32_t strata_pack_h2(float lo_k, float hi_k) {
    __half2 h = __floats2half2_rn(lo_k, hi_k);
    return *reinterpret_cast<uint32_t *>(&h);
}

// Smem2: the int8 kernel's shared memory, `sizeof(StrataSmem2)` bytes of it dynamic (45,584).
#define STRATA_CH2 32
struct StrataSmem2 {
    int8_t kv[2][4][2][STRATA_CH2][64];
    float sc[2][4][2][STRATA_CH2];
    float part[4][16][STRATA_CH2 + 1];
    float p[16][STRATA_CH2 + 1];
    float qmax[4];
    float alpha[16];
    float lsum[16];
    float mrow[16];
};

__device__ __forceinline__ int strata_swz(int cell, int byte) {
    return cell * 64 + ((((byte >> 4) ^ (cell >> 1)) & 3) << 4) + (byte & 15);
}
__device__ __forceinline__ void strata_cp_async16(void * smem, const void * gmem, bool valid) {
    const unsigned sa = (unsigned) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(sa), "l"(gmem), "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void strata_cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void strata_cp_async_wait1() { asm volatile("cp.async.wait_group 1;\n" ::); }

// prompt_attn_i8_kernel, Strata's QsaAttnPools arriving as its int8 pointers. Launch: grid (n_q (at most 65,535 a
// launch), n_head_kv), 128 threads, sizeof(StrataSmem2) bytes of dynamic shared memory; scale_log2 = log2(e) / 16.
extern "C" __global__ void __launch_bounds__(STRATA_PA_THREADS) strata_prompt_attn_i8(
        const float * __restrict__ q, const int8_t * k_q, const int8_t * v_q, const uint16_t * k_scale,
        const uint16_t * v_scale, const int32_t * page_table, const int32_t * __restrict__ ids,
        const int32_t * __restrict__ steps, int n_kv_heads, int page_size, float scale_log2,
        float * __restrict__ attn, int cap) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    StrataSmem2 & S = *reinterpret_cast<StrataSmem2 *>(smem_raw);
    const int qi = blockIdx.x, kvh = blockIdx.y;
    const int n_head = n_kv_heads * STRATA_G;
    q += (size_t) qi * n_head * STRATA_HD + (size_t) kvh * STRATA_G * STRATA_HD;
    attn += (size_t) qi * n_head * STRATA_HD + (size_t) kvh * STRATA_G * STRATA_HD;
    ids += (size_t) qi * cap;
    const int n = __ldg(steps + (size_t) qi * STRATA_STEP_COUNT + STRATA_STEP_WIDTH);
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int gid = lane >> 2, tig = lane & 3;
    const int dim0 = warp * 64;

    float qm = 0.0f;
    for (int i = t; i < STRATA_G * STRATA_HD; i += STRATA_PA_THREADS) qm = fmaxf(qm, fabsf(q[i]));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) qm = fmaxf(qm, __shfl_xor_sync(0xffffffffu, qm, o));
    if (lane == 0) S.qmax[warp] = qm;
    if (t < 16) { S.mrow[t] = -INFINITY; S.lsum[t] = 0.0f; }
    __syncthreads();
    qm = fmaxf(fmaxf(S.qmax[0], S.qmax[1]), fmaxf(S.qmax[2], S.qmax[3]));
    int qe = 0;
    if (qm > 0.0f) frexpf(qm, &qe);
    const float qup = ldexpf(1.0f, 14 - qe), qdown = ldexpf(scale_log2, qe - 14);
    uint32_t qh[4][4], ql[4][4];
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int row = gid + (r & 1) * 8, col = dim0 + kk * 16 + 2 * tig + (r >> 1) * 8;
            float2 x = make_float2(0.f, 0.f);
            if (row < STRATA_G) x = *reinterpret_cast<const float2 *>(q + (size_t) row * STRATA_HD + col);
            x.x *= qup;
            x.y *= qup;
            const __half2 hi = __floats2half2_rn(x.x, x.y);
            const float2 hf = __half22float2(hi);
            const __half2 lo = __floats2half2_rn(x.x - hf.x, x.y - hf.y);
            qh[kk][r] = *reinterpret_cast<const uint32_t *>(&hi);
            ql[kk][r] = *reinterpret_cast<const uint32_t *>(&lo);
        }
    }

    const int n_chunks = (n + STRATA_CH2 - 1) / STRATA_CH2;
    auto cell_of = [&](int c) -> int { return c < n ? __ldg(ids + c) : -1; };
    auto row_of = [&](int cell) -> long long {
        if (cell < 0) return -1;
        const long long page = (long long) __ldg(page_table + cell / page_size);
        return (page * n_kv_heads + kvh) * page_size + (cell % page_size);
    };
    auto issue = [&](long long r, int st, float & ksr, float & vsr) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int idx = lane + 32 * j, cell = idx >> 2, pc = idx & 3;
            const long long rr = __shfl_sync(0xffffffffu, r, cell);
            const bool ok = rr >= 0;
            const size_t off = ok ? (size_t) rr * STRATA_HD + dim0 + pc * 16 : 0;
            strata_cp_async16(&S.kv[st][warp][0][0][0] + strata_swz(cell, pc * 16), k_q + off, ok);
            strata_cp_async16(&S.kv[st][warp][1][0][0] + strata_swz(cell, pc * 16), v_q + off, ok);
        }
        ksr = r >= 0 ? __half2float(__ushort_as_half(__ldg(k_scale + r * (STRATA_HD / STRATA_KV_Q8_GROUP) + warp))) : 0.0f;
        vsr = r >= 0 ? __half2float(__ushort_as_half(__ldg(v_scale + r * (STRATA_HD / STRATA_KV_Q8_GROUP) + warp))) : 0.0f;
    };
    float ksn, vsn;
    issue(row_of(cell_of(lane)), 0, ksn, vsn);
    strata_cp_async_commit();
    S.sc[0][warp][0][lane] = ksn;
    S.sc[0][warp][1][lane] = vsn;
    long long r_next = row_of(cell_of(STRATA_CH2 + lane));
    int cell_next2 = cell_of(2 * STRATA_CH2 + lane);

    float acc[8][4];
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.0f;

    for (int ci = 0; ci < n_chunks; ++ci) {
        const int st = ci & 1, c0 = ci * STRATA_CH2;
        const bool more = ci + 1 < n_chunks;
        if (more) issue(r_next, st ^ 1, ksn, vsn);
        strata_cp_async_commit();
        r_next = row_of(cell_next2);
        cell_next2 = cell_of((ci + 3) * STRATA_CH2 + lane);
        strata_cp_async_wait1();
        __syncwarp();
        const int8_t * K = &S.kv[st][warp][0][0][0];
        const int8_t * V = &S.kv[st][warp][1][0][0];
#pragma unroll
        for (int nt = 0; nt < STRATA_CH2 / 8; ++nt) {
            float tg[4] = {0.f, 0.f, 0.f, 0.f};
            const int cell = nt * 8 + gid;
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                uint32_t b[2];
                b[0] = strata_i8x2_to_h2(*reinterpret_cast<const uint16_t *>(K + strata_swz(cell, kk * 16 + 2 * tig)));
                b[1] = strata_i8x2_to_h2(*reinterpret_cast<const uint16_t *>(K + strata_swz(cell, kk * 16 + 2 * tig + 8)));
                strata_mma16816(tg, qh[kk], b);
                strata_mma16816(tg, ql[kk], b);
            }
            const int c = nt * 8 + 2 * tig;
            const float s0 = S.sc[st][warp][0][c], s1 = S.sc[st][warp][0][c + 1];
            S.part[warp][gid][c] = tg[0] * s0;
            S.part[warp][gid][c + 1] = tg[1] * s1;
            S.part[warp][gid + 8][c] = tg[2] * s0;
            S.part[warp][gid + 8][c + 1] = tg[3] * s1;
        }
        __syncthreads();
        {
            const int r = t >> 3, sub = t & 7;
            float x[4], mx = -INFINITY;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int c = sub * 4 + j;
                x[j] = c0 + c < n ? (((S.part[0][r][c] + S.part[1][r][c]) + S.part[2][r][c]) + S.part[3][r][c]) * qdown
                                  : -INFINITY;
                mx = fmaxf(mx, x[j]);
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
            const float m_old = S.mrow[r];
            const float m_new = fmaxf(m_old, mx);
            float sum = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float e = x[j] == -INFINITY ? 0.0f : exp2f(x[j] - m_new);
                S.p[r][sub * 4 + j] = e;
                sum += e;
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
            __syncwarp();
            if (sub == 0) {
                const float a = m_old == -INFINITY ? 0.0f : exp2f(m_old - m_new);
                S.alpha[r] = a;
                S.lsum[r] = fmaf(S.lsum[r], a, sum);
                S.mrow[r] = m_new;
            }
        }
        __syncthreads();
        {
            float vmax = S.sc[st][warp][1][lane];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) vmax = fmaxf(vmax, __shfl_xor_sync(0xffffffffu, vmax, o));
            const float vup = vmax > 0.0f ? 16384.0f / vmax : 0.0f, vdown = vmax * (1.0f / 16384.0f);
            float tmp[8][4];
#pragma unroll
            for (int j = 0; j < 8; ++j) tmp[j][0] = tmp[j][1] = tmp[j][2] = tmp[j][3] = 0.0f;
#pragma unroll
            for (int ks = 0; ks < STRATA_CH2 / 16; ++ks) {
                const int cA = ks * 16 + 2 * tig, cB = cA + 8;
                const float w0 = S.sc[st][warp][1][cA] * vup, w1 = S.sc[st][warp][1][cA + 1] * vup,
                            w2 = S.sc[st][warp][1][cB] * vup, w3 = S.sc[st][warp][1][cB + 1] * vup;
                const float p00 = S.p[gid][cA] * w0, p01 = S.p[gid][cA + 1] * w1;
                const float p10 = S.p[gid + 8][cA] * w0, p11 = S.p[gid + 8][cA + 1] * w1;
                const float p02 = S.p[gid][cB] * w2, p03 = S.p[gid][cB + 1] * w3;
                const float p12 = S.p[gid + 8][cB] * w2, p13 = S.p[gid + 8][cB + 1] * w3;
                uint32_t ah[4], al[4];
                ah[0] = strata_pack_h2(p00, p01);
                ah[1] = strata_pack_h2(p10, p11);
                ah[2] = strata_pack_h2(p02, p03);
                ah[3] = strata_pack_h2(p12, p13);
                {
                    const __half2 * h = reinterpret_cast<const __half2 *>(ah);
                    float2 f;
                    f = __half22float2(h[0]); al[0] = strata_pack_h2(p00 - f.x, p01 - f.y);
                    f = __half22float2(h[1]); al[1] = strata_pack_h2(p10 - f.x, p11 - f.y);
                    f = __half22float2(h[2]); al[2] = strata_pack_h2(p02 - f.x, p03 - f.y);
                    f = __half22float2(h[3]); al[3] = strata_pack_h2(p12 - f.x, p13 - f.y);
                }
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int d = j * 8 + gid;
                    const uint32_t x0 = (uint8_t) V[strata_swz(cA, d)] | ((uint32_t) (uint8_t) V[strata_swz(cA + 1, d)] << 8);
                    const uint32_t x1 = (uint8_t) V[strata_swz(cB, d)] | ((uint32_t) (uint8_t) V[strata_swz(cB + 1, d)] << 8);
                    uint32_t b[2];
                    b[0] = strata_i8x2_to_h2(x0);
                    b[1] = strata_i8x2_to_h2(x1);
                    strata_mma16816(tmp[j], ah, b);
                    strata_mma16816(tmp[j], al, b);
                }
            }
            const float a0 = S.alpha[gid], a1 = S.alpha[gid + 8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                acc[j][0] = fmaf(acc[j][0], a0, tmp[j][0] * vdown);
                acc[j][1] = fmaf(acc[j][1], a0, tmp[j][1] * vdown);
                acc[j][2] = fmaf(acc[j][2], a1, tmp[j][2] * vdown);
                acc[j][3] = fmaf(acc[j][3], a1, tmp[j][3] * vdown);
            }
        }
        if (more) {
            S.sc[st ^ 1][warp][0][lane] = ksn;
            S.sc[st ^ 1][warp][1][lane] = vsn;
        }
        __syncwarp();
    }
    __syncthreads();
    const float l0 = S.lsum[gid], l1 = S.lsum[gid + 8];
    const float i0 = l0 > 0.0f ? 1.0f / l0 : 0.0f, i1 = l1 > 0.0f ? 1.0f / l1 : 0.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int d = dim0 + j * 8 + 2 * tig;
        *reinterpret_cast<float2 *>(attn + (size_t) gid * STRATA_HD + d) = make_float2(acc[j][0] * i0, acc[j][1] * i0);
        if (gid + 8 < STRATA_G)
            *reinterpret_cast<float2 *>(attn + (size_t) (gid + 8) * STRATA_HD + d) = make_float2(acc[j][2] * i1, acc[j][3] * i1);
    }
}

// strata_prompt_attn_i8_smem reports sizeof(StrataSmem2) to the host, so the launch's dynamic shared memory is the
// compiler's size of the layout rather than a transcription of it.
extern "C" __global__ void strata_prompt_attn_i8_smem(uint32_t * bytes) {
    if (threadIdx.x == 0 && blockIdx.x == 0) *bytes = (uint32_t) sizeof(StrataSmem2);
}

// ---- kv_stream.cu: KV streaming for a QSA layer whose cells outgrow its VRAM slots (kv_stream.hpp). The layer's
// authoritative cells sit in pinned, device-mapped host memory at the identity layout; VRAM holds n_slots pages and
// the page table is a residency map, block -> slot or -1. After the selection and before the attention, the resolve
// makes every block the selection names resident: hits take the call's epoch and their clock bit, a missing block is
// claimed once (-1 -> -2) into the miss list, one victim slot a miss is taken by a clock sweep that never takes a slot
// the call uses, and the copy reads the missed blocks from the host copy into their slots. The map's ints: ctl[0]
// epoch, ctl[1] hand, ctl[2] misses of the last call, ctl[3] overflow, u64 counters at ctl[4] misses, ctl[6] lookups,
// ctl[8] calls (kKvCtlInts = 16).
#define STRATA_KV_RESOLVE_THREADS 1024
#define STRATA_KV_CTL_INTS 16

// Block-wide exclusive prefix sum of one int per thread (1,024 threads); total is the sum over the block.
__device__ int strata_kv_block_scan(int v, int * warp_sums, int & total) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    int x = v;
    for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o) x += y;
    }
    if (lane == 31) warp_sums[w] = x;
    __syncthreads();
    if (w == 0) {
        int t = warp_sums[lane];
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, t, o);
            if (lane >= o) t += y;
        }
        warp_sums[lane] = t;
    }
    __syncthreads();
    total = warp_sums[31];
    const int excl = x - v + (w > 0 ? warp_sums[w - 1] : 0);
    __syncthreads();
    return excl;
}

// resolve_kernel: KvStreamMap's arrays as its pointers. Launch: 1 block of 1,024 threads.
extern "C" __global__ void __launch_bounds__(STRATA_KV_RESOLVE_THREADS) strata_kv_stream_resolve(
        int32_t * page_table, int32_t * slot_block, int32_t * slot_stamp, int32_t * slot_ref, int32_t * ctl,
        int32_t * miss_block, int32_t * miss_slot, long long n_slots,
        const int32_t * __restrict__ ids, const int32_t * __restrict__ steps, int n_q, int cap, int page_size) {
    const int RT = STRATA_KV_RESOLVE_THREADS;
    __shared__ int s_nmiss, s_lookups, s_cut;
    __shared__ int warp_sums[32];
    const int epoch = ctl[0] + 1;
    if (threadIdx.x == 0) { s_nmiss = 0; s_lookups = 0; }
    __syncthreads();
    // 1. hits take this epoch and their reference bit; a missing block is claimed exactly once (-1 -> -2)
    int lookups = 0;
    for (int q = 0; q < n_q; ++q) {
        const int width = steps[q * 4 + STRATA_STEP_WIDTH];
        const int32_t * qi = ids + (long long) q * cap;
        for (int i = threadIdx.x; i < width; i += RT) {
            const int b = qi[i] / page_size;
            if (i > 0 && qi[i - 1] / page_size == b) continue;   // ids are ascending: one lookup per block
            ++lookups;
            const int sl = page_table[b];
            if (sl >= 0) {
                slot_stamp[sl] = epoch;
                slot_ref[sl] = 1;
            } else if (sl == -1 && atomicCAS(&page_table[b], -1, -2) == -1) {
                miss_block[atomicAdd(&s_nmiss, 1)] = b;
            }
        }
    }
    if (lookups > 0) atomicAdd(&s_lookups, lookups);
    __syncthreads();
    // 2. one victim per miss: a clock sweep from the hand. A slot this call uses (stamp == epoch) is never taken;
    //    a referenced one loses its bit as the hand passes it and is taken on the next pass.
    const int need = s_nmiss, n = (int) n_slots;
    int hand = ctl[1], got = 0;
    for (int scanned = 0; got < need && scanned < 3 * n; scanned += RT) {
        const int j = (int) (((long long) hand + threadIdx.x) % n);
        const bool mine = slot_stamp[j] == epoch;
        const bool cand = !mine && (slot_block[j] < 0 || slot_ref[j] == 0);
        int total = 0;
        const int rank = strata_kv_block_scan(cand ? 1 : 0, warp_sums, total);
        const int want = need - got;
        if (threadIdx.x == 0) s_cut = RT;
        __syncthreads();
        if (cand && rank == want - 1) s_cut = threadIdx.x + 1;   // the hand stops just past the last slot taken
        __syncthreads();
        const int cut = s_cut;
        if (cand && rank < want) {
            miss_slot[got + rank] = j;
            slot_stamp[j] = epoch;   // taken: a sweep that wraps around must not take it twice
        } else if (threadIdx.x < cut && !mine) {
            slot_ref[j] = 0;
        }
        got += total < want ? total : want;
        hand = (int) (((long long) hand + cut) % n);
        __syncthreads();
    }
    // 3. re-point the table; the copy kernel fills the slots
    const int placed = got < need ? got : need;
    for (int k = threadIdx.x; k < need; k += RT) {
        const int b = miss_block[k];
        if (k >= placed) { page_table[b] = -1; continue; }   // overflow: never happens with a legal n_slots
        const int sl = miss_slot[k];
        const int old = slot_block[sl];
        if (old >= 0) page_table[old] = -1;
        slot_block[sl] = b;
        slot_stamp[sl] = epoch;
        slot_ref[sl] = 1;
        page_table[b] = sl;
    }
    if (threadIdx.x == 0) {
        ctl[0] = epoch;
        ctl[1] = hand;
        ctl[2] = placed;
        if (placed < need) ctl[3] = 1;
        unsigned long long * c = reinterpret_cast<unsigned long long *>(ctl + 4);
        c[0] += (unsigned long long) placed;
        c[1] += (unsigned long long) s_lookups;
        c[2] += 1ull;
    }
}

// copy_kernel: one block per missed block (grid-stride), its int8 runs (codes K, codes V, scales K, scales V) from the
// host copy into its slot, 16 B a thread. Runs as its pointers and lengths. Launch: 96 blocks of 128 threads.
extern "C" __global__ void strata_kv_stream_copy(
        const int32_t * ctl, const int32_t * miss_block, const int32_t * miss_slot,
        const uint8_t * src0, const uint8_t * src1, const uint8_t * src2, const uint8_t * src3,
        uint8_t * dst0, uint8_t * dst1, uint8_t * dst2, uint8_t * dst3, int len0, int len1, int len2, int len3) {
    const uint8_t * src[4] = {src0, src1, src2, src3};
    uint8_t * dst[4] = {dst0, dst1, dst2, dst3};
    const int len[4] = {len0, len1, len2, len3};
    const int need = ctl[2];
    for (int k = blockIdx.x; k < need; k += gridDim.x) {
        const long long b = miss_block[k], sl = miss_slot[k];
        for (int a = 0; a < 4; ++a) {
            const uint4 * s = reinterpret_cast<const uint4 *>(src[a] + b * len[a]);
            uint4 * d = reinterpret_cast<uint4 *>(dst[a] + sl * len[a]);
            for (int i = threadIdx.x; i < len[a] / 16; i += blockDim.x) d[i] = s[i];
        }
    }
}

// reset_kernel: every block evicted, every slot free, the control ints zero. Launch: 128 blocks of 256 threads.
extern "C" __global__ void strata_kv_stream_reset(
        int32_t * page_table, int32_t * slot_block, int32_t * slot_stamp, int32_t * slot_ref, int32_t * ctl,
        long long n_blocks, long long n_slots) {
    const long long i0 = (long long) blockIdx.x * blockDim.x + threadIdx.x, st = (long long) gridDim.x * blockDim.x;
    for (long long i = i0; i < n_blocks; i += st) page_table[i] = -1;
    for (long long i = i0; i < n_slots; i += st) {
        slot_block[i] = -1;
        slot_stamp[i] = -1;
        slot_ref[i] = 0;
    }
    if (i0 < STRATA_KV_CTL_INTS) ctl[i0] = 0;
}

// ---- prefill/kernels.cu kv_append_kernel, its int8 path (an int8 state never takes the fp16 one): a chunk's T
// cells from pos0 into the resident slots through the page table, the host copy and the prompt read's staging pool
// (both at the identity layout), one value a thread, the scale fp16(max|x| / 127) by __float2half_rn as prefill's
// hf writes it and the codes against the stored scale. Its geometry is the one prefill.cpp compiles for: 2 K/V heads
// of 256 values, 4 groups of 64. Launch: grid (T, 2, 8), 64 threads.
extern "C" __global__ void strata_prefill_kv_append_q8(
        const float * __restrict__ K, const float * __restrict__ V, long long pos0,
        const int32_t * __restrict__ table, long long page_size,
        int8_t * k_q, int8_t * v_q, uint16_t * k_scale, uint16_t * v_scale,
        int8_t * host_k_q, int8_t * host_v_q, uint16_t * host_k_scale, uint16_t * host_v_scale,
        int8_t * stage_k_q, int8_t * stage_v_q, uint16_t * stage_k_scale, uint16_t * stage_v_scale) {
    const long long t = blockIdx.x;
    const int kvh = blockIdx.y, g = blockIdx.z >> 1;
    const bool is_v = (blockIdx.z & 1) != 0;
    const int d = g * 64 + threadIdx.x;
    const float x = (is_v ? V : K)[t * 512 + kvh * 256 + d];
    const long long pos = pos0 + t;
    const long long page = table[pos / page_size];
    const long long row = (page * 2 + kvh) * page_size + pos % page_size;
    const long long row_id = ((pos / page_size) * 2 + kvh) * page_size + pos % page_size;
    float a = fabsf(x);
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    __shared__ float wm[2];
    if ((threadIdx.x & 31) == 0) wm[threadIdx.x >> 5] = a;
    __syncthreads();
    const float amax = fmaxf(wm[0], wm[1]);
    const uint16_t sb = __half_as_ushort(__float2half_rn(amax / 127.0f));
    const float sf = __half2float(__ushort_as_half(sb));
    int q = 0;
    if (sf > 0.0f) { q = __float2int_rn(x / sf); q = q < -127 ? -127 : (q > 127 ? 127 : q); }
    if (page >= 0) {
        (is_v ? v_q : k_q)[row * 256 + d] = (int8_t) q;
        if (threadIdx.x == 0) (is_v ? v_scale : k_scale)[row * 4 + g] = sb;
    }
    if (host_k_q != nullptr) {
        (is_v ? host_v_q : host_k_q)[row_id * 256 + d] = (int8_t) q;
        if (threadIdx.x == 0) (is_v ? host_v_scale : host_k_scale)[row_id * 4 + g] = sb;
    }
    if (stage_k_q != nullptr) {
        (is_v ? stage_v_q : stage_k_q)[row_id * 256 + d] = (int8_t) q;
        if (threadIdx.x == 0) (is_v ? stage_v_scale : stage_k_scale)[row_id * 4 + g] = sb;
    }
}

// Strata's gated-delta verify kernels (verify_kernels.cu), compiled with this module's flags as Strata compiles them.
#include "strata_gdn.cuh"

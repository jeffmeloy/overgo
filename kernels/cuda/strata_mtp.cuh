// strata_mtp.cuh - the kernels Strata's draft layer launches (src/core/mtp.cpp) beyond those its verify window shares,
// ported literally for its decode round (strata-port/decode).
//
// Ported from Strata (https://github.com/Niko1221/Strata, commit 296b59d2), MIT License, Copyright (c) 2026 Niko1221
// and the Strata contributors (licenses/Strata-MIT.txt), from src/kernels/cuda/s2_expert_grouped.cu (the S2 blob's
// geometry, f16_ld, expand_codes, load_x_chunk, chunk_s, stage_chunk, gu_grouped_t_kernel, down_grouped_t_kernel,
// swiglu_quantize_q8_0_scaled_kernel, group_resident_kernel), quantize_act.cu (quantize_q8_0_scaled_kernel, its CUDA
// form), verify_kernels.cu (add_streams_broadcast_kernel, gather_rows_kernel, map_ids_kernel, mtp_select_kernel,
// copy_row_to_first_kernel, TopRow and row_top_prob_split_kernel, window_ids_kernel) and elementwise.cu (MappedCopy and
// copy_from_mapped_multi_kernel).
//
// Strata compiles these files WITHOUT fast math, as this module is compiled. The S2 kernels are the ones Strata's
// grouped path takes on an aligned activation (new_grouped: STRATA_OLD_GROUPED unset), the SwiGLU and the scaled Q8_0
// quantize fused (STRATA_S2_SWIGLU_Q8 unset). The host sides are translated in
// internal/cuda/executor/launch_strata_mtp.go.

namespace strata_mtp {

// ---- s2_expert_grouped.cu: THE BLOB'S OWN GEOMETRY (the draft layer's 512 experts, Q2_0 codes and fp16 scales).
constexpr int H = 2560;
constexpr int FF = 640;
constexpr int QK = 64;                       // Q2_0's group: one fp16 scale per 64 weights
constexpr int ROW_GU = H / 4;                // 640 B of codes per gate/up row (2 bits per element)
constexpr int ROW_D = FF / 4;                // 160 B per down row
constexpr int SC_GU = H / QK;                // 40 fp16 scales per gate/up row
constexpr int SC_D = FF / QK;                // 10 per down row
constexpr size_t O_D_CODES = (size_t) 2 * FF * ROW_GU;
constexpr size_t O_GU_SCALES = O_D_CODES + (size_t) H * ROW_D;
constexpr size_t O_D_SCALES = O_GU_SCALES + (size_t) 2 * FF * SC_GU * 2;
constexpr int THREADS = 256;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xFFFFFFFFu, v, off);
    return v;
}

// fp16 at an even address, one 16-bit load (`f16_at` reads it as two bytes); same value.
__device__ __forceinline__ float f16_ld(const uint8_t* p) {
    return __half2float(__ushort_as_half(*(const unsigned short*) p));
}

// A chunk's two code words as the eight `dp4a` operands that pair with `X[0..7]`: `m[4h + f]` byte `b` is the
// code of element `16h + 4b + f`.
__device__ __forceinline__ void expand_codes(uint2 cb, int m[8]) {
    const unsigned M = 0x03030303u;
    m[0] = (int) (cb.x & M);
    m[1] = (int) ((cb.x >> 2) & M);
    m[2] = (int) ((cb.x >> 4) & M);
    m[3] = (int) ((cb.x >> 6) & M);
    m[4] = (int) (cb.y & M);
    m[5] = (int) ((cb.y >> 2) & M);
    m[6] = (int) ((cb.y >> 4) & M);
    m[7] = (int) ((cb.y >> 6) & M);
}

// The 32 int8 of the `block_q8_0` at `xb` regrouped into `X[0..7]` (see above), and their sum `hx`.
//
// **NINE ALIGNED WORD LOADS INSTEAD OF 32 BYTE LOADS.**  The caller guarantees a 4-byte aligned activation row, so
// chunk `c`'s int8 start at `34c + 2`, i.e. 0 bytes (odd `c`) or 2 bytes (even `c`) into an aligned word; the
// words are read aligned and funnel-shifted into place.  The ninth word is read only in the 2-byte case, and then
// it lies inside block `c + 1` of the same row: both row lengths (80 and 20 chunks) are even, so an even chunk is
// never a row's last.
__device__ __forceinline__ int load_x_chunk(const uint8_t* __restrict__ xb, int X[8]) {
    const uint8_t* q = xb + 2;
    const int off = (int) ((uintptr_t) q & 3);
    const unsigned* p = (const unsigned*) (q - off);
    const unsigned sh = (unsigned) off * 8;
    unsigned v[9];
#pragma unroll
    for (int j = 0; j < 8; ++j) v[j] = __ldg(p + j);
    v[8] = off != 0 ? __ldg(p + 8) : 0u;
    unsigned n[8];
    int hx = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        // natural word j: x[4j .. 4j+3]; a 64-bit shift (sh is 0, 8, 16 or 24) rather than __funnelshift_r, so the same
        // source needs no CUDA-only intrinsic.
        n[j] = (unsigned) ((((unsigned long long) v[j + 1] << 32) | v[j]) >> sh);
        hx = STRATA_DP4A(0x01010101, (int) n[j], hx);
    }
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const unsigned t0 = __byte_perm(n[4 * h], n[4 * h + 1], 0x5140);      // a0 b0 a1 b1
        const unsigned t1 = __byte_perm(n[4 * h], n[4 * h + 1], 0x7362);      // a2 b2 a3 b3
        const unsigned t2 = __byte_perm(n[4 * h + 2], n[4 * h + 3], 0x5140);  // c0 d0 c1 d1
        const unsigned t3 = __byte_perm(n[4 * h + 2], n[4 * h + 3], 0x7362);  // c2 d2 c3 d3
        X[4 * h + 0] = (int) __byte_perm(t0, t2, 0x5410);                    // a0 b0 c0 d0
        X[4 * h + 1] = (int) __byte_perm(t0, t2, 0x7632);                    // a1 b1 c1 d1
        X[4 * h + 2] = (int) __byte_perm(t1, t3, 0x5410);                    // a2 b2 c2 d2
        X[4 * h + 3] = (int) __byte_perm(t1, t3, 0x7632);                    // a3 b3 c3 d3
    }
    return hx;
}

// `s` for one chunk: the eight expanded code words against the eight regrouped activation words.
__device__ __forceinline__ int chunk_s(const int m[8], const int X[8]) {
    int s = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) s = STRATA_DP4A(m[j], X[j], s);
    return s;
}

__device__ __forceinline__ void swiglu_quantize_q8_0_scaled_kernel(float* __restrict__ gate_up, long long n_pairs,
                                                                   uint8_t* __restrict__ blocks,
                                                                   float* __restrict__ scales) {
    const long long n_blocks = n_pairs >> 5;
    const long long b = ((long long) blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (b >= n_blocks) return;
    const int lane = threadIdx.x & 31;
    const long long idx = b * 32 + lane;
    const float g = gate_up[idx];
    const float u = gate_up[n_pairs + idx];
    const float xv = (g / (1.0f + __expf(-g))) * u;
    gate_up[idx] = xv;
    uint8_t* out = blocks + b * 34;

    float amax = fabsf(xv);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
    const float s = amax > 0.f ? amax / 127.f : 0.f;
    const float inv = s > 0.f ? 1.f / s : 0.f;
    if (lane == 0) {
        scales[b] = s;
        const uint16_t d16bits = strata_f16_from_f32(s);
        out[0] = (uint8_t) (d16bits & 0xFF);
        out[1] = (uint8_t) (d16bits >> 8);
    }
    const float t = xv * inv;
    const float r = t + (t >= 0.f ? 0.5f : -0.5f);
    int v = (int) r;
    v = v < -127 ? -127 : (v > 127 ? 127 : v);
    out[2 + lane] = (uint8_t) (int8_t) v;
}

// Plan v0.3 P6: experts GROUPED - one blob pointer per group (a VRAM slot or a mapped host blob read over PCIe),
// every entry of the group (a token routed to that expert) computed from ONE read of each row.
constexpr int GU_CHUNKS = (H / 32 + 31) / 32;   // 3: chunks of a gate/up row per lane (80 chunks / 32 lanes)
constexpr int GMAX = 8;                          // entries per group (tokens routed to one expert in a window)
constexpr int GU_ROWS = 32;                      // gate/up rows per block: 4 per warp
constexpr int D_ROWS = 64;                       // down rows per block: 8 per warp

// Stages chunk `i = k * NC + c` of a group: its regrouped words at `xs_w[j * STRIDE + i]` and (dx, hx) at `xs_dh[i]`.
template <int STRIDE>
__device__ __forceinline__ void stage_chunk(const uint8_t* __restrict__ xb, float dx, int* xs_w, int2* xs_dh, int i) {
    int X[8];
    const int hx = load_x_chunk(xb, X);
#pragma unroll
    for (int j = 0; j < 8; ++j) xs_w[j * STRIDE + i] = X[j];
    xs_dh[i] = make_int2(__float_as_int(dx), hx);
}

__device__ __forceinline__ void gu_grouped_t_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                    const int32_t* __restrict__ grp_start,
                                                    const int32_t* __restrict__ n_groups,
                                                    const int32_t* __restrict__ ent_tok,
                                                    const uint8_t* __restrict__ x_q8_0,
                                                    const float* __restrict__ x_scales,
                                                    float* __restrict__ gate_up, int cap_entries) {
    constexpr int NC = H / 32;
    __shared__ int xs_w[8 * GMAX * NC];          // 20 KB: word j of entry k's chunk c at [j][k * NC + c]
    __shared__ int2 xs_dh[GMAX * NC];            // 5 KB: (dx as bits, hx)
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const int e0 = grp_start[g], ne = min(grp_start[g + 1] - e0, GMAX);
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    for (int i = t; i < ne * NC; i += blockDim.x) {
        const int k = i / NC, c = i - k * NC;
        const int tok = ent_tok[e0 + k];
        const uint8_t* xb = x_q8_0 + (size_t) tok * (size_t) NC * 34 + (size_t) c * 34;
        const float dx = x_scales ? x_scales[(size_t) tok * NC + c] : f16_ld(xb);
        stage_chunk<GMAX * NC>(xb, dx, xs_w, xs_dh, i);
    }
    __syncthreads();
    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const int row0 = blockIdx.x * GU_ROWS;                 // even: a pair is always (gate r, up r)
    for (int pp = warp; pp < GU_ROWS / 2; pp += 8) {
        const int i = row0 + 2 * pp;
        const uint8_t* codes = blob + (size_t) i * ROW_GU;
        const uint8_t* scales = blob + O_GU_SCALES + (size_t) i * SC_GU * 2;
        int m0[GU_CHUNKS][8], m1[GU_CHUNKS][8];
        float dw0[GU_CHUNKS], dw1[GU_CHUNKS];
#pragma unroll
        for (int q = 0; q < GU_CHUNKS; ++q) {
            const int c = lane + 32 * q;
            if (c < NC) {
                expand_codes(*(const uint2*) (codes + (size_t) c * 8), m0[q]);
                expand_codes(*(const uint2*) (codes + ROW_GU + (size_t) c * 8), m1[q]);
                dw0[q] = f16_ld(scales + (size_t) (c >> 1) * 2);
                dw1[q] = f16_ld(scales + SC_GU * 2 + (size_t) (c >> 1) * 2);
            }
        }
        for (int k = 0; k < ne; ++k) {
            float acc0 = 0.0f, acc1 = 0.0f;
#pragma unroll
            for (int q = 0; q < GU_CHUNKS; ++q) {
                const int c = lane + 32 * q;
                if (c >= NC) break;
                const int at = k * NC + c;
                int X[8];
#pragma unroll
                for (int j = 0; j < 8; ++j) X[j] = xs_w[j * (GMAX * NC) + at];
                const int2 dh = xs_dh[at];
                const float dx = __int_as_float(dh.x);
                acc0 += dw0[q] * dx * (float) (chunk_s(m0[q], X) - dh.y);
                acc1 += dw1[q] * dx * (float) (chunk_s(m1[q], X) - dh.y);
            }
            const float s0 = warp_sum(acc0);
            const float s1 = warp_sum(acc1);
            if (lane == 0) {
                const int e = e0 + k, r = i >> 1;
                gate_up[(size_t) e * FF + (size_t) r] = s0;
                gate_up[(size_t) cap_entries * FF + (size_t) e * FF + (size_t) r] = s1;
            }
        }
    }
}

__device__ __forceinline__ void down_grouped_t_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                      const int32_t* __restrict__ grp_start,
                                                      const int32_t* __restrict__ n_groups,
                                                      const int32_t* __restrict__ ent_dst,
                                                      const uint8_t* __restrict__ h_q8_0,
                                                      const float* __restrict__ h_scales,
                                                      float* __restrict__ out) {
    constexpr int NC = FF / 32;
    __shared__ int hs_w[8 * GMAX * NC];
    __shared__ int2 hs_dh[GMAX * NC];
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const int e0 = grp_start[g], ne = min(grp_start[g + 1] - e0, GMAX);
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    for (int i = t; i < ne * NC; i += blockDim.x) {
        const int k = i / NC, c = i - k * NC;
        const uint8_t* xb = h_q8_0 + (size_t) (e0 + k) * (size_t) NC * 34 + (size_t) c * 34;
        const float dx = h_scales ? h_scales[(size_t) (e0 + k) * NC + c] : f16_ld(xb);
        stage_chunk<GMAX * NC>(xb, dx, hs_w, hs_dh, i);
    }
    __syncthreads();
    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const int row0 = blockIdx.x * D_ROWS;
    for (int pp = warp; pp < D_ROWS / 2; pp += 8) {
        const int r = row0 + 2 * pp;
        const uint8_t* codes = blob + O_D_CODES + (size_t) r * ROW_D;
        const uint8_t* scales = blob + O_D_SCALES + (size_t) r * SC_D * 2;
        const int c = lane;                                // lanes 0..19 hold one chunk each, as above
        int m0[8], m1[8];
        float dw0 = 0.0f, dw1 = 0.0f;
        if (c < NC) {
            expand_codes(*(const uint2*) (codes + (size_t) c * 8), m0);
            expand_codes(*(const uint2*) (codes + ROW_D + (size_t) c * 8), m1);
            dw0 = f16_ld(scales + (size_t) (c >> 1) * 2);
            dw1 = f16_ld(scales + SC_D * 2 + (size_t) (c >> 1) * 2);
        }
        for (int k = 0; k < ne; ++k) {
            float acc0 = 0.0f, acc1 = 0.0f;
            if (c < NC) {
                const int at = k * NC + c;
                int X[8];
#pragma unroll
                for (int j = 0; j < 8; ++j) X[j] = hs_w[j * (GMAX * NC) + at];
                const int2 dh = hs_dh[at];
                const float dx = __int_as_float(dh.x);
                acc0 += dw0 * dx * (float) (chunk_s(m0, X) - dh.y);
                acc1 += dw1 * dx * (float) (chunk_s(m1, X) - dh.y);
            }
            const float s0 = warp_sum(acc0);
            const float s1 = warp_sum(acc1);
            if (lane == 0) {
                const size_t o = (size_t) ent_dst[e0 + k] * H + (size_t) r;
                out[o] = s0;
                out[o + 1] = s1;
            }
        }
    }
}

// Plan v0.3 P6: groups built on the device when every expert is resident at `base + id * blob` (the MTP layer):
// one block of 128 threads, groups in first-appearance order, entries of a group in routing order.
__device__ __forceinline__ void group_resident_kernel(const int32_t* __restrict__ ids, int n, int k_per_tok,
                                                      const uint8_t* base, long long blob,
                                                      unsigned long long* __restrict__ grp_ptr,
                                                      int32_t* __restrict__ grp_start, int32_t* __restrict__ counts,
                                                      int32_t* __restrict__ ent_dst, int32_t* __restrict__ ent_tok) {
    __shared__ int e_s[128], first_s[128], size_s[128], gidx_s[128], gstart_s[129];
    const int i = threadIdx.x;
    const int e = i < n ? ids[i] : -1;
    e_s[i] = e;
    __syncthreads();
    int first = i, rank = 0, size = 0;
    if (i < n) {
        for (int j = 0; j < i; ++j)
            if (e_s[j] == e) { if (first == i) first = j; ++rank; }
        if (first == i)
            for (int j = i; j < n; ++j) size += e_s[j] == e;
    }
    first_s[i] = first;
    size_s[i] = (i < n && first == i) ? size : 0;
    __syncthreads();
    if (i == 0) {
        int gi = 0, acc = 0;
        for (int j = 0; j < n; ++j)
            if (first_s[j] == j) {
                gidx_s[j] = gi;
                gstart_s[gi] = acc;
                grp_ptr[gi] = (unsigned long long) (base + (size_t) e_s[j] * (size_t) blob);
                grp_start[gi] = acc;
                acc += size_s[j];
                ++gi;
            }
        grp_start[gi] = acc;
        counts[0] = gi;
        counts[1] = acc;
    }
    __syncthreads();
    if (i < n) {
        const int at = gstart_s[gidx_s[first]] + rank;
        ent_dst[at] = i;
        ent_tok[at] = i / k_per_tok;
    }
}

// ---- quantize_act.cu: a warp a block, lane i on value i: the block's |max| is a max over the same values (0 for none
// above it, as the serial scan from 0 through fmaxf), so every byte is the per-thread loop's.
constexpr int QK8_0 = 32;
__device__ __forceinline__ void quantize_q8_0_scaled_kernel(const float* __restrict__ x, uint8_t* __restrict__ blocks,
                                                            float* __restrict__ scales, long long n_blocks) {
    const long long b = ((long long) blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int i = (int) (threadIdx.x & 31);
    if (b >= n_blocks) return;
    const float xi = x[b * QK8_0 + i];
    uint8_t* out = blocks + b * 34;

    float amax = fmaxf(0.0f, fabsf(xi));
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    // VERBATIM from `cpu/expert.cpp:144-145`, including the `amax > 0` guard, so the fp32 value written here
    // is bit-identical to the `s` the CPU path used.
    const float s = amax > 0.f ? amax / 127.f : 0.f;
    const float inv = s > 0.f ? 1.f / s : 0.f;
    if (i == 0) {
        scales[b] = s;
        const uint16_t d16bits = strata_f16_from_f32(s);
        out[0] = (uint8_t) (d16bits & 0xFF);
        out[1] = (uint8_t) (d16bits >> 8);
    }
    // VERBATIM from `cpu/expert.cpp:159-162`: reciprocal multiply, then `t + copysign(0.5, t)` truncated
    // toward zero, which is `lround`'s rule - round half away from zero.
    const float t = xi * inv;
    const float r = t + (t >= 0.f ? 0.5f : -0.5f);
    int v = (int) r;
    v = v < -127 ? -127 : (v > 127 ? 127 : v);
    out[2 + i] = (uint8_t) (int8_t) v;
}

// ---- elementwise.cu: up to eight copies out of mapped memory in one launch, copy y of the grid's rows.
struct MappedCopy {
    void* dst = nullptr;
    const void* src = nullptr;
    int64_t words = 0;
};
struct MappedCopies {
    MappedCopy c[8];
};

__device__ __forceinline__ void copy_from_mapped_multi_kernel(const MappedCopies& a) {
    const MappedCopy c = a.c[blockIdx.y];
    const int64_t i0 = (int64_t) blockIdx.x * blockDim.x + threadIdx.x, st = (int64_t) gridDim.x * blockDim.x;
    if ((c.words & 3) == 0 && ((uintptr_t) c.dst & 15) == 0 && ((uintptr_t) c.src & 15) == 0) {
        const volatile float4* s = (const volatile float4*) c.src;
        for (int64_t i = i0; i < c.words / 4; i += st) ((float4*) c.dst)[i] = const_cast<const float4*>(s)[i];
    } else {
        const volatile int32_t* s = (const volatile int32_t*) c.src;
        for (int64_t i = i0; i < c.words; i += st) ((int32_t*) c.dst)[i] = s[i];
    }
}

// ---- verify_kernels.cu: the draft layer's input and selection kernels.
__device__ __forceinline__ void add_streams_broadcast_kernel(const float* __restrict__ h, const float* __restrict__ e,
                                                             float* __restrict__ R, int64_t n, int hc) {
    const int t = blockIdx.y;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * hc) return;
    R[(size_t) t * n * hc + i] = h[(size_t) t * n * hc + i] + e[(size_t) t * n + i % n];
}

// E = the widest element the row size divides into (16, 4 or 1 bytes): a Q6_K head row of 2560 values is 2100 bytes
template<typename E>
__device__ __forceinline__ void gather_rows_kernel(const E* __restrict__ src, long long row_e,
                                                   const int32_t* __restrict__ ids, long long n, E* __restrict__ dst) {
    const long long total = n * row_e;
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (long long) gridDim.x * blockDim.x) {
        const long long r = i / row_e, o = i - r * row_e;
        dst[i] = src[(long long) ids[r] * row_e + o];
    }
}

__device__ __forceinline__ void map_ids_kernel(int32_t* ids, const int32_t* __restrict__ table, int n) {
    const int i = threadIdx.x;
    if (i < n) ids[i] = table[ids[i]];
}

__device__ __forceinline__ void mtp_select_kernel(const float* __restrict__ R_src, int64_t stride,
                                                  const int32_t* __restrict__ ids, const int32_t* __restrict__ row_dev,
                                                  float* __restrict__ R_dst, int32_t* __restrict__ tok_dst,
                                                  int32_t* out, int j, const float* probs, float* out_p) {
    const int row = *row_dev;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < stride; i += (int64_t) gridDim.x * blockDim.x)
        R_dst[i] = R_src[(size_t) row * stride + i];
    __syncthreads();
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const int32_t tok = ids[row];
        *tok_dst = tok;
        if (probs != nullptr && out_p != nullptr) ((volatile float*) out_p)[j] = probs[row];
        __threadfence_system();
        if (out != nullptr) ((volatile int32_t*) out)[j] = tok;
    }
}

__device__ __forceinline__ void copy_row_to_first_kernel(const int32_t* __restrict__ row_dev, float* a, int64_t a_n,
                                                         float* b, int64_t b_n, float* c, int64_t c_n) {
    const int64_t row = *row_dev;
    if (row == 0) return;
    const int64_t i0 = (int64_t) blockIdx.x * blockDim.x + threadIdx.x, st = (int64_t) gridDim.x * blockDim.x;
    for (int64_t i = i0; i < a_n; i += st) a[i] = a[row * a_n + i];
    for (int64_t i = i0; i < b_n; i += st) b[i] = b[row * b_n + i];
    for (int64_t i = i0; i < c_n; i += st) c[i] = c[row * c_n + i];
}

// a row's top-probability scratch: its counter, then its 32 warps' sums
struct TopRow {
    unsigned counter;
    unsigned pad[3];
    float parts[32];
};

// grid (kTopBlocks, rows), 1024 / kTopBlocks threads: thread j of block b is row_top_prob's thread b * blockDim + j
__device__ __forceinline__ void row_top_prob_split_kernel(const float* __restrict__ logits, int n_vocab,
                                                          const int32_t* __restrict__ ids, float* __restrict__ probs,
                                                          TopRow* __restrict__ rows) {
    const int row = (int) blockIdx.y;
    float* parts = rows[row].parts;
    const float* l = logits + (size_t) row * n_vocab;
    const float m = l[ids[row]];
    const int vt = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    float s = 0.0f;
    for (int i = vt; i < n_vocab; i += 1024) s += __expf(l[i] - m);
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if ((threadIdx.x & 31) == 0) parts[vt >> 5] = s;
    __threadfence();
    __syncthreads();
    __shared__ bool last;
    if (threadIdx.x == 0) last = atomicAdd(&rows[row].counter, 1u) == (unsigned) (gridDim.x - 1);
    __syncthreads();
    if (!last || threadIdx.x != 0) return;
    __threadfence();
    float tot = 0.0f;
    for (int w = 0; w < 32; ++w) tot += ((volatile const float*) parts)[w];
    probs[row] = 1.0f / tot;
    rows[row].counter = 0;
}

__device__ __forceinline__ void window_ids_kernel(int32_t* steps, int window, int32_t* ids, long long stride) {
    const int q = blockIdx.y;
    int32_t* st = steps + q * 4;
    const int n_kv = st[1];
    const int start = n_kv > window ? n_kv - window : 0;
    const int width = n_kv - start;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < width; j += gridDim.x * blockDim.x)
        ids[q * stride + j] = start + j;
    __syncthreads();
    if (blockIdx.x == 0 && threadIdx.x == 0) st[3] = width;
}

}  // namespace strata_mtp

// ---- the entries, each with the launch shape its host side gives it (launch_strata_mtp.go).
extern "C" __global__ void __launch_bounds__(256) strata_gu_grouped_t(
        const unsigned long long* __restrict__ grp_ptr, const int32_t* __restrict__ grp_start,
        const int32_t* __restrict__ n_groups, const int32_t* __restrict__ ent_tok, const uint8_t* __restrict__ x_q8_0,
        const float* __restrict__ x_scales, float* __restrict__ gate_up, int cap_entries) {
    strata_mtp::gu_grouped_t_kernel(grp_ptr, grp_start, n_groups, ent_tok, x_q8_0, x_scales, gate_up, cap_entries);
}
extern "C" __global__ void __launch_bounds__(256) strata_down_grouped_t(
        const unsigned long long* __restrict__ grp_ptr, const int32_t* __restrict__ grp_start,
        const int32_t* __restrict__ n_groups, const int32_t* __restrict__ ent_dst, const uint8_t* __restrict__ h_q8_0,
        const float* __restrict__ h_scales, float* __restrict__ out) {
    strata_mtp::down_grouped_t_kernel(grp_ptr, grp_start, n_groups, ent_dst, h_q8_0, h_scales, out);
}
extern "C" __global__ void strata_swiglu_quantize_q8_0_scaled(float* __restrict__ gate_up, long long n_pairs,
                                                              uint8_t* __restrict__ blocks, float* __restrict__ scales) {
    strata_mtp::swiglu_quantize_q8_0_scaled_kernel(gate_up, n_pairs, blocks, scales);
}
extern "C" __global__ void strata_group_resident(const int32_t* __restrict__ ids, int n, int k_per_tok,
                                                 const uint8_t* base, long long blob,
                                                 unsigned long long* __restrict__ grp_ptr,
                                                 int32_t* __restrict__ grp_start, int32_t* __restrict__ counts,
                                                 int32_t* __restrict__ ent_dst, int32_t* __restrict__ ent_tok) {
    strata_mtp::group_resident_kernel(ids, n, k_per_tok, base, blob, grp_ptr, grp_start, counts, ent_dst, ent_tok);
}
extern "C" __global__ void strata_quantize_q8_0_scaled(const float* __restrict__ x, uint8_t* __restrict__ blocks,
                                                       float* __restrict__ scales, long long n_blocks) {
    strata_mtp::quantize_q8_0_scaled_kernel(x, blocks, scales, n_blocks);
}
extern "C" __global__ void strata_copy_from_mapped_multi(const strata_mtp::MappedCopies a) {
    strata_mtp::copy_from_mapped_multi_kernel(a);
}
extern "C" __global__ void strata_add_streams_broadcast(const float* __restrict__ h, const float* __restrict__ e,
                                                        float* __restrict__ R, int64_t n, int hc) {
    strata_mtp::add_streams_broadcast_kernel(h, e, R, n, hc);
}
extern "C" __global__ void strata_gather_rows_u32(const uint32_t* __restrict__ src, long long row_e,
                                                  const int32_t* __restrict__ ids, long long n, uint32_t* __restrict__ dst) {
    strata_mtp::gather_rows_kernel<uint32_t>(src, row_e, ids, n, dst);
}
extern "C" __global__ void strata_map_ids(int32_t* ids, const int32_t* __restrict__ table, int n) {
    strata_mtp::map_ids_kernel(ids, table, n);
}
extern "C" __global__ void strata_mtp_select(const float* __restrict__ R_src, int64_t stride,
                                             const int32_t* __restrict__ ids, const int32_t* __restrict__ row_dev,
                                             float* __restrict__ R_dst, int32_t* __restrict__ tok_dst, int32_t* out,
                                             int j, const float* probs, float* out_p) {
    strata_mtp::mtp_select_kernel(R_src, stride, ids, row_dev, R_dst, tok_dst, out, j, probs, out_p);
}
extern "C" __global__ void strata_copy_row_to_first(const int32_t* __restrict__ row_dev, float* a, int64_t a_n,
                                                    float* b, int64_t b_n, float* c, int64_t c_n) {
    strata_mtp::copy_row_to_first_kernel(row_dev, a, a_n, b, b_n, c, c_n);
}
extern "C" __global__ void strata_row_top_prob_split(const float* __restrict__ logits, int n_vocab,
                                                     const int32_t* __restrict__ ids, float* __restrict__ probs,
                                                     strata_mtp::TopRow* __restrict__ rows) {
    strata_mtp::row_top_prob_split_kernel(logits, n_vocab, ids, probs, rows);
}
extern "C" __global__ void strata_window_ids(int32_t* steps, int window, int32_t* ids, long long stride) {
    strata_mtp::window_ids_kernel(steps, window, ids, stride);
}

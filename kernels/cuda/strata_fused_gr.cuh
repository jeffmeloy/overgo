// strata_fused_gr.cuh - Strata's hyper-connection read over the GGUF's Q8_0 projections, ported literally for its decode
// round (strata-port/decode).
//
// Ported from Strata (https://github.com/Niko1221/Strata, commit 296b59d2), MIT License, Copyright (c) 2026 Niko1221
// and the Strata contributors (licenses/Strata-MIT.txt), from src/kernels/cuda/fused_gr.cu (gr_down_q8_kernel,
// gr_up_q8_kernel, launch_q8_t; and the default read the drafter takes: gr_norm_split_kernel, gr_down_staged_kernel,
// gr_up_multi_kernel and their helpers) and include/strata/kernels/fused_gr.hpp (FusedGrArgs, kFusedGrMaxT).
//
// Strata compiles fused_gr.cu WITHOUT fast math, as this module is compiled. The read is the S23 one Strata's bench
// takes (STRATA_HC_Q8=1): the hyper-connection down and up projections' Q8_0 bytes as the GGUF stores them, the inject
// rows as the pack's bf16 (the GGUF's f32 values are bf16-exact). Each template instance is an extern "C" entry of its
// own, the GrMulti argument passed by value as Strata passes it; the host side (fused_gr_read_multi, launch_q8) is
// translated in internal/cuda/executor/launch_strata_fused_gr.go. The q8_1 tail of the up kernel (STRATA_QFUSE, opt-in
// in Strata, off as benched) is not ported: the round quantizes the mixed rows itself.

namespace strata_gr {

constexpr int N = 2560;         // n_embd
constexpr int HC = 4;           // streams
constexpr int D = N * HC;       // 10240
constexpr int LR = 320;         // hc_lr
constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;
constexpr int UPM_COLS = 16;                      // columns per block (x 4 streams = 64 rows, 8 per warp)
constexpr int UPM_BLOCKS = N / UPM_COLS;          // 160
constexpr int PR = LR + HC;                       // partial rows per (token, stream): 320 down + 4 inject
constexpr int kFusedGrMaxT = 8;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + __expf(-x)); }

struct FusedGrArgs {
    const float* R = nullptr;          ///< (hc, n_embd), read by `down`; `up` updates it in place when apply
    float* R_out = nullptr;            ///< == R for the in-place update
    bool apply = false;                ///< fold the previous half's gr_write
    const float* bo_prev = nullptr;    ///< that half's block output, n_embd
    const float* inj_prev = nullptr;   ///< that half's injection, hc
    const float* w_norm = nullptr;     ///< (hc * n_embd) f32
    const uint16_t* w_down = nullptr;  ///< bf16 [hc_lr][hc*n_embd]
    const uint16_t* w_up = nullptr;    ///< bf16 [hc*n_embd][hc_lr]
    const uint16_t* w_inject = nullptr;///< bf16 [hc][hc*n_embd], or null (the final mixer)
    float eps = 1e-6f;
    float* lo = nullptr;               ///< workspace, hc_lr floats
    float* rs = nullptr;               ///< workspace, hc floats
    float* inject_out = nullptr;       ///< hc floats (when w_inject)
    float* mixed = nullptr;            ///< n_embd
    /// S23 experiment (STRATA_HC_Q8=1): the GGUF's Q8_0 projections (null: the BF16 ones above); fused_gr_read_multi
    /// only
    const uint8_t* q8_down = nullptr;  ///< Q8_0 [hc_lr][hc*n_embd]
    const uint8_t* q8_up = nullptr;    ///< Q8_0 [hc*n_embd][hc_lr]
    const uint8_t* q8_inject = nullptr;///< Q8_0 [hc][hc*n_embd]
    /// S26 STRATA_QFUSE=1 (fused_gr_read_multi, the default and Q8_0 reads): also write `mixed`'s q8_1 image here (the
    /// bytes native_quantize_q8_1 would write); q8_cnt = n_embd / 32 zeroed counters owned by the caller (token 0's)
    uint8_t* q8_mixed = nullptr;
    unsigned* q8_cnt = nullptr;
};

struct GrMulti {
    FusedGrArgs a[kFusedGrMaxT];
    float* xn;
    int T;
};

// ================================ S23 experiment (STRATA_HC_Q8=1): the read with the GGUF's Q8_0 projections ======
// The pack holds the hyper-connection projections as BF16 (iq_pack.py --compat-bf16 rounds the GGUF's Q8_0 values to
// BF16: 1.20 GiB per verify step); this read takes the Q8_0 bytes as stored (0.65 GiB - and the GGUF's own values,
// not a rounding of them).  The layout is the stream split of the v3 read above, at 4 chunks per stream: `down` runs
// on (10 groups of 32 rows + the inject rows) x 16 chunks of 640 columns = 176 blocks, each staging its chunk of
// R' * w_norm (unscaled) for the T tokens and writing the chunk's partial dots and sums of squares; `up` sums them
// per stream in a fixed order, applies rs, SwiGLU-free silu, and runs the default epilogue.  Another summation order
// and other weights than the default read: an output-changing step (measured, quality-checked).
constexpr int Q8B = 34;                             // Q8_0 block: fp16 d + 32 int8
constexpr int Q8_RPW = 4;                           // down rows per warp
constexpr int Q8_RG = LR / (WARPS * Q8_RPW);        // 10 row groups (+1: the inject rows)
constexpr int Q8_KC = 640, Q8_NKC = D / Q8_KC;      // 16 chunks, 4 per stream
constexpr int Q8_CPS = N / Q8_KC;                   // chunks per stream
constexpr int Q8_SPB = Q8_KC / 128;                 // 5 steps of 4 Q8_0 blocks (8 lanes x 4 values each)
__device__ __forceinline__ uint32_t q8_ld16(const uint8_t* p) { return *(const uint16_t*) p; }
__device__ __forceinline__ float q8_half(uint32_t h) { return __half2float(__ushort_as_half((unsigned short) h)); }
__device__ __forceinline__ float4 q8_four(uint32_t q, float d) {
    return make_float4(d * (float) (int8_t) (q & 255u), d * (float) (int8_t) ((q >> 8) & 255u),
                       d * (float) (int8_t) ((q >> 16) & 255u), d * (float) (int8_t) (q >> 24));
}

template <int T>
__device__ __forceinline__ void gr_down_q8_kernel(const GrMulti& m, float* __restrict__ part, float* __restrict__ ssg) {
    __shared__ __align__(16) float xs[T][Q8_KC];
    __shared__ float red[WARPS][T];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int rg = blockIdx.x, kc = blockIdx.y, c = kc / Q8_CPS;
    const bool inj = rg == Q8_RG;
    // S25: the inject rows are F32 in the GGUF with BF16-exact values, so by default they are read as the pack's BF16
    // (exact); a Q8_0 copy (STRATA_HC_Q8_INJECT=1) is the other option
    const bool inj_bf16 = inj && m.a[0].q8_inject == nullptr;
    const int nrows = inj ? (((m.a[0].q8_inject != nullptr || m.a[0].w_inject != nullptr) && warp < HC) ? 1 : 0) : Q8_RPW;
    const int row0 = inj ? warp : (rg * WARPS + warp) * Q8_RPW;
    const uint8_t* wb = inj ? m.a[0].q8_inject : m.a[0].q8_down;
    const int sub = lane & 7, bl = lane >> 3;       // lanes 8b..8b+7: one Q8_0 block, 4 values each
    uint32_t q[Q8_RPW][Q8_SPB];
    float dq[Q8_RPW][Q8_SPB];
#pragma unroll
    for (int r = 0; r < Q8_RPW; ++r) {
        if (r >= nrows || inj_bf16) break;
        const uint8_t* row = wb + (size_t) (row0 + r) * (D / 32) * Q8B + (size_t) kc * (Q8_KC / 32) * Q8B;
#pragma unroll
        for (int s = 0; s < Q8_SPB; ++s) {
            const uint8_t* blk = row + (4 * s + bl) * Q8B;
            dq[r][s] = q8_half(q8_ld16(blk));
            q[r][s] = q8_ld16(blk + 2 + 4 * sub) | q8_ld16(blk + 4 + 4 * sub) << 16;
        }
    }
    float ssp[T];
#pragma unroll
    for (int k = 0; k < T; ++k) {
        ssp[k] = 0.0f;
        const FusedGrArgs& a = m.a[k];
        const float gw = a.apply ? 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC) : 0.0f;
        for (int i = t; i < Q8_KC / 4; i += THREADS) {
            const int col = kc * Q8_KC + 4 * i;
            float4 r = *reinterpret_cast<const float4*>(a.R + col);
            if (a.apply) {
                const float4 b = *reinterpret_cast<const float4*>(a.bo_prev + (col - c * N));
                r.x = fmaf(b.x, gw, r.x); r.y = fmaf(b.y, gw, r.y);
                r.z = fmaf(b.z, gw, r.z); r.w = fmaf(b.w, gw, r.w);
            }
            const float4 g = *reinterpret_cast<const float4*>(a.w_norm + col);
            ssp[k] += r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w;
            *reinterpret_cast<float4*>(&xs[k][4 * i]) = make_float4(r.x * g.x, r.y * g.y, r.z * g.z, r.w * g.w);
        }
    }
#pragma unroll
    for (int k = 0; k < T; ++k) {
        const float v = warp_sum(ssp[k]);
        if (lane == 0) red[warp][k] = v;
    }
    __syncthreads();
    if (rg == 0 && t < T) {
        float s = 0.0f;
        for (int w = 0; w < WARPS; ++w) s += red[w][t];
        ssg[t * Q8_NKC + kc] = s;
    }
    if (nrows == 0) return;
#pragma unroll
    for (int r = 0; r < Q8_RPW; ++r) {
        if (r >= nrows) break;
        float acc[T];
#pragma unroll
        for (int k = 0; k < T; ++k) acc[k] = 0.0f;
#pragma unroll
        for (int s = 0; s < Q8_SPB; ++s) {
            const int v = 32 * (4 * s + bl) + 4 * sub;
            float4 w;
            if (inj_bf16) {
                const uint2 b = *reinterpret_cast<const uint2*>(m.a[0].w_inject + (size_t) row0 * D + (size_t) kc * Q8_KC + v);
                w = make_float4(__uint_as_float(b.x << 16), __uint_as_float(b.x & 0xffff0000u),
                                __uint_as_float(b.y << 16), __uint_as_float(b.y & 0xffff0000u));
            } else {
                w = q8_four(q[r][s], dq[r][s]);
            }
#pragma unroll
            for (int k = 0; k < T; ++k) {
                const float4 x = *reinterpret_cast<const float4*>(&xs[k][v]);
                acc[k] = fmaf(w.x, x.x, acc[k]); acc[k] = fmaf(w.y, x.y, acc[k]);
                acc[k] = fmaf(w.z, x.z, acc[k]); acc[k] = fmaf(w.w, x.w, acc[k]);
            }
        }
        const int prow = inj ? LR + warp : row0 + r;
#pragma unroll
        for (int k = 0; k < T; ++k) {
            const float v = warp_sum(acc[k]);
            if (lane == 0) part[((size_t) k * Q8_NKC + kc) * PR + prow] = v;
        }
    }
}

template <int T>
__device__ __forceinline__ void gr_up_q8_kernel(const GrMulti& m, const float* __restrict__ part,
                                                const float* __restrict__ ssg) {
    __shared__ __align__(16) float lo[T][LR];
    __shared__ float rsS[T][HC];
    __shared__ float g[T][HC][UPM_COLS];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int d0 = blockIdx.x * UPM_COLS;
    if (t < T * HC) {
        const int k = t / HC, c = t - k * HC;
        float ss = 0.0f;
#pragma unroll
        for (int j = 0; j < Q8_CPS; ++j) ss += ssg[k * Q8_NKC + c * Q8_CPS + j];
        const float r = rsqrtf(ss / (float) N + m.a[k].eps);
        rsS[k][c] = r;
        if (blockIdx.x == 0) m.a[k].rs[c] = r;
    }
    __syncthreads();
    for (int i = t; i < T * PR; i += THREADS) {
        const int k = i / PR, r = i - k * PR;
        if (r >= LR && (blockIdx.x != 0 || m.a[k].w_inject == nullptr)) continue;
        float sum = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) {
            float p = 0.0f;
#pragma unroll
            for (int j = 0; j < Q8_CPS; ++j) p += part[((size_t) k * Q8_NKC + c * Q8_CPS + j) * PR + r];
            sum = fmaf(rsS[k][c], p, sum);
        }
        if (r < LR) {
            const float x = sum / (float) HC;
            lo[k][r] = x / (1.0f + __expf(-x));
        } else {
            m.a[k].inject_out[r - LR] = sum;
        }
    }
    __syncthreads();
    constexpr int RPW8 = HC * UPM_COLS / WARPS;     // 8 rows per warp
#pragma unroll 1
    for (int qq = 0; qq < RPW8; ++qq) {
        const int r = warp + qq * WARPS;
        const int c = r / UPM_COLS, dd = r - c * UPM_COLS, i = c * N + d0 + dd;
        const uint8_t* row = m.a[0].q8_up + (size_t) i * (LR / 32) * Q8B;
        float4 w[3];
#pragma unroll
        for (int s = 0; s < 3; ++s) {   // lane: values 4 lane + 128 s (the third step: lanes 0-15)
            const int v = 4 * lane + 128 * s;
            if (v < LR) {
                const uint8_t* blk = row + (v >> 5) * Q8B;
                w[s] = q8_four(q8_ld16(blk + 2 + (v & 31)) | q8_ld16(blk + 4 + (v & 31)) << 16, q8_half(q8_ld16(blk)));
            } else {
                w[s] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
        float rv = 0.0f, wn = 0.0f, bo = 0.0f, ip = 0.0f;
        bool apply = false;
        if (lane < T) {
            const FusedGrArgs& a = m.a[lane];
            rv = a.R[i];
            wn = a.w_norm[i];
            apply = a.apply;
            if (apply) { bo = a.bo_prev[d0 + dd]; ip = a.inj_prev[c]; }
        }
        float mine = 0.0f;
#pragma unroll
        for (int k = 0; k < T; ++k) {
            float acc = 0.0f;
#pragma unroll
            for (int s = 0; s < 3; ++s) {
                const int v = 4 * lane + 128 * s;
                if (v < LR) {
                    const float4 x = *reinterpret_cast<const float4*>(&lo[k][v]);
                    acc = fmaf(w[s].x, x.x, acc); acc = fmaf(w[s].y, x.y, acc);
                    acc = fmaf(w[s].z, x.z, acc); acc = fmaf(w[s].w, x.w, acc);
                }
            }
            acc = warp_sum(acc);
            if (lane == k) mine = acc;
        }
        if (lane < T) {
            if (apply) {
                rv = fmaf(bo, 2.0f * sigmoidf_(ip / (float) HC), rv);
                m.a[lane].R_out[i] = rv;
            }
            const float x = rv * wn * rsS[lane][c];
            g[lane][c][dd] = x * sigmoidf_(mine);
        }
    }
    __syncthreads();
    for (int i = t; i < T * UPM_COLS; i += THREADS) {
        const int k = i / UPM_COLS, col = i - k * UPM_COLS;
        float sum = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) sum += g[k][c][col];
        m.a[k].mixed[d0 + col] = sum / (float) HC;
    }
}

// ================================ The default read over the bf16 projections, as the drafter takes it ==============
// fused_gr_read_multi without q8_down (the draft layer's hyper-connections are the rt pack's bf16): launch_multi at the
// variant fused_gr_check accepts on this card, kHcStaged - gr_norm_split_kernel, gr_down_staged_kernel<T, true> (a
// launch of exactly T <= 6 tokens, STRATA_NO_MULTI_GR unset), gr_up_multi_kernel<T, true>. q8_mixed is null (no
// STRATA_QFUSE counters), so gr_up_multi_kernel's q8_1 tail never runs and is not ported.

constexpr int DOWN_BLOCKS = LR / WARPS;          // 40 blocks of 8 rows; one more for the inject rows

// 8 bf16 packed in a uint4, unpacked to two float4 once when reused across tokens.
struct Bf16x8 { float4 a, b; };
__device__ __forceinline__ Bf16x8 unpack8(const uint4 w) {
    return {
        make_float4(__uint_as_float(w.x << 16), __uint_as_float(w.x & 0xffff0000u),
                    __uint_as_float(w.y << 16), __uint_as_float(w.y & 0xffff0000u)),
        make_float4(__uint_as_float(w.z << 16), __uint_as_float(w.z & 0xffff0000u),
                    __uint_as_float(w.w << 16), __uint_as_float(w.w & 0xffff0000u))
    };
}
__device__ __forceinline__ float dot8u(const Bf16x8& w, const float4 x0, const float4 x1) {
    float acc = 0.0f;
    acc = fmaf(w.a.x, x0.x, acc);
    acc = fmaf(w.a.y, x0.y, acc);
    acc = fmaf(w.a.z, x0.z, acc);
    acc = fmaf(w.a.w, x0.w, acc);
    acc = fmaf(w.b.x, x1.x, acc);
    acc = fmaf(w.b.y, x1.y, acc);
    acc = fmaf(w.b.z, x1.z, acc);
    acc = fmaf(w.b.w, x1.w, acc);
    return acc;
}
__device__ __forceinline__ float dot8u_ptr(const Bf16x8& w, const float* x) {
    const float4* x4 = reinterpret_cast<const float4*>(x);
    return dot8u(w, x4[0], x4[1]);
}

// `gr_up_kernel` for T tokens: each row of w_up read once; the T dots reduced by xor so every lane holds every
// sum, and lane k runs token k's epilogue - the T epilogues in parallel instead of one after another.
template <int MAX_T = kFusedGrMaxT, bool EXACT_T = false>
__device__ __forceinline__ void gr_up_multi_kernel(const GrMulti& m) {
    __shared__ __align__(16) float lo[MAX_T][LR];
    __shared__ float g[MAX_T][HC][UPM_COLS];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int T = EXACT_T ? MAX_T : m.T;
    const int d0 = blockIdx.x * UPM_COLS;
    for (int i = t; i < T * LR; i += THREADS) lo[i / LR][i % LR] = m.a[i / LR].lo[i % LR];
    __syncthreads();
    for (int r = warp; r < HC * UPM_COLS; r += WARPS) {
        const int c = r / UPM_COLS, dd = r - c * UPM_COLS, i = c * N + d0 + dd;
        const uint4* w4 = reinterpret_cast<const uint4*>(m.a[0].w_up + (size_t) i * LR);
        const Bf16x8 wa = unpack8(__ldg(w4 + lane));
        const Bf16x8 wb = unpack8(lane < LR / 8 - 32 ? __ldg(w4 + 32 + lane) : make_uint4(0, 0, 0, 0));
        // the epilogue inputs of this lane's token, fetched while the dots run
        float rv = 0.0f, wn = 0.0f, rsc = 0.0f, bo = 0.0f, ip = 0.0f;
        bool apply = false;
        if (lane < T) {
            const FusedGrArgs& a = m.a[lane];
            rv = a.R[i];
            wn = a.w_norm[i];
            rsc = a.rs[c];
            apply = a.apply;
            if (apply) { bo = a.bo_prev[d0 + dd]; ip = a.inj_prev[c]; }
        }
        float mine = 0.0f;
#pragma unroll
        for (int k = 0; k < MAX_T; ++k) {
            if (!EXACT_T && k >= T) break;
            float acc = dot8u_ptr(wa, lo[k] + lane * 8);
            if (lane < LR / 8 - 32) acc += dot8u_ptr(wb, lo[k] + (32 + lane) * 8);
            acc = warp_sum(acc);
            if (lane == k) mine = acc;
        }
        if (lane < T) {
            if (apply) {
                rv = fmaf(bo, 2.0f * sigmoidf_(ip / (float) HC), rv);
                m.a[lane].R_out[i] = rv;
            }
            const float x = rv * wn * rsc;
            g[lane][c][dd] = x * sigmoidf_(mine);
        }
    }
    __syncthreads();
    for (int i = t; i < T * UPM_COLS; i += THREADS) {
        const int k = i / UPM_COLS, col = i - k * UPM_COLS;
        float s = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) s += g[k][c][col];
        m.a[k].mixed[d0 + col] = s / (float) HC;
    }
}

// The bitwise-equal split read: the norm one block a (token, stream), the down projection over staged tiles.
constexpr int H_TILE = 1280;                          // staged tile: half a stream = 160 chunks of 8, 5 per lane
constexpr int HQ = H_TILE / 8 / 32;
constexpr int N_HTILES = D / H_TILE;                  // 8
static_assert(N % H_TILE == 0, "a staged tile never straddles two streams");
static_assert(H_TILE % 256 == 0, "a staged tile holds whole rounds of 32 chunks: the plain read's lane order");

__device__ __forceinline__ void gr_norm_split_kernel(const GrMulti& m) {
    __shared__ float part[WARPS];
    __shared__ float s_rs;
    const FusedGrArgs& a = m.a[blockIdx.x];
    const int c = blockIdx.y;
    float* xn = m.xn + (size_t) blockIdx.x * D;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const float gw = a.apply ? 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC) : 0.0f;
    float ss = 0.0f;
    for (int i = t * 4; i < D; i += THREADS * 4) {
        if (i / N != c) continue;                       // another stream's element: another block's
        const int d = i - c * N;
        float4 r = *reinterpret_cast<const float4*>(a.R + i);
        if (a.apply) {
            const float4 b = *reinterpret_cast<const float4*>(a.bo_prev + d);
            r.x = fmaf(b.x, gw, r.x); r.y = fmaf(b.y, gw, r.y);
            r.z = fmaf(b.z, gw, r.z); r.w = fmaf(b.w, gw, r.w);
        }
        const float sq = r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w;
        ss += sq;
        // Retain the unscaled product in the existing scratch instead of reloading R
        // and recomputing the pending write after the reduction.
        const float4 g = *reinterpret_cast<const float4*>(a.w_norm + i);
        *reinterpret_cast<float4*>(xn + i) = make_float4(r.x * g.x, r.y * g.y, r.z * g.z, r.w * g.w);
    }
    const float v = warp_sum(ss);
    if (lane == 0) part[warp] = v;
    __syncthreads();
    if (t == 0) {
        float s = 0.0f;
        for (int w = 0; w < WARPS; ++w) s += part[w];
        s_rs = rsqrtf(s / (float) N + a.eps);
        a.rs[c] = s_rs;
    }
    __syncthreads();
    const float rs = s_rs;
    for (int d = t; d < N; d += THREADS) xn[c * N + d] *= rs;
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define STRATA_GR_CP_ASYNC 1
#endif
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
#if defined(STRATA_GR_CP_ASYNC)
    const unsigned sa = (unsigned) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(sa), "l"(gmem) : "memory");
#else
    *reinterpret_cast<float4*>(smem) = *reinterpret_cast<const float4*>(gmem);
#endif
}
__device__ __forceinline__ void cp_async_commit() {
#if defined(STRATA_GR_CP_ASYNC)
    asm volatile("cp.async.commit_group;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void cp_async_wait1() {
#if defined(STRATA_GR_CP_ASYNC)
    asm volatile("cp.async.wait_group 1;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void cp_async_wait0() {
#if defined(STRATA_GR_CP_ASYNC)
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
#endif
}

// Stage tile `h` of every token into `buf`: [T][2 planes][160 chunks] float4, plane 0 = floats 0-3 of a chunk.
__device__ __forceinline__ void stage_htile(const GrMulti& m, int T, int h, float4* buf, int t) {
    for (int i = t; i < T * (H_TILE / 4); i += THREADS) {
        const int k = i / (H_TILE / 4), s4 = i - k * (H_TILE / 4);   // s4: float4 of the tile, chunk s4/2, half s4&1
        const float* src = m.xn + (size_t) k * D + (size_t) h * H_TILE + (size_t) s4 * 4;
        cp_async16(buf + (size_t) k * (H_TILE / 4) + (s4 & 1) * (H_TILE / 8) + (s4 >> 1), src);
    }
}

template <int MAX_T = kFusedGrMaxT, bool EXACT_T = false>
__device__ __forceinline__ void gr_down_staged_kernel(const GrMulti& m) {
    extern __shared__ __align__(16) float4 hbuf[];      // 2 buffers x [T][2][160] float4
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int T = EXACT_T ? MAX_T : m.T;
    const bool inject_block = blockIdx.x == DOWN_BLOCKS;
    const int row = inject_block ? warp : blockIdx.x * WARPS + warp;
    const bool active = !(inject_block && (m.a[0].w_inject == nullptr || warp >= HC));
    const uint16_t* wrow = (inject_block ? m.a[0].w_inject : m.a[0].w_down) + (size_t) (active ? row : 0) * D;
    const uint4* w4 = reinterpret_cast<const uint4*>(wrow);
    const size_t buf_f4 = (size_t) T * (H_TILE / 4);    // float4 per buffer (the second one follows the first)
    float acc[MAX_T];
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) acc[k] = 0.0f;
    uint4 wv[HQ], wnext[HQ];
    if (active) {
#pragma unroll
        for (int q = 0; q < HQ; ++q) wv[q] = __ldg(w4 + lane + 32 * q);
    }
    stage_htile(m, T, 0, hbuf, t);
    cp_async_commit();
    stage_htile(m, T, 1, hbuf + buf_f4, t);
    cp_async_commit();
#pragma unroll 1
    for (int h = 0; h < N_HTILES; ++h) {
        if (active && h + 1 < N_HTILES) {
#pragma unroll
            for (int q = 0; q < HQ; ++q) wnext[q] = __ldg(w4 + (h + 1) * (H_TILE / 8) + lane + 32 * q);
        }
        if (h + 1 < N_HTILES) cp_async_wait1();         // tile h has landed (h + 1 may still be on its way)
        else cp_async_wait0();
        __syncthreads();
        const float4* cur = hbuf + (h & 1) * buf_f4;
        if (active) {
#pragma unroll
            for (int q = 0; q < HQ; ++q) {
                const int j = lane + 32 * q;
                const Bf16x8 wvq = unpack8(wv[q]);
#pragma unroll
                for (int k = 0; k < MAX_T; ++k) {
                    if (EXACT_T || k < T) {
                        const float4* pk = cur + (size_t) k * (H_TILE / 4);
                        acc[k] += dot8u(wvq, pk[j], pk[H_TILE / 8 + j]);
                    }
                }
            }
        }
        __syncthreads();                                // every warp is done with buffer h & 1
        if (h + 2 < N_HTILES) stage_htile(m, T, h + 2, hbuf + (h & 1) * buf_f4, t);
        cp_async_commit();                              // an empty group at the end keeps the wait counts simple
        if (h + 1 < N_HTILES) {
#pragma unroll
            for (int q = 0; q < HQ; ++q) wv[q] = wnext[q];
        }
    }
    if (!active) return;
    float s[MAX_T];
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) s[k] = (EXACT_T || k < T) ? warp_sum(acc[k]) : 0.0f;
    // lane k writes token k (every lane holds every sum after the xor reduction)
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) {
        if ((!EXACT_T && k >= T) || lane != k) continue;
        if (inject_block) {
            m.a[k].inject_out[row] = s[k];
        } else {
            const float x = s[k] / (float) HC;
            m.a[k].lo[row] = x / (1.0f + __expf(-x));
        }
    }
}

}  // namespace strata_gr

// ---- launch_q8_t's instances, a token count each: gr_down_q8_kernel<T> on (Q8_RG + 1, Q8_NKC) blocks, then
// gr_up_q8_kernel<T> on UPM_BLOCKS, THREADS threads each.
#define STRATA_GR_Q8(T)                                                                                                 \
    extern "C" __global__ void __launch_bounds__(strata_gr::THREADS) strata_gr_down_q8_t##T(                           \
            const strata_gr::GrMulti m, float * __restrict__ part, float * __restrict__ ssg) {                         \
        strata_gr::gr_down_q8_kernel<T>(m, part, ssg);                                                                 \
    }                                                                                                                  \
    extern "C" __global__ void __launch_bounds__(strata_gr::THREADS) strata_gr_up_q8_t##T(                             \
            const strata_gr::GrMulti m, const float * __restrict__ part, const float * __restrict__ ssg) {             \
        strata_gr::gr_up_q8_kernel<T>(m, part, ssg);                                                                   \
    }
STRATA_GR_Q8(1)
STRATA_GR_Q8(2)
STRATA_GR_Q8(3)
STRATA_GR_Q8(4)
STRATA_GR_Q8(5)
STRATA_GR_Q8(6)
STRATA_GR_Q8(7)
STRATA_GR_Q8(8)
#undef STRATA_GR_Q8

// ---- launch_multi's instances at kHcStaged: gr_norm_split_kernel on (T, HC) blocks, gr_down_staged_kernel<T, true> on
// DOWN_BLOCKS + 1 blocks with 2 * T * H_TILE floats of dynamic shared memory, gr_up_multi_kernel<T, true> on UPM_BLOCKS;
// THREADS threads each.
extern "C" __global__ void __launch_bounds__(strata_gr::THREADS) strata_gr_norm_split(const strata_gr::GrMulti m) {
    strata_gr::gr_norm_split_kernel(m);
}
#define STRATA_GR_STAGED(T)                                                                                             \
    extern "C" __global__ void __launch_bounds__(strata_gr::THREADS) strata_gr_down_staged_t##T(                       \
            const strata_gr::GrMulti m) {                                                                              \
        strata_gr::gr_down_staged_kernel<T, true>(m);                                                                  \
    }                                                                                                                  \
    extern "C" __global__ void __launch_bounds__(strata_gr::THREADS) strata_gr_up_multi_t##T(const strata_gr::GrMulti m) { \
        strata_gr::gr_up_multi_kernel<T, true>(m);                                                                     \
    }
STRATA_GR_STAGED(1)
STRATA_GR_STAGED(2)
STRATA_GR_STAGED(3)
STRATA_GR_STAGED(4)
STRATA_GR_STAGED(5)
STRATA_GR_STAGED(6)
#undef STRATA_GR_STAGED

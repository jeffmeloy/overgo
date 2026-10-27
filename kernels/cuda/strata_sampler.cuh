// strata_sampler.cuh - Strata's sampler chain for its decode round's sampled windows (strata-port/decode), ported
// literally: penalties -> top_k -> top_p -> min_p -> temperature -> pick, the pick one Philox 4x32-10 draw keyed by the
// seed and the row's position.
//
// Ported from Strata (https://github.com/Niko1221/Strata, commit 296b59d2), MIT License, Copyright (c) 2026 Niko1221
// and the Strata contributors (licenses/Strata-MIT.txt), from src/kernels/cuda/sampler.cu (philox4x32_round,
// philox_uniform, gumbel_exp, history_count, apply_penalties, sampled_k, take_first, warp_first, sampled_tail_warp,
// warp_merge_lists, sampler_split_part_kernel, sampler_split_merge_kernel) and include/strata/kernels/sampler.hpp
// (SamplerParams), the chain llama.cpp's default order (common/common.h at 3cf03257).
//
// Strata compiles sampler.cu WITHOUT fast math, as this module is compiled. The sampled path is the split top_k, the
// default (STRATA_OLD_SAMPLER and STRATA_SAMPLER_ONE_BLOCK unset) and what a verify window's host-side sampling takes
// (its stream is not capturing); the host side (sample_tokens) is translated in
// internal/cuda/executor/launch_strata_sampler.go.

namespace strata_sampler {

struct SamplerParams {
    int top_k = 20;              // sampled path: 1..64 as given; 0 (off) or more than 64 keep the widest list, 64
    float top_p = 0.95f;         // 1.0 disables the filter
    float min_p = 0.0f;          // 0 disables; keeps tokens with p >= min_p * p_max (a prefix of the top_k list)
    float temperature = 1.0f;    // <= 0 means greedy
    int min_keep = 1;            // top_p keeps at least this many
    int penalty_last_n = 0;      // 0 disables; the window over `history` to count occurrences in
    float penalty_repeat = 1.0f;
    float penalty_freq = 0.0f;
    float penalty_present = 0.0f;
    uint64_t seed = 0;           // drives Philox, which is counter-based on (seed, token index)
    uint64_t counter = 0;        // absolute draw index of row 0; advance across decode calls
    bool greedy = false;
    const float* logit_bias = nullptr; // optional device array [n_vocab], added BEFORE penalties/filters; -inf bans
    bool gumbel = false;         // STRATA_SPEC_GUMBEL=1: Gumbel-max pick keyed by (seed, counter, token id) - see sampler.cu
};

// Philox 4x32-10, the counter-based generator the phase asks for.  Counter-based matters because it makes the
// stream a function of (seed, position) rather than of how many draws came before - so a batch can be sampled
// in any order and a run is reproducible.
__device__ __forceinline__ uint32_t philox4x32_round(uint32_t& c0, uint32_t& c1, uint32_t& c2, uint32_t& c3,
                                                     uint32_t k0, uint32_t k1) {
    const uint32_t hi0 = __umulhi(0x9E3779B9u, c0);
    const uint32_t hi1 = __umulhi(0xBB67AE85u, c2);
    const uint32_t lo0 = 0x9E3779B9u * c0;
    const uint32_t lo1 = 0xBB67AE85u * c2;
    const uint32_t n0 = hi1 ^ c1 ^ k0;
    const uint32_t n1 = lo1;
    const uint32_t n2 = hi0 ^ c3 ^ k1;
    const uint32_t n3 = lo0;
    c0 = n0; c1 = n1; c2 = n2; c3 = n3;
    return 0;
}

__device__ __forceinline__ float philox_uniform(uint64_t seed, uint64_t counter) {
    uint32_t c0 = (uint32_t) counter, c1 = (uint32_t) (counter >> 32);
    uint32_t c2 = (uint32_t) seed, c3 = (uint32_t) (seed >> 32);
    for (int i = 0; i < 10; ++i) {
        philox4x32_round(c0, c1, c2, c3, (uint32_t) i, 0u);
    }
    // 24 bits of mantissa, so the value is uniform in [0,1) with no rounding to 1.0
    return (float) (c0 >> 8) * (1.0f / 16777216.0f);
}

// GUMBEL-MAX COUPLING (STRATA_SPEC_GUMBEL=1, off as benched): the pick is argmax_i p_i / E_i with E_i = -log(u_i) and
// u_i a hash of (seed, counter, TOKEN ID), an exact sample of p.
__device__ __forceinline__ double gumbel_exp(uint64_t seed, uint64_t counter, uint32_t token) {
    uint64_t z = seed ^ (counter * 0x9E3779B97F4A7C15ull) ^ ((uint64_t) token * 0xD1B54A32D192ED03ull);
    z += 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z ^= z >> 31;
    const double u = ((double) (z >> 11) + 0.5) * (1.0 / 9007199254740992.0);   // (0, 1), never 0 or 1
    return -log(u);
}

// `count_in_history` and the penalty application, transcribed from `llama_sampler_penalties_apply`.
// The repeat penalty MULTIPLIES for non-positive logits and DIVIDES for positive ones - dividing
// unconditionally is the natural reading of the source paper and it INVERTS the penalty on half the
// vocabulary.  The presence penalty is `float(count > 0)`, a boolean, not the count.
__device__ __forceinline__ int history_count(const int* __restrict__ h, int n, int v) {
    int c = 0;
    for (int i = 0; i < n; ++i) if (h[i] == v) ++c;
    return c;
}

__device__ __forceinline__ float apply_penalties(float logit, int count, const SamplerParams& p) {
    if (count <= 0) return logit;
    if (logit <= 0.0f) logit *= p.penalty_repeat;
    else               logit /= p.penalty_repeat;
    logit -= (float) count * p.penalty_freq + (count > 0 ? 1.0f : 0.0f) * p.penalty_present;
    return logit;
}

constexpr int kSelMax = 64;               // the widest top_k list, `sampler_kernel`'s KMAX
constexpr unsigned kFullMask = 0xFFFFFFFFu;

// top_k 1..64 as given; 0 ("off") and anything wider keep 64; never more than the vocabulary
__host__ __device__ __forceinline__ int sampled_k(int top_k, int n_vocab) {
    int k = (top_k > 0 && top_k < kSelMax) ? top_k : kSelMax;
    return k > n_vocab ? n_vocab : k;
}

// (bv, bi) <- the first of (bv, bi) and (ov, oi) in the selection order
__device__ __forceinline__ void take_first(float& bv, int& bi, float ov, int oi) {
    if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
}

// The first of the warp's 32 candidates, left in EVERY lane.  An XOR butterfly is exact here: "the first of two" is
// associative and commutative in a strict total order (two lanes never hold different candidates that compare
// equal - an id is in one lane at most, and two sentinels are the same pair), so every lane ends with the same pair.
__device__ __forceinline__ void warp_first(float& bv, int& bi) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const float ov = __shfl_xor_sync(kFullMask, bv, off);
        const int oi = __shfl_xor_sync(kFullMask, bi, off);
        take_first(bv, bi, ov, oi);
    }
}

/// **THE TAIL ON ONE WARP, WITH `sampler_kernel`'S ARITHMETIC.**  `sel_ids` / `sel_logit` (shared, `k` entries) are
/// the top_k list in selection order; the 32 lanes of one warp call this and lane 0 writes `out[t]`.  The lanes share
/// the `exp`s (one per entry, into `ex`) and then the quotients, and lane 0 alone runs the two ORDERED sums and the two
/// cumulative scans.
__device__ void sampled_tail_warp(const int* sel_ids, const float* sel_logit, int k, const SamplerParams& p, int t,
                                  int* __restrict__ out, double* ex) {
    const int lane = (int) (threadIdx.x & 31);
    const float inv_t = p.temperature > 0.0f ? 1.0f / p.temperature : 0.0f;
    int n_keep = k;
    float mx = sel_logit[0];
    for (int i = 1; i < k; ++i) mx = fmaxf(mx, sel_logit[i]);
    if (p.top_p < 1.0f) {
        for (int i = lane; i < k; i += 32) ex[i] = exp((double) sel_logit[i] - (double) mx);
        __syncwarp();
        double sum = 0.0;
        if (lane == 0)
            for (int i = 0; i < k; ++i) sum += ex[i];
        sum = __shfl_sync(kFullMask, sum, 0);
        __syncwarp();                            // lane 0 has read every `ex` before it is overwritten
        for (int i = lane; i < k; i += 32) ex[i] = ex[i] / sum;
        __syncwarp();
        int cut = k;
        if (lane == 0) {
            double cum = 0.0;
            for (int i = 0; i < k; ++i) {
                cum += ex[i];
                if (cum >= (double) p.top_p) { cut = i + 1; break; }
            }
        }
        cut = __shfl_sync(kFullMask, cut, 0);
        if (cut < p.min_keep) cut = p.min_keep < k ? p.min_keep : k;
        n_keep = cut;
        __syncwarp();                            // `ex` is written again below
    }
    // min_p on top_p's survivors, as in `sampler_kernel` (every lane, the same float arithmetic)
    if (p.min_p > 0.0f) {
        const float thresh = sel_logit[0] + logf(p.min_p);
        for (int i = 0; i < n_keep; ++i)
            if (sel_logit[i] < thresh) { n_keep = i; break; }
    }
    // temperature, then one Philox draw
    float smx = sel_logit[0] * inv_t;
    for (int i = 1; i < n_keep; ++i) smx = fmaxf(smx, sel_logit[i] * inv_t);
    for (int i = lane; i < n_keep; i += 32) ex[i] = exp((double) (sel_logit[i] * inv_t) - (double) smx);
    __syncwarp();
    double sum = 0.0;
    if (lane == 0)
        for (int i = 0; i < n_keep; ++i) sum += ex[i];
    sum = __shfl_sync(kFullMask, sum, 0);
    __syncwarp();
    for (int i = lane; i < n_keep; i += 32) ex[i] = ex[i] / sum;
    __syncwarp();
    if (lane == 0) {
        int pi = n_keep > 0 ? n_keep - 1 : 0;
        int pick = sel_ids[pi];
        if (p.gumbel) {
            double best = -1.0;
            for (int i = 0; i < n_keep; ++i) {
                const int id = sel_ids[i];
                const double r = ex[i] / gumbel_exp(p.seed, p.counter + (uint64_t) t, (uint32_t) id);
                if (r > best) { best = r; pick = sel_ids[i]; pi = i; }
            }
        } else {
            const float u = philox_uniform(p.seed, p.counter + (uint64_t) t);
            double cum = 0.0;
            for (int i = 0; i < n_keep; ++i) {
                cum += ex[i];
                if ((double) u < cum) { pick = sel_ids[i]; pi = i; break; }
            }
        }
        out[t] = pick;
    }
}

// ---- THE SPLIT top_k: a warp keeps the top_k of 1,024 logits held in registers, four warps make a block of 4,096
// logits that merges their lists, and a row is 61 such blocks over the whole GPU; one warp per row then merges the 61
// lists and runs the tail.
constexpr int kSplitPerLane = 32;                               // logits per lane, in registers
constexpr int kSplitWarpSpan = 32 * kSplitPerLane;              // 1,024 logits per warp
constexpr int kSplitWarps = 4;
constexpr int kSplitBlockSpan = kSplitWarps * kSplitWarpSpan;   // 4,096 logits per block
constexpr int kSplitMaxBlocks = 64;                             // lists the merge holds: n_vocab <= 262,144

/// Merge `nl` (<= 64) lists of `k` candidates - list L at `lists[L * stride]`, each in the selection order and
/// padded with sentinels - into their first `k`: `sink(i, value, id)` runs in every lane for i = 0..k-1 with the
/// same pair.  Lane owns lists `lane` and `lane + 32`; a round takes the first of all heads and advances the list
/// it came from.  An id is in one list at most (the lists cover disjoint logits), so exactly one head matches.
template <typename Sink>
__device__ __forceinline__ void warp_merge_lists(const int2* lists, int nl, int stride, int k, int n_vocab,
                                                 Sink&& sink) {
    const int lane = (int) (threadIdx.x & 31);
    float hv[2];
    int hi[2], pos[2];
#pragma unroll
    for (int m = 0; m < 2; ++m) {
        const int L = lane + 32 * m;
        pos[m] = 0;
        hv[m] = __int_as_float(0xff800000);
        hi[m] = n_vocab;
        if (L < nl) {
            const int2 c = lists[(size_t) L * stride];
            hi[m] = c.x;
            hv[m] = __int_as_float(c.y);
        }
    }
    for (int i = 0; i < k; ++i) {
        float bv = hv[0];
        int bi = hi[0];
        take_first(bv, bi, hv[1], hi[1]);
        warp_first(bv, bi);
        sink(i, bv, bi);
        if (bi < n_vocab) {
#pragma unroll
            for (int m = 0; m < 2; ++m) {
                if (hi[m] != bi) continue;
                if (++pos[m] < k) {
                    const int2 c = lists[(size_t) (lane + 32 * m) * stride + pos[m]];
                    hi[m] = c.x;
                    hv[m] = __int_as_float(c.y);
                } else {
                    hi[m] = n_vocab;
                    hv[m] = __int_as_float(0xff800000);
                }
            }
        }
    }
}

/// **SPLIT STAGE 1: THE top_k OF EACH 4,096-LOGIT BLOCK.**  Grid (blocks per row, rows), 128 threads.  Each warp
/// loads its 1,024 penalised logits once into registers and runs `k` warp-argmax rounds over them; warp 0 then merges
/// the four warp lists into the block's list in `cand` (row t, block b, entry i at `(t * n_blocks + b) * k + i`, as
/// (id, value bits)).
__device__ __forceinline__ void sampler_split_part_kernel(const float* __restrict__ logits, int n_vocab,
                                                          const int* __restrict__ history, int history_len,
                                                          const SamplerParams& p, int k, int n_blocks,
                                                          int2* __restrict__ cand) {
    const int t = blockIdx.y;
    const float* l = logits + (size_t) t * n_vocab;
    const int warp = (int) (threadIdx.x >> 5), lane = (int) (threadIdx.x & 31);
    const int blo = (int) blockIdx.x * kSplitBlockSpan;

    const int* hrow = history ? history + (size_t) t * history_len : nullptr;
    int hlen = 0;
    if (hrow) {
        hlen = p.penalty_last_n < history_len ? p.penalty_last_n : history_len;
        if (hlen < 0) hlen = 0;
        hrow += history_len - hlen;          // the window is the TAIL
    }
    // The membership bitmap of THIS BLOCK'S 4,096 logits (512 bytes, not the vocabulary's 31 KB): the same test,
    // and a hit pays the same exact count over the whole window.
    __shared__ unsigned int bits[kSplitBlockSpan / 32];
    const bool use_bits = hrow != nullptr && hlen > 0;
    if (use_bits) {
        for (int w = threadIdx.x; w < kSplitBlockSpan / 32; w += blockDim.x) bits[w] = 0u;
        __syncthreads();
        for (int i = threadIdx.x; i < hlen; i += blockDim.x) {
            const int h = hrow[i];
            if (h >= 0 && h < n_vocab && h >= blo && h - blo < kSplitBlockSpan)
                atomicOr(&bits[(h - blo) >> 5], 1u << ((h - blo) & 31));
        }
        __syncthreads();
    }

    // This warp's logits, penalised: `apply_penalties` with a zero count returns the logit unchanged, so only the
    // bitmap's hits go through it.  Past the vocabulary: -inf, which no round picks.
    const int lo = blo + warp * kSplitWarpSpan;
    float s[kSplitPerLane];
#pragma unroll
    for (int j = 0; j < kSplitPerLane; ++j) {
        const int v = lo + 32 * j + lane;
        s[j] = v < n_vocab ? (p.logit_bias ? l[v] + p.logit_bias[v] : l[v]) : __int_as_float(0xff800000);
    }
    if (use_bits) {
#pragma unroll
        for (int j = 0; j < kSplitPerLane; ++j) {
            const int v = lo + 32 * j + lane, b = v - blo;
            if (v < n_vocab && (bits[b >> 5] & (1u << (b & 31))))
                s[j] = apply_penalties(s[j], history_count(hrow, hlen, v), p);
        }
    }

    __shared__ int2 wl[kSplitWarps][kSelMax];
    float prev_v = __int_as_float(0x7f800000);   // +inf and id -1: round 0 takes every logit
    int prev_i = -1;
    int i = 0;
    for (; i < k; ++i) {
        // two chains (even and odd j), each walked in ascending id with a strict `>`, so each keeps its first in
        // the order; `take_first` then orders the two
        float b0 = __int_as_float(0xff800000), b1 = __int_as_float(0xff800000);
        int i0 = n_vocab, i1 = n_vocab;
#pragma unroll
        for (int j = 0; j < kSplitPerLane; j += 2) {
            const int v0 = lo + 32 * j + lane, v1 = v0 + 32;
            const float x0 = s[j], x1 = s[j + 1];
            if ((x0 < prev_v || (x0 == prev_v && v0 > prev_i)) && x0 > b0) { b0 = x0; i0 = v0; }
            if ((x1 < prev_v || (x1 == prev_v && v1 > prev_i)) && x1 > b1) { b1 = x1; i1 = v1; }
        }
        take_first(b0, i0, b1, i1);
        warp_first(b0, i0);
        if (i0 >= n_vocab) break;                // the same in every lane: nothing left in these 1,024 logits
        if (lane == 0) wl[warp][i] = make_int2(i0, __float_as_int(b0));
        prev_v = b0;
        prev_i = i0;
    }
    for (int r = i + lane; r < k; r += 32) wl[warp][r] = make_int2(n_vocab, (int) 0xff800000u);   // sentinels
    __syncthreads();
    if (warp == 0) {
        int2* dst = cand + ((size_t) t * n_blocks + blockIdx.x) * k;
        warp_merge_lists(&wl[0][0], kSplitWarps, kSelMax, k, n_vocab, [&](int r, float v, int id) {
            if (lane == 0) dst[r] = make_int2(id, __float_as_int(v));
        });
    }
}

/// **SPLIT STAGE 2: ONE WARP PER ROW MERGES THE BLOCK LISTS, THEN RUNS THE TAIL.**  The row's lists (at most 64 x
/// 64 entries, 32 KB) are copied to shared memory, merged into the row's top_k list, and `sampled_tail_warp` picks.
__device__ __forceinline__ void sampler_split_merge_kernel(const int2* __restrict__ cand, int n_blocks, int n_vocab,
                                                           const SamplerParams& p, int k, int* __restrict__ out) {
    const int t = blockIdx.x;
    const int lane = (int) threadIdx.x;
    __shared__ int2 lists[kSplitMaxBlocks * kSelMax];
    __shared__ int sel_ids[kSelMax];
    __shared__ float sel_logit[kSelMax];
    __shared__ double ex[kSelMax];
    const int2* src = cand + (size_t) t * n_blocks * k;
    for (int e = lane; e < n_blocks * k; e += 32) lists[e] = src[e];
    __syncwarp();
    warp_merge_lists(lists, n_blocks, k, k, n_vocab, [&](int i, float v, int id) {
        if (lane == 0) { sel_ids[i] = id < n_vocab ? id : 0; sel_logit[i] = v; }
    });
    __syncwarp();
    sampled_tail_warp(sel_ids, sel_logit, k, p, t, out, ex);
}

}  // namespace strata_sampler

extern "C" __global__ void __launch_bounds__(strata_sampler::kSplitWarps * 32) strata_sampler_split_part(
        const float* __restrict__ logits, int n_vocab, const int* __restrict__ history, int history_len,
        const strata_sampler::SamplerParams p, int k, int n_blocks, int2* __restrict__ cand) {
    strata_sampler::sampler_split_part_kernel(logits, n_vocab, history, history_len, p, k, n_blocks, cand);
}
extern "C" __global__ void __launch_bounds__(32) strata_sampler_split_merge(
        const int2* __restrict__ cand, int n_blocks, int n_vocab, const strata_sampler::SamplerParams p, int k,
        int* __restrict__ out) {
    strata_sampler::sampler_split_merge_kernel(cand, n_blocks, n_vocab, p, k, out);
}

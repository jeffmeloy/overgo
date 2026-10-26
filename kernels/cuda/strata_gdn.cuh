// strata_gdn.cu - Strata's gated-delta verify kernels, ported literally over our state's layout.
//
// Ported from Strata (https://github.com/Niko1221/Strata, commit 296b59d2), MIT License,
// Copyright (c) 2026 Niko1221 and the Strata contributors; the license text is licenses/Strata-MIT.txt.
// Source: src/kernels/cuda/verify_kernels.cu (gdn_conv_l2_body, gdn_step_norm_multi_kernel, gdn_conv_commit_kernel,
// gdn_step_commit_body).
//
// This module compiles WITHOUT -use_fast_math, as Strata compiles verify_kernels.cu. Each kernel keeps Strata's body
// and launch geometry (internal/cuda/executor/launch_strata_gdn.go), its template instantiations extern "C" entries of
// their own. One thing differs: the recurrent state's layout. Strata holds a layer's state [key 128][value head][value
// 128]; ours, which the prompt read's scan writes, is [value head][value 128][key 128]. Each thread holds the state
// elements Strata's does, its 32 keys of one value column, read and written at our addresses (base and row stride).
//
// Layouts: the conv history [channel][3] (Strata's and ours); the qkv and h rows [token][channel], channels q (key
// heads x 128), k, then v (value heads x 128); gate and beta [token][value head]; z and y [token][value head][128].
// A kept count (n_keep) is read from device memory, so one captured graph serves every count.

#include <stdint.h>

// verify_kernels.cu S, RG, RPG: the state size, its row groups and a group's rows.
#define STRATA_GDN_S 128
#define STRATA_GDN_RG 4
#define STRATA_GDN_RPG (STRATA_GDN_S / STRATA_GDN_RG)

// ---- gdn_conv_l2_body / gdn_conv_l2_multi_kernel: token t = t_begin + blockIdx.y; the window of token t is
// [hist0, hist1, hist2, x_0, ..., x_t], its last four entries; the conv, its SiLU and, over each q and k head's 128
// channels, the L2 norm. `commit` (one token): the history then keeps it - [hist1, hist2, x_0], each thread its
// channel's, after reading them. Launch: grid (channels / 128, tokens), 128 threads.
extern "C" __global__ void __launch_bounds__(STRATA_GDN_S) strata_gdn_conv_l2_multi(
        float * hist, const float * __restrict__ qkv, const float * __restrict__ w, float * __restrict__ h,
        int C, int qk_heads, float eps, int t_begin, int commit) {
    __shared__ float part[STRATA_GDN_S / 32];
    const int bx = blockIdx.x, by = blockIdx.y;
    const int t = t_begin + by;
    const int c = bx * STRATA_GDN_S + threadIdx.x;
    float win[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int src = t + j;
        win[j] = src < 3 ? hist[c * 3 + src] : qkv[(size_t) (src - 3) * C + c];
    }
    const float v0 = win[0], v1 = win[1], v2 = win[2], x = qkv[(size_t) t * C + c];
    if (commit) {
        hist[c * 3] = v1;
        hist[c * 3 + 1] = v2;
        hist[c * 3 + 2] = x;
    }
    float sum = v0 * w[c * 4] + v1 * w[c * 4 + 1] + v2 * w[c * 4 + 2] + x * w[c * 4 + 3];
    float y = sum / (1.0f + __expf(-sum));
    if (bx < qk_heads) {
        float sq = y * y;
        for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
        if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = sq;
        __syncthreads();
        const float ss = part[0] + part[1] + part[2] + part[3];
        y *= rsqrtf(ss + eps);
    }
    h[(size_t) t * C + c] = y;
}

// ---- gdn_step_norm_multi_kernel<ALL_OUT, Q = false>: the recurrence of one value head (blockIdx.x) from the state over
// the window's tokens, each token's output normed (gamma, eps) and gated by sigmoid(z). ALL_OUT reads the state and
// leaves it; otherwise the recurrence runs over the kept count's tokens (all of them without one), outputs from
// t_out_begin, and with a kept count writes the state after them. Launch: grid value heads, block (128, 4).
template <bool ALL_OUT>
__device__ __forceinline__ void strata_gdn_step_norm_body(
        float * __restrict__ state, const float * __restrict__ hbuf, int C, const float * __restrict__ gate,
        const float * __restrict__ beta, const float * __restrict__ z, const float * __restrict__ gamma, float eps,
        float * __restrict__ y, int h_k, int h_v, int T, const int32_t * __restrict__ n_keep, int t_out_begin) {
    const int S = STRATA_GDN_S, RG = STRATA_GDN_RG, RPG = STRATA_GDN_RPG;
    __shared__ float sk[2][STRATA_GDN_S], sq[2][STRATA_GDN_S];
    __shared__ float red_kv[STRATA_GDN_RG][STRATA_GDN_S];
    __shared__ float red_o[STRATA_GDN_RG][STRATA_GDN_S];
    __shared__ float wsum[STRATA_GDN_S / 32];
    const int head = blockIdx.x;
    const int col = threadIdx.x;
    const int rg = threadIdx.y;
    const int qh = head % h_k;
    const int qk = S * h_k;
    const int value_dim = S * h_v;
    const int n = ALL_OUT ? T : (n_keep ? *n_keep : T);
    if (n > 0) {
        if (rg == 0) sk[0][col] = hbuf[qk + qh * S + col];
        else if (rg == 1 && (ALL_OUT || 0 >= t_out_begin)) sq[0][col] = hbuf[qh * S + col];
    }
    const float gam = (rg == 0) ? gamma[col] : 0.0f;
    float s[STRATA_GDN_RPG];
    // Our layout holds value column c's 128 keys contiguous at the head's state + c * 128, where a thread holds 32 keys
    // of one column: each row group's 32 keys of every column pass through shared memory, read as whole lines, and each
    // thread of the group takes its column's - the values Strata's thread holds, the traffic coalesced.
    __shared__ float tile[STRATA_GDN_S][STRATA_GDN_RPG + 1];
    float * head_state = state + (size_t) head * S * S;
    const int tid = rg * S + col;
    for (int group = 0; group < RG; ++group) {
        for (int index = tid; index < S * RPG; index += S * RG) {
            const int c = index / RPG, r = index % RPG;
            tile[c][r] = head_state[(size_t) c * S + group * RPG + r];
        }
        __syncthreads();
        if (rg == group) {
#pragma unroll
            for (int r = 0; r < RPG; ++r) s[r] = tile[col][r];
        }
        __syncthreads();
    }
    for (int t = 0; t < n; ++t) {
        const int cur = t & 1, nxt = (t + 1) & 1;
        const float * ht = hbuf + (size_t) t * C;
        const bool need_out = ALL_OUT || (t >= t_out_begin);
        __syncthreads();
        if (t + 1 < n) {
            const float * ht_next = ht + C;
            const bool need_next = ALL_OUT || (t + 1 >= t_out_begin);
            if (rg == 0) sk[nxt][col] = ht_next[qk + qh * S + col];
            else if (rg == 1 && need_next) sq[nxt][col] = ht_next[qh * S + col];
        }
        const float g = __expf(gate[(size_t) t * h_v + head]);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) kv = fmaf(s[r], sk[cur][rg * RPG + r], kv);
        red_kv[rg][col] = kv;
        __syncthreads();
        const float kv_col = red_kv[0][col] + red_kv[1][col] + red_kv[2][col] + red_kv[3][col];
        const float delta = (ht[2 * qk + head * S + col] - g * kv_col) * beta[(size_t) t * h_v + head];
        if (!need_out) {
#pragma unroll
            for (int r = 0; r < RPG; ++r) s[r] = fmaf(g, s[r], sk[cur][rg * RPG + r] * delta);
            continue;
        }
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            s[r] = fmaf(g, s[r], sk[cur][rg * RPG + r] * delta);
            o = fmaf(s[r], sq[cur][rg * RPG + r], o);
        }
        red_o[rg][col] = o;
        __syncthreads();
        float oc = 0.0f, sq_part = 0.0f;
        if (rg == 0) {
            oc = (red_o[0][col] + red_o[1][col] + red_o[2][col] + red_o[3][col]) * rsqrtf((float) S);
            sq_part = oc * oc;
        }
        for (int o2 = 16; o2 > 0; o2 >>= 1) sq_part += __shfl_xor_sync(0xffffffffu, sq_part, o2);
        if (rg == 0 && (col & 31) == 0) wsum[col >> 5] = sq_part;
        __syncthreads();
        if (rg == 0) {
            const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
            const float scale = rsqrtf(ss / (float) S + eps);
            const float zz = z[(size_t) t * value_dim + head * S + col];
            const float yv = oc * scale * gam * (1.0f / (1.0f + __expf(-zz)));
            y[(size_t) t * value_dim + head * S + col] = yv;
        }
    }
    if (!ALL_OUT && n_keep != nullptr && n > 0) {
        // The state goes back as it came: a row group's keys through shared memory, written as whole lines.
        for (int group = 0; group < RG; ++group) {
            __syncthreads();
            if (rg == group) {
#pragma unroll
                for (int r = 0; r < RPG; ++r) tile[col][r] = s[r];
            }
            __syncthreads();
            for (int index = tid; index < S * RPG; index += S * RG) {
                const int c = index / RPG, r = index % RPG;
                head_state[(size_t) c * S + group * RPG + r] = tile[c][r];
            }
        }
    }
}

extern "C" __global__ void __launch_bounds__(STRATA_GDN_S * STRATA_GDN_RG) strata_gdn_step_norm_multi_all(
        float * __restrict__ state, const float * __restrict__ hbuf, int C, const float * __restrict__ gate,
        const float * __restrict__ beta, const float * __restrict__ z, const float * __restrict__ gamma, float eps,
        float * __restrict__ y, int h_k, int h_v, int T) {
    strata_gdn_step_norm_body<true>(state, hbuf, C, gate, beta, z, gamma, eps, y, h_k, h_v, T, nullptr, 0);
}

extern "C" __global__ void __launch_bounds__(STRATA_GDN_S * STRATA_GDN_RG) strata_gdn_step_norm_multi_kept(
        float * __restrict__ state, const float * __restrict__ hbuf, int C, const float * __restrict__ gate,
        const float * __restrict__ beta, const float * __restrict__ z, const float * __restrict__ gamma, float eps,
        float * __restrict__ y, int h_k, int h_v, int T, const int32_t * __restrict__ n_keep, int t_out_begin) {
    strata_gdn_step_norm_body<false>(state, hbuf, C, gate, beta, z, gamma, eps, y, h_k, h_v, T, n_keep, t_out_begin);
}

// ---- gdn_conv_commit_kernel: the history keeps the last three of [hist(3) | x_0..x_{n-1}], n the kept count.
// Launch: grid ceil(channels / 256), 256 threads.
extern "C" __global__ void strata_gdn_conv_commit(float * __restrict__ hist, const float * __restrict__ qkv, int C,
                                                  const int32_t * __restrict__ n_keep) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const int n = *n_keep;
    if (n <= 0) return;
    float seq[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int src = n + j;
        seq[j] = src < 3 ? hist[c * 3 + src] : qkv[(size_t) (src - 3) * C + c];
    }
    hist[c * 3] = seq[0];
    hist[c * 3 + 1] = seq[1];
    hist[c * 3 + 2] = seq[2];
}

// ---- gdn_step_commit_kernel<DBG = false> / gdn_step_commit_body: the state replayed over the kept count's tokens,
// no outputs; each value head's 128 columns over 4 blocks of 32. Launch: grid (value heads, 4), block (32, 4).
extern "C" __global__ void __launch_bounds__(32 * STRATA_GDN_RG) strata_gdn_step_commit(
        float * __restrict__ state, const float * __restrict__ hbuf, int C, const float * __restrict__ gate,
        const float * __restrict__ beta, int h_k, int h_v, const int32_t * __restrict__ n_keep) {
    const int S = STRATA_GDN_S, RPG = STRATA_GDN_RPG;
    const int n = *n_keep;
    if (n <= 0) return;
    __shared__ float sk[STRATA_GDN_S];
    __shared__ float red[STRATA_GDN_RG][32];
    const int head = blockIdx.x;
    const int col_local = threadIdx.x;
    const int col = blockIdx.y * 32 + col_local;
    const int rg = threadIdx.y;
    const int tid = rg * 32 + col_local;
    const int qh = head % h_k;
    const int qk = S * h_k;
    float s[STRATA_GDN_RPG];
    // The block's 32 value columns' 128 keys pass through shared memory as whole lines, each thread taking its column's
    // 32 keys of its row group - the values Strata's thread holds, the traffic coalesced in our layout.
    __shared__ float tile[32][STRATA_GDN_S + 1];
    float * columns = state + ((size_t) head * S + blockIdx.y * 32) * S;
    for (int index = tid; index < 32 * S; index += 32 * STRATA_GDN_RG) {
        const int c = index / S, k = index % S;
        tile[c][k] = columns[(size_t) c * S + k];
    }
    __syncthreads();
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = tile[col_local][rg * RPG + r];
    for (int t = 0; t < n; ++t) {
        const float * ht = hbuf + (size_t) t * C;
        __syncthreads();
        sk[tid] = ht[qk + qh * S + tid];
        __syncthreads();
        const float g = __expf(gate[(size_t) t * h_v + head]);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) kv = fmaf(s[r], sk[rg * RPG + r], kv);
        red[rg][col_local] = kv;
        __syncthreads();
        const float kv_col = red[0][col_local] + red[1][col_local] + red[2][col_local] + red[3][col_local];
        const float delta = (ht[2 * qk + head * S + col] - g * kv_col) * beta[(size_t) t * h_v + head];
#pragma unroll
        for (int r = 0; r < RPG; ++r) s[r] = fmaf(g, s[r], sk[rg * RPG + r] * delta);
    }
    __syncthreads();
#pragma unroll
    for (int r = 0; r < RPG; ++r) tile[col_local][rg * RPG + r] = s[r];
    __syncthreads();
    for (int index = tid; index < 32 * S; index += 32 * STRATA_GDN_RG) {
        const int c = index / S, k = index % S;
        columns[(size_t) c * S + k] = tile[c][k];
    }
}

// ---- copy_indexed_kernel: dst's n values become src's from row index (read from device memory) on, rows of stride
// values; nothing when the index is negative. The commit's copy of a window's history after its kept tokens.
// Launch: min(ceil(n / 256), 64) blocks of 256 threads.
extern "C" __global__ void strata_copy_indexed(float * __restrict__ dst, const float * __restrict__ src, long long stride,
                                               const int32_t * __restrict__ index, long long n) {
    const int idx = *index;
    if (idx < 0) return;
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long) gridDim.x * blockDim.x)
        dst[i] = src[(size_t) idx * stride + i];
}

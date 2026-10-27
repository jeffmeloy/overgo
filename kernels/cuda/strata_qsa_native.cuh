// strata_qsa_native.cuh - Strata's native QSA indexer append, ported literally into the operation module.
//
// Ported from Strata (https://github.com/Niko1221/Strata, commit 296b59d2), MIT License, Copyright (c) 2026 Niko1221
// and the Strata contributors (licenses/Strata-MIT.txt), from src/kernels/cuda/native_qsa_indexer.cu, which Strata
// adapted from llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d (src/models/qwen4exp.cpp;
// ggml/src/ggml-cuda/{set-rows.cu,norm.cu,rope.cu}):
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
// Strata compiles the native_* files WITH --use_fast_math, so these kernels sit in the operation module, which
// compiles the same way, and not in strata_qsa.cu, which keeps Strata's IEEE files. The prompt path appends a chunk's
// indexer keys with the batch form (append_first, append_blocks, append_tail); the per-step form (append, n_steps
// cells at a position stride) is the sequential reference the batch is held bit-identical to, and the form a verify
// window appends with. Only Strata's TAB=false kernels are here: the float64 angle table is opt-in in Strata
// (STRATA_ROPE_TABLE=1). Keys are rounded through F16 as llama.cpp's SET_ROWS stores them; a completed block's key
// is the F32 mean of its four keys in cell order, RMS-normed, scaled by gamma and rotated at the block's first cell's
// position; the spare (cell 0) is the mean of four copies of cell 0 at angle 0.

#define STRATA_NATIVE_D 128
#define STRATA_NATIVE_R 4
#define STRATA_NATIVE_ROT 64
#define STRATA_NATIVE_THREADS 256

__device__ __forceinline__ float strata_native_warp_sum(float x) {
#pragma unroll
    for (int offset = 16; offset; offset >>= 1) x += __shfl_xor_sync(0xffffffffu, x, offset);
    return x;
}

__device__ __forceinline__ int strata_native_mrope_pos(const int32_t * tab, int pos, int pair) {
    return tab ? __ldg(tab + (size_t) pos * 3 + pair % 3) : pos;
}

// rope_scaling.hpp's rope_yarn_ramp and rope_scaled_angle.
__device__ __forceinline__ float strata_native_yarn_ramp(float low, float high, int pair) {
    const float y = ((float) pair - low) / (high - low > 0.001f ? high - low : 0.001f);
    const float clamped = y < 0.0f ? 0.0f : (y > 1.0f ? 1.0f : y);
    return 1.0f - clamped;
}
__device__ __forceinline__ void strata_native_scaled_angle(
        float theta_extrap, float freq_scale, float corr_low, float corr_high, float ext_factor, float mscale_in,
        int pair, float & cos_out, float & sin_out) {
    float theta = freq_scale * theta_extrap;
    float mscale = mscale_in;
    if (ext_factor != 0.0f) {
        const float ramp_mix = strata_native_yarn_ramp(corr_low, corr_high, pair) * ext_factor;
        theta = theta * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    cos_out = cosf(theta) * mscale;
    sin_out = sinf(theta) * mscale;
}

// append_body<false>: n_steps cells, cell i's position at pos_dev[i * pos_stride] (a position outside the capacity
// skips its cell).
__device__ __forceinline__ void strata_native_append_body(
        const float * __restrict__ raw, const int32_t * __restrict__ pos_dev, int pos_stride, int n_steps,
        int pos_base, const float * __restrict__ gamma, float epsilon, float * __restrict__ tail,
        float * __restrict__ dead, float * __restrict__ pooled, int32_t * __restrict__ block_pos, int max_cells,
        float theta_scale, float freq_scale, float corr_low, float corr_high, float ext_factor, float mscale,
        const int32_t * __restrict__ mtab) {
    __shared__ float values[STRATA_NATIVE_D];
    __shared__ float partials[32];
    const int d = threadIdx.x;
    for (int step_idx = 0; step_idx < n_steps; ++step_idx) {
        const int pos = pos_dev[(size_t) step_idx * pos_stride];
        if (pos >= 0 && pos < max_cells) {
            const float * raw_step = raw + (size_t) step_idx * STRATA_NATIVE_D;
            const int slot = pos % STRATA_NATIVE_R;
            float incoming = 0.0f;
            if (d < STRATA_NATIVE_D) {
                incoming = __half2float(__float2half_rn(raw_step[d]));
                if (slot < STRATA_NATIVE_R - 1) tail[slot * STRATA_NATIVE_D + d] = incoming;
            }
            if (pos == 0 || slot == STRATA_NATIVE_R - 1) {
                float mean = 0.0f;
                if (d < STRATA_NATIVE_D) {
                    float sum = pos == 0 ? incoming : tail[d];
#pragma unroll
                    for (int j = 1; j < STRATA_NATIVE_R; ++j)
                        sum = __fadd_rn(sum, pos == 0 || j == STRATA_NATIVE_R - 1 ? incoming : tail[j * STRATA_NATIVE_D + d]);
                    mean = __fmaf_rn(0.25f, sum, 0.0f);
                }
                float square_sum = 0.0f;
                if (d < STRATA_NATIVE_D) square_sum += mean * mean;
                square_sum = strata_native_warp_sum(square_sum);
                const int lane = d % 32;
                if (lane == 0) partials[d / 32] = square_sum;
                __syncthreads();
                square_sum = lane < STRATA_NATIVE_THREADS / 32 ? partials[lane] : 0.0f;
                square_sum = strata_native_warp_sum(square_sum);
                const float scale = rsqrtf(square_sum / STRATA_NATIVE_D + epsilon);
                if (d < STRATA_NATIVE_D) values[d] = scale * mean * gamma[d];
                __syncthreads();
                if (d < STRATA_NATIVE_D) {
                    const int b = pos / STRATA_NATIVE_R;
                    const int rope_pos = pos == 0 ? 0 : pos_base + STRATA_NATIVE_R * b;
                    float y = values[d];
                    if (d < STRATA_NATIVE_ROT) {
                        const int pair = d % (STRATA_NATIVE_ROT / 2);
                        float c, s;
                        const float theta_extrap = (pos == 0 ? 0 : strata_native_mrope_pos(mtab, rope_pos, pair)) * powf(theta_scale, float(pair));
                        strata_native_scaled_angle(theta_extrap, freq_scale, corr_low, corr_high, ext_factor, mscale, pair, c, s);
                        const float a = values[pair], z = values[pair + STRATA_NATIVE_ROT / 2];
                        y = d < STRATA_NATIVE_ROT / 2 ? a * c - z * s : a * s + z * c;
                    }
                    pooled[(size_t) b * STRATA_NATIVE_D + d] = y;
                    if (pos == 0) dead[d] = y;
                    else pooled[(size_t) (b + 1) * STRATA_NATIVE_D + d] = dead[d];
                    if (d == 0 && pos != 0) *block_pos = rope_pos;
                }
            }
        }
        if (n_steps > 1) __syncthreads();
    }
}

// append<false>: native_qsa_indexer_append (n_steps 1) and native_qsa_indexer_append_steps. Launch: 1 block, 256.
extern "C" __global__ void strata_native_indexer_append(
        const float * __restrict__ raw, const int32_t * __restrict__ pos_dev, int pos_stride, int n_steps,
        int pos_base, const float * __restrict__ gamma, float epsilon, float * __restrict__ tail,
        float * __restrict__ dead, float * __restrict__ pooled, int32_t * __restrict__ block_pos, int max_cells,
        float theta_scale, float freq_scale, float corr_low, float corr_high, float ext_factor, float mscale,
        const int32_t * __restrict__ mtab) {
    strata_native_append_body(raw, pos_dev, pos_stride, n_steps, pos_base, gamma, epsilon, tail, dead, pooled,
                              block_pos, max_cells, theta_scale, freq_scale, corr_low, corr_high, ext_factor,
                              mscale, mtab);
}

// QsaIndexerLayers (native_qsa_indexer.hpp): the layers append_layers takes, by value.
#define STRATA_NATIVE_INDEXER_LAYERS 16
struct strata_native_indexer_layers {
    const float * gamma[STRATA_NATIVE_INDEXER_LAYERS];
    const float * tail_src[STRATA_NATIVE_INDEXER_LAYERS];
    float * tail[STRATA_NATIVE_INDEXER_LAYERS];
    float * dead[STRATA_NATIVE_INDEXER_LAYERS];
    float * pooled[STRATA_NATIVE_INDEXER_LAYERS];
    int32_t * block_pos[STRATA_NATIVE_INDEXER_LAYERS];
};

// append_layers<false>: the commit's appends of several QSA layers in one launch, one block per layer (blockIdx.x);
// each block first restores its layer's tail from the window's snapshot (the commit's copy_from_mapped), then runs
// append's body. Launch: n_layers blocks, 256 threads.
extern "C" __global__ void strata_native_indexer_append_layers(
        const float * __restrict__ raw0, long long raw_stride, const int32_t * __restrict__ pos_dev, int pos_stride,
        int n_steps, int pos_base, float epsilon, strata_native_indexer_layers L, int max_cells, float theta_scale,
        float freq_scale, float corr_low, float corr_high, float ext_factor, float mscale,
        const int32_t * __restrict__ mtab) {
    const int i = blockIdx.x;
    float * tail = L.tail[i];
    if (L.tail_src[i] != nullptr) {
        for (int j = threadIdx.x; j < (STRATA_NATIVE_R - 1) * STRATA_NATIVE_D; j += blockDim.x) tail[j] = L.tail_src[i][j];
        __syncthreads();
    }
    strata_native_append_body(raw0 + (long long) i * raw_stride, pos_dev, pos_stride, n_steps, pos_base, L.gamma[i], epsilon,
                              tail, L.dead[i], L.pooled[i], L.block_pos[i], max_cells, theta_scale, freq_scale, corr_low,
                              corr_high, ext_factor, mscale, mtab);
}

__device__ __forceinline__ float strata_native_pooled_value(
        const float * values, int d, int rope_pos, float theta_scale, float freq_scale, float corr_low,
        float corr_high, float ext_factor, float mscale, const int32_t * mtab, bool zero_pos) {
    float y = values[d];
    if (d < STRATA_NATIVE_ROT) {
        const int pair = d % (STRATA_NATIVE_ROT / 2);
        float c, s;
        const float theta_extrap = (zero_pos ? 0 : strata_native_mrope_pos(mtab, rope_pos, pair)) * powf(theta_scale, float(pair));
        strata_native_scaled_angle(theta_extrap, freq_scale, corr_low, corr_high, ext_factor, mscale, pair, c, s);
        const float a = values[pair], z = values[pair + STRATA_NATIVE_ROT / 2];
        y = d < STRATA_NATIVE_ROT / 2 ? a * c - z * s : a * s + z * c;
    }
    return y;
}

__device__ __forceinline__ float strata_native_norm_scale(float mean, float * partials, int d) {
    float square_sum = d < STRATA_NATIVE_D ? mean * mean : 0.0f;
    square_sum = strata_native_warp_sum(square_sum);
    const int lane = d % 32;
    if (lane == 0) partials[d / 32] = square_sum;
    __syncthreads();
    square_sum = lane < STRATA_NATIVE_THREADS / 32 ? partials[lane] : 0.0f;
    square_sum = strata_native_warp_sum(square_sum);
    return square_sum;
}

// append_first<false>: cell 0 of a sequence, the spare (every gather index names cell 0), into pooled[0] and dead.
// Launch: 1 block, 256 threads.
extern "C" __global__ void strata_native_indexer_append_first(
        const float * __restrict__ raw, const float * __restrict__ gamma, float epsilon, float * __restrict__ dead,
        float * __restrict__ pooled, float theta_scale, float freq_scale, float corr_low, float corr_high,
        float ext_factor, float mscale, const int32_t * __restrict__ mtab) {
    const int d = threadIdx.x;
    __shared__ float values[STRATA_NATIVE_D];
    __shared__ float partials[32];
    float mean = 0.0f;
    if (d < STRATA_NATIVE_D) {
        const float incoming = __half2float(__float2half_rn(raw[d]));
        float sum = incoming;
#pragma unroll
        for (int j = 1; j < STRATA_NATIVE_R; ++j) sum = __fadd_rn(sum, incoming);
        mean = __fmaf_rn(0.25f, sum, 0.0f);
    }
    const float square_sum = strata_native_norm_scale(mean, partials, d);
    const float scale = rsqrtf(square_sum / STRATA_NATIVE_D + epsilon);
    if (d < STRATA_NATIVE_D) values[d] = scale * mean * gamma[d];
    __syncthreads();
    if (d >= STRATA_NATIVE_D) return;
    const float y = strata_native_pooled_value(values, d, 0, theta_scale, freq_scale, corr_low, corr_high, ext_factor,
                                               mscale, mtab, true);
    pooled[d] = y;
    dead[d] = y;
}

// append_blocks<false>: the pooled key of each block completed in a batch (its last cell 4b+3 in [p0, p0 + n)), a
// block a CTA, its keys this batch's rows (cell >= p0) or the tail the previous batch left; only the batch's last
// completed block writes the spare after it. Launch: last - first + 1 blocks, 256 threads.
extern "C" __global__ void strata_native_indexer_append_blocks(
        const float * __restrict__ raw, int64_t n, int64_t p0, int pos_base, const float * __restrict__ gamma,
        float epsilon, const float * __restrict__ tail, const float * __restrict__ dead, float * __restrict__ pooled,
        int32_t * __restrict__ block_pos, int64_t first_block, int64_t last_block, float theta_scale,
        float freq_scale, float corr_low, float corr_high, float ext_factor, float mscale,
        const int32_t * __restrict__ mtab) {
    const int64_t b = first_block + blockIdx.x;
    const int d = threadIdx.x;
    __shared__ float values[STRATA_NATIVE_D];
    __shared__ float partials[32];
    float mean = 0.0f;
    if (d < STRATA_NATIVE_D) {
        auto key = [&](int j) -> float {
            const int64_t cell = b * STRATA_NATIVE_R + j;
            return cell >= p0 ? __half2float(__float2half_rn(raw[(cell - p0) * STRATA_NATIVE_D + d])) : tail[j * STRATA_NATIVE_D + d];
        };
        float sum = key(0);
#pragma unroll
        for (int j = 1; j < STRATA_NATIVE_R; ++j) sum = __fadd_rn(sum, key(j));
        mean = __fmaf_rn(0.25f, sum, 0.0f);
    }
    const float square_sum = strata_native_norm_scale(mean, partials, d);
    const float scale = rsqrtf(square_sum / STRATA_NATIVE_D + epsilon);
    if (d < STRATA_NATIVE_D) values[d] = scale * mean * gamma[d];
    __syncthreads();
    if (d >= STRATA_NATIVE_D) return;
    const int rope_pos = pos_base + STRATA_NATIVE_R * (int) b;
    pooled[(size_t) b * STRATA_NATIVE_D + d] = strata_native_pooled_value(values, d, rope_pos, theta_scale, freq_scale,
                                                                          corr_low, corr_high, ext_factor, mscale, mtab, false);
    if (b == last_block) {
        pooled[(size_t) (b + 1) * STRATA_NATIVE_D + d] = dead[d];
        if (d == 0) *block_pos = rope_pos;
    }
}

// append_tail: after a batch, tail slot s holds the key of the batch's last cell with cell % 4 == s (s < 3), if any.
// Launch: 3 blocks, 128 threads.
extern "C" __global__ void strata_native_indexer_append_tail(
        const float * __restrict__ raw, int64_t n, int64_t p0, float * __restrict__ tail) {
    const int s = blockIdx.x, d = threadIdx.x;
    if (d >= STRATA_NATIVE_D) return;
    const int64_t last = p0 + n - 1;
    int64_t cell = last - ((last % STRATA_NATIVE_R) - s + STRATA_NATIVE_R) % STRATA_NATIVE_R;
    if (cell < p0) return;
    tail[s * STRATA_NATIVE_D + d] = __half2float(__float2half_rn(raw[(cell - p0) * STRATA_NATIVE_D + d]));
}

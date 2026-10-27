// strata_native_qsa.cuh - Strata's native QSA gate and fused per-head norm and rotation, ported literally for its
// decode round (strata-port/decode).
//
// Ported from Strata (https://github.com/Niko1221/Strata, commit 296b59d2), MIT License, Copyright (c) 2026 Niko1221
// and the Strata contributors (licenses/Strata-MIT.txt), from src/kernels/cuda/native_qsa.cu (gate, norm),
// native_rope.cu (norm_rope_kernel, apply), include/strata/kernels/mrope.hpp (mrope_pos) and rope_scaling.hpp
// (rope_yarn_ramp, rope_scaled_angle), which Strata adapted from llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d
// (ggml/src/ggml-cuda/{norm.cu,common.cuh,unary.cu,rope.cu}):
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
// Strata compiles native_qsa.cu and native_rope.cu WITH --use_fast_math, as this module is compiled. norm_rope_kernel
// is its TAB=false instance, the analytic angles (the float64 angle table, STRATA_ROPE_TABLE, is opt-in in Strata), so
// its RopeTab argument, which that instance never reads, is not passed. The host sides (native_qsa_gate_apply,
// native_qsa_rms_norm_rope) are translated in internal/cuda/executor/launch_strata_native_qsa.go.

namespace strata_native_qsa {

__device__ __forceinline__ int mrope_pos(const int32_t* tab, int pos, int pair) {
    return tab ? __ldg(tab + (size_t) pos * 3 + pair % 3) : pos;
}

// ggml/src/ggml-cuda/rope.cu `rope_yarn_ramp`: `1 - clamp(y, 0, 1)` with
// `y = (i0 / 2 - low) / max(0.001, high - low)`, by pair index.
__device__ __forceinline__ float rope_yarn_ramp(float low, float high, int pair) {
    const float y = ((float) pair - low) / (high - low > 0.001f ? high - low : 0.001f);
    const float clamped = y < 0.0f ? 0.0f : (y > 1.0f ? 1.0f : y);
    return 1.0f - clamped;
}

/// The scaled angle for one pair, float32, the ANALYTIC paths' contract (pinned ggml arithmetic).
__device__ __forceinline__ void rope_scaled_angle(float theta_extrap, float freq_scale, float corr_low,
                                                  float corr_high, float ext_factor, float mscale_in,
                                                  int pair, float& cos_out, float& sin_out) {
    // The interpolation, always applied; the correction mixed in on top (ggml-cuda/rope.cu rope_yarn).
    float theta = freq_scale * theta_extrap;
    float mscale = mscale_in;
    if (ext_factor != 0.0f) {
        const float ramp_mix = rope_yarn_ramp(corr_low, corr_high, pair) * ext_factor;
        theta = theta * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        // "Get n-d magnitude scaling corrected for interpolation" - the log term only when correcting.
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    cos_out = cosf(theta) * mscale;
    sin_out = sinf(theta) * mscale;
}

}  // namespace strata_native_qsa

extern "C" __global__ void strata_native_qsa_gate(const float* attn, const float* __restrict__ q_full, float* output,
                                                  int n_head, int head_dim) {
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= std::size_t(n_head) * head_dim) return;
    const std::size_t head = i / head_dim, channel = i % head_dim;
    const float raw = q_full[head * 2 * head_dim + head_dim + channel];
    const float sigmoid = 1.0f / (1.0f + expf(-raw));
    output[i] = attn[i] * sigmoid;
}

extern "C" __global__ void strata_native_norm_rope(const float* x, int in_stride, const float* __restrict__ gamma,
                                                   float* out, int n_cols, int n_rot, float epsilon, float theta_scale,
                                                   float freq_scale, float corr_low, float corr_high,
                                                   float ext_factor, float mscale, const int* positions,
                                                   const int32_t* mtab) {
    using namespace strata_native_qsa;
    const int tid = threadIdx.x;
    const int row = blockIdx.x;
    const float* in_row = x + size_t(row) * in_stride;
    float* out_row = out + size_t(row) * n_cols;
    float partial = 0.0f;
    for (size_t col = tid; col < size_t(n_cols); col += 256) {
        const float xv = in_row[col];
        partial += xv * xv;
    }
    const int half_rot = n_rot / 2;
    const float x_tid = tid < n_cols ? in_row[tid] : 0.0f;
    const float x_hi  = tid < half_rot ? in_row[tid + half_rot] : 0.0f;
    __shared__ float sums[32];
#pragma unroll
    for (int offset = 16; offset; offset >>= 1)
        partial += __shfl_xor_sync(0xffffffffu, partial, offset, 32);
    const int lane = tid % 32;
    if (lane == 0) sums[tid / 32] = partial;
    __syncthreads();
    partial = lane < 256 / 32 ? sums[lane] : 0.0f;
#pragma unroll
    for (int offset = 16; offset; offset >>= 1)
        partial += __shfl_xor_sync(0xffffffffu, partial, offset, 32);
    const float mean = partial / n_cols;
    const float scale = rsqrtf(mean + epsilon);
    if (tid >= n_rot && tid < n_cols) {
        out_row[tid] = (scale * x_tid) * gamma[tid];
    } else if (tid < half_rot) {
        const int pair = tid;
        float a = (scale * x_tid) * gamma[pair];
        float b = (scale * x_hi) * gamma[pair + half_rot];
        asm volatile("mov.f32 %0, %1;" : "=f"(a) : "f"(a));
        asm volatile("mov.f32 %0, %1;" : "=f"(b) : "f"(b));
        float c, s;
        const float theta_extrap = mrope_pos(mtab, positions[row], pair) * powf(theta_scale, float(pair));
        rope_scaled_angle(theta_extrap, freq_scale, corr_low, corr_high, ext_factor, mscale, pair, c, s);
        float bs = b * s;
        float bc = b * c;
        asm volatile("mov.f32 %0, %1;" : "=f"(bs) : "f"(bs));
        asm volatile("mov.f32 %0, %1;" : "=f"(bc) : "f"(bc));
        out_row[pair] = fmaf(a, c, -bs);
        out_row[pair + half_rot] = fmaf(a, s, bc);
    }
}

// native_qsa.cu's norm<BlockSize> (native_qsa_rms_norm_weighted: 256 threads below 1,024 columns, else 1,024): a row a
// block, its sum of squares reduced through two warp sums, the row scaled by rsqrt(mean + epsilon) and gamma.
template<int BlockSize>
__device__ __forceinline__ void strata_native_qsa_norm(const float* input, const float* __restrict__ gamma, float* output,
                                                       int n_cols, float epsilon) {
    const int tid = threadIdx.x;
    const std::size_t row_offset = std::size_t(blockIdx.x) * n_cols;
    input += row_offset; output += row_offset;
    float partial = 0.0f;
    for (std::size_t col = tid; col < std::size_t(n_cols); col += BlockSize) {
        const float x = input[col];
        partial += x * x;
    }
    __shared__ float sums[32];
#pragma unroll
    for (int offset = 16; offset; offset >>= 1)
        partial += __shfl_xor_sync(0xffffffffu, partial, offset, 32);
    const int lane = tid % 32;
    if (lane == 0) sums[tid / 32] = partial;
    // All reads of input for the reduction precede this barrier. Afterwards,
    // each thread reads/writes only its own elements, allowing exact in-place use.
    __syncthreads();
    partial = lane < BlockSize / 32 ? sums[lane] : 0.0f;
#pragma unroll
    for (int offset = 16; offset; offset >>= 1)
        partial += __shfl_xor_sync(0xffffffffu, partial, offset, 32);
    const float mean = partial / n_cols;
    const float scale = rsqrtf(mean + epsilon);
    for (std::size_t col = tid; col < std::size_t(n_cols); col += BlockSize)
        output[col] = scale * input[col] * gamma[col];
}
extern "C" __global__ void strata_native_qsa_norm_256(const float* input, const float* __restrict__ gamma, float* output,
                                                      int n_cols, float epsilon) {
    strata_native_qsa_norm<256>(input, gamma, output, n_cols, epsilon);
}
extern "C" __global__ void strata_native_qsa_norm_1024(const float* input, const float* __restrict__ gamma, float* output,
                                                       int n_cols, float epsilon) {
    strata_native_qsa_norm<1024>(input, gamma, output, n_cols, epsilon);
}

// native_rope.cu's apply<false> (native_rope_apply): a (pair, row) a thread, the rotated pairs by the analytic scaled
// angle, the rest copied where x is not out; 32 threads a block in place at a 64-wide rotation, else 128.
extern "C" __global__ void strata_native_rope_apply(const float* x, float* out, int rows, int width, int n_rot,
                                                    float theta_scale, float freq_scale, float corr_low, float corr_high,
                                                    float ext_factor, float mscale, const int* positions,
                                                    const int32_t* mtab) {
    using namespace strata_native_qsa;
    const int row = blockIdx.y;
    const int pair = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows || pair >= width / 2) return;
    const size_t start = size_t(row) * width;
    if (pair >= n_rot / 2) {
        if (x != out) {
            out[start + 2 * pair] = x[start + 2 * pair];
            out[start + 2 * pair + 1] = x[start + 2 * pair + 1];
        }
        return;
    }
    float c, s;
    const float theta_extrap = mrope_pos(mtab, positions[row], pair) * powf(theta_scale, float(pair));
    rope_scaled_angle(theta_extrap, freq_scale, corr_low, corr_high, ext_factor, mscale, pair, c, s);
    const float a = x[start + pair], b = x[start + pair + n_rot / 2];
    out[start + pair] = a * c - b * s;
    out[start + pair + n_rot / 2] = a * s + b * c;
}

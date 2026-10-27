// strata_native_ple.cuh - Strata's native lexical-injection (PLE) post-ops and weighted RMS norm, ported literally for
// its decode round (strata-port/decode).
//
// Ported from Strata (https://github.com/Niko1221/Strata, commit 296b59d2), MIT License, Copyright (c) 2026 Niko1221
// and the Strata contributors (licenses/Strata-MIT.txt), from src/kernels/cuda/native_ple_postops.cu (gate_kernel,
// broadcast_kernel, conv_residual_kernel) and native_gr_norm.cu (weighted_rms_norm), which Strata adapted from llama.cpp
// 3cf03257f219afbe7334045ff7c6a06ac68c627d (ggml/src/ggml-cuda/{norm.cu,sumrows.cu,ssm-conv.cu}):
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
// Strata compiles both files WITH --use_fast_math, as this module is compiled. These are the kernels of
// native_ple_postops (the per-token path the verify window takes at layer 1) and weighted_rms_norm at the block sizes
// native_gr_rms_norm_weighted_multi picks (256 below 1,024 columns, else 1,024). The host sides are translated in
// internal/cuda/executor/launch_strata_native_ple.go.

namespace strata_native_ple {

constexpr int N = 2560, H = 4, D = N * H, HISTORY = 9;

__device__ float warp_sum(float x) {
    for (int offset = 16; offset; offset >>= 1) x += __shfl_xor_sync(0xffffffffu, x, offset);
    return x;
}

__device__ __forceinline__ float norm_warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) value += __shfl_xor_sync(0xffffffffu, value, offset, 32);
    return value;
}

template <int BlockSize>
__device__ __forceinline__ void weighted_rms_norm(const float* __restrict__ input, const float* __restrict__ gamma,
                                                  float* __restrict__ output, int n_cols, float epsilon) {
    const int tid = threadIdx.x;
    const std::size_t row = std::size_t(blockIdx.y) * gridDim.x + blockIdx.x;
    const std::size_t row_offset = row * n_cols;
    input += row_offset;
    gamma += std::size_t(blockIdx.x) * n_cols;
    output += row_offset;
    float partial = 0.0f;
    for (int col = tid; col < n_cols; col += BlockSize) {
        const float value = input[col];
        partial += value * value;
    }

    // Pinned block_reduce<SUM,BlockSize>: every warp repeats the final XOR
    // reduction. There is no warp-0-only broadcast or downward-shuffle tree.
    __shared__ float sums[32];
    partial = norm_warp_sum(partial);
    const int lane = tid % 32;
    if (lane == 0) sums[tid / 32] = partial;
    __syncthreads();
    partial = 0.0f;
    if (lane < BlockSize / 32) partial = sums[lane];
    partial = norm_warp_sum(partial);

    const float mean = partial / n_cols;
    const float scale = rsqrtf(mean + epsilon);
    for (int col = tid; col < n_cols; col += BlockSize) {
        output[col] = scale * input[col] * gamma[col];
    }
}

}  // namespace strata_native_ple

extern "C" __global__ void strata_native_weighted_rms_norm_256(const float* __restrict__ input,
                                                               const float* __restrict__ gamma,
                                                               float* __restrict__ output, int n_cols, float epsilon) {
    strata_native_ple::weighted_rms_norm<256>(input, gamma, output, n_cols, epsilon);
}

extern "C" __global__ void strata_native_weighted_rms_norm_1024(const float* __restrict__ input,
                                                                const float* __restrict__ gamma,
                                                                float* __restrict__ output, int n_cols, float epsilon) {
    strata_native_ple::weighted_rms_norm<1024>(input, gamma, output, n_cols, epsilon);
}

extern "C" __global__ void strata_native_ple_gate(const float* key, const float* query, float* gate, float scale) {
    using namespace strata_native_ple;
    // SUM_ROWS selects 512 threads for four rows on the target GPU. Preserve
    // its eight partial lanes and materialized MUL rounding (never a dot FMA).
    float sums[8] = {};
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int d = threadIdx.x + j * 512;
        const float p = d < N ? __fmul_rn(key[blockIdx.x * N + d], query[blockIdx.x * N + d]) : 0.0f;
        sums[j] += p;
    }
    float sum = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) sum += sums[j];
    __shared__ float partials[32];
    sum = warp_sum(sum);
    const int lane = threadIdx.x % 32;
    if (!lane) partials[threadIdx.x / 32] = sum;
    __syncthreads();
    sum = lane < 16 ? partials[lane] : 0.0f;
    sum = warp_sum(sum);
    if (threadIdx.x == 0) {
        const float s = __fmaf_rn(scale, sum, 0.0f); // ggml SCALE's zero bias
        const float mag = sqrtf(fmaxf(fabsf(s), 1e-6f));
        const float sign = float((s > 0.0f) - (s < 0.0f));
        gate[blockIdx.x] = 1.0f / (1.0f + expf(-__fmul_rn(sign, mag)));
    }
}

extern "C" __global__ void strata_native_ple_broadcast(const float* value, const float* gate, float* gated) {
    using namespace strata_native_ple;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < D) gated[i] = __fmul_rn(value[i % N], gate[i / N]);
}

extern "C" __global__ void strata_native_ple_conv_residual(const float* history, const float* normalized,
                                                           const uint16_t* weights, const float* hidden,
                                                           const float* gated, float* conv, float* result) {
    using namespace strata_native_ple;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= D) return;
    float sum = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const float x = k == 3 ? normalized[c] : history[c * HISTORY + 3 * k];
        const float w = __half2float(__ushort_as_half(weights[c * 4 + k]));
        const float term = __fmul_rn(x, w);
        sum = k == 0 ? term : __fadd_rn(sum, term);
    }
    const float activation = sum / (1.0f + expf(-sum));
    conv[c] = activation;
    // Exact hidden/result alias is safe: each thread owns one element.
    result[c] = __fadd_rn(hidden[c], __fadd_rn(gated[c], activation));
}

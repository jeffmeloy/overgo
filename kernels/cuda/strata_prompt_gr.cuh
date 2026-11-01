// strata_prompt_gr.cuh - the hyper-connection read of Strata's prompt path, which its batched draft-layer pass runs over a
// batch of rows (Prefill::draft_kv, E-9): the row scale and the BF16 image of the normed rows, the low-rank SiLU, and
// the gated mix, the down and up products between them taken as BF16 GEMMs.
//
// Ported from Strata (https://github.com/Niko1221/Strata, commit 296b59d2), MIT License, Copyright (c) 2026 Niko1221
// and the Strata contributors (licenses/Strata-MIT.txt), from src/prefill/kernels.cu (warp_sum, bf, sigm, block_sum,
// gr_norm_rs_kernel, gr_silu_kernel and gr_mix_r_kernel, their CUDA forms without the BF16 second operands). The
// host side is internal/cuda/executor/strata_drafter_prompt_kv.go.

namespace strata_prompt_gr {

// The hyper-connection's geometry: the embedding, the streams, a row of every stream and the low rank.
constexpr int N = 2560, HC = 4, D = N * HC, LR = 320;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ uint16_t bf(float f) {
    uint32_t u = __float_as_uint(f);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}

__device__ __forceinline__ float sigm(float x) { return 1.0f / (1.0f + __expf(-x)); }

// block-wide sum for blockDim.x <= 1024, result broadcast
__device__ float block_sum(float v, float* sh) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) sh[w] = v;
    __syncthreads();
    const int nw = (blockDim.x + 31) >> 5;
    float t = (threadIdx.x < nw) ? sh[threadIdx.x] : 0.0f;
    if (w == 0) t = warp_sum(t);
    if (threadIdx.x == 0) sh[0] = t;
    __syncthreads();
    return sh[0];
}

// F-1: the row scale only (and the BF16 image); gr_mix_r_kernel recomputes r * rs * w itself, in the same order,
// so the FP32 copy of the normalized rows (T x 10240 floats) is neither written nor read
__device__ __forceinline__ void gr_norm_rs_kernel(const float* __restrict__ R, const float* __restrict__ w, float eps,
                                                  float* __restrict__ rs_out, uint16_t* __restrict__ xn16) {
    __shared__ float sh[32];
    const int64_t row = blockIdx.x;                 // t * 4 + c
    const int c = (int) (row % HC);
    const float* r = R + row * N;
    const int64_t xo = (row / HC) * D + (int64_t) c * N;   // the BF16 image's row
    float ss = 0.0f;
    for (int d = threadIdx.x; d < N; d += blockDim.x) ss += r[d] * r[d];
    const float rs = rsqrtf(block_sum(ss, sh) / (float) N + eps);
    if (threadIdx.x == 0) rs_out[row] = rs;
    for (int d = threadIdx.x; d < N; d += blockDim.x) xn16[xo + d] = bf(r[d] * rs * w[c * N + d]);
}

__device__ __forceinline__ void gr_silu_kernel(const float* __restrict__ lo, uint16_t* __restrict__ lo16, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float x = lo[i] / (float) HC;
    lo16[i] = bf(x / (1.0f + __expf(-x)));
}

__device__ __forceinline__ void gr_mix_r_kernel(const float* __restrict__ R, const float* __restrict__ rs,
                                                const float* __restrict__ w, const float* __restrict__ g,
                                                float* __restrict__ mixed, int64_t T) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * N) return;
    const int64_t t = i / N, d = i % N;
    float s = 0.0f;
#pragma unroll
    for (int c = 0; c < HC; ++c) {
        const int64_t j = t * D + c * N + d;
        const float x = R[j] * rs[t * HC + c] * w[c * N + d];   // gr_norm_kernel's value, bit for bit
        s = fmaf(x, sigm(g[j]), s);
    }
    mixed[i] = s / (float) HC;
}

}  // namespace strata_prompt_gr

extern "C" __global__ void __launch_bounds__(256) strata_prompt_gr_norm_rs(const float* __restrict__ R,
        const float* __restrict__ w, float eps, float* __restrict__ rs_out, uint16_t* __restrict__ xn16) {
    strata_prompt_gr::gr_norm_rs_kernel(R, w, eps, rs_out, xn16);
}
extern "C" __global__ void strata_prompt_gr_silu(const float* __restrict__ lo, uint16_t* __restrict__ lo16, long long n) {
    strata_prompt_gr::gr_silu_kernel(lo, lo16, n);
}
extern "C" __global__ void strata_prompt_gr_mix_r(const float* __restrict__ R, const float* __restrict__ rs,
        const float* __restrict__ w, const float* __restrict__ g, float* __restrict__ mixed, long long T) {
    strata_prompt_gr::gr_mix_r_kernel(R, rs, w, g, mixed, T);
}

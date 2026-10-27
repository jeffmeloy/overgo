// strata_native_gr_postops.cuh - Strata's native hyper-connection write, ported literally for its decode round
// (strata-port/decode).
//
// Ported from Strata (https://github.com/Niko1221/Strata, commit 296b59d2), MIT License, Copyright (c) 2026 Niko1221
// and the Strata contributors (licenses/Strata-MIT.txt), from src/kernels/cuda/native_gr_postops.cu (post, post_multi),
// compiled with --use_fast_math as Strata compiles it. The round writes a layer's residual through gr_write_multi, which
// takes native_gr_post_multi where --native sets gr_native_mmvf (the bench's configuration); the host side is
// translated in internal/cuda/executor/launch_strata_fused_gr.go.

namespace strata_native_gr {

constexpr int THREADS = 256;
__device__ __forceinline__ float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ float scale_zero_bias(float x, float scale) {
    return __fmaf_rn(scale, x, 0.0f);
}

}  // namespace strata_native_gr

extern "C" __global__ void strata_native_gr_post(const float* residual, const float* __restrict__ block_out,
                                                 const float* __restrict__ inject, float* output,
                                                 int n_embd, int hc, float scale) {
    using namespace strata_native_gr;
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= std::size_t(n_embd) * hc) return;
    const std::size_t t = blockIdx.y;
    residual += t * std::size_t(n_embd) * hc;
    block_out += t * std::size_t(n_embd);
    inject += t * std::size_t(hc);
    output += t * std::size_t(n_embd) * hc;
    const int c = int(i / n_embd), d = int(i % n_embd);
    const float weight = scale_zero_bias(sigmoid(scale_zero_bias(inject[c], scale)), 2.0f);
    // Exact residual/output alias is supported; no other thread reads residual[i].
    output[i] = __fmaf_rn(block_out[d], weight, residual[i]);
}

// strata_native_moe.cuh - Strata's routed-expert combine for its decode round (strata-port/decode), ported literally.
//
// Ported from Strata (https://github.com/Niko1221/Strata, commit 296b59d2), MIT License, Copyright (c) 2026 Niko1221
// and the Strata contributors (licenses/Strata-MIT.txt), from src/kernels/cuda/native_moe.cu (combine_k10_vec4), whose
// arithmetic Strata adapted from ggml's CUDA moe-weighted-reduction.cu at 3cf03257:
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
// native_moe_combine_multi takes this float4 form at ten experts with a shared row and 16-byte aligned rows, as
// Flash-Next's round always has; its scalar form is not ported. The host side is translated in
// internal/cuda/executor/launch_strata_native_moe.go.

// combine_k10_vec4: each token's ten expert parts weighted, the first product rounded and then one FMA an expert in
// order, plus the shared row.
extern "C" __global__ void strata_native_combine_k10_vec4(const float4* __restrict__ parts4,
                                                          const float* __restrict__ weights,
                                                          const float4* __restrict__ shared4,
                                                          float4* __restrict__ output4, int64_t n4) {
    const int64_t tk = blockIdx.y;
    parts4 += tk * 10 * n4;
    weights += tk * 10;
    shared4 += tk * n4;
    output4 += tk * n4;
    const int64_t c4 = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c4 >= n4) return;
    const float w0 = __ldg(weights + 0);
    const float4 p0 = parts4[c4];
    float4 sum = make_float4(p0.x * w0, p0.y * w0, p0.z * w0, p0.w * w0);
#pragma unroll
    for (int expert = 1; expert < 10; ++expert) {
        const float w = __ldg(weights + expert);
        const float4 p = parts4[int64_t(expert) * n4 + c4];
        // the documented contract spelled out: the first product rounded, then one FMA per expert in order. Left to the
        // compiler's contraction, a build may fuse a different product (gfx1151: the float4 kernel then differed from the
        // scalar one, native_multi_parity)
        sum.x = fmaf(p.x, w, sum.x);
        sum.y = fmaf(p.y, w, sum.y);
        sum.z = fmaf(p.z, w, sum.z);
        sum.w = fmaf(p.w, w, sum.w);
    }
    const float4 sh = shared4[c4];
    sum.x += sh.x;
    sum.y += sh.y;
    sum.z += sh.z;
    sum.w += sh.w;
    output4[c4] = sum;
}

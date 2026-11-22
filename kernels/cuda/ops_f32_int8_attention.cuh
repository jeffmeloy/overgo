// ops_f32_int8_attention.cuh: comfy_kitchen's int8 attention (comfy_kitchen/backends/cuda/sage_attention at v0.2.40,
// Copyright (c) 2025 Comfy Org and NVIDIA CORPORATION & AFFILIATES, Apache-2.0; its attention derived from
// SageAttention, https://github.com/thu-ml/SageAttention commit d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5, Copyright (c)
// 2024 by SageAttention team, Apache-2.0; licenses/comfy-kitchen-Apache-2.0.txt), as ComfyUI's MOSS runs its blocks'
// attention on this card (sm_89): heads of 128 channels, no mask, keys in 128-key tiles. ops_f32.cu includes it once.
//
// Four launches, as sage_sdpa makes them: int8_attention_anchor_f32 (detect_k_anchor) picks each key head's anchor
// key, or none; int8_attention_quantize_qk_f32 (quant_qk_fused) rotates each query and key row by a signed H128,
// keys first less their anchor, and quantizes it to INT8, a scale per eight-row group of a warp's 32 query rows and
// per two-row group of its 32 key rows; int8_attention_quantize_v_f32 (quant_v_int8_kernel, 512 threads, or its
// _short_ form at 128 threads for 256 keys or fewer) quantizes V to INT8 per channel, transposed and its keys permuted
// for the value product; int8_attention_bf16_f32 (qk_int_sv_i8_attn_kernel) computes Q.K on INT8 tensor cores, the
// softmax online in F32 with its probabilities rounded to U8, and P.V on U8 x INT8 tensor cores.
//
// Modifications from comfy_kitchen's source: the templates are fixed at the configuration this path takes
// (quant_qk_fused<NR 4, NL 16, BLKQ 128, WARPQ 32, BLKK 128, WARPK 128, one channel tile, ROTATION 128, aligned>,
// quant_v_int8_kernel at 128 and 512 threads, qk_int_sv_i8_attn_kernel<CTA_Q 128, CTA_K 128, WARP_Q 16, WARP_K 128, head
// 128, int8, per-thread scales, F32 accumulation, BF16 output, the denominator on CUDA cores, no mask, V's scale fused,
// the F32 probabilities fused> with its Ada query cache), so their dead branches are gone; the inputs are F32 rounded
// to BF16 as they load, the values comfy_kitchen reads from its BF16 tensors; the output's BF16 values are written as
// F32; and the names are in a namespace of their own.

namespace kitchen_attention {

// ---- math.cuh, cp_async.cuh, mma.cuh, permuted_smem.cuh (SageAttention) ----

constexpr float log2e = 1.44269504088896340736f;
// S_U8_OFFSET: the online maximum's shift so exp2(score - maximum) spans the whole U8 range.
constexpr float s_u8_offset = 7.9943534f;

__forceinline__ __device__ float ptx_exp2(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__forceinline__ __device__ float ptx_rcp(float x) {
    float y;
    asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

// i2f_exact: int32 to F32, exact for |x| <= 2^22; sm_89 converts on the FMA pipe.
__forceinline__ __device__ float i2f_exact(int32_t x) {
    return __int2float_rn(x);
}

__device__ __forceinline__ void commit_group() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <size_t n> __device__ __forceinline__ void wait_group() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(n));
}

__device__ __forceinline__ void load_128b(uint4 * smem_ptr, const uint4 * gmem_ptr) {
    const uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2, %3;\n" ::"r"(smem_int_ptr), "l"(gmem_ptr), "n"(16), "r"(16));
}

// pred_load_128b without zero fill: a false predicate leaves the shared bytes as they were.
__device__ __forceinline__ void pred_load_128b(uint4 * smem_ptr, const uint4 * gmem_ptr, bool predicate) {
    const uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("{\n"
                 " .reg .pred p;\n"
                 " setp.ne.b32 p, %0, 0;\n"
                 " @p cp.async.cg.shared.global.L2::128B [%1], [%2], %3;\n"
                 "}\n" ::"r"((int)predicate),
                 "r"(smem_int_ptr), "l"(gmem_ptr), "n"(16));
}

__device__ __forceinline__ void ldmatrix_m8n8x4(uint32_t * R, const uint4 * smem_ptr) {
    const uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(R[0]), "=r"(R[1]), "=r"(R[2]), "=r"(R[3])
                 : "r"(smem_int_ptr));
}

// The m16n16k32 products, two m16n8k32 instructions each: signed INT8 Q.K and unsigned U8 P by signed INT8 V.
template <bool init>
__device__ __forceinline__ void mma_s8s8s32(int32_t * C, const uint32_t * A, const uint32_t * B) {
    if constexpr (!init) {
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                     : "=r"(C[0]), "=r"(C[1]), "=r"(C[2]), "=r"(C[3])
                     : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                     : "=r"(C[4]), "=r"(C[5]), "=r"(C[6]), "=r"(C[7])
                     : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]), "r"(C[4]), "r"(C[5]), "r"(C[6]), "r"(C[7]));
    } else {
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                     : "=r"(C[0]), "=r"(C[1]), "=r"(C[2]), "=r"(C[3])
                     : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(0), "r"(0), "r"(0), "r"(0));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                     : "=r"(C[4]), "=r"(C[5]), "=r"(C[6]), "=r"(C[7])
                     : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]), "r"(0), "r"(0), "r"(0), "r"(0));
    }
}

template <bool init>
__device__ __forceinline__ void mma_u8s8s32(int32_t * C, const uint32_t * A, const uint32_t * B) {
    if constexpr (!init) {
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                     : "=r"(C[0]), "=r"(C[1]), "=r"(C[2]), "=r"(C[3])
                     : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                     : "=r"(C[4]), "=r"(C[5]), "=r"(C[6]), "=r"(C[7])
                     : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]), "r"(C[4]), "r"(C[5]), "r"(C[6]), "r"(C[7]));
    } else {
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                     : "=r"(C[0]), "=r"(C[1]), "=r"(C[2]), "=r"(C[3])
                     : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(0), "r"(0), "r"(0), "r"(0));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                     : "=r"(C[4]), "=r"(C[5]), "=r"(C[6]), "=r"(C[7])
                     : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]), "r"(0), "r"(0), "r"(0), "r"(0));
    }
}

// pack_u8x4: four F32 values rounded and saturated into one U8 word, the first in the low byte.
__device__ __forceinline__ uint32_t pack_u8x4(float a, float b, float c, float d) {
    uint32_t qa, qb, qc, qd;
    asm volatile("cvt.rni.s32.f32 %0, %1;" : "=r"(qa) : "f"(a));
    asm volatile("cvt.rni.s32.f32 %0, %1;" : "=r"(qb) : "f"(b));
    asm volatile("cvt.rni.s32.f32 %0, %1;" : "=r"(qc) : "f"(c));
    asm volatile("cvt.rni.s32.f32 %0, %1;" : "=r"(qd) : "f"(d));
    uint32_t qdc, packed;
    asm volatile("cvt.pack.sat.u8.s32.b32 %0, %1, %2, 0;" : "=r"(qdc) : "r"(qd), "r"(qc));
    asm volatile("cvt.pack.sat.u8.s32.b32 %0, %1, %2, %3;" : "=r"(packed) : "r"(qb), "r"(qa), "r"(qdc));
    return packed;
}

// smem_t in its 128-byte swizzle: offsets count 16-byte cells, a row stride of them.
template <uint32_t stride> struct smem_t {
    uint4 * base;
    template <typename T> __device__ __forceinline__ smem_t(T * pointer) : base((uint4 *)pointer) {
        static_assert(stride % 8 == 0, "Stride must be multiple of 8 for 128B swizzle mode");
    }
    static __device__ __forceinline__ uint32_t get_permuted_offset(const uint32_t & i, const uint32_t & j) {
        return i * stride + (j ^ (i % 8));
    }
    template <uint32_t step_size> static __device__ __forceinline__ uint32_t advance_offset_by_column(const uint32_t & offset) {
        static_assert(step_size % 8 == 0, "Unsupported step size");
        return offset + step_size;
    }
    template <uint32_t step_size>
    static __device__ __forceinline__ uint32_t advance_offset_by_column(const uint32_t & offset, const uint32_t & step_idx) {
        static_assert(step_size == 2, "Unsupported step size");
        return (offset ^ (0x2 + (0x4 * (step_idx % 2 == 1)))) + (step_idx % 4 == 3) * 8;
    }
    template <uint32_t step_size> static __device__ __forceinline__ uint32_t advance_offset_by_row(const uint32_t & offset) {
        static_assert(step_size == 4 || step_size % 8 == 0, "Unsupported step size");
        if constexpr (step_size == 4) {
            return (offset ^ 0x4) + step_size * stride;
        } else {
            return offset + step_size * stride;
        }
    }
    __device__ __forceinline__ void ldmatrix_m8n8x4(const uint32_t & offset, uint32_t * R) const {
        kitchen_attention::ldmatrix_m8n8x4(R, base + offset);
    }
    template <typename T> __device__ __forceinline__ void load_128b_async(const uint32_t & offset, const T * gptr, bool predicate) const {
        pred_load_128b(base + offset, reinterpret_cast<const uint4 *>(gptr), predicate);
    }
    template <typename T> __device__ __forceinline__ void load_128b_async(const uint32_t & offset, const T * gptr) const {
        load_128b(base + offset, reinterpret_cast<const uint4 *>(gptr));
    }
};

// ---- float_utils.cuh (comfy_kitchen) ----

__device__ __forceinline__ float warp_reduce_fmax(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, off));
    return v;
}

__device__ __forceinline__ int8_t float_to_int8_rn(float value) {
    int32_t quantized;
    asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(quantized) : "f"(value));
    return static_cast<int8_t>(quantized);
}

__device__ __forceinline__ int8_t quant_int8_rcp(float v, float inv_scale) {
    return float_to_int8_rn(v * inv_scale);
}

__device__ __forceinline__ void store4_i8(int8_t * ptr, int8_t a, int8_t b, int8_t c, int8_t d) {
    *reinterpret_cast<int32_t *>(ptr) =
        (uint32_t)(uint8_t)a | ((uint32_t)(uint8_t)b << 8) | ((uint32_t)(uint8_t)c << 16) | ((uint32_t)(uint8_t)d << 24);
}

// bf16_value is an F32 value as BF16 holds it, the value comfy_kitchen reads from its BF16 tensor.
__device__ __forceinline__ float bf16_value(float value) {
    return __bfloat162float(__float2bfloat16_rn(value));
}

// load4: VectorLoader4's load of four channels, each rounded to BF16, as one vector where they start 16-byte aligned
// and one at a time where they do not: a head may be bound as a view of another tensor, aligned to 4 bytes alone,
// which a vector read faults on (fusion-edges-all).
__forceinline__ __device__ void load4(const float * ptr, float * out) {
    const float4 raw = (reinterpret_cast<size_t>(ptr) & (sizeof(float4) - 1)) == 0
        ? __ldg(reinterpret_cast<const float4 *>(ptr))
        : make_float4(__ldg(ptr), __ldg(ptr + 1), __ldg(ptr + 2), __ldg(ptr + 3));
    out[0] = bf16_value(raw.x);
    out[1] = bf16_value(raw.y);
    out[2] = bf16_value(raw.z);
    out[3] = bf16_value(raw.w);
}

// ---- quant_qk_int8.cu (comfy_kitchen) ----

__forceinline__ __device__ void convrot4(float * values) {
    const float x0 = values[0];
    const float x1 = values[1];
    const float x2 = values[2];
    const float x3 = values[3];
    const float a0 = x0 + x1;
    const float a1 = x0 - x1;
    const float a2 = x2 + x3;
    const float a3 = x2 - x3;
    values[0] = (a0 + a2) * 0.5f;
    values[1] = (a1 + a3) * 0.5f;
    values[2] = (a0 - a2) * 0.5f;
    values[3] = (a1 - a3) * 0.5f;
}

// A fixed random diagonal before the Hadamard; Q and K take the same signs, so their exact dot product is unchanged.
__forceinline__ __device__ void apply_convrot_sign128(float * values, const int lane) {
    constexpr uint32_t signs_0 = 0x1035997bu;
    constexpr uint32_t signs_1 = 0x8087f5eeu;
    constexpr uint32_t signs_2 = 0xee2e4e1au;
    constexpr uint32_t signs_3 = 0x71132418u;
    const uint32_t signs = lane < 8 ? signs_0 : lane < 16 ? signs_1 : lane < 24 ? signs_2 : signs_3;
    const int shift = (lane & 7) * 4;
#pragma unroll
    for (int channel = 0; channel < 4; ++channel) {
        const uint32_t flip = ((signs >> (shift + channel)) & 1u) ^ 1u;
        values[channel] = __uint_as_float(__float_as_uint(values[channel]) ^ (flip << 31));
    }
}

// A normalized Walsh-Hadamard H128 across a warp's four-channel groups.
__forceinline__ __device__ void convrot128_plain(float * values) {
    convrot4(values);
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int bit = 1; bit < 32; bit <<= 1) {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const float other = __shfl_xor_sync(0xffffffffu, values[c], bit);
            values[c] = (lane & bit) ? other - values[c] : values[c] + other;
        }
    }
#pragma unroll
    for (int c = 0; c < 4; ++c)
        values[c] *= 0.1767766952966369f;
}

__forceinline__ __device__ void convrot128(float * values) {
    apply_convrot_sign128(values, threadIdx.x & 31);
    convrot128_plain(values);
}

// The fixed configuration: 128 channels, a 128-row query block of four 32-row warp blocks, four rows of a warp a group
// (NR); a 128-key block, a warp's 32 keys in sixteen pairs (NL).
constexpr int quant_channels = 128;
constexpr int quant_nr = 4;
constexpr int quant_nl = 16;
constexpr int quant_blkq = 128;
constexpr int quant_warpq = 32;
constexpr int quant_warpk = 128;

#pragma nv_diag_suppress 1056
__forceinline__ __device__ void process_q(const float * __restrict__ in, int8_t * __restrict__ out, float * __restrict__ sc_buf,
                                          const int oblk, const int L, const int C, const int64_t stride_n) {
    constexpr int NSUB = quant_blkq / quant_warpq;
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
#pragma unroll
    for (int g = 0; g < 2; ++g) {
        const int otld = wid * 2 + g;
        const int base = (oblk / NSUB) * quant_blkq + (oblk % NSUB) * quant_warpq + otld;
        float v[quant_nr * 4];
        float mx = 0.f;
#pragma unroll
        for (int i = 0; i < quant_nr * 4; ++i)
            v[i] = 0.f;
        const int ch = lane << 2;
#pragma unroll
        for (int j = 0; j < quant_nr; ++j) {
            const int n = base + j * 8;
            const int vi = j * 4;
            if (n < L) {
                load4(&in[(int64_t)n * stride_n + ch], &v[vi]);
                mx = fmaxf(mx, fmaxf(fmaxf(fabsf(v[vi]), fabsf(v[vi + 1])), fmaxf(fabsf(v[vi + 2]), fabsf(v[vi + 3]))));
            }
        }
#pragma unroll
        for (int j = 0; j < quant_nr; ++j)
            convrot128(&v[j * 4]);
        mx = 0.f;
#pragma unroll
        for (int j = 0; j < quant_nr * 4; ++j)
            mx = fmaxf(mx, fabsf(v[j]));
        mx = warp_reduce_fmax(mx);
        const float sc = mx / 127.f + 1e-7f;
        const float inv_sc = 1.f / sc;
        if (lane == 0)
            sc_buf[oblk * 8 + otld] = sc;
#pragma unroll
        for (int j = 0; j < quant_nr; ++j) {
            const int n = base + j * 8;
            const int vi = j * 4;
            if (n < L) {
                store4_i8(&out[(int64_t)n * C + ch], quant_int8_rcp(v[vi], inv_sc), quant_int8_rcp(v[vi + 1], inv_sc),
                          quant_int8_rcp(v[vi + 2], inv_sc), quant_int8_rcp(v[vi + 3], inv_sc));
            }
        }
    }
}

// process_k: an anchor_index of zero or more subtracts that key from every key before the maximum, the rotation and
// the quantization, which shifts every score of a query alike.
__forceinline__ __device__ void process_k(const float * __restrict__ in, int8_t * __restrict__ out, float * __restrict__ sc_buf,
                                          const int oblk, const int L, const int C, const int anchor_index, const int64_t stride_n) {
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    const int otld = wid;
    float bias[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
        bias[i] = 0.f;
    const int ch = lane << 2;
    if (anchor_index >= 0) {
        const int64_t anchor_offset = (int64_t)anchor_index * stride_n + ch;
        load4(&in[anchor_offset], &bias[0]);
    }
    float v[2 * quant_nl * 4];
    float mx = 0.f;
#pragma unroll
    for (int i = 0; i < 2 * quant_nl * 4; ++i)
        v[i] = 0.f;
#pragma unroll
    for (int j = 0; j < quant_nl; ++j) {
#pragma unroll
        for (int p = 0; p < 2; ++p) {
            const int n = oblk * quant_warpk + j * 8 + otld * 2 + p;
            const int vi = (j * 2 + p) * 4;
            if (n < L) {
                load4(&in[(int64_t)n * stride_n + ch], &v[vi]);
#pragma unroll
                for (int c = 0; c < 4; ++c)
                    v[vi + c] -= bias[c];
                mx = fmaxf(mx, fmaxf(fmaxf(fabsf(v[vi]), fabsf(v[vi + 1])), fmaxf(fabsf(v[vi + 2]), fabsf(v[vi + 3]))));
            }
        }
    }
#pragma unroll
    for (int j = 0; j < 2 * quant_nl; ++j)
        convrot128(&v[j * 4]);
    mx = 0.f;
#pragma unroll
    for (int j = 0; j < 2 * quant_nl * 4; ++j)
        mx = fmaxf(mx, fabsf(v[j]));
    mx = warp_reduce_fmax(mx);
    const float sc = mx / 127.f + 1e-7f;
    const float inv_sc = 1.f / sc;
    if (lane == 0)
        sc_buf[oblk * 4 + otld] = sc;
#pragma unroll
    for (int j = 0; j < quant_nl; ++j) {
#pragma unroll
        for (int p = 0; p < 2; ++p) {
            const int n = oblk * quant_warpk + j * 8 + otld * 2 + p;
            const int vi = (j * 2 + p) * 4;
            if (n < L) {
                store4_i8(&out[(int64_t)n * C + ch], quant_int8_rcp(v[vi], inv_sc), quant_int8_rcp(v[vi + 1], inv_sc),
                          quant_int8_rcp(v[vi + 2], inv_sc), quant_int8_rcp(v[vi + 3], inv_sc));
            }
        }
    }
}
#pragma nv_diag_default 1056

constexpr int center_detect_threads = 128;
constexpr int center_samples = 9;
constexpr int center_max_channels = 256;

// ---- quant_v_int8.cu (comfy_kitchen) ----

constexpr int v_d_tile = 8;

// The INT8 product's 16-key inverse permutation: {b3,b2,b1,b0} to {b2,b0,b3,b1}.
__device__ __forceinline__ int inv_perm16(int w) {
    return (w & 1) | (((w >> 3) & 1) << 1) | (((w >> 1) & 1) << 2) | (((w >> 2) & 1) << 3);
}

// load_tile reads eight channels of V, each four as one vector where they start 16-byte aligned (load_f32x4_any).
__device__ __forceinline__ void load_tile(const float * ptr, float * out_vals) {
    const float4 v0 = load_f32x4_any(ptr);
    const float4 v1 = load_f32x4_any(ptr + 4);
    out_vals[0] = bf16_value(v0.x);
    out_vals[1] = bf16_value(v0.y);
    out_vals[2] = bf16_value(v0.z);
    out_vals[3] = bf16_value(v0.w);
    out_vals[4] = bf16_value(v1.x);
    out_vals[5] = bf16_value(v1.y);
    out_vals[6] = bf16_value(v1.z);
    out_vals[7] = bf16_value(v1.w);
}

// quant_v_int8: a block a (sequence, head, eight-channel tile); the channels' absolute maxima over the keys, then each
// key's channels quantized by them, written [sequence][head][channel][padded key] with each 16 keys permuted, the
// padding zero.
template <int THREADS>
__device__ __forceinline__ void quant_v_int8(const float * __restrict__ v, int8_t * __restrict__ out, float * __restrict__ scale_out,
                                             int N, int padded_N, int H, int D, int64_t sb, int64_t sh, int64_t sn) {
    const int d_tiles = D / v_d_tile;
    const int d_tile = blockIdx.x % d_tiles;
    const int bh = blockIdx.x / d_tiles;
    const int h = bh % H;
    const int b = bh / H;
    const int d0 = d_tile * v_d_tile;
    const float * base = v + b * sb + h * sh + d0;
    constexpr int WARPS = THREADS / 32;

    float mx[v_d_tile];
#pragma unroll
    for (int i = 0; i < v_d_tile; ++i)
        mx[i] = 0.f;
    int n = threadIdx.x;
    const int N_body = N - 3 * THREADS;
    for (; n < N_body; n += 4 * THREADS) {
        float t0[v_d_tile], t1[v_d_tile], t2[v_d_tile], t3[v_d_tile];
        load_tile(base + (int64_t)n * sn, t0);
        load_tile(base + (int64_t)(n + THREADS) * sn, t1);
        load_tile(base + (int64_t)(n + 2 * THREADS) * sn, t2);
        load_tile(base + (int64_t)(n + 3 * THREADS) * sn, t3);
#pragma unroll
        for (int di = 0; di < v_d_tile; ++di) {
            float a = fabsf(t0[di]);
            float b = fabsf(t1[di]);
            float c = fabsf(t2[di]);
            float d = fabsf(t3[di]);
            mx[di] = fmaxf(mx[di], fmaxf(fmaxf(a, b), fmaxf(c, d)));
        }
    }
    for (; n < N; n += THREADS) {
        float tmp[v_d_tile];
        load_tile(base + (int64_t)n * sn, tmp);
#pragma unroll
        for (int di = 0; di < v_d_tile; ++di)
            mx[di] = fmaxf(mx[di], fabsf(tmp[di]));
    }
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int di = 0; di < v_d_tile; ++di)
        mx[di] = warp_reduce_fmax(mx[di]);
    __shared__ float warp_mx[v_d_tile][WARPS];
    __shared__ float inv_sc_sh[v_d_tile];
    if (lane == 0) {
#pragma unroll
        for (int di = 0; di < v_d_tile; ++di)
            warp_mx[di][warp] = mx[di];
    }
    __syncthreads();
    if (threadIdx.x < v_d_tile) {
        float val = 0.f;
#pragma unroll
        for (int w = 0; w < WARPS; ++w)
            val = fmaxf(val, warp_mx[threadIdx.x][w]);
        float sc = fmaxf(val * (1.f / 127.f), 1e-12f);
        scale_out[(b * H + h) * D + d0 + threadIdx.x] = sc;
        inv_sc_sh[threadIdx.x] = 1.f / sc;
    }
    __syncthreads();
    float inv_sc[v_d_tile];
#pragma unroll
    for (int di = 0; di < v_d_tile; ++di)
        inv_sc[di] = inv_sc_sh[di];
    const int64_t out_row = static_cast<int64_t>((b * H + h) * D + d0);
    for (int src = N - 1 - threadIdx.x; src >= 0; src -= THREADS) {
        const int w = src & 15;
        const int dst = (src & ~15) | inv_perm16(w);
        float tmp[v_d_tile];
        load_tile(base + (int64_t)src * sn, tmp);
#pragma unroll
        for (int di = 0; di < v_d_tile; ++di) {
            out[(out_row + di) * padded_N + dst] = float_to_int8_rn(tmp[di] * inv_sc[di]);
        }
    }
    for (int src = N + threadIdx.x; src < padded_N; src += THREADS) {
        const int w = src & 15;
        const int dst = (src & ~15) | inv_perm16(w);
#pragma unroll
        for (int di = 0; di < v_d_tile; ++di)
            out[(out_row + di) * padded_N + dst] = 0;
    }
}

// ---- attn_utils.cuh and qk_int_sv_i8_cuda.cuh (SageAttention) ----

// The fixed configuration: 128 query rows a block in eight warps of 16, 128 keys a tile, heads of 128 channels.
constexpr uint32_t CTA_Q = 128;
constexpr uint32_t CTA_K = 128;
constexpr uint32_t WARP_Q = 16;
constexpr uint32_t WARP_K = 128;
constexpr uint32_t head_dim = 128;
constexpr uint32_t num_warps_q = CTA_Q / WARP_Q;
constexpr uint32_t num_warps_k = CTA_K / WARP_K;
constexpr uint32_t num_warps = num_warps_q * num_warps_k;
constexpr uint32_t num_tiles_q = WARP_Q / 16;
constexpr uint32_t num_tiles_k = WARP_K / 16;
constexpr uint32_t num_tiles_qk_inner = head_dim / 32;
constexpr uint32_t num_tiles_v = head_dim / 16;
constexpr uint32_t PACK_SIZE_QK = 16;
constexpr uint32_t PACK_SIZE_V = 16;
constexpr uint32_t PACK_SIZE_O = 8;
constexpr uint32_t QK_SMEM_STRIDE = head_dim;
constexpr uint32_t O_SMEM_STRIDE = head_dim;
constexpr uint32_t V_SMEM_STRIDE = CTA_K;
// Shared memory: two K tiles, the Q tile cached in registers, then V; the output's BF16 rows alias the K tiles.
constexpr uint32_t K_smem_idx_offset = 0;
constexpr uint32_t V_smem_idx_offset = 2 * CTA_K;
constexpr uint32_t shared_bytes = CTA_Q * head_dim + 2 * CTA_K * head_dim;
constexpr uint32_t line_lanes = 8;
constexpr uint32_t copy_lines_per_warp = 4;
constexpr uint32_t QK_smem_iters_row = QK_SMEM_STRIDE / (line_lanes * PACK_SIZE_QK);
constexpr uint32_t K_smem_iters_col = CTA_K / (num_warps * copy_lines_per_warp);
constexpr uint32_t V_smem_iters_row = V_SMEM_STRIDE / (line_lanes * PACK_SIZE_V);
constexpr uint32_t V_smem_iters_col = head_dim / (num_warps * copy_lines_per_warp);
constexpr uint32_t O_smem_iters_row = O_SMEM_STRIDE / (line_lanes * PACK_SIZE_O);
constexpr uint32_t O_smem_iters_col = CTA_Q / (num_warps * copy_lines_per_warp);
typedef smem_t<QK_SMEM_STRIDE / PACK_SIZE_QK> smem_qk;
typedef smem_t<V_SMEM_STRIDE / PACK_SIZE_V> smem_v;
typedef smem_t<O_SMEM_STRIDE / PACK_SIZE_O> smem_o;

__device__ __forceinline__ uint32_t div_ceil(uint32_t m, uint32_t n) { return (m + n - 1) / n; }
__device__ __forceinline__ uint32_t get_warp_id() { return threadIdx.y; }
__device__ __forceinline__ uint32_t get_lane_id() { return threadIdx.x; }
__device__ __forceinline__ uint32_t get_warp_idx_q() { return get_warp_id() / num_warps_k; }
__device__ __forceinline__ uint32_t get_warp_idx_k() { return get_warp_id() % num_warps_k; }

// load_global_to_share: a tile's rows copied by cp.async, each lane a 16-byte cell of a row, the lane pointer and the
// shared offset left where the next tile's copy starts.
template <uint32_t smem_iters_row, uint32_t smem_iters_col, uint32_t stride, uint32_t CTA>
__device__ __forceinline__ void load_global_to_share(int8_t ** lane_ptr, uint32_t & smem_offset, const uint32_t & gmem_stride,
                                                     const smem_t<stride> & smem) {
    constexpr uint32_t pack_size = 16;
#pragma unroll
    for (uint32_t i = 0; i < smem_iters_col; i++) {
#pragma unroll
        for (uint32_t j = 0; j < smem_iters_row; j++) {
            smem.load_128b_async(smem_offset, *lane_ptr);
            *lane_ptr += (line_lanes * pack_size);
            smem_offset = smem.template advance_offset_by_column<line_lanes>(smem_offset);
        }
        smem_offset = smem.template advance_offset_by_row<copy_lines_per_warp>(smem_offset - (smem_iters_row * line_lanes));
        *lane_ptr += ((copy_lines_per_warp * gmem_stride) - (smem_iters_row * line_lanes * pack_size));
    }
    smem_offset -= (smem_iters_col * copy_lines_per_warp * stride);
    *lane_ptr += (CTA - smem_iters_col * copy_lines_per_warp) * gmem_stride;
}

// The same, each row copied while its index is below max_len.
template <uint32_t smem_iters_row, uint32_t smem_iters_col, uint32_t stride, uint32_t CTA>
__device__ __forceinline__ void load_global_to_share(int8_t ** lane_ptr, uint32_t & smem_offset, const uint32_t & gmem_stride,
                                                     const smem_t<stride> & smem, uint32_t base_idx, uint32_t max_len) {
    constexpr uint32_t pack_size = 16;
#pragma unroll
    for (uint32_t i = 0; i < smem_iters_col; i++) {
#pragma unroll
        for (uint32_t j = 0; j < smem_iters_row; j++) {
            smem.load_128b_async(smem_offset, *lane_ptr, base_idx < max_len);
            *lane_ptr += (line_lanes * pack_size);
            smem_offset = smem.template advance_offset_by_column<line_lanes>(smem_offset);
        }
        smem_offset = smem.template advance_offset_by_row<copy_lines_per_warp>(smem_offset - (smem_iters_row * line_lanes));
        *lane_ptr += ((copy_lines_per_warp * gmem_stride) - (smem_iters_row * line_lanes * pack_size));
        base_idx += copy_lines_per_warp;
    }
    smem_offset -= (smem_iters_col * copy_lines_per_warp * stride);
    *lane_ptr += (CTA - smem_iters_col * copy_lines_per_warp) * gmem_stride;
}

__device__ __forceinline__ void load_int8_V_global_to_share(int8_t ** lane_ptr, uint32_t & smem_offset, const uint32_t & gmem_stride,
                                                            const smem_v & smem) {
    constexpr uint32_t pack_size_int8 = 16;
    constexpr uint32_t stride = V_SMEM_STRIDE / PACK_SIZE_V;
#pragma unroll
    for (uint32_t i = 0; i < V_smem_iters_col; i++) {
#pragma unroll
        for (uint32_t j = 0; j < V_smem_iters_row; j++) {
            smem.load_128b_async(smem_offset, *lane_ptr);
            *lane_ptr += (line_lanes * pack_size_int8);
            smem_offset = smem.template advance_offset_by_column<line_lanes>(smem_offset);
        }
        smem_offset = smem.template advance_offset_by_row<copy_lines_per_warp>(smem_offset - (V_smem_iters_row * line_lanes));
        *lane_ptr += ((copy_lines_per_warp * gmem_stride) - (V_smem_iters_row * line_lanes * pack_size_int8));
    }
    smem_offset -= (V_smem_iters_col * copy_lines_per_warp * stride);
    *lane_ptr += CTA_K;
    *lane_ptr -= (V_smem_iters_col * copy_lines_per_warp) * gmem_stride;
}

// compute_int_qk_cached: the tile's INT8 scores from the warp's cached query fragments and the key tile.
__device__ __forceinline__ void compute_int_qk_cached(const smem_qk & smem_K, int32_t RS[][num_tiles_k][8],
                                                      uint32_t RQ[][num_tiles_qk_inner][4], uint32_t & offset_K) {
    constexpr uint32_t stride = QK_SMEM_STRIDE / PACK_SIZE_QK;
    uint32_t RK[4];
#pragma unroll
    for (uint32_t iter = 0; iter < 1; iter++) {
#pragma unroll
        for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
            smem_K.ldmatrix_m8n8x4(offset_K, RK);
            offset_K = smem_K.template advance_offset_by_row<16>(offset_K);
#pragma unroll
            for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
                mma_s8s8s32<true>(RS[fq][fk], RQ[fq][iter], RK);
            }
        }
        offset_K = smem_K.template advance_offset_by_column<2>(offset_K - (num_tiles_k * 16 * stride), iter);
    }
#pragma unroll
    for (uint32_t iter = 1; iter < num_tiles_qk_inner; iter++) {
#pragma unroll
        for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
            smem_K.ldmatrix_m8n8x4(offset_K, RK);
            offset_K = smem_K.template advance_offset_by_row<16>(offset_K);
#pragma unroll
            for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
                mma_s8s8s32<false>(RS[fq][fk], RQ[fq][iter], RK);
            }
        }
        offset_K = smem_K.template advance_offset_by_column<2>(offset_K - (num_tiles_k * 16 * stride), iter);
    }
    offset_K -= (2 * num_tiles_qk_inner);
}

__device__ __forceinline__ void apply_out_of_bound_mask(const uint32_t & K_idx_lane_base, float RS[][num_tiles_k][8],
                                                        const uint32_t & kv_len, const float mask_value) {
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
        for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
#pragma unroll
            for (uint32_t k = 0; k < 8; k++) {
                const uint32_t kv_idx = K_idx_lane_base + fk * 16 + 8 * (k / 4) + k % 2;
                const bool out_of_boundary = (kv_idx >= kv_len);
                if (out_of_boundary) {
                    RS[fq][fk][k] = mask_value;
                }
            }
        }
    }
}

// update_mdo with an exponent offset and the scale applied here (exp_offset, no fuse_scale): the running maximum,
// denominator and output rescaled by the tile, its scores raised to probabilities in place.
__device__ __forceinline__ void update_mdo(float RS[][num_tiles_k][8], float RO[][num_tiles_v][8], float m[][2], float d[][2],
                                           float pv_scale[][2], const float & sm_scale, const float exp_offset_value) {
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
        for (uint32_t k = 0; k < 2; k++) {
            const float m_prev = m[fq][k];
            float m_temp = -50000.0f;
#pragma unroll
            for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
                const float m_local = max(max(RS[fq][fk][k * 2 + 0], RS[fq][fk][k * 2 + 1]), max(RS[fq][fk][k * 2 + 4], RS[fq][fk][k * 2 + 5]));
                m_temp = max(m_temp, m_local);
            }
            m_temp = fmaf(m_temp, sm_scale, -exp_offset_value);
            m_temp = max(m_temp, __shfl_xor_sync(0xffffffff, m_temp, 0x1));
            m_temp = max(m_temp, __shfl_xor_sync(0xffffffff, m_temp, 0x2));
            const float tile_m = m_temp;
            m[fq][k] = max(m_prev, tile_m);
            const float smaller_scale = ptx_exp2(-fabsf(m_prev - tile_m));
            const float o_scale = m_prev < tile_m ? smaller_scale : 1.0f;
            pv_scale[fq][k] = tile_m < m_prev ? smaller_scale : 1.0f;
            d[fq][k] *= o_scale;
#pragma unroll
            for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
                RO[fq][fv][k * 2 + 0] *= o_scale;
                RO[fq][fv][k * 2 + 1] *= o_scale;
                RO[fq][fv][k * 2 + 4] *= o_scale;
                RO[fq][fv][k * 2 + 5] *= o_scale;
            }
            const float negative_m = -tile_m;
#pragma unroll
            for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
                RS[fq][fk][k * 2 + 0] = ptx_exp2(fmaf(RS[fq][fk][k * 2 + 0], sm_scale, negative_m));
                RS[fq][fk][k * 2 + 1] = ptx_exp2(fmaf(RS[fq][fk][k * 2 + 1], sm_scale, negative_m));
                RS[fq][fk][k * 2 + 4] = ptx_exp2(fmaf(RS[fq][fk][k * 2 + 4], sm_scale, negative_m));
                RS[fq][fk][k * 2 + 5] = ptx_exp2(fmaf(RS[fq][fk][k * 2 + 5], sm_scale, negative_m));
            }
        }
    }
}

struct PackedU8RowSum {
    uint32_t probabilities;
    float denominator;
};

// pack_scaled_exp2_u8x4 for scores that cannot be NaN: four probabilities packed to U8 and their exact sum.
__device__ __forceinline__ PackedU8RowSum pack_scaled_exp2_u8x4(const int32_t a, const int32_t b, const int32_t c, const int32_t d,
                                                                const float scale, const float negative_m) {
    const float probability_a = ptx_exp2(fmaf(i2f_exact(a), scale, negative_m));
    const float probability_b = ptx_exp2(fmaf(i2f_exact(b), scale, negative_m));
    const float probability_c = ptx_exp2(fmaf(i2f_exact(c), scale, negative_m));
    const float probability_d = ptx_exp2(fmaf(i2f_exact(d), scale, negative_m));
    const uint32_t probabilities = pack_u8x4(probability_a, probability_b, probability_c, probability_d);
    const uint32_t denominator = __dp4a(probabilities, 0x01010101u, 0u);
    return {probabilities, i2f_exact(static_cast<int32_t>(denominator))};
}

// update_mdo_i32_u8: the softmax step of a whole tile from its INT32 scores, the probabilities packed to U8 as they
// are made and summed from their U8 values; the tile's probability scale left in RS[fq][0][k].
__device__ __forceinline__ void update_mdo_i32_u8(int32_t RS[][num_tiles_k][8], float RO[][num_tiles_v][8], float m[][2], float d[][2],
                                                  const float sm_scale, const float exp_offset_value, uint32_t RS_u8[][num_tiles_k / 2][4]) {
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
        for (uint32_t k = 0; k < 2; k++) {
            const float m_prev = m[fq][k];
            int32_t m_temp_i32 = -2147483647 - 1;
#pragma unroll
            for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
                const int32_t m_local = max(max(RS[fq][fk][k * 2], RS[fq][fk][k * 2 + 1]), max(RS[fq][fk][k * 2 + 4], RS[fq][fk][k * 2 + 5]));
                m_temp_i32 = max(m_temp_i32, m_local);
            }
            float m_temp = fmaf(__int2float_rz(m_temp_i32), sm_scale, -exp_offset_value);
            m_temp = max(m_temp, __shfl_xor_sync(0xffffffff, m_temp, 0x1));
            m_temp = max(m_temp, __shfl_xor_sync(0xffffffff, m_temp, 0x2));
            const float tile_m = m_temp;
            m[fq][k] = max(m_prev, tile_m);
            const float smaller_scale = ptx_exp2(-fabsf(m_prev - tile_m));
            const float o_scale = m_prev < tile_m ? smaller_scale : 1.0f;
            const float tile_scale = tile_m < m_prev ? smaller_scale : 1.0f;
            d[fq][k] *= o_scale;
#pragma unroll
            for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
                RO[fq][fv][k * 2] *= o_scale;
                RO[fq][fv][k * 2 + 1] *= o_scale;
                RO[fq][fv][k * 2 + 4] *= o_scale;
                RO[fq][fv][k * 2 + 5] *= o_scale;
            }
            const float negative_m = -tile_m;
#pragma unroll
            for (uint32_t fk = 0; fk < num_tiles_k / 2; fk++) {
                const PackedU8RowSum probabilities_0 = pack_scaled_exp2_u8x4(
                    RS[fq][fk * 2][k * 2], RS[fq][fk * 2][k * 2 + 1], RS[fq][fk * 2][k * 2 + 4], RS[fq][fk * 2][k * 2 + 5], sm_scale, negative_m);
                const PackedU8RowSum probabilities_1 = pack_scaled_exp2_u8x4(
                    RS[fq][fk * 2 + 1][k * 2], RS[fq][fk * 2 + 1][k * 2 + 1], RS[fq][fk * 2 + 1][k * 2 + 4], RS[fq][fk * 2 + 1][k * 2 + 5],
                    sm_scale, negative_m);
                RS_u8[fq][fk][k] = probabilities_0.probabilities;
                RS_u8[fq][fk][k + 2] = probabilities_1.probabilities;
                d[fq][k] += probabilities_0.denominator * tile_scale;
                d[fq][k] += probabilities_1.denominator * tile_scale;
            }
            RS[fq][0][k] = __float_as_int(tile_scale);
        }
    }
}

__device__ __forceinline__ void RS_to_u8(float RS[][num_tiles_k][8], uint32_t RS_u8[][num_tiles_k / 2][4]) {
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
        for (uint32_t fk = 0; fk < num_tiles_k / 2; fk++) {
            RS_u8[fq][fk][0] = pack_u8x4(RS[fq][fk * 2][0], RS[fq][fk * 2][1], RS[fq][fk * 2][4], RS[fq][fk * 2][5]);
            RS_u8[fq][fk][1] = pack_u8x4(RS[fq][fk * 2][2], RS[fq][fk * 2][3], RS[fq][fk * 2][6], RS[fq][fk * 2][7]);
            RS_u8[fq][fk][2] = pack_u8x4(RS[fq][fk * 2 + 1][0], RS[fq][fk * 2 + 1][1], RS[fq][fk * 2 + 1][4], RS[fq][fk * 2 + 1][5]);
            RS_u8[fq][fk][3] = pack_u8x4(RS[fq][fk * 2 + 1][2], RS[fq][fk * 2 + 1][3], RS[fq][fk * 2 + 1][6], RS[fq][fk * 2 + 1][7]);
        }
    }
}

__device__ __forceinline__ void accumulate_d(float RS[][num_tiles_k][8], float d[][2], float pv_scale[][2]) {
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
        for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
            const float tile_sum_0 = (RS[fq][fk][0] + RS[fq][fk][1]) + (RS[fq][fk][4] + RS[fq][fk][5]);
            const float tile_sum_1 = (RS[fq][fk][2] + RS[fq][fk][3]) + (RS[fq][fk][6] + RS[fq][fk][7]);
            d[fq][0] += tile_sum_0 * pv_scale[fq][0];
            d[fq][1] += tile_sum_1 * pv_scale[fq][1];
        }
    }
}

// compute_int8_sv: the tile's U8 probabilities by its INT8 values, each output's exact INT32 sum added to it times the
// tile's probability scale (RS_scale[fq][0][k]).
__device__ __forceinline__ void compute_int8_sv(const smem_v & smem_V, int32_t RS_scale[][num_tiles_k][8],
                                                uint32_t RS_u8[][num_tiles_k / 2][4], float RO[][num_tiles_v][8]) {
    uint32_t smem_V_row_base = get_lane_id() % 8 + (get_lane_id() / 16) * 8;
    uint32_t smem_V_col_base = (get_lane_id() / 8) % 2;
    uint32_t offsets_V[num_tiles_k / 2];
#pragma unroll
    for (uint32_t fk = 0; fk < num_tiles_k / 2; fk++) {
        offsets_V[fk] = smem_V.get_permuted_offset(smem_V_row_base, smem_V_col_base + fk * 2);
    }
#pragma unroll
    for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
        uint32_t RV[num_tiles_k / 2][4];
#pragma unroll
        for (uint32_t fk = 0; fk < num_tiles_k / 2; fk++) {
            smem_V.ldmatrix_m8n8x4(offsets_V[fk], RV[fk]);
            offsets_V[fk] = smem_V.template advance_offset_by_row<16>(offsets_V[fk]);
        }
#pragma unroll
        for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
            int32_t partial[8];
            mma_u8s8s32<true>(partial, RS_u8[fq][0], RV[0]);
#pragma unroll
            for (uint32_t fk = 1; fk < num_tiles_k / 2; fk++) {
                mma_u8s8s32<false>(partial, RS_u8[fq][fk], RV[fk]);
            }
#pragma unroll
            for (uint32_t k = 0; k < 8; k++) {
                RO[fq][fv][k] = fmaf(i2f_exact(partial[k]), __int_as_float(RS_scale[fq][0][(k % 4) / 2]), RO[fq][fv][k]);
            }
        }
    }
}

// normalize_d with the denominator on CUDA cores: each row's sum gathered over its quad, the output divided by it.
__device__ __forceinline__ void normalize_d(float RO[][num_tiles_v][8], float m[][2], float d[][2]) {
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
        for (uint32_t k = 0; k < 2; k++) {
            d[fq][k] += __shfl_xor_sync(0xffffffff, d[fq][k], 0x1);
            d[fq][k] += __shfl_xor_sync(0xffffffff, d[fq][k], 0x2);
        }
    }
    float d_rcp[num_tiles_q][2];
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
        for (uint32_t k = 0; k < 2; k++) {
            d_rcp[fq][k] = ptx_rcp(d[fq][k]);
        }
    }
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
        for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
#pragma unroll
            for (uint32_t k = 0; k < 8; k++) {
                RO[fq][fv][k] *= d_rcp[fq][(k % 4) / 2];
            }
        }
    }
}

// The last tile's softmax: its scores past kv_len masked, the generic F32 step, its probabilities packed to U8 and
// summed in F32, and the tile's probability scale left in RS[fq][0][k].
__device__ __forceinline__ void last_tile_softmax(int32_t RS[][num_tiles_k][8], float RO[][num_tiles_v][8], float m[][2], float d[][2],
                                                  uint32_t RS_u8[][num_tiles_k / 2][4], const uint32_t K_idx_lane_base,
                                                  const uint32_t kv_len, const float sm_scale) {
    float RS_soft[num_tiles_q][num_tiles_k][8];
    float pv_scale[num_tiles_q][2];
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
        for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
#pragma unroll
            for (uint32_t k = 0; k < 8; k++) {
                RS_soft[fq][fk][k] = i2f_exact(RS[fq][fk][k]);
            }
        }
    }
    apply_out_of_bound_mask(K_idx_lane_base, RS_soft, kv_len, -1.0e30f);
    update_mdo(RS_soft, RO, m, d, pv_scale, sm_scale, s_u8_offset);
    RS_to_u8(RS_soft, RS_u8);
    accumulate_d(RS_soft, d, pv_scale);
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
        for (uint32_t k = 0; k < 2; k++)
            RS[fq][0][k] = __float_as_int(pv_scale[fq][k]);
    }
}

// attention: qk_int_sv_i8_attn_body at the fixed configuration on sm_89, which caches each warp's query fragments in
// registers and keeps two key tiles in shared memory. A block takes CTA_Q query rows of one head of one sequence.
__device__ __forceinline__ void attention(
        int8_t * __restrict__ Q, int8_t * __restrict__ K, int8_t * __restrict__ V, float * __restrict__ O,
        const float * __restrict__ Q_scale, const float * __restrict__ K_scale, const float * __restrict__ V_scale,
        const uint32_t qo_len, const uint32_t kv_len, const uint32_t num_kv_groups,
        const uint32_t stride_bz_q, const uint32_t stride_seq_q, const uint32_t stride_h_q,
        const uint32_t stride_bz_k, const uint32_t stride_seq_k, const uint32_t stride_h_k,
        const uint32_t stride_bz_v, const uint32_t stride_h_v, const uint32_t stride_d_v,
        const uint32_t stride_bz_o, const uint32_t stride_seq_o, const uint32_t stride_h_o, float sm_scale) {
    extern __shared__ __align__(16) int8_t kitchen_attention_smem[];
    int8_t * smem = kitchen_attention_smem;
    const uint32_t lane_id = get_lane_id();
    const uint32_t warp_id = get_warp_id();
    const uint32_t batch_id = blockIdx.z;
    const uint32_t bx = blockIdx.x;
    const uint32_t num_qo_heads = gridDim.y;
    const uint32_t head_id = blockIdx.y;

    sm_scale *= log2e;

    int32_t RS[num_tiles_q][num_tiles_k][8];
    float RO[num_tiles_q][num_tiles_v][8];
    float m[num_tiles_q][2];
    float d[num_tiles_q][2];

    // Q scales are 8 a 32-row warp block, a pair of 16-row warps sharing one; K scales are 4 a 128-key tile.
    uint32_t q_scale_idx, k_scale_idx;
    {
        constexpr uint32_t quant_warps_q = CTA_Q / 32;
        const uint32_t num_warp_block_q = gridDim.x * quant_warps_q;
        q_scale_idx = batch_id * num_qo_heads * (num_warp_block_q * 8) + head_id * (num_warp_block_q * 8) + bx * (quant_warps_q * 8) +
                      (get_warp_idx_q() / 2) * 8 + lane_id / 4;
    }
    {
        const uint32_t num_warp_block_k = div_ceil(kv_len, CTA_K) * (CTA_K / WARP_K);
        k_scale_idx = batch_id * (num_qo_heads / num_kv_groups) * (num_warp_block_k * 4) +
                      (head_id / num_kv_groups) * (num_warp_block_k * 4) + get_warp_idx_k() * 4 + lane_id % 4;
    }
    constexpr uint32_t k_scale_advance_offset = (CTA_K / WARP_K) * 4;

#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
        for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
#pragma unroll
            for (uint32_t k = 0; k < 8; k++) {
                RO[fq][fv][k] = 0.0f;
            }
        }
    }
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
        for (uint32_t k = 0; k < 2; k++) {
            m[fq][k] = -50000.0f;
            d[fq][k] = 1.0f;
        }
    }

    smem_qk smem_K(smem + K_smem_idx_offset * QK_SMEM_STRIDE);
    smem_v smem_V(smem + V_smem_idx_offset * QK_SMEM_STRIDE);
    smem_o smem_O(smem);

    int8_t * K_lane_base_ptr = K + batch_id * stride_bz_k + (head_id / num_kv_groups) * stride_h_k +
                               (CTA_K / num_warps * warp_id + lane_id / line_lanes) * stride_seq_k + (lane_id % line_lanes) * PACK_SIZE_QK;
    int8_t * V_lane_base_ptr = V + batch_id * stride_bz_v + (head_id / num_kv_groups) * stride_h_v + head_dim / num_warps * warp_id * stride_d_v +
                               lane_id / line_lanes * stride_d_v + (lane_id % line_lanes) * PACK_SIZE_V;
    uint32_t K_smem_offset_load = smem_K.get_permuted_offset(warp_id * copy_lines_per_warp * K_smem_iters_col + lane_id / line_lanes, lane_id % line_lanes);
    uint32_t V_smem_offset_load = smem_V.get_permuted_offset(warp_id * copy_lines_per_warp * V_smem_iters_col + lane_id / line_lanes, lane_id % line_lanes);
    uint32_t K_smem_offset_mma = smem_K.get_permuted_offset(get_warp_idx_k() * WARP_K + lane_id % 8 + (lane_id / 16) * 8, (lane_id / 8) % 2);
    uint32_t K_idx_lane_base = get_warp_idx_k() * WARP_K + 2 * (lane_id % 4);
    uint32_t K_load_idx_lane_base = CTA_K / num_warps * warp_id + lane_id / line_lanes;

    const uint32_t num_iterations = div_ceil(kv_len, CTA_K);

    // Each warp's query fragments, read once into registers.
    uint32_t RQ_cached[num_tiles_q][num_tiles_qk_inner][4];
    {
        const int8_t * query = Q + batch_id * stride_bz_q + head_id * stride_h_q;
#pragma unroll
        for (uint32_t fq = 0; fq < num_tiles_q; ++fq) {
            const uint32_t row = bx * CTA_Q + get_warp_idx_q() * WARP_Q + fq * 16 + lane_id / 4;
#pragma unroll
            for (uint32_t inner = 0; inner < num_tiles_qk_inner; ++inner) {
                const uint32_t col = inner * 32 + (lane_id % 4) * 4;
                RQ_cached[fq][inner][0] = row < qo_len ? __ldg(reinterpret_cast<const uint32_t *>(query + row * stride_seq_q + col)) : 0;
                RQ_cached[fq][inner][1] = row + 8 < qo_len ? __ldg(reinterpret_cast<const uint32_t *>(query + (row + 8) * stride_seq_q + col)) : 0;
                RQ_cached[fq][inner][2] = row < qo_len ? __ldg(reinterpret_cast<const uint32_t *>(query + row * stride_seq_q + col + 16)) : 0;
                RQ_cached[fq][inner][3] = row + 8 < qo_len ? __ldg(reinterpret_cast<const uint32_t *>(query + (row + 8) * stride_seq_q + col + 16)) : 0;
            }
        }
    }

    load_global_to_share<QK_smem_iters_row, K_smem_iters_col, QK_SMEM_STRIDE / PACK_SIZE_QK, CTA_K>(
        &K_lane_base_ptr, K_smem_offset_load, stride_seq_k, smem_K, K_load_idx_lane_base, kv_len);
    commit_group();

    const float q_scale = Q_scale[q_scale_idx];
    const float original_sm_scale = sm_scale;
    float dequant_scale = q_scale * K_scale[k_scale_idx + 0 * k_scale_advance_offset];
    sm_scale = original_sm_scale * dequant_scale;

    // V is padded to a whole CTA_K tile by its quantizer.
    load_int8_V_global_to_share(&V_lane_base_ptr, V_smem_offset_load, stride_d_v, smem_V);
    commit_group();

    K_load_idx_lane_base += CTA_K;

#pragma unroll
    for (uint32_t iter = 1; iter < num_iterations - 1; iter++) {
        wait_group<0>();
        __syncthreads();
        uint32_t RS_u8[num_tiles_q][num_tiles_k / 2][4];
        {
            smem_qk next_K(smem + ((iter & 1) * CTA_K) * QK_SMEM_STRIDE);
            load_global_to_share<QK_smem_iters_row, K_smem_iters_col, QK_SMEM_STRIDE / PACK_SIZE_QK, CTA_K>(
                &K_lane_base_ptr, K_smem_offset_load, stride_seq_k, next_K);
            commit_group();
        }
        compute_int_qk_cached(smem_K, RS, RQ_cached, K_smem_offset_mma);
        update_mdo_i32_u8(RS, RO, m, d, sm_scale, s_u8_offset, RS_u8);
        K_idx_lane_base += CTA_K;
        dequant_scale = q_scale * K_scale[k_scale_idx + iter * k_scale_advance_offset];
        sm_scale = original_sm_scale * dequant_scale;
        compute_int8_sv(smem_V, RS, RS_u8, RO);
        __syncthreads();
        smem_K.base = reinterpret_cast<uint4 *>(smem + ((iter & 1) * CTA_K) * QK_SMEM_STRIDE);
        load_int8_V_global_to_share(&V_lane_base_ptr, V_smem_offset_load, stride_d_v, smem_V);
        commit_group();
        K_load_idx_lane_base += CTA_K;
    }

    // The second last tile: the last key tile is copied with its rows past kv_len left alone.
    if (num_iterations > 1) {
        wait_group<0>();
        __syncthreads();
        uint32_t RS_u8[num_tiles_q][num_tiles_k / 2][4];
        {
            smem_qk next_K(smem + (((num_iterations - 1) & 1) * CTA_K) * QK_SMEM_STRIDE);
            load_global_to_share<QK_smem_iters_row, K_smem_iters_col, QK_SMEM_STRIDE / PACK_SIZE_QK, CTA_K>(
                &K_lane_base_ptr, K_smem_offset_load, stride_seq_k, next_K, K_load_idx_lane_base, kv_len);
            commit_group();
        }
        compute_int_qk_cached(smem_K, RS, RQ_cached, K_smem_offset_mma);
        update_mdo_i32_u8(RS, RO, m, d, sm_scale, s_u8_offset, RS_u8);
        K_idx_lane_base += CTA_K;
        dequant_scale = q_scale * K_scale[k_scale_idx + (num_iterations - 1) * k_scale_advance_offset];
        sm_scale = original_sm_scale * dequant_scale;
        compute_int8_sv(smem_V, RS, RS_u8, RO);
        __syncthreads();
        smem_K.base = reinterpret_cast<uint4 *>(smem + (((num_iterations - 1) & 1) * CTA_K) * QK_SMEM_STRIDE);
        load_int8_V_global_to_share(&V_lane_base_ptr, V_smem_offset_load, stride_d_v, smem_V);
        commit_group();
        K_load_idx_lane_base += CTA_K;
    }

    // The last tile, its keys past kv_len masked.
    {
        wait_group<0>();
        __syncthreads();
        uint32_t RS_u8[num_tiles_q][num_tiles_k / 2][4];
        compute_int_qk_cached(smem_K, RS, RQ_cached, K_smem_offset_mma);
        last_tile_softmax(RS, RO, m, d, RS_u8, K_idx_lane_base, kv_len, sm_scale);
        K_idx_lane_base += CTA_K;
        compute_int8_sv(smem_V, RS, RS_u8, RO);
        __syncthreads();
    }

    normalize_d(RO, m, d);

    // V's per-channel scale.
    {
        float v_scale[4];
        const float * V_scale_base_ptr =
            V_scale + batch_id * (num_qo_heads / num_kv_groups) * head_dim + (head_id / num_kv_groups) * head_dim + (lane_id % 4) * 2;
#pragma unroll
        for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
            ((float2 *)v_scale)[0] = *((const float2 *)(V_scale_base_ptr + fv * 16));
            ((float2 *)v_scale)[1] = *((const float2 *)(V_scale_base_ptr + fv * 16 + 8));
#pragma unroll
            for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
                for (uint32_t k = 0; k < 8; k++) {
                    const float scale_value = v_scale[(k / 4) * 2 + (k % 2)];
                    RO[fq][fv][k] *= scale_value;
                }
            }
        }
    }

    // The output rounded to BF16 into shared memory, each warp its own rows.
    const uint32_t smem_O_row_base = get_warp_idx_q() * WARP_Q + lane_id / 4;
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
        for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
            uint32_t offset_O = smem_O.get_permuted_offset(smem_O_row_base + fq * 16, fv * (16 / PACK_SIZE_O));
            uint32_t RO_f16[4];
#pragma unroll
            for (uint32_t k = 0; k < 4; k++) {
                ((nv_bfloat162 *)RO_f16)[k] = __float22bfloat162_rn(((float2 *)RO[fq][fv])[k]);
            }
            ((uint32_t *)(smem_O.base + offset_O))[lane_id % 4] = RO_f16[0];
            ((uint32_t *)(smem_O.base + offset_O + 8 * (O_SMEM_STRIDE / PACK_SIZE_O)))[lane_id % 4] = RO_f16[1];
            offset_O = smem_O.get_permuted_offset(smem_O_row_base + fq * 16, fv * (16 / PACK_SIZE_O) + 1);
            ((uint32_t *)(smem_O.base + offset_O))[lane_id % 4] = RO_f16[2];
            ((uint32_t *)(smem_O.base + offset_O + 8 * (O_SMEM_STRIDE / PACK_SIZE_O)))[lane_id % 4] = RO_f16[3];
        }
    }
    __syncwarp();

    // Shared memory to the output, each lane eight BF16 values written as F32.
    float * O_lane_ptr = O + batch_id * stride_bz_o + head_id * stride_h_o +
                         (bx * CTA_Q + WARP_Q * get_warp_idx_q() + lane_id / line_lanes) * stride_seq_o + lane_id % line_lanes * PACK_SIZE_O;
    uint32_t offset_O = smem_O.get_permuted_offset(get_warp_idx_q() * WARP_Q + lane_id / line_lanes, lane_id % line_lanes);
    uint32_t O_load_idx_lane_base = bx * CTA_Q + CTA_Q / num_warps * warp_id + lane_id / line_lanes;
#pragma unroll
    for (uint32_t i = 0; i < O_smem_iters_col; i++) {
#pragma unroll
        for (uint32_t j = 0; j < O_smem_iters_row; j++) {
            if (O_load_idx_lane_base < qo_len) {
                const uint4 cell = *(smem_O.base + offset_O);
                const nv_bfloat162 * pairs = reinterpret_cast<const nv_bfloat162 *>(&cell);
                const float2 f0 = __bfloat1622float2(pairs[0]);
                const float2 f1 = __bfloat1622float2(pairs[1]);
                const float2 f2 = __bfloat1622float2(pairs[2]);
                const float2 f3 = __bfloat1622float2(pairs[3]);
                reinterpret_cast<float4 *>(O_lane_ptr)[0] = make_float4(f0.x, f0.y, f1.x, f1.y);
                reinterpret_cast<float4 *>(O_lane_ptr)[1] = make_float4(f2.x, f2.y, f3.x, f3.y);
            }
            O_lane_ptr += (line_lanes * PACK_SIZE_O);
            offset_O = smem_O.template advance_offset_by_column<line_lanes>(offset_O);
        }
        offset_O = smem_O.template advance_offset_by_row<copy_lines_per_warp>(offset_O - (O_smem_iters_row * line_lanes));
        O_lane_ptr += ((copy_lines_per_warp * stride_seq_o) - (O_smem_iters_row * line_lanes * PACK_SIZE_O));
        O_load_idx_lane_base += copy_lines_per_warp;
    }
}

} // namespace kitchen_attention

// int8_attention_anchor_f32 (detect_k_anchor): of nine keys sampled evenly over a sequence's key head, the one whose
// subtraction leaves the least sampled energy, written as its key index where that lowers the energy without raising
// the sampled absolute maximum past 1.125 times, else -1. Grid (key heads, sequences), 128 threads; the keys F32,
// rounded to BF16 as they load, at the strides given in elements.
extern "C" __global__ void __launch_bounds__(kitchen_attention::center_detect_threads) int8_attention_anchor_f32(
        const float * k_in, int * anchor_indices, int Lk, int C, int H_kv,
        unsigned int stride_b, unsigned int stride_h, unsigned int stride_n) {
    using namespace kitchen_attention;
    const int h = blockIdx.x;
    const int b = blockIdx.y;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int64_t bh_offset = (int64_t)b * stride_b + (int64_t)h * stride_h;
    __shared__ float samples[center_samples * center_max_channels];
    __shared__ float warp_original_energy[4];
    __shared__ float warp_original_max[4];
    __shared__ float warp_candidate_distance[center_samples][4];
    __shared__ float warp_best_energy[4];
    __shared__ float warp_best_max[4];
    __shared__ int selected_candidate;
    for (int index = tid; index < center_samples * C; index += center_detect_threads) {
        const int sample = index / C;
        const int channel = index - sample * C;
        const int row = sample * (Lk - 1) / (center_samples - 1);
        samples[index] = bf16_value(__ldg(&k_in[bh_offset + (int64_t)row * stride_n + channel]));
    }
    __syncthreads();
    float original_energy = 0.f;
    float original_max = 0.f;
    float candidate_distance[center_samples];
#pragma unroll
    for (int candidate = 0; candidate < center_samples; ++candidate) {
        candidate_distance[candidate] = 0.f;
    }
    for (int channel = tid; channel < C; channel += center_detect_threads) {
        float channel_sum = 0.f;
#pragma unroll
        for (int sample = 0; sample < center_samples; ++sample) {
            const float value = samples[sample * C + channel];
            original_energy = fmaf(value, value, original_energy);
            original_max = fmaxf(original_max, fabsf(value));
            channel_sum += value;
        }
#pragma unroll
        for (int candidate = 0; candidate < center_samples; ++candidate) {
            const float distance = center_samples * samples[candidate * C + channel] - channel_sum;
            candidate_distance[candidate] = fmaf(distance, distance, candidate_distance[candidate]);
        }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        original_energy += __shfl_down_sync(0xffffffffu, original_energy, offset);
        original_max = fmaxf(original_max, __shfl_down_sync(0xffffffffu, original_max, offset));
#pragma unroll
        for (int candidate = 0; candidate < center_samples; ++candidate) {
            candidate_distance[candidate] += __shfl_down_sync(0xffffffffu, candidate_distance[candidate], offset);
        }
    }
    if (lane == 0) {
        warp_original_energy[warp] = original_energy;
        warp_original_max[warp] = original_max;
#pragma unroll
        for (int candidate = 0; candidate < center_samples; ++candidate) {
            warp_candidate_distance[candidate][warp] = candidate_distance[candidate];
        }
    }
    __syncthreads();
    if (tid == 0) {
        int best_candidate = 0;
        float best_distance = 3.402823466e+38F;
#pragma unroll
        for (int candidate = 0; candidate < center_samples; ++candidate) {
            float distance = 0.f;
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                distance += warp_candidate_distance[candidate][w];
            }
            if (distance < best_distance) {
                best_candidate = candidate;
                best_distance = distance;
            }
        }
        selected_candidate = best_candidate;
    }
    __syncthreads();
    float best_energy = 0.f;
    float best_max = 0.f;
    for (int channel = tid; channel < C; channel += center_detect_threads) {
        const float anchor = samples[selected_candidate * C + channel];
#pragma unroll
        for (int sample = 0; sample < center_samples; ++sample) {
            const float residual = samples[sample * C + channel] - anchor;
            best_energy = fmaf(residual, residual, best_energy);
            best_max = fmaxf(best_max, fabsf(residual));
        }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        best_energy += __shfl_down_sync(0xffffffffu, best_energy, offset);
        best_max = fmaxf(best_max, __shfl_down_sync(0xffffffffu, best_max, offset));
    }
    if (lane == 0) {
        warp_best_energy[warp] = best_energy;
        warp_best_max[warp] = best_max;
    }
    __syncthreads();
    if (tid == 0) {
        float total_original_energy = 0.f;
        float total_original_max = 0.f;
        float total_best_energy = 0.f;
        float total_best_max = 0.f;
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            total_original_energy += warp_original_energy[w];
            total_original_max = fmaxf(total_original_max, warp_original_max[w]);
            total_best_energy += warp_best_energy[w];
            total_best_max = fmaxf(total_best_max, warp_best_max[w]);
        }
        const bool improves_range = total_best_energy < total_original_energy && total_best_max <= total_original_max * 1.125f;
        anchor_indices[b * H_kv + h] = improves_range ? selected_candidate * (Lk - 1) / (center_samples - 1) : -1;
    }
}

// int8_attention_quantize_qk_f32 (quant_qk_fused): blocks below q_oblk_count quantize a 32-row warp block of a query
// head, the rest a 128-key block of a key head, each into [sequence][head][token][channel] INT8 and its scales. Grid
// (query warp blocks plus key blocks, the larger head count, sequences), 128 threads; the inputs F32, rounded to BF16
// as they load, at the strides given in elements.
extern "C" __global__ void __launch_bounds__(128, 3) int8_attention_quantize_qk_f32(
        const float * q_in, signed char * q_out, float * q_sb, const float * k_in, signed char * k_out, float * k_sb,
        const int * anchor_indices, int Lq, int Lk, int C, int q_oblk_count, int H_q, int H_kv, int q_sc_per_h, int k_sc_per_h,
        unsigned int q_stride_b, unsigned int q_stride_h, unsigned int q_stride_n,
        unsigned int k_stride_b, unsigned int k_stride_h, unsigned int k_stride_n) {
    using namespace kitchen_attention;
    const int h = blockIdx.y, b = blockIdx.z;
    if (blockIdx.x < (unsigned)q_oblk_count) {
        if (h >= H_q)
            return;
        const int64_t in_bh = (int64_t)b * q_stride_b + (int64_t)h * q_stride_h;
        const int64_t out_bh = ((int64_t)b * H_q + h) * Lq * C;
        const int64_t sbh = ((int64_t)b * H_q + h) * q_sc_per_h;
        process_q(q_in + in_bh, reinterpret_cast<int8_t *>(q_out) + out_bh, q_sb + sbh, blockIdx.x, Lq, C, q_stride_n);
    } else {
        if (h >= H_kv)
            return;
        const int64_t in_bh = (int64_t)b * k_stride_b + (int64_t)h * k_stride_h;
        const int64_t out_bh = ((int64_t)b * H_kv + h) * Lk * C;
        const int64_t sbh = ((int64_t)b * H_kv + h) * k_sc_per_h;
        const int anchor_index = __ldg(&anchor_indices[b * H_kv + h]);
        process_k(k_in + in_bh, reinterpret_cast<int8_t *>(k_out) + out_bh, k_sb + sbh, (int)blockIdx.x - q_oblk_count, Lk, C, anchor_index,
                  k_stride_n);
    }
}

// int8_attention_quantize_v_f32 (quant_v_int8_kernel<512>, for more than 256 keys): V to INT8 per channel, written
// [sequence][head][channel][padded keys], each 16 keys permuted for the value product. Grid sequences x heads x
// channels / 8, 512 threads; V F32, rounded to BF16 as it loads, at the strides given in elements.
extern "C" __global__ void int8_attention_quantize_v_f32(
        const float * v, signed char * out, float * scale_out, int N, int padded_N, int H, int D,
        unsigned int sb, unsigned int sh, unsigned int sn) {
    kitchen_attention::quant_v_int8<512>(v, reinterpret_cast<int8_t *>(out), scale_out, N, padded_N, H, D, sb, sh, sn);
}

// int8_attention_quantize_v_short_f32 (quant_v_int8_kernel<128>): the same for 256 keys or fewer, 128 threads.
extern "C" __global__ void int8_attention_quantize_v_short_f32(
        const float * v, signed char * out, float * scale_out, int N, int padded_N, int H, int D,
        unsigned int sb, unsigned int sh, unsigned int sn) {
    kitchen_attention::quant_v_int8<128>(v, reinterpret_cast<int8_t *>(out), scale_out, N, padded_N, H, D, sb, sh, sn);
}

// int8_attention_bf16_f32 (qk_int_sv_i8_attn_kernel): the attention of the quantized heads, each output rounded to
// BF16 and written as F32 at the output strides given in elements. Grid (query tokens / 128 rounded up, query heads,
// sequences), block (32, 8), kitchen_attention::shared_bytes of dynamic shared memory; the query heads share a key head
// in groups of num_kv_groups; sm_scale is the scores' scale.
extern "C" __global__ void int8_attention_bf16_f32(
        const signed char * Q, const signed char * K, const signed char * V, float * O,
        const float * Q_scale, const float * K_scale, const float * V_scale,
        unsigned int qo_len, unsigned int kv_len, unsigned int num_kv_groups,
        unsigned int stride_bz_q, unsigned int stride_seq_q, unsigned int stride_h_q,
        unsigned int stride_bz_k, unsigned int stride_seq_k, unsigned int stride_h_k,
        unsigned int stride_bz_v, unsigned int stride_h_v, unsigned int stride_d_v,
        unsigned int stride_bz_o, unsigned int stride_seq_o, unsigned int stride_h_o, float sm_scale) {
    kitchen_attention::attention(
        const_cast<int8_t *>(reinterpret_cast<const int8_t *>(Q)), const_cast<int8_t *>(reinterpret_cast<const int8_t *>(K)),
        const_cast<int8_t *>(reinterpret_cast<const int8_t *>(V)), O, Q_scale, K_scale, V_scale, qo_len, kv_len, num_kv_groups,
        stride_bz_q, stride_seq_q, stride_h_q, stride_bz_k, stride_seq_k, stride_h_k, stride_bz_v, stride_h_v, stride_d_v,
        stride_bz_o, stride_seq_o, stride_h_o, sm_scale);
}

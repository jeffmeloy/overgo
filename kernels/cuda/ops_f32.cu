#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <mma.h>
#include "iq_tables_generated.cuh"

enum GGMLStorageType : unsigned int {
    GGML_F32 = 0,
    GGML_Q4_0 = 2,
    GGML_Q4_1 = 3,
    GGML_Q5_0 = 6,
    GGML_Q5_1 = 7,
    GGML_Q8_0 = 8,
    GGML_Q8_1 = 9,
    GGML_Q2_K = 10,
    GGML_Q3_K = 11,
    GGML_Q4_K = 12,
    GGML_Q5_K = 13,
    GGML_Q6_K = 14,
    GGML_Q8_K = 15,
    GGML_IQ2_XXS = 16,
    GGML_IQ2_XS = 17,
    GGML_IQ3_XXS = 18,
    GGML_IQ1_S = 19,
    GGML_IQ4_NL = 20,
    GGML_IQ3_S = 21,
    GGML_IQ2_S = 22,
    GGML_IQ4_XS = 23,
    GGML_IQ1_M = 29,
    GGML_TQ1_0 = 34,
    GGML_TQ2_0 = 35,
    GGML_MXFP4 = 39,
    GGML_NVFP4 = 40,
    GGML_Q1_0 = 41,
    GGML_Q2_0 = 42,
};

constexpr unsigned int CUDA_WARP_WIDTH = 32;
constexpr unsigned int Q8_0_BLOCK_WIDTH = 32;
constexpr unsigned int Q8_0_BLOCK_BYTES = 34;
constexpr unsigned int Q8_0_SCALE_BYTES = 2;
constexpr unsigned int Q8_0_VECTORS_PER_WARP = 4;
constexpr unsigned int Q8_INPUT_BLOCK_BYTES = 36;

__device__ __forceinline__ int load_i32_unaligned(const unsigned char * source) {
    const size_t address = reinterpret_cast<size_t>(source);
    const unsigned int * aligned = reinterpret_cast<const unsigned int *>(
        address & ~(size_t) (sizeof(unsigned int) - 1));
    const unsigned int shift = (unsigned int) (address & (sizeof(unsigned int) - 1)) * 8;
    return (int) __funnelshift_r(aligned[0], aligned[1], shift);
}

__device__ __forceinline__ void store_q8_input_warp(
        unsigned char * output,
        unsigned int block,
        float value,
        unsigned int lane) {
    float maximum = fabsf(value);
    for (unsigned int offset = CUDA_WARP_WIDTH / 2; offset > 0; offset /= 2) {
        maximum = fmaxf(maximum, __shfl_down_sync(0xffffffff, maximum, offset));
    }
    maximum = __shfl_sync(0xffffffff, maximum, 0);
    const float scale = maximum / 127.0f;
    unsigned char * destination = output + block * Q8_INPUT_BLOCK_BYTES;
    if (lane == 0) {
        *reinterpret_cast<float *>(destination) = scale;
    }
    const int quantized = maximum == 0.0f
        ? 0
        : max(-127, min(127, __float2int_rn(value / scale)));
    *(reinterpret_cast<signed char *>(destination + sizeof(float)) + lane) =
        (signed char) quantized;
}

__device__ __forceinline__ float activated_gate_value(
        float gate,
        unsigned int activation) {
    if (activation == 1) {
        return gate / (1.0f + expf(-gate));
    }
    return 1.0f / (1.0f + expf(-gate));
}

__device__ float block_sum_f32(float value, float * partial) {
    const unsigned int lane = threadIdx.x;
    partial[lane] = value;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (lane < stride) {
            partial[lane] += partial[lane + stride];
        }
        __syncthreads();
    }
    return partial[0];
}

__device__ float block_max_f32(float value, float * partial) {
    const unsigned int lane = threadIdx.x;
    partial[lane] = value;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (lane < stride) {
            partial[lane] = fmaxf(partial[lane], partial[lane + stride]);
        }
        __syncthreads();
    }
    return partial[0];
}

extern "C" __global__ void f32_to_bf16(
        const float * input,
        unsigned short * output,
        unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    unsigned int bits = __float_as_uint(input[index]);
    bits += 0x7fffU + ((bits >> 16) & 1U);
    output[index] = (unsigned short) (bits >> 16);
}

// F16 -> F32 lossless upconvert (prefill weight staging feeds SGEMM). F16 has
// fewer exponent/mantissa bits than F32, so every value expands exactly.
extern "C" __global__ void f16_to_f32(
        const unsigned short * input,
        float * output,
        unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    output[index] = __half2float(__ushort_as_half(input[index]));
}

// BF16 -> F32 lossless upconvert (prefill weight staging feeds SGEMM). BF16 is
// the high 16 bits of the F32 encoding, so the expansion is an exact bit shift
// and matches the resident F32-copy dequant (quant.Dequantize BF16) exactly.
extern "C" __global__ void bf16_to_f32(
        const unsigned short * input,
        float * output,
        unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    output[index] = __uint_as_float(((unsigned int) input[index]) << 16);
}

// e4m3 (OCP F8_E4M3FN) -> f32, exact by construction: every e4m3 value is
// representable in f32, so this is pure integer bit assembly (no float rounding,
// -use_fast_math immune). Bit-identical to Go dtype.F8E4M3ToFloat32. Layout:
// 1 sign, 4 exponent (bias 7), 3 mantissa; NO Inf; the sole NaN is S.1111.111.
__device__ __forceinline__ float fp8_e4m3_decode(unsigned int b) {
    const unsigned int s = (b & 0x80u) << 24;            // e4m3 sign bit7 -> f32 bit31
    const unsigned int e = (b >> 3) & 0x0fu;
    const unsigned int m = b & 0x07u;
    if (e == 0u) {
        // subnormal: m * 2^-9 (m in 0..7, exact in f32); sign OR'd back in
        const float v = (float) m * (1.0f / 512.0f);
        return __uint_as_float(s | __float_as_uint(v));
    }
    if (e == 0x0fu && m == 0x07u) {
        return __uint_as_float(s | 0x7fc00000u);         // E4M3FN NaN, sign preserved
    }
    // normal: (1 + m/8) * 2^(e-7); exponent field e-7+127 = e+120, mantissa m<<20
    return __uint_as_float(s | ((e + 120u) << 23) | (m << 20));
}

// FP8(E4M3) -> F32 prefill upconvert with per-output-row scale folded in. Mirrors
// bf16_to_f32 (prefill weight staging feeds SGEMM) but E4M3 carries a per-row F32
// weight_scale, so the staged F32 weight is scale[row] * e4m3(byte). One block per
// row keeps scale[row] in a register across the row.
extern "C" __global__ void fp8_to_f32(
        const unsigned char * input,
        const float * scale,
        float * output,
        unsigned int inner,
        unsigned int rows) {
    const unsigned int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    const float row_scale = scale[row];
    const unsigned char * weight_row = input + (size_t) row * inner;
    float * output_row = output + (size_t) row * inner;
    for (unsigned int column = threadIdx.x; column < inner; column += blockDim.x) {
        output_row[column] = fp8_e4m3_decode((unsigned int) weight_row[column]) * row_scale;
    }
}

extern "C" __global__ void quantize_q8_0_input_f32(
        const float * input,
        unsigned char * output,
        unsigned int blocks) {
    const unsigned int block = blockIdx.x;
    const unsigned int lane = threadIdx.x;
    if (block >= blocks || lane >= Q8_0_BLOCK_WIDTH) {
        return;
    }
    const float value = input[block * Q8_0_BLOCK_WIDTH + lane];
    store_q8_input_warp(output, block, value, lane);
}

extern "C" __global__ void add_f32(
        const float * input_a,
        const float * input_b,
        float * output,
        unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = input_a[index] + input_b[index];
    }
}

extern "C" __global__ void multiply_f32(
        const float * input_a,
        const float * input_b,
        float * output,
        unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = input_a[index] * input_b[index];
    }
}

extern "C" __global__ void divide_f32(
        const float * input_a,
        const float * input_b,
        float * output,
        unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = input_a[index] / input_b[index];
    }
}

extern "C" __global__ void clamp_f32(
        const float * input,
        float * output,
        float minimum,
        float maximum,
        unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = fminf(fmaxf(input[index], minimum), maximum);
    }
}

extern "C" __global__ void bf16_round_f32(
        const float * input,
        float * output,
        unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        unsigned int raw = __float_as_uint(input[index]);
        if ((raw & 0x7f800000u) != 0x7f800000u) {
            raw += 0x7fffu + ((raw >> 16) & 1u);
            raw &= 0xffff0000u;
        }
        output[index] = __uint_as_float(raw);
    }
}

__device__ unsigned int broadcast_index(
        unsigned int index,
        unsigned int input_0,
        unsigned int input_1,
        unsigned int input_2,
        unsigned int input_3,
        unsigned int output_0,
        unsigned int output_1,
        unsigned int output_2) {
    const unsigned int coordinate_0 = index % output_0;
    index /= output_0;
    const unsigned int coordinate_1 = index % output_1;
    index /= output_1;
    const unsigned int coordinate_2 = index % output_2;
    const unsigned int coordinate_3 = index / output_2;
    return
        (input_0 == 1 ? 0 : coordinate_0) +
        input_0 * (
            (input_1 == 1 ? 0 : coordinate_1) +
            input_1 * (
                (input_2 == 1 ? 0 : coordinate_2) +
                input_2 * (input_3 == 1 ? 0 : coordinate_3)));
}

extern "C" __global__ void broadcast_add_f32(
        const float * input_a,
        const float * input_b,
        float * output,
        unsigned int count,
        unsigned int a_0,
        unsigned int a_1,
        unsigned int a_2,
        unsigned int a_3,
        unsigned int b_0,
        unsigned int b_1,
        unsigned int b_2,
        unsigned int b_3,
        unsigned int output_0,
        unsigned int output_1,
        unsigned int output_2) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        const unsigned int index_a = broadcast_index(
            index, a_0, a_1, a_2, a_3, output_0, output_1, output_2);
        const unsigned int index_b = broadcast_index(
            index, b_0, b_1, b_2, b_3, output_0, output_1, output_2);
        output[index] = input_a[index_a] + input_b[index_b];
    }
}

extern "C" __global__ void broadcast_multiply_f32(
        const float * input_a,
        const float * input_b,
        float * output,
        unsigned int count,
        unsigned int a_0,
        unsigned int a_1,
        unsigned int a_2,
        unsigned int a_3,
        unsigned int b_0,
        unsigned int b_1,
        unsigned int b_2,
        unsigned int b_3,
        unsigned int output_0,
        unsigned int output_1,
        unsigned int output_2) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        const unsigned int index_a = broadcast_index(
            index, a_0, a_1, a_2, a_3, output_0, output_1, output_2);
        const unsigned int index_b = broadcast_index(
            index, b_0, b_1, b_2, b_3, output_0, output_1, output_2);
        output[index] = input_a[index_a] * input_b[index_b];
    }
}

extern "C" __global__ void broadcast_divide_f32(
        const float * input_a,
        const float * input_b,
        float * output,
        unsigned int count,
        unsigned int a_0,
        unsigned int a_1,
        unsigned int a_2,
        unsigned int a_3,
        unsigned int b_0,
        unsigned int b_1,
        unsigned int b_2,
        unsigned int b_3,
        unsigned int output_0,
        unsigned int output_1,
        unsigned int output_2) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        const unsigned int index_a = broadcast_index(
            index, a_0, a_1, a_2, a_3, output_0, output_1, output_2);
        const unsigned int index_b = broadcast_index(
            index, b_0, b_1, b_2, b_3, output_0, output_1, output_2);
        output[index] = input_a[index_a] / input_b[index_b];
    }
}

extern "C" __global__ void scale_f32(
        const float * input,
        float * output,
        float scale,
        unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = input[index] * scale;
    }
}

extern "C" __global__ void copy_f32(
        const float * input,
        float * output,
        unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = input[index];
    }
}

// copy_token_offset_f32: cache append with device-resident row offset so the
// destination argument stays byte-identical across decode steps.
extern "C" __global__ void copy_token_offset_f32(
        const float * input,
        float * output,
        const unsigned int * offset_rows,
        unsigned int inner,
        unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        const unsigned long long base =
            (unsigned long long) offset_rows[0] * (unsigned long long) inner;
        output[base + index] = input[index];
    }
}

extern "C" __global__ void silu_f32(
        const float * input,
        float * output,
        unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        const float value = input[index];
        output[index] = value / (1.0f + expf(-value));
    }
}

// silu_backward_f32: grad_input = grad_output * silu'(x), where
// silu(x) = x*sigmoid(x) and silu'(x) = s*(1 + x*(1-s)), s = sigmoid(x).
extern "C" __global__ void silu_backward_f32(
        const float * grad_output,
        const float * input,
        float * grad_input,
        unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        const float x = input[index];
        const float s = 1.0f / (1.0f + expf(-x));
        grad_input[index] = grad_output[index] * s * (1.0f + x * (1.0f - s));
    }
}

// rms_norm_backward_f32: VJP of affine RMSNorm y = x * rsqrt(mean(x^2)+eps) * w.
// One thread per row. dscale (the weight gradient) accumulates across rows via
// atomicAdd, so the caller must zero-initialize it before launch.
extern "C" __global__ void rms_norm_backward_f32(
        const float * dy,
        const float * x,
        const float * weight,
        float * dx,
        float * dscale,
        unsigned int rows,
        unsigned int d,
        float eps) {
    const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    const float * xr = x + (size_t)row * d;
    const float * dyr = dy + (size_t)row * d;
    float * dxr = dx + (size_t)row * d;
    float ss = 0.0f;
    float dotGX = 0.0f;
    for (unsigned int i = 0; i < d; i++) {
        ss += xr[i] * xr[i];
        dotGX += dyr[i] * weight[i] * xr[i];
    }
    const float inv = rsqrtf(ss / (float)d + eps);
    const float coef = inv * inv * inv / (float)d * dotGX;
    for (unsigned int i = 0; i < d; i++) {
        dxr[i] = inv * weight[i] * dyr[i] - coef * xr[i];
        atomicAdd(&dscale[i], dyr[i] * xr[i] * inv);
    }
}

// softmax_backward_f32: VJP of a row-wise softmax. Given the softmax output p
// and its cotangent dp, ds_i = p_i*(dp_i - sum_j p_j*dp_j). One thread per row.
extern "C" __global__ void softmax_backward_f32(
        const float * p,
        const float * dp,
        float * ds,
        unsigned int rows,
        unsigned int d) {
    const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    const float * pr = p + (size_t)row * d;
    const float * dpr = dp + (size_t)row * d;
    float * dsr = ds + (size_t)row * d;
    float dot = 0.0f;
    for (unsigned int i = 0; i < d; i++) {
        dot += pr[i] * dpr[i];
    }
    for (unsigned int i = 0; i < d; i++) {
        dsr[i] = pr[i] * (dpr[i] - dot);
    }
}

// rope_half_backward_f32: VJP of split-half (HF-layout) rotary embedding. dx is
// [seq, n_heads*hd]; inv_freq is [hd/2]. Rotates each gradient pair (i, i+hd/2)
// by the negated angle pos*inv_freq[i]. One thread per (position, head, pair).
extern "C" __global__ void rope_half_backward_f32(
        float * dx,
        const float * inv_freq,
        unsigned int seq,
        unsigned int n_heads,
        unsigned int hd) {
    const unsigned int half = hd / 2;
    const unsigned int total = seq * n_heads * half;
    const unsigned int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= total) {
        return;
    }
    const unsigned int i = t % half;
    const unsigned int head = (t / half) % n_heads;
    const unsigned int pos = (t / half) / n_heads;
    const float a = (float)pos * inv_freq[i];
    const float c = cosf(a);
    const float s = sinf(a);
    const size_t base = (size_t)pos * n_heads * hd + (size_t)head * hd;
    const float d1 = dx[base + i];
    const float d2 = dx[base + i + half];
    dx[base + i] = c * d1 + s * d2;
    dx[base + i + half] = -s * d1 + c * d2;
}

// rope_half_f32: split-half (HF-layout) rotary embedding forward. x is
// [seq, n_heads*hd]; inv_freq is [hd/2]. Rotates each pair (i, i+hd/2) by angle
// pos*inv_freq[i]. One thread per (position, head, pair). Forward rotation R;
// rope_half_backward_f32 applies its transpose.
extern "C" __global__ void rope_half_f32(
        float * x,
        const float * inv_freq,
        unsigned int seq,
        unsigned int n_heads,
        unsigned int hd) {
    const unsigned int half = hd / 2;
    const unsigned int total = seq * n_heads * half;
    const unsigned int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= total) {
        return;
    }
    const unsigned int i = t % half;
    const unsigned int head = (t / half) % n_heads;
    const unsigned int pos = (t / half) / n_heads;
    const float a = (float)pos * inv_freq[i];
    const float c = cosf(a);
    const float s = sinf(a);
    const size_t base = (size_t)pos * n_heads * hd + (size_t)head * hd;
    const float x1 = x[base + i];
    const float x2 = x[base + i + half];
    x[base + i] = c * x1 - s * x2;
    x[base + i + half] = s * x1 + c * x2;
}

// head_major_f32: reshape [seq, n_heads*hd] (position-major) ->
// [n_heads, seq, hd] (head-major, each head contiguous) so per-head device GEMMs
// address contiguous sub-buffers. One thread per output element (gather).
extern "C" __global__ void head_major_f32(
        const float * input,
        float * output,
        unsigned int seq,
        unsigned int n_heads,
        unsigned int hd) {
    const unsigned int total = seq * n_heads * hd;
    const unsigned int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= total) {
        return;
    }
    const unsigned int x = t % hd;
    const unsigned int i = (t / hd) % seq;
    const unsigned int h = (t / hd) / seq;
    output[t] = input[((size_t)i * n_heads + h) * hd + x];
}

// head_major_inverse_f32: reshape [n_heads, seq, hd] (head-major) ->
// [seq, n_heads*hd] (position-major), the inverse of head_major_f32. One thread
// per output element (gather).
extern "C" __global__ void head_major_inverse_f32(
        const float * input,
        float * output,
        unsigned int seq,
        unsigned int n_heads,
        unsigned int hd) {
    const unsigned int total = seq * n_heads * hd;
    const unsigned int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= total) {
        return;
    }
    const unsigned int x = t % hd;
    const unsigned int h = (t / hd) % n_heads;
    const unsigned int i = (t / hd) / n_heads;
    output[t] = input[((size_t)h * seq + i) * hd + x];
}

extern "C" __global__ void activated_gate_f32(
        const float * gate,
        const float * up,
        float * output,
        unsigned int activation,
        unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = activated_gate_value(gate[index], activation) * up[index];
    }
}

extern "C" __global__ void activated_gate_q8_0_f32(
        const float * gate,
        const float * up,
        float * output,
        unsigned char * quantized,
        unsigned int activation,
        unsigned int count) {
    const unsigned int block = blockIdx.x;
    const unsigned int lane = threadIdx.x;
    const unsigned int index = block * Q8_0_BLOCK_WIDTH + lane;
    if (index >= count || lane >= Q8_0_BLOCK_WIDTH) {
        return;
    }
    const float value = activated_gate_value(gate[index], activation) * up[index];
    output[index] = value;
    store_q8_input_warp(quantized, block, value, lane);
}

extern "C" __global__ void gelu_f32(
        const float * input,
        float * output,
        unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        if (input[index] <= -10.0f) {
            output[index] = 0.0f;
            return;
        }
        if (input[index] >= 10.0f) {
            output[index] = input[index];
            return;
        }
        const float value = __half2float(__float2half_rn(input[index]));
        const float inner = 0.7978845608028654f * value *
                (1.0f + 0.044715f * value * value);
        output[index] = __half2float(__float2half_rn(
                0.5f * value * (1.0f + tanhf(inner))));
    }
}

extern "C" __global__ void gelu_erf_f32(
        const float * input,
        float * output,
        unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        const float value = input[index];
        output[index] = 0.5f * value * (1.0f + erff(value * 0.7071067811865475f));
    }
}

// gelu_tanh_exact_f32: the elementwise tanh-GELU chain fused into one pass,
// replicating the primitive kernel sequence op-for-op (__f*_rn blocks FMA
// contraction, so every intermediate rounds exactly like the separate
// multiply/scale/add/tanh launches it replaces). bias_width != 0 folds the
// preceding broadcast bias add (bias[index % bias_width]) into the same pass.
extern "C" __global__ void gelu_tanh_exact_f32(
        const float * input,
        const float * bias,
        float * output,
        float cubic_coefficient,
        float inner_scale,
        float half_scale,
        unsigned int bias_width,
        unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        float x = input[index];
        if (bias_width != 0u) {
            x = __fadd_rn(x, bias[index % bias_width]);
        }
        const float squared = __fmul_rn(x, x);
        const float cubic = __fmul_rn(squared, x);
        const float shifted = __fadd_rn(x, __fmul_rn(cubic, cubic_coefficient));
        const float inner = __fmul_rn(shifted, inner_scale);
        const float half = __fmul_rn(x, half_scale);
        output[index] = __fadd_rn(half, __fmul_rn(half, tanhf(inner)));
    }
}

extern "C" __global__ void xielu_f32(
        const float * input,
        float * output,
        float alpha_n,
        float alpha_p,
        float beta,
        float epsilon,
        unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        const float value = input[index];
        if (value > 0.0f) {
            output[index] = alpha_p * value * value + beta * value;
        } else {
            const float minimum = fminf(value, epsilon);
            output[index] = (expm1f(minimum) - value) * alpha_n + beta * value;
        }
    }
}

extern "C" __global__ void relu_squared_f32(
        const float * input,
        float * output,
        unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        const float value = input[index];
        output[index] = value > 0.0f ? value * value : 0.0f;
    }
}

extern "C" __global__ void relu_f32(
		const float * input,
		float * output,
		unsigned int count) {
	const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < count) {
		output[index] = fmaxf(input[index], 0.0f);
	}
}

extern "C" __global__ void conv_1d_same_f32(
		const float * input,
		const float * weight,
		const float * bias,
		float * output,
		unsigned int channels_in,
		unsigned int tokens,
		unsigned int kernel,
		unsigned int channels_out,
		unsigned int depthwise,
		unsigned int count) {
	const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= count) {
		return;
	}
	const unsigned int channel_out = index % channels_out;
	const unsigned int token = index / channels_out;
	const int padding = (int) kernel / 2;
	float sum = bias[channel_out];
	for (unsigned int tap = 0; tap < kernel; ++tap) {
		const int source_token = (int) token + (int) tap - padding;
		if (source_token < 0 || source_token >= (int) tokens) {
			continue;
		}
		if (depthwise) {
			sum += input[(unsigned int) source_token * channels_in + channel_out] *
				weight[channel_out * kernel + tap];
			continue;
		}
		for (unsigned int channel_in = 0; channel_in < channels_in; ++channel_in) {
			const unsigned int weight_offset =
				(channel_out * channels_in + channel_in) * kernel + tap;
			sum += input[(unsigned int) source_token * channels_in + channel_in] * weight[weight_offset];
		}
	}
	output[index] = sum;
}

extern "C" __global__ void conv_2d_f32(
		const float * input,
		const float * weight,
		const float * bias,
		float * output,
		unsigned int channels_in,
		unsigned int input_w,
		unsigned int input_h,
		unsigned int kernel_w,
		unsigned int kernel_h,
		unsigned int weight_channels,
		unsigned int channels_out,
		unsigned int output_w,
		unsigned int output_h,
		unsigned int stride_x,
		unsigned int stride_y,
		unsigned int pad_left,
		unsigned int pad_top,
		unsigned int depthwise,
		unsigned int has_bias,
		unsigned int count) {
	const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= count) {
		return;
	}
	const unsigned int channel_out = index % channels_out;
	const unsigned int spatial = index / channels_out;
	const unsigned int x = spatial % output_w;
	const unsigned int y = spatial / output_w;
	float sum = has_bias ? bias[channel_out] : 0.0f;
	for (unsigned int ky = 0; ky < kernel_h; ++ky) {
		const int source_y = (int) (y * stride_y + ky) - (int) pad_top;
		if (source_y < 0 || source_y >= (int) input_h) {
			continue;
		}
		for (unsigned int kx = 0; kx < kernel_w; ++kx) {
			const int source_x = (int) (x * stride_x + kx) - (int) pad_left;
			if (source_x < 0 || source_x >= (int) input_w) {
				continue;
			}
			if (depthwise) {
				const unsigned int input_offset = channel_out + channels_in * ((unsigned int) source_x + input_w * (unsigned int) source_y);
				const unsigned int weight_offset = kx + kernel_w * (ky + kernel_h * channel_out);
				sum += input[input_offset] * weight[weight_offset];
				continue;
			}
			for (unsigned int channel_in = 0; channel_in < channels_in; ++channel_in) {
				const unsigned int input_offset = channel_in + channels_in * ((unsigned int) source_x + input_w * (unsigned int) source_y);
				const unsigned int weight_offset = kx + kernel_w * (ky + kernel_h * (channel_in + weight_channels * channel_out));
				sum += input[input_offset] * weight[weight_offset];
			}
		}
	}
	output[index] = sum;
}

// conv_2d_im2col_f32: bounded spatial tile lowering for cuBLAS Conv2D.
extern "C" __global__ void conv_2d_im2col_f32(
		const float * input,
		float * lowered,
		unsigned int channels_in,
		unsigned int input_w,
		unsigned int input_h,
		unsigned int kernel_w,
		unsigned int kernel_h,
		unsigned int output_w,
		unsigned int output_h,
		unsigned int stride_x,
		unsigned int stride_y,
		unsigned int pad_left,
		unsigned int pad_top,
		unsigned int start,
		unsigned int columns,
		unsigned int inner,
		unsigned int count) {
	const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= count) {
		return;
	}
	const unsigned int column = index / inner;
	unsigned int reduction = index - column * inner;
	const unsigned int kx = reduction % kernel_w;
	reduction /= kernel_w;
	const unsigned int ky = reduction % kernel_h;
	const unsigned int channel = reduction / kernel_h;
	const unsigned int spatial = start + column;
	const unsigned int x = spatial % output_w;
	const unsigned int y = spatial / output_w;
	const int source_x = (int) (x * stride_x + kx) - (int) pad_left;
	const int source_y = (int) (y * stride_y + ky) - (int) pad_top;
	float value = 0.0f;
	if (source_x >= 0 && source_x < (int) input_w &&
			source_y >= 0 && source_y < (int) input_h) {
		value = input[channel + channels_in * ((unsigned int) source_x + input_w * (unsigned int) source_y)];
	}
	lowered[index] = value;
}

extern "C" __global__ void window_partition_2d_f32(
		const float * input,
		float * output,
		unsigned int channels,
		unsigned int width,
		unsigned int height,
		unsigned int window,
		unsigned int count) {
	const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= count) {
		return;
	}
	const unsigned int channel = index % channels;
	unsigned int item = index / channels;
	const unsigned int local_x = item % window;
	item /= window;
	const unsigned int local_y = item % window;
	const unsigned int batch = item / window;
	const unsigned int windows_x = (width + window - 1) / window;
	const unsigned int x = (batch % windows_x) * window + local_x;
	const unsigned int y = (batch / windows_x) * window + local_y;
	output[index] = x < width && y < height ? input[channel + channels * (x + width * y)] : 0.0f;
}

extern "C" __global__ void window_unpartition_2d_f32(
		const float * input,
		float * output,
		unsigned int channels,
		unsigned int width,
		unsigned int height,
		unsigned int window,
		unsigned int count) {
	const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= count) {
		return;
	}
	const unsigned int channel = index % channels;
	const unsigned int spatial = index / channels;
	const unsigned int x = spatial % width;
	const unsigned int y = spatial / width;
	const unsigned int windows_x = (width + window - 1) / window;
	const unsigned int batch = x / window + windows_x * (y / window);
	const unsigned int local_x = x % window;
	const unsigned int local_y = y % window;
	output[index] = input[channel + channels * (local_x + window * (local_y + window * batch))];
}

__device__ float sam_relative_value(
		const float * table,
		unsigned int width,
		unsigned int source_length,
		unsigned int channel,
		unsigned int target_index,
		unsigned int target_length) {
	if (source_length == target_length) {
		return table[channel + width * target_index];
	}
	const float coordinate = ((float) target_index + 0.5f) * (float) source_length / (float) target_length - 0.5f;
	const int raw_low = (int) floorf(coordinate);
	const unsigned int low = (unsigned int) max(0, min((int) source_length - 1, raw_low));
	const unsigned int high = min(source_length - 1, low + 1);
	const float factor = fmaxf(0.0f, fminf(1.0f, coordinate - (float) low));
	return table[channel + width * low] * (1.0f - factor) + table[channel + width * high] * factor;
}

__device__ float sam_attention_score(
		const float * query,
		const float * key,
		const float * relative_w,
		const float * relative_h,
		unsigned int key_width,
		unsigned int query_heads,
		unsigned int key_heads,
		unsigned int tokens,
		unsigned int batch,
		unsigned int query_token,
		unsigned int key_token,
		unsigned int query_head,
		unsigned int key_head,
		unsigned int spatial_size,
		unsigned int relative_w_length,
		unsigned int relative_h_length,
		float scale,
		float relative_scale) {
	const unsigned int query_base = key_width * (query_head + query_heads * (query_token + tokens * batch));
	const unsigned int key_base = key_width * (key_head + key_heads * (key_token + tokens * batch));
	const unsigned int query_x = query_token % spatial_size;
	const unsigned int query_y = query_token / spatial_size;
	const unsigned int key_x = key_token % spatial_size;
	const unsigned int key_y = key_token / spatial_size;
	const unsigned int target_length = 2 * spatial_size - 1;
	float score = 0.0f;
	for (unsigned int channel = 0; channel < key_width; ++channel) {
		const float q = query[query_base + channel];
		score += q * key[key_base + channel] * scale;
		score += q * sam_relative_value(relative_w, key_width, relative_w_length, channel,
			query_x - key_x + spatial_size - 1, target_length) * relative_scale;
		score += q * sam_relative_value(relative_h, key_width, relative_h_length, channel,
			query_y - key_y + spatial_size - 1, target_length) * relative_scale;
	}
	return score;
}

extern "C" __global__ void sam_attention_f32(
		const float * query,
		const float * key,
		const float * value,
		const float * relative_w,
		const float * relative_h,
		float * output,
		unsigned int key_width,
		unsigned int value_width,
		unsigned int query_heads,
		unsigned int key_heads,
		unsigned int tokens,
		unsigned int batches,
		unsigned int spatial_size,
		unsigned int relative_w_length,
		unsigned int relative_h_length,
		float scale,
		float relative_scale,
		unsigned int count) {
	const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= count) {
		return;
	}
	const unsigned int channel = index % value_width;
	unsigned int item = index / value_width;
	const unsigned int query_head = item % query_heads;
	item /= query_heads;
	const unsigned int query_token = item % tokens;
	const unsigned int batch = item / tokens;
	if (batch >= batches) {
		return;
	}
	const unsigned int key_head = query_head / (query_heads / key_heads);
	float maximum = -3.402823466e+38F;
	for (unsigned int key_token = 0; key_token < tokens; ++key_token) {
		maximum = fmaxf(maximum, sam_attention_score(query, key, relative_w, relative_h,
			key_width, query_heads, key_heads, tokens, batch, query_token, key_token,
			query_head, key_head, spatial_size, relative_w_length, relative_h_length, scale, relative_scale));
	}
	float sum = 0.0f;
	float result = 0.0f;
	for (unsigned int key_token = 0; key_token < tokens; ++key_token) {
		const float probability = expf(sam_attention_score(query, key, relative_w, relative_h,
			key_width, query_heads, key_heads, tokens, batch, query_token, key_token,
			query_head, key_head, spatial_size, relative_w_length, relative_h_length, scale, relative_scale) - maximum);
		const unsigned int value_base = value_width * (key_head + key_heads * (key_token + tokens * batch));
		sum += probability;
		result += probability * value[value_base + channel];
	}
	output[index] = result / sum;
}

extern "C" __global__ void group_norm_f32(
		const float * input,
		const float * weight,
		const float * bias,
		float * output,
		unsigned int channels,
		unsigned int tokens,
		unsigned int groups,
		float epsilon,
		unsigned int count) {
	const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= count) {
		return;
	}
	const unsigned int channel = index % channels;
	const unsigned int channels_per_group = channels / groups;
	const unsigned int first_channel = (channel / channels_per_group) * channels_per_group;
	const unsigned int values_per_group = channels_per_group * tokens;
	float mean = 0.0f;
	for (unsigned int token = 0; token < tokens; ++token) {
		for (unsigned int item = 0; item < channels_per_group; ++item) {
			mean += input[token * channels + first_channel + item];
		}
	}
	mean /= (float) values_per_group;
	float variance = 0.0f;
	for (unsigned int token = 0; token < tokens; ++token) {
		for (unsigned int item = 0; item < channels_per_group; ++item) {
			const float delta = input[token * channels + first_channel + item] - mean;
			variance += delta * delta;
		}
	}
	const float normalized = (input[index] - mean) * rsqrtf(variance / (float) values_per_group + epsilon);
	output[index] = normalized * weight[channel] + bias[channel];
}

extern "C" __global__ void sigmoid_f32(
        const float * input,
        float * output,
        unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = 1.0f / (1.0f + expf(-input[index]));
    }
}

extern "C" __global__ void softplus_f32(
        const float * input,
        float * output,
        unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        const float value = input[index];
        output[index] = fmaxf(value, 0.0f) + log1pf(expf(-fabsf(value)));
    }
}

extern "C" __global__ void tanh_f32(
		const float * input,
		float * output,
		unsigned int count) {
	const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < count) {
		output[index] = tanhf(input[index]);
	}
}

extern "C" __global__ void exp_f32(
		const float * input,
		float * output,
		unsigned int count) {
	const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < count) {
		output[index] = expf(input[index]);
	}
}

extern "C" __global__ void l2_norm_f32(
        const float * input,
        float * output,
        unsigned int width,
        unsigned int rows,
        float epsilon) {
    const unsigned int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    const unsigned int offset = row * width;
    float sum_squares = 0.0f;
    for (unsigned int column = threadIdx.x; column < width; column += blockDim.x) {
        const float value = input[offset + column];
        sum_squares += value * value;
    }
    __shared__ float partial[256];
    sum_squares = block_sum_f32(sum_squares, partial);
    const float inverse = 1.0f / fmaxf(sqrtf(sum_squares), epsilon);
    for (unsigned int column = threadIdx.x; column < width; column += blockDim.x) {
        output[offset + column] = input[offset + column] * inverse;
    }
}

// l2_norm_backward_f32: VJP of l2_norm_f32 (per row, width columns). With
// inv = 1/max(sqrt(sum(x^2)),eps): unclamped (norm>eps)
// dX = inv*dY - inv^3 * x * (dY.x); clamped (norm<=eps) dX = inv*dY. One thread
// per row; row math in double to mirror the host f64 golden (L2NormBackward).
extern "C" __global__ void l2_norm_backward_f32(
        const float * input,
        const float * grad_output,
        float * grad_input,
        unsigned int width,
        unsigned int rows,
        float epsilon) {
    const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    const unsigned int offset = row * width;
    double sum_squares = 0.0;
    for (unsigned int column = 0; column < width; ++column) {
        const double value = (double)input[offset + column];
        sum_squares += value * value;
    }
    const double norm = sqrt(sum_squares);
    const double inverse = 1.0 / fmax(norm, (double)epsilon);
    const bool clamped = norm <= (double)epsilon;
    double dot = 0.0;
    if (!clamped) {
        for (unsigned int column = 0; column < width; ++column) {
            dot += (double)grad_output[offset + column] * (double)input[offset + column];
        }
    }
    const double inverse_cubed = inverse * inverse * inverse;
    for (unsigned int column = 0; column < width; ++column) {
        double gradient = inverse * (double)grad_output[offset + column];
        if (!clamped) {
            gradient -= inverse_cubed * (double)input[offset + column] * dot;
        }
        grad_input[offset + column] = (float)gradient;
    }
}

// short_conv_backward_f32: VJP of SiLU(depthwise causal conv1d, left-pad k-1) --
// the qwen3.5 GDN-mix short convolution (host ShortConvForward). x/grad_output/
// grad_input are [channels, tokens] channel-major; weights/grad_weights are
// [channels, k]; bias/grad_bias are per channel (has_bias==0 skips both). One
// thread per channel (channels independent). The per-token SiLU'(pre)*dY product
// is staged into the dconv_scratch double buffer, then reused for grad_weights
// (per tap) and grad_input (per position); all accumulation in double to mirror
// the host f64 golden (ShortConvBackward).
extern "C" __global__ void short_conv_backward_f32(
        const float * x,
        const float * grad_output,
        const float * weights,
        const float * bias,
        float * grad_input,
        float * grad_weights,
        float * grad_bias,
        double * dconv_scratch,
        unsigned int channels,
        unsigned int tokens,
        unsigned int k,
        unsigned int has_bias) {
    const unsigned int channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= channels) {
        return;
    }
    const float * x_row = x + (size_t)channel * tokens;
    const float * dy_row = grad_output + (size_t)channel * tokens;
    const float * weight_row = weights + (size_t)channel * k;
    double * dconv = dconv_scratch + (size_t)channel * tokens;
    double dbias = 0.0;
    for (unsigned int t = 0; t < tokens; ++t) {
        double pre = has_bias ? (double)bias[channel] : 0.0;
        for (unsigned int j = 0; j < k; ++j) {
            const int ti = (int)t - (int)(k - 1) + (int)j;
            if (ti < 0 || ti >= (int)tokens) {
                continue;
            }
            pre += (double)x_row[ti] * (double)weight_row[j];
        }
        const double s = 1.0 / (1.0 + exp(-pre));
        const double dc = (double)dy_row[t] * s * (1.0 + pre * (1.0 - s));
        dconv[t] = dc;
        dbias += dc;
    }
    if (has_bias) {
        grad_bias[channel] = (float)dbias;
    }
    for (unsigned int j = 0; j < k; ++j) {
        double dweight = 0.0;
        for (unsigned int t = 0; t < tokens; ++t) {
            const int ti = (int)t - (int)(k - 1) + (int)j;
            if (ti < 0 || ti >= (int)tokens) {
                continue;
            }
            dweight += dconv[t] * (double)x_row[ti];
        }
        grad_weights[(size_t)channel * k + j] = (float)dweight;
    }
    for (unsigned int p = 0; p < tokens; ++p) {
        double dx = 0.0;
        for (unsigned int j = 0; j < k; ++j) {
            const int t = (int)p + (int)(k - 1) - (int)j;
            if (t < 0 || t >= (int)tokens) {
                continue;
            }
            dx += dconv[t] * (double)weight_row[j];
        }
        grad_input[(size_t)channel * tokens + p] = (float)dx;
    }
}

extern "C" __global__ void ssm_conv_f32(
        const float * input,
        const float * weights,
        float * output,
        unsigned int window,
        unsigned int channels,
        unsigned int tokens,
        unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const unsigned int channel = index % channels;
    const unsigned int row = index / channels;
    const unsigned int token = row % tokens;
    const unsigned int sequence = row / tokens;
    const unsigned int kernel_size = window - tokens + 1;
    const unsigned int input_base =
        sequence * window * channels + channel * window + token;
    const unsigned int weight_base = channel * kernel_size;
    float sum = 0.0f;
    for (unsigned int tap = 0; tap < kernel_size; ++tap) {
        sum += input[input_base + tap] * weights[weight_base + tap];
    }
    output[index] = sum;
}

extern "C" __global__ void ssm_scan_f32(
        const float * input_state,
        const float * x,
        const float * dt,
        const float * a,
        const float * beta,
        const float * c,
        float * output,
        unsigned int state_width,
        unsigned int dimension,
        unsigned int heads,
        unsigned int tokens,
        unsigned int sequences,
        unsigned int groups,
        unsigned int a_width) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int count = heads * sequences;
    if (index >= count) {
        return;
    }
    const unsigned int head = index % heads;
    const unsigned int sequence = index / heads;
    const unsigned int group = head / (heads / groups);
    const unsigned int attention_elements =
        dimension * heads * tokens * sequences;
    const unsigned int state_base =
        attention_elements + index * dimension * state_width;
    const unsigned int source_state_base =
        index * dimension * state_width;
    for (unsigned int i = 0; i < dimension * state_width; ++i) {
        output[state_base + i] = input_state[source_state_base + i];
    }
    for (unsigned int token = 0; token < tokens; ++token) {
        float delta = dt[head + heads * (token + tokens * sequence)];
        delta = fmaxf(delta, 0.0f) + log1pf(expf(-fabsf(delta)));
        for (unsigned int inner = 0; inner < dimension; ++inner) {
            const unsigned int x_index =
                inner + dimension * (head + heads * (token + tokens * sequence));
            const float x_delta = x[x_index] * delta;
            float sum = 0.0f;
            for (unsigned int column = 0; column < state_width; ++column) {
                const unsigned int state_index =
                    state_base + inner * state_width + column;
                const unsigned int bc_index = column + state_width *
                    (group + groups * (token + tokens * sequence));
                const float next = output[state_index] *
                    expf(delta * a[(column % a_width) + a_width * head]) +
                    beta[bc_index] * x_delta;
                output[state_index] = next;
                sum += next * c[bc_index];
            }
            output[x_index] = sum;
        }
    }
}

extern "C" __global__ void transpose_2d_f32(
        const float * input,
        float * output,
        unsigned int width,
        unsigned int rows,
        unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        const unsigned int old_row = index % rows;
        const unsigned int old_column = index / rows;
        output[index] = input[old_column + width * old_row];
    }
}

extern "C" __global__ void group_slice_f32(
        const float * input,
        float * output,
        unsigned int input_width,
        unsigned int offset,
        unsigned int width,
        unsigned int groups,
        unsigned int stride,
        unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        const unsigned int column = index % width;
        const unsigned int remainder = index / width;
        const unsigned int group = remainder % groups;
        const unsigned int outer = remainder / groups;
        output[index] = input[
            outer * input_width + offset + group * stride + column];
    }
}

extern "C" __global__ void flat_slice_f32(
        const float * input,
        float * output,
        unsigned int offset,
        unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = input[offset + index];
    }
}

extern "C" __global__ void gated_delta_net_f32(
        const float * query,
        const float * key,
        const float * value,
        const float * gate,
        const float * beta,
        const float * input_state,
        float * output,
        unsigned int size,
        unsigned int query_heads,
        unsigned int key_heads,
        unsigned int heads,
        unsigned int tokens,
        unsigned int sequences,
        unsigned int gate_width,
        unsigned int repeat_interleave) {
    const unsigned int index = blockIdx.x;
    const unsigned int count = heads * sequences;
    if (index >= count) {
        return;
    }
    const unsigned int head = index % heads;
    const unsigned int sequence = index / heads;
    const unsigned int attention_elements =
        size * heads * tokens * sequences;
    const unsigned int state_elements = size * size;
    float * state = output + attention_elements + index * state_elements;
    const float * source_state = input_state + index * state_elements;
    const float scale = rsqrtf((float) size);
    const unsigned int lane = threadIdx.x % CUDA_WARP_WIDTH;
    const unsigned int warp = threadIdx.x / CUDA_WARP_WIDTH;
    const unsigned int warps = blockDim.x / CUDA_WARP_WIDTH;
    for (unsigned int row = warp; row < size; row += warps) {
        float * state_row = state + row * size;
        const float * source_row = source_state + row * size;
        for (unsigned int column = lane; column < size; column += CUDA_WARP_WIDTH) {
            state_row[column] = source_row[column];
        }
        for (unsigned int token = 0; token < tokens; ++token) {
            const unsigned int value_base =
                ((sequence * tokens + token) * heads + head) * size;
            const unsigned int query_head = repeat_interleave
                ? head / (heads / query_heads)
                : head % query_heads;
            const unsigned int key_head = repeat_interleave
                ? head / (heads / key_heads)
                : head % key_heads;
            const unsigned int query_base =
                ((sequence * tokens + token) * query_heads + query_head) * size;
            const unsigned int key_base =
                ((sequence * tokens + token) * key_heads + key_head) * size;
            const unsigned int gate_base =
                ((sequence * tokens + token) * heads + head) * gate_width;
            const float beta_value =
                beta[(sequence * tokens + token) * heads + head];
            if (gate_width == 1) {
                const float gate_scale = expf(gate[gate_base]);
                for (unsigned int column = lane; column < size; column += CUDA_WARP_WIDTH) {
                    state_row[column] *= gate_scale;
                }
            } else {
                for (unsigned int column = lane; column < size; column += CUDA_WARP_WIDTH) {
                    state_row[column] *= expf(gate[gate_base + column]);
                }
            }
            float dot = 0.0f;
            for (unsigned int column = lane; column < size; column += CUDA_WARP_WIDTH) {
                dot += state_row[column] * key[key_base + column];
            }
            for (unsigned int offset = CUDA_WARP_WIDTH / 2; offset > 0; offset /= 2) {
                dot += __shfl_down_sync(0xffffffff, dot, offset);
            }
            dot = __shfl_sync(0xffffffff, dot, 0);
            const float delta =
                (value[value_base + row] - dot) * beta_value;
            for (unsigned int column = lane; column < size; column += CUDA_WARP_WIDTH) {
                state_row[column] += delta * key[key_base + column];
            }
            dot = 0.0f;
            for (unsigned int column = lane; column < size; column += CUDA_WARP_WIDTH) {
                dot += state_row[column] * query[query_base + column];
            }
            for (unsigned int offset = CUDA_WARP_WIDTH / 2; offset > 0; offset /= 2) {
                dot += __shfl_down_sync(0xffffffff, dot, offset);
            }
            if (lane == 0) {
                output[value_base + row] = dot * scale;
            }
        }
    }
}

// gated_delta_net_backward_f32: VJP of gated_delta_net_f32, the BPTT through the
// per-(sequence,head,row) gated delta recurrence. One thread per (sequence,head,
// row): the token loop is inherently sequential per row, rows are independent.
// A forward recompute pass writes each token's post-update state s_t into the
// safter scratch; sprime (post-gate, pre-delta) is recomputed from safter[t-1].
// The reverse pass carries ds through ds_carry scratch. Internal math is double
// to mirror the host f64 reference; grouped q/k, dgate, and dbeta accumulate
// across rows/heads via atomicAdd, so the caller must zero-initialize
// d_query/d_key/d_value/d_gate/d_beta before launch (d_input_state is written
// once per element). scratch buffers safter (count*size*tokens*size) and
// ds_carry (count*size*size) are doubles; they need no pre-init.
extern "C" __global__ void gated_delta_net_backward_f32(
        const float * query,
        const float * key,
        const float * value,
        const float * gate,
        const float * beta,
        const float * input_state,
        const float * d_output,
        float * d_query,
        float * d_key,
        float * d_value,
        float * d_gate,
        float * d_beta,
        float * d_input_state,
        double * safter,
        double * ds_carry,
        unsigned int size,
        unsigned int query_heads,
        unsigned int key_heads,
        unsigned int heads,
        unsigned int tokens,
        unsigned int sequences,
        unsigned int gate_width,
        unsigned int repeat_interleave) {
    const unsigned int index = blockIdx.x;
    const unsigned int count = heads * sequences;
    if (index >= count) {
        return;
    }
    const unsigned int head = index % heads;
    const unsigned int sequence = index / heads;
    const unsigned int query_head = repeat_interleave
        ? head / (heads / query_heads)
        : head % query_heads;
    const unsigned int key_head = repeat_interleave
        ? head / (heads / key_heads)
        : head % key_heads;
    const double scale = 1.0 / sqrt((double) size);
    const unsigned int state_elements = size * size;
    for (unsigned int row = threadIdx.x; row < size; row += blockDim.x) {
        const float * state_in = input_state + index * state_elements + row * size;
        double * row_safter = safter + ((size_t)(index * size + row)) * tokens * size;
        double * ds = ds_carry + ((size_t)(index * size + row)) * size;
        // Forward recompute: write s_t (post-update state) into row_safter[t].
        for (unsigned int token = 0; token < tokens; ++token) {
            const unsigned int value_base =
                ((sequence * tokens + token) * heads + head) * size;
            const unsigned int key_base =
                ((sequence * tokens + token) * key_heads + key_head) * size;
            const unsigned int gate_base =
                ((sequence * tokens + token) * heads + head) * gate_width;
            const double beta_value =
                (double) beta[(sequence * tokens + token) * heads + head];
            const double * prev = token == 0
                ? (const double *) 0
                : row_safter + (size_t)(token - 1) * size;
            double dot = 0.0;
            for (unsigned int column = 0; column < size; ++column) {
                const double g = gate_width == 1
                    ? (double) gate[gate_base]
                    : (double) gate[gate_base + column];
                const double base = token == 0
                    ? (double) state_in[column]
                    : prev[column];
                dot += base * exp(g) * (double) key[key_base + column];
            }
            const double delta =
                ((double) value[value_base + row] - dot) * beta_value;
            double * cur = row_safter + (size_t)token * size;
            for (unsigned int column = 0; column < size; ++column) {
                const double g = gate_width == 1
                    ? (double) gate[gate_base]
                    : (double) gate[gate_base + column];
                const double base = token == 0
                    ? (double) state_in[column]
                    : prev[column];
                cur[column] = base * exp(g) + delta * (double) key[key_base + column];
            }
        }
        // Reverse pass over tokens.
        for (unsigned int column = 0; column < size; ++column) {
            ds[column] = 0.0;
        }
        for (int token = (int) tokens - 1; token >= 0; --token) {
            const unsigned int value_base =
                ((sequence * tokens + token) * heads + head) * size;
            const unsigned int query_base =
                ((sequence * tokens + token) * query_heads + query_head) * size;
            const unsigned int key_base =
                ((sequence * tokens + token) * key_heads + key_head) * size;
            const unsigned int gate_base =
                ((sequence * tokens + token) * heads + head) * gate_width;
            const unsigned int beta_index =
                (sequence * tokens + token) * heads + head;
            const double beta_value = (double) beta[beta_index];
            const double go_value = (double) d_output[value_base + row] * scale;
            const double * saf_cur = row_safter + (size_t)token * size;
            const double * prev = token == 0
                ? (const double *) 0
                : row_safter + (size_t)(token - 1) * size;
            // dsnew = ds + go*q ; dQuery += go*s_t ; dDelta = dsnew . k
            double d_delta = 0.0;
            for (unsigned int column = 0; column < size; ++column) {
                const double dsnew = ds[column] + go_value * (double) query[query_base + column];
                atomicAdd(&d_query[query_base + column], (float)(go_value * saf_cur[column]));
                d_delta += dsnew * (double) key[key_base + column];
            }
            // recompute dot from s'_t (= prev . exp(gate))
            double dot = 0.0;
            for (unsigned int column = 0; column < size; ++column) {
                const double g = gate_width == 1
                    ? (double) gate[gate_base]
                    : (double) gate[gate_base + column];
                const double base = token == 0
                    ? (double) state_in[column]
                    : prev[column];
                dot += base * exp(g) * (double) key[key_base + column];
            }
            const double v_minus = (double) value[value_base + row] - dot;
            const double delta = v_minus * beta_value;
            atomicAdd(&d_value[value_base + row], (float)(d_delta * beta_value));
            atomicAdd(&d_beta[beta_index], (float)(d_delta * v_minus));
            const double d_dot = -d_delta * beta_value;
            // dSp = dsnew + d_dot*k ; dKey += delta*dsnew + d_dot*s' ;
            // dGate += dSp*s' ; ds = dSp*exp(gate)
            double gate_accum = 0.0;
            for (unsigned int column = 0; column < size; ++column) {
                const double g = gate_width == 1
                    ? (double) gate[gate_base]
                    : (double) gate[gate_base + column];
                const double gexp = exp(g);
                const double sprime = (token == 0
                    ? (double) state_in[column]
                    : prev[column]) * gexp;
                const double kc = (double) key[key_base + column];
                const double dsnew = ds[column] + go_value * (double) query[query_base + column];
                const double dsp = dsnew + d_dot * kc;
                atomicAdd(&d_key[key_base + column], (float)(delta * dsnew + d_dot * sprime));
                if (gate_width == 1) {
                    gate_accum += dsp * sprime;
                } else {
                    atomicAdd(&d_gate[gate_base + column], (float)(dsp * sprime));
                }
                ds[column] = dsp * gexp;
            }
            if (gate_width == 1) {
                atomicAdd(&d_gate[gate_base], (float) gate_accum);
            }
        }
        for (unsigned int column = 0; column < size; ++column) {
            d_input_state[index * state_elements + row * size + column] = (float) ds[column];
        }
    }
}

extern "C" __global__ void gated_linear_attention_f32(
		const float * key,
		const float * value,
		const float * receptance,
		const float * decay,
		const float * input_state,
		float * output,
		unsigned int width,
		unsigned int key_heads,
		unsigned int heads,
		unsigned int tokens,
		unsigned int sequences,
		float scale) {
	const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int count = heads * sequences;
	if (index >= count) {
		return;
	}
	const unsigned int head = index % heads;
	const unsigned int key_head = head / (heads / key_heads);
	const unsigned int sequence = index / heads;
	const unsigned int attention_elements = width * heads * tokens * sequences;
	const unsigned int state_elements = width * width;
	float * state = output + attention_elements + index * state_elements;
	const float * source_state = input_state + index * state_elements;
	for (unsigned int i = 0; i < state_elements; ++i) {
		state[i] = source_state[i];
	}
	for (unsigned int token = 0; token < tokens; ++token) {
		const unsigned int vector_base =
			((sequence * tokens + token) * heads + head) * width;
		const unsigned int key_base =
			((sequence * tokens + token) * key_heads + key_head) * width;
		for (unsigned int row = 0; row < width; ++row) {
			float sum = 0.0f;
			float * state_row = state + row * width;
			for (unsigned int column = 0; column < width; ++column) {
				const unsigned int vector_index = vector_base + column;
				const float effective_key = key[key_base + column] *
					(1.0f - decay[vector_index]);
				const float next = state_row[column] * decay[vector_index] +
					effective_key * value[key_base + row];
				state_row[column] = next;
				sum += receptance[vector_index] * next;
			}
			output[vector_base + row] = sum * scale;
		}
	}
}

extern "C" __global__ void rwkv6_f32(
		const float * key,
		const float * value,
		const float * receptance,
		const float * first,
		const float * decay,
		const float * input_state,
		float * output,
		unsigned int width,
		unsigned int heads,
		unsigned int tokens,
		unsigned int sequences) {
	const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int count = heads * sequences;
	if (index >= count) {
		return;
	}
	const unsigned int head = index % heads;
	const unsigned int sequence = index / heads;
	const unsigned int attention_elements = width * heads * tokens * sequences;
	const unsigned int state_elements = width * width;
	float * state = output + attention_elements + index * state_elements;
	const float * source_state = input_state + index * state_elements;
	for (unsigned int i = 0; i < state_elements; ++i) {
		state[i] = source_state[i];
	}
	for (unsigned int token = 0; token < tokens; ++token) {
		const unsigned int vector_base =
			((sequence * tokens + token) * heads + head) * width;
		for (unsigned int column = 0; column < width; ++column) {
			output[vector_base + column] = 0.0f;
		}
		for (unsigned int row = 0; row < width; ++row) {
			float * state_row = state + row * width;
			const float key_value = key[vector_base + row];
			const float receptance_value = receptance[vector_base + row];
			const float first_value = first[head * width + row];
			const float decay_value = decay[vector_base + row];
			for (unsigned int column = 0; column < width; ++column) {
				const float kv = key_value * value[vector_base + column];
				const float previous = state_row[column];
				output[vector_base + column] +=
					(previous + kv * first_value) * receptance_value;
				state_row[column] = previous * decay_value + kv;
			}
		}
	}
}

extern "C" __global__ void sum_rows_f32(
		const float * input,
		float * output,
		unsigned int width,
		unsigned int rows) {
	const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row >= rows) {
		return;
	}
	float sum = 0.0f;
	for (unsigned int column = 0; column < width; ++column) {
		sum += input[row * width + column];
	}
	output[row] = sum;
}

extern "C" __global__ void fwht_f32(
		const float * input,
		float * output,
		unsigned int width,
		unsigned int rows) {
	const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row >= rows) {
		return;
	}
	const size_t base = (size_t) row * width;
	for (unsigned int column = 0; column < width; ++column) {
		output[base + column] = input[base + column];
	}
	for (unsigned int stride = 1; stride < width; stride *= 2) {
		for (unsigned int block = 0; block < width; block += 2 * stride) {
			for (unsigned int offset = 0; offset < stride; ++offset) {
				const size_t first = base + block + offset;
				const size_t second = first + stride;
				const float a = output[first];
				const float b = output[second];
				output[first] = a + b;
				output[second] = a - b;
			}
		}
	}
	const float scale = rsqrtf((float) width);
	for (unsigned int column = 0; column < width; ++column) {
		output[base + column] *= scale;
	}
}

__device__ bool top_k_before(
		float left,
		unsigned int left_index,
		float right,
		unsigned int right_index) {
	const bool left_nan = isnan(left);
	const bool right_nan = isnan(right);
	if (left_nan != right_nan) {
		return !left_nan;
	}
	if (left == right || left_nan) {
		return left_index < right_index;
	}
	return left > right;
}

extern "C" __global__ void argmax_f32(
		const float * input,
		float * output,
		unsigned int width,
		unsigned int rows) {
	const unsigned int row = blockIdx.x;
	const unsigned int lane = threadIdx.x;
	if (row >= rows) {
		return;
	}
	const size_t base = (size_t) row * width;
	unsigned int best_index = 0xffffffffU;
	float best_value = 0.0f;
	unsigned int invalid = 0;
	for (unsigned int candidate = lane; candidate < width; candidate += blockDim.x) {
		const float value = input[base + candidate];
		invalid |= isnan(value);
		if (best_index == 0xffffffffU || top_k_before(value, candidate, best_value, best_index)) {
			best_index = candidate;
			best_value = value;
		}
	}
	__shared__ float values[256];
	__shared__ unsigned int indices[256];
	__shared__ unsigned int invalids[256];
	values[lane] = best_value;
	indices[lane] = best_index;
	invalids[lane] = invalid;
	__syncthreads();
	for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
		if (lane < stride) {
			invalids[lane] |= invalids[lane + stride];
			const unsigned int right_index = indices[lane + stride];
			if (right_index != 0xffffffffU &&
				(indices[lane] == 0xffffffffU || top_k_before(
					values[lane + stride], right_index, values[lane], indices[lane]))) {
				values[lane] = values[lane + stride];
				indices[lane] = right_index;
			}
		}
		__syncthreads();
	}
	if (lane == 0) {
		output[row] = invalids[0] ? -1.0f : (float) indices[0];
	}
}

extern "C" __global__ void top_k_f32(
		const float * input,
		float * output,
		unsigned int width,
		unsigned int k,
		unsigned int rows) {
	const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row >= rows) {
		return;
	}
	const size_t input_base = (size_t) row * width;
	const size_t output_base = (size_t) row * k;
	for (unsigned int slot = 0; slot < k; ++slot) {
		output[output_base + slot] = -1.0f;
	}
	for (unsigned int candidate = 0; candidate < width; ++candidate) {
		unsigned int insertion = k;
		const float candidate_value = input[input_base + candidate];
		for (unsigned int slot = 0; slot < k; ++slot) {
			const int selected = (int) output[output_base + slot];
			if (selected < 0 || top_k_before(
					candidate_value, candidate,
					input[input_base + (unsigned int) selected], (unsigned int) selected)) {
				insertion = slot;
				break;
			}
		}
		if (insertion == k) {
			continue;
		}
		for (unsigned int slot = k - 1; slot > insertion; --slot) {
			output[output_base + slot] = output[output_base + slot - 1];
		}
		output[output_base + insertion] = (float) candidate;
	}
}

extern "C" __global__ void top_k_pairs_f32(
		const float * partials,
		float * output,
		unsigned int candidates,
		unsigned int k,
		unsigned int rows) {
	constexpr unsigned int max_k = 64;
	constexpr unsigned int pair_values = 2;
	const unsigned int row = blockIdx.x;
	const unsigned int lane = threadIdx.x;
	if (row >= rows || k > max_k) {
		return;
	}
	const size_t input_base = (size_t) row * candidates * pair_values;
	const size_t output_base = (size_t) row * k * pair_values;
	unsigned int selected[max_k];
	__shared__ float values[256];
	__shared__ unsigned int indices[256];
	for (unsigned int slot = 0; slot < k; ++slot) {
		float best_value = 0.0f;
		unsigned int best = 0xffffffffU;
		for (unsigned int candidate = lane; candidate < candidates; candidate += blockDim.x) {
			const size_t pair = input_base + candidate * pair_values;
			const float raw = partials[pair];
			if (raw < 0.0f) {
				continue;
			}
			const unsigned int id = (unsigned int) raw;
			bool used = false;
			for (unsigned int previous = 0; previous < slot; ++previous) {
				used |= selected[previous] == id;
			}
			const float value = partials[pair + 1];
			if (!used && (best == 0xffffffffU || top_k_before(value, id, best_value, best))) {
				best = id;
				best_value = value;
			}
		}
		values[lane] = best_value;
		indices[lane] = best;
		__syncthreads();
		for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
			if (lane < stride) {
				const unsigned int right = indices[lane + stride];
				if (right != 0xffffffffU &&
					(indices[lane] == 0xffffffffU || top_k_before(
						values[lane + stride], right, values[lane], indices[lane]))) {
					values[lane] = values[lane + stride];
					indices[lane] = right;
				}
			}
			__syncthreads();
		}
		selected[slot] = indices[0];
		if (lane == 0) {
			output[output_base + slot * pair_values] = (float) indices[0];
			output[output_base + slot * pair_values + 1] = values[0];
		}
		__syncthreads();
	}
}

extern "C" __global__ void top_k_partials_f32(
		const float * input,
		float * output,
		unsigned int width,
		unsigned int k,
		unsigned int chunk,
		unsigned int chunks,
		unsigned int rows) {
	constexpr unsigned int max_k = 64;
	constexpr unsigned int pair_values = 2;
	const unsigned int index = blockIdx.x;
	const unsigned int lane = threadIdx.x;
	if (index >= chunks * rows || k > max_k) {
		return;
	}
	const unsigned int row = index / chunks;
	const unsigned int part = index % chunks;
	const unsigned int first = part * chunk;
	const unsigned int last = min(first + chunk, width);
	const size_t input_base = (size_t) row * width;
	const size_t output_base = (size_t) index * k * pair_values;
	unsigned int selected[max_k];
	for (unsigned int slot = 0; slot < k; ++slot) {
		float best_value = 0.0f;
		unsigned int best = 0xffffffffU;
		for (unsigned int candidate = first + lane; candidate < last; candidate += CUDA_WARP_WIDTH) {
			bool used = false;
			for (unsigned int previous = 0; previous < slot; ++previous) {
				used |= selected[previous] == candidate;
			}
			const float value = input[input_base + candidate];
			if (!used && (best == 0xffffffffU || top_k_before(value, candidate, best_value, best))) {
				best = candidate;
				best_value = value;
			}
		}
		for (unsigned int offset = CUDA_WARP_WIDTH / 2; offset > 0; offset >>= 1) {
			const float right_value = __shfl_down_sync(0xffffffff, best_value, offset);
			const unsigned int right = __shfl_down_sync(0xffffffff, best, offset);
			if (right != 0xffffffffU &&
				(best == 0xffffffffU || top_k_before(right_value, right, best_value, best))) {
				best = right;
				best_value = right_value;
			}
		}
		best = __shfl_sync(0xffffffff, best, 0);
		best_value = __shfl_sync(0xffffffff, best_value, 0);
		selected[slot] = best;
		if (lane == 0) {
			output[output_base + slot * pair_values] =
				best == 0xffffffffU ? -1.0f : (float) best;
			output[output_base + slot * pair_values + 1] = best_value;
		}
	}
}

extern "C" __global__ void gather_last_f32(
		const float * input,
		const float * indices,
		float * output,
		unsigned int inner,
		unsigned int index_count,
		unsigned int input_rows,
		unsigned int count) {
	const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= count) {
		return;
	}
	const unsigned int inner_index = index % inner;
	const unsigned int index_position = index / inner;
	if (index_position >= index_count) {
		return;
	}
	const float raw = indices[index_position];
	if (!isfinite(raw) || raw < 0.0f || raw >= (float) input_rows) {
		output[index] = 0.0f;
		return;
	}
	const unsigned int row = (unsigned int) raw;
	if (raw != (float) row) {
		output[index] = 0.0f;
		return;
	}
	output[index] = input[(size_t) row * inner + inner_index];
}

extern "C" __global__ void gather_last_q8_0_f32(
		const unsigned char * input,
		const float * indices,
		float * output,
		unsigned int inner,
		unsigned int index_count,
		unsigned int input_rows,
		unsigned int count) {
	const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= count) {
		return;
	}
	const unsigned int column = index % inner;
	const unsigned int index_position = index / inner;
	if (index_position >= index_count) {
		return;
	}
	const float raw = indices[index_position];
	if (!isfinite(raw) || raw < 0.0f || raw >= (float) input_rows) {
		output[index] = 0.0f;
		return;
	}
	const unsigned int row = (unsigned int) raw;
	if (raw != (float) row) {
		output[index] = 0.0f;
		return;
	}
	const unsigned int blocks_per_row = inner / Q8_0_BLOCK_WIDTH;
	const unsigned int block_index = row * blocks_per_row + column / Q8_0_BLOCK_WIDTH;
	const unsigned char * block = input + block_index * Q8_0_BLOCK_BYTES;
	const float scale = __half2float(*reinterpret_cast<const __half *>(block));
	const signed char quantized = *(reinterpret_cast<const signed char *>(
		block + Q8_0_SCALE_BYTES) + column % Q8_0_BLOCK_WIDTH);
	output[index] = scale * (float) quantized;
}

extern "C" __global__ void sparse_attention_f32(
		const float * query,
		const float * key,
		const float * value,
		const float * indices,
		float * output,
		unsigned int key_width,
		unsigned int value_width,
		unsigned int query_heads,
		unsigned int key_value_heads,
		unsigned int query_tokens,
		unsigned int key_value_tokens,
		unsigned int selected,
		float scale,
		unsigned int causal,
		unsigned int query_start,
		unsigned int count) {
	const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= count) {
		return;
	}
	const unsigned int value_channel = index % value_width;
	const unsigned int row = index / value_width;
	const unsigned int query_head = row % query_heads;
	const unsigned int query_token = row / query_heads;
	if (query_token >= query_tokens) {
		return;
	}
	const unsigned int key_value_head = query_head / (query_heads / key_value_heads);
	const size_t query_offset =
		((size_t) query_token * query_heads + query_head) * key_width;
	const size_t index_base = (size_t) query_token * selected;
	float maximum = -3.402823466e+38F;
	unsigned int valid_count = 0;
	for (unsigned int slot = 0; slot < selected; ++slot) {
		const float raw = indices[index_base + slot];
		if (!isfinite(raw) || raw < 0.0f || raw >= (float) key_value_tokens) {
			continue;
		}
		const unsigned int key_token = (unsigned int) raw;
		if (raw != (float) key_token || causal && key_token > query_start + query_token) {
			continue;
		}
		const size_t key_offset =
			((size_t) key_token * key_value_heads + key_value_head) * key_width;
		float dot = 0.0f;
		for (unsigned int channel = 0; channel < key_width; ++channel) {
			dot += query[query_offset + channel] * key[key_offset + channel];
		}
		maximum = fmaxf(maximum, dot * scale);
		++valid_count;
	}
	if (valid_count == 0) {
		output[index] = 0.0f;
		return;
	}
	float sum = 0.0f;
	float weighted = 0.0f;
	for (unsigned int slot = 0; slot < selected; ++slot) {
		const float raw = indices[index_base + slot];
		if (!isfinite(raw) || raw < 0.0f || raw >= (float) key_value_tokens) {
			continue;
		}
		const unsigned int key_token = (unsigned int) raw;
		if (raw != (float) key_token || causal && key_token > query_start + query_token) {
			continue;
		}
		const size_t key_offset =
			((size_t) key_token * key_value_heads + key_value_head) * key_width;
		float dot = 0.0f;
		for (unsigned int channel = 0; channel < key_width; ++channel) {
			dot += query[query_offset + channel] * key[key_offset + channel];
		}
		const float probability = expf(dot * scale - maximum);
		const size_t value_offset =
			((size_t) key_token * key_value_heads + key_value_head) * value_width;
		sum += probability;
		weighted += probability * value[value_offset + value_channel];
	}
	output[index] = weighted / sum;
}

extern "C" __global__ void indexer_score_f32(
		const float * query,
		const float * key,
		const float * weights,
		float * output,
		unsigned int width,
		unsigned int heads,
		unsigned int query_tokens,
		unsigned int key_tokens,
		float scale,
		unsigned int query_start,
		unsigned int count) {
	const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= count) {
		return;
	}
	const unsigned int key_token = index % key_tokens;
	const unsigned int query_token = index / key_tokens;
	if (query_token >= query_tokens) {
		return;
	}
	if (key_token > query_start + query_token) {
		output[index] = -__int_as_float(0x7f800000);
		return;
	}
	float score = 0.0f;
	for (unsigned int head = 0; head < heads; ++head) {
		const size_t query_offset = ((size_t) query_token * heads + head) * width;
		const size_t key_offset = (size_t) key_token * width;
		float dot = 0.0f;
		for (unsigned int channel = 0; channel < width; ++channel) {
			dot += query[query_offset + channel] * key[key_offset + channel];
		}
		if (dot > 0.0f) {
			score += dot * weights[(size_t) query_token * heads + head];
		}
	}
	output[index] = score * scale;
}

extern "C" __global__ void rwkv7_f32(
		const float * receptance,
		const float * decay,
		const float * key,
		const float * value,
		const float * a,
		const float * b_vector,
		const float * input_state,
		float * output,
		unsigned int width,
		unsigned int heads,
		unsigned int tokens,
		unsigned int sequences) {
	const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int count = heads * sequences;
	if (index >= count) {
		return;
	}
	const unsigned int head = index % heads;
	const unsigned int sequence = index / heads;
	const unsigned int attention_elements = width * heads * tokens * sequences;
	const unsigned int state_elements = width * width;
	float * state = output + attention_elements + index * state_elements;
	const float * source_state = input_state + index * state_elements;
	for (unsigned int i = 0; i < state_elements; ++i) {
		state[i] = source_state[i];
	}
	for (unsigned int token = 0; token < tokens; ++token) {
		const unsigned int vector_base =
			((sequence * tokens + token) * heads + head) * width;
		for (unsigned int row = 0; row < width; ++row) {
			float * state_row = state + row * width;
			float state_a = 0.0f;
			for (unsigned int column = 0; column < width; ++column) {
				state_a += a[vector_base + column] * state_row[column];
			}
			float result = 0.0f;
			for (unsigned int column = 0; column < width; ++column) {
				const float next = state_row[column] * decay[vector_base + column] +
					value[vector_base + row] * key[vector_base + column] +
					state_a * b_vector[vector_base + column];
				state_row[column] = next;
				result += next * receptance[vector_base + column];
			}
			output[vector_base + row] = result;
		}
	}
}

extern "C" __global__ void rms_norm_f32(
        const float * input,
        float * output,
        unsigned int width,
        unsigned int rows,
        float epsilon) {
    const unsigned int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    const unsigned int offset = row * width;
    float sum_squares = 0.0f;
    for (unsigned int column = threadIdx.x; column < width; column += blockDim.x) {
        const float value = input[offset + column];
        sum_squares += value * value;
    }
    __shared__ float partial[256];
    sum_squares = block_sum_f32(sum_squares, partial);
    const float inverse = rsqrtf(sum_squares / (float) width + epsilon);
    for (unsigned int column = threadIdx.x; column < width; column += blockDim.x) {
        output[offset + column] = input[offset + column] * inverse;
    }
}

extern "C" __global__ void weighted_rms_norm_f32(
        const float * input,
        const float * weight,
        float * output,
        unsigned int width,
        unsigned int rows,
        float epsilon) {
    const unsigned int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    const unsigned int offset = row * width;
    float sum_squares = 0.0f;
    for (unsigned int column = threadIdx.x; column < width; column += blockDim.x) {
        const float value = input[offset + column];
        sum_squares += value * value;
    }
    __shared__ float partial[256];
    sum_squares = block_sum_f32(sum_squares, partial);
    const float inverse = rsqrtf(sum_squares / (float) width + epsilon);
    for (unsigned int column = threadIdx.x; column < width; column += blockDim.x) {
        output[offset + column] = input[offset + column] * inverse * weight[column];
    }
}

extern "C" __global__ void weighted_rms_norm_add_f32(
        const float * left,
        const float * right,
        const float * weight,
        float * output,
        unsigned int width,
        unsigned int rows,
        float epsilon) {
    const unsigned int row = blockIdx.x;
    if (row >= rows) return;
    const unsigned int offset = row * width;
    float sum_squares = 0.0f;
    for (unsigned int column = threadIdx.x; column < width; column += blockDim.x) {
        const float merged = left[offset + column] + right[offset + column];
        sum_squares += merged * merged;
    }
    __shared__ float partial[256];
    sum_squares = block_sum_f32(sum_squares, partial);
    const float inverse = rsqrtf(sum_squares / (float) width + epsilon);
    for (unsigned int column = threadIdx.x; column < width; column += blockDim.x) {
        const float merged = left[offset + column] + right[offset + column];
        output[offset + column] = merged * inverse * weight[column];
    }
}

extern "C" __global__ void weighted_rms_norm_q8_0_f32(
        const float * input,
        const float * weight,
        float * output,
        unsigned char * quantized,
        unsigned int width,
        unsigned int rows,
        float epsilon) {
    const unsigned int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    const unsigned int offset = row * width;
    float sum_squares = 0.0f;
    for (unsigned int column = threadIdx.x; column < width; column += blockDim.x) {
        const float value = input[offset + column];
        sum_squares += value * value;
    }
    __shared__ float partial[256];
    sum_squares = block_sum_f32(sum_squares, partial);
    const float inverse = rsqrtf(sum_squares / (float) width + epsilon);
    const unsigned int blocks_per_row = width / Q8_0_BLOCK_WIDTH;
    const unsigned int lane = threadIdx.x % CUDA_WARP_WIDTH;
    for (unsigned int column = threadIdx.x; column < width; column += blockDim.x) {
        const float value = input[offset + column] * inverse * weight[column];
        output[offset + column] = value;
        store_q8_input_warp(
            quantized,
            row * blocks_per_row + column / Q8_0_BLOCK_WIDTH,
            value,
            lane);
    }
}

extern "C" __global__ void weighted_rms_norm_add_q8_0_f32(
        const float * left,
        const float * right,
        const float * weight,
        float * output,
        unsigned char * quantized,
        unsigned int width,
        unsigned int rows,
        float epsilon) {
    const unsigned int row = blockIdx.x;
    if (row >= rows) return;
    const unsigned int offset = row * width;
    float sum_squares = 0.0f;
    for (unsigned int column = threadIdx.x; column < width; column += blockDim.x) {
        const float merged = left[offset + column] + right[offset + column];
        sum_squares += merged * merged;
    }
    __shared__ float partial[256];
    sum_squares = block_sum_f32(sum_squares, partial);
    const float inverse = rsqrtf(sum_squares / (float) width + epsilon);
    const unsigned int blocks_per_row = width / Q8_0_BLOCK_WIDTH;
    const unsigned int lane = threadIdx.x % CUDA_WARP_WIDTH;
    for (unsigned int column = threadIdx.x; column < width; column += blockDim.x) {
        const float merged = left[offset + column] + right[offset + column];
        const float normalized = merged * inverse * weight[column];
        output[offset + column] = normalized;
        store_q8_input_warp(
            quantized,
            row * blocks_per_row + column / Q8_0_BLOCK_WIDTH,
            normalized,
            lane);
    }
}

extern "C" __global__ void weighted_rms_gate_f32(
        const float * left,
        const float * right,
        const float * gate,
        const float * weight,
        float * output,
        unsigned int activation,
        unsigned int use_add,
        unsigned int width,
        unsigned int rows,
        float epsilon) {
    const unsigned int row = blockIdx.x;
    if (row >= rows) return;
    const unsigned int offset = row * width;
    float sum_squares = 0.0f;
    for (unsigned int column = threadIdx.x; column < width; column += blockDim.x) {
        const float source = left[offset + column] +
            (use_add != 0 ? right[offset + column] : 0.0f);
        sum_squares += source * source;
    }
    __shared__ float partial[256];
    sum_squares = block_sum_f32(sum_squares, partial);
    const float inverse = rsqrtf(sum_squares / (float) width + epsilon);
    for (unsigned int column = threadIdx.x; column < width; column += blockDim.x) {
        const unsigned int index = offset + column;
        const float source = left[index] + (use_add != 0 ? right[index] : 0.0f);
        output[index] = source * inverse * weight[column] *
            activated_gate_value(gate[index], activation);
    }
}

extern "C" __global__ void weighted_rms_gate_q8_0_f32(
        const float * left,
        const float * right,
        const float * gate,
        const float * weight,
        float * output,
        unsigned char * quantized,
        unsigned int activation,
        unsigned int use_add,
        unsigned int width,
        unsigned int rows,
        float epsilon) {
    const unsigned int row = blockIdx.x;
    if (row >= rows) return;
    const unsigned int offset = row * width;
    float sum_squares = 0.0f;
    for (unsigned int column = threadIdx.x; column < width; column += blockDim.x) {
        const float source = left[offset + column] +
            (use_add != 0 ? right[offset + column] : 0.0f);
        sum_squares += source * source;
    }
    __shared__ float partial[256];
    sum_squares = block_sum_f32(sum_squares, partial);
    const float inverse = rsqrtf(sum_squares / (float) width + epsilon);
    const unsigned int blocks_per_row = width / Q8_0_BLOCK_WIDTH;
    const unsigned int lane = threadIdx.x % CUDA_WARP_WIDTH;
    for (unsigned int column = threadIdx.x; column < width; column += blockDim.x) {
        const unsigned int index = offset + column;
        const float source = left[index] + (use_add != 0 ? right[index] : 0.0f);
        const float value = source * inverse * weight[column] *
            activated_gate_value(gate[index], activation);
        output[index] = value;
        store_q8_input_warp(
            quantized,
            row * blocks_per_row + column / Q8_0_BLOCK_WIDTH,
            value,
            lane);
    }
}

extern "C" __global__ void layer_norm_f32(
        const float * input,
        float * output,
        unsigned int width,
        unsigned int rows,
        float epsilon) {
    const unsigned int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    const unsigned int offset = row * width;
    float sum = 0.0f;
    for (unsigned int column = threadIdx.x; column < width; column += blockDim.x) {
        sum += input[offset + column];
    }
    __shared__ float partial[256];
    sum = block_sum_f32(sum, partial);
    const float mean = sum / (float) width;
    float sum_squares = 0.0f;
    for (unsigned int column = threadIdx.x; column < width; column += blockDim.x) {
        const float centered = input[offset + column] - mean;
        sum_squares += centered * centered;
    }
    sum_squares = block_sum_f32(sum_squares, partial);
    const float inverse = rsqrtf(sum_squares / (float) width + epsilon);
    for (unsigned int column = threadIdx.x; column < width; column += blockDim.x) {
        output[offset + column] = (input[offset + column] - mean) * inverse;
    }
}

// layer_norm_modulate_f32: layer_norm_f32 with the per-channel modulation
// epilogue fused into the write-back pass (identical reduction, identical
// per-element rounding via __f*_rn: the unfused chain materialized the
// normalized value then applied one op per kernel pass).
// adaptive != 0: out = n + n*scale + shift (adaptive shift-scale);
// adaptive == 0: out = n*scale + shift (affine norm).
extern "C" __global__ void layer_norm_modulate_f32(
        const float * input,
        const float * scale_vector,
        const float * shift_vector,
        float * output,
        unsigned int width,
        unsigned int rows,
        float epsilon,
        unsigned int adaptive) {
    const unsigned int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    const unsigned int offset = row * width;
    float sum = 0.0f;
    for (unsigned int column = threadIdx.x; column < width; column += blockDim.x) {
        sum += input[offset + column];
    }
    __shared__ float partial[256];
    sum = block_sum_f32(sum, partial);
    const float mean = sum / (float) width;
    float sum_squares = 0.0f;
    for (unsigned int column = threadIdx.x; column < width; column += blockDim.x) {
        const float centered = input[offset + column] - mean;
        sum_squares += centered * centered;
    }
    sum_squares = block_sum_f32(sum_squares, partial);
    const float inverse = rsqrtf(sum_squares / (float) width + epsilon);
    for (unsigned int column = threadIdx.x; column < width; column += blockDim.x) {
        const float normalized = (input[offset + column] - mean) * inverse;
        const float scaled = __fmul_rn(normalized, scale_vector[column]);
        const float value = adaptive != 0u ? __fadd_rn(normalized, scaled) : scaled;
        output[offset + column] = __fadd_rn(value, shift_vector[column]);
    }
}

// broadcast_gate_add_f32: out = residual + value*gate[channel] — the gated
// residual join (rank-1 gate broadcast over tokens) in one pass, rounding
// exactly like the broadcast_multiply + add pair it replaces.
extern "C" __global__ void broadcast_gate_add_f32(
        const float * value,
        const float * gate,
        const float * residual,
        float * output,
        unsigned int width,
        unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = __fadd_rn(residual[index], __fmul_rn(value[index], gate[index % width]));
    }
}

extern "C" __global__ void softmax_f32(
        const float * input,
        float * output,
        unsigned int width,
        unsigned int rows) {
    const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    const unsigned int offset = row * width;
    float maximum = input[offset];
    for (unsigned int column = 1; column < width; ++column) {
        maximum = fmaxf(maximum, input[offset + column]);
    }
    float sum = 0.0f;
    for (unsigned int column = 0; column < width; ++column) {
        const float value = expf(input[offset + column] - maximum);
        output[offset + column] = value;
        sum += value;
    }
    for (unsigned int column = 0; column < width; ++column) {
        output[offset + column] /= sum;
    }
}

// causal_softmax_f32: row-wise softmax over the causal prefix of a square
// [rows, rows] score matrix. Row i (a query) softmaxes over columns 0..i (keys)
// and zeros columns i+1..rows-1. One thread per row. Feeds the device attention
// backward so the softmax p is computed in-place from resident scores instead of
// uploaded from a host [heads, seq, seq] allocation.
extern "C" __global__ void causal_softmax_f32(
        const float * scores,
        float * probabilities,
        unsigned int rows) {
    const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    const unsigned int offset = row * rows;
    const unsigned int keys = row + 1;
    float maximum = scores[offset];
    for (unsigned int column = 1; column < keys; ++column) {
        maximum = fmaxf(maximum, scores[offset + column]);
    }
    float sum = 0.0f;
    for (unsigned int column = 0; column < keys; ++column) {
        const float value = expf(scores[offset + column] - maximum);
        probabilities[offset + column] = value;
        sum += value;
    }
    for (unsigned int column = 0; column < keys; ++column) {
        probabilities[offset + column] /= sum;
    }
    for (unsigned int column = keys; column < rows; ++column) {
        probabilities[offset + column] = 0.0f;
    }
}

extern "C" __global__ void mul_mat_f32(
        const float * left,
        const float * right,
        float * output,
        unsigned int inner,
        unsigned int left_rows,
        unsigned int right_rows) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int count = left_rows * right_rows;
    if (index >= count) {
        return;
    }
    const unsigned int left_row = index % left_rows;
    const unsigned int right_row = index / left_rows;
    float sum = 0.0f;
    for (unsigned int column = 0; column < inner; ++column) {
        sum += left[left_row * inner + column] * right[right_row * inner + column];
    }
    output[index] = sum;
}

extern "C" __global__ void get_rows_f32(
        const float * table,
        const unsigned int * rows,
        float * output,
        unsigned int width,
        unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        const unsigned int column = index % width;
        const unsigned int output_row = index / width;
        output[index] = table[rows[output_row] * width + column];
    }
}

__device__ static float rope_yarn_corr_dim(
		unsigned int rotary_dimensions,
		unsigned int original_context,
		float rotations,
		float frequency_base) {
	return (float) rotary_dimensions * logf(
		(float) original_context / (rotations * 2.0f * 3.14159265358979323846f)
	) / (2.0f * logf(frequency_base));
}

__device__ static void rope_yarn_angles(
		float theta_extrapolated,
		float frequency_scale,
		unsigned int pair,
		unsigned int rotary_dimensions,
		unsigned int original_context,
		float frequency_base,
		float ext_factor,
		float attention_factor,
		float beta_fast,
		float beta_slow,
		float * cosine,
		float * sine) {
	float theta = frequency_scale * theta_extrapolated;
	float magnitude = attention_factor;
	if (ext_factor != 0.0f && original_context != 0) {
		float low = floorf(rope_yarn_corr_dim(
			rotary_dimensions, original_context, beta_fast, frequency_base));
		float high = ceilf(rope_yarn_corr_dim(
			rotary_dimensions, original_context, beta_slow, frequency_base));
		low = fmaxf(0.0f, fminf((float) rotary_dimensions - 1.0f, low));
		high = fmaxf(0.0f, fminf((float) rotary_dimensions - 1.0f, high));
		const float y = ((float) pair - low) / fmaxf(0.001f, high - low);
		const float ramp = 1.0f - fminf(1.0f, fmaxf(0.0f, y));
		const float mix = ramp * ext_factor;
		theta = theta * (1.0f - mix) + theta_extrapolated * mix;
		magnitude *= 1.0f + 0.1f * logf(1.0f / frequency_scale);
	}
	sincosf(theta, sine, cosine);
	*cosine *= magnitude;
	*sine *= magnitude;
}

extern "C" __global__ void rope_neox_f32(
        const float * input,
        const unsigned int * positions,
        const float * frequency_factors,
        float * output,
        unsigned int width,
        unsigned int heads,
        unsigned int tokens,
        unsigned int rotary_dimensions,
        float frequency_base,
        float frequency_scale,
		unsigned int original_context,
		float ext_factor,
		float attention_factor,
		float beta_fast,
		float beta_slow,
        unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const unsigned int column = index % width;
    if (column >= rotary_dimensions) {
        output[index] = input[index];
        return;
    }
    const unsigned int row = index / width;
    const unsigned int token = (row / heads) % tokens;
    const unsigned int half = rotary_dimensions / 2;
    const unsigned int pair = column % half;
    const unsigned int pair_offset = row * width + pair;
	const float theta_extrapolated =
		(float) positions[token] *
        powf(frequency_base, -2.0f * (float) pair / (float) rotary_dimensions) /
        (frequency_factors == nullptr ? 1.0f : frequency_factors[pair]);
    float sine;
    float cosine;
	rope_yarn_angles(theta_extrapolated, frequency_scale, pair, rotary_dimensions,
		original_context, frequency_base, ext_factor, attention_factor,
		beta_fast, beta_slow, &cosine, &sine);
    const float x0 = input[pair_offset];
    const float x1 = input[pair_offset + half];
    output[index] = column < half
        ? x0 * cosine - x1 * sine
        : x0 * sine + x1 * cosine;
}

extern "C" __global__ void rope_normal_f32(
        const float * input,
        const unsigned int * positions,
        const float * frequency_factors,
        float * output,
        unsigned int width,
        unsigned int heads,
        unsigned int tokens,
        unsigned int rotary_dimensions,
        float frequency_base,
        float frequency_scale,
		unsigned int original_context,
		float ext_factor,
		float attention_factor,
		float beta_fast,
		float beta_slow,
        unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const unsigned int column = index % width;
    if (column >= rotary_dimensions) {
        output[index] = input[index];
        return;
    }
    const unsigned int row = index / width;
    const unsigned int token = (row / heads) % tokens;
    const unsigned int pair = column / 2;
    const unsigned int pair_offset = row * width + pair * 2;
	const float theta_extrapolated =
		(float) positions[token] *
        powf(frequency_base, -2.0f * (float) pair / (float) rotary_dimensions) /
        (frequency_factors == nullptr ? 1.0f : frequency_factors[pair]);
    float sine;
    float cosine;
	rope_yarn_angles(theta_extrapolated, frequency_scale, pair, rotary_dimensions,
		original_context, frequency_base, ext_factor, attention_factor,
		beta_fast, beta_slow, &cosine, &sine);
    const float x0 = input[pair_offset];
    const float x1 = input[pair_offset + 1];
    output[index] = (column & 1) == 0
        ? x0 * cosine - x1 * sine
        : x0 * sine + x1 * cosine;
}

// rope_normal rotated DIRECTLY into its cache-append slot: the standalone
// append copy launch does not exist. Offset is device-resident (token units of
// inner elements), keeping decode launches byte-identical across steps.
extern "C" __global__ void rope_append_normal_f32(
        const float * input,
        const unsigned int * positions,
        const float * frequency_factors,
        float * output,
        const unsigned int * offset,
        unsigned int inner,
        unsigned int width,
        unsigned int heads,
        unsigned int tokens,
        unsigned int rotary_dimensions,
        float frequency_base,
        float frequency_scale,
		unsigned int original_context,
		float ext_factor,
		float attention_factor,
		float beta_fast,
		float beta_slow,
        unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    float * destination = output + (size_t) offset[0] * inner + index;
    const unsigned int column = index % width;
    if (column >= rotary_dimensions) {
        *destination = input[index];
        return;
    }
    const unsigned int row = index / width;
    const unsigned int token = (row / heads) % tokens;
    const unsigned int pair = column / 2;
    const unsigned int pair_offset = row * width + pair * 2;
	const float theta_extrapolated =
		(float) positions[token] *
        powf(frequency_base, -2.0f * (float) pair / (float) rotary_dimensions) /
        (frequency_factors == nullptr ? 1.0f : frequency_factors[pair]);
    float sine;
    float cosine;
	rope_yarn_angles(theta_extrapolated, frequency_scale, pair, rotary_dimensions,
		original_context, frequency_base, ext_factor, attention_factor,
		beta_fast, beta_slow, &cosine, &sine);
    const float x0 = input[pair_offset];
    const float x1 = input[pair_offset + 1];
    *destination = (column & 1) == 0
        ? x0 * cosine - x1 * sine
        : x0 * sine + x1 * cosine;
}

// rope_neox rotated DIRECTLY into its cache-append slot; mirrors
// rope_append_normal_f32 with half-split pair indexing.
extern "C" __global__ void rope_append_neox_f32(
        const float * input,
        const unsigned int * positions,
        const float * frequency_factors,
        float * output,
        const unsigned int * offset,
        unsigned int inner,
        unsigned int width,
        unsigned int heads,
        unsigned int tokens,
        unsigned int rotary_dimensions,
        float frequency_base,
        float frequency_scale,
		unsigned int original_context,
		float ext_factor,
		float attention_factor,
		float beta_fast,
		float beta_slow,
        unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    float * destination = output + (size_t) offset[0] * inner + index;
    const unsigned int column = index % width;
    if (column >= rotary_dimensions) {
        *destination = input[index];
        return;
    }
    const unsigned int row = index / width;
    const unsigned int token = (row / heads) % tokens;
    const unsigned int half = rotary_dimensions / 2;
    const unsigned int pair = column % half;
    const unsigned int pair_offset = row * width + pair;
	const float theta_extrapolated =
		(float) positions[token] *
        powf(frequency_base, -2.0f * (float) pair / (float) rotary_dimensions) /
        (frequency_factors == nullptr ? 1.0f : frequency_factors[pair]);
    float sine;
    float cosine;
	rope_yarn_angles(theta_extrapolated, frequency_scale, pair, rotary_dimensions,
		original_context, frequency_base, ext_factor, attention_factor,
		beta_fast, beta_slow, &cosine, &sine);
    const float x0 = input[pair_offset];
    const float x1 = input[pair_offset + half];
    *destination = column < half
        ? x0 * cosine - x1 * sine
        : x0 * sine + x1 * cosine;
}

extern "C" __global__ void rope_multi_f32(
        const float * input,
        const unsigned int * positions,
        float * output,
        unsigned int width,
        unsigned int heads,
        unsigned int tokens,
        unsigned int rotary_dimensions,
        float frequency_base,
        float frequency_scale,
        unsigned int section_0,
        unsigned int section_1,
        unsigned int section_2,
        unsigned int section_3,
        unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const unsigned int column = index % width;
    if (column >= rotary_dimensions) {
        output[index] = input[index];
        return;
    }
    const unsigned int row = index / width;
    const unsigned int token = (row / heads) % tokens;
    const unsigned int pair = column / 2;
    const unsigned int section_pairs =
        section_0 + section_1 + section_2 + section_3;
    const unsigned int sector = pair % section_pairs;
    unsigned int axis = 0;
    if (sector >= section_0 + section_1 + section_2) {
        axis = 3;
    } else if (sector >= section_0 + section_1) {
        axis = 2;
    } else if (sector >= section_0) {
        axis = 1;
    }
    const unsigned int pair_offset = row * width + pair * 2;
    const float theta =
        (float) positions[axis * tokens + token] * frequency_scale *
        powf(frequency_base, -2.0f * (float) pair / (float) rotary_dimensions);
    float sine;
    float cosine;
    sincosf(theta, &sine, &cosine);
    const float x0 = input[pair_offset];
    const float x1 = input[pair_offset + 1];
    output[index] = (column & 1) == 0
        ? x0 * cosine - x1 * sine
        : x0 * sine + x1 * cosine;
}

__device__ float moe_expert_value(
        const void * weights,
        size_t index,
        unsigned int storage);

extern "C" __global__ void lora_merge_f32(
		const float * base,
		const float * a,
		const float * b,
		float * output,
		unsigned int inner,
		unsigned int rows,
		unsigned int rank,
		unsigned int groups,
		float scale,
		unsigned int count) {
	const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= count) return;
	const unsigned int column = index % inner;
	const unsigned int row = (index / inner) % rows;
	const unsigned int group = index / (inner * rows);
	float delta = 0.0f;
	for (unsigned int component = 0; component < rank; ++component) {
		const size_t a_index = ((size_t) group * rank + component) * inner + column;
		const size_t b_index = ((size_t) group * rows + row) * rank + component;
		delta += a[a_index] * b[b_index];
	}
	output[index] = base[index] + scale * delta;
}

extern "C" __global__ void moe_f32(
        const float * input,
		const float * router_input,
        const float * router,
		const void * gate,
		const void * up,
		const void * down,
		const float * selection_bias,
		const float * expert_scale,
		const float * router_bias,
		const float * gate_bias,
		const float * up_bias,
		const float * down_bias,
		const float * selected_experts,
        float * output,
        unsigned int hidden,
		unsigned int router_hidden,
        unsigned int tokens,
        unsigned int experts,
        unsigned int top_k,
        unsigned int intermediate,
        unsigned int normalize_top_k,
		unsigned int routing,
        float routed_scale,
		unsigned int expert_storage,
		unsigned int gated,
		unsigned int fused_gate_up,
		unsigned int activation,
		unsigned int expert_index_divisor,
		float swiglu_clamp,
        unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const unsigned int output_channel = index % hidden;
    const unsigned int token = index / hidden;
    const float * x = input + (size_t) token * hidden;
	const float * router_x = router_input + (size_t) token * router_hidden;

    float maximum = -3.402823466e+38F;
    for (unsigned int expert = 0; expert < experts; ++expert) {
		const float * weight = router + (size_t) expert * router_hidden;
        float logit = 0.0f;
		for (unsigned int channel = 0; channel < router_hidden; ++channel) {
            logit += router_x[channel] * weight[channel];
        }
		if (router_bias) logit += router_bias[expert];
        maximum = fmaxf(maximum, logit);
    }
	float denominator = 0.0f;
	if (routing == 1) for (unsigned int expert = 0; expert < experts; ++expert) {
		const float * weight = router + (size_t) expert * router_hidden;
        float logit = 0.0f;
		for (unsigned int channel = 0; channel < router_hidden; ++channel) {
            logit += router_x[channel] * weight[channel];
        }
		if (router_bias) logit += router_bias[expert];
        denominator += expf(logit - maximum);
    }

    unsigned int selected[16];
    float route_weights[16];
    float selected_sum = 0.0f;
    for (unsigned int slot = 0; slot < top_k; ++slot) {
		if (selected_experts) {
			const unsigned int expert = (unsigned int) selected_experts[(size_t) token * top_k + slot];
			const float * weight = router + (size_t) expert * router_hidden;
			float logit = 0.0f;
			for (unsigned int channel = 0; channel < router_hidden; ++channel) {
				logit += router_x[channel] * weight[channel];
			}
			if (router_bias) logit += router_bias[expert];
			const float probability = routing == 4
				? sqrtf(fmaxf(logit, 0.0f) + log1pf(expf(-fabsf(logit))))
				: routing == 2 ? 1.0f / (1.0f + expf(-logit))
				: routing == 3 ? logit : expf(logit - maximum) / denominator;
			selected[slot] = expert;
			route_weights[slot] = probability;
			selected_sum += probability;
			continue;
		}
        int best = -1;
		float best_score = -3.402823466e+38F;
		float best_probability = 0.0f;
        for (unsigned int expert = 0; expert < experts; ++expert) {
            bool used = false;
            for (unsigned int prior = 0; prior < slot; ++prior) {
                used = used || selected[prior] == expert;
            }
            if (used) continue;
			const float * weight = router + (size_t) expert * router_hidden;
            float logit = 0.0f;
			for (unsigned int channel = 0; channel < router_hidden; ++channel) {
                logit += router_x[channel] * weight[channel];
            }
			if (router_bias) logit += router_bias[expert];
			const float probability = routing == 4
				? sqrtf(fmaxf(logit, 0.0f) + log1pf(expf(-fabsf(logit))))
				: routing == 2 ? 1.0f / (1.0f + expf(-logit))
				: routing == 3 ? logit : expf(logit - maximum) / denominator;
			const float score = probability + (selection_bias ? selection_bias[expert] : 0.0f);
			if (best < 0 || score > best_score) {
                best = (int) expert;
				best_score = score;
				best_probability = probability;
            }
        }
        selected[slot] = (unsigned int) best;
		route_weights[slot] = best_probability;
        selected_sum += route_weights[slot];
    }
	if (routing == 3) {
		float selected_maximum = -3.402823466e+38F;
		for (unsigned int slot = 0; slot < top_k; ++slot) selected_maximum = fmaxf(selected_maximum, route_weights[slot]);
		selected_sum = 0.0f;
		for (unsigned int slot = 0; slot < top_k; ++slot) {
			route_weights[slot] = expf(route_weights[slot] - selected_maximum);
			selected_sum += route_weights[slot];
		}
		for (unsigned int slot = 0; slot < top_k; ++slot) route_weights[slot] /= selected_sum;
	}

    float result = 0.0f;
	for (unsigned int slot = 0; slot < top_k; ++slot) {
		const unsigned int routed_expert = selected[slot];
		const unsigned int expert = routed_expert / expert_index_divisor;
		float route = route_weights[slot] * routed_scale;
		if (normalize_top_k) route /= fmaxf(selected_sum, 6.103515625e-5f);
		float expert_output = 0.0f;
		for (unsigned int inner = 0; inner < intermediate; ++inner) {
			const size_t gate_offset = ((size_t) expert * (fused_gate_up ? 2 : 1) * intermediate + inner) * hidden;
			const size_t up_offset = gate_offset + (fused_gate_up ? (size_t) intermediate * hidden : 0);
			float gate_dot = 0.0f;
			float up_dot = 0.0f;
			for (unsigned int channel = 0; channel < hidden; ++channel) {
				if (gated) gate_dot += x[channel] * moe_expert_value(
					gate, gate_offset + channel, expert_storage);
				up_dot += x[channel] * moe_expert_value(
					up, up_offset + channel, expert_storage);
            }
			if (gate_bias) {
				gate_dot += gate_bias[(size_t) expert * intermediate + inner];
				up_dot += up_bias[(size_t) expert * intermediate + inner];
			}
			float activated;
			if (activation == 2) {
				activated = gated ? fmaxf(gate_dot, 0.0f) * up_dot : fmaxf(up_dot, 0.0f);
			} else if (activation == 3) {
				const float value = gated ? gate_dot : up_dot;
				float gelu;
				if (value <= -10.0f) {
					gelu = 0.0f;
				} else if (value >= 10.0f) {
					gelu = value;
				} else {
					const float rounded = __half2float(__float2half_rn(value));
					const float inner = 0.7978845608028654f * rounded *
							(1.0f + 0.044715f * rounded * rounded);
					gelu = __half2float(__float2half_rn(
							0.5f * rounded * (1.0f + tanhf(inner))));
				}
				activated = gated ? gelu * up_dot : gelu;
			} else if (activation == 4) {
				const float gate_value = fminf(gate_dot, 7.0f);
				const float up_value = fminf(7.0f, fmaxf(-7.0f, up_dot));
				activated = gate_value / (1.0f + expf(-1.702f * gate_value)) * (up_value + 1.0f);
			} else if (activation == 5) {
				activated = fmaxf(up_dot, 0.0f);
				activated *= activated;
			} else {
				if (gated) {
					if (swiglu_clamp > 0.0f) {
						up_dot = fminf(swiglu_clamp, fmaxf(-swiglu_clamp, up_dot));
						if (routing == 4) gate_dot = fminf(swiglu_clamp, gate_dot);
					}
					float gate_activation = gate_dot / (1.0f + expf(-gate_dot));
					if (swiglu_clamp > 0.0f && routing != 4) gate_activation = fminf(swiglu_clamp, gate_activation);
					activated = gate_activation * up_dot;
				} else {
					activated = up_dot / (1.0f + expf(-up_dot));
				}
			}
            const size_t down_offset = ((size_t) expert * hidden + output_channel) * intermediate + inner;
			expert_output += activated * moe_expert_value(
				down, down_offset, expert_storage);
        }
		if (down_bias) expert_output += down_bias[(size_t) expert * hidden + output_channel];
		if (expert_scale) expert_output *= expert_scale[expert];
        result += route * expert_output;
    }
    output[index] = result;
}

extern "C" __global__ void moe_grouped_f32(
        const float * input,
		const float * router_input,
        const float * router,
		const void * gate,
		const void * up,
		const void * down,
		const float * selection_bias,
		const float * expert_scale,
		const float * router_bias,
		const float * gate_bias,
		const float * up_bias,
		const float * down_bias,
		const float * selected_experts,
        float * output,
        unsigned int hidden,
		unsigned int router_hidden,
        unsigned int tokens,
        unsigned int experts,
        unsigned int top_k,
        unsigned int intermediate,
        unsigned int normalize_top_k,
		unsigned int routing,
        float routed_scale,
		unsigned int expert_storage,
		unsigned int gated,
		unsigned int fused_gate_up,
		unsigned int activation,
		unsigned int expert_index_divisor,
		float swiglu_clamp) {
    const unsigned int token = blockIdx.x;
    if (token >= tokens) return;
    const float * x = input + (size_t) token * hidden;
    const float * router_x = router_input + (size_t) token * router_hidden;
    extern __shared__ unsigned char shared_storage[];
    unsigned int * selected = reinterpret_cast<unsigned int *>(shared_storage);
    float * routes = reinterpret_cast<float *>(selected + top_k);
    float * activations = routes + top_k;
    float * selected_sum = activations + blockDim.x;

    if (threadIdx.x == 0) {
        float maximum = -3.402823466e+38F;
        for (unsigned int expert = 0; expert < experts; ++expert) {
            const float * weight = router + (size_t) expert * router_hidden;
            float logit = 0.0f;
            for (unsigned int channel = 0; channel < router_hidden; ++channel) {
                logit += router_x[channel] * weight[channel];
            }
            if (router_bias) logit += router_bias[expert];
            maximum = fmaxf(maximum, logit);
        }
        float denominator = 0.0f;
        if (routing == 1) {
            for (unsigned int expert = 0; expert < experts; ++expert) {
                const float * weight = router + (size_t) expert * router_hidden;
                float logit = 0.0f;
                for (unsigned int channel = 0; channel < router_hidden; ++channel) {
                    logit += router_x[channel] * weight[channel];
                }
                if (router_bias) logit += router_bias[expert];
                denominator += expf(logit - maximum);
            }
        }
        *selected_sum = 0.0f;
        for (unsigned int slot = 0; slot < top_k; ++slot) {
            if (selected_experts) {
                const unsigned int expert = (unsigned int) selected_experts[(size_t) token * top_k + slot];
                const float * weight = router + (size_t) expert * router_hidden;
                float logit = 0.0f;
                for (unsigned int channel = 0; channel < router_hidden; ++channel) {
                    logit += router_x[channel] * weight[channel];
                }
                if (router_bias) logit += router_bias[expert];
                const float probability = routing == 4
                    ? sqrtf(fmaxf(logit, 0.0f) + log1pf(expf(-fabsf(logit))))
                    : routing == 2 ? 1.0f / (1.0f + expf(-logit))
                    : routing == 3 ? logit : expf(logit - maximum) / denominator;
                selected[slot] = expert;
                routes[slot] = probability;
                *selected_sum += probability;
                continue;
            }
            int best = -1;
            float best_score = -3.402823466e+38F;
            float best_probability = 0.0f;
            for (unsigned int expert = 0; expert < experts; ++expert) {
                bool used = false;
                for (unsigned int prior = 0; prior < slot; ++prior) used = used || selected[prior] == expert;
                if (used) continue;
                const float * weight = router + (size_t) expert * router_hidden;
                float logit = 0.0f;
                for (unsigned int channel = 0; channel < router_hidden; ++channel) {
                    logit += router_x[channel] * weight[channel];
                }
                if (router_bias) logit += router_bias[expert];
                const float probability = routing == 4
                    ? sqrtf(fmaxf(logit, 0.0f) + log1pf(expf(-fabsf(logit))))
                    : routing == 2 ? 1.0f / (1.0f + expf(-logit))
                    : routing == 3 ? logit : expf(logit - maximum) / denominator;
                const float score = probability + (selection_bias ? selection_bias[expert] : 0.0f);
                if (best < 0 || score > best_score) {
                    best = (int) expert;
                    best_score = score;
                    best_probability = probability;
                }
            }
            selected[slot] = (unsigned int) best;
            routes[slot] = best_probability;
            *selected_sum += best_probability;
        }
        if (routing == 3) {
            float selected_maximum = -3.402823466e+38F;
            for (unsigned int slot = 0; slot < top_k; ++slot) selected_maximum = fmaxf(selected_maximum, routes[slot]);
            *selected_sum = 0.0f;
            for (unsigned int slot = 0; slot < top_k; ++slot) {
                routes[slot] = expf(routes[slot] - selected_maximum);
                *selected_sum += routes[slot];
            }
            for (unsigned int slot = 0; slot < top_k; ++slot) routes[slot] /= *selected_sum;
        }
    }
    for (unsigned int output_channel = threadIdx.x; output_channel < hidden; output_channel += blockDim.x) {
        output[(size_t) token * hidden + output_channel] = 0.0f;
    }
    __syncthreads();

    for (unsigned int slot = 0; slot < top_k; ++slot) {
        const unsigned int expert = selected[slot] / expert_index_divisor;
        float route = routes[slot] * routed_scale;
        if (normalize_top_k) route /= fmaxf(*selected_sum, 6.103515625e-5f);
        const float expert_multiplier = expert_scale ? expert_scale[expert] : 1.0f;
        for (unsigned int tile = 0; tile < intermediate; tile += blockDim.x) {
            const unsigned int inner_index = tile + threadIdx.x;
            if (inner_index < intermediate) {
                const size_t gate_offset =
                    ((size_t) expert * (fused_gate_up ? 2 : 1) * intermediate + inner_index) * hidden;
                const size_t up_offset = gate_offset + (fused_gate_up ? (size_t) intermediate * hidden : 0);
                float gate_dot = 0.0f;
                float up_dot = 0.0f;
                for (unsigned int channel = 0; channel < hidden; ++channel) {
                    if (gated) gate_dot += x[channel] * moe_expert_value(gate, gate_offset + channel, expert_storage);
                    up_dot += x[channel] * moe_expert_value(up, up_offset + channel, expert_storage);
                }
                if (gate_bias) {
                    gate_dot += gate_bias[(size_t) expert * intermediate + inner_index];
                    up_dot += up_bias[(size_t) expert * intermediate + inner_index];
                }
                float activated;
                if (activation == 2) {
                    activated = gated ? fmaxf(gate_dot, 0.0f) * up_dot : fmaxf(up_dot, 0.0f);
                } else if (activation == 3) {
                    const float source = gated ? gate_dot : up_dot;
                    float gelu;
                    if (source <= -10.0f) {
                        gelu = 0.0f;
                    } else if (source >= 10.0f) {
                        gelu = source;
                    } else {
                        const float rounded = __half2float(__float2half_rn(source));
                        const float argument = 0.7978845608028654f * rounded *
                            (1.0f + 0.044715f * rounded * rounded);
                        gelu = __half2float(__float2half_rn(
                            0.5f * rounded * (1.0f + tanhf(argument))));
                    }
                    activated = gated ? gelu * up_dot : gelu;
                } else if (activation == 4) {
                    const float gate_value = fminf(gate_dot, 7.0f);
                    const float up_value = fminf(7.0f, fmaxf(-7.0f, up_dot));
                    activated = gate_value / (1.0f + expf(-1.702f * gate_value)) * (up_value + 1.0f);
                } else if (activation == 5) {
                    activated = fmaxf(up_dot, 0.0f);
                    activated *= activated;
                } else if (gated) {
                    if (swiglu_clamp > 0.0f) {
                        up_dot = fminf(swiglu_clamp, fmaxf(-swiglu_clamp, up_dot));
                        if (routing == 4) gate_dot = fminf(swiglu_clamp, gate_dot);
                    }
                    float gate_activation = gate_dot / (1.0f + expf(-gate_dot));
                    if (swiglu_clamp > 0.0f && routing != 4) gate_activation = fminf(swiglu_clamp, gate_activation);
                    activated = gate_activation * up_dot;
                } else {
                    activated = up_dot / (1.0f + expf(-up_dot));
                }
                activations[threadIdx.x] = activated;
            }
            __syncthreads();
            const unsigned int tile_count = intermediate - tile < blockDim.x ? intermediate - tile : blockDim.x;
            for (unsigned int output_channel = threadIdx.x; output_channel < hidden; output_channel += blockDim.x) {
                float partial = 0.0f;
                const size_t down_offset = ((size_t) expert * hidden + output_channel) * intermediate + tile;
                for (unsigned int inner = 0; inner < tile_count; ++inner) {
                    partial += activations[inner] * moe_expert_value(down, down_offset + inner, expert_storage);
                }
                if (tile + tile_count == intermediate && down_bias) {
                    partial += down_bias[(size_t) expert * hidden + output_channel];
                }
                output[(size_t) token * hidden + output_channel] += route * expert_multiplier * partial;
            }
            __syncthreads();
        }
    }
}

extern "C" __global__ void repeat_heads_f32(
		const float * input,
		float * output,
		unsigned int width,
		unsigned int heads,
		unsigned int tokens,
		unsigned int count) {
	const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= count) return;
	const unsigned int token = index / (width * heads);
	const unsigned int column = index % width;
	if (token < tokens) output[index] = input[(size_t) token * width + column];
}

extern "C" __global__ void attention_f32(
        const float * query,
        const float * key,
        const float * value,
        const float * relative_bias,
		const float * sinks,
		const float * block_ids,
		const float * key_bias,
        float * output,
        unsigned int key_width,
        unsigned int value_width,
        unsigned int query_heads,
        unsigned int key_value_heads,
        unsigned int query_tokens,
        unsigned int key_value_tokens,
        unsigned int sequences,
        float scale,
        float softcap,
        float max_alibi_bias,
        unsigned int causal,
        unsigned int query_start,
        unsigned int window,
		unsigned int symmetric_window,
		unsigned int relative_buckets,
		unsigned int relative_bidirectional,
		unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const unsigned int value_channel = index % value_width;
    unsigned int row = index / value_width;
    const unsigned int query_head = row % query_heads;
    row /= query_heads;
    const unsigned int query_token = row % query_tokens;
    const unsigned int sequence = row / query_tokens;
    const unsigned int group_size = query_heads / key_value_heads;
    const unsigned int key_value_head = query_head / group_size;
    const unsigned int query_position = query_start + query_token;
    const unsigned int causal_limit = query_position + 1;
    unsigned int key_limit = causal
        ? causal_limit
        : key_value_tokens;
    if (causal && block_ids != nullptr && block_ids[query_position] >= 0.0f) {
        key_limit = key_value_tokens;
    }
    unsigned int key_first =
        window > 0 && causal_limit > window ? causal_limit - window : 0;
    if (symmetric_window == 2) {
        key_first = ((query_start + query_token) / window) * window;
    } else if (symmetric_window) {
        const unsigned int half_window = window / 2;
        const unsigned int query_position = query_start + query_token;
        key_first = query_position > half_window ? query_position - half_window : 0;
        const unsigned int symmetric_limit = query_position + half_window + 1;
        key_limit = symmetric_limit < key_value_tokens ? symmetric_limit : key_value_tokens;
    }
    const unsigned int query_offset =
        ((sequence * query_tokens + query_token) * query_heads + query_head) * key_width;

	unsigned int n_head_log2 = 1;
	while (n_head_log2 * 2 <= query_heads) {
		n_head_log2 *= 2;
	}
	float alibi_slope = 0.0f;
	if (max_alibi_bias > 0.0f) {
		const float m0 = powf(2.0f, -max_alibi_bias / (float) n_head_log2);
		const float m1 = powf(2.0f, -(max_alibi_bias / 2.0f) / (float) n_head_log2);
		alibi_slope = query_head < n_head_log2
			? powf(m0, (float) (query_head + 1))
			: powf(m1, (float) (2 * (query_head - n_head_log2) + 1));
	}

    float maximum = -3.402823466e+38F;
	if (sinks != nullptr) {
		maximum = sinks[query_head];
	}
    for (unsigned int key_token = key_first; key_token < key_limit; ++key_token) {
		if (causal && key_token >= causal_limit && (block_ids == nullptr ||
			block_ids[query_position] < 0.0f ||
			block_ids[key_token] != block_ids[query_position])) continue;
        const unsigned int key_offset =
            ((sequence * key_value_tokens + key_token) * key_value_heads + key_value_head) * key_width;
        float dot = 0.0f;
        for (unsigned int channel = 0; channel < key_width; ++channel) {
            dot += query[query_offset + channel] * key[key_offset + channel];
        }
        float score = dot * scale;
		if (alibi_slope > 0.0f) {
			const int query_position = (int) query_start + (int) query_token;
			const int distance = query_position - (int) key_token;
			score -= (float) (distance < 0 ? -distance : distance) * alibi_slope;
		}
        if (relative_bias != nullptr) {
			unsigned int buckets = relative_buckets;
			int distance = (int) key_token - ((int) query_start + (int) query_token);
			unsigned int bucket = 0;
			if (relative_bidirectional) {
				buckets /= 2;
				bucket = distance > 0 ? buckets : 0;
			} else if (distance > 0) {
				distance = 0;
			}
			const unsigned int max_exact = buckets / 2;
			distance = distance < 0 ? -distance : distance;
            if ((unsigned int) distance < max_exact) {
                bucket += (unsigned int) distance;
            } else {
                unsigned int large = max_exact + (unsigned int) floorf(
                    logf((float) distance / (float) max_exact) *
					(float) (buckets - max_exact) /
					logf(128.0f / (float) max_exact));
				bucket += large < buckets ? large : buckets - 1;
            }
            score += relative_bias[bucket * query_heads + query_head];
        }
        if (softcap > 0.0f) {
            score = softcap * tanhf(score / softcap);
        }
        if (key_bias != nullptr) score += key_bias[key_token];
        maximum = fmaxf(maximum, score);
    }

	float sum = sinks != nullptr ? expf(sinks[query_head] - maximum) : 0.0f;
    float weighted = 0.0f;
    for (unsigned int key_token = key_first; key_token < key_limit; ++key_token) {
		if (causal && key_token >= causal_limit && (block_ids == nullptr ||
			block_ids[query_position] < 0.0f ||
			block_ids[key_token] != block_ids[query_position])) continue;
        const unsigned int key_offset =
            ((sequence * key_value_tokens + key_token) * key_value_heads + key_value_head) * key_width;
        float dot = 0.0f;
        for (unsigned int channel = 0; channel < key_width; ++channel) {
            dot += query[query_offset + channel] * key[key_offset + channel];
        }
        float score = dot * scale;
		if (alibi_slope > 0.0f) {
			const int query_position = (int) query_start + (int) query_token;
			const int distance = query_position - (int) key_token;
			score -= (float) (distance < 0 ? -distance : distance) * alibi_slope;
		}
        if (relative_bias != nullptr) {
			unsigned int buckets = relative_buckets;
			int distance = (int) key_token - ((int) query_start + (int) query_token);
			unsigned int bucket = 0;
			if (relative_bidirectional) {
				buckets /= 2;
				bucket = distance > 0 ? buckets : 0;
			} else if (distance > 0) {
				distance = 0;
			}
			const unsigned int max_exact = buckets / 2;
			distance = distance < 0 ? -distance : distance;
            if ((unsigned int) distance < max_exact) {
                bucket += (unsigned int) distance;
            } else {
                unsigned int large = max_exact + (unsigned int) floorf(
                    logf((float) distance / (float) max_exact) *
					(float) (buckets - max_exact) /
					logf(128.0f / (float) max_exact));
				bucket += large < buckets ? large : buckets - 1;
            }
            score += relative_bias[bucket * query_heads + query_head];
        }
        if (softcap > 0.0f) {
            score = softcap * tanhf(score / softcap);
        }
        if (key_bias != nullptr) score += key_bias[key_token];
        const float probability = expf(score - maximum);
        const unsigned int value_offset =
            ((sequence * key_value_tokens + key_token) * key_value_heads + key_value_head) * value_width;
        sum += probability;
        weighted += probability * value[value_offset + value_channel];
    }
    output[index] = weighted / sum;
}

// key_value_token_count: device scalar so decode launches stay byte-identical
// across steps; key_value_stride is the KV capacity used for addressing and
// shared-memory partitioning.
extern "C" __global__ void attention_decode_f32(
        const float * query,
        const float * key,
        const float * value,
        float * output,
        unsigned int key_width,
        unsigned int value_width,
        unsigned int query_heads,
        unsigned int key_value_heads,
        const unsigned int * key_value_token_count,
        unsigned int key_value_stride,
        unsigned int sequences,
        float scale) {
    const unsigned int row = blockIdx.x;
    const unsigned int query_head = row % query_heads;
    const unsigned int sequence = row / query_heads;
    if (sequence >= sequences) {
        return;
    }
    const unsigned int key_value_tokens = key_value_token_count[0];
    const unsigned int group_size = query_heads / key_value_heads;
    const unsigned int key_value_head = query_head / group_size;
    const unsigned int query_offset =
        (sequence * query_heads + query_head) * key_width;
    extern __shared__ float shared[];
    float * scores = shared;
    float * partial = shared + key_value_stride;
    float local_maximum = -3.402823466e+38F;
    for (unsigned int token = threadIdx.x; token < key_value_tokens; token += blockDim.x) {
        const unsigned int key_offset =
            ((sequence * key_value_stride + token) * key_value_heads + key_value_head) * key_width;
        float dot = 0.0f;
        for (unsigned int channel = 0; channel < key_width; ++channel) {
            dot += query[query_offset + channel] * key[key_offset + channel];
        }
        const float score = dot * scale;
        scores[token] = score;
        local_maximum = fmaxf(local_maximum, score);
    }
    const float maximum = block_max_f32(local_maximum, partial);
    float local_sum = 0.0f;
    for (unsigned int token = threadIdx.x; token < key_value_tokens; token += blockDim.x) {
        const float probability = expf(scores[token] - maximum);
        scores[token] = probability;
        local_sum += probability;
    }
    const float sum = block_sum_f32(local_sum, partial);
    for (unsigned int channel = threadIdx.x; channel < value_width; channel += blockDim.x) {
        float weighted = 0.0f;
        for (unsigned int token = 0; token < key_value_tokens; ++token) {
            const unsigned int value_offset =
                ((sequence * key_value_stride + token) * key_value_heads + key_value_head) * value_width;
            weighted += scores[token] * value[value_offset + channel];
        }
        output[(sequence * query_heads + query_head) * value_width + channel] = weighted / sum;
    }
}

extern "C" __global__ void attention_online_f32(
        const float * query,
        const float * key,
        const float * value,
        const float * relative_bias,
        const float * sinks,
        const float * block_ids,
        const float * key_bias,
        float * output,
        unsigned int key_width,
        unsigned int value_width,
        unsigned int query_heads,
        unsigned int key_value_heads,
        unsigned int query_tokens,
        unsigned int key_value_tokens,
        unsigned int sequences,
        float scale,
        float softcap,
        float max_alibi_bias,
        unsigned int causal,
        unsigned int query_start,
        unsigned int window,
        unsigned int symmetric_window,
        unsigned int relative_buckets,
        unsigned int relative_bidirectional) {
    unsigned int row = blockIdx.x;
    const unsigned int query_head = row % query_heads;
    row /= query_heads;
    const unsigned int query_token = row % query_tokens;
    const unsigned int sequence = row / query_tokens;
    if (sequence >= sequences || value_width > blockDim.x) return;

    const unsigned int group_size = query_heads / key_value_heads;
    const unsigned int key_value_head = query_head / group_size;
    const unsigned int query_position = query_start + query_token;
    const unsigned int causal_limit = query_position + 1;
    unsigned int key_limit = causal ? causal_limit : key_value_tokens;
    if (causal && block_ids != nullptr && block_ids[query_position] >= 0.0f) {
        key_limit = key_value_tokens;
    }
    unsigned int key_first = window > 0 && causal_limit > window ? causal_limit - window : 0;
    if (symmetric_window == 2) {
        key_first = (query_position / window) * window;
    } else if (symmetric_window) {
        const unsigned int half_window = window / 2;
        key_first = query_position > half_window ? query_position - half_window : 0;
        const unsigned int symmetric_limit = query_position + half_window + 1;
        key_limit = symmetric_limit < key_value_tokens ? symmetric_limit : key_value_tokens;
    }

    unsigned int n_head_log2 = 1;
    while (n_head_log2 * 2 <= query_heads) n_head_log2 *= 2;
    float alibi_slope = 0.0f;
    if (max_alibi_bias > 0.0f) {
        const float m0 = powf(2.0f, -max_alibi_bias / (float) n_head_log2);
        const float m1 = powf(2.0f, -(max_alibi_bias / 2.0f) / (float) n_head_log2);
        alibi_slope = query_head < n_head_log2
            ? powf(m0, (float) (query_head + 1))
            : powf(m1, (float) (2 * (query_head - n_head_log2) + 1));
    }

    __shared__ float shared_maximum;
    __shared__ float shared_sum;
    __shared__ float shared_alpha;
    __shared__ float shared_beta;
    if (threadIdx.x == 0) {
        shared_maximum = sinks != nullptr ? sinks[query_head] : -3.402823466e+38F;
        shared_sum = sinks != nullptr ? 1.0f : 0.0f;
    }
    __syncthreads();

    float weighted = 0.0f;
    const unsigned int query_offset =
        ((sequence * query_tokens + query_token) * query_heads + query_head) * key_width;
    for (unsigned int key_token = key_first; key_token < key_limit; ++key_token) {
        if (threadIdx.x == 0) {
            bool included = !(causal && key_token >= causal_limit && (block_ids == nullptr ||
                block_ids[query_position] < 0.0f ||
                block_ids[key_token] != block_ids[query_position]));
            if (!included) {
                shared_alpha = 1.0f;
                shared_beta = 0.0f;
            } else {
                const unsigned int key_offset =
                    ((sequence * key_value_tokens + key_token) * key_value_heads + key_value_head) * key_width;
                float dot = 0.0f;
                for (unsigned int channel = 0; channel < key_width; ++channel) {
                    dot += query[query_offset + channel] * key[key_offset + channel];
                }
                float score = dot * scale;
                if (alibi_slope > 0.0f) {
                    const int distance = (int) query_position - (int) key_token;
                    score -= (float) (distance < 0 ? -distance : distance) * alibi_slope;
                }
                if (relative_bias != nullptr) {
                    unsigned int buckets = relative_buckets;
                    int distance = (int) key_token - (int) query_position;
                    unsigned int bucket = 0;
                    if (relative_bidirectional) {
                        buckets /= 2;
                        bucket = distance > 0 ? buckets : 0;
                    } else if (distance > 0) {
                        distance = 0;
                    }
                    const unsigned int max_exact = buckets / 2;
                    distance = distance < 0 ? -distance : distance;
                    if ((unsigned int) distance < max_exact) {
                        bucket += (unsigned int) distance;
                    } else {
                        unsigned int large = max_exact + (unsigned int) floorf(
                            logf((float) distance / (float) max_exact) *
                            (float) (buckets - max_exact) /
                            logf(128.0f / (float) max_exact));
                        bucket += large < buckets ? large : buckets - 1;
                    }
                    score += relative_bias[bucket * query_heads + query_head];
                }
                if (softcap > 0.0f) score = softcap * tanhf(score / softcap);
                if (key_bias != nullptr) score += key_bias[key_token];
                const float next_maximum = fmaxf(shared_maximum, score);
                shared_alpha = expf(shared_maximum - next_maximum);
                shared_beta = expf(score - next_maximum);
                shared_sum = shared_sum * shared_alpha + shared_beta;
                shared_maximum = next_maximum;
            }
        }
        __syncthreads();
        if (threadIdx.x < value_width) {
            const unsigned int value_offset =
                ((sequence * key_value_tokens + key_token) * key_value_heads + key_value_head) * value_width;
            weighted = weighted * shared_alpha + shared_beta * value[value_offset + threadIdx.x];
        }
        __syncthreads();
    }
    if (threadIdx.x < value_width) {
        const unsigned int output_offset =
            ((sequence * query_tokens + query_token) * query_heads + query_head) * value_width;
        output[output_offset + threadIdx.x] = weighted / shared_sum;
    }
}

// attention_online_init_f32: zero the query-chunk output rows and reset
// the per-row online softmax stats (max=-inf, sum=0) ahead of the key-tile
// loop. Stats layout: [max rows][sum rows].
extern "C" __global__ void attention_online_init_f32(
        float * output,
        float * stats,
        unsigned int output_count,
        unsigned int stat_rows,
        unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    if (index < output_count) {
        output[index] = 0.0f;
    }
    if (index < stat_rows) {
        stats[index] = -3.402823466e+38F;
        stats[stat_rows + index] = 0.0f;
    }
}

// attention_online_softmax_f32: exact online softmax over one L2-resident
// score tile (SGEMM QK^T staging, layout [heads][chunk][key_chunk]). One
// block per (head, row): update the running max, rescale the accumulated
// output row (skip the exact *1.0 identity), write probabilities in place
// for the PV accumulate, fold the tile sum into the running sum. FP64
// shared reductions. Port of the proven external online softmax structure;
// output rows live in the interleaved [token][head][channel] layout.
extern "C" __global__ void attention_online_softmax_f32(
        float * scores,
        float * output,
        float * stats,
        const float * key_bias,
        unsigned int key_start,
        unsigned int keys,
        unsigned int key_chunk,
        unsigned int rows,
        unsigned int chunk,
        unsigned int width,
        unsigned int heads,
        float scale) {
    extern __shared__ double reduce_shared[];
    __shared__ float alpha_shared;
    __shared__ float maximum_shared;
    const unsigned int head = blockIdx.x / rows;
    const unsigned int row = blockIdx.x % rows;
    const unsigned int tid = threadIdx.x;
    const unsigned int base = (head * chunk + row) * key_chunk;
    const unsigned int stat = head * chunk + row;
    const unsigned int stat_rows = heads * chunk;
    float local_maximum = -3.402823466e+38F;
    for (unsigned int j = tid; j < keys; j += blockDim.x) {
        const float bias = key_bias != nullptr ? key_bias[key_start + j] : 0.0f;
        local_maximum = fmaxf(local_maximum, scores[base + j] * scale + bias);
    }
    reduce_shared[tid] = (double) local_maximum;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reduce_shared[tid] = fmax(reduce_shared[tid], reduce_shared[tid + stride]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        const float old = stats[stat];
        const float next = fmaxf(old, (float) reduce_shared[0]);
        maximum_shared = next;
        alpha_shared = old <= -3.4028233e+38F ? 0.0f : expf(old - next);
        stats[stat] = next;
    }
    __syncthreads();
    const float alpha = alpha_shared;
    const float maximum = maximum_shared;
    if (alpha != 1.0f) {
        float * output_row = output + (row * heads + head) * width;
        for (unsigned int channel = tid; channel < width; channel += blockDim.x) {
            output_row[channel] *= alpha;
        }
    }
    double sum = 0.0;
    for (unsigned int j = tid; j < keys; j += blockDim.x) {
        const float bias = key_bias != nullptr ? key_bias[key_start + j] : 0.0f;
        const float probability = expf(scores[base + j] * scale + bias - maximum);
        scores[base + j] = probability;
        sum += (double) probability;
    }
    reduce_shared[tid] = sum;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reduce_shared[tid] += reduce_shared[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        stats[stat_rows + stat] = stats[stat_rows + stat] * alpha + (float) reduce_shared[0];
    }
}

// attention_online_finalize_f32: divide the accumulated interleaved output
// rows by the final online sums.
extern "C" __global__ void attention_online_finalize_f32(
        float * output,
        const float * stats,
        unsigned int chunk,
        unsigned int width,
        unsigned int heads,
        unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const unsigned int z = index / width;
    const unsigned int head = z % heads;
    const unsigned int row = z / heads;
    const float sum = stats[heads * chunk + head * chunk + row];
    if (sum > 0.0f) {
        output[index] /= sum;
    }
}

// Tiled exact flash-style attention for large non-causal workloads with
// optional per-key bias (no relative bias/sinks/blocks/softcap/alibi/window). One block owns a
// 32-query-row tile of one head; K/V stream through shared memory in
// 32-token tiles with an exact online softmax, so scores never touch
// global memory. Each of 8 warps owns 4 query rows; within a row, lane j
// owns tile token j for the score and broadcasts its probability for the
// value accumulation. Dynamic shared: Q tile (32*key_width) + K tile
// transposed (key_width*32) + V tile (32*value_width) floats.
extern "C" __global__ void attention_tiled_f32(
        const float * query,
        const float * key,
        const float * value,
        const float * key_bias,
        float * output,
        unsigned int key_width,
        unsigned int value_width,
        unsigned int query_heads,
        unsigned int key_value_heads,
        unsigned int query_tokens,
        unsigned int key_value_tokens,
        unsigned int sequences,
        float scale) {
    const unsigned int TILE = 32u;
    const unsigned int CW = 4u; // output channels per lane (width <= 128)
    const unsigned int row_tile = blockIdx.x;
    const unsigned int query_head = blockIdx.y;
    const unsigned int sequence = blockIdx.z;
    if (sequence >= sequences || query_head >= query_heads ||
        key_width > TILE * CW || value_width > TILE * CW) {
        return;
    }
    const unsigned int group_size = query_heads / key_value_heads;
    const unsigned int key_value_head = query_head / group_size;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31u;

    extern __shared__ float shared[];
    float * q_tile = shared;                       // [TILE][key_width]
    float * k_tile = q_tile + TILE * key_width;    // [key_width][TILE] transposed
    float * v_tile = k_tile + TILE * key_width;    // [TILE][value_width]

    // Cooperative Q tile load (rows beyond the sequence zero-fill).
    const unsigned int row_base = row_tile * TILE;
    for (unsigned int index = threadIdx.x; index < TILE * key_width; index += blockDim.x) {
        const unsigned int row = index / key_width;
        const unsigned int channel = index % key_width;
        const unsigned int token = row_base + row;
        q_tile[index] = token < query_tokens
            ? query[((sequence * query_tokens + token) * query_heads + query_head) * key_width + channel]
            : 0.0f;
    }
    __syncthreads();

    // Per-warp row state: 4 rows, per-lane CW output channels each.
    float maximum[4] = {-3.402823466e+38F, -3.402823466e+38F, -3.402823466e+38F, -3.402823466e+38F};
    float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float accumulator[4][CW];
    for (unsigned int r = 0; r < 4; ++r) {
        for (unsigned int i = 0; i < CW; ++i) {
            accumulator[r][i] = 0.0f;
        }
    }

    for (unsigned int tile_base = 0; tile_base < key_value_tokens; tile_base += TILE) {
        // Cooperative K (transposed) and V tile loads.
        for (unsigned int index = threadIdx.x; index < TILE * key_width; index += blockDim.x) {
            const unsigned int token = index / key_width;
            const unsigned int channel = index % key_width;
            const unsigned int kv = tile_base + token;
            k_tile[channel * TILE + token] = kv < key_value_tokens
                ? key[((sequence * key_value_tokens + kv) * key_value_heads + key_value_head) * key_width + channel]
                : 0.0f;
        }
        for (unsigned int index = threadIdx.x; index < TILE * value_width; index += blockDim.x) {
            const unsigned int token = index / value_width;
            const unsigned int channel = index % value_width;
            const unsigned int kv = tile_base + token;
            v_tile[index] = kv < key_value_tokens
                ? value[((sequence * key_value_tokens + kv) * key_value_heads + key_value_head) * value_width + channel]
                : 0.0f;
        }
        __syncthreads();

        for (unsigned int r = 0; r < 4; ++r) {
            const unsigned int row = warp * 4u + r;
            // Lane j scores tile token j against the shared Q row.
            float score = -3.402823466e+38F;
            if (tile_base + lane < key_value_tokens) {
                float dot = 0.0f;
                const float * q_row = q_tile + row * key_width;
                const float * k_column = k_tile + lane;
                for (unsigned int channel = 0; channel < key_width; ++channel) {
                    dot += q_row[channel] * k_column[channel * TILE];
                }
                score = dot * scale + (key_bias != nullptr ? key_bias[tile_base + lane] : 0.0f);
            }
            float tile_maximum = score;
            for (unsigned int offset = 16; offset > 0; offset >>= 1) {
                tile_maximum = fmaxf(tile_maximum, __shfl_xor_sync(0xffffffffu, tile_maximum, offset));
            }
            const float next_maximum = fmaxf(maximum[r], tile_maximum);
            const float alpha = expf(maximum[r] - next_maximum);
            maximum[r] = next_maximum;
            const float probability = tile_base + lane < key_value_tokens
                ? expf(score - next_maximum)
                : 0.0f;
            float tile_sum = probability;
            for (unsigned int offset = 16; offset > 0; offset >>= 1) {
                tile_sum += __shfl_xor_sync(0xffffffffu, tile_sum, offset);
            }
            sum[r] = sum[r] * alpha + tile_sum;
            for (unsigned int i = 0; i < CW; ++i) {
                accumulator[r][i] *= alpha;
            }
            for (unsigned int j = 0; j < TILE; ++j) {
                const float p = __shfl_sync(0xffffffffu, probability, j);
                const float * v_row = v_tile + j * value_width;
                for (unsigned int i = 0; i < CW; ++i) {
                    const unsigned int channel = lane + 32u * i;
                    if (channel < value_width) {
                        accumulator[r][i] += p * v_row[channel];
                    }
                }
            }
        }
        __syncthreads();
    }

    for (unsigned int r = 0; r < 4; ++r) {
        const unsigned int row = warp * 4u + r;
        const unsigned int token = row_base + row;
        if (token >= query_tokens || sum[r] <= 0.0f) {
            continue;
        }
        const unsigned int output_offset =
            ((sequence * query_tokens + token) * query_heads + query_head) * value_width;
        for (unsigned int i = 0; i < CW; ++i) {
            const unsigned int channel = lane + 32u * i;
            if (channel < value_width) {
                output[output_offset + channel] = accumulator[r][i] / sum[r];
            }
        }
    }
}

// attention_bf16_stage_rows: issue one 64-token panel's cp.async copies of a
// pre-packed BF16 tensor into the given shared stage (16 bytes = 8 bf16
// channels per copy; pointer pre-offset to the sequence and KV head); rows
// past the sequence zero-fill directly.
__device__ __forceinline__ void attention_bf16_stage_rows(
        const unsigned short * base,
        __nv_bfloat16 * stage,
        unsigned int tile_base,
        unsigned int key_value_tokens,
        unsigned int token_stride) {
    const unsigned int W = 128u;
    const unsigned int BN = 64u;
    const unsigned int LD = W + 8u;
    const unsigned int copies = BN * (W / 8u);
    for (unsigned int copy = threadIdx.x; copy < copies; copy += blockDim.x) {
        const unsigned int token = copy / (W / 8u);
        const unsigned int channel = (copy % (W / 8u)) * 8u;
        __nv_bfloat16 * destination = stage + token * LD + channel;
        const unsigned int kv = tile_base + token;
        if (kv < key_value_tokens) {
            __pipeline_memcpy_async(
                destination, base + (size_t) kv * token_stride + channel, 16);
        } else {
            for (unsigned int i = 0; i < 8u; ++i) {
                destination[i] = __float2bfloat16(0.0f);
            }
        }
    }
}

// attention_ldx4 / attention_ldx4_trans / attention_mma: the ldmatrix +
// mma.sync primitives behind the BF16 flash kernel. nvcc's wmma path lowers
// generic-space fragment loads to scalar loads (no LDSM), which leaves the
// tensor pipe idle; explicit ldmatrix restores 16-bytes-per-lane fragment
// loads. mma.m16n8k16 drives the same HMMA.16816 hardware ops as wmma
// m16n16k16 (identical per-element FMA order), so numerics stay in the
// declared kernel class. Register mappings below are the architected
// sm_80+ fragment layouts, verified by a basis-product calibration probe.
__device__ __forceinline__ unsigned int attention_shared_address(const void * pointer) {
    return (unsigned int) __cvta_generic_to_shared(pointer);
}

__device__ __forceinline__ void attention_ldx4(
        unsigned int address,
        unsigned int & r0, unsigned int & r1, unsigned int & r2, unsigned int & r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(address));
}

__device__ __forceinline__ void attention_ldx4_trans(
        unsigned int address,
        unsigned int & r0, unsigned int & r1, unsigned int & r2, unsigned int & r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(address));
}

__device__ __forceinline__ void attention_mma(
        float * c,
        const unsigned int * a,
        unsigned int b0, unsigned int b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// BF16 tensor-core flash attention for head width 128 (the fused
// Attention(BF16Round(q), BF16Round(k), BF16Round(v)) form). Storage
// rounding only: Q rounds to BF16 on stage-in (round-to-nearest-even,
// identical to bf16_round_f32) and K/V arrive pre-rounded to BF16 by the
// launcher's f32_to_bf16 pack (the same rounding class); products
// accumulate in F32 through mma.m16n8k16 fragments (ldmatrix-fed), and the
// online softmax runs in exact F32 on the staged score tile; probabilities
// round to BF16 for the value product. One 512-thread block (16 warps) owns
// 64 query rows of one head; K/V stream in 64-token panels: K
// double-buffers through cp.async (the next panel loads during the value
// product) while V single-buffers (its load overlaps scores and softmax).
// O accumulators stay register-resident with architected row mapping, so
// the online rescale is a per-register multiply (alpha is exactly 1 on
// steady-state panels, and x*1.0f is exact). The O spill scratch used by
// the final store aliases the Q+S region (Q is register-resident after the
// fragment preload and S is consumed into P first), keeping dynamic shared
// at K(2x64x136 bf16) + V(64x136 bf16) + Q/S-alias(64x136 bf16 + 64x68
// f32, >= O 64x132 f32) + P(64x72 bf16) + row stats = 97024 bytes.
extern "C" __global__ void __launch_bounds__(512, 1) attention_tiled_bf16_f32(
        const float * query,
        const unsigned short * key,
        const unsigned short * value,
        const float * key_bias,
        float * output,
        unsigned int query_heads,
        unsigned int key_value_heads,
        unsigned int query_tokens,
        unsigned int key_value_tokens,
        unsigned int sequences,
        float scale) {
    const unsigned int W = 128u;
    const unsigned int BM = 64u;
    const unsigned int BN = 64u;
    const unsigned int QLD = W + 8u;  // bf16 lds, multiple of 8
    const unsigned int KLD = W + 8u;
    const unsigned int VLD = W + 8u;
    const unsigned int SLD = BN + 4u; // f32 ld, multiple of 4
    const unsigned int PLD = BN + 8u; // bf16 ld, multiple of 8
    const unsigned int OLD = W + 4u;  // f32 ld, multiple of 4
    const unsigned int row_tile = blockIdx.x;
    const unsigned int query_head = blockIdx.y;
    const unsigned int sequence = blockIdx.z;
    if (sequence >= sequences || query_head >= query_heads) {
        return;
    }
    const unsigned int key_value_head = query_head / (query_heads / key_value_heads);
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31u;

    extern __shared__ unsigned char attention_shared[];
    __nv_bfloat16 * k_tile = (__nv_bfloat16 *) attention_shared;    // [2][BN][KLD]
    __nv_bfloat16 * v_tile = k_tile + 2u * BN * KLD;                // [BN][VLD]
    unsigned char * qs_region = (unsigned char *) (v_tile + BN * VLD);
    __nv_bfloat16 * q_tile = (__nv_bfloat16 *) qs_region;           // [BM][QLD]
    float * s_tile = (float *) (q_tile + BM * QLD);                 // [BM][SLD]
    // O spill scratch (final store only) aliases Q+S: Q lives only until
    // the fragment preload and S is dead once P is written.
    float * o_tile = (float *) qs_region;                           // [BM][OLD]
    __nv_bfloat16 * p_tile = (__nv_bfloat16 *) (s_tile + BM * SLD); // [BM][PLD]
    float * alpha_row = (float *) (p_tile + BM * PLD);              // [BM]
    float * maximum_row = alpha_row + BM;                           // [BM]
    float * sum_row = maximum_row + BM;                             // [BM]

    const unsigned int row_base = row_tile * BM;
    for (unsigned int index = threadIdx.x; index < BM * W; index += blockDim.x) {
        const unsigned int row = index >> 7;
        const unsigned int channel = index & 127u;
        const unsigned int token = row_base + row;
        q_tile[row * QLD + channel] = __float2bfloat16(token < query_tokens
            ? query[((sequence * query_tokens + token) * query_heads + query_head) * W + channel]
            : 0.0f);
    }
    for (unsigned int index = threadIdx.x; index < BM; index += blockDim.x) {
        maximum_row[index] = -3.402823466e+38F;
        sum_row[index] = 0.0f;
    }

    // Warp tiling: warp w owns S rows (w%4)*16 x columns (w/4)*16 and O rows
    // (w%4)*16 x columns (w/4)*32. Architected fragment lane roles: g = row
    // within the 8-row group, t = the lane's element pair.
    const unsigned int s_row0 = (warp & 3u) * 16u;
    const unsigned int s_column0 = (warp >> 2) * 16u;
    const unsigned int o_column_base = (warp >> 2) * 32u;
    const unsigned int g = lane >> 2;
    const unsigned int t = lane & 3u;
    // ldmatrix lane->address roles (16x16 operand = four 8x8 tiles).
    const unsigned int a_row = (lane & 7u) + ((lane >> 3u) & 1u) * 8u;  // tiles: row-halves first
    const unsigned int a_col = (lane >> 4u) * 8u;
    const unsigned int bk_row = (lane & 7u) + (lane >> 4u) * 8u;        // tiles: column-halves first
    const unsigned int bk_col = ((lane >> 3u) & 1u) * 8u;

    // O accumulators: four 16x8 column tiles, register-resident across the
    // whole stream; element rows are architected (g and g+8).
    float weighted[4][4];
    for (unsigned int tile = 0; tile < 4u; ++tile) {
        for (unsigned int element = 0; element < 4u; ++element) {
            weighted[tile][element] = 0.0f;
        }
    }
    __syncthreads();

    // Q fragments are panel-invariant: load the warp's score-row strip once
    // (q_tile is dead afterwards; the O spill scratch overlays it).
    unsigned int q_frag[8][4];
    for (unsigned int k = 0; k < 8u; ++k) {
        attention_ldx4(
            attention_shared_address(q_tile + (s_row0 + a_row) * QLD + k * 16u + a_col),
            q_frag[k][0], q_frag[k][1], q_frag[k][2], q_frag[k][3]);
    }
    __syncthreads();

    // Pre-offset K/V to the sequence and KV head; preload K panel 0.
    const unsigned int kv_token_stride = key_value_heads * W;
    const unsigned short * key_base = key +
        sequence * key_value_tokens * kv_token_stride + key_value_head * W;
    const unsigned short * value_base = value +
        sequence * key_value_tokens * kv_token_stride + key_value_head * W;
    attention_bf16_stage_rows(key_base, k_tile, 0u, key_value_tokens, kv_token_stride);
    __pipeline_commit();

    unsigned int stage = 0u;
    for (unsigned int tile_base = 0; tile_base < key_value_tokens; tile_base += BN, stage ^= 1u) {
        const bool has_next = tile_base + BN < key_value_tokens;
        // K for this tile is the only outstanding group; the barrier both
        // publishes it and closes out the previous value product, so v_tile
        // is free for this panel's V copy (which then overlaps scores).
        __pipeline_wait_prior(0);
        __syncthreads();
        attention_bf16_stage_rows(
            value_base, v_tile, tile_base, key_value_tokens, kv_token_stride);
        __pipeline_commit();
        const __nv_bfloat16 * k_stage = k_tile + stage * BN * KLD;

        {
            // S tile: two n-halves of the warp's 16x16 score tile.
            float scores[2][4] = {{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}};
            const __nv_bfloat16 * k_rows = k_stage + (s_column0 + bk_row) * KLD + bk_col;
            for (unsigned int k = 0; k < 8u; ++k) {
                unsigned int b0, b1, b2, b3;
                attention_ldx4(
                    attention_shared_address(k_rows + k * 16u), b0, b1, b2, b3);
                attention_mma(scores[0], q_frag[k], b0, b1);
                attention_mma(scores[1], q_frag[k], b2, b3);
            }
            float * s_low = s_tile + (s_row0 + g) * SLD + s_column0 + 2u * t;
            float * s_high = s_tile + (s_row0 + 8u + g) * SLD + s_column0 + 2u * t;
            *(float2 *) s_low = make_float2(scores[0][0], scores[0][1]);
            *(float2 *) s_high = make_float2(scores[0][2], scores[0][3]);
            *(float2 *) (s_low + 8u) = make_float2(scores[1][0], scores[1][1]);
            *(float2 *) (s_high + 8u) = make_float2(scores[1][2], scores[1][3]);
        }
        // The next K panel loads during softmax and the value product.
        if (has_next) {
            attention_bf16_stage_rows(
                key_base, k_tile + (stage ^ 1u) * BN * KLD,
                tile_base + BN, key_value_tokens, kv_token_stride);
            __pipeline_commit();
        }
        __syncthreads();

        // Exact online softmax on the staged F32 scores: eight threads per
        // row, reduced with warp shuffles.
        {
            const unsigned int row = threadIdx.x >> 3;
            const unsigned int part = threadIdx.x & 7u;
            const unsigned int remaining = key_value_tokens - tile_base;
            const unsigned int valid = remaining < BN ? remaining : BN;
            // One shared read per column: the scaled scores stay in
            // registers across the max and probability passes.
            float scaled[8];
            for (unsigned int i = 0; i < 8u; ++i) {
                const unsigned int key_token = tile_base + part + 8u * i;
                scaled[i] = s_tile[row * SLD + part + 8u * i] * scale +
                    (key_bias != nullptr && key_token < key_value_tokens ? key_bias[key_token] : 0.0f);
            }
            float local_maximum = -3.402823466e+38F;
            for (unsigned int i = 0; i < 8u; ++i) {
                if (part + 8u * i < valid) {
                    local_maximum = fmaxf(local_maximum, scaled[i]);
                }
            }
            local_maximum = fmaxf(local_maximum, __shfl_xor_sync(0xffffffffu, local_maximum, 1));
            local_maximum = fmaxf(local_maximum, __shfl_xor_sync(0xffffffffu, local_maximum, 2));
            local_maximum = fmaxf(local_maximum, __shfl_xor_sync(0xffffffffu, local_maximum, 4));
            const float next_maximum = fmaxf(maximum_row[row], local_maximum);
            const float alpha = expf(maximum_row[row] - next_maximum);
            float local_sum = 0.0f;
            for (unsigned int i = 0; i < 8u; ++i) {
                const float probability = part + 8u * i < valid
                    ? expf(scaled[i] - next_maximum)
                    : 0.0f;
                p_tile[row * PLD + part + 8u * i] = __float2bfloat16(probability);
                local_sum += probability;
            }
            local_sum += __shfl_xor_sync(0xffffffffu, local_sum, 1);
            local_sum += __shfl_xor_sync(0xffffffffu, local_sum, 2);
            local_sum += __shfl_xor_sync(0xffffffffu, local_sum, 4);
            if (part == 0) {
                maximum_row[row] = next_maximum;
                sum_row[row] = sum_row[row] * alpha + local_sum;
                alpha_row[row] = alpha;
            }
        }
        // V panel must be resident before the value product: all-but-one
        // leaves only the next K (or nothing on the last panel)
        // outstanding. One barrier publishes P, the row stats, and V.
        __pipeline_wait_prior(has_next ? 1 : 0);
        __syncthreads();

        {
            // Register-resident rescale: each accumulator element's row is
            // architected (g in the low half, g+8 in the high), so the
            // running-max correction is a plain multiply. Steady-state
            // panels carry alpha == 1 exactly and x*1.0f is exact.
            const float alpha_low = alpha_row[s_row0 + g];
            const float alpha_high = alpha_row[s_row0 + 8u + g];
            for (unsigned int tile = 0; tile < 4u; ++tile) {
                weighted[tile][0] *= alpha_low;
                weighted[tile][1] *= alpha_low;
                weighted[tile][2] *= alpha_high;
                weighted[tile][3] *= alpha_high;
            }
        }

        {
            // Value product: P fragments load once per token quarter and
            // feed all four O column tiles.
            for (unsigned int k = 0; k < 4u; ++k) {
                unsigned int p_frag[4];
                attention_ldx4(
                    attention_shared_address(
                        p_tile + (s_row0 + a_row) * PLD + k * 16u + a_col),
                    p_frag[0], p_frag[1], p_frag[2], p_frag[3]);
                const __nv_bfloat16 * v_rows =
                    v_tile + (k * 16u + a_row) * VLD + o_column_base + a_col;
                unsigned int b0, b1, b2, b3;
                attention_ldx4_trans(
                    attention_shared_address(v_rows), b0, b1, b2, b3);
                attention_mma(weighted[0], p_frag, b0, b1);
                attention_mma(weighted[1], p_frag, b2, b3);
                attention_ldx4_trans(
                    attention_shared_address(v_rows + 16u), b0, b1, b2, b3);
                attention_mma(weighted[2], p_frag, b0, b1);
                attention_mma(weighted[3], p_frag, b2, b3);
            }
        }
        // No trailing barrier: the next panel's top barrier (after the K
        // wait) closes this value product before v_tile is rewritten.
    }

    for (unsigned int tile = 0; tile < 4u; ++tile) {
        const unsigned int column = o_column_base + tile * 8u + 2u * t;
        *(float2 *) (o_tile + (s_row0 + g) * OLD + column) =
            make_float2(weighted[tile][0], weighted[tile][1]);
        *(float2 *) (o_tile + (s_row0 + 8u + g) * OLD + column) =
            make_float2(weighted[tile][2], weighted[tile][3]);
    }
    __syncthreads();

    for (unsigned int index = threadIdx.x; index < BM * W; index += blockDim.x) {
        const unsigned int row = index >> 7;
        const unsigned int channel = index & 127u;
        const unsigned int token = row_base + row;
        if (token < query_tokens && sum_row[row] > 0.0f) {
            output[((sequence * query_tokens + token) * query_heads + query_head) * W + channel] =
                o_tile[row * OLD + channel] / sum_row[row];
        }
    }
}

extern "C" __global__ void concat_f32(
        const float * left,
        const float * right,
        float * output,
        unsigned int inner,
        unsigned int left_axis,
        unsigned int right_axis,
        unsigned int axis,
        unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        const unsigned int inner_index = index % inner;
        const unsigned int axis_outer = index / inner;
        const unsigned int output_axis = left_axis + right_axis;
        const unsigned int axis_index = axis_outer % output_axis;
        const unsigned int outer_index = axis_outer / output_axis;
        output[index] = axis_index < left_axis
            ? left[(outer_index * left_axis + axis_index) * inner + inner_index]
            : right[(outer_index * right_axis + axis_index - left_axis) * inner + inner_index];
    }
}

extern "C" __global__ void get_rows_q8_0_f32(
        const unsigned char * table,
        const unsigned int * rows,
        float * output,
        unsigned int width,
        unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const unsigned int column = index % width;
    const unsigned int output_row = index / width;
    const unsigned int blocks_per_row = width / Q8_0_BLOCK_WIDTH;
    const unsigned int block_index =
        rows[output_row] * blocks_per_row + column / Q8_0_BLOCK_WIDTH;
    const unsigned char * block = table + block_index * Q8_0_BLOCK_BYTES;
    const float scale = __half2float(*reinterpret_cast<const __half *>(block));
    const signed char quantized =
        *(reinterpret_cast<const signed char *>(block + Q8_0_SCALE_BYTES) +
            column % Q8_0_BLOCK_WIDTH);
    output[index] = scale * (float) quantized;
}

extern "C" __global__ void mul_mat_q8_0_f32(
        const unsigned char * left,
        const float * right,
        float * output,
        unsigned int inner,
        unsigned int left_rows,
        unsigned int right_rows) {
    const unsigned int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int warp_index = thread_index / CUDA_WARP_WIDTH;
    const unsigned int right_tiles =
        (right_rows + Q8_0_VECTORS_PER_WARP - 1) / Q8_0_VECTORS_PER_WARP;
    const unsigned int left_row = warp_index % left_rows;
    const unsigned int right_tile = warp_index / left_rows;
    if (right_tile >= right_tiles) {
        return;
    }
    const unsigned int lane = threadIdx.x % CUDA_WARP_WIDTH;
    const unsigned int right_first = right_tile * Q8_0_VECTORS_PER_WARP;
    const unsigned int blocks_per_row = inner / Q8_0_BLOCK_WIDTH;
    const unsigned char * weight_row =
        left + left_row * blocks_per_row * Q8_0_BLOCK_BYTES;
    float sums[Q8_0_VECTORS_PER_WARP] = {};
    for (unsigned int block_index = 0; block_index < blocks_per_row; ++block_index) {
        const unsigned char * block =
            weight_row + block_index * Q8_0_BLOCK_BYTES;
        float scale = lane == 0
            ? __half2float(*reinterpret_cast<const __half *>(block))
            : 0.0f;
        scale = __shfl_sync(0xffffffff, scale, 0);
        const signed char * quantized =
            reinterpret_cast<const signed char *>(block + Q8_0_SCALE_BYTES);
        const unsigned int column = block_index * Q8_0_BLOCK_WIDTH + lane;
        const float weight = scale * (float) quantized[lane];
#pragma unroll
        for (unsigned int vector = 0; vector < Q8_0_VECTORS_PER_WARP; ++vector) {
            const unsigned int right_row = right_first + vector;
            if (right_row < right_rows) {
                sums[vector] += weight * right[right_row * inner + column];
            }
        }
    }
#pragma unroll
    for (unsigned int vector = 0; vector < Q8_0_VECTORS_PER_WARP; ++vector) {
        float sum = sums[vector];
        for (unsigned int offset = CUDA_WARP_WIDTH / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        const unsigned int right_row = right_first + vector;
        if (lane == 0 && right_row < right_rows) {
            output[right_row * left_rows + left_row] = sum;
        }
    }
}

__device__ __forceinline__ float q8_0_input_dot_row(
        const unsigned char * left,
        const unsigned char * right,
        unsigned int inner,
        unsigned int left_row,
        unsigned int lane) {
    const unsigned int blocks_per_row = inner / Q8_0_BLOCK_WIDTH;
    const unsigned char * weight_row =
        left + (size_t) left_row * blocks_per_row * Q8_0_BLOCK_BYTES;
    float sum = 0.0f;
    for (unsigned int block = lane; block < blocks_per_row; block += CUDA_WARP_WIDTH) {
        const unsigned char * weight_block = weight_row + block * Q8_0_BLOCK_BYTES;
        const unsigned char * input_block = right + block * Q8_INPUT_BLOCK_BYTES;
        const int * input_quants = reinterpret_cast<const int *>(
            input_block + sizeof(float));
        int dot = 0;
#pragma unroll
        for (unsigned int word = 0; word < Q8_0_BLOCK_WIDTH / sizeof(int); ++word) {
            const int weight_quant = load_i32_unaligned(
                weight_block + Q8_0_SCALE_BYTES + word * sizeof(int));
            dot = __dp4a(weight_quant, input_quants[word], dot);
        }
        const float weight_scale = __half2float(
            *reinterpret_cast<const __half *>(weight_block));
        const float input_scale = *reinterpret_cast<const float *>(input_block);
        sum += weight_scale * input_scale * (float) dot;
    }
    for (unsigned int offset = CUDA_WARP_WIDTH / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    return sum;
}

extern "C" __global__ void mul_mat_q8_0_input_f32(
        const unsigned char * left,
        const unsigned char * right,
        float * output,
        unsigned int inner,
        unsigned int left_rows,
        unsigned int right_rows) {
    const unsigned int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int warp_index = thread_index / CUDA_WARP_WIDTH;
    const unsigned int left_row = warp_index % left_rows;
    const unsigned int right_row = warp_index / left_rows;
    if (right_row >= right_rows) {
        return;
    }
    const unsigned int lane = threadIdx.x % CUDA_WARP_WIDTH;
    const unsigned char * input_row =
        right + (size_t) right_row * (inner / Q8_0_BLOCK_WIDTH) * Q8_INPUT_BLOCK_BYTES;
    const float sum = q8_0_input_dot_row(left, input_row, inner, left_row, lane);
    if (lane == 0) {
        output[right_row * left_rows + left_row] = sum;
    }
}

// warp-per-row BF16 matvec: packed pair loads keep coalesced 128B segments;
// fp32 input and accumulation. Decode-regime replacement for staged GEMMEx.
extern "C" __global__ void mul_mat_bf16_f32(
        const unsigned int * left,
        const float * right,
        float * output,
        unsigned int inner,
        unsigned int left_rows,
        unsigned int right_rows) {
    const unsigned int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int warp_index = thread_index / CUDA_WARP_WIDTH;
    const unsigned int left_row = warp_index % left_rows;
    const unsigned int right_row = warp_index / left_rows;
    if (right_row >= right_rows) {
        return;
    }
    const unsigned int lane = threadIdx.x % CUDA_WARP_WIDTH;
    const unsigned int pairs = inner / 2;
    const unsigned int * weight_row = left + (size_t) left_row * pairs;
    const float * input_row = right + (size_t) right_row * inner;
    float sum = 0.0f;
    for (unsigned int pair = lane; pair < pairs; pair += CUDA_WARP_WIDTH) {
        const unsigned int packed = weight_row[pair];
        const float low = __uint_as_float(packed << 16);
        const float high = __uint_as_float(packed & 0xffff0000U);
        sum += low * input_row[2 * pair] + high * input_row[2 * pair + 1];
    }
    for (unsigned int offset = CUDA_WARP_WIDTH / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    if (lane == 0) {
        output[right_row * left_rows + left_row] = sum;
    }
}

// warp-per-row F16 matvec: packed pair loads keep coalesced 128B segments;
// each F16 half upconverts to F32 per element BEFORE the FMA (lossless), same
// F32 accumulation as the BF16 decode kernel. Decode-regime native-F16 read
// (half the per-token weight bandwidth of the F32 fallback).
extern "C" __global__ void mul_mat_f16_f32(
        const unsigned int * left,
        const float * right,
        float * output,
        unsigned int inner,
        unsigned int left_rows,
        unsigned int right_rows) {
    const unsigned int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int warp_index = thread_index / CUDA_WARP_WIDTH;
    const unsigned int left_row = warp_index % left_rows;
    const unsigned int right_row = warp_index / left_rows;
    if (right_row >= right_rows) {
        return;
    }
    const unsigned int lane = threadIdx.x % CUDA_WARP_WIDTH;
    const unsigned int pairs = inner / 2;
    const unsigned int * weight_row = left + (size_t) left_row * pairs;
    const float * input_row = right + (size_t) right_row * inner;
    float sum = 0.0f;
    for (unsigned int pair = lane; pair < pairs; pair += CUDA_WARP_WIDTH) {
        const unsigned int packed = weight_row[pair];
        const float low = __half2float(__ushort_as_half((unsigned short) (packed & 0xffffU)));
        const float high = __half2float(__ushort_as_half((unsigned short) (packed >> 16)));
        sum += low * input_row[2 * pair] + high * input_row[2 * pair + 1];
    }
    for (unsigned int offset = CUDA_WARP_WIDTH / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    if (lane == 0) {
        output[right_row * left_rows + left_row] = sum;
    }
}

// warp-per-row FP8(E4M3) matvec with per-output-row scale. Native-dtype decode
// sibling of mul_mat_bf16_f32: lanes read packed uint32 quads of e4m3 bytes
// (coalesced 128B segments), decode 4 values in registers, F32 accumulate,
// warp-shuffle reduce, and the per-row F32 scale folds in ONCE at lane 0
// (y = scale[row] * dot(e4m3_row, x)). inner must be a multiple of 4.
extern "C" __global__ void mul_mat_fp8_f32(
        const unsigned int * left,
        const float * scale,
        const float * right,
        float * output,
        unsigned int inner,
        unsigned int left_rows,
        unsigned int right_rows) {
    const unsigned int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int warp_index = thread_index / CUDA_WARP_WIDTH;
    const unsigned int left_row = warp_index % left_rows;
    const unsigned int right_row = warp_index / left_rows;
    if (right_row >= right_rows) {
        return;
    }
    const unsigned int lane = threadIdx.x % CUDA_WARP_WIDTH;
    const unsigned int quads = inner / 4;
    const unsigned int * weight_row = left + (size_t) left_row * quads;
    const float * input_row = right + (size_t) right_row * inner;
    float sum = 0.0f;
    for (unsigned int quad = lane; quad < quads; quad += CUDA_WARP_WIDTH) {
        const unsigned int packed = weight_row[quad];
        const unsigned int base = 4u * quad;
        sum += fp8_e4m3_decode(packed & 0xffu)          * input_row[base]
             + fp8_e4m3_decode((packed >> 8) & 0xffu)   * input_row[base + 1]
             + fp8_e4m3_decode((packed >> 16) & 0xffu)  * input_row[base + 2]
             + fp8_e4m3_decode((packed >> 24) & 0xffu)  * input_row[base + 3];
    }
    for (unsigned int offset = CUDA_WARP_WIDTH / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    if (lane == 0) {
        output[right_row * left_rows + left_row] = sum * scale[left_row];
    }
}

__device__ __forceinline__ float bf16_dot_row(
        const unsigned int * left,
        const float * input_row,
        unsigned int pairs,
        unsigned int row,
        unsigned int lane) {
    const unsigned int * weight_row = left + (size_t) row * pairs;
    float sum = 0.0f;
    for (unsigned int pair = lane; pair < pairs; pair += CUDA_WARP_WIDTH) {
        const unsigned int packed = weight_row[pair];
        const float low = __uint_as_float(packed << 16);
        const float high = __uint_as_float(packed & 0xffff0000U);
        sum += low * input_row[2 * pair] + high * input_row[2 * pair + 1];
    }
    for (unsigned int offset = CUDA_WARP_WIDTH / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    return sum;
}

// BF16 matvec with residual epilogue: output = W.x + addend (one token).
extern "C" __global__ void mul_mat_bf16_add_f32(
        const unsigned int * left,
        const float * right,
        const float * addend,
        float * output,
        unsigned int inner,
        unsigned int left_rows) {
    const unsigned int warp_index =
        (blockIdx.x * blockDim.x + threadIdx.x) / CUDA_WARP_WIDTH;
    if (warp_index >= left_rows) {
        return;
    }
    const unsigned int lane = threadIdx.x % CUDA_WARP_WIDTH;
    const float sum = bf16_dot_row(left, right, inner / 2, warp_index, lane);
    if (lane == 0) {
        output[warp_index] = sum + addend[warp_index];
    }
}

// fused BF16 gate/up matvec pair with activation epilogue:
// output = activate(Wg.x) * (Wu.x) (one token).
extern "C" __global__ void mul_mat_bf16_gate_f32(
        const unsigned int * gate,
        const unsigned int * up,
        const float * right,
        float * output,
        unsigned int inner,
        unsigned int rows,
        unsigned int activation) {
    const unsigned int warp_index =
        (blockIdx.x * blockDim.x + threadIdx.x) / CUDA_WARP_WIDTH;
    if (warp_index >= rows) {
        return;
    }
    const unsigned int lane = threadIdx.x % CUDA_WARP_WIDTH;
    const unsigned int pairs = inner / 2;
    const float gated = bf16_dot_row(gate, right, pairs, warp_index, lane);
    const float scaled = bf16_dot_row(up, right, pairs, warp_index, lane);
    if (lane == 0) {
        output[warp_index] = activated_gate_value(gated, activation) * scaled;
    }
}

// BF16 matvec appended DIRECTLY into its cache slot: the standalone append
// copy launch does not exist. Offset is device-resident (token units of
// append_inner elements).
extern "C" __global__ void mul_mat_bf16_append_f32(
        const unsigned int * left,
        const float * right,
        float * output,
        const unsigned int * offset,
        unsigned int inner,
        unsigned int left_rows,
        unsigned int append_inner) {
    const unsigned int warp_index =
        (blockIdx.x * blockDim.x + threadIdx.x) / CUDA_WARP_WIDTH;
    if (warp_index >= left_rows) {
        return;
    }
    const unsigned int lane = threadIdx.x % CUDA_WARP_WIDTH;
    const float sum = bf16_dot_row(left, right, inner / 2, warp_index, lane);
    if (lane == 0) {
        output[(size_t) offset[0] * append_inner + warp_index] = sum;
    }
}

// BF16 head projection fused with block-level argmax: logits never
// materialize. Mirrors the Q8 partials contract; the shared reduction kernel
// argmax_q8_0_input_partials_f32 finishes the selection.
extern "C" __global__ void mul_mat_bf16_argmax_partials_f32(
        const unsigned int * left,
        const float * right,
        float * partials,
        unsigned int inner,
        unsigned int left_rows) {
    constexpr unsigned int warps_per_block = 8;
    constexpr unsigned int partial_values = 2;
    const unsigned int warp = threadIdx.x / CUDA_WARP_WIDTH;
    const unsigned int lane = threadIdx.x % CUDA_WARP_WIDTH;
    const unsigned int row = blockIdx.x * warps_per_block + warp;
    const bool valid = row < left_rows;
    const float sum = valid ? bf16_dot_row(left, right, inner / 2, row, lane) : 0.0f;
    __shared__ float values[warps_per_block];
    __shared__ unsigned int indices[warps_per_block];
    __shared__ unsigned int invalids[warps_per_block];
    if (lane == 0) {
        values[warp] = sum;
        indices[warp] = valid ? row : 0xffffffffU;
        invalids[warp] = valid && isnan(sum);
    }
    __syncthreads();
    if (threadIdx.x != 0) {
        return;
    }
    unsigned int best = 0xffffffffU;
    float best_value = 0.0f;
    unsigned int invalid = 0;
    for (unsigned int item = 0; item < warps_per_block; ++item) {
        invalid |= invalids[item];
        const unsigned int index = indices[item];
        if (index != 0xffffffffU &&
            (best == 0xffffffffU || top_k_before(values[item], index, best_value, best))) {
            best = index;
            best_value = values[item];
        }
    }
    partials[blockIdx.x * partial_values] = best_value;
    partials[blockIdx.x * partial_values + 1] = invalid ? -1.0f : (float) best;
}

extern "C" __global__ void mul_mat_q8_0_input_argmax_partials_f32(
        const unsigned char * left,
        const unsigned char * right,
        float * partials,
        unsigned int inner,
        unsigned int left_rows) {
    constexpr unsigned int warps_per_block = 8;
    constexpr unsigned int partial_values = 2;
    const unsigned int warp = threadIdx.x / CUDA_WARP_WIDTH;
    const unsigned int lane = threadIdx.x % CUDA_WARP_WIDTH;
    const unsigned int row = blockIdx.x * warps_per_block + warp;
    const bool valid = row < left_rows;
    const float sum = valid ? q8_0_input_dot_row(left, right, inner, row, lane) : 0.0f;
    __shared__ float values[warps_per_block];
    __shared__ unsigned int indices[warps_per_block];
    __shared__ unsigned int invalids[warps_per_block];
    if (lane == 0) {
        values[warp] = sum;
        indices[warp] = valid ? row : 0xffffffffU;
        invalids[warp] = valid && isnan(sum);
    }
    __syncthreads();
    if (threadIdx.x != 0) {
        return;
    }
    unsigned int best = 0xffffffffU;
    float best_value = 0.0f;
    unsigned int invalid = 0;
    for (unsigned int item = 0; item < warps_per_block; ++item) {
        invalid |= invalids[item];
        const unsigned int index = indices[item];
        if (index != 0xffffffffU &&
            (best == 0xffffffffU || top_k_before(values[item], index, best_value, best))) {
            best = index;
            best_value = values[item];
        }
    }
    partials[blockIdx.x * partial_values] = best_value;
    partials[blockIdx.x * partial_values + 1] = invalid ? -1.0f : (float) best;
}

extern "C" __global__ void argmax_q8_0_input_partials_f32(
        const float * partials,
        float * output,
        unsigned int count) {
    constexpr unsigned int partial_values = 2;
    const unsigned int lane = threadIdx.x;
    float best_value = 0.0f;
    unsigned int best = 0xffffffffU;
    unsigned int invalid = 0;
    for (unsigned int item = lane; item < count; item += blockDim.x) {
        const float value = partials[item * partial_values];
        const float raw_index = partials[item * partial_values + 1];
        invalid |= raw_index < 0.0f;
        const unsigned int index = (unsigned int) raw_index;
        if (raw_index >= 0.0f &&
            (best == 0xffffffffU || top_k_before(value, index, best_value, best))) {
            best = index;
            best_value = value;
        }
    }
    __shared__ float values[256];
    __shared__ unsigned int indices[256];
    __shared__ unsigned int invalids[256];
    values[lane] = best_value;
    indices[lane] = best;
    invalids[lane] = invalid;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            invalids[lane] |= invalids[lane + stride];
            const unsigned int right_index = indices[lane + stride];
            if (right_index != 0xffffffffU &&
                (indices[lane] == 0xffffffffU || top_k_before(
                    values[lane + stride], right_index, values[lane], indices[lane]))) {
                values[lane] = values[lane + stride];
                indices[lane] = right_index;
            }
        }
        __syncthreads();
    }
    if (lane == 0) {
        output[0] = invalids[0] ? -1.0f : (float) indices[0];
    }
}

__device__ unsigned int load_u32_le(const unsigned char * source) {
    return
        (unsigned int) source[0] |
        ((unsigned int) source[1] << 8) |
        ((unsigned int) source[2] << 16) |
        ((unsigned int) source[3] << 24);
}

__device__ float dequant_q4_0_value(
        const unsigned char * block,
        unsigned int column) {
    const float scale =
        __half2float(*reinterpret_cast<const __half *>(block));
    const unsigned int lane = column % 16;
    const unsigned int packed = block[2 + lane];
    const unsigned int value =
        column < 16 ? packed & 0x0f : packed >> 4;
    return scale * (float) ((int) value - 8);
}

__device__ float dequant_q4_1_value(
        const unsigned char * block,
        unsigned int column) {
    const float scale =
        __half2float(*reinterpret_cast<const __half *>(block));
    const float minimum =
        __half2float(*reinterpret_cast<const __half *>(block + 2));
    const unsigned int lane = column % 16;
    const unsigned int packed = block[4 + lane];
    const unsigned int value =
        column < 16 ? packed & 0x0f : packed >> 4;
    return scale * (float) value + minimum;
}

__device__ float dequant_q5_0_value(
        const unsigned char * block,
        unsigned int column) {
    const float scale =
        __half2float(*reinterpret_cast<const __half *>(block));
    const unsigned int high = load_u32_le(block + 2);
    const unsigned int lane = column % 16;
    const unsigned int packed = block[6 + lane];
    unsigned int value =
        column < 16 ? packed & 0x0f : packed >> 4;
    value |= ((high >> column) & 1) << 4;
    return scale * (float) ((int) value - 16);
}

__device__ float dequant_q5_1_value(
        const unsigned char * block,
        unsigned int column) {
    const float scale =
        __half2float(*reinterpret_cast<const __half *>(block));
    const float minimum =
        __half2float(*reinterpret_cast<const __half *>(block + 2));
    const unsigned int high = load_u32_le(block + 4);
    const unsigned int lane = column % 16;
    const unsigned int packed = block[8 + lane];
    unsigned int value =
        column < 16 ? packed & 0x0f : packed >> 4;
    value |= ((high >> column) & 1) << 4;
    return scale * (float) value + minimum;
}

__device__ float dequant_q8_1_value(
        const unsigned char * block,
        unsigned int column) {
    const float scale =
        __half2float(*reinterpret_cast<const __half *>(block));
    const signed char value =
        *(reinterpret_cast<const signed char *>(block + 4) + column);
    return scale * (float) value;
}

__device__ __constant__ float mxfp4_values[16] = {
    0.0f, 1.0f, 2.0f, 3.0f,
    4.0f, 6.0f, 8.0f, 12.0f,
    0.0f, -1.0f, -2.0f, -3.0f,
    -4.0f, -6.0f, -8.0f, -12.0f
};

__device__ float dequant_mxfp4_value(
        const unsigned char * block,
        unsigned int column) {
    const unsigned int exponent = block[0];
    const unsigned int scale_bits = exponent < 2 ?
        0x00200000u << exponent :
        (exponent - 1) << 23;
    const float scale = __uint_as_float(scale_bits);
    const unsigned int lane = column % 16;
    const unsigned int packed = block[1 + lane];
    const unsigned int value =
        column < 16 ? packed & 0x0f : packed >> 4;
    return scale * mxfp4_values[value];
}

__device__ float ue4m3_to_float(unsigned int value) {
    if (value == 0 || value == 0x7f) {
        return 0.0f;
    }
    const unsigned int exponent = (value >> 3) & 0x0f;
    const unsigned int mantissa = value & 0x07;
    if (exponent == 0) {
        return (float) mantissa / 1024.0f;
    }
    const float exponent_scale =
        __uint_as_float((119u + exponent) << 23);
    return (1.0f + (float) mantissa / 8.0f) * exponent_scale;
}

__device__ float dequant_nvfp4_value(
        const unsigned char * block,
        unsigned int column) {
    const unsigned int sub_block = column / 16;
    const unsigned int within = column % 16;
    const unsigned int lane = within % 8;
    const float scale = ue4m3_to_float(block[sub_block]);
    const unsigned int packed =
        block[4 + sub_block * 8 + lane];
    const unsigned int value =
        within < 8 ? packed & 0x0f : packed >> 4;
    return scale * mxfp4_values[value];
}

#define DEFINE_CLASSIC_QUANT_KERNELS(TAG, BLOCK_BYTES, DEQUANT) \
extern "C" __global__ void get_rows_##TAG##_f32( \
        const unsigned char * table, \
        const unsigned int * rows, \
        float * output, \
        unsigned int width, \
        unsigned int count) { \
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x; \
    if (index >= count) { \
        return; \
    } \
    const unsigned int column = index % width; \
    const unsigned int output_row = index / width; \
    const unsigned int blocks_per_row = width / 32; \
    const unsigned int block_index = \
        rows[output_row] * blocks_per_row + column / 32; \
    output[index] = DEQUANT( \
        table + block_index * BLOCK_BYTES, \
        column % 32); \
} \
extern "C" __global__ void mul_mat_##TAG##_f32( \
        const unsigned char * left, \
        const float * right, \
        float * output, \
        unsigned int inner, \
        unsigned int left_rows, \
        unsigned int right_rows) { \
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x; \
    const unsigned int count = left_rows * right_rows; \
    if (index >= count) { \
        return; \
    } \
    const unsigned int left_row = index % left_rows; \
    const unsigned int right_row = index / left_rows; \
    const unsigned int blocks_per_row = inner / 32; \
    const unsigned char * weight_row = \
        left + left_row * blocks_per_row * BLOCK_BYTES; \
    const float * input_row = right + right_row * inner; \
    float sum = 0.0f; \
    for (unsigned int block_index = 0; block_index < blocks_per_row; ++block_index) { \
        const unsigned char * block = weight_row + block_index * BLOCK_BYTES; \
        const unsigned int column_base = block_index * 32; \
        for (unsigned int column = 0; column < 32; ++column) { \
            sum += DEQUANT(block, column) * input_row[column_base + column]; \
        } \
    } \
    output[index] = sum; \
}

DEFINE_CLASSIC_QUANT_KERNELS(q4_0, 18, dequant_q4_0_value)
DEFINE_CLASSIC_QUANT_KERNELS(q4_1, 20, dequant_q4_1_value)
DEFINE_CLASSIC_QUANT_KERNELS(q5_0, 22, dequant_q5_0_value)
DEFINE_CLASSIC_QUANT_KERNELS(q5_1, 24, dequant_q5_1_value)
DEFINE_CLASSIC_QUANT_KERNELS(q8_1, 36, dequant_q8_1_value)
DEFINE_CLASSIC_QUANT_KERNELS(mxfp4, 17, dequant_mxfp4_value)

__device__ float dequant_q1_0_value(
        const unsigned char * block,
        unsigned int column) {
    const float scale =
        __half2float(*reinterpret_cast<const __half *>(block));
    const unsigned int value =
        (block[2 + column / 8] >> (column % 8)) & 1;
    return value != 0 ? scale : -scale;
}

__device__ float dequant_q2_0_value(
        const unsigned char * block,
        unsigned int column) {
    const float scale =
        __half2float(*reinterpret_cast<const __half *>(block));
    const unsigned int value =
        (block[2 + column / 4] >> (2 * (column % 4))) & 0x03;
    return scale * (float) ((int) value - 1);
}

__device__ float dequant_tq2_0_value(
        const unsigned char * block,
        unsigned int column) {
    const float scale =
        __half2float(*reinterpret_cast<const __half *>(block + 64));
    const unsigned int half = column / 128;
    const unsigned int within = column % 128;
    const unsigned int pair = within / 32;
    const unsigned int lane = within % 32;
    const unsigned int value =
        (block[half * 32 + lane] >> (pair * 2)) & 0x03;
    return scale * (float) ((int) value - 1);
}

__device__ float dequant_tq1_0_value(
        const unsigned char * block,
        unsigned int column) {
    const unsigned int powers_of_three[5] = {1, 3, 9, 27, 81};
    const float scale =
        __half2float(*reinterpret_cast<const __half *>(block + 52));
    unsigned int packed;
    unsigned int trit;
    if (column < 160) {
        trit = column / 32;
        packed = block[column % 32];
    } else if (column < 240) {
        const unsigned int within = column - 160;
        trit = within / 16;
        packed = block[32 + within % 16];
    } else {
        const unsigned int within = column - 240;
        trit = within / 4;
        packed = block[48 + within % 4];
    }
    packed = (packed * powers_of_three[trit]) & 0xff;
    const unsigned int value = (packed * 3) >> 8;
    return scale * (float) ((int) value - 1);
}

__device__ float dequant_q8_K_value(
        const unsigned char * block,
        unsigned int column) {
    const float scale = *reinterpret_cast<const float *>(block);
    const signed char value =
        *(reinterpret_cast<const signed char *>(block + 4) + column);
    return scale * (float) value;
}

__device__ unsigned int read_u16_le(const unsigned char * value) {
    return (unsigned int) value[0] | ((unsigned int) value[1] << 8);
}

__device__ unsigned int read_u32_le(const unsigned char * value) {
    return
        (unsigned int) value[0] |
        ((unsigned int) value[1] << 8) |
        ((unsigned int) value[2] << 16) |
        ((unsigned int) value[3] << 24);
}

__device__ unsigned int iq_sign_mask(unsigned int index) {
    index &= 0x7f;
    unsigned int parity = index;
    parity ^= parity >> 4;
    parity ^= parity >> 2;
    parity ^= parity >> 1;
    return index | ((parity & 1) << 7);
}

__device__ float iq_grid64_lane(unsigned long long grid, unsigned int lane) {
    return (float) ((grid >> (lane * 8)) & 0xff);
}

__device__ float iq_grid32_lane(unsigned int grid, unsigned int lane) {
    return (float) ((grid >> (lane * 8)) & 0xff);
}

__device__ float iq_grid1_lane(unsigned long long grid, unsigned int lane) {
    return (float) (signed char) ((grid >> (lane * 8)) & 0xff);
}

__device__ float dequant_iq2_xxs_value(
        const unsigned char * block,
        unsigned int column) {
    const float scale =
        __half2float(*reinterpret_cast<const __half *>(block));
    const unsigned int group = column / 32;
    const unsigned int within = column % 32;
    const unsigned int sub_group = within / 8;
    const unsigned int lane = within % 8;
    const unsigned char * packed = block + 2 + group * 8;
    const unsigned int first = read_u32_le(packed);
    const unsigned int second = read_u32_le(packed + 4);
    const unsigned int grid_index =
        (first >> (sub_group * 8)) & 0xff;
    const unsigned int signs = iq_sign_mask(
        (second >> (sub_group * 7)) & 0x7f);
    float value =
        scale * (0.5f + (float) (second >> 28)) * 0.25f *
        iq_grid64_lane(iq2XXSGrid[grid_index], lane);
    return (signs & (1u << lane)) != 0 ? -value : value;
}

__device__ float dequant_iq2_xs_value(
        const unsigned char * block,
        unsigned int column) {
    const float scale =
        __half2float(*reinterpret_cast<const __half *>(block));
    const unsigned int group = column / 32;
    const unsigned int within = column % 32;
    const unsigned int sub_group = within / 8;
    const unsigned int lane = within % 8;
    const unsigned int packed = read_u16_le(
        block + 2 + (group * 4 + sub_group) * 2);
    const unsigned int packed_scale = block[66 + group];
    const unsigned int group_scale =
        sub_group < 2 ? packed_scale & 0x0f : packed_scale >> 4;
    const unsigned int signs = iq_sign_mask(packed >> 9);
    float value =
        scale * (0.5f + (float) group_scale) * 0.25f *
        iq_grid64_lane(iq2XSGrid[packed & 0x01ff], lane);
    return (signs & (1u << lane)) != 0 ? -value : value;
}

__device__ float dequant_iq2_s_value(
        const unsigned char * block,
        unsigned int column) {
    const float scale =
        __half2float(*reinterpret_cast<const __half *>(block));
    const unsigned int group = column / 32;
    const unsigned int within = column % 32;
    const unsigned int sub_group = within / 8;
    const unsigned int lane = within % 8;
    const unsigned char * quantized = block + 2;
    const unsigned int grid_index =
        quantized[group * 4 + sub_group] |
        ((((unsigned int) block[66 + group]) << (8 - 2 * sub_group)) & 0x0300);
    const unsigned int signs = quantized[32 + group * 4 + sub_group];
    const unsigned int packed_scale = block[74 + group];
    const unsigned int group_scale =
        sub_group < 2 ? packed_scale & 0x0f : packed_scale >> 4;
    float value =
        scale * (0.5f + (float) group_scale) * 0.25f *
        iq_grid64_lane(iq2SGrid[grid_index], lane);
    return (signs & (1u << lane)) != 0 ? -value : value;
}

__device__ float dequant_iq3_xxs_value(
        const unsigned char * block,
        unsigned int column) {
    const float scale =
        __half2float(*reinterpret_cast<const __half *>(block));
    const unsigned int group = column / 32;
    const unsigned int within = column % 32;
    const unsigned int sub_group = within / 8;
    const unsigned int lane = within % 8;
    const unsigned int packed = read_u32_le(block + 66 + group * 4);
    const unsigned int signs = iq_sign_mask(
        (packed >> (sub_group * 7)) & 0x7f);
    const unsigned int grid_index =
        block[2 + group * 8 + sub_group * 2 + lane / 4];
    float value =
        scale * (0.5f + (float) (packed >> 28)) * 0.5f *
        iq_grid32_lane(iq3XXSGrid[grid_index], lane % 4);
    return (signs & (1u << lane)) != 0 ? -value : value;
}

__device__ float dequant_iq3_s_value(
        const unsigned char * block,
        unsigned int column) {
    const float scale =
        __half2float(*reinterpret_cast<const __half *>(block));
    const unsigned int pair = column / 64;
    const unsigned int within_pair = column % 64;
    const unsigned int half = within_pair / 32;
    const unsigned int within = within_pair % 32;
    const unsigned int sub_group = within / 8;
    const unsigned int lane = within % 8;
    const unsigned int q_base = pair * 16 + half * 8;
    const unsigned int high = block[66 + pair * 2 + half];
    const unsigned int q_index = q_base + sub_group * 2 + lane / 4;
    const unsigned int high_shift =
        lane < 4 ? 8 - 2 * sub_group : 7 - 2 * sub_group;
    const unsigned int grid_index =
        block[2 + q_index] | ((high << high_shift) & 0x0100);
    const unsigned int signs =
        block[74 + pair * 8 + half * 4 + sub_group];
    const unsigned int packed_scale = block[106 + pair];
    const unsigned int group_scale =
        half == 0 ? packed_scale & 0x0f : packed_scale >> 4;
    float value =
        scale * (float) (1 + 2 * group_scale) *
        iq_grid32_lane(iq3SGrid[grid_index], lane % 4);
    return (signs & (1u << lane)) != 0 ? -value : value;
}

__device__ float dequant_iq1_s_value(
        const unsigned char * block,
        unsigned int column) {
    const float scale =
        __half2float(*reinterpret_cast<const __half *>(block));
    const unsigned int group = column / 32;
    const unsigned int within = column % 32;
    const unsigned int sub_group = within / 8;
    const unsigned int lane = within % 8;
    const unsigned int packed_high = read_u16_le(block + 34 + group * 2);
    const unsigned int grid_index =
        block[2 + group * 4 + sub_group] |
        (((packed_high >> (sub_group * 3)) & 7) << 8);
    const float delta =
        (packed_high & 0x8000) != 0 ? -0.125f : 0.125f;
    return
        scale * (float) (2 * ((packed_high >> 12) & 7) + 1) *
        (iq_grid1_lane(iq1SGrid[grid_index], lane) + delta);
}

__device__ float half_bits_to_float(unsigned int bits) {
    union {
        unsigned short bits;
        __half value;
    } converted;
    converted.bits = (unsigned short) bits;
    return __half2float(converted.value);
}

__device__ float dequant_iq1_m_value(
        const unsigned char * block,
        unsigned int column) {
    const unsigned int group = column / 32;
    const unsigned int within = column % 32;
    const unsigned int sub_group = within / 8;
    const unsigned int lane = within % 8;
    const unsigned int scale_words[4] = {
        read_u16_le(block + 48),
        read_u16_le(block + 50),
        read_u16_le(block + 52),
        read_u16_le(block + 54)
    };
    const unsigned int scale_bits =
        (scale_words[0] >> 12) |
        ((scale_words[1] >> 8) & 0x00f0) |
        ((scale_words[2] >> 4) & 0x0f00) |
        (scale_words[3] & 0xf000);
    const unsigned int scale_word = scale_words[group / 2];
    const unsigned int scale_shift = 6 * (group % 2) + 3 * (sub_group / 2);
    const float group_scale =
        half_bits_to_float(scale_bits) *
        (float) (2 * ((scale_word >> scale_shift) & 7) + 1);
    const unsigned int q_base = group * 4;
    const unsigned int high_base = group * 2 + sub_group / 2;
    const unsigned int high = block[32 + high_base];
    const unsigned int high_shift = sub_group % 2 == 0 ? 8 : 4;
    const unsigned int grid_index =
        block[q_base + sub_group] |
        ((high << high_shift) & 0x0700);
    const unsigned int delta_bit = sub_group % 2 == 0 ? 0x08 : 0x80;
    const float delta = (high & delta_bit) != 0 ? -0.125f : 0.125f;
    return group_scale *
        (iq_grid1_lane(iq1SGrid[grid_index], lane) + delta);
}

#define DEFINE_SMALL_QUANT_KERNELS(TAG, BLOCK_WIDTH, BLOCK_BYTES, DEQUANT) \
extern "C" __global__ void get_rows_##TAG##_f32( \
        const unsigned char * table, \
        const unsigned int * rows, \
        float * output, \
        unsigned int width, \
        unsigned int count) { \
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x; \
    if (index >= count) { \
        return; \
    } \
    const unsigned int column = index % width; \
    const unsigned int output_row = index / width; \
    const unsigned int blocks_per_row = width / BLOCK_WIDTH; \
    const unsigned int block_index = \
        rows[output_row] * blocks_per_row + column / BLOCK_WIDTH; \
    output[index] = DEQUANT( \
        table + block_index * BLOCK_BYTES, \
        column % BLOCK_WIDTH); \
} \
extern "C" __global__ void mul_mat_##TAG##_f32( \
        const unsigned char * left, \
        const float * right, \
        float * output, \
        unsigned int inner, \
        unsigned int left_rows, \
        unsigned int right_rows) { \
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x; \
    const unsigned int count = left_rows * right_rows; \
    if (index >= count) { \
        return; \
    } \
    const unsigned int left_row = index % left_rows; \
    const unsigned int right_row = index / left_rows; \
    const unsigned int blocks_per_row = inner / BLOCK_WIDTH; \
    const unsigned char * weight_row = \
        left + left_row * blocks_per_row * BLOCK_BYTES; \
    const float * input_row = right + right_row * inner; \
    float sum = 0.0f; \
    for (unsigned int block_index = 0; block_index < blocks_per_row; ++block_index) { \
        const unsigned char * block = weight_row + block_index * BLOCK_BYTES; \
        const unsigned int column_base = block_index * BLOCK_WIDTH; \
        for (unsigned int column = 0; column < BLOCK_WIDTH; ++column) { \
            sum += DEQUANT(block, column) * input_row[column_base + column]; \
        } \
    } \
    output[index] = sum; \
}

DEFINE_SMALL_QUANT_KERNELS(q1_0, 128, 18, dequant_q1_0_value)
DEFINE_SMALL_QUANT_KERNELS(q2_0, 64, 18, dequant_q2_0_value)
DEFINE_SMALL_QUANT_KERNELS(tq1_0, 256, 54, dequant_tq1_0_value)
DEFINE_SMALL_QUANT_KERNELS(tq2_0, 256, 66, dequant_tq2_0_value)
DEFINE_SMALL_QUANT_KERNELS(q8_K, 256, 292, dequant_q8_K_value)
DEFINE_SMALL_QUANT_KERNELS(nvfp4, 64, 36, dequant_nvfp4_value)
DEFINE_SMALL_QUANT_KERNELS(iq2_xxs, 256, 66, dequant_iq2_xxs_value)
DEFINE_SMALL_QUANT_KERNELS(iq2_xs, 256, 74, dequant_iq2_xs_value)
DEFINE_SMALL_QUANT_KERNELS(iq2_s, 256, 82, dequant_iq2_s_value)
DEFINE_SMALL_QUANT_KERNELS(iq3_xxs, 256, 98, dequant_iq3_xxs_value)
DEFINE_SMALL_QUANT_KERNELS(iq3_s, 256, 110, dequant_iq3_s_value)
DEFINE_SMALL_QUANT_KERNELS(iq1_s, 256, 50, dequant_iq1_s_value)
DEFINE_SMALL_QUANT_KERNELS(iq1_m, 256, 56, dequant_iq1_m_value)

__device__ float dequant_q2_K_value(
        const unsigned char * block,
        unsigned int column) {
    const unsigned char * scales = block;
    const unsigned char * quantized = block + 16;
    const float scale =
        __half2float(*reinterpret_cast<const __half *>(block + 80));
    const float minimum =
        __half2float(*reinterpret_cast<const __half *>(block + 82));
    const unsigned int group = column / 16;
    const unsigned int lane = column % 16;
    const unsigned int group_in_half = group % 8;
    const unsigned int quantized_index =
        (group / 8) * 32 + (group_in_half % 2) * 16 + lane;
    const unsigned int shift = (group_in_half / 2) * 2;
    const unsigned int value = (quantized[quantized_index] >> shift) & 0x03;
    return scale * (float) (scales[group] & 0x0f) * (float) value -
        minimum * (float) (scales[group] >> 4);
}

extern "C" __global__ void get_rows_q2_K_f32(
        const unsigned char * table,
        const unsigned int * rows,
        float * output,
        unsigned int width,
        unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const unsigned int column = index % width;
    const unsigned int output_row = index / width;
    const unsigned int blocks_per_row = width / 256;
    const unsigned int block_index =
        rows[output_row] * blocks_per_row + column / 256;
    output[index] = dequant_q2_K_value(
        table + block_index * 84,
        column % 256);
}

extern "C" __global__ void mul_mat_q2_K_f32(
        const unsigned char * left,
        const float * right,
        float * output,
        unsigned int inner,
        unsigned int left_rows,
        unsigned int right_rows) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int count = left_rows * right_rows;
    if (index >= count) {
        return;
    }
    const unsigned int left_row = index % left_rows;
    const unsigned int right_row = index / left_rows;
    const unsigned int blocks_per_row = inner / 256;
    const unsigned char * weight_row =
        left + left_row * blocks_per_row * 84;
    const float * input_row = right + right_row * inner;
    float sum = 0.0f;
    for (unsigned int block_index = 0; block_index < blocks_per_row; ++block_index) {
        const unsigned char * block = weight_row + block_index * 84;
        const unsigned int column_base = block_index * 256;
        for (unsigned int column = 0; column < 256; ++column) {
            sum += dequant_q2_K_value(block, column) *
                input_row[column_base + column];
        }
    }
    output[index] = sum;
}

__device__ float dequant_q3_K_value(
        const unsigned char * block,
        unsigned int column) {
    const unsigned char * high = block;
    const unsigned char * quantized = block + 32;
    const unsigned char * scales = block + 96;
    const float scale =
        __half2float(*reinterpret_cast<const __half *>(block + 108));
    const unsigned int group = column / 16;
    const unsigned int lane = column % 16;
    unsigned int low_scale = scales[group % 8];
    low_scale = group >= 8 ? low_scale >> 4 : low_scale & 0x0f;
    const unsigned int high_scale =
        (scales[8 + group % 4] >> (2 * (group / 4))) & 0x03;
    const int group_scale = (int) (low_scale | (high_scale << 4)) - 32;
    const unsigned int group_in_half = group % 8;
    const unsigned int quantized_index =
        (group / 8) * 32 + (group_in_half % 2) * 16 + lane;
    const unsigned int shift = (group_in_half / 2) * 2;
    int value = (int) ((quantized[quantized_index] >> shift) & 0x03);
    const unsigned int mask = 1u << (group / 2);
    if ((high[(group_in_half % 2) * 16 + lane] & mask) == 0) {
        value -= 4;
    }
    return scale * (float) group_scale * (float) value;
}

extern "C" __global__ void get_rows_q3_K_f32(
        const unsigned char * table,
        const unsigned int * rows,
        float * output,
        unsigned int width,
        unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const unsigned int column = index % width;
    const unsigned int output_row = index / width;
    const unsigned int blocks_per_row = width / 256;
    const unsigned int block_index =
        rows[output_row] * blocks_per_row + column / 256;
    output[index] = dequant_q3_K_value(
        table + block_index * 110,
        column % 256);
}

extern "C" __global__ void mul_mat_q3_K_f32(
        const unsigned char * left,
        const float * right,
        float * output,
        unsigned int inner,
        unsigned int left_rows,
        unsigned int right_rows) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int count = left_rows * right_rows;
    if (index >= count) {
        return;
    }
    const unsigned int left_row = index % left_rows;
    const unsigned int right_row = index / left_rows;
    const unsigned int blocks_per_row = inner / 256;
    const unsigned char * weight_row =
        left + left_row * blocks_per_row * 110;
    const float * input_row = right + right_row * inner;
    float sum = 0.0f;
    for (unsigned int block_index = 0; block_index < blocks_per_row; ++block_index) {
        const unsigned char * block = weight_row + block_index * 110;
        const unsigned int column_base = block_index * 256;
        for (unsigned int column = 0; column < 256; ++column) {
            sum += dequant_q3_K_value(block, column) *
                input_row[column_base + column];
        }
    }
    output[index] = sum;
}

__device__ void scale_min_k4(
        unsigned int index,
        const unsigned char * packed,
        unsigned int * scale,
        unsigned int * minimum) {
    if (index < 4) {
        *scale = packed[index] & 0x3f;
        *minimum = packed[index + 4] & 0x3f;
        return;
    }
    *scale =
        (packed[index + 4] & 0x0f) |
        ((packed[index - 4] >> 6) << 4);
    *minimum =
        (packed[index + 4] >> 4) |
        ((packed[index] >> 6) << 4);
}

__device__ float dequant_q4_K_value(
        const unsigned char * block,
        unsigned int column) {
    const float scale =
        __half2float(*reinterpret_cast<const __half *>(block));
    const float minimum =
        __half2float(*reinterpret_cast<const __half *>(block + 2));
    const unsigned char * scales = block + 4;
    const unsigned char * quantized = block + 16;
    const unsigned int group = column / 32;
    const unsigned int lane = column % 32;
    unsigned int group_scale;
    unsigned int group_minimum;
    scale_min_k4(group, scales, &group_scale, &group_minimum);
    const unsigned int packed = quantized[(group / 2) * 32 + lane];
    const unsigned int value =
        group % 2 == 0 ? packed & 0x0f : packed >> 4;
    return scale * (float) group_scale * (float) value -
        minimum * (float) group_minimum;
}

extern "C" __global__ void get_rows_q4_K_f32(
        const unsigned char * table,
        const unsigned int * rows,
        float * output,
        unsigned int width,
        unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const unsigned int column = index % width;
    const unsigned int output_row = index / width;
    const unsigned int blocks_per_row = width / 256;
    const unsigned int block_index =
        rows[output_row] * blocks_per_row + column / 256;
    output[index] = dequant_q4_K_value(
        table + block_index * 144,
        column % 256);
}

extern "C" __global__ void mul_mat_q4_K_f32(
        const unsigned char * left,
        const float * right,
        float * output,
        unsigned int inner,
        unsigned int left_rows,
        unsigned int right_rows) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int count = left_rows * right_rows;
    if (index >= count) {
        return;
    }
    const unsigned int left_row = index % left_rows;
    const unsigned int right_row = index / left_rows;
    const unsigned int blocks_per_row = inner / 256;
    const unsigned char * weight_row =
        left + left_row * blocks_per_row * 144;
    const float * input_row = right + right_row * inner;
    float sum = 0.0f;
    for (unsigned int block_index = 0; block_index < blocks_per_row; ++block_index) {
        const unsigned char * block = weight_row + block_index * 144;
        const unsigned int column_base = block_index * 256;
        for (unsigned int column = 0; column < 256; ++column) {
            sum += dequant_q4_K_value(block, column) *
                input_row[column_base + column];
        }
    }
    output[index] = sum;
}

__device__ float dequant_q5_K_value(
        const unsigned char * block,
        unsigned int column) {
    const float scale =
        __half2float(*reinterpret_cast<const __half *>(block));
    const float minimum =
        __half2float(*reinterpret_cast<const __half *>(block + 2));
    const unsigned char * scales = block + 4;
    const unsigned char * high = block + 16;
    const unsigned char * quantized = block + 48;
    const unsigned int group = column / 32;
    const unsigned int lane = column % 32;
    unsigned int group_scale;
    unsigned int group_minimum;
    scale_min_k4(group, scales, &group_scale, &group_minimum);
    const unsigned int packed = quantized[(group / 2) * 32 + lane];
    unsigned int value =
        group % 2 == 0 ? packed & 0x0f : packed >> 4;
    if ((high[lane] & (1u << group)) != 0) {
        value += 16;
    }
    return scale * (float) group_scale * (float) value -
        minimum * (float) group_minimum;
}

extern "C" __global__ void get_rows_q5_K_f32(
        const unsigned char * table,
        const unsigned int * rows,
        float * output,
        unsigned int width,
        unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const unsigned int column = index % width;
    const unsigned int output_row = index / width;
    const unsigned int blocks_per_row = width / 256;
    const unsigned int block_index =
        rows[output_row] * blocks_per_row + column / 256;
    output[index] = dequant_q5_K_value(
        table + block_index * 176,
        column % 256);
}

extern "C" __global__ void mul_mat_q5_K_f32(
        const unsigned char * left,
        const float * right,
        float * output,
        unsigned int inner,
        unsigned int left_rows,
        unsigned int right_rows) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int count = left_rows * right_rows;
    if (index >= count) {
        return;
    }
    const unsigned int left_row = index % left_rows;
    const unsigned int right_row = index / left_rows;
    const unsigned int blocks_per_row = inner / 256;
    const unsigned char * weight_row =
        left + left_row * blocks_per_row * 176;
    const float * input_row = right + right_row * inner;
    float sum = 0.0f;
    for (unsigned int block_index = 0; block_index < blocks_per_row; ++block_index) {
        const unsigned char * block = weight_row + block_index * 176;
        const unsigned int column_base = block_index * 256;
        for (unsigned int column = 0; column < 256; ++column) {
            sum += dequant_q5_K_value(block, column) *
                input_row[column_base + column];
        }
    }
    output[index] = sum;
}

__device__ __constant__ float iq4nl_values[16] = {
    -127.0f, -104.0f, -83.0f, -65.0f,
    -49.0f, -35.0f, -22.0f, -10.0f,
    1.0f, 13.0f, 25.0f, 38.0f,
    53.0f, 69.0f, 89.0f, 113.0f
};

__device__ float dequant_iq4_nl_value(
        const unsigned char * block,
        unsigned int column) {
    const float scale =
        __half2float(*reinterpret_cast<const __half *>(block));
    const unsigned int lane = column % 16;
    const unsigned int packed = block[2 + lane];
    const unsigned int value =
        column < 16 ? packed & 0x0f : packed >> 4;
    return scale * iq4nl_values[value];
}

DEFINE_CLASSIC_QUANT_KERNELS(iq4_nl, 18, dequant_iq4_nl_value)

__device__ float dequant_iq4_xs_value(
        const unsigned char * block,
        unsigned int column) {
    const float scale =
        __half2float(*reinterpret_cast<const __half *>(block));
    const unsigned int high_scales =
        *reinterpret_cast<const unsigned short *>(block + 2);
    const unsigned char * low_scales = block + 4;
    const unsigned char * quantized = block + 8;
    const unsigned int group = column / 32;
    const unsigned int within = column % 32;
    unsigned int packed_scale =
        (low_scales[group / 2] >> (4 * (group % 2))) & 0x0f;
    packed_scale |= ((high_scales >> (2 * group)) & 0x03) << 4;
    const unsigned int packed =
        quantized[group * 16 + within % 16];
    const unsigned int value =
        within < 16 ? packed & 0x0f : packed >> 4;
    return scale * (float) ((int) packed_scale - 32) *
        iq4nl_values[value];
}

extern "C" __global__ void get_rows_iq4_xs_f32(
        const unsigned char * table,
        const unsigned int * rows,
        float * output,
        unsigned int width,
        unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const unsigned int column = index % width;
    const unsigned int output_row = index / width;
    const unsigned int blocks_per_row = width / 256;
    const unsigned int block_index =
        rows[output_row] * blocks_per_row + column / 256;
    output[index] = dequant_iq4_xs_value(
        table + block_index * 136,
        column % 256);
}

extern "C" __global__ void mul_mat_iq4_xs_f32(
        const unsigned char * left,
        const float * right,
        float * output,
        unsigned int inner,
        unsigned int left_rows,
        unsigned int right_rows) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int count = left_rows * right_rows;
    if (index >= count) {
        return;
    }
    const unsigned int left_row = index % left_rows;
    const unsigned int right_row = index / left_rows;
    const unsigned int blocks_per_row = inner / 256;
    const unsigned char * weight_row =
        left + left_row * blocks_per_row * 136;
    const float * input_row = right + right_row * inner;
    float sum = 0.0f;
    for (unsigned int block_index = 0; block_index < blocks_per_row; ++block_index) {
        const unsigned char * block = weight_row + block_index * 136;
        const unsigned int column_base = block_index * 256;
        for (unsigned int column = 0; column < 256; ++column) {
            sum += dequant_iq4_xs_value(block, column) *
                input_row[column_base + column];
        }
    }
    output[index] = sum;
}

__device__ float dequant_q6_K_value(
        const unsigned char * block,
        unsigned int column) {
    const unsigned char * lower = block;
    const unsigned char * high = block + 128;
    const signed char * scales =
        reinterpret_cast<const signed char *>(block + 192);
    const float scale =
        __half2float(*reinterpret_cast<const __half *>(block + 208));
    const unsigned int group = column / 128;
    const unsigned int within = column % 128;
    const unsigned int quarter = within / 32;
    const unsigned int lane = within % 32;
    const unsigned int lower_base = group * 64;
    const unsigned int high_base = group * 32;
    const unsigned int scale_base = group * 8 + lane / 16;
    unsigned int quantized;
    unsigned int scale_offset;
    if (quarter == 0) {
        quantized =
            (lower[lower_base + lane] & 0x0f) |
            ((high[high_base + lane] & 0x03) << 4);
        scale_offset = 0;
    } else if (quarter == 1) {
        quantized =
            (lower[lower_base + lane + 32] & 0x0f) |
            (((high[high_base + lane] >> 2) & 0x03) << 4);
        scale_offset = 2;
    } else if (quarter == 2) {
        quantized =
            (lower[lower_base + lane] >> 4) |
            (((high[high_base + lane] >> 4) & 0x03) << 4);
        scale_offset = 4;
    } else {
        quantized =
            (lower[lower_base + lane + 32] >> 4) |
            (((high[high_base + lane] >> 6) & 0x03) << 4);
        scale_offset = 6;
    }
    return scale *
        (float) scales[scale_base + scale_offset] *
        (float) ((int) quantized - 32);
}

extern "C" __global__ void get_rows_q6_K_f32(
        const unsigned char * table,
        const unsigned int * rows,
        float * output,
        unsigned int width,
        unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const unsigned int column = index % width;
    const unsigned int output_row = index / width;
    const unsigned int blocks_per_row = width / 256;
    const unsigned int block_index =
        rows[output_row] * blocks_per_row + column / 256;
    output[index] = dequant_q6_K_value(
        table + block_index * 210,
        column % 256);
}

extern "C" __global__ void mul_mat_q6_K_f32(
        const unsigned char * left,
        const float * right,
        float * output,
        unsigned int inner,
        unsigned int left_rows,
        unsigned int right_rows) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int count = left_rows * right_rows;
    if (index >= count) {
        return;
    }
    const unsigned int left_row = index % left_rows;
    const unsigned int right_row = index / left_rows;
    const unsigned int blocks_per_row = inner / 256;
    const unsigned char * weight_row =
        left + left_row * blocks_per_row * 210;
    const float * input_row = right + right_row * inner;
    float sum = 0.0f;
    for (unsigned int block_index = 0; block_index < blocks_per_row; ++block_index) {
        const unsigned char * block = weight_row + block_index * 210;
        const unsigned int column_base = block_index * 256;
        for (unsigned int column = 0; column < 256; ++column) {
            sum += dequant_q6_K_value(block, column) *
                input_row[column_base + column];
        }
    }
    output[index] = sum;
}

__device__ float moe_expert_value(
        const void * weights,
        size_t index,
        unsigned int storage) {
    if (storage == GGML_F32) {
        return reinterpret_cast<const float *>(weights)[index];
    }
    const unsigned char * bytes = reinterpret_cast<const unsigned char *>(weights);
#define MOE_DEQUANT_CASE(TYPE, WIDTH, BLOCK_BYTES, DEQUANT) \
    case TYPE: { \
        const unsigned char * block = \
            bytes + (index / WIDTH) * BLOCK_BYTES; \
        return DEQUANT(block, (unsigned int) (index % WIDTH)); \
    }
    switch (storage) {
    MOE_DEQUANT_CASE(GGML_Q4_0, 32, 18, dequant_q4_0_value)
    MOE_DEQUANT_CASE(GGML_Q4_1, 32, 20, dequant_q4_1_value)
    MOE_DEQUANT_CASE(GGML_Q5_0, 32, 22, dequant_q5_0_value)
    MOE_DEQUANT_CASE(GGML_Q5_1, 32, 24, dequant_q5_1_value)
    case GGML_Q8_0: {
        const unsigned char * block = bytes + (index / 32) * 34;
        const float scale = __half2float(*reinterpret_cast<const __half *>(block));
        const signed char value =
            *(reinterpret_cast<const signed char *>(block + 2) + index % 32);
        return scale * (float) value;
    }
    MOE_DEQUANT_CASE(GGML_Q8_1, 32, 36, dequant_q8_1_value)
    MOE_DEQUANT_CASE(GGML_Q2_K, 256, 84, dequant_q2_K_value)
    MOE_DEQUANT_CASE(GGML_Q3_K, 256, 110, dequant_q3_K_value)
    MOE_DEQUANT_CASE(GGML_Q4_K, 256, 144, dequant_q4_K_value)
    MOE_DEQUANT_CASE(GGML_Q5_K, 256, 176, dequant_q5_K_value)
    MOE_DEQUANT_CASE(GGML_Q6_K, 256, 210, dequant_q6_K_value)
    MOE_DEQUANT_CASE(GGML_Q8_K, 256, 292, dequant_q8_K_value)
    MOE_DEQUANT_CASE(GGML_IQ2_XXS, 256, 66, dequant_iq2_xxs_value)
    MOE_DEQUANT_CASE(GGML_IQ2_XS, 256, 74, dequant_iq2_xs_value)
    MOE_DEQUANT_CASE(GGML_IQ3_XXS, 256, 98, dequant_iq3_xxs_value)
    MOE_DEQUANT_CASE(GGML_IQ1_S, 256, 50, dequant_iq1_s_value)
    MOE_DEQUANT_CASE(GGML_IQ4_NL, 32, 18, dequant_iq4_nl_value)
    MOE_DEQUANT_CASE(GGML_IQ3_S, 256, 110, dequant_iq3_s_value)
    MOE_DEQUANT_CASE(GGML_IQ2_S, 256, 82, dequant_iq2_s_value)
    MOE_DEQUANT_CASE(GGML_IQ4_XS, 256, 136, dequant_iq4_xs_value)
    MOE_DEQUANT_CASE(GGML_IQ1_M, 256, 56, dequant_iq1_m_value)
    MOE_DEQUANT_CASE(GGML_TQ1_0, 256, 54, dequant_tq1_0_value)
    MOE_DEQUANT_CASE(GGML_TQ2_0, 256, 66, dequant_tq2_0_value)
    MOE_DEQUANT_CASE(GGML_MXFP4, 32, 17, dequant_mxfp4_value)
    MOE_DEQUANT_CASE(GGML_NVFP4, 64, 36, dequant_nvfp4_value)
    MOE_DEQUANT_CASE(GGML_Q1_0, 128, 18, dequant_q1_0_value)
    MOE_DEQUANT_CASE(GGML_Q2_0, 64, 18, dequant_q2_0_value)
    default:
        return 0.0f;
    }
#undef MOE_DEQUANT_CASE
}

// ---- Wan causal 3-D VAE decode (streamed chunk-major, latentvideo) ----

// vae_causal_conv3d_f32: stride-1 same-padded causal 3-D convolution over one
// streamed chunk with a temporal prefix cache of cache_t frames. 64x64
// position-by-channel tiles with 4x4 register blocking per thread (8 shared
// reads per 16 FMA) and per-block position decomposition precomputed once;
// fp32 FMA accumulation in the fixed ascending kernel-value order, fused
// mandatory bias. Output dims equal input dims (stride 1 everywhere in the
// decoder). Covers 3x3x3, temporal 3x1x1, and 1x1x1 pointwise convs.
#define VAE_CONV_TILE 64
#define VAE_CONV_K 16
#define VAE_CONV_SPAN 16
extern "C" __global__ void vae_causal_conv3d_f32(
        float * out,
        const float * x,
        const float * cache,
        const float * weight,
        const float * bias,
        int c_in,
        int c_out,
        int in_t,
        int in_h,
        int in_w,
        int kt_n,
        int kh_n,
        int kw_n,
        int pad_t,
        int pad_h,
        int pad_w,
        int cache_t) {
    __shared__ float activations[VAE_CONV_K][VAE_CONV_TILE];
    __shared__ float weights[VAE_CONV_K][VAE_CONV_TILE];
    __shared__ int pos_t[VAE_CONV_TILE];
    __shared__ int pos_h[VAE_CONV_TILE];
    __shared__ int pos_w[VAE_CONV_TILE];
    const int lane = threadIdx.x;
    const int tx = lane % VAE_CONV_SPAN;
    const int ty = lane / VAE_CONV_SPAN;
    const int positions = in_t * in_h * in_w;
    const int channel_tiles = (c_out + VAE_CONV_TILE - 1) / VAE_CONV_TILE;
    const int position_tile = blockIdx.x / channel_tiles;
    const int channel_tile = blockIdx.x - position_tile * channel_tiles;
    const int kernel_values = c_in * kt_n * kh_n * kw_n;
    const int kernel_plane = kt_n * kh_n * kw_n;
    if (lane < VAE_CONV_TILE) {
        const int position = position_tile * VAE_CONV_TILE + lane;
        const int clamped = position < positions ? position : positions - 1;
        const int ot = clamped / (in_h * in_w);
        const int prem = clamped - ot * in_h * in_w;
        pos_t[lane] = ot;
        pos_h[lane] = prem / in_w;
        pos_w[lane] = prem - (prem / in_w) * in_w;
    }
    float acc[4][4];
    for (int i = 0; i < 4; ++i) {
        const int position = position_tile * VAE_CONV_TILE + ty + i * VAE_CONV_SPAN;
        for (int j = 0; j < 4; ++j) {
            const int co = channel_tile * VAE_CONV_TILE + tx + j * VAE_CONV_SPAN;
            acc[i][j] = (position < positions && co < c_out) ? bias[co] : 0.0f;
        }
    }
    __syncthreads();
    for (int base = 0; base < kernel_values; base += VAE_CONV_K) {
        for (int flat = lane; flat < VAE_CONV_K * VAE_CONV_TILE; flat += blockDim.x) {
            const int k = flat / VAE_CONV_TILE;
            const int m = flat - k * VAE_CONV_TILE;
            const int input_kernel = base + k;
            const int position = position_tile * VAE_CONV_TILE + m;
            float value = 0.0f;
            if (position < positions && input_kernel < kernel_values) {
                const int ci = input_kernel / kernel_plane;
                int z = input_kernel - ci * kernel_plane;
                const int kt = z / (kh_n * kw_n);
                z -= kt * kh_n * kw_n;
                const int kh = z / kw_n;
                const int kw = z - kh * kw_n;
                const int padded_t = pos_t[m] + kt - (2 * pad_t - cache_t);
                const int ih = pos_h[m] + kh - pad_h;
                const int iw = pos_w[m] + kw - pad_w;
                if (padded_t >= 0 && padded_t < cache_t + in_t &&
                        ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                    if (padded_t < cache_t) {
                        value = cache[((ci * cache_t + padded_t) * in_h + ih) * in_w + iw];
                    } else {
                        value = x[((ci * in_t + padded_t - cache_t) * in_h + ih) * in_w + iw];
                    }
                }
            }
            activations[k][m] = value;
            const int load_co = channel_tile * VAE_CONV_TILE + m;
            weights[k][m] = (load_co < c_out && input_kernel < kernel_values)
                ? weight[load_co * kernel_values + input_kernel]
                : 0.0f;
        }
        __syncthreads();
        for (int k = 0; k < VAE_CONV_K; ++k) {
            float a[4];
            float b[4];
            for (int i = 0; i < 4; ++i) {
                a[i] = activations[k][ty + i * VAE_CONV_SPAN];
                b[i] = weights[k][tx + i * VAE_CONV_SPAN];
            }
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
                }
            }
        }
        __syncthreads();
    }
    for (int i = 0; i < 4; ++i) {
        const int position = position_tile * VAE_CONV_TILE + ty + i * VAE_CONV_SPAN;
        if (position >= positions) continue;
        for (int j = 0; j < 4; ++j) {
            const int co = channel_tile * VAE_CONV_TILE + tx + j * VAE_CONV_SPAN;
            if (co < c_out) {
                out[co * positions + position] = acc[i][j];
            }
        }
    }
}

// vae_channel_rms_norm_f32: RMS over channels at each position, sqrt(C)
// scale, 1e-12 zero guard, f64 accumulation (reference engine convention),
// optional fused SiLU applied after the f32 rounding (host rounding order).
// One block per position; shared = blockDim.x doubles.
extern "C" __global__ void vae_channel_rms_norm_f32(
        float * out,
        const float * x,
        const float * gamma,
        int channels,
        int plane,
        int apply_silu) {
    extern __shared__ double vae_rms_reduce[];
    const int pos = blockIdx.x;
    const int tid = threadIdx.x;
    double sum = 0.0;
    for (int ch = tid; ch < channels; ch += blockDim.x) {
        const double v = (double) x[ch * plane + pos];
        sum += v * v;
    }
    vae_rms_reduce[tid] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride; stride >>= 1) {
        if (tid < stride) {
            vae_rms_reduce[tid] += vae_rms_reduce[tid + stride];
        }
        __syncthreads();
    }
    double norm = sqrt(vae_rms_reduce[0]);
    if (norm < 1.0e-12) {
        norm = 1.0e-12;
    }
    const double scale = sqrt((double) channels) / norm;
    for (int ch = tid; ch < channels; ch += blockDim.x) {
        const int idx = ch * plane + pos;
        float value = (float) ((double) x[idx] * scale * (double) gamma[ch]);
        if (apply_silu) {
            const double s = (double) value;
            value = (float) (s / (1.0 + exp(-s)));
        }
        out[idx] = value;
    }
}

// vae_spatial_attention_f32: QKV-packed per-frame spatial self-attention.
// Layout qkv[section*channels*groups*frame + (ch*groups+g)*frame + pos],
// sections q=0,k=1,v=2; groups is the frame count, frame the spatial extent.
// One block per (group, query); shared = frame floats (8-aligned) + one
// double per thread. Residual add stays with the caller.
extern "C" __global__ void vae_spatial_attention_f32(
        const float * qkv,
        float * out,
        int channels,
        int groups,
        int frame,
        float scale) {
    extern __shared__ unsigned char vae_attention_scratch[];
    float * scores = (float *) vae_attention_scratch;
    const int score_bytes = ((frame * (int) sizeof(float) + 7) / 8) * 8;
    double * reduce = (double *) (vae_attention_scratch + score_bytes);
    const int qi = blockIdx.x % frame;
    const int g = blockIdx.x / frame;
    const int tid = threadIdx.x;
    const int koff = channels * groups * frame;
    const int voff = 2 * channels * groups * frame;
    for (int kj = tid; kj < frame; kj += blockDim.x) {
        float dot = 0.0f;
        for (int ch = 0; ch < channels; ++ch) {
            dot = fmaf(qkv[(ch * groups + g) * frame + qi],
                       qkv[koff + (ch * groups + g) * frame + kj], dot);
        }
        scores[kj] = dot * scale;
    }
    __syncthreads();
    float local_max = -3.402823466e38f;
    for (int j = tid; j < frame; j += blockDim.x) {
        local_max = fmaxf(local_max, scores[j]);
    }
    reduce[tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride; stride >>= 1) {
        if (tid < stride) {
            reduce[tid] = fmax(reduce[tid], reduce[tid + stride]);
        }
        __syncthreads();
    }
    const float mx = (float) reduce[0];
    __syncthreads();
    double sum = 0.0;
    for (int j = tid; j < frame; j += blockDim.x) {
        const float e = expf(scores[j] - mx);
        scores[j] = e;
        sum += e;
    }
    reduce[tid] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride; stride >>= 1) {
        if (tid < stride) {
            reduce[tid] += reduce[tid + stride];
        }
        __syncthreads();
    }
    const float inv = (float) (1.0 / reduce[0]);
    for (int ch = tid; ch < channels; ch += blockDim.x) {
        float acc = 0.0f;
        for (int j = 0; j < frame; ++j) {
            acc = fmaf(scores[j] * inv, qkv[voff + (ch * groups + g) * frame + j], acc);
        }
        out[(ch * groups + g) * frame + qi] = acc;
    }
}

// vae_upsample2d_f32: nearest-neighbor 2x spatial upsample fused with a
// same-padded 3x3 Conv2d per frame (Wan resample stack), fp32 FMA, fused
// mandatory bias. One thread per output element.
extern "C" __global__ void vae_upsample2d_f32(
        float * out,
        const float * x,
        const float * weight,
        const float * bias,
        int c_in,
        int c_out,
        int frames,
        int height,
        int width,
        int out_h,
        int out_w,
        unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const int plane = frames * out_h * out_w;
    const int co = (int) index / plane;
    int rem = (int) index - co * plane;
    const int ti = rem / (out_h * out_w);
    rem -= ti * out_h * out_w;
    const int oh = rem / out_w;
    const int ow = rem - oh * out_w;
    float acc = bias[co];
    for (int ci = 0; ci < c_in; ++ci) {
        for (int kh = 0; kh < 3; ++kh) {
            const int uh = oh + kh - 1;
            if (uh < 0 || uh >= out_h) {
                continue;
            }
            const int ih = uh / 2;
            for (int kw = 0; kw < 3; ++kw) {
                const int uw = ow + kw - 1;
                if (uw < 0 || uw >= out_w) {
                    continue;
                }
                const int iw = uw / 2;
                acc = fmaf(x[((ci * frames + ti) * height + ih) * width + iw],
                           weight[((co * c_in + ci) * 3 + kh) * 3 + kw], acc);
            }
        }
    }
    out[index] = acc;
}

// vae_time_interleave_f32: temporal upsample reshape - the time conv's
// doubled channel output [2c][t][s] interleaves to [c][2t][s]; even output
// frames from the first channel half.
extern "C" __global__ void vae_time_interleave_f32(
        float * out,
        const float * convolved,
        int channels,
        int frames,
        int spatial,
        unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const int out_frames = 2 * frames;
    const int ch = (int) index / (out_frames * spatial);
    int rem = (int) index - ch * out_frames * spatial;
    const int ti = rem / spatial;
    const int pos = rem - ti * spatial;
    out[index] = convolved[(((ti & 1) * channels + ch) * frames + ti / 2) * spatial + pos];
}

// vae_temporal_cache_update_f32: reference temporal_cache_update semantics.
// mode 0: keep the last two input frames, joining the prior cache's final
// frame when the chunk is a single frame; mode 1: zero prefix then the chunk
// (temporal upsample first-cache convention after the skipped chunk).
extern "C" __global__ void vae_temporal_cache_update_f32(
        float * out,
        const float * current,
        const float * prior,
        int channels,
        int current_frames,
        int prior_frames,
        int out_frames,
        int spatial,
        int mode,
        unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const int ch = (int) index / (out_frames * spatial);
    int rem = (int) index - ch * out_frames * spatial;
    const int ti = rem / spatial;
    const int pos = rem - ti * spatial;
    if (mode) {
        out[index] = ti == 0 ? 0.0f : current[(ch * current_frames + ti - 1) * spatial + pos];
        return;
    }
    if (current_frames >= 2) {
        const int src = current_frames - out_frames + ti;
        out[index] = current[(ch * current_frames + src) * spatial + pos];
        return;
    }
    if (prior_frames > 0 && ti == 0) {
        out[index] = prior[(ch * prior_frames + prior_frames - 1) * spatial + pos];
        return;
    }
    const int src = ti - (prior_frames > 0 ? 1 : 0);
    out[index] = current[(ch * current_frames + src) * spatial + pos];
}

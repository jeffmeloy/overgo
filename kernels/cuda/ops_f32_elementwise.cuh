// ops_f32_elementwise.cuh: conversions between float formats, elementwise and broadcast arithmetic, scales, copies and SiLU, in ops_f32.cu's order. ops_f32.cu
// includes it once; it compiles only there.

extern "C" __global__ void f32_to_bf16(
        const float * input,
        unsigned short * output,
        unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    output[index] = bf16_pack(input[index]);
}

// F32 -> F16 round-to-nearest-even pack (tensor-core prefill stages the F32
// activation as F16 next to a resident F16 weight; cuBLAS multiplies both in
// F16 and accumulates in F32). Values beyond the F16 range saturate to
// infinity exactly as __float2half_rn defines.
extern "C" __global__ void f32_to_f16(
        const float * input,
        unsigned short * output,
        unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    output[index] = __half_as_ushort(__float2half_rn(input[index]));
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

// fp8_to_f16: the f16 twin of fp8_to_f32 for the tensor-core prefill of
// native fp8 weights: e4m3 decoded and scaled per row into half
// precision, the operand cuBLAS multiplies with F32 accumulation.
extern "C" __global__ void fp8_to_f16(
        const unsigned char * input,
        const float * scale,
        __half * output,
        unsigned int inner,
        unsigned int rows) {
    const unsigned int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    const float row_scale = scale[row];
    const unsigned char * weight_row = input + (size_t) row * inner;
    __half * output_row = output + (size_t) row * inner;
    for (unsigned int column = threadIdx.x; column < inner; column += blockDim.x) {
        output_row[column] = __float2half(fp8_e4m3_decode((unsigned int) weight_row[column]) * row_scale);
    }
}

// FP8_E4M3_MAX: the largest finite OCP E4M3FN magnitude, where each
// activation column's absolute maximum lands.
constexpr float FP8_E4M3_MAX = 448.0f;
constexpr unsigned int FP8_QUAD_WIDTH = 4;

// quantize_fp8_columns_f32: the activation side of the FP8 tensor-core
// prefill. One block per token column of `inner` F32 values (inner a
// multiple of four): the column's absolute maximum sets
// scale = amax / 448, stored per column, and every value is written as
// e4m3(value / scale), round to nearest even and saturating -- the host
// dtype.Float32ToF8E4M3 contract. Both divisions are the correctly
// rounded __fdiv_rn so the bytes match the host encoder under
// -use_fast_math. An all-zero column stores scale 0 and zero bytes.
extern "C" __global__ void quantize_fp8_columns_f32(
        const float * input,
        unsigned char * output,
        float * scale,
        unsigned int inner,
        unsigned int columns) {
    const unsigned int column = blockIdx.x;
    if (column >= columns) {
        return;
    }
    const float4 * source = reinterpret_cast<const float4 *>(input + (size_t) column * inner);
    const unsigned int quads = inner / FP8_QUAD_WIDTH;
    float amax = 0.0f;
    for (unsigned int quad = threadIdx.x; quad < quads; quad += blockDim.x) {
        const float4 value = source[quad];
        amax = fmaxf(amax, fmaxf(fmaxf(fabsf(value.x), fabsf(value.y)), fmaxf(fabsf(value.z), fabsf(value.w))));
    }
    for (unsigned int offset = CUDA_WARP_WIDTH / 2; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, offset));
    }
    __shared__ float warp_amax[CUDA_WARP_WIDTH];
    if (threadIdx.x % CUDA_WARP_WIDTH == 0) {
        warp_amax[threadIdx.x / CUDA_WARP_WIDTH] = amax;
    }
    __syncthreads();
    const unsigned int warps = (blockDim.x + CUDA_WARP_WIDTH - 1) / CUDA_WARP_WIDTH;
    amax = 0.0f;
    for (unsigned int warp = 0; warp < warps; ++warp) {
        amax = fmaxf(amax, warp_amax[warp]);
    }
    const float column_scale = __fdiv_rn(amax, FP8_E4M3_MAX);
    if (threadIdx.x == 0) {
        scale[column] = column_scale;
    }
    unsigned int * destination = reinterpret_cast<unsigned int *>(output + (size_t) column * inner);
    for (unsigned int quad = threadIdx.x; quad < quads; quad += blockDim.x) {
        float4 value = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (column_scale > 0.0f) {
            const float4 raw = source[quad];
            value = make_float4(
                __fdiv_rn(raw.x, column_scale), __fdiv_rn(raw.y, column_scale),
                __fdiv_rn(raw.z, column_scale), __fdiv_rn(raw.w, column_scale));
        }
        const unsigned int low = __nv_cvt_float2_to_fp8x2(make_float2(value.x, value.y), __NV_SATFINITE, __NV_E4M3);
        const unsigned int high = __nv_cvt_float2_to_fp8x2(make_float2(value.z, value.w), __NV_SATFINITE, __NV_E4M3);
        destination[quad] = low | (high << 16);
    }
}

// scale_fp8_product_f32: the epilogue of the FP8 tensor-core prefill.
// The e4m3 product lands column-major [columns][rows] in F32; entry
// (row, column) takes its weight row's scale and its activation
// column's scale. rows is a multiple of four, so each thread scales
// four rows of one column; quads = rows * columns / 4.
extern "C" __global__ void scale_fp8_product_f32(
        float * output,
        const float * row_scale,
        const float * column_scale,
        unsigned int rows,
        unsigned int quads) {
    const unsigned int quad = blockIdx.x * blockDim.x + threadIdx.x;
    if (quad >= quads) {
        return;
    }
    const unsigned int element = quad * FP8_QUAD_WIDTH;
    const unsigned int column = element / rows;
    const unsigned int row = element - column * rows;
    const float factor = column_scale[column];
    const float4 weight = *reinterpret_cast<const float4 *>(row_scale + row);
    float4 value = reinterpret_cast<float4 *>(output)[quad];
    value.x *= weight.x * factor;
    value.y *= weight.y * factor;
    value.z *= weight.z * factor;
    value.w *= weight.w * factor;
    reinterpret_cast<float4 *>(output)[quad] = value;
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

// bf16_rounded is value's nearest BF16, ties to even, as bf16_round_f32 stores it; infinities and NaNs pass unchanged.
__device__ __forceinline__ float bf16_rounded(float value) {
    unsigned int raw = __float_as_uint(value);
    if ((raw & 0x7f800000u) != 0x7f800000u) {
        raw += 0x7fffu + ((raw >> 16) & 1u);
        raw &= 0xffff0000u;
    }
    return __uint_as_float(raw);
}

// INT8_CONVROT_GROUP is the width of comfy_kitchen's ConvRot rotation, rotated by H4's Kronecker power in four radix-4
// stages (INT8_CONVROT_RADIX), each halving (INT8_CONVROT_HALF). INT8_CONVROT_MAX_INNER is the widest row the ConvRot
// kernels hold in shared memory, MOSS's widest projection input (its feed-forward's 8,192). INT8_INVERSE_127 is 1/127
// rounded to F32, which a row's absolute maximum is multiplied by for its scale, at least INT8_SCALE_FLOOR. A
// stochastic rounding's noise is a 32-bit hash's top bits (below INT8_NOISE_SHIFT) times INT8_NOISE_UNIT, 2^-24.
constexpr unsigned int INT8_CONVROT_GROUP = 256;
constexpr unsigned int INT8_CONVROT_RADIX = 4;
constexpr unsigned int INT8_CONVROT_MAX_INNER = 8192;
constexpr float INT8_CONVROT_HALF = 0.5f;
constexpr float INT8_INVERSE_127 = 1.0f / 127.0f;
constexpr float INT8_SCALE_FLOOR = 1e-30f;
constexpr float INT8_LOWEST = -128.0f;
constexpr float INT8_HIGHEST = 127.0f;
constexpr unsigned int INT8_NOISE_SHIFT = 8;
constexpr float INT8_NOISE_UNIT = 5.9604644775390625e-08f;

// int8_convrot_rotate rotates each INT8_CONVROT_GROUP-wide group of the inner values a block holds in shared memory by
// H4's Kronecker power, H4 = [[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]], as comfy_kitchen's convrot64 kernels do:
// one radix-4 stage a base-4 digit of a value's index in its group, from the least significant, each output half the
// left-to-right sum of its four signed inputs, so the four stages divide by 16. The block's threads share each stage's
// butterflies and meet after it.
__device__ __forceinline__ void int8_convrot_rotate(float * values, unsigned int inner) {
    const unsigned int per_group = INT8_CONVROT_GROUP / INT8_CONVROT_RADIX;
    const unsigned int butterflies = inner / INT8_CONVROT_RADIX;
    for (unsigned int stride = 1; stride < INT8_CONVROT_GROUP; stride *= INT8_CONVROT_RADIX) {
        for (unsigned int butterfly = threadIdx.x; butterfly < butterflies; butterfly += blockDim.x) {
            const unsigned int group = butterfly / per_group;
            const unsigned int within = butterfly - group * per_group;
            const unsigned int high = within / stride;
            const unsigned int base = group * INT8_CONVROT_GROUP + high * stride * INT8_CONVROT_RADIX + (within - high * stride);
            const float x0 = values[base];
            const float x1 = values[base + stride];
            const float x2 = values[base + 2 * stride];
            const float x3 = values[base + 3 * stride];
            values[base] = __fmul_rn(INT8_CONVROT_HALF, __fsub_rn(__fadd_rn(__fadd_rn(x0, x1), x2), x3));
            values[base + stride] = __fmul_rn(INT8_CONVROT_HALF, __fadd_rn(__fsub_rn(__fadd_rn(x0, x1), x2), x3));
            values[base + 2 * stride] = __fmul_rn(INT8_CONVROT_HALF, __fadd_rn(__fadd_rn(__fsub_rn(x0, x1), x2), x3));
            values[base + 3 * stride] = __fmul_rn(INT8_CONVROT_HALF, __fadd_rn(__fadd_rn(__fsub_rn(x1, x0), x2), x3));
        }
        __syncthreads();
    }
}

// int8_noise is the noise comfy_kitchen's stochastic rounding adds at key, a value's flat index plus the seed: the
// key's halves exclusive-ored, hashed by PCG's output permutation, its top 24 bits times 2^-24, rounded to BF16.
__device__ __forceinline__ float int8_noise(unsigned long long key) {
    const unsigned int folded = (unsigned int) key ^ (unsigned int) (key >> 32);
    const unsigned int state = folded * 747796405u + 2891336453u;
    const unsigned int word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return bf16_rounded(__fmul_rn(__uint2float_rn(((word >> 22u) ^ word) >> INT8_NOISE_SHIFT), INT8_NOISE_UNIT));
}

// int8_convrot_quantize quantizes a row of inner values a block holds in shared memory, rotated (int8_convrot_rotate),
// to int8 with one scale, as comfy_kitchen's convrot64_quantize_row does from a BF16 row: the scale is the row's
// absolute maximum times 1/127, at least 1e-30, and each value's quotient is the value and the scale, each rounded to
// BF16, divided and rounded to BF16. Nearest rounding rounds the quotient to an integer, ties to even; stochastic
// rounding (STOCHASTIC) adds the noise at the value's flat index, first plus its index in the row, plus seed, as a BF16
// addition rounds, and rounds down. Both saturate to int8. Every operation is a correctly rounded intrinsic, so
// -use_fast_math changes no bit.
template <bool STOCHASTIC>
__device__ __forceinline__ void int8_convrot_quantize(
        const float * values,
        float * warp_amax,
        signed char * destination,
        float * scale,
        unsigned int inner,
        unsigned long long first,
        unsigned long long seed) {
    float amax = 0.0f;
    for (unsigned int index = threadIdx.x; index < inner; index += blockDim.x) {
        amax = fmaxf(amax, fabsf(values[index]));
    }
    for (unsigned int offset = CUDA_WARP_WIDTH / 2; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, offset));
    }
    if (threadIdx.x % CUDA_WARP_WIDTH == 0) {
        warp_amax[threadIdx.x / CUDA_WARP_WIDTH] = amax;
    }
    __syncthreads();
    const unsigned int warps = (blockDim.x + CUDA_WARP_WIDTH - 1) / CUDA_WARP_WIDTH;
    amax = 0.0f;
    for (unsigned int warp = 0; warp < warps; ++warp) {
        amax = fmaxf(amax, warp_amax[warp]);
    }
    const float row_scale = fmaxf(__fmul_rn(amax, INT8_INVERSE_127), INT8_SCALE_FLOOR);
    if (threadIdx.x == 0) {
        *scale = row_scale;
    }
    const float divisor = bf16_rounded(row_scale);
    for (unsigned int index = threadIdx.x; index < inner; index += blockDim.x) {
        const float quotient = bf16_rounded(__fdiv_rn(bf16_rounded(values[index]), divisor));
        const float rounded = STOCHASTIC ? floorf(bf16_rounded(__fadd_rn(quotient, int8_noise(first + index + seed)))) : rintf(quotient);
        destination[index] = (signed char) __float2int_rn(fminf(fmaxf(rounded, INT8_LOWEST), INT8_HIGHEST));
    }
}

// quantize_int8_convrot_columns_f32: the activation side of the int8 ConvRot tensor-core product, as comfy_kitchen's
// CUDA int8_linear forms it (moss-int8, its product matched bit for bit): one block per token column of `inner` F32
// values, inner a multiple of INT8_CONVROT_GROUP and at most INT8_CONVROT_MAX_INNER, rotated (int8_convrot_rotate) and
// quantized to the nearest (int8_convrot_quantize) with its scale stored per column.
extern "C" __global__ void quantize_int8_convrot_columns_f32(
        const float * input,
        signed char * output,
        float * scale,
        unsigned int inner,
        unsigned int columns) {
    __shared__ float values[INT8_CONVROT_MAX_INNER];
    __shared__ float warp_amax[CUDA_WARP_WIDTH];
    const unsigned int column = blockIdx.x;
    if (column >= columns || inner > INT8_CONVROT_MAX_INNER || inner % INT8_CONVROT_GROUP != 0) {
        return;
    }
    const float * source = input + (size_t) column * inner;
    for (unsigned int index = threadIdx.x; index < inner; index += blockDim.x) {
        values[index] = source[index];
    }
    __syncthreads();
    int8_convrot_rotate(values, inner);
    int8_convrot_quantize<false>(values, warp_amax, output + (size_t) column * inner, scale + column, inner, 0, 0);
}

// quantize_int8_convrot_rows_stochastic_f32: an int8 ConvRot weight made of `rows` rows of `inner` F32 values, each a
// BF16 value, as comfy_kitchen's quantize_int8_convrot_weight makes one with a stochastic rounding seeded by seed (the
// requantize ComfyUI runs after merging a LoRA into the weight): one block per row, rotated (int8_convrot_rotate) and
// quantized stochastically (int8_convrot_quantize) at the flat indexes of the weight whose row first_row the input's
// first row is, its int8 rows written to output and its scales to scale.
extern "C" __global__ void quantize_int8_convrot_rows_stochastic_f32(
        const float * input,
        signed char * output,
        float * scale,
        unsigned int inner,
        unsigned int rows,
        unsigned long long first_row,
        unsigned long long seed) {
    __shared__ float values[INT8_CONVROT_MAX_INNER];
    __shared__ float warp_amax[CUDA_WARP_WIDTH];
    const unsigned int row = blockIdx.x;
    if (row >= rows || inner > INT8_CONVROT_MAX_INNER || inner % INT8_CONVROT_GROUP != 0) {
        return;
    }
    const float * source = input + (size_t) row * inner;
    for (unsigned int index = threadIdx.x; index < inner; index += blockDim.x) {
        values[index] = source[index];
    }
    __syncthreads();
    int8_convrot_rotate(values, inner);
    int8_convrot_quantize<true>(values, warp_amax, output + (size_t) row * inner, scale + row, inner, (first_row + row) * inner, seed);
}

// dequantize_int8_convrot_rows_f32: an int8 ConvRot weight's `rows` rows of `inner` int8 values and their F32 scales
// decoded to BF16 values in F32 as comfy_kitchen's CUDA dequantize_int8_convrot_groups64_kernel decodes them: one block
// per row, each value its int8 times its row's scale in F32, rotated (int8_convrot_rotate), rounded to BF16.
extern "C" __global__ void dequantize_int8_convrot_rows_f32(
        const signed char * input,
        const float * scale,
        float * output,
        unsigned int inner,
        unsigned int rows) {
    __shared__ float values[INT8_CONVROT_MAX_INNER];
    const unsigned int row = blockIdx.x;
    if (row >= rows || inner > INT8_CONVROT_MAX_INNER || inner % INT8_CONVROT_GROUP != 0) {
        return;
    }
    const signed char * source = input + (size_t) row * inner;
    const float row_scale = scale[row];
    for (unsigned int index = threadIdx.x; index < inner; index += blockDim.x) {
        values[index] = __fmul_rn(__int2float_rn(source[index]), row_scale);
    }
    __syncthreads();
    int8_convrot_rotate(values, inner);
    float * destination = output + (size_t) row * inner;
    for (unsigned int index = threadIdx.x; index < inner; index += blockDim.x) {
        destination[index] = bf16_rounded(values[index]);
    }
}

// scale_int8_product_bf16_f32: the epilogue of the int8 ConvRot tensor-core product, as comfy_kitchen's int8_linear
// scales it: cuBLASLt's exact int32 product lands column-major [columns][rows] in the F32 output it is written back
// over; entry (row, column) is its product times its activation column's scale and then its weight row's, each product
// rounded to F32, rounded to BF16. rows is a multiple of four, so each thread scales four rows of one column;
// quads = rows * columns / 4.
extern "C" __global__ void scale_int8_product_bf16_f32(
        float * output,
        const float * row_scale,
        const float * column_scale,
        unsigned int rows,
        unsigned int quads) {
    const unsigned int quad = blockIdx.x * blockDim.x + threadIdx.x;
    if (quad >= quads) {
        return;
    }
    const unsigned int element = quad * FP8_QUAD_WIDTH;
    const unsigned int column = element / rows;
    const unsigned int row = element - column * rows;
    const float factor = column_scale[column];
    const float4 weight = *reinterpret_cast<const float4 *>(row_scale + row);
    const int4 product = reinterpret_cast<const int4 *>(output)[quad];
    reinterpret_cast<float4 *>(output)[quad] = make_float4(
        bf16_rounded(__fmul_rn(__fmul_rn(__int2float_rn(product.x), factor), weight.x)),
        bf16_rounded(__fmul_rn(__fmul_rn(__int2float_rn(product.y), factor), weight.y)),
        bf16_rounded(__fmul_rn(__fmul_rn(__int2float_rn(product.z), factor), weight.z)),
        bf16_rounded(__fmul_rn(__fmul_rn(__int2float_rn(product.w), factor), weight.w)));
}

// scale_int8_product_gated_add_bf16_f32: the epilogue of an int8 ConvRot product whose entries a gated residual sum
// reads alone (gated_add_bf16_round_f32): each entry scaled as scale_int8_product_bf16_f32 scales it, then the residual
// plus the entry times its row's gate rounded to BF16, rounded to BF16, written over the product. rows is a multiple of
// four, so each thread sums four rows of one column; quads = rows * columns / 4.
extern "C" __global__ void scale_int8_product_gated_add_bf16_f32(
        float * output,
        const float * row_scale,
        const float * column_scale,
        const float * residual,
        const float * gate,
        unsigned int rows,
        unsigned int quads) {
    const unsigned int quad = blockIdx.x * blockDim.x + threadIdx.x;
    if (quad >= quads) {
        return;
    }
    const unsigned int element = quad * FP8_QUAD_WIDTH;
    const unsigned int column = element / rows;
    const unsigned int row = element - column * rows;
    const float factor = column_scale[column];
    const float4 weight = *reinterpret_cast<const float4 *>(row_scale + row);
    const float4 gates = *reinterpret_cast<const float4 *>(gate + row);
    const float4 stream = reinterpret_cast<const float4 *>(residual)[quad];
    const int4 product = reinterpret_cast<const int4 *>(output)[quad];
    reinterpret_cast<float4 *>(output)[quad] = make_float4(
        bf16_rounded(stream.x + bf16_rounded(bf16_rounded(__fmul_rn(__fmul_rn(__int2float_rn(product.x), factor), weight.x)) * gates.x)),
        bf16_rounded(stream.y + bf16_rounded(bf16_rounded(__fmul_rn(__fmul_rn(__int2float_rn(product.y), factor), weight.y)) * gates.y)),
        bf16_rounded(stream.z + bf16_rounded(bf16_rounded(__fmul_rn(__fmul_rn(__int2float_rn(product.z), factor), weight.z)) * gates.z)),
        bf16_rounded(stream.w + bf16_rounded(bf16_rounded(__fmul_rn(__fmul_rn(__int2float_rn(product.w), factor), weight.w)) * gates.w)));
}

// elementwise_store is what an elementwise kernel stores of value: value, or, for the <name>_bf16_round entry a BF16
// rounding of its output folds into, its nearest BF16.
template <bool ROUND>
__device__ __forceinline__ float elementwise_store(float value) {
    return ROUND ? bf16_rounded(value) : value;
}

template <bool ROUND>
__device__ __forceinline__ void add_elementwise(
        const float * input_a,
        const float * input_b,
        float * output,
        unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = elementwise_store<ROUND>(input_a[index] + input_b[index]);
    }
}

extern "C" __global__ void add_f32(
        const float * input_a,
        const float * input_b,
        float * output,
        unsigned int count) {
    add_elementwise<false>(input_a, input_b, output, count);
}

extern "C" __global__ void add_bf16_round_f32(
        const float * input_a,
        const float * input_b,
        float * output,
        unsigned int count) {
    add_elementwise<true>(input_a, input_b, output, count);
}

// gated_add_bf16_round_f32: a gated residual sum as MOSS's denoiser adds a block's projection to its stream (latentimage
// rotaryGraph.joint): each value the residual's plus the update's times its channel's gate rounded to BF16, rounded to
// BF16, over rows of width values and a gate of width, reading each value once.
extern "C" __global__ void gated_add_bf16_round_f32(
        const float * residual,
        const float * update,
        const float * gate,
        float * output,
        unsigned int width,
        unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = bf16_rounded(residual[index] + bf16_rounded(update[index] * gate[index % width]));
    }
}

template <bool ROUND>
__device__ __forceinline__ void multiply_elementwise(
        const float * input_a,
        const float * input_b,
        float * output,
        unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = elementwise_store<ROUND>(input_a[index] * input_b[index]);
    }
}

extern "C" __global__ void multiply_f32(
        const float * input_a,
        const float * input_b,
        float * output,
        unsigned int count) {
    multiply_elementwise<false>(input_a, input_b, output, count);
}

extern "C" __global__ void multiply_bf16_round_f32(
        const float * input_a,
        const float * input_b,
        float * output,
        unsigned int count) {
    multiply_elementwise<true>(input_a, input_b, output, count);
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

extern "C" __global__ void signed_sqrt_f32(
    const float* input, float* output, float minimum, unsigned int count) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        const float x = input[i];
        output[i] = isnan(x) ? x : x == 0.0f ? 0.0f : copysignf(sqrtf(fmaxf(fabsf(x), minimum)), x);
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

template <bool ROUND>
__device__ __forceinline__ void broadcast_add_elementwise(
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
        output[index] = elementwise_store<ROUND>(input_a[index_a] + input_b[index_b]);
    }
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
    broadcast_add_elementwise<false>(input_a, input_b, output, count, a_0, a_1, a_2, a_3, b_0, b_1, b_2, b_3, output_0, output_1, output_2);
}

extern "C" __global__ void broadcast_add_bf16_round_f32(
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
    broadcast_add_elementwise<true>(input_a, input_b, output, count, a_0, a_1, a_2, a_3, b_0, b_1, b_2, b_3, output_0, output_1, output_2);
}

template <bool ROUND>
__device__ __forceinline__ void broadcast_multiply_elementwise(
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
        output[index] = elementwise_store<ROUND>(input_a[index_a] * input_b[index_b]);
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
    broadcast_multiply_elementwise<false>(input_a, input_b, output, count, a_0, a_1, a_2, a_3, b_0, b_1, b_2, b_3, output_0, output_1, output_2);
}

extern "C" __global__ void broadcast_multiply_bf16_round_f32(
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
    broadcast_multiply_elementwise<true>(input_a, input_b, output, count, a_0, a_1, a_2, a_3, b_0, b_1, b_2, b_3, output_0, output_1, output_2);
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

template <bool ROUND>
__device__ __forceinline__ void scale_elementwise(
        const float * input,
        float * output,
        float scale,
        unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = elementwise_store<ROUND>(input[index] * scale);
    }
}

extern "C" __global__ void scale_f32(
        const float * input,
        float * output,
        float scale,
        unsigned int count) {
    scale_elementwise<false>(input, output, scale, count);
}

extern "C" __global__ void scale_bf16_round_f32(
        const float * input,
        float * output,
        float scale,
        unsigned int count) {
    scale_elementwise<true>(input, output, scale, count);
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

// copy_token_offset_bf16_cache_f32: copy_token_offset_f32 into a BF16 cache, each value rounded to the nearest BF16.
extern "C" __global__ void copy_token_offset_bf16_cache_f32(
        const float * input,
        __nv_bfloat16 * output,
        const unsigned int * offset_rows,
        unsigned int inner,
        unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        const unsigned long long base =
            (unsigned long long) offset_rows[0] * (unsigned long long) inner;
        output[base + index] = __float2bfloat16_rn(input[index]);
    }
}

// concat_bf16 is concat_f32 over BF16 values: a cache's rows joined along an axis.
extern "C" __global__ void concat_bf16(
        const __nv_bfloat16 * left,
        const __nv_bfloat16 * right,
        __nv_bfloat16 * output,
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

// copy_bf16 copies count BF16 values.
extern "C" __global__ void copy_bf16(
        const __nv_bfloat16 * input,
        __nv_bfloat16 * output,
        unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = input[index];
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

extern "C" __global__ void relu_backward_f32(
        const float * grad_output,
        const float * input,
        float * grad_input,
        unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        grad_input[index] = input[index] > 0.0f ? grad_output[index] : 0.0f;
    }
}

extern "C" __global__ void strided_row_copy_f32(
        const float * source,
        float * destination,
        unsigned int rows,
        unsigned int width,
        unsigned int source_stride,
        unsigned int source_offset,
        unsigned int destination_stride,
        unsigned int destination_offset) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < rows * width) {
        const unsigned int row = index / width;
        const unsigned int column = index - row * width;
        destination[(size_t) row * destination_stride + destination_offset + column] =
            source[(size_t) row * source_stride + source_offset + column];
    }
}

// indexed_row_scatter_add_f32 adds each source row into the destination row its index names, one thread per column
// adding the source rows in order, so rows that share an index sum to the same bytes run to run, as an atomic add per
// element did not (fixed-order/do).
extern "C" __global__ void indexed_row_scatter_add_f32(
        const float * source,
        const unsigned int * rows,
        float * destination,
        unsigned int count,
        unsigned int width) {
    const unsigned int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= width) {
        return;
    }
    for (unsigned int row = 0; row < count; row++) {
        destination[(size_t) rows[row] * width + column] += source[(size_t) row * width + column];
    }
}

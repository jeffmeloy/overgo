#include <math.h>
#include <cuda_bf16.h>

__device__ __forceinline__ float native_bf16(float value) {
    return __bfloat162float(__float2bfloat16_rn(value));
}

// Native rotary coefficients and both products round separately before the sum.
// The first axis owns pairs outside the height and width interleaved sections.
extern "C" __global__ void rope_native_bf16_f32(
        const float * input, const unsigned int * positions, const float * frequencies, float * output,
        unsigned int width, unsigned int heads, unsigned int tokens, unsigned int rotary,
        unsigned int height_section, unsigned int width_section, unsigned int multi,
        unsigned int positions_f32, unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const unsigned int column = index % width;
    if (column >= rotary) { output[index] = input[index]; return; }
    const unsigned int half = rotary / 2, pair = column % half, token = (index / width / heads) % tokens;
    unsigned int axis = 0;
    if (multi && pair % 3 == 1 && pair < height_section * 3) axis = 1;
    if (multi && pair % 3 == 2 && pair < width_section * 3) axis = 2;
    const float position = positions_f32 ? reinterpret_cast<const float *>(positions)[token * 4 + axis] : (float)positions[axis * tokens + token];
    const float angle = __fmul_rn(position, frequencies[pair]);
    const float cosine = native_bf16(cosf(angle)), sine = native_bf16(sinf(angle));
    const unsigned int partner = column < half ? index + half : index - half;
    const float rotated = column < half ? -input[partner] : input[partner];
    output[index] = native_bf16(__fadd_rn(native_bf16(__fmul_rn(input[index], cosine)), native_bf16(__fmul_rn(rotated, sine))));
}

// Native attention preserves full head batching and symmetric source scaling.
// Staging is admitted and reused by the shared BLAS owner.
extern "C" __global__ void attention_native_pack_f32(
        const float * query, const float * key, const float * value, float * output,
        float * packed_query, float * packed_key, float * packed_value, const float * packed_output,
        unsigned int width, unsigned int value_width, unsigned int heads, unsigned int key_heads,
        unsigned int queries, unsigned int keys, unsigned int groups, float scale,
        unsigned int phase, unsigned int count) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    if (phase != 0) {
        const unsigned int d = i % value_width, t = (i / value_width) % queries;
        const unsigned int group = i / (value_width * queries), h = group % heads, b = group / heads;
        output[((b * queries + t) * heads + h) * value_width + d] = packed_output[i];
        return;
    }
    if (i < (size_t) groups * queries * width) {
        const unsigned int d = i % width, t = (i / width) % queries;
        const unsigned int group = i / (width * queries), h = group % heads, b = group / heads;
        packed_query[i] = __fmul_rn(query[((b * queries + t) * heads + h) * width + d], scale);
    }
    if (i < (size_t) groups * keys * width) {
        const unsigned int d = i % width, t = (i / width) % keys;
        const unsigned int group = i / (width * keys), h = (group % heads) / (heads / key_heads), b = group / heads;
        packed_key[i] = __fmul_rn(key[((b * keys + t) * key_heads + h) * width + d], scale);
    }
    if (i < (size_t) groups * keys * value_width) {
        const unsigned int d = i % value_width, t = (i / value_width) % keys;
        const unsigned int group = i / (value_width * keys), h = (group % heads) / (heads / key_heads), b = group / heads;
        packed_value[i] = value[((b * keys + t) * key_heads + h) * value_width + d];
    }
}

// PersistentSoftmax.cuh uses at most 2048 values and 32 lanes per F32 row.
// Wider validated rows use SoftMax.cu's 1024-thread register and block reduction.
extern "C" __global__ void attention_native_softmax_f32(
        float * scores, unsigned int keys, unsigned int queries, unsigned int rows, unsigned int causal) {
    const unsigned int row = blockIdx.x, lane = threadIdx.x, warp = blockDim.x;
    if (row >= rows) return;
    if (keys > 2048) {
        __shared__ float partial[32];
        float values[4];
        const unsigned int local_lane = lane % 32, group = lane / 32;
        const float lowest = -3.4028234663852886e+38f;
        const float negative_infinity = __int_as_float(0xff800000);
        float maximum = lowest;
        const unsigned int iterations = (keys + blockDim.x - 1) / blockDim.x;
        for (unsigned int i = 0; i < iterations; ++i) {
            const unsigned int k = lane + i * blockDim.x;
            values[i] = k < keys && (!causal || k <= row % queries) ? scores[(size_t) row * keys + k] : negative_infinity;
            if (k < keys) maximum = fmaxf(maximum, values[i]);
        }
        for (unsigned int offset = 16; offset > 0; offset >>= 1) maximum = fmaxf(maximum, __shfl_down_sync(0xffffffff, maximum, offset));
        __syncthreads();
        if (local_lane == 0) partial[group] = maximum;
        __syncthreads();
        maximum = lane < 32 ? partial[local_lane] : lowest;
        if (group == 0) {
            for (unsigned int offset = 16; offset > 0; offset >>= 1) maximum = fmaxf(maximum, __shfl_down_sync(0xffffffff, maximum, offset));
        }
        if (lane == 0) partial[0] = maximum;
        __syncthreads();
        maximum = partial[0];
        float sum = 0.0f;
        for (unsigned int i = 0; i < iterations; ++i) {
            if (lane + i * blockDim.x < keys) sum = __fadd_rn(sum, expf(__fsub_rn(values[i], maximum)));
        }
        for (unsigned int offset = 16; offset > 0; offset >>= 1) sum = __fadd_rn(sum, __shfl_down_sync(0xffffffff, sum, offset));
        __syncthreads();
        if (local_lane == 0) partial[group] = sum;
        __syncthreads();
        sum = lane < 32 ? partial[local_lane] : 0.0f;
        if (group == 0) {
            for (unsigned int offset = 16; offset > 0; offset >>= 1) sum = __fadd_rn(sum, __shfl_down_sync(0xffffffff, sum, offset));
        }
        if (lane == 0) partial[0] = sum;
        __syncthreads();
        sum = partial[0];
        for (unsigned int i = 0; i < iterations; ++i) {
            const unsigned int k = lane + i * blockDim.x;
            if (k < keys) scores[(size_t) row * keys + k] = __fdiv_rn(expf(__fsub_rn(values[i], maximum)), sum);
        }
        return;
    }
    float values[64];
    const unsigned int iterations = (keys + warp - 1) / warp;
    const float negative_infinity = __int_as_float(0xff800000);
    float maximum = negative_infinity;
    for (unsigned int i = 0; i < iterations; ++i) {
        const unsigned int k = lane + i * warp;
        values[i] = k < keys && (!causal || k <= row % queries) ? scores[(size_t) row * keys + k] : negative_infinity;
        maximum = fmaxf(maximum, values[i]);
    }
    const unsigned int mask = __activemask();
    for (unsigned int offset = warp / 2; offset > 0; offset >>= 1) maximum = fmaxf(maximum, __shfl_xor_sync(mask, maximum, offset, warp));
    float sum = 0.0f;
    for (unsigned int i = 0; i < iterations; ++i) {
        values[i] = expf(__fsub_rn(values[i], maximum));
        sum = __fadd_rn(sum, values[i]);
    }
    for (unsigned int offset = warp / 2; offset > 0; offset >>= 1) sum = __fadd_rn(sum, __shfl_xor_sync(mask, sum, offset, warp));
    for (unsigned int i = 0; i < iterations; ++i) {
        const unsigned int k = lane + i * warp;
        if (k < keys) scores[(size_t) row * keys + k] = __fdiv_rn(values[i], sum);
    }
}
// Native F32 math follows Torch's scalar functions and Reduce.cuh input-vector order.
// Kernel callers derive the source lane geometry and refuse cross-CTA reductions.
extern "C" __global__ void exp_native_f32(const float * input, float * output, unsigned int count) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) output[i] = expf(input[i]);
}

extern "C" __global__ void softplus_native_f32(const float * input, float * output, unsigned int count) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) output[i] = input[i] > 20.0f ? input[i] : log1pf(expf(input[i]));
}

extern "C" __global__ void rms_norm_native_f32(
        const float * input, float * output, unsigned int width, unsigned int rows,
        float epsilon, unsigned int lanes, unsigned int reduction_rows, unsigned int bf16_l2) {
    const unsigned int row = blockIdx.x, lane = threadIdx.x;
    const size_t base = (size_t) row * width;
    const unsigned int threads = lanes * reduction_rows;
    // ReduceSumProdKernel.cu loads eight BF16 values and four F32 values per lane.
    const unsigned int vector = bf16_l2 ? 8 : 4;
    float sums[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (unsigned int col = lane * vector; col < width; col += threads * vector) {
        for (unsigned int part = 0; part < vector; ++part) {
            const float x = input[base + col + part];
            const float square = __fmul_rn(x, x);
            sums[part] = __fadd_rn(sums[part], bf16_l2 ? native_bf16(square) : square);
        }
    }
    __shared__ float partial[512];
    float sum = sums[0];
    for (unsigned int part = 1; part < vector; ++part) sum = __fadd_rn(sum, sums[part]);
    partial[lane] = sum;
    __syncthreads();
    for (unsigned int offset = lanes / 2; offset > 0; offset >>= 1) {
        if (lane % lanes < offset) partial[lane] = __fadd_rn(partial[lane], partial[lane + offset]);
        __syncthreads();
    }
    for (unsigned int offset = reduction_rows / 2; offset > 0; offset >>= 1) {
        if (lane % lanes == 0 && lane / lanes < offset) partial[lane] = __fadd_rn(partial[lane], partial[lane + offset * lanes]);
        __syncthreads();
    }
    const float mean = bf16_l2 ? native_bf16(partial[0]) : __fmul_rn(partial[0], __fdiv_rn((float) rows, (float) ((size_t) rows * width)));
    const float shifted = __fadd_rn(mean, epsilon);
    const float inverse = bf16_l2 ? native_bf16(rsqrtf(native_bf16(shifted))) : rsqrtf(shifted);
    for (unsigned int col = lane; col < width; col += threads) {
        const float value = __fmul_rn(input[base + col], inverse);
        output[base + col] = bf16_l2 ? native_bf16(value) : value;
    }
}


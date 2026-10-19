#include <math.h>

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
// Keep the source XOR reduction, exponential and division order.
extern "C" __global__ void attention_native_softmax_f32(
        float * scores, unsigned int keys, unsigned int queries, unsigned int rows, unsigned int causal) {
    const unsigned int row = blockIdx.x, lane = threadIdx.x, warp = blockDim.x;
    if (row >= rows) return;
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
        float epsilon, unsigned int lanes, unsigned int reduction_rows) {
    const unsigned int row = blockIdx.x, lane = threadIdx.x;
    const size_t base = (size_t) row * width;
    const unsigned int threads = lanes * reduction_rows;
    float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (unsigned int col = lane * 4; col < width; col += threads * 4) {
        for (unsigned int part = 0; part < 4; ++part) {
            const float x = input[base + col + part];
            sums[part] = __fadd_rn(sums[part], __fmul_rn(x, x));
        }
    }
    __shared__ float partial[512];
    partial[lane] = __fadd_rn(__fadd_rn(__fadd_rn(sums[0], sums[1]), sums[2]), sums[3]);
    __syncthreads();
    for (unsigned int offset = lanes / 2; offset > 0; offset >>= 1) {
        if (lane % lanes < offset) partial[lane] = __fadd_rn(partial[lane], partial[lane + offset]);
        __syncthreads();
    }
    for (unsigned int offset = reduction_rows / 2; offset > 0; offset >>= 1) {
        if (lane % lanes == 0 && lane / lanes < offset) partial[lane] = __fadd_rn(partial[lane], partial[lane + offset * lanes]);
        __syncthreads();
    }
    const float mean = __fmul_rn(partial[0], __fdiv_rn((float) rows, (float) ((size_t) rows * width)));
    const float inverse = rsqrtf(__fadd_rn(mean, epsilon));
    for (unsigned int col = lane; col < width; col += threads) output[base + col] = __fmul_rn(input[base + col], inverse);
}


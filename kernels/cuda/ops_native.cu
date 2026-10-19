#include <math.h>
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


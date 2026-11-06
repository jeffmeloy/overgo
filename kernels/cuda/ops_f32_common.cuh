// ops_f32_common.cuh: the storage types, constants and shared device helpers every family reads, in ops_f32.cu's order. ops_f32.cu
// includes it once; it compiles only there.

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
    GGML_BF16 = 30,
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
// Q8_INPUT_SPAN_MAX: input columns one warp serves per weight read in the
// q8-input mul_mat kernels and mul_mat_f32 — the speculation verify batch bound.
constexpr unsigned int Q8_INPUT_SPAN_MAX = 8;
// Q8_INPUT_BLOCK_WARPS = the warps of a mul_mat_q8_0_input_f32 block (the default 256 threads), which take a row each or share rows.
constexpr unsigned int Q8_INPUT_BLOCK_WARPS = 8;
// F32_QUAD_WIDTH = the floats of one 16-byte read.
constexpr unsigned int F32_QUAD_WIDTH = sizeof(float4) / sizeof(float);
// F32_ROW_WARPS_MAX = the warps of the largest block, which bounds the warps that share a row of mul_mat_f32.
constexpr unsigned int F32_ROW_WARPS_MAX = 32;
// ROW_READS_IN_FLIGHT = the weight reads a thread of an F32 row dot issues before it uses any.
//   - a one-token product reads each weight once, so the reads in flight bound its rate where few warps share the card
constexpr unsigned int ROW_READS_IN_FLIGHT = 4;
// DECODE_READS_IN_FLIGHT = the key channels or keys' values a thread of attention_decode_f32 reads before it adds any.
constexpr unsigned int DECODE_READS_IN_FLIGHT = 8;
// NORM_ROW_VALUES = the values a normalization thread holds in registers, so a row of up to blockDim.x times as many is
// read once.
constexpr unsigned int NORM_ROW_VALUES = 16;

__device__ __forceinline__ int load_i32_unaligned(const unsigned char * source) {
    const size_t address = reinterpret_cast<size_t>(source);
    const unsigned int * aligned = reinterpret_cast<const unsigned int *>(
        address & ~(size_t) (sizeof(unsigned int) - 1));
    const unsigned int shift = (unsigned int) (address & (sizeof(unsigned int) - 1)) * 8;
    // The second word only when the address is unaligned: at a tensor's last word it lies past the allocation.
    if (shift == 0) {
        return (int) aligned[0];
    }
    return (int) __funnelshift_r(aligned[0], aligned[1], shift);
}

// load_f32x4 reads four consecutive activations that start 16-byte aligned.
__device__ __forceinline__ float4 load_f32x4(const float * source) {
    return *reinterpret_cast<const float4 *>(source);
}

// warp_sum_down adds a value across a warp's lanes by halving shuffles, and lane 0 holds the sum.
__device__ __forceinline__ float warp_sum_down(float value) {
    for (unsigned int offset = CUDA_WARP_WIDTH / 2; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
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
    const float total = partial[0];
    // Close reduction lifetime before callers reuse shared storage.
    __syncthreads();
    return total;
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

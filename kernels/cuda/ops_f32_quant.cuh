// ops_f32_quant.cuh: quantized row gathers, products and dequantization: Q8_0, BF16, F16 and FP8, the classic, small and K quants, in ops_f32.cu's order. ops_f32.cu
// includes it once; it compiles only there.

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
    const unsigned char * block = table + (size_t) block_index * Q8_0_BLOCK_BYTES;
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

// q8_0_input_dot_row returns in lane 0 a Q8_0 weight row's dot with one q8 input column over its blocks first,
// first + step, ...; a lane takes whole blocks.
//   - first = lane and step = CUDA_WARP_WIDTH => the warp alone takes the row, as the greedy head's argmax partials
//     and mul_mat_q8_0_input_f32's one-warp rows do, so the argmax reads the logits' bits
__device__ __forceinline__ float q8_0_input_dot_row(
        const unsigned char * left,
        const unsigned char * right,
        unsigned int inner,
        unsigned int left_row,
        unsigned int first,
        unsigned int step) {
    const unsigned int blocks_per_row = inner / Q8_0_BLOCK_WIDTH;
    const unsigned char * weight_row =
        left + (size_t) left_row * blocks_per_row * Q8_0_BLOCK_BYTES;
    float sum = 0.0f;
    for (unsigned int block = first; block < blocks_per_row; block += step) {
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

// q8_0_input_span_dots sets sums[span] in lane 0 to a Q8_0 weight row's dot with each of right_rows q8 input
// columns over its blocks first, first + step, ...; a lane takes whole blocks and reads each once for every column.
__device__ __forceinline__ void q8_0_input_span_dots(
        const unsigned char * left,
        const unsigned char * right,
        unsigned int blocks_per_row,
        unsigned int left_row,
        unsigned int first,
        unsigned int step,
        unsigned int right_rows,
        float * sums) {
    const unsigned char * weight_row =
        left + (size_t) left_row * blocks_per_row * Q8_0_BLOCK_BYTES;
    const unsigned int input_stride = blocks_per_row * Q8_INPUT_BLOCK_BYTES;
    for (unsigned int block = first; block < blocks_per_row; block += step) {
        const unsigned char * weight_block = weight_row + block * Q8_0_BLOCK_BYTES;
        int weight_words[Q8_0_BLOCK_WIDTH / sizeof(int)];
#pragma unroll
        for (unsigned int word = 0; word < Q8_0_BLOCK_WIDTH / sizeof(int); ++word) {
            weight_words[word] = load_i32_unaligned(
                weight_block + Q8_0_SCALE_BYTES + word * sizeof(int));
        }
        const float weight_scale = __half2float(
            *reinterpret_cast<const __half *>(weight_block));
#pragma unroll
        for (unsigned int span = 0; span < Q8_INPUT_SPAN_MAX; ++span) {
            if (span >= right_rows) {
                break;
            }
            const unsigned char * input_block =
                right + span * input_stride + block * Q8_INPUT_BLOCK_BYTES;
            const int * input_quants = reinterpret_cast<const int *>(
                input_block + sizeof(float));
            int dot = 0;
#pragma unroll
            for (unsigned int word = 0; word < Q8_0_BLOCK_WIDTH / sizeof(int); ++word) {
                dot = __dp4a(weight_words[word], input_quants[word], dot);
            }
            const float input_scale = *reinterpret_cast<const float *>(input_block);
            sums[span] += weight_scale * input_scale * (float) dot;
        }
    }
#pragma unroll
    for (unsigned int span = 0; span < Q8_INPUT_SPAN_MAX; ++span) {
        if (span >= right_rows) {
            break;
        }
        sums[span] = warp_sum_down(sums[span]);
    }
}

// mul_mat_q8_0_input_f32: one warp per output row serves every input
// column from a single weight read — the span verify batch reads the
// weights once for the whole draft. row_warps warps share a row, taking
// its blocks in turn, when a long row sits among few rows, so the launch
// still fills the card; the row's first warp adds their sums in warp order.
extern "C" __global__ void mul_mat_q8_0_input_f32(
        const unsigned char * left,
        const unsigned char * right,
        float * output,
        unsigned int inner,
        unsigned int left_rows,
        unsigned int right_rows,
        unsigned int row_warps) {
    __shared__ float partials[Q8_INPUT_BLOCK_WARPS][Q8_INPUT_SPAN_MAX];
    if (right_rows > Q8_INPUT_SPAN_MAX) {
        return;
    }
    const unsigned int warp = threadIdx.x / CUDA_WARP_WIDTH;
    const unsigned int part = warp % row_warps;
    const unsigned int left_row = blockIdx.x * (Q8_INPUT_BLOCK_WARPS / row_warps) + warp / row_warps;
    const unsigned int lane = threadIdx.x % CUDA_WARP_WIDTH;
    const unsigned int first = part * CUDA_WARP_WIDTH + lane;
    const unsigned int step = row_warps * CUDA_WARP_WIDTH;
    const unsigned int blocks_per_row = inner / Q8_0_BLOCK_WIDTH;
    float sums[Q8_INPUT_SPAN_MAX];
#pragma unroll
    for (unsigned int span = 0; span < Q8_INPUT_SPAN_MAX; ++span) {
        sums[span] = 0.0f;
    }
    if (left_row < left_rows && right_rows == 1) {
        sums[0] = q8_0_input_dot_row(left, right, inner, left_row, first, step);
    } else if (left_row < left_rows) {
        q8_0_input_span_dots(left, right, blocks_per_row, left_row, first, step, right_rows, sums);
    }
    if (row_warps > 1) {
        if (lane == 0) {
#pragma unroll
            for (unsigned int span = 0; span < Q8_INPUT_SPAN_MAX; ++span) {
                partials[warp][span] = sums[span];
            }
        }
        __syncthreads();
        if (part == 0 && lane == 0) {
#pragma unroll
            for (unsigned int span = 0; span < Q8_INPUT_SPAN_MAX; ++span) {
                float sum = 0.0f;
                for (unsigned int other = 0; other < row_warps; ++other) {
                    sum += partials[warp + other][span];
                }
                sums[span] = sum;
            }
        }
    }
    if (left_row >= left_rows || part != 0 || lane != 0) {
        return;
    }
#pragma unroll
    for (unsigned int span = 0; span < Q8_INPUT_SPAN_MAX; ++span) {
        if (span >= right_rows) {
            break;
        }
        output[span * left_rows + left_row] = sums[span];
    }
}

// q8_0_input_product_row sums the block's warp's Q8_0 weight row with each of right_rows q8 input columns as
// mul_mat_q8_0_input_f32 sums it, into sums, and returns whether this thread holds a row's sums (the row's first lane of
// its first warp), setting left_row; partials is the block's scratch for the row's warps.
__device__ __forceinline__ bool q8_0_input_product_row(
        const unsigned char * left,
        const unsigned char * right,
        unsigned int inner,
        unsigned int left_rows,
        unsigned int right_rows,
        unsigned int row_warps,
        float (*partials)[Q8_INPUT_SPAN_MAX],
        float * sums,
        unsigned int & left_row) {
    const unsigned int warp = threadIdx.x / CUDA_WARP_WIDTH;
    const unsigned int part = warp % row_warps;
    left_row = blockIdx.x * (Q8_INPUT_BLOCK_WARPS / row_warps) + warp / row_warps;
    const unsigned int lane = threadIdx.x % CUDA_WARP_WIDTH;
    const unsigned int first = part * CUDA_WARP_WIDTH + lane;
    const unsigned int step = row_warps * CUDA_WARP_WIDTH;
    const unsigned int blocks_per_row = inner / Q8_0_BLOCK_WIDTH;
#pragma unroll
    for (unsigned int span = 0; span < Q8_INPUT_SPAN_MAX; ++span) {
        sums[span] = 0.0f;
    }
    if (left_row < left_rows && right_rows == 1) {
        sums[0] = q8_0_input_dot_row(left, right, inner, left_row, first, step);
    } else if (left_row < left_rows) {
        q8_0_input_span_dots(left, right, blocks_per_row, left_row, first, step, right_rows, sums);
    }
    if (row_warps > 1) {
        if (lane == 0) {
#pragma unroll
            for (unsigned int span = 0; span < Q8_INPUT_SPAN_MAX; ++span) {
                partials[warp][span] = sums[span];
            }
        }
        __syncthreads();
        if (part == 0 && lane == 0) {
#pragma unroll
            for (unsigned int span = 0; span < Q8_INPUT_SPAN_MAX; ++span) {
                float sum = 0.0f;
                for (unsigned int other = 0; other < row_warps; ++other) {
                    sum += partials[warp + other][span];
                }
                sums[span] = sum;
            }
        }
    }
    return left_row < left_rows && part == 0 && lane == 0;
}

// mul_mat_q8_0_input_pair_f32 is two mul_mat_q8_0_input_f32 products of one q8 input in one launch, a gated layer's
// gate and up rows: a warp sums its row of each, so each output's bits are the single product's.
extern "C" __global__ void mul_mat_q8_0_input_pair_f32(
        const unsigned char * gate,
        const unsigned char * up,
        const unsigned char * right,
        float * gate_output,
        float * up_output,
        unsigned int inner,
        unsigned int left_rows,
        unsigned int right_rows,
        unsigned int row_warps) {
    __shared__ float gate_partials[Q8_INPUT_BLOCK_WARPS][Q8_INPUT_SPAN_MAX];
    __shared__ float up_partials[Q8_INPUT_BLOCK_WARPS][Q8_INPUT_SPAN_MAX];
    if (right_rows > Q8_INPUT_SPAN_MAX) {
        return;
    }
    float gate_sums[Q8_INPUT_SPAN_MAX], up_sums[Q8_INPUT_SPAN_MAX];
    unsigned int left_row;
    const bool holds = q8_0_input_product_row(gate, right, inner, left_rows, right_rows, row_warps, gate_partials, gate_sums, left_row);
    q8_0_input_product_row(up, right, inner, left_rows, right_rows, row_warps, up_partials, up_sums, left_row);
    if (!holds) {
        return;
    }
#pragma unroll
    for (unsigned int span = 0; span < Q8_INPUT_SPAN_MAX; ++span) {
        if (span >= right_rows) {
            break;
        }
        gate_output[span * left_rows + left_row] = gate_sums[span];
        up_output[span * left_rows + left_row] = up_sums[span];
    }
}

// mul_mat_q8_0_input_gated_add_f32 is a mul_mat_q8_0_input_f32 product scaled by the sigmoid of its column's gate and
// added to addend, as sigmoid_f32, broadcast_multiply_f32 and add_f32 round them: a gated shared expert's down product
// into the routed output.
extern "C" __global__ void mul_mat_q8_0_input_gated_add_f32(
        const unsigned char * left,
        const unsigned char * right,
        const float * gate,
        const float * addend,
        float * output,
        unsigned int inner,
        unsigned int left_rows,
        unsigned int right_rows,
        unsigned int row_warps) {
    __shared__ float partials[Q8_INPUT_BLOCK_WARPS][Q8_INPUT_SPAN_MAX];
    if (right_rows > Q8_INPUT_SPAN_MAX) {
        return;
    }
    float sums[Q8_INPUT_SPAN_MAX];
    unsigned int left_row;
    if (!q8_0_input_product_row(left, right, inner, left_rows, right_rows, row_warps, partials, sums, left_row)) {
        return;
    }
#pragma unroll
    for (unsigned int span = 0; span < Q8_INPUT_SPAN_MAX; ++span) {
        if (span >= right_rows) {
            break;
        }
        const unsigned int index = span * left_rows + left_row;
        const float sigmoid = 1.0f / (1.0f + expf(-gate[span]));
        output[index] = __fadd_rn(addend[index], __fmul_rn(sums[span], sigmoid));
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
    const float sum = valid ? q8_0_input_dot_row(left, right, inner, row, lane, CUDA_WARP_WIDTH) : 0.0f;
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
        table + (size_t) block_index * BLOCK_BYTES, \
        column % 32); \
} \
extern "C" __global__ void mul_mat_##TAG##_f32( \
        const unsigned char * left, \
        const float * right, \
        float * output, \
        unsigned int inner, \
        unsigned int left_rows, \
        unsigned int right_rows) { \
    const unsigned int thread_index = blockIdx.x * blockDim.x + threadIdx.x; \
    const unsigned int warp_index = thread_index / CUDA_WARP_WIDTH; \
    if (warp_index >= left_rows * right_rows) { \
        return; \
    } \
    const unsigned int lane = threadIdx.x % CUDA_WARP_WIDTH; \
    const unsigned int left_row = warp_index % left_rows; \
    const unsigned int right_row = warp_index / left_rows; \
    const unsigned int blocks_per_row = inner / 32; \
    const unsigned char * weight_row = \
        left + (size_t) left_row * blocks_per_row * BLOCK_BYTES; \
    const float * input_row = right + (size_t) right_row * inner; \
    float sum = 0.0f; \
    for (unsigned int column = lane; column < inner; column += CUDA_WARP_WIDTH) { \
        const unsigned char * block = \
            weight_row + (column / 32) * BLOCK_BYTES; \
        sum += DEQUANT(block, column % 32) * input_row[column]; \
    } \
    for (unsigned int offset = CUDA_WARP_WIDTH / 2; offset > 0; offset /= 2) { \
        sum += __shfl_down_sync(0xffffffffu, sum, offset); \
    } \
    if (lane == 0) { \
        output[warp_index] = sum; \
    } \
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

// dequant_rows_q1_0_f16 expands Q1_0 weight rows to f16 for the staged
// tensor-core prefill, as dequant_rows_q8_0_f16 does for Q8_0: a 128-wide
// block is one f16 scale and 128 sign bits, each element the scale or its
// negation. Without this the Q1_0 27B prefilled through the per-vector span
// kernel at its decode rate, 48 prompt tokens a second against 472 for the
// same model at Q6_K.
extern "C" __global__ void dequant_rows_q1_0_f16(
        const unsigned char * weight,
        __half * output,
        unsigned int inner,
        unsigned int start_row,
        unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const unsigned int block_width = 128u;
    const unsigned int block_bytes = 18u;
    const unsigned int column = index % inner;
    const unsigned int row = start_row + index / inner;
    const unsigned int blocks_per_row = inner / block_width;
    const unsigned char * block = weight +
        ((size_t) row * blocks_per_row + column / block_width) * block_bytes;
    output[index] = __float2half(dequant_q1_0_value(block, column % block_width));
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
    const unsigned int first = load_u32_le(packed);
    const unsigned int second = load_u32_le(packed + 4);
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
    const unsigned int packed = load_u32_le(block + 66 + group * 4);
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

#define DEFINE_SMALL_QUANT_GET_ROWS(TAG, BLOCK_WIDTH, BLOCK_BYTES, DEQUANT) \
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
        table + (size_t) block_index * BLOCK_BYTES, \
        column % BLOCK_WIDTH); \
}

#define DEFINE_SMALL_QUANT_MUL_MAT(TAG, BLOCK_WIDTH, BLOCK_BYTES, DEQUANT) \
extern "C" __global__ void mul_mat_##TAG##_f32( \
        const unsigned char * left, \
        const float * right, \
        float * output, \
        unsigned int inner, \
        unsigned int left_rows, \
        unsigned int right_rows) { \
    const unsigned int thread_index = blockIdx.x * blockDim.x + threadIdx.x; \
    const unsigned int warp_index = thread_index / CUDA_WARP_WIDTH; \
    if (warp_index >= left_rows * right_rows) { \
        return; \
    } \
    const unsigned int lane = threadIdx.x % CUDA_WARP_WIDTH; \
    const unsigned int left_row = warp_index % left_rows; \
    const unsigned int right_row = warp_index / left_rows; \
    const unsigned int blocks_per_row = inner / BLOCK_WIDTH; \
    const unsigned char * weight_row = \
        left + (size_t) left_row * blocks_per_row * BLOCK_BYTES; \
    const float * input_row = right + (size_t) right_row * inner; \
    float sum = 0.0f; \
    for (unsigned int column = lane; column < inner; column += CUDA_WARP_WIDTH) { \
        const unsigned char * block = \
            weight_row + (column / BLOCK_WIDTH) * BLOCK_BYTES; \
        sum += DEQUANT(block, column % BLOCK_WIDTH) * input_row[column]; \
    } \
    for (unsigned int offset = CUDA_WARP_WIDTH / 2; offset > 0; offset /= 2) { \
        sum += __shfl_down_sync(0xffffffffu, sum, offset); \
    } \
    if (lane == 0) { \
        output[warp_index] = sum; \
    } \
}

#define DEFINE_SMALL_QUANT_KERNELS(TAG, BLOCK_WIDTH, BLOCK_BYTES, DEQUANT) \
DEFINE_SMALL_QUANT_GET_ROWS(TAG, BLOCK_WIDTH, BLOCK_BYTES, DEQUANT) \
DEFINE_SMALL_QUANT_MUL_MAT(TAG, BLOCK_WIDTH, BLOCK_BYTES, DEQUANT)

DEFINE_SMALL_QUANT_GET_ROWS(q1_0, 128, 18, dequant_q1_0_value)

constexpr unsigned int Q1_0_BLOCK_WIDTH = 128;
constexpr unsigned int Q1_0_BLOCK_BYTES = 18;
constexpr unsigned int Q1_0_SCALE_BYTES = 2;

// mul_mat_q1_0_f32: one warp per output, as the other small-quant kernels,
// with a block's work hoisted out of its columns. Lane l owns columns l,
// l+32, l+64, ...; in a 128-wide block those are bit (l % 8) of sign bytes
// l/8, l/8+4, l/8+8 and l/8+12, so the lane reads the block's scale once and
// takes its four columns in ascending order. Each lane adds the products the
// per-element loop added, in the order it added them; the outputs are the
// same bits. The block, sign and input addresses step by constants so every
// load is a base plus an immediate.
extern "C" __global__ void mul_mat_q1_0_f32(
        const unsigned char * left,
        const float * right,
        float * output,
        unsigned int inner,
        unsigned int left_rows,
        unsigned int right_rows) {
    const unsigned int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int warp_index = thread_index / CUDA_WARP_WIDTH;
    if (warp_index >= left_rows * right_rows) {
        return;
    }
    const unsigned int lane = threadIdx.x % CUDA_WARP_WIDTH;
    const unsigned int left_row = warp_index % left_rows;
    const unsigned int right_row = warp_index / left_rows;
    const unsigned int blocks_per_row = inner / Q1_0_BLOCK_WIDTH;
    const unsigned char * block =
        left + (size_t) left_row * blocks_per_row * Q1_0_BLOCK_BYTES;
    const unsigned char * signs = block + Q1_0_SCALE_BYTES + lane / 8;
    const unsigned int mask = 1u << (lane % 8);
    const float * input = right + (size_t) right_row * inner + lane;
    float sum = 0.0f;
    for (unsigned int index = 0; index < blocks_per_row; ++index) {
        const float scale =
            __half2float(*reinterpret_cast<const __half *>(block));
        const float negative = -scale;
#pragma unroll
        for (unsigned int word = 0; word < Q1_0_BLOCK_WIDTH / CUDA_WARP_WIDTH; ++word) {
            const unsigned int sign = signs[word * (CUDA_WARP_WIDTH / 8)];
            sum += ((sign & mask) != 0 ? scale : negative) *
                input[word * CUDA_WARP_WIDTH];
        }
        block += Q1_0_BLOCK_BYTES;
        signs += Q1_0_BLOCK_BYTES;
        input += Q1_0_BLOCK_WIDTH;
    }
    for (unsigned int offset = CUDA_WARP_WIDTH / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    if (lane == 0) {
        output[warp_index] = sum;
    }
}

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

DEFINE_SMALL_QUANT_KERNELS(q2_K, 256, 84, dequant_q2_K_value)

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

DEFINE_SMALL_QUANT_KERNELS(q3_K, 256, 110, dequant_q3_K_value)

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

DEFINE_SMALL_QUANT_KERNELS(q4_K, 256, 144, dequant_q4_K_value)

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

DEFINE_SMALL_QUANT_KERNELS(q5_K, 256, 176, dequant_q5_K_value)

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

DEFINE_SMALL_QUANT_KERNELS(iq4_xs, 256, 136, dequant_iq4_xs_value)

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

// dequant_rows_<type>_f16: expand `count` consecutive weight elements
// starting at row `start_row` into f16, row-major [row][inner], for the
// tensor-core prefill of quantized weights: the weight is read once per
// forward and multiplied by cuBLAS's half-precision GEMM with F32
// accumulation, instead of once per eight prompt columns by the span
// kernels. The rounding to f16 is bounded in the executor's parity test
// against the F32 reference.
extern "C" __global__ void dequant_rows_q8_0_f16(
        const unsigned char * weight,
        __half * output,
        unsigned int inner,
        unsigned int start_row,
        unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const unsigned int column = index % inner;
    const unsigned int row = start_row + index / inner;
    const unsigned int blocks_per_row = inner / Q8_0_BLOCK_WIDTH;
    const unsigned char * block = weight +
        ((size_t) row * blocks_per_row + column / Q8_0_BLOCK_WIDTH) * Q8_0_BLOCK_BYTES;
    const float scale = __half2float(*reinterpret_cast<const __half *>(block));
    const signed char quantized =
        *(reinterpret_cast<const signed char *>(block + Q8_0_SCALE_BYTES) + column % Q8_0_BLOCK_WIDTH);
    output[index] = __float2half(scale * (float) quantized);
}

// DEFINE_DEQUANT_ROWS_F16 expands rows of a weight of BLOCK_WIDTH-column blocks of BLOCK_BYTES bytes each to f16,
// one column per thread, through its dequantizer.
#define DEFINE_DEQUANT_ROWS_F16(TAG, BLOCK_WIDTH, BLOCK_BYTES, DEQUANT) \
extern "C" __global__ void dequant_rows_##TAG##_f16( \
        const unsigned char * weight, \
        __half * output, \
        unsigned int inner, \
        unsigned int start_row, \
        unsigned int count) { \
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x; \
    if (index >= count) { \
        return; \
    } \
    const unsigned int column = index % inner; \
    const unsigned int row = start_row + index / inner; \
    const unsigned int blocks_per_row = inner / BLOCK_WIDTH; \
    const unsigned char * block = weight + ((size_t) row * blocks_per_row + column / BLOCK_WIDTH) * BLOCK_BYTES; \
    output[index] = __float2half(DEQUANT(block, column % BLOCK_WIDTH)); \
}

DEFINE_DEQUANT_ROWS_F16(q4_K, 256, 144, dequant_q4_K_value)

DEFINE_DEQUANT_ROWS_F16(q5_K, 256, 176, dequant_q5_K_value)

DEFINE_DEQUANT_ROWS_F16(q6_K, 256, 210, dequant_q6_K_value)

// dequant_rows_iq4_nl_f16 and dequant_rows_iq4_xs_f16 expand the non-linear 4-bit types the UD mixes carry beside
// their K-quants, so those weights prefill on tensor cores with the rest instead of the per-row dot kernels.
DEFINE_DEQUANT_ROWS_F16(iq4_nl, 32, 18, dequant_iq4_nl_value)

DEFINE_DEQUANT_ROWS_F16(iq4_xs, 256, 136, dequant_iq4_xs_value)

DEFINE_SMALL_QUANT_KERNELS(q6_K, 256, 210, dequant_q6_K_value)

// mul_mat_q4_K_input_f32: integer fast path for decode — per 32-column
// group the contribution splits into d*scale*dot(q,x) minus
// dmin*min*sum(x), both reduced with dp4a over the q8 input blocks.
// Bit layout mirrors dequant_q4_K_value exactly.
extern "C" __global__ void mul_mat_q4_K_input_f32(
        const unsigned char * left,
        const unsigned char * right,
        float * output,
        unsigned int inner,
        unsigned int left_rows,
        unsigned int right_rows) {
    const unsigned int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int left_row = thread_index / CUDA_WARP_WIDTH;
    if (left_row >= left_rows || right_rows > Q8_INPUT_SPAN_MAX) {
        return;
    }
    const unsigned int lane = threadIdx.x % CUDA_WARP_WIDTH;
    const unsigned int superblocks = inner / 256;
    const unsigned char * weight_row =
        left + (size_t) left_row * superblocks * 144;
    if (right_rows == 1) {
        float sum = 0.0f;
        for (unsigned int super = 0; super < superblocks; ++super) {
            const unsigned char * block = weight_row + super * 144;
            const float d =
                __half2float(*reinterpret_cast<const __half *>(block));
            const float dmin =
                __half2float(*reinterpret_cast<const __half *>(block + 2));
            const unsigned char * scales = block + 4;
            const unsigned char * quantized = block + 16;
            for (unsigned int run = lane; run < 64; run += CUDA_WARP_WIDTH) {
                const unsigned int column = run * 4;
                const unsigned int group = column / 32;
                const unsigned int qlane = column % 32;
                unsigned int group_scale;
                unsigned int group_minimum;
                scale_min_k4(group, scales, &group_scale, &group_minimum);
                const unsigned int q_word = (unsigned int) load_i32_unaligned(
                    quantized + (group / 2) * 32 + qlane);
                const unsigned int packed = group % 2 == 0
                    ? (q_word & 0x0f0f0f0fu)
                    : ((q_word >> 4) & 0x0f0f0f0fu);
                const unsigned char * input_block = right +
                    (super * 8 + group) * Q8_INPUT_BLOCK_BYTES;
                const float input_scale =
                    *reinterpret_cast<const float *>(input_block);
                const int input_word = load_i32_unaligned(
                    input_block + sizeof(float) + qlane);
                const int dot = __dp4a((int) packed, input_word, 0);
                const int input_sum = __dp4a(0x01010101, input_word, 0);
                sum += input_scale *
                    (d * (float) ((int) group_scale * dot) -
                        dmin * (float) ((int) group_minimum * input_sum));
            }
        }
        for (unsigned int offset = CUDA_WARP_WIDTH / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
        }
        if (lane == 0) {
            output[left_row] = sum;
        }
        return;
    }
    const unsigned int input_stride = (inner / 32) * Q8_INPUT_BLOCK_BYTES;
    float sums[Q8_INPUT_SPAN_MAX];
#pragma unroll
    for (unsigned int span = 0; span < Q8_INPUT_SPAN_MAX; ++span) {
        sums[span] = 0.0f;
    }
    for (unsigned int super = 0; super < superblocks; ++super) {
        const unsigned char * block = weight_row + super * 144;
        const float d =
            __half2float(*reinterpret_cast<const __half *>(block));
        const float dmin =
            __half2float(*reinterpret_cast<const __half *>(block + 2));
        const unsigned char * scales = block + 4;
        const unsigned char * quantized = block + 16;
        for (unsigned int run = lane; run < 64; run += CUDA_WARP_WIDTH) {
            const unsigned int column = run * 4;
            const unsigned int group = column / 32;
            const unsigned int qlane = column % 32;
            unsigned int group_scale;
            unsigned int group_minimum;
            scale_min_k4(group, scales, &group_scale, &group_minimum);
            const unsigned int q_word = (unsigned int) load_i32_unaligned(
                quantized + (group / 2) * 32 + qlane);
            const unsigned int packed = group % 2 == 0
                ? (q_word & 0x0f0f0f0fu)
                : ((q_word >> 4) & 0x0f0f0f0fu);
            const unsigned int input_offset =
                (super * 8 + group) * Q8_INPUT_BLOCK_BYTES;
#pragma unroll
            for (unsigned int span = 0; span < Q8_INPUT_SPAN_MAX; ++span) {
                if (span >= right_rows) {
                    break;
                }
                const unsigned char * input_block =
                    right + span * input_stride + input_offset;
                const float input_scale =
                    *reinterpret_cast<const float *>(input_block);
                const int input_word = load_i32_unaligned(
                    input_block + sizeof(float) + qlane);
                const int dot = __dp4a((int) packed, input_word, 0);
                const int input_sum = __dp4a(0x01010101, input_word, 0);
                sums[span] += input_scale *
                    (d * (float) ((int) group_scale * dot) -
                        dmin * (float) ((int) group_minimum * input_sum));
            }
        }
    }
#pragma unroll
    for (unsigned int span = 0; span < Q8_INPUT_SPAN_MAX; ++span) {
        if (span >= right_rows) {
            break;
        }
        float sum = sums[span];
        for (unsigned int offset = CUDA_WARP_WIDTH / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
        }
        if (lane == 0) {
            output[span * left_rows + left_row] = sum;
        }
    }
}

// mul_mat_q5_K_input_f32: the q4_K split plus the fifth bit from the
// high plane. Bit layout mirrors dequant_q5_K_value exactly.
extern "C" __global__ void mul_mat_q5_K_input_f32(
        const unsigned char * left,
        const unsigned char * right,
        float * output,
        unsigned int inner,
        unsigned int left_rows,
        unsigned int right_rows) {
    const unsigned int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int left_row = thread_index / CUDA_WARP_WIDTH;
    if (left_row >= left_rows || right_rows > Q8_INPUT_SPAN_MAX) {
        return;
    }
    const unsigned int lane = threadIdx.x % CUDA_WARP_WIDTH;
    const unsigned int superblocks = inner / 256;
    const unsigned char * weight_row =
        left + (size_t) left_row * superblocks * 176;
    if (right_rows == 1) {
        float sum = 0.0f;
        for (unsigned int super = 0; super < superblocks; ++super) {
            const unsigned char * block = weight_row + super * 176;
            const float d =
                __half2float(*reinterpret_cast<const __half *>(block));
            const float dmin =
                __half2float(*reinterpret_cast<const __half *>(block + 2));
            const unsigned char * scales = block + 4;
            const unsigned char * high = block + 16;
            const unsigned char * quantized = block + 48;
            for (unsigned int run = lane; run < 64; run += CUDA_WARP_WIDTH) {
                const unsigned int column = run * 4;
                const unsigned int group = column / 32;
                const unsigned int qlane = column % 32;
                unsigned int group_scale;
                unsigned int group_minimum;
                scale_min_k4(group, scales, &group_scale, &group_minimum);
                const unsigned int q_word = (unsigned int) load_i32_unaligned(
                    quantized + (group / 2) * 32 + qlane);
                const unsigned int high_word = (unsigned int) load_i32_unaligned(
                    high + qlane);
                const unsigned int low = group % 2 == 0
                    ? (q_word & 0x0f0f0f0fu)
                    : ((q_word >> 4) & 0x0f0f0f0fu);
                const unsigned int packed = low |
                    (((high_word >> group) & 0x01010101u) << 4);
                const unsigned char * input_block = right +
                    (super * 8 + group) * Q8_INPUT_BLOCK_BYTES;
                const float input_scale =
                    *reinterpret_cast<const float *>(input_block);
                const int input_word = load_i32_unaligned(
                    input_block + sizeof(float) + qlane);
                const int dot = __dp4a((int) packed, input_word, 0);
                const int input_sum = __dp4a(0x01010101, input_word, 0);
                sum += input_scale *
                    (d * (float) ((int) group_scale * dot) -
                        dmin * (float) ((int) group_minimum * input_sum));
            }
        }
        for (unsigned int offset = CUDA_WARP_WIDTH / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
        }
        if (lane == 0) {
            output[left_row] = sum;
        }
        return;
    }
    const unsigned int input_stride = (inner / 32) * Q8_INPUT_BLOCK_BYTES;
    float sums[Q8_INPUT_SPAN_MAX];
#pragma unroll
    for (unsigned int span = 0; span < Q8_INPUT_SPAN_MAX; ++span) {
        sums[span] = 0.0f;
    }
    for (unsigned int super = 0; super < superblocks; ++super) {
        const unsigned char * block = weight_row + super * 176;
        const float d =
            __half2float(*reinterpret_cast<const __half *>(block));
        const float dmin =
            __half2float(*reinterpret_cast<const __half *>(block + 2));
        const unsigned char * scales = block + 4;
        const unsigned char * high = block + 16;
        const unsigned char * quantized = block + 48;
        for (unsigned int run = lane; run < 64; run += CUDA_WARP_WIDTH) {
            const unsigned int column = run * 4;
            const unsigned int group = column / 32;
            const unsigned int qlane = column % 32;
            unsigned int group_scale;
            unsigned int group_minimum;
            scale_min_k4(group, scales, &group_scale, &group_minimum);
            const unsigned int q_word = (unsigned int) load_i32_unaligned(
                quantized + (group / 2) * 32 + qlane);
            const unsigned int high_word = (unsigned int) load_i32_unaligned(
                high + qlane);
            const unsigned int low = group % 2 == 0
                ? (q_word & 0x0f0f0f0fu)
                : ((q_word >> 4) & 0x0f0f0f0fu);
            const unsigned int packed = low |
                (((high_word >> group) & 0x01010101u) << 4);
            const unsigned int input_offset =
                (super * 8 + group) * Q8_INPUT_BLOCK_BYTES;
#pragma unroll
            for (unsigned int span = 0; span < Q8_INPUT_SPAN_MAX; ++span) {
                if (span >= right_rows) {
                    break;
                }
                const unsigned char * input_block =
                    right + span * input_stride + input_offset;
                const float input_scale =
                    *reinterpret_cast<const float *>(input_block);
                const int input_word = load_i32_unaligned(
                    input_block + sizeof(float) + qlane);
                const int dot = __dp4a((int) packed, input_word, 0);
                const int input_sum = __dp4a(0x01010101, input_word, 0);
                sums[span] += input_scale *
                    (d * (float) ((int) group_scale * dot) -
                        dmin * (float) ((int) group_minimum * input_sum));
            }
        }
    }
#pragma unroll
    for (unsigned int span = 0; span < Q8_INPUT_SPAN_MAX; ++span) {
        if (span >= right_rows) {
            break;
        }
        float sum = sums[span];
        for (unsigned int offset = CUDA_WARP_WIDTH / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
        }
        if (lane == 0) {
            output[span * left_rows + left_row] = sum;
        }
    }
}

// q6_K_input_span_row stores a Q6_K weight row's dot with each of COLUMNS q8 input columns, each summed as the
// one-column path sums it, the weights read once for every column.
//   - a count the launch fixes unrolls the column loop: one bound read each run held two columns at 60 percent of the
//     card's DRAM rate where one column reached 88 (Flash-Next's head, 2560 x 248,320)
template <unsigned int COLUMNS>
__device__ __forceinline__ void q6_K_input_span_row(
        const unsigned char * weight_row, const unsigned char * right, float * output, unsigned int superblocks,
        unsigned int left_rows, unsigned int left_row, unsigned int lane) {
    const unsigned int input_stride = superblocks * 8 * Q8_INPUT_BLOCK_BYTES;
    float sums[COLUMNS] = {};
    for (unsigned int super = 0; super < superblocks; ++super) {
        const unsigned char * block = weight_row + super * 210;
        const unsigned char * lower = block;
        const unsigned char * high = block + 128;
        const signed char * scales =
            reinterpret_cast<const signed char *>(block + 192);
        const float d =
            __half2float(*reinterpret_cast<const __half *>(block + 208));
        for (unsigned int run = lane; run < 64; run += CUDA_WARP_WIDTH) {
            const unsigned int column = run * 4;
            const unsigned int group = column / 128;
            const unsigned int within = column % 128;
            const unsigned int quarter = within / 32;
            const unsigned int qlane = within % 32;
            const unsigned int low_word = (unsigned int) load_i32_unaligned(
                lower + group * 64 + qlane +
                (quarter == 1 || quarter == 3 ? 32 : 0));
            const unsigned int high_word = (unsigned int) load_i32_unaligned(
                high + group * 32 + qlane);
            const unsigned int low = quarter < 2
                ? (low_word & 0x0f0f0f0fu)
                : ((low_word >> 4) & 0x0f0f0f0fu);
            const unsigned int packed = low |
                (((high_word >> (quarter * 2)) & 0x03030303u) << 4);
            const signed char block_scale =
                scales[group * 8 + quarter * 2 + qlane / 16];
            const unsigned int input_offset =
                (super * 8 + column / 32) * Q8_INPUT_BLOCK_BYTES;
#pragma unroll
            for (unsigned int span = 0; span < COLUMNS; ++span) {
                const unsigned char * input_block =
                    right + span * input_stride + input_offset;
                const float input_scale =
                    *reinterpret_cast<const float *>(input_block);
                const int input_word = *reinterpret_cast<const int *>(
                    input_block + sizeof(float) + (column % 32));
                int dot = __dp4a((int) packed, input_word, 0);
                dot -= 32 * __dp4a(0x01010101, input_word, 0);
                sums[span] += d * input_scale * (float) ((int) block_scale * dot);
            }
        }
    }
#pragma unroll
    for (unsigned int span = 0; span < COLUMNS; ++span) {
        float sum = sums[span];
        for (unsigned int offset = CUDA_WARP_WIDTH / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
        }
        if (lane == 0) {
            output[span * left_rows + left_row] = sum;
        }
    }
}

// mul_mat_q6_K_input_f32: integer fast path for decode — the input
// vector arrives as q8 blocks (quantize_q8_0_input_f32), and each
// four-column run reduces through one dp4a with the q6_K -32 offset
// folded through a ones-vector dot. Bit layout mirrors
// dequant_q6_K_value exactly.
extern "C" __global__ void mul_mat_q6_K_input_f32(
        const unsigned char * left,
        const unsigned char * right,
        float * output,
        unsigned int inner,
        unsigned int left_rows,
        unsigned int right_rows) {
    const unsigned int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int left_row = thread_index / CUDA_WARP_WIDTH;
    if (left_row >= left_rows || right_rows > Q8_INPUT_SPAN_MAX) {
        return;
    }
    const unsigned int lane = threadIdx.x % CUDA_WARP_WIDTH;
    const unsigned int superblocks = inner / 256;
    const unsigned char * weight_row =
        left + (size_t) left_row * superblocks * 210;
    switch (right_rows) {
    case 1: q6_K_input_span_row<1>(weight_row, right, output, superblocks, left_rows, left_row, lane); return;
    case 2: q6_K_input_span_row<2>(weight_row, right, output, superblocks, left_rows, left_row, lane); return;
    case 3: q6_K_input_span_row<3>(weight_row, right, output, superblocks, left_rows, left_row, lane); return;
    case 4: q6_K_input_span_row<4>(weight_row, right, output, superblocks, left_rows, left_row, lane); return;
    case 5: q6_K_input_span_row<5>(weight_row, right, output, superblocks, left_rows, left_row, lane); return;
    case 6: q6_K_input_span_row<6>(weight_row, right, output, superblocks, left_rows, left_row, lane); return;
    case 7: q6_K_input_span_row<7>(weight_row, right, output, superblocks, left_rows, left_row, lane); return;
    default: q6_K_input_span_row<Q8_INPUT_SPAN_MAX>(weight_row, right, output, superblocks, left_rows, left_row, lane);
    }
}

__device__ float moe_expert_value(
        const void * weights,
        size_t index,
        unsigned int storage) {
    if (storage == GGML_F32) {
        return reinterpret_cast<const float *>(weights)[index];
    }
    if (storage == GGML_BF16) {
        return __uint_as_float((unsigned int) reinterpret_cast<const unsigned short *>(weights)[index] << 16);
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

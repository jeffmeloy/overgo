// ops_f32_norms.cuh: RMS and layer norms with their fused forms, softmaxes and the GQA head reduction, in ops_f32.cu's order. ops_f32.cu
// includes it once; it compiles only there.

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
        unsigned int weight_rows,
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
        output[offset + column] = input[offset + column] * inverse * weight[(row % weight_rows) * width + column];
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
        unsigned int weight_rows,
        float epsilon) {
    const unsigned int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    const unsigned int offset = row * width;
    const float * row_weight = weight + (row % weight_rows) * width;
    const unsigned int blocks_per_row = width / Q8_0_BLOCK_WIDTH;
    const unsigned int lane = threadIdx.x % CUDA_WARP_WIDTH;
    __shared__ float partial[256];
    // A row of at most NORM_ROW_VALUES values a thread is read once, every load in flight at once, and normalized from
    // the registers; the sum keeps the strided loop's order, so either way gives the same bits.
    if (width <= NORM_ROW_VALUES * blockDim.x) {
        float values[NORM_ROW_VALUES], weights[NORM_ROW_VALUES];
#pragma unroll
        for (unsigned int index = 0; index < NORM_ROW_VALUES; ++index) {
            const unsigned int column = threadIdx.x + index * blockDim.x;
            values[index] = column < width ? input[offset + column] : 0.0f;
            weights[index] = column < width ? row_weight[column] : 0.0f;
        }
        float sum_squares = 0.0f;
#pragma unroll
        for (unsigned int index = 0; index < NORM_ROW_VALUES; ++index) {
            sum_squares += values[index] * values[index];
        }
        sum_squares = block_sum_f32(sum_squares, partial);
        const float inverse = rsqrtf(sum_squares / (float) width + epsilon);
#pragma unroll
        for (unsigned int index = 0; index < NORM_ROW_VALUES; ++index) {
            const unsigned int column = threadIdx.x + index * blockDim.x;
            if (column >= width) {
                break;
            }
            const float value = values[index] * inverse * weights[index];
            output[offset + column] = value;
            store_q8_input_warp(quantized, row * blocks_per_row + column / Q8_0_BLOCK_WIDTH, value, lane);
        }
        return;
    }
    float sum_squares = 0.0f;
    for (unsigned int column = threadIdx.x; column < width; column += blockDim.x) {
        const float value = input[offset + column];
        sum_squares += value * value;
    }
    sum_squares = block_sum_f32(sum_squares, partial);
    const float inverse = rsqrtf(sum_squares / (float) width + epsilon);
    for (unsigned int column = threadIdx.x; column < width; column += blockDim.x) {
        const float value = input[offset + column] * inverse * row_weight[column];
        output[offset + column] = value;
        store_q8_input_warp(quantized, row * blocks_per_row + column / Q8_0_BLOCK_WIDTH, value, lane);
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

// causal_softmax_batched_f32: independent square causal matrices.
extern "C" __global__ void causal_softmax_batched_f32(
        const float * scores,
        float * probabilities,
        unsigned int batches,
        unsigned int rows) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int count = batches * rows;
    if (index >= count) return;
    const unsigned int row = index % rows;
    const unsigned int offset = index * rows;
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

// reduce_gqa_heads_f32: sum query-head gradients into KV-head storage.
extern "C" __global__ void reduce_gqa_heads_f32(
        const float * expanded,
        float * reduced,
        unsigned int query_heads,
        unsigned int key_value_heads,
        unsigned int elements_per_head,
        unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const unsigned int key_value_head = index / elements_per_head;
    const unsigned int element = index % elements_per_head;
    const unsigned int group = query_heads / key_value_heads;
    float sum = 0.0f;
    for (unsigned int member = 0; member < group; ++member) {
        const unsigned int head = key_value_head * group + member;
        sum += expanded[head * elements_per_head + element];
    }
    reduced[index] = sum;
}

// f32_row_dots adds one thread's share of an F32 weight row's dot with each of columns input columns to sums.
//   - input column c starts at input + c * inner, and SPAN >= columns
//   - the row, the input and inner keep 16-byte alignment => the thread takes every threads-th quad of weights from quad thread, read 16 bytes at a time
//   - otherwise the thread takes every threads-th weight from weight thread
template <unsigned int SPAN>
__device__ __forceinline__ void f32_row_dots(
        const float * weights, const float * input, unsigned int inner, unsigned int thread, unsigned int threads,
        unsigned int columns, float * sums) {
    if (((reinterpret_cast<size_t>(weights) | reinterpret_cast<size_t>(input)) & (sizeof(float4) - 1)) == 0 &&
            inner % F32_QUAD_WIDTH == 0) {
        const unsigned int quads = inner / F32_QUAD_WIDTH;
        for (unsigned int turn = thread; turn < quads; turn += ROW_READS_IN_FLIGHT * threads) {
            // The reads of ROW_READS_IN_FLIGHT quads are issued before any is used, so they are in flight together.
            float4 weight[ROW_READS_IN_FLIGHT];
            #pragma unroll
            for (unsigned int read = 0; read < ROW_READS_IN_FLIGHT; ++read) {
                const unsigned int quad = turn + read * threads;
                weight[read] = quad < quads ? load_f32x4(weights + F32_QUAD_WIDTH * quad) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
            #pragma unroll
            for (unsigned int read = 0; read < ROW_READS_IN_FLIGHT; ++read) {
                const unsigned int quad = turn + read * threads;
                #pragma unroll
                for (unsigned int column = 0; column < SPAN; ++column) {
                    if (column < columns && quad < quads) {
                        const float4 x = load_f32x4(input + (size_t) column * inner + F32_QUAD_WIDTH * quad);
                        const float4 w = weight[read];
                        sums[column] += w.x * x.x + w.y * x.y + w.z * x.z + w.w * x.w;
                    }
                }
            }
        }
        return;
    }
    for (unsigned int index = thread; index < inner; index += threads) {
        const float weight = weights[index];
        #pragma unroll
        for (unsigned int column = 0; column < SPAN; ++column) {
            if (column < columns) sums[column] += weight * input[(size_t) column * inner + index];
        }
    }
}

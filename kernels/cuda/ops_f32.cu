#include <cuda_fp16.h>
#include "iq_tables_generated.cuh"

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

extern "C" __global__ void l2_norm_f32(
        const float * input,
        float * output,
        unsigned int width,
        unsigned int rows,
        float epsilon) {
    const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    const unsigned int offset = row * width;
    float sum_squares = 0.0f;
    for (unsigned int column = 0; column < width; ++column) {
        const float value = input[offset + column];
        sum_squares += value * value;
    }
    const float inverse = 1.0f / fmaxf(sqrtf(sum_squares), epsilon);
    for (unsigned int column = 0; column < width; ++column) {
        output[offset + column] = input[offset + column] * inverse;
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
        unsigned int gate_width) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
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
    for (unsigned int i = 0; i < state_elements; ++i) {
        state[i] = source_state[i];
    }

    const float scale = rsqrtf((float) size);
    for (unsigned int token = 0; token < tokens; ++token) {
        const unsigned int value_base =
            ((sequence * tokens + token) * heads + head) * size;
        const unsigned int query_base =
            ((sequence * tokens + token) * query_heads +
                head % query_heads) * size;
        const unsigned int key_base =
            ((sequence * tokens + token) * key_heads +
                head % key_heads) * size;
        const unsigned int gate_base =
            ((sequence * tokens + token) * heads + head) * gate_width;
        const float beta_value =
            beta[(sequence * tokens + token) * heads + head];

        if (gate_width == 1) {
            const float gate_scale = expf(gate[gate_base]);
            for (unsigned int i = 0; i < state_elements; ++i) {
                state[i] *= gate_scale;
            }
        } else {
            for (unsigned int row = 0; row < size; ++row) {
                for (unsigned int column = 0; column < size; ++column) {
                    state[row * size + column] *=
                        expf(gate[gate_base + column]);
                }
            }
        }

        for (unsigned int row = 0; row < size; ++row) {
            float * state_row = state + row * size;
            float dot = 0.0f;
            for (unsigned int column = 0; column < size; ++column) {
                dot += state_row[column] * key[key_base + column];
            }
            const float delta =
                (value[value_base + row] - dot) * beta_value;
            for (unsigned int column = 0; column < size; ++column) {
                state_row[column] += delta * key[key_base + column];
            }
            dot = 0.0f;
            for (unsigned int column = 0; column < size; ++column) {
                dot += state_row[column] * query[query_base + column];
            }
            output[value_base + row] = dot * scale;
        }
    }
}

extern "C" __global__ void rms_norm_f32(
        const float * input,
        float * output,
        unsigned int width,
        unsigned int rows,
        float epsilon) {
    const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    const unsigned int offset = row * width;
    float sum_squares = 0.0f;
    for (unsigned int column = 0; column < width; ++column) {
        const float value = input[offset + column];
        sum_squares += value * value;
    }
    const float inverse = rsqrtf(sum_squares / (float) width + epsilon);
    for (unsigned int column = 0; column < width; ++column) {
        output[offset + column] = input[offset + column] * inverse;
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

extern "C" __global__ void rope_neox_f32(
        const float * input,
        const unsigned int * positions,
        float * output,
        unsigned int width,
        unsigned int heads,
        unsigned int tokens,
        unsigned int rotary_dimensions,
        float frequency_base,
        float frequency_scale,
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
    const float theta =
        (float) positions[token] * frequency_scale *
        powf(frequency_base, -2.0f * (float) pair / (float) rotary_dimensions);
    float sine;
    float cosine;
    sincosf(theta, &sine, &cosine);
    const float x0 = input[pair_offset];
    const float x1 = input[pair_offset + half];
    output[index] = column < half
        ? x0 * cosine - x1 * sine
        : x0 * sine + x1 * cosine;
}

extern "C" __global__ void rope_normal_f32(
        const float * input,
        const unsigned int * positions,
        float * output,
        unsigned int width,
        unsigned int heads,
        unsigned int tokens,
        unsigned int rotary_dimensions,
        float frequency_base,
        float frequency_scale,
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
    const float theta =
        (float) positions[token] * frequency_scale *
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

extern "C" __global__ void rope_multi_f32(
        const float * input,
        const unsigned int * positions,
        float * output,
        unsigned int width,
        unsigned int heads,
        unsigned int tokens,
        unsigned int rotary_dimensions,
        float frequency_base,
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
    const unsigned int half = rotary_dimensions / 2;
    const unsigned int pair = column % half;
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
    const unsigned int pair_offset = row * width + pair;
    const float theta =
        (float) positions[axis * tokens + token] *
        powf(frequency_base, -2.0f * (float) pair / (float) rotary_dimensions);
    float sine;
    float cosine;
    sincosf(theta, &sine, &cosine);
    const float x0 = input[pair_offset];
    const float x1 = input[pair_offset + half];
    output[index] = column < half
        ? x0 * cosine - x1 * sine
        : x0 * sine + x1 * cosine;
}

extern "C" __global__ void attention_f32(
        const float * query,
        const float * key,
        const float * value,
        const float * relative_bias,
        float * output,
        unsigned int key_width,
        unsigned int value_width,
        unsigned int query_heads,
        unsigned int key_value_heads,
        unsigned int query_tokens,
        unsigned int key_value_tokens,
        float scale,
        unsigned int causal,
        unsigned int query_start,
        unsigned int window,
        unsigned int relative_buckets,
        unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const unsigned int value_channel = index % value_width;
    unsigned int row = index / value_width;
    const unsigned int query_head = row % query_heads;
    const unsigned int query_token = row / query_heads;
    const unsigned int group_size = query_heads / key_value_heads;
    const unsigned int key_value_head = query_head / group_size;
    const unsigned int key_limit = causal
        ? query_start + query_token + 1
        : key_value_tokens;
    const unsigned int key_first =
        window > 0 && key_limit > window ? key_limit - window : 0;
    const unsigned int query_offset =
        (query_token * query_heads + query_head) * key_width;

    float maximum = -3.402823466e+38F;
    for (unsigned int key_token = key_first; key_token < key_limit; ++key_token) {
        const unsigned int key_offset =
            (key_token * key_value_heads + key_value_head) * key_width;
        float dot = 0.0f;
        for (unsigned int channel = 0; channel < key_width; ++channel) {
            dot += query[query_offset + channel] * key[key_offset + channel];
        }
        float score = dot * scale;
        if (relative_bias != nullptr) {
            const unsigned int half = relative_buckets / 2;
            const unsigned int max_exact = half / 2;
            int distance = (int) key_token - (int) query_token;
            unsigned int bucket = distance > 0 ? half : 0;
            distance = distance < 0 ? -distance : distance;
            if ((unsigned int) distance < max_exact) {
                bucket += (unsigned int) distance;
            } else {
                unsigned int large = max_exact + (unsigned int) floorf(
                    logf((float) distance / (float) max_exact) *
                    (float) (half - max_exact) /
                    logf(128.0f / (float) max_exact));
                bucket += large < half ? large : half - 1;
            }
            score += relative_bias[bucket * query_heads + query_head];
        }
        maximum = fmaxf(maximum, score);
    }

    float sum = 0.0f;
    float weighted = 0.0f;
    for (unsigned int key_token = key_first; key_token < key_limit; ++key_token) {
        const unsigned int key_offset =
            (key_token * key_value_heads + key_value_head) * key_width;
        float dot = 0.0f;
        for (unsigned int channel = 0; channel < key_width; ++channel) {
            dot += query[query_offset + channel] * key[key_offset + channel];
        }
        float score = dot * scale;
        if (relative_bias != nullptr) {
            const unsigned int half = relative_buckets / 2;
            const unsigned int max_exact = half / 2;
            int distance = (int) key_token - (int) query_token;
            unsigned int bucket = distance > 0 ? half : 0;
            distance = distance < 0 ? -distance : distance;
            if ((unsigned int) distance < max_exact) {
                bucket += (unsigned int) distance;
            } else {
                unsigned int large = max_exact + (unsigned int) floorf(
                    logf((float) distance / (float) max_exact) *
                    (float) (half - max_exact) /
                    logf(128.0f / (float) max_exact));
                bucket += large < half ? large : half - 1;
            }
            score += relative_bias[bucket * query_heads + query_head];
        }
        const float probability = expf(score - maximum);
        const unsigned int value_offset =
            (key_token * key_value_heads + key_value_head) * value_width;
        sum += probability;
        weighted += probability * value[value_offset + value_channel];
    }
    output[index] = weighted / sum;
}

extern "C" __global__ void concat_f32(
        const float * left,
        const float * right,
        float * output,
        unsigned int left_count,
        unsigned int left_width,
        unsigned int right_width,
        unsigned int axis,
        unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        if (axis == 0) {
            const unsigned int output_width = left_width + right_width;
            const unsigned int column = index % output_width;
            const unsigned int outer = index / output_width;
            output[index] = column < left_width
                ? left[outer * left_width + column]
                : right[outer * right_width + column - left_width];
        } else {
            output[index] = index < left_count
                ? left[index]
                : right[index - left_count];
        }
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
    const unsigned int blocks_per_row = width / 32;
    const unsigned int block_index =
        rows[output_row] * blocks_per_row + column / 32;
    const unsigned char * block = table + block_index * 34;
    const float scale = __half2float(*reinterpret_cast<const __half *>(block));
    const signed char quantized =
        *(reinterpret_cast<const signed char *>(block + 2) + column % 32);
    output[index] = scale * (float) quantized;
}

extern "C" __global__ void mul_mat_q8_0_f32(
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
    const unsigned int blocks_per_row = inner / 32;
    const unsigned char * weight_row =
        left + left_row * blocks_per_row * 34;
    const float * input_row = right + right_row * inner;
    float sum = 0.0f;
    for (unsigned int block_index = 0; block_index < blocks_per_row; ++block_index) {
        const unsigned char * block = weight_row + block_index * 34;
        const float scale = __half2float(*reinterpret_cast<const __half *>(block));
        const signed char * quantized =
            reinterpret_cast<const signed char *>(block + 2);
        const unsigned int column_base = block_index * 32;
        float block_sum = 0.0f;
        for (unsigned int column = 0; column < 32; ++column) {
            block_sum += (float) quantized[column] * input_row[column_base + column];
        }
        sum += scale * block_sum;
    }
    output[index] = sum;
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

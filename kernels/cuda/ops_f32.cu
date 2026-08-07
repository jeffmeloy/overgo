#include <cuda_fp16.h>
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
        const float probability = expf(score - maximum);
        const unsigned int value_offset =
            ((sequence * key_value_tokens + key_token) * key_value_heads + key_value_head) * value_width;
        sum += probability;
        weighted += probability * value[value_offset + value_channel];
    }
    output[index] = weighted / sum;
}

extern "C" __global__ void attention_decode_f32(
        const float * query,
        const float * key,
        const float * value,
        float * output,
        unsigned int key_width,
        unsigned int value_width,
        unsigned int query_heads,
        unsigned int key_value_heads,
        unsigned int key_value_tokens,
        unsigned int sequences,
        float scale) {
    const unsigned int row = blockIdx.x;
    const unsigned int query_head = row % query_heads;
    const unsigned int sequence = row / query_heads;
    if (sequence >= sequences) {
        return;
    }
    const unsigned int group_size = query_heads / key_value_heads;
    const unsigned int key_value_head = query_head / group_size;
    const unsigned int query_offset =
        (sequence * query_heads + query_head) * key_width;
    extern __shared__ float shared[];
    float * scores = shared;
    float * partial = shared + key_value_tokens;
    float local_maximum = -3.402823466e+38F;
    for (unsigned int token = threadIdx.x; token < key_value_tokens; token += blockDim.x) {
        const unsigned int key_offset =
            ((sequence * key_value_tokens + token) * key_value_heads + key_value_head) * key_width;
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
                ((sequence * key_value_tokens + token) * key_value_heads + key_value_head) * value_width;
            weighted += scores[token] * value[value_offset + channel];
        }
        output[(sequence * query_heads + query_head) * value_width + channel] = weighted / sum;
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

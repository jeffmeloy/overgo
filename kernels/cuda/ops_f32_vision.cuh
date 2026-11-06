// ops_f32_vision.cuh: convolutions, windows, pixel shuffles, SAM attention and group norm, with their backward passes, in ops_f32.cu's order. ops_f32.cu
// includes it once; it compiles only there.

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

extern "C" __global__ void pixel_shuffle_2d_f32(
		const float * input,
		float * output,
		unsigned int channels,
		unsigned int input_w,
		unsigned int input_h,
		unsigned int scale,
		unsigned int count) {
	const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= count) {
		return;
	}
	const unsigned int channel = index % channels;
	const unsigned int spatial = index / channels;
	const unsigned int output_w = input_w * scale;
	const unsigned int x = spatial % output_w;
	const unsigned int y = spatial / output_w;
	const unsigned int subpixel = (y % scale) * scale + (x % scale);
	const unsigned int expanded_channels = channels * scale * scale;
	const unsigned int source_spatial = x / scale + input_w * (y / scale);
	output[index] = input[channel + channels * subpixel + expanded_channels * source_spatial];
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

// A group's moments combine by Welford's method, pairwise down each warp and then across the block's warps.
// The reduction runs over planar channels while storage stays channel-innermost.
struct group_moments_f32 {
	float mean, m2, count;
};

__device__ __forceinline__ group_moments_f32 group_combine_f32(group_moments_f32 a, group_moments_f32 b) {
	if (a.count == 0.0f) return b;
	if (b.count == 0.0f) return a;
	const float delta = __fsub_rn(b.mean, a.mean);
	const float count = __fadd_rn(a.count, b.count);
	const float ratio = __fdiv_rn(b.count, count);
	return {fmaf(delta, ratio, a.mean),
		fmaf(__fmul_rn(__fmul_rn(delta, delta), a.count), ratio, __fadd_rn(a.m2, b.m2)), count};
}

__device__ __forceinline__ group_moments_f32 group_warp_f32(group_moments_f32 value) {
	for (unsigned int offset = 16; offset; offset >>= 1) {
		group_moments_f32 other = {
			__shfl_down_sync(0xffffffff, value.mean, offset),
			__shfl_down_sync(0xffffffff, value.m2, offset),
			__shfl_down_sync(0xffffffff, value.count, offset)};
		value = group_combine_f32(value, other);
	}
	return value;
}

// sam_attention_grouped_f32 computes sam_attention_f32's outputs with a block
// a query token, head and batch: each key's score is computed once, with the
// same function, into shared memory, and each value channel sums the
// probabilities and weighted values over the keys in order, where
// sam_attention_f32 computed every score again for each value channel, twice.
extern "C" __global__ void sam_attention_grouped_f32(
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
		float relative_scale) {
	extern __shared__ float probabilities[];
	__shared__ float maxima[32];
	const unsigned int query_token = blockIdx.x;
	const unsigned int query_head = blockIdx.y;
	const unsigned int batch = blockIdx.z;
	if (query_token >= tokens || query_head >= query_heads || batch >= batches) {
		return;
	}
	const unsigned int key_head = query_head / (query_heads / key_heads);
	float maximum = -3.402823466e+38F;
	for (unsigned int key_token = threadIdx.x; key_token < tokens; key_token += blockDim.x) {
		const float score = sam_attention_score(query, key, relative_w, relative_h,
			key_width, query_heads, key_heads, tokens, batch, query_token, key_token,
			query_head, key_head, spatial_size, relative_w_length, relative_h_length, scale, relative_scale);
		probabilities[key_token] = score;
		maximum = fmaxf(maximum, score);
	}
	for (unsigned int offset = CUDA_WARP_WIDTH / 2; offset > 0; offset /= 2) {
		maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, offset));
	}
	if (threadIdx.x % CUDA_WARP_WIDTH == 0) {
		maxima[threadIdx.x / CUDA_WARP_WIDTH] = maximum;
	}
	__syncthreads();
	maximum = maxima[0];
	for (unsigned int warp = 1; warp < blockDim.x / CUDA_WARP_WIDTH; ++warp) {
		maximum = fmaxf(maximum, maxima[warp]);
	}
	for (unsigned int key_token = threadIdx.x; key_token < tokens; key_token += blockDim.x) {
		probabilities[key_token] = expf(probabilities[key_token] - maximum);
	}
	__syncthreads();
	for (unsigned int channel = threadIdx.x; channel < value_width; channel += blockDim.x) {
		float sum = 0.0f;
		float result = 0.0f;
		for (unsigned int key_token = 0; key_token < tokens; ++key_token) {
			const float probability = probabilities[key_token];
			const unsigned int value_base = value_width * (key_head + key_heads * (key_token + tokens * batch));
			sum += probability;
			result += probability * value[value_base + channel];
		}
		output[channel + value_width * (query_head + query_heads * (query_token + tokens * batch))] = result / sum;
	}
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

extern "C" __global__ void group_norm_native_f32(
		const float * input,
		const float * weight,
		const float * bias,
		float * output,
		unsigned int channels,
		unsigned int tokens,
		unsigned int groups,
		float epsilon,
		unsigned int count) {
	const unsigned int channels_per_group = channels / groups;
	const unsigned int first_channel = blockIdx.x * channels_per_group;
	const unsigned int values_per_group = channels_per_group * tokens;
	group_moments_f32 value = {0.0f, 0.0f, 0.0f};
	unsigned int observations = 0;
	for (unsigned int j = threadIdx.x; j < values_per_group; j += blockDim.x) {
		const float x = input[(j % tokens) * channels + first_channel + j / tokens];
		const float delta = __fsub_rn(x, value.mean);
		value.count = (float) ++observations;
		value.mean = __fadd_rn(value.mean, __fdiv_rn(delta, value.count));
		value.m2 = fmaf(delta, __fsub_rn(x, value.mean), value.m2);
	}
	value = group_warp_f32(value);
	__shared__ group_moments_f32 partials[32];
	__shared__ float mean, inverse;
	if (blockDim.x > 32) {
		if ((threadIdx.x % 32) == 0) partials[threadIdx.x / 32] = value;
		__syncthreads();
		value = threadIdx.x < blockDim.x / 32 ? partials[threadIdx.x] : group_moments_f32{0.0f, 0.0f, 0.0f};
		if (threadIdx.x < 32) value = group_warp_f32(value);
	}
	if (threadIdx.x == 0) {
		mean = value.mean;
		inverse = rsqrtf(__fadd_rn(__fdiv_rn(value.m2, value.count), epsilon));
	}
	__syncthreads();
	for (unsigned int j = threadIdx.x; j < values_per_group; j += blockDim.x) {
		const unsigned int channel = first_channel + j % channels_per_group;
		const unsigned int index = (j / channels_per_group) * channels + channel;
		if (tokens == 1) {
			output[index] = fmaf(__fmul_rn(__fsub_rn(input[index], mean), inverse), weight[channel], bias[channel]);
		} else {
			const float scale = __fmul_rn(inverse, weight[channel]);
			const float shift = __fadd_rn(__fmul_rn(-scale, mean), bias[channel]);
			output[index] = fmaf(scale, input[index], shift);
		}
	}
}

// group_norm_grouped_f32 computes group_norm_f32's outputs with a block a
// group: its first thread sums the group's mean and variance once, in
// group_norm_f32's order, and the block normalizes the group's values,
// where group_norm_f32 summed the whole group again for every value.
extern "C" __global__ void group_norm_grouped_f32(
		const float * input,
		const float * weight,
		const float * bias,
		float * output,
		unsigned int channels,
		unsigned int tokens,
		unsigned int groups,
		float epsilon) {
	__shared__ float statistics[2];
	const unsigned int group = blockIdx.x;
	if (group >= groups) {
		return;
	}
	const unsigned int channels_per_group = channels / groups;
	const unsigned int first_channel = group * channels_per_group;
	const unsigned int values_per_group = channels_per_group * tokens;
	if (threadIdx.x == 0) {
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
		statistics[0] = mean;
		statistics[1] = variance;
	}
	__syncthreads();
	const float mean = statistics[0];
	const float variance = statistics[1];
	for (unsigned int item = threadIdx.x; item < values_per_group; item += blockDim.x) {
		const unsigned int index = (item / channels_per_group) * channels + first_channel + item % channels_per_group;
		const unsigned int channel = index % channels;
		const float normalized = (input[index] - mean) * rsqrtf(variance / (float) values_per_group + epsilon);
		output[index] = normalized * weight[channel] + bias[channel];
	}
}

template <bool ROUND>
__device__ __forceinline__ void sigmoid_elementwise(
        const float * input,
        float * output,
        unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = elementwise_store<ROUND>(1.0f / (1.0f + expf(-input[index])));
    }
}

// ops_f32_rows.cuh: the F32 product, gated delta convolutions and gates, row gathers and rotary embeddings, in ops_f32.cu's order. ops_f32.cu
// includes it once; it compiles only there.

// mul_mat_f32 multiplies F32 weight rows by up to Q8_INPUT_SPAN_MAX input columns, a block a row.
//   - the block's threads split the row by f32_row_dots, so a few long rows still fill the card
//   - each warp sums its lanes, and thread 0 adds the warps' sums in warp order
//   - output column c holds left_rows values from output + c * left_rows
extern "C" __global__ void mul_mat_f32(
        const float * left,
        const float * right,
        float * output,
        unsigned int inner,
        unsigned int left_rows,
        unsigned int right_rows) {
    __shared__ float partials[F32_ROW_WARPS_MAX][Q8_INPUT_SPAN_MAX];
    const unsigned int row = blockIdx.x;
    if (row >= left_rows || right_rows > Q8_INPUT_SPAN_MAX) return;
    const unsigned int warp = threadIdx.x / CUDA_WARP_WIDTH;
    const unsigned int lane = threadIdx.x % CUDA_WARP_WIDTH;
    const float * weights = left + (size_t) row * inner;
    float sums[Q8_INPUT_SPAN_MAX] = {};
    // A span's columns take the instantiation of their count: eight sums for two columns ran Flash-Next's 4-, 48- and
    // 1-row products 7.7, 3.3 and 3.1 us a launch against 2.3, 1.8 and 1.4 for one column.
    if (right_rows == 1) {
        f32_row_dots<1>(weights, right, inner, threadIdx.x, blockDim.x, right_rows, sums);
    } else if (right_rows == 2) {
        f32_row_dots<2>(weights, right, inner, threadIdx.x, blockDim.x, right_rows, sums);
    } else if (right_rows <= 4) {
        f32_row_dots<4>(weights, right, inner, threadIdx.x, blockDim.x, right_rows, sums);
    } else {
        f32_row_dots<Q8_INPUT_SPAN_MAX>(weights, right, inner, threadIdx.x, blockDim.x, right_rows, sums);
    }
    #pragma unroll
    for (unsigned int column = 0; column < Q8_INPUT_SPAN_MAX; ++column) {
        if (column >= right_rows) break;
        const float sum = warp_sum_down(sums[column]);
        if (lane == 0) partials[warp][column] = sum;
    }
    __syncthreads();
    if (threadIdx.x != 0) return;
    for (unsigned int column = 0; column < right_rows; ++column) {
        float sum = 0.0f;
        for (unsigned int part = 0; part < blockDim.x / CUDA_WARP_WIDTH; ++part) sum += partials[part][column];
        output[(size_t) column * left_rows + row] = sum;
    }
}

// GATED_DELTA_FUSED_KERNEL and GATED_DELTA_FUSED_STATE_WIDTH = the largest convolution kernel and the widest state
// gated_delta_conv_f32 takes (tensor.GatedDeltaFused).
#define GATED_DELTA_FUSED_KERNEL 8
#define GATED_DELTA_FUSED_STATE_WIDTH 1024

// gated_delta_conv_f32 is tensor.Builder.GatedDeltaConv: a block takes one head's state_width channels, a thread one
// channel, holding its window and taps in registers. A channel's value at a token adds its taps in order over the
// window's rows and the token's, as ssm_conv_f32 does, then takes the SiLU; a query or key head divides its values by
// max(their L2 norm, epsilon), summed as l2_norm_f32 sums it. The window after each token is written for the next
// state and a span's checkpoints.
extern "C" __global__ void gated_delta_conv_f32(
        const float * state,
        const float * qkv,
        const float * weights,
        float * output,
        unsigned int channels,
        unsigned int state_width,
        unsigned int key_heads,
        unsigned int value_heads,
        unsigned int kernel,
        unsigned int tokens,
        float epsilon) {
    __shared__ float sums[GATED_DELTA_FUSED_STATE_WIDTH];
    const unsigned int head = blockIdx.x;
    const unsigned int channel = head * state_width + threadIdx.x;
    const unsigned int history = kernel - 1;
    const unsigned int key_channels = key_heads * state_width;
    const unsigned int value_channels = value_heads * state_width;
    const bool normalized = head < 2 * key_heads;
    float window[GATED_DELTA_FUSED_KERNEL];
    float taps[GATED_DELTA_FUSED_KERNEL];
#pragma unroll
    for (unsigned int index = 0; index < GATED_DELTA_FUSED_KERNEL; ++index) {
        window[index] = index < history ? state[channel * history + index] : 0.0f;
        taps[index] = index < kernel ? weights[channel * kernel + index] : 0.0f;
    }
    for (unsigned int token = 0; token < tokens; ++token) {
        const float x = qkv[token * channels + channel];
        float sum = 0.0f;
#pragma unroll
        for (unsigned int index = 0; index < GATED_DELTA_FUSED_KERNEL; ++index) {
            if (index < history) {
                sum += window[index] * taps[index];
            } else if (index == history) {
                sum += x * taps[index];
            }
        }
        float value = sum / (1.0f + expf(-sum));
        // The window moves on by the token, and the rows after it are written.
        float * after = output + (size_t) channels * (tokens + token * history) + (size_t) channel * history;
#pragma unroll
        for (unsigned int index = 0; index < GATED_DELTA_FUSED_KERNEL; ++index) {
            if (index + 1 < history) {
                window[index] = window[index + 1];
            } else if (index + 1 == history) {
                window[index] = x;
            }
            if (index < history) {
                after[index] = window[index];
            }
        }
        if (normalized) {
            // l2_norm_f32's reduction tree, so the norm's bits are its.
            const float total = block_sum_f32(value * value, sums);
            value *= 1.0f / fmaxf(sqrtf(total), epsilon);
        }
        if (channel < key_channels) {
            output[token * key_channels + channel] = value;
        } else if (channel < 2 * key_channels) {
            output[(size_t) key_channels * tokens + token * key_channels + channel - key_channels] = value;
        } else {
            output[(size_t) 2 * key_channels * tokens + token * value_channels + channel - 2 * key_channels] = value;
        }
    }
}

// gated_delta_conv_row is the convolution input's row at time index of channel: the state's history rows, then the
// tokens' qkv rows.
__device__ __forceinline__ float gated_delta_conv_row(
        const float * state, const float * qkv, unsigned int channel, unsigned int channels, unsigned int history,
        unsigned int index) {
    return index < history ? state[channel * history + index] : qkv[(size_t) (index - history) * channels + channel];
}

// gated_delta_conv_prompt_f32 is tensor.Builder.GatedDeltaPromptConv, gated_delta_conv_f32 for a prompt's tokens: a block
// takes one head's state_width channels at one token (blockIdx.x the token, blockIdx.y the head), a thread one channel,
// reading the rows that end at its token where they lie, so the tokens run side by side. A value adds its taps in
// gated_delta_conv_f32's order, takes the same SiLU and the same norm, and only the window after the last token is
// written.
extern "C" __global__ void gated_delta_conv_prompt_f32(
        const float * state,
        const float * qkv,
        const float * weights,
        float * output,
        unsigned int channels,
        unsigned int state_width,
        unsigned int key_heads,
        unsigned int value_heads,
        unsigned int kernel,
        unsigned int tokens,
        float epsilon) {
    __shared__ float sums[GATED_DELTA_FUSED_STATE_WIDTH];
    const unsigned int token = blockIdx.x;
    const unsigned int head = blockIdx.y;
    const unsigned int channel = head * state_width + threadIdx.x;
    const unsigned int history = kernel - 1;
    const unsigned int key_channels = key_heads * state_width;
    const unsigned int value_channels = value_heads * state_width;
    float sum = 0.0f;
    for (unsigned int index = 0; index < kernel; ++index) {
        sum += gated_delta_conv_row(state, qkv, channel, channels, history, token + index) * weights[channel * kernel + index];
    }
    float value = sum / (1.0f + expf(-sum));
    if (head < 2 * key_heads) {
        // l2_norm_f32's reduction tree, so the norm's bits are its.
        const float total = block_sum_f32(value * value, sums);
        value *= 1.0f / fmaxf(sqrtf(total), epsilon);
    }
    if (channel < key_channels) {
        output[(size_t) token * key_channels + channel] = value;
    } else if (channel < 2 * key_channels) {
        output[(size_t) key_channels * tokens + (size_t) token * key_channels + channel - key_channels] = value;
    } else {
        output[(size_t) 2 * key_channels * tokens + (size_t) token * value_channels + channel - 2 * key_channels] = value;
    }
    if (token + 1 == tokens) {
        float * after = output + (size_t) channels * tokens + (size_t) channel * history;
        for (unsigned int index = 0; index < history; ++index) {
            after[index] = gated_delta_conv_row(state, qkv, channel, channels, history, tokens + index);
        }
    }
}

// gated_delta_gates_f32 is tensor.Builder.GatedDeltaGates: a block an alpha row, then a beta row, each product summed
// as mul_mat_f32 sums it (f32_row_dots, then the warps' sums in warp order), so its bits are the composed product's; the
// alpha rows' gate is then softplus(product + time step) * decay and the beta rows' the sigmoid of the product, as
// broadcast_add_f32, softplus_f32, broadcast_multiply_f32 and sigmoid_f32 round them.
extern "C" __global__ void gated_delta_gates_f32(
        const float * input,
        const float * alpha,
        const float * beta,
        const float * time_step,
        const float * decay,
        float * output,
        unsigned int embedding,
        unsigned int heads,
        unsigned int tokens) {
    __shared__ float partials[F32_ROW_WARPS_MAX][Q8_INPUT_SPAN_MAX];
    const unsigned int row = blockIdx.x;
    if (row >= 2 * heads || tokens > Q8_INPUT_SPAN_MAX) return;
    const bool is_beta = row >= heads;
    const unsigned int head = is_beta ? row - heads : row;
    const float * weights = (is_beta ? beta : alpha) + (size_t) head * embedding;
    const unsigned int warp = threadIdx.x / CUDA_WARP_WIDTH;
    const unsigned int lane = threadIdx.x % CUDA_WARP_WIDTH;
    float sums[Q8_INPUT_SPAN_MAX] = {};
    if (tokens == 1) {
        f32_row_dots<1>(weights, input, embedding, threadIdx.x, blockDim.x, tokens, sums);
    } else if (tokens == 2) {
        f32_row_dots<2>(weights, input, embedding, threadIdx.x, blockDim.x, tokens, sums);
    } else if (tokens <= 4) {
        f32_row_dots<4>(weights, input, embedding, threadIdx.x, blockDim.x, tokens, sums);
    } else {
        f32_row_dots<Q8_INPUT_SPAN_MAX>(weights, input, embedding, threadIdx.x, blockDim.x, tokens, sums);
    }
    #pragma unroll
    for (unsigned int token = 0; token < Q8_INPUT_SPAN_MAX; ++token) {
        if (token >= tokens) break;
        const float sum = warp_sum_down(sums[token]);
        if (lane == 0) partials[warp][token] = sum;
    }
    __syncthreads();
    if (threadIdx.x != 0) return;
    for (unsigned int token = 0; token < tokens; ++token) {
        float sum = 0.0f;
        for (unsigned int part = 0; part < blockDim.x / CUDA_WARP_WIDTH; ++part) sum += partials[part][token];
        if (is_beta) {
            output[heads * tokens + token * heads + head] = 1.0f / (1.0f + expf(-sum));
        } else {
            const float value = sum + time_step[head];
            output[token * heads + head] = (fmaxf(value, 0.0f) + log1pf(expf(-fabsf(value)))) * decay[head];
        }
    }
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
        output[index] = table[(size_t) rows[output_row] * width + column];
    }
}

extern "C" __global__ void get_rows_bf16_f32(
        const __nv_bfloat16 * table,
        const unsigned int * rows,
        float * output,
        unsigned int width,
        unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        const unsigned int column = index % width;
        const unsigned int output_row = index / width;
        output[index] = __bfloat162float(table[(size_t) rows[output_row] * width + column]);
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

// rope_normal rotated DIRECTLY into its cache-append slot: the standalone
// append copy launch does not exist. Offset is device-resident (token units of
// inner elements), keeping decode launches byte-identical across steps.
extern "C" __global__ void rope_append_normal_f32(
        const float * input,
        const unsigned int * positions,
        const float * frequency_factors,
        float * output,
        const unsigned int * offset,
        unsigned int inner,
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
    float * destination = output + (size_t) offset[0] * inner + index;
    const unsigned int column = index % width;
    if (column >= rotary_dimensions) {
        *destination = input[index];
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
    *destination = (column & 1) == 0
        ? x0 * cosine - x1 * sine
        : x0 * sine + x1 * cosine;
}

// rope_neox rotated DIRECTLY into its cache-append slot; mirrors
// rope_append_normal_f32 with half-split pair indexing.
extern "C" __global__ void rope_append_neox_f32(
        const float * input,
        const unsigned int * positions,
        const float * frequency_factors,
        float * output,
        const unsigned int * offset,
        unsigned int inner,
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
    float * destination = output + (size_t) offset[0] * inner + index;
    const unsigned int column = index % width;
    if (column >= rotary_dimensions) {
        *destination = input[index];
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
    *destination = column < half
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
        unsigned int interleaved_sections,
        unsigned int split_half,
        unsigned int positions_f32,
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
    const unsigned int pair = split_half ? column % half : column / 2;
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
    if (interleaved_sections) {
        // IMRoPE cycles temporal, height and width; exhausted sections use the extra axis.
        constexpr unsigned int spatial_axes = 3;
        const unsigned int sections[] = {section_0, section_1, section_2};
        axis = sector % spatial_axes;
        if (sector >= spatial_axes * sections[axis]) {
            axis = spatial_axes;
        }
    }
    const unsigned int pair_offset = row * width + (split_half ? pair : pair * 2);
    const float position = positions_f32
        ? reinterpret_cast<const float *>(positions)[token * 4 + axis]
        : (float) positions[axis * tokens + token];
    const float theta =
        position * frequency_scale *
        powf(frequency_base, -2.0f * (float) pair / (float) rotary_dimensions);
    float sine;
    float cosine;
    sincosf(theta, &sine, &cosine);
    const float x0 = input[pair_offset];
    const float x1 = input[pair_offset + (split_half ? half : 1)];
    const bool first = split_half ? column < half : (column & 1) == 0;
    output[index] = first
        ? x0 * cosine - x1 * sine
        : x0 * sine + x1 * cosine;
}

__device__ float moe_expert_value(
        const void * weights,
        size_t index,
        unsigned int storage);

struct block_dot_codebooks;

__device__ __forceinline__ float moe_down_channel(
        const unsigned int * route_state, const float * activations, const void * down,
        const float * expert_scale, const float * down_bias, unsigned int hidden, unsigned int top_k,
        unsigned int intermediate, unsigned int normalize_top_k, float routed_scale, unsigned int expert_storage,
        unsigned int expert_index_divisor, unsigned int indexed_experts, unsigned int index,
        unsigned int lane, unsigned int width, const block_dot_codebooks * codebooks);

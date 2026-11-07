// ops_f32_unary.cuh: unary math, L2 norms, state-space convolution and scan, transposes and slices, in ops_f32.cu's order. ops_f32.cu
// includes it once; it compiles only there.

extern "C" __global__ void sigmoid_f32(
        const float * input,
        float * output,
        unsigned int count) {
    sigmoid_elementwise<false>(input, output, count);
}

extern "C" __global__ void sigmoid_bf16_round_f32(
        const float * input,
        float * output,
        unsigned int count) {
    sigmoid_elementwise<true>(input, output, count);
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

template <bool ROUND>
__device__ __forceinline__ void tanh_elementwise(
		const float * input,
		float * output,
		unsigned int count) {
	const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < count) {
		output[index] = elementwise_store<ROUND>(tanhf(input[index]));
	}
}

extern "C" __global__ void tanh_f32(
		const float * input,
		float * output,
		unsigned int count) {
    tanh_elementwise<false>(input, output, count);
}

extern "C" __global__ void tanh_bf16_round_f32(
		const float * input,
		float * output,
		unsigned int count) {
    tanh_elementwise<true>(input, output, count);
}

extern "C" __global__ void atan_f32(
		const float * input,
		float * output,
		unsigned int count) {
	const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < count) {
		output[index] = atanf(input[index]);
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

// l2_norm_backward_f32: VJP of l2_norm_f32 (per row, width columns). With
// inv = 1/max(sqrt(sum(x^2)),eps): unclamped (norm>eps)
// dX = inv*dY - inv^3 * x * (dY.x); clamped (norm<=eps) dX = inv*dY. One thread
// per row; row math in double to mirror the host f64 golden (L2NormBackward).
extern "C" __global__ void l2_norm_backward_f32(
        const float * input,
        const float * grad_output,
        float * grad_input,
        unsigned int width,
        unsigned int rows,
        float epsilon) {
    const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    const unsigned int offset = row * width;
    double sum_squares = 0.0;
    for (unsigned int column = 0; column < width; ++column) {
        const double value = (double)input[offset + column];
        sum_squares += value * value;
    }
    const double norm = sqrt(sum_squares);
    const double inverse = 1.0 / fmax(norm, (double)epsilon);
    const bool clamped = norm <= (double)epsilon;
    double dot = 0.0;
    if (!clamped) {
        for (unsigned int column = 0; column < width; ++column) {
            dot += (double)grad_output[offset + column] * (double)input[offset + column];
        }
    }
    const double inverse_cubed = inverse * inverse * inverse;
    for (unsigned int column = 0; column < width; ++column) {
        double gradient = inverse * (double)grad_output[offset + column];
        if (!clamped) {
            gradient -= inverse_cubed * (double)input[offset + column] * dot;
        }
        grad_input[offset + column] = (float)gradient;
    }
}

// short_conv_backward_f32: VJP of SiLU(depthwise causal conv1d, left-pad k-1) --
// the qwen3.5 GDN-mix short convolution (host ShortConvForward). x/grad_output/
// grad_input are [channels, tokens] channel-major; weights/grad_weights are
// [channels, k]; bias/grad_bias are per channel (has_bias==0 skips both). One
// thread per channel (channels independent). The per-token SiLU'(pre)*dY product
// is staged into the dconv_scratch double buffer, then reused for grad_weights
// (per tap) and grad_input (per position); all accumulation in double to mirror
// the host f64 golden (ShortConvBackward).
extern "C" __global__ void short_conv_backward_f32(
        const float * x,
        const float * grad_output,
        const float * weights,
        const float * bias,
        float * grad_input,
        float * grad_weights,
        float * grad_bias,
        double * dconv_scratch,
        unsigned int channels,
        unsigned int tokens,
        unsigned int k,
        unsigned int has_bias) {
    const unsigned int channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= channels) {
        return;
    }
    const float * x_row = x + (size_t)channel * tokens;
    const float * dy_row = grad_output + (size_t)channel * tokens;
    const float * weight_row = weights + (size_t)channel * k;
    double * dconv = dconv_scratch + (size_t)channel * tokens;
    double dbias = 0.0;
    for (unsigned int t = 0; t < tokens; ++t) {
        double pre = has_bias ? (double)bias[channel] : 0.0;
        for (unsigned int j = 0; j < k; ++j) {
            const int ti = (int)t - (int)(k - 1) + (int)j;
            if (ti < 0 || ti >= (int)tokens) {
                continue;
            }
            pre += (double)x_row[ti] * (double)weight_row[j];
        }
        const double s = 1.0 / (1.0 + exp(-pre));
        const double dc = (double)dy_row[t] * s * (1.0 + pre * (1.0 - s));
        dconv[t] = dc;
        dbias += dc;
    }
    if (has_bias) {
        grad_bias[channel] = (float)dbias;
    }
    for (unsigned int j = 0; j < k; ++j) {
        double dweight = 0.0;
        for (unsigned int t = 0; t < tokens; ++t) {
            const int ti = (int)t - (int)(k - 1) + (int)j;
            if (ti < 0 || ti >= (int)tokens) {
                continue;
            }
            dweight += dconv[t] * (double)x_row[ti];
        }
        grad_weights[(size_t)channel * k + j] = (float)dweight;
    }
    for (unsigned int p = 0; p < tokens; ++p) {
        double dx = 0.0;
        for (unsigned int j = 0; j < k; ++j) {
            const int t = (int)p + (int)(k - 1) - (int)j;
            if (t < 0 || t >= (int)tokens) {
                continue;
            }
            dx += dconv[t] * (double)weight_row[j];
        }
        grad_input[(size_t)channel * tokens + p] = (float)dx;
    }
}

// depthwise_causal_conv_f32 is the qwen3.5 gated-delta mix's short convolution before its SiLU (hostmath.ShortConvForward
// without the SiLU) over token-major rows [tokens, channels], a thread an output value adding its taps in order:
// output[t][c] = sum_j weights[c][j] * input[t - (kernel - 1) + j][c], inputs before the first token zero. With
// transpose set it is that convolution's VJP with respect to its input: output[t][c] = sum_j weights[c][j] *
// input[t + (kernel - 1) - j][c], inputs past the last token zero.
extern "C" __global__ void depthwise_causal_conv_f32(
        const float * input,
        const float * weights,
        float * output,
        unsigned int channels,
        unsigned int tokens,
        unsigned int kernel,
        unsigned int transpose) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= channels * tokens) {
        return;
    }
    const unsigned int channel = index % channels, token = index / channels;
    float sum = 0.0f;
    for (unsigned int tap = 0; tap < kernel; ++tap) {
        const unsigned int lag = kernel - 1 - tap;
        if (transpose != 0 ? token + lag < tokens : token >= lag) {
            const unsigned int source = transpose != 0 ? token + lag : token - lag;
            sum += weights[channel * kernel + tap] * input[(size_t) source * channels + channel];
        }
    }
    output[index] = sum;
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

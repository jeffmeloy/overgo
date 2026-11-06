// ops_f32_activations.cuh: activations, gates and stream mixes, MOSS rotations, half rope and head layouts, with the backward passes training reads, in ops_f32.cu's order. ops_f32.cu
// includes it once; it compiles only there.

extern "C" __global__ void attention_score_affine_backward_f32(
        const float * grad_scores,
        const float * query,
        const float * key,
        float * grad_lag_bias,
        float * grad_scale,
        unsigned int sequence,
        unsigned int head_dim,
        unsigned int lag_count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int query_row = index / sequence;
    const unsigned int key_row = index - query_row * sequence;
    if (query_row >= sequence || key_row > query_row) {
        return;
    }
    const float grad = grad_scores[index];
    if (grad == 0.0f) {
        return;
    }
    const unsigned int lag = min(lag_count - 1, query_row - key_row);
    float dot = 0.0f;
    for (unsigned int channel = 0; channel < head_dim; ++channel) {
        dot += query[(size_t) query_row * head_dim + channel] *
            key[(size_t) key_row * head_dim + channel];
    }
    atomicAdd(&grad_lag_bias[lag], grad);
    atomicAdd(grad_scale, grad * dot);
}

// rms_norm_backward_f32: VJP of affine RMSNorm y = x * rsqrt(mean(x^2)+eps) * w.
// One thread per row. dscale (the weight gradient) accumulates across rows via
// atomicAdd, so the caller must zero-initialize it before launch.
extern "C" __global__ void rms_norm_backward_f32(
        const float * dy,
        const float * x,
        const float * weight,
        float * dx,
        float * dscale,
        unsigned int rows,
        unsigned int d,
        float eps) {
    const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    const float * xr = x + (size_t)row * d;
    const float * dyr = dy + (size_t)row * d;
    float * dxr = dx + (size_t)row * d;
    float ss = 0.0f;
    float dotGX = 0.0f;
    for (unsigned int i = 0; i < d; i++) {
        ss += xr[i] * xr[i];
        dotGX += dyr[i] * weight[i] * xr[i];
    }
    const float inv = rsqrtf(ss / (float)d + eps);
    const float coef = inv * inv * inv / (float)d * dotGX;
    for (unsigned int i = 0; i < d; i++) {
        dxr[i] = inv * weight[i] * dyr[i] - coef * xr[i];
        atomicAdd(&dscale[i], dyr[i] * xr[i] * inv);
    }
}

// embedding_gather_rows_f32: frozen lexical lookup for resident training.
extern "C" __global__ void embedding_gather_rows_f32(
		const float * table,
		const unsigned int * ids,
		float * output,
		unsigned int rows,
		unsigned int width) {
	const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int count = rows * width;
	if (index >= count) {
		return;
	}
	const unsigned int row = index / width;
	const unsigned int column = index - row * width;
	output[index] = table[(size_t) ids[row] * width + column];
}

// softmax_ce_grad_rows_f32: mean next-token CE; logits become dLogits.
extern "C" __global__ void softmax_ce_grad_rows_f32(
		float * logits,
		const unsigned int * targets,
		float * losses,
		unsigned int rows,
		unsigned int columns,
		float scale) {
	const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row >= rows) {
		return;
	}
	float * values = logits + (size_t) row * columns;
	float maximum = -3.402823466e+38F;
	for (unsigned int column = 0; column < columns; ++column) {
		maximum = fmaxf(maximum, values[column]);
	}
	float sum = 0.0F;
	for (unsigned int column = 0; column < columns; ++column) {
		sum += expf(values[column] - maximum);
	}
	const unsigned int target = targets[row];
	losses[row] = (logf(sum) + maximum - values[target]) * scale;
	const float inverse = scale / sum;
	for (unsigned int column = 0; column < columns; ++column) {
		values[column] = expf(values[column] - maximum) * inverse;
	}
	values[target] -= scale;
}

// softmax_ce_grad_scaled_rows_f32: softmax_ce_grad_rows_f32 with a scale for
// each row in place of one for all. A row's loss and its dLogits are those of
// the next-token CE times the row's scale, which may be negative or zero: a
// row scaled by zero contributes nothing, and rows that share a scale give the
// arithmetic softmax_ce_grad_rows_f32 gives at that scale.
extern "C" __global__ void softmax_ce_grad_scaled_rows_f32(
		float * logits,
		const unsigned int * targets,
		const float * scales,
		float * losses,
		unsigned int rows,
		unsigned int columns) {
	const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row >= rows) {
		return;
	}
	const float scale = scales[row];
	float * values = logits + (size_t) row * columns;
	float maximum = -3.402823466e+38F;
	for (unsigned int column = 0; column < columns; ++column) {
		maximum = fmaxf(maximum, values[column]);
	}
	float sum = 0.0F;
	for (unsigned int column = 0; column < columns; ++column) {
		sum += expf(values[column] - maximum);
	}
	const unsigned int target = targets[row];
	losses[row] = (logf(sum) + maximum - values[target]) * scale;
	const float inverse = scale / sum;
	for (unsigned int column = 0; column < columns; ++column) {
		values[column] = expf(values[column] - maximum) * inverse;
	}
	values[target] -= scale;
}

// softmax_backward_f32: VJP of a row-wise softmax. Given the softmax output p
// and its cotangent dp, ds_i = p_i*(dp_i - sum_j p_j*dp_j). One thread per row.
extern "C" __global__ void softmax_backward_f32(
        const float * p,
        const float * dp,
        float * ds,
        unsigned int rows,
        unsigned int d) {
    const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    const float * pr = p + (size_t)row * d;
    const float * dpr = dp + (size_t)row * d;
    float * dsr = ds + (size_t)row * d;
    float dot = 0.0f;
    for (unsigned int i = 0; i < d; i++) {
        dot += pr[i] * dpr[i];
    }
    for (unsigned int i = 0; i < d; i++) {
        dsr[i] = pr[i] * (dpr[i] - dot);
    }
}

// moss_situ_f32: the MOSS feed-forward activation of [rows, 2*width] rows, each
// its width hidden values then its width gate values:
// (4 tanh(gate/4) sigmoid(gate)) * (25 tanh(hidden/25)). One thread per value.
extern "C" __global__ void moss_situ_f32(
        const float * fused,
        float * output,
        unsigned int width,
        unsigned int rows) {
    const size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= (size_t)rows * width) {
        return;
    }
    const size_t row = index / width;
    const size_t column = index % width;
    const float hidden = fused[row * 2 * width + column];
    const float gate = fused[row * 2 * width + width + column];
    const float sigmoid = 1.0f / (1.0f + expf(-gate));
    output[index] = (4.0f * tanhf(gate / 4.0f) * sigmoid) * (25.0f * tanhf(hidden / 25.0f));
}

// moss_situ_bf16_round_f32 is MOSS's SITU activation as the denoiser composes it (rotaryGraph.situ), each operation's
// F32 result rounded to BF16 as bf16_round_f32 rounds it: bounded(gate) times sigmoid(gate), times bounded(up), where
// bounded scales by its bound's inverse, takes the tanh and scales by the bound, the products in that order. The
// training kernel moss_situ_f32 divides by the bounds and rounds nothing.
extern "C" __global__ void moss_situ_bf16_round_f32(
        const float * gate,
        const float * up,
        float * output,
        float gate_inverse,
        float gate_bound,
        float hidden_inverse,
        float hidden_bound,
        unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        const float value = gate[index];
        const float bounded_gate = bf16_rounded(bf16_rounded(tanhf(bf16_rounded(value * gate_inverse))) * gate_bound);
        const float bounded_hidden = bf16_rounded(bf16_rounded(tanhf(bf16_rounded(up[index] * hidden_inverse))) * hidden_bound);
        const float sigmoid = bf16_rounded(1.0f / (1.0f + expf(-value)));
        output[index] = bf16_rounded(bf16_rounded(bounded_gate * sigmoid) * bounded_hidden);
    }
}

// moss_rotate_bf16_round_f32 is MOSS's rotary embedding as the denoiser composes it (rotaryGraph.rotate), each product
// and the sum rounded to BF16 as bf16_round_f32 rounds it: the heads times the cosine, plus the half-turned heads (each
// head's second half negated, then its first half) times the sine. The heads are [width, heads, tokens] and the tables
// [width, 1, tokens], read for every head.
extern "C" __global__ void moss_rotate_bf16_round_f32(
        const float * input,
        const float * cosine,
        const float * sine,
        float * output,
        unsigned int width,
        unsigned int heads,
        unsigned int tokens) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= width * heads * tokens) {
        return;
    }
    const unsigned int channel = index % width;
    const unsigned int token = index / (width * heads);
    const unsigned int half = width / 2u;
    const float turned = channel < half ? -input[index + half] : input[index - half];
    const unsigned int table = token * width + channel;
    output[index] = bf16_rounded(bf16_rounded(input[index] * cosine[table]) + bf16_rounded(turned * sine[table]));
}

// moss_situ_backward_f32: VJP of moss_situ_f32 into its fused input.
extern "C" __global__ void moss_situ_backward_f32(
        const float * fused,
        const float * d_output,
        float * d_fused,
        unsigned int width,
        unsigned int rows) {
    const size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= (size_t)rows * width) {
        return;
    }
    const size_t row = index / width;
    const size_t column = index % width;
    const size_t hidden_at = row * 2 * width + column;
    const size_t gate_at = hidden_at + width;
    const float hidden_tanh = tanhf(fused[hidden_at] / 25.0f);
    const float gate_tanh = tanhf(fused[gate_at] / 4.0f);
    const float sigmoid = 1.0f / (1.0f + expf(-fused[gate_at]));
    const float gate_value = 4.0f * gate_tanh * sigmoid;
    const float d_gate_value = (1.0f - gate_tanh * gate_tanh) * sigmoid + 4.0f * gate_tanh * sigmoid * (1.0f - sigmoid);
    d_fused[hidden_at] = d_output[index] * gate_value * (1.0f - hidden_tanh * hidden_tanh);
    d_fused[gate_at] = d_output[index] * 25.0f * hidden_tanh * d_gate_value;
}

// moss_rotate_f32: the MOSS rotary rotation of [tokens, heads, head_width] rows
// by a [tokens, head_width] table of sines then cosines, shared by the heads.
// sign 1 is the rotation and -1 its transpose. One thread per pair.
extern "C" __global__ void moss_rotate_f32(
        const float * x,
        const float * rope,
        float * output,
        unsigned int heads,
        unsigned int head_width,
        unsigned int tokens,
        float sign) {
    const unsigned int half = head_width / 2;
    const size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= (size_t)tokens * heads * half) {
        return;
    }
    const size_t pair = index % half;
    const size_t head = (index / half) % heads;
    const size_t token = (index / half) / heads;
    const float sine = rope[token * head_width + pair];
    const float cosine = rope[token * head_width + half + pair];
    const size_t base = (token * heads + head) * head_width + pair;
    const float first = x[base];
    const float second = x[base + half];
    output[base] = first * cosine - sign * second * sine;
    output[base + half] = second * cosine + sign * first * sine;
}

// moss_row_affine_f32: output = x * scale + shift for rows of width values, with
// one scale and one shift vector for every row. One thread per value.
extern "C" __global__ void moss_row_affine_f32(
        const float * x,
        const float * scale,
        const float * shift,
        float * output,
        unsigned int width,
        unsigned int rows) {
    const size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= (size_t)rows * width) {
        return;
    }
    const size_t column = index % width;
    output[index] = x[index] * scale[column] + shift[column];
}

// rope_half_backward_f32: VJP of split-half (HF-layout) rotary embedding. dx is
// [seq, n_heads*hd]; inv_freq is [hd/2]. Rotates each gradient pair (i, i+hd/2)
// by the negated angle pos*inv_freq[i]. One thread per (position, head, pair).
extern "C" __global__ void rope_half_backward_f32(
        float * dx,
        const float * inv_freq,
        unsigned int seq,
        unsigned int n_heads,
        unsigned int hd) {
    const unsigned int half = hd / 2;
    const unsigned int total = seq * n_heads * half;
    const unsigned int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= total) {
        return;
    }
    const unsigned int i = t % half;
    const unsigned int head = (t / half) % n_heads;
    const unsigned int pos = (t / half) / n_heads;
    const float a = (float)pos * inv_freq[i];
    const float c = cosf(a);
    const float s = sinf(a);
    const size_t base = (size_t)pos * n_heads * hd + (size_t)head * hd;
    const float d1 = dx[base + i];
    const float d2 = dx[base + i + half];
    dx[base + i] = c * d1 + s * d2;
    dx[base + i + half] = -s * d1 + c * d2;
}

// rope_half_f32: split-half (HF-layout) rotary embedding forward. x is
// [seq, n_heads*hd]; inv_freq is [hd/2]. Rotates each pair (i, i+hd/2) by angle
// pos*inv_freq[i]. One thread per (position, head, pair). Forward rotation R;
// rope_half_backward_f32 applies its transpose.
extern "C" __global__ void rope_half_f32(
        float * x,
        const float * inv_freq,
        unsigned int seq,
        unsigned int n_heads,
        unsigned int hd) {
    const unsigned int half = hd / 2;
    const unsigned int total = seq * n_heads * half;
    const unsigned int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= total) {
        return;
    }
    const unsigned int i = t % half;
    const unsigned int head = (t / half) % n_heads;
    const unsigned int pos = (t / half) / n_heads;
    const float a = (float)pos * inv_freq[i];
    const float c = cosf(a);
    const float s = sinf(a);
    const size_t base = (size_t)pos * n_heads * hd + (size_t)head * hd;
    const float x1 = x[base + i];
    const float x2 = x[base + i + half];
    x[base + i] = c * x1 - s * x2;
    x[base + i + half] = s * x1 + c * x2;
}

// head_major_f32: reshape [seq, n_heads*hd] (position-major) ->
// [n_heads, seq, hd] (head-major, each head contiguous) so per-head device GEMMs
// address contiguous sub-buffers. One thread per output element (gather).
extern "C" __global__ void head_major_f32(
        const float * input,
        float * output,
        unsigned int seq,
        unsigned int n_heads,
        unsigned int hd) {
    const unsigned int total = seq * n_heads * hd;
    const unsigned int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= total) {
        return;
    }
    const unsigned int x = t % hd;
    const unsigned int i = (t / hd) % seq;
    const unsigned int h = (t / hd) / seq;
    output[t] = input[((size_t)i * n_heads + h) * hd + x];
}

// head_major_inverse_f32: reshape [n_heads, seq, hd] (head-major) ->
// [seq, n_heads*hd] (position-major), the inverse of head_major_f32. One thread
// per output element (gather).
extern "C" __global__ void head_major_inverse_f32(
        const float * input,
        float * output,
        unsigned int seq,
        unsigned int n_heads,
        unsigned int hd) {
    const unsigned int total = seq * n_heads * hd;
    const unsigned int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= total) {
        return;
    }
    const unsigned int x = t % hd;
    const unsigned int h = (t / hd) % n_heads;
    const unsigned int i = (t / hd) / n_heads;
    output[t] = input[((size_t)h * seq + i) * hd + x];
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

// stream_value is a stream's value at at: the gate's activation times values
// where a gate is given, values alone otherwise, rounded as activated_gate_f32
// rounds it.
__device__ __forceinline__ float stream_value(
        const float * gate,
        const float * values,
        unsigned int at,
        unsigned int activation) {
    if (gate == nullptr) {
        return values[at];
    }
    return __fmul_rn(activated_gate_value(gate[at], activation), values[at]);
}

// stream_sum_value sums a token's streams, each a slice of width values, in
// stream order and scales the sum, each step rounded as the group slices, adds
// and scale it replaces round theirs: no product folds into an add.
__device__ __forceinline__ float stream_sum_value(
        const float * gate,
        const float * values,
        unsigned int index,
        unsigned int activation,
        unsigned int width,
        unsigned int streams,
        float scale) {
    const unsigned int base = (index / width) * width * streams + index % width;
    float sum = stream_value(gate, values, base, activation);
    for (unsigned int stream = 1; stream < streams; ++stream) {
        sum = __fadd_rn(sum, stream_value(gate, values, base + stream * width, activation));
    }
    return __fmul_rn(sum, scale);
}

// stream_sum_f32 writes each token's scaled sum of its streams: a gated
// hyper-connection mix's group slices, adds and scale in one pass, with the
// activated gate folded in where gate is given.
extern "C" __global__ void stream_sum_f32(
        const float * gate,
        const float * values,
        float * output,
        unsigned int activation,
        unsigned int width,
        unsigned int streams,
        float scale,
        unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = stream_sum_value(gate, values, index, activation, width, streams, scale);
    }
}

// stream_sum_q8_0_f32 is stream_sum_f32 that also writes its output as the
// Q8_0 input of the product reading it, a block of 32 values a warp.
extern "C" __global__ void stream_sum_q8_0_f32(
        const float * gate,
        const float * values,
        float * output,
        unsigned char * quantized,
        unsigned int activation,
        unsigned int width,
        unsigned int streams,
        float scale,
        unsigned int count) {
    const unsigned int block = blockIdx.x;
    const unsigned int lane = threadIdx.x;
    const unsigned int index = block * Q8_0_BLOCK_WIDTH + lane;
    if (index >= count || lane >= Q8_0_BLOCK_WIDTH) {
        return;
    }
    const float value = stream_sum_value(gate, values, index, activation, width, streams, scale);
    output[index] = value;
    store_q8_input_warp(quantized, block, value, lane);
}

// stream_combine_f32 adds to each stream of residual the token's current row
// times the stream's gate, as broadcast_multiply_f32 and add_f32 round it: a
// hyper-connection combine's repeat, product and add in one pass. With scaled
// set the gate is outer * sigmoid(inner * gate), rounded as the injection's
// scale_f32, sigmoid_f32 and scale_f32 round it.
extern "C" __global__ void stream_combine_f32(
        const float * residual,
        const float * current,
        const float * gate,
        float * output,
        unsigned int width,
        unsigned int streams,
        unsigned int scaled,
        float inner,
        float outer,
        unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const unsigned int row = index / width;
    float value = gate[row];
    if (scaled != 0) {
        value = __fmul_rn(1.0f / (1.0f + expf(-__fmul_rn(value, inner))), outer);
    }
    output[index] = __fadd_rn(residual[index], __fmul_rn(current[(row / streams) * width + index % width], value));
}

// scaled_activation_f32 writes the activation of the scaled input, rounded
// as scale_f32 and then silu_f32 or sigmoid_f32 round it.
extern "C" __global__ void scaled_activation_f32(
        const float * input,
        float * output,
        unsigned int activation,
        float scale,
        unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = activated_gate_value(__fmul_rn(input[index], scale), activation);
    }
}

// scaled_activation_q8_0_f32 is scaled_activation_f32 that also writes its
// output as the Q8_0 input of the product reading it, a block of 32 values a
// warp.
extern "C" __global__ void scaled_activation_q8_0_f32(
        const float * input,
        float * output,
        unsigned char * quantized,
        unsigned int activation,
        float scale,
        unsigned int count) {
    const unsigned int block = blockIdx.x;
    const unsigned int lane = threadIdx.x;
    const unsigned int index = block * Q8_0_BLOCK_WIDTH + lane;
    if (index >= count || lane >= Q8_0_BLOCK_WIDTH) {
        return;
    }
    const float value = activated_gate_value(__fmul_rn(input[index], scale), activation);
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

// gelu_tanh_exact_f32: the elementwise tanh-GELU chain fused into one pass,
// replicating the primitive kernel sequence op-for-op (__f*_rn blocks FMA
// contraction, so every intermediate rounds exactly like the separate
// multiply/scale/add/tanh launches it replaces). bias_width != 0 folds the
// preceding broadcast bias add (bias[index % bias_width]) into the same pass.
extern "C" __global__ void gelu_tanh_exact_f32(
        const float * input,
        const float * bias,
        float * output,
        float cubic_coefficient,
        float inner_scale,
        float half_scale,
        unsigned int bias_width,
        unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        float x = input[index];
        if (bias_width != 0u) {
            x = __fadd_rn(x, bias[index % bias_width]);
        }
        const float squared = __fmul_rn(x, x);
        const float cubic = __fmul_rn(squared, x);
        const float shifted = __fadd_rn(x, __fmul_rn(cubic, cubic_coefficient));
        const float inner = __fmul_rn(shifted, inner_scale);
        const float half = __fmul_rn(x, half_scale);
        output[index] = __fadd_rn(half, __fmul_rn(half, tanhf(inner)));
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

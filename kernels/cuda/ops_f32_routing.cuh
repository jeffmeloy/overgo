// ops_f32_routing.cuh: LoRA merges, expert routing and its hidden and down products, and head repeats, in ops_f32.cu's order. ops_f32.cu
// includes it once; it compiles only there.

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

// Routed scratch holds intermediate activations, selected experts and weights.
// Strict dots keep sequential sums independent of launch geometry.
// Native dots reduce one row across a warp without expanded weight storage (moe_hidden_dot_f32, moe_down_dot_f32).
// Down projections retain slot order and restart partial sums every MOE_DOWN_TILE rows.
constexpr unsigned int MOE_DOWN_TILE = 256;

// moe_router_logits_f32 gives each warp one expert of one token across the whole grid.
// Lanes read consecutive channels at once, and lane 0 adds their products in channel
// order, so each logit has the bits of a serial sum.
extern "C" __global__ void moe_router_logits_f32(
        const float * router_input,
        const float * router,
        const float * router_bias,
        float * logits,
        unsigned int router_hidden,
        unsigned int tokens,
        unsigned int experts) {
    const unsigned int token = blockIdx.y;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int expert = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (token >= tokens || expert >= experts) return;
    const float * router_x = router_input + (size_t) token * router_hidden;
    const float * weight = router + (size_t) expert * router_hidden;
    float logit = 0.0f;
    for (unsigned int base = 0; base < router_hidden; base += 32u) {
        const unsigned int width = min(32u, router_hidden - base);
        const float x = lane < width ? router_x[base + lane] : 0.0f;
        const float w = lane < width ? weight[base + lane] : 0.0f;
        for (unsigned int channel = 0; channel < width; ++channel) {
            const float channel_x = __shfl_sync(0xffffffffu, x, channel);
            const float channel_w = __shfl_sync(0xffffffffu, w, channel);
            logit += channel_x * channel_w;
        }
    }
    if (lane == 0u) logits[(size_t) token * experts + expert] = router_bias ? logit + router_bias[expert] : logit;
}

// moe_router_dot_f32 is moe_router_logits_f32 for native execution: the warp splits each logit's channels.
//   - each lane sums its share of the channels by f32_row_dots, and the warp adds the lanes' sums by halving shuffles
extern "C" __global__ void moe_router_dot_f32(
        const float * router_input,
        const float * router,
        const float * router_bias,
        float * logits,
        unsigned int router_hidden,
        unsigned int tokens,
        unsigned int experts) {
    const unsigned int token = blockIdx.y;
    const unsigned int lane = threadIdx.x % CUDA_WARP_WIDTH;
    const unsigned int expert = blockIdx.x * (blockDim.x / CUDA_WARP_WIDTH) + threadIdx.x / CUDA_WARP_WIDTH;
    if (token >= tokens || expert >= experts) return;
    float logit = 0.0f;
    f32_row_dots<1>(router + (size_t) expert * router_hidden, router_input + (size_t) token * router_hidden,
        router_hidden, lane, CUDA_WARP_WIDTH, 1, &logit);
    logit = warp_sum_down(logit);
    if (lane == 0) logits[(size_t) token * experts + expert] = router_bias ? logit + router_bias[expert] : logit;
}

// MOE_ROUTE_WARP_EXPERTS: the most experts one warp selects among, a lane's chosen experts a bit of one word.
constexpr unsigned int MOE_ROUTE_WARP_EXPERTS = CUDA_WARP_WIDTH * 32;

// moe_route_f32 turns one token's expert logits into its routed experts and weights.
//   - routed_experts != nullptr => each routed bank expert (expert / expert_index_divisor) sets its flag there
//   - a flag is a plain store of one value, because the flags may be host memory the card maps, which a device atomic does not reach over PCIe
extern "C" __global__ void moe_route_f32(
        const float * router_logits,
        const float * selection_bias,
        const float * selected_experts,
        unsigned int * route_state,
        unsigned char * routed_experts,
        unsigned int tokens,
        unsigned int experts,
        unsigned int top_k,
        unsigned int routing,
        unsigned int expert_index_divisor) {
    const unsigned int token = blockIdx.x;
    if (token >= tokens) return;
    extern __shared__ float shared[];
    float * logits = shared;
    __shared__ float warp_score[32];
    __shared__ int warp_expert[32];
    __shared__ float block_scalar;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int warps = blockDim.x >> 5;
    for (unsigned int expert = threadIdx.x; expert < experts; expert += blockDim.x) {
        logits[expert] = router_logits[(size_t) token * experts + expert];
    }
    __syncthreads();
    float partial = -3.402823466e+38F;
    for (unsigned int expert = threadIdx.x; expert < experts; expert += blockDim.x) partial = fmaxf(partial, logits[expert]);
    for (unsigned int offset = 16u; offset > 0u; offset >>= 1) partial = fmaxf(partial, __shfl_down_sync(0xffffffffu, partial, offset));
    if (lane == 0u) warp_score[warp] = partial;
    __syncthreads();
    if (threadIdx.x == 0u) {
        float maximum = -3.402823466e+38F;
        for (unsigned int index = 0; index < warps; ++index) maximum = fmaxf(maximum, warp_score[index]);
        block_scalar = maximum;
    }
    __syncthreads();
    const float maximum = block_scalar;
    float denominator = 0.0f;
    if (routing == 1) {
        // The softmax denominator is summed across the block: one thread's serial sum over 512 experts took about
        // half of a Flash-Next decode layer's 12 us of routing.
        float sum = 0.0f;
        for (unsigned int expert = threadIdx.x; expert < experts; expert += blockDim.x) sum += expf(logits[expert] - maximum);
        for (unsigned int offset = 16u; offset > 0u; offset >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, offset);
        if (lane == 0u) warp_score[warp] = sum;
        __syncthreads();
        if (threadIdx.x == 0u) {
            float total = 0.0f;
            for (unsigned int index = 0; index < warps; ++index) total += warp_score[index];
            block_scalar = total;
        }
        __syncthreads();
        denominator = block_scalar;
    }
    for (unsigned int expert = threadIdx.x; expert < experts; expert += blockDim.x) {
        const float logit = logits[expert];
        logits[expert] = routing == 4
            ? sqrtf(fmaxf(logit, 0.0f) + log1pf(expf(-fabsf(logit))))
            : routing == 2 ? 1.0f / (1.0f + expf(-logit))
            : routing == 3 ? logit : expf(logit - maximum) / denominator;
    }
    __syncthreads();
    unsigned int * selected = route_state + (size_t) token * (2 * top_k + 1);
    float * routes = reinterpret_cast<float *>(selected + top_k);
    float * selected_sum = routes + top_k;
    if (selected_experts) {
        if (threadIdx.x != 0u) return;
        *selected_sum = 0.0f;
        for (unsigned int slot = 0; slot < top_k; ++slot) {
            const unsigned int expert = (unsigned int) selected_experts[(size_t) token * top_k + slot];
            const float probability = logits[expert];
            selected[slot] = expert;
            routes[slot] = probability;
            *selected_sum += probability;
        }
    } else {
        // Each slot takes the highest score among unchosen experts, the lowest index on a tie, as a serial scan would.
        if (experts <= MOE_ROUTE_WARP_EXPERTS) {
            // One warp selects: lane l holds experts l, l + 32, ..., and a slot is a warp argmax, with no block barrier
            // (ten of them took about half of a Flash-Next decode layer's routing).
            if (warp != 0u) return;
            unsigned int chosen_bits = 0u;
            float sum = 0.0f;
            for (unsigned int slot = 0; slot < top_k; ++slot) {
                int best = -1;
                float best_score = -3.402823466e+38F;
                for (unsigned int expert = lane, bit = 0; expert < experts; expert += CUDA_WARP_WIDTH, ++bit) {
                    if (chosen_bits >> bit & 1u) continue;
                    const float score = logits[expert] + (selection_bias ? selection_bias[expert] : 0.0f);
                    if (best < 0 || score > best_score) {
                        best = (int) expert;
                        best_score = score;
                    }
                }
                // A butterfly leaves every lane the warp's winner under one total order.
                for (unsigned int offset = 16u; offset > 0u; offset >>= 1) {
                    const float other_score = __shfl_xor_sync(0xffffffffu, best_score, offset);
                    const int other = __shfl_xor_sync(0xffffffffu, best, offset);
                    if (other >= 0 && (best < 0 || other_score > best_score || (other_score == best_score && other < best))) {
                        best = other;
                        best_score = other_score;
                    }
                }
                if (best >= 0 && (unsigned int) best % CUDA_WARP_WIDTH == lane) chosen_bits |= 1u << ((unsigned int) best / CUDA_WARP_WIDTH);
                if (lane == 0u) {
                    selected[slot] = (unsigned int) best;
                    routes[slot] = logits[best];
                }
                sum += logits[best];
            }
            if (lane != 0u) return;
            *selected_sum = sum;
        } else {
            if (threadIdx.x == 0u) *selected_sum = 0.0f;
            __syncthreads();
            for (unsigned int slot = 0; slot < top_k; ++slot) {
                int best = -1;
                float best_score = -3.402823466e+38F;
                for (unsigned int expert = threadIdx.x; expert < experts; expert += blockDim.x) {
                    bool used = false;
                    for (unsigned int prior = 0; prior < slot; ++prior) used = used || selected[prior] == expert;
                    if (used) continue;
                    const float score = logits[expert] + (selection_bias ? selection_bias[expert] : 0.0f);
                    if (best < 0 || score > best_score) {
                        best = (int) expert;
                        best_score = score;
                    }
                }
                for (unsigned int offset = 16u; offset > 0u; offset >>= 1) {
                    const float other_score = __shfl_down_sync(0xffffffffu, best_score, offset);
                    const int other = __shfl_down_sync(0xffffffffu, best, offset);
                    if (other >= 0 && (best < 0 || other_score > best_score || (other_score == best_score && other < best))) {
                        best = other;
                        best_score = other_score;
                    }
                }
                if (lane == 0u) {
                    warp_score[warp] = best_score;
                    warp_expert[warp] = best;
                }
                __syncthreads();
                if (threadIdx.x == 0u) {
                    int chosen = -1;
                    float chosen_score = -3.402823466e+38F;
                    for (unsigned int index = 0; index < warps; ++index) {
                        const int candidate = warp_expert[index];
                        const float candidate_score = warp_score[index];
                        if (candidate >= 0 && (chosen < 0 || candidate_score > chosen_score || (candidate_score == chosen_score && candidate < chosen))) {
                            chosen = candidate;
                            chosen_score = candidate_score;
                        }
                    }
                    selected[slot] = (unsigned int) chosen;
                    routes[slot] = logits[chosen];
                    *selected_sum += logits[chosen];
                }
                __syncthreads();
            }
            if (threadIdx.x != 0u) return;
        }
    }
    if (routed_experts) {
        for (unsigned int slot = 0; slot < top_k; ++slot) {
            routed_experts[selected[slot] / expert_index_divisor] = 1;
        }
    }
    if (routing == 3) {
        float selected_maximum = -3.402823466e+38F;
        for (unsigned int slot = 0; slot < top_k; ++slot) selected_maximum = fmaxf(selected_maximum, routes[slot]);
        *selected_sum = 0.0f;
        for (unsigned int slot = 0; slot < top_k; ++slot) {
            routes[slot] = expf(routes[slot] - selected_maximum);
            *selected_sum += routes[slot];
        }
        for (unsigned int slot = 0; slot < top_k; ++slot) routes[slot] /= *selected_sum;
    }
}

__device__ __forceinline__ float moe_activation(float gate_dot, float up_dot, unsigned int gated,
        unsigned int activation, unsigned int routing, float swiglu_clamp) {
    float activated;
    if (activation == 2) {
        activated = gated ? fmaxf(gate_dot, 0.0f) * up_dot : fmaxf(up_dot, 0.0f);
    } else if (activation == 3) {
        const float source = gated ? gate_dot : up_dot;
        float gelu;
        if (source <= -10.0f) {
            gelu = 0.0f;
        } else if (source >= 10.0f) {
            gelu = source;
        } else {
            const float rounded = __half2float(__float2half_rn(source));
            const float argument = 0.7978845608028654f * rounded *
                (1.0f + 0.044715f * rounded * rounded);
            gelu = __half2float(__float2half_rn(
                0.5f * rounded * (1.0f + tanhf(argument))));
        }
        activated = gated ? gelu * up_dot : gelu;
    } else if (activation == 4) {
        const float gate_value = fminf(gate_dot, 7.0f);
        const float up_value = fminf(7.0f, fmaxf(-7.0f, up_dot));
        activated = gate_value / (1.0f + expf(-1.702f * gate_value)) * (up_value + 1.0f);
    } else if (activation == 5) {
        activated = fmaxf(up_dot, 0.0f);
        activated *= activated;
    } else if (gated) {
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
    return activated;
}

extern "C" __global__ void moe_hidden_f32(
        const float * input,
        const unsigned int * route_state,
        const void * gate,
        const void * up,
        const float * gate_bias,
        const float * up_bias,
        float * activations,
        unsigned int hidden,
        unsigned int top_k,
        unsigned int intermediate,
        unsigned int routing,
        unsigned int gate_storage,
        unsigned int up_storage,
        unsigned int gated,
        unsigned int fused_gate_up,
        unsigned int activation,
        unsigned int expert_index_divisor,
        float swiglu_clamp,
        unsigned int indexed_experts,
        unsigned int count) {
    // Strict execution transposes sector reads while preserving sum order.
    constexpr unsigned int sector_bytes = 32;
    constexpr unsigned int channels = sector_bytes / sizeof(float);
    constexpr unsigned int stride = channels + 1;
    constexpr unsigned int rows_per_load = CUDA_WARP_WIDTH / channels;
    const unsigned int tiles = (intermediate - 1) / blockDim.x + 1;
    const unsigned int selected = blockIdx.x / tiles;
    if (selected >= count / intermediate) return;
    const unsigned int inner_index = (blockIdx.x % tiles) * blockDim.x + threadIdx.x;
    const unsigned int lane = threadIdx.x % CUDA_WARP_WIDTH;
    const unsigned int warp_row = inner_index - lane;
    if (warp_row >= intermediate) return;
    const unsigned int index = selected * intermediate + inner_index;
    const unsigned int slot = selected % top_k;
    const unsigned int token = selected / top_k;
    const float * x = input + (size_t) token * hidden;
    const unsigned int expert =
        route_state[(size_t) token * (2 * top_k + 1) + slot] / expert_index_divisor;
    if (indexed_experts) {
        if (gated) gate = reinterpret_cast<const void * const *>(gate)[expert];
        up = reinterpret_cast<const void * const *>(up)[expert];
    }
    const size_t gate_base =
        (size_t) (indexed_experts ? 0 : expert) * (fused_gate_up ? 2 : 1) * intermediate * hidden;
    const size_t up_base = gate_base + (fused_gate_up ? (size_t) intermediate * hidden : 0);
    extern __shared__ float weights[];
    float * gate_tile = weights + (threadIdx.x / CUDA_WARP_WIDTH) * CUDA_WARP_WIDTH * stride;
    float * up_tile = gate_tile + blockDim.x * stride;
    float gate_dot = 0.0f;
    float up_dot = 0.0f;
    for (unsigned int base = 0; base < hidden; base += channels) {
        const unsigned int channel = base + lane % channels;
        #pragma unroll 1
        for (unsigned int row = lane / channels; row < CUDA_WARP_WIDTH; row += rows_per_load) {
            if (warp_row + row < intermediate && channel < hidden) {
                const size_t offset = (size_t) (warp_row + row) * hidden + channel;
                if (gated) gate_tile[row * stride + lane % channels] =
                    moe_expert_value(gate, gate_base + offset, gate_storage);
                up_tile[row * stride + lane % channels] =
                    moe_expert_value(up, up_base + offset, up_storage);
            }
        }
        __syncwarp();
        if (inner_index < intermediate) {
            #pragma unroll
            for (unsigned int channel = 0; channel < channels; ++channel) {
                if (base + channel < hidden) {
                    if (gated) gate_dot += x[base + channel] * gate_tile[lane * stride + channel];
                    up_dot += x[base + channel] * up_tile[lane * stride + channel];
                }
            }
        }
        __syncwarp();
    }
    if (inner_index >= intermediate) return;
    if (gate_bias) {
        gate_dot += gate_bias[(size_t) expert * intermediate + inner_index];
        up_dot += up_bias[(size_t) expert * intermediate + inner_index];
    }
    activations[index] = moe_activation(gate_dot, up_dot, gated, activation, routing, swiglu_clamp);
}

extern "C" __global__ void moe_down_f32(
        const unsigned int * route_state,
        const float * activations,
        const void * down,
        const float * expert_scale,
        const float * down_bias,
        float * output,
        unsigned int hidden,
        unsigned int top_k,
        unsigned int intermediate,
        unsigned int normalize_top_k,
        float routed_scale,
        unsigned int expert_storage,
        unsigned int expert_index_divisor,
        unsigned int indexed_experts,
        unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    output[index] = moe_down_channel(route_state, activations, down, expert_scale, down_bias, hidden, top_k,
        intermediate, normalize_top_k, routed_scale, expert_storage, expert_index_divisor, indexed_experts,
        index, 0, 1, nullptr);
}

// moe_host_slots_f32 places the slot outputs the CPU computed for experts in
// host memory, packed row after row, at their slots for moe_combine_f32.
//   - slots[positions[r] * hidden + c] = rows[r * hidden + c]
extern "C" __global__ void moe_host_slots_f32(
        const float * rows, const unsigned int * positions, float * slots, unsigned int hidden, unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const unsigned int row = index / hidden, channel = index % hidden;
    slots[(size_t) positions[row] * hidden + channel] = rows[index];
}

extern "C" __global__ void moe_combine_f32(
        const unsigned int * route_state, const float * slots, const float * expert_scale, float * output,
        unsigned int hidden, unsigned int top_k, unsigned int normalize, float scale,
        unsigned int divisor, unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const unsigned int token = index / hidden, channel = index % hidden;
    const unsigned int * selected = route_state + (size_t) token * (2 * top_k + 1);
    const float * routes = reinterpret_cast<const float *>(selected + top_k);
    float result = 0.0f;
    for (unsigned int slot = 0; slot < top_k; ++slot) {
        float route = routes[slot] * scale;
        if (normalize) route /= fmaxf(routes[top_k], 6.103515625e-5f);
        const float multiplier = expert_scale ? expert_scale[selected[slot] / divisor] : 1.0f;
        result += route * multiplier * slots[((size_t) token * top_k + slot) * hidden + channel];
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

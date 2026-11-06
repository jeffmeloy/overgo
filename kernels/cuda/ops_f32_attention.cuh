// ops_f32_attention.cuh: attention: dense, decode, online and tiled on tensor cores, with concatenation, in ops_f32.cu's order. ops_f32.cu
// includes it once; it compiles only there.

// relative_position_bucket is the bucket of relative_bias that holds a key distance tokens from its query, the key's
// position less the query's: exact under half the buckets, logarithmic out to 128 tokens beyond, and with bidirectional
// set each direction its own half.
__device__ __forceinline__ unsigned int relative_position_bucket(int distance, unsigned int buckets, unsigned int bidirectional) {
    unsigned int bucket = 0;
    if (bidirectional) {
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
    return bucket;
}

extern "C" __global__ void attention_f32(
        const float * query,
        const float * key,
        const float * value,
        const float * relative_bias,
		const float * sinks,
		const float * block_ids,
		const float * key_bias,
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
            score += relative_bias[relative_position_bucket(
                (int) key_token - ((int) query_start + (int) query_token), relative_buckets, relative_bidirectional) * query_heads + query_head];
        }
        if (softcap > 0.0f) {
            score = softcap * tanhf(score / softcap);
        }
        if (key_bias != nullptr) score += key_bias[key_token];
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
            score += relative_bias[relative_position_bucket(
                (int) key_token - ((int) query_start + (int) query_token), relative_buckets, relative_bidirectional) * query_heads + query_head];
        }
        if (softcap > 0.0f) {
            score = softcap * tanhf(score / softcap);
        }
        if (key_bias != nullptr) score += key_bias[key_token];
        const float probability = expf(score - maximum);
        const unsigned int value_offset =
            ((sequence * key_value_tokens + key_token) * key_value_heads + key_value_head) * value_width;
        sum += probability;
        weighted += probability * value[value_offset + value_channel];
    }
    output[index] = weighted / sum;
}

// key_value_token_count: device scalar so decode launches stay byte-identical
// across steps; key_value_stride is the KV capacity used for addressing and
// shared-memory partitioning. window > 0 is a causal sliding window: the
// query at position key_value_tokens-1 reads the last `window` keys, derived
// from the device token count so a windowed layer replays the same launch
// too. softcap > 0 applies the tanh score cap before the softmax.
// The keys are walked in tiles of tile_tokens with an online softmax:
// each tile's scores live in shared memory, the running maximum and sum
// are rescaled when a tile raises the maximum, and the value
// accumulation is rescaled with them, so the launch's shared memory is
// one tile and the block's partials rather than the cache capacity. The
// capacity-sized buffer put every decode past the 48 KB launch limit,
// 12k keys, on the batched GEMM path at a hundred launches per token
// with a graph instantiated every third token, and that path read the
// whole capacity as keys.
// query_tokens > 1 is a causal span (a speculative verify step): its
// queries are the last query_tokens positions, so query t reads the keys
// through its own position, the device count less the queries after it,
// and its row walks them exactly as a one-query step at that count does.
// cache_read widens one cache element to f32: an F32 cache's as it is, a BF16 cache's exactly.
__device__ __forceinline__ float cache_read(const float * cache, size_t index) {
    return cache[index];
}

__device__ __forceinline__ float cache_read(const __nv_bfloat16 * cache, size_t index) {
    return __bfloat162float(cache[index]);
}

template <typename KV>
__device__ __forceinline__ void attention_decode(
        const float * query,
        const KV * key,
        const KV * value,
        float * output,
        unsigned int key_width,
        unsigned int value_width,
        unsigned int query_heads,
        unsigned int key_value_heads,
        const unsigned int * key_value_token_count,
        unsigned int key_value_stride,
        unsigned int sequences,
        unsigned int query_tokens,
        float scale,
        unsigned int window,
        float softcap,
        unsigned int tile_tokens,
        unsigned int splits,
        float * partials) {
    // splits blocks share one (sequence, query, head) row, each walking
    // its slice of the keys; with one split the block writes the output,
    // with more it leaves its maximum, sum, and unnormalized output for
    // attention_decode_combine_f32.
    const unsigned int row = blockIdx.x / splits;
    const unsigned int split = blockIdx.x % splits;
    const unsigned int query_head = row % query_heads;
    const unsigned int query_row = row / query_heads;
    const unsigned int query_token = query_row % query_tokens;
    const unsigned int sequence = query_row / query_tokens;
    if (sequence >= sequences) {
        return;
    }
    const unsigned int key_value_tokens = key_value_token_count[0] - (query_tokens - 1 - query_token);
    const unsigned int window_first = window > 0 && key_value_tokens > window ? key_value_tokens - window : 0;
    const unsigned int span = (key_value_tokens - window_first + splits - 1) / splits;
    const unsigned int key_first = window_first + split * span;
    const unsigned int key_last = min(key_value_tokens, key_first + span);
    float * partial_row = partials + (row * splits + split) * (value_width + 2);
    if (key_first >= key_last) {
        // An empty slice contributes nothing: a minimal maximum weighs
        // it to zero in the combine.
        if (threadIdx.x == 0) {
            partial_row[0] = -3.402823466e+38F;
            partial_row[1] = 0.0f;
        }
        for (unsigned int channel = threadIdx.x; channel < value_width; channel += blockDim.x) {
            partial_row[2 + channel] = 0.0f;
        }
        return;
    }
    const unsigned int group_size = query_heads / key_value_heads;
    const unsigned int key_value_head = query_head / group_size;
    const unsigned int query_offset =
        (query_row * query_heads + query_head) * key_width;
    extern __shared__ float shared[];
    float * scores = shared;
    float * partial = shared + tile_tokens;
    // The running output lives in shared memory, one float per value
    // channel, so the value pass keeps the one-channel-per-thread loop
    // whose loads coalesce across the block.
    float * accumulator = partial + blockDim.x;
    for (unsigned int channel = threadIdx.x; channel < value_width; channel += blockDim.x) {
        accumulator[channel] = 0.0f;
    }
    float running_maximum = -3.402823466e+38F;
    float running_sum = 0.0f;
    for (unsigned int tile_first = key_first; tile_first < key_last; tile_first += tile_tokens) {
        const unsigned int tile_end = min(key_last, tile_first + tile_tokens);
        float local_maximum = -3.402823466e+38F;
        for (unsigned int token = tile_first + threadIdx.x; token < tile_end; token += blockDim.x) {
            const unsigned int key_offset =
                ((sequence * key_value_stride + token) * key_value_heads + key_value_head) * key_width;
            float dot = 0.0f;
            unsigned int channel = 0;
            // DECODE_READS_IN_FLIGHT channels are read before any is added, in channel order, so the walk waits on
            // one read in that many and keeps the serial sum's bits.
            for (; channel + DECODE_READS_IN_FLIGHT <= key_width; channel += DECODE_READS_IN_FLIGHT) {
                float keys[DECODE_READS_IN_FLIGHT];
#pragma unroll
                for (unsigned int read = 0; read < DECODE_READS_IN_FLIGHT; ++read) {
                    keys[read] = cache_read(key, key_offset + channel + read);
                }
#pragma unroll
                for (unsigned int read = 0; read < DECODE_READS_IN_FLIGHT; ++read) {
                    dot += query[query_offset + channel + read] * keys[read];
                }
            }
            for (; channel < key_width; ++channel) {
                dot += query[query_offset + channel] * cache_read(key, key_offset + channel);
            }
            float score = dot * scale;
            if (softcap > 0.0f) {
                score = softcap * tanhf(score / softcap);
            }
            scores[token - tile_first] = score;
            local_maximum = fmaxf(local_maximum, score);
        }
        const float tile_maximum = block_max_f32(local_maximum, partial);
        // Every thread has read the tile maximum before the partials are
        // reused by the sum below.
        __syncthreads();
        const float maximum = fmaxf(running_maximum, tile_maximum);
        const float correction = expf(running_maximum - maximum);
        float local_sum = 0.0f;
        for (unsigned int token = tile_first + threadIdx.x; token < tile_end; token += blockDim.x) {
            const float probability = expf(scores[token - tile_first] - maximum);
            scores[token - tile_first] = probability;
            local_sum += probability;
        }
        const float tile_sum = block_sum_f32(local_sum, partial);
        running_sum = running_sum * correction + tile_sum;
        // As in llama.cpp fattn-vec, independent thread groups accumulate
        // disjoint keys for the same value channels, then combine. Derive
        // groups from the live channel count: narrow heads no longer leave
        // most threads idle through the serial key walk. The completed
        // softmax reduction's scratch holds the partials, adding no storage.
        for (unsigned int channel_first = 0; channel_first < value_width; channel_first += blockDim.x) {
            const unsigned int value_lanes = min(value_width - channel_first, blockDim.x);
            const unsigned int value_groups = blockDim.x / value_lanes;
            const unsigned int group = threadIdx.x / value_lanes;
            const unsigned int channel = channel_first + threadIdx.x % value_lanes;
            float weighted = 0.0f;
            if (group < value_groups) {
                unsigned int token = tile_first + group;
                // DECODE_READS_IN_FLIGHT keys' values are read before any is added, in key order: one block a head
                // walked a 235-key cache in 48 us, waiting on each key's read.
                for (; token + (DECODE_READS_IN_FLIGHT - 1) * value_groups < tile_end;
                        token += DECODE_READS_IN_FLIGHT * value_groups) {
                    float values[DECODE_READS_IN_FLIGHT];
#pragma unroll
                    for (unsigned int read = 0; read < DECODE_READS_IN_FLIGHT; ++read) {
                        const unsigned int value_offset = ((sequence * key_value_stride + token + read * value_groups) *
                            key_value_heads + key_value_head) * value_width;
                        values[read] = cache_read(value, value_offset + channel);
                    }
#pragma unroll
                    for (unsigned int read = 0; read < DECODE_READS_IN_FLIGHT; ++read) {
                        weighted += scores[token + read * value_groups - tile_first] * values[read];
                    }
                }
                for (; token < tile_end; token += value_groups) {
                    const unsigned int value_offset =
                        ((sequence * key_value_stride + token) * key_value_heads + key_value_head) * value_width;
                    weighted += scores[token - tile_first] * cache_read(value, value_offset + channel);
                }
            }
            if (value_groups > 1) {
                partial[threadIdx.x] = weighted;
                __syncthreads();
                if (threadIdx.x < value_lanes) {
                    for (unsigned int other = 1; other < value_groups; ++other) {
                        weighted += partial[other * value_lanes + threadIdx.x];
                    }
                }
                // Every group has finished reading before the next chunk
                // or softmax tile can reuse the same scratch.
                __syncthreads();
            }
            if (threadIdx.x < value_lanes) {
                accumulator[channel_first + threadIdx.x] = accumulator[channel_first + threadIdx.x] * correction + weighted;
            }
        }
        running_maximum = maximum;
        // The next tile overwrites the scores every thread just read.
        __syncthreads();
    }
    if (splits == 1) {
        for (unsigned int channel = threadIdx.x; channel < value_width; channel += blockDim.x) {
            output[row * value_width + channel] = accumulator[channel] / running_sum;
        }
        return;
    }
    if (threadIdx.x == 0) {
        partial_row[0] = running_maximum;
        partial_row[1] = running_sum;
    }
    for (unsigned int channel = threadIdx.x; channel < value_width; channel += blockDim.x) {
        partial_row[2 + channel] = accumulator[channel];
    }
}

// attention_decode_f32 reads an F32 cache, attention_decode_bf16_cache_f32 a BF16 one, each key and value widened to
// f32 as it is read, so both take the same products in the same order.
extern "C" __global__ void attention_decode_f32(
        const float * query,
        const float * key,
        const float * value,
        float * output,
        unsigned int key_width,
        unsigned int value_width,
        unsigned int query_heads,
        unsigned int key_value_heads,
        const unsigned int * key_value_token_count,
        unsigned int key_value_stride,
        unsigned int sequences,
        unsigned int query_tokens,
        float scale,
        unsigned int window,
        float softcap,
        unsigned int tile_tokens,
        unsigned int splits,
        float * partials) {
    attention_decode<float>(query, key, value, output, key_width, value_width, query_heads, key_value_heads, key_value_token_count,
        key_value_stride, sequences, query_tokens, scale, window, softcap, tile_tokens, splits, partials);
}

extern "C" __global__ void attention_decode_bf16_cache_f32(
        const float * query,
        const __nv_bfloat16 * key,
        const __nv_bfloat16 * value,
        float * output,
        unsigned int key_width,
        unsigned int value_width,
        unsigned int query_heads,
        unsigned int key_value_heads,
        const unsigned int * key_value_token_count,
        unsigned int key_value_stride,
        unsigned int sequences,
        unsigned int query_tokens,
        float scale,
        unsigned int window,
        float softcap,
        unsigned int tile_tokens,
        unsigned int splits,
        float * partials) {
    attention_decode<__nv_bfloat16>(query, key, value, output, key_width, value_width, query_heads, key_value_heads, key_value_token_count,
        key_value_stride, sequences, query_tokens, scale, window, softcap, tile_tokens, splits, partials);
}

// attention_decode_combine_f32: one block per (sequence, query, head) row folds
// the split partials of attention_decode_f32 into the output, each split
// weighted by the exponential of its maximum against the row maximum
// (the flash-attention combine of the llama.cpp lineage). A row no split
// weighed, a selected attention's with no valid slot, outputs zeros.
extern "C" __global__ void attention_decode_combine_f32(
        const float * partials,
        float * output,
        unsigned int value_width,
        unsigned int splits,
        unsigned int rows) {
    const unsigned int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    const float * partial_rows = partials + row * splits * (value_width + 2);
    float maximum = -3.402823466e+38F;
    for (unsigned int split = 0; split < splits; ++split) {
        maximum = fmaxf(maximum, partial_rows[split * (value_width + 2)]);
    }
    float denominator = 0.0f;
    for (unsigned int split = 0; split < splits; ++split) {
        const float * partial_row = partial_rows + split * (value_width + 2);
        denominator += expf(partial_row[0] - maximum) * partial_row[1];
    }
    for (unsigned int channel = threadIdx.x; channel < value_width; channel += blockDim.x) {
        float numerator = 0.0f;
        for (unsigned int split = 0; split < splits; ++split) {
            const float * partial_row = partial_rows + split * (value_width + 2);
            numerator += expf(partial_row[0] - maximum) * partial_row[2 + channel];
        }
        output[row * value_width + channel] = attention_row_value(numerator, denominator);
    }
}

template <typename KV>
__device__ __forceinline__ void attention_online(
        const float * query,
        const KV * key,
        const KV * value,
        const float * relative_bias,
        const float * sinks,
        const float * block_ids,
        const float * key_bias,
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
        unsigned int relative_bidirectional) {
    unsigned int row = blockIdx.x;
    const unsigned int query_head = row % query_heads;
    row /= query_heads;
    const unsigned int query_token = row % query_tokens;
    const unsigned int sequence = row / query_tokens;
    if (sequence >= sequences || value_width > blockDim.x) return;

    const unsigned int group_size = query_heads / key_value_heads;
    const unsigned int key_value_head = query_head / group_size;
    const unsigned int query_position = query_start + query_token;
    const unsigned int causal_limit = query_position + 1;
    unsigned int key_limit = causal ? causal_limit : key_value_tokens;
    if (causal && block_ids != nullptr && block_ids[query_position] >= 0.0f) {
        key_limit = key_value_tokens;
    }
    unsigned int key_first = window > 0 && causal_limit > window ? causal_limit - window : 0;
    if (symmetric_window == 2) {
        key_first = (query_position / window) * window;
    } else if (symmetric_window) {
        const unsigned int half_window = window / 2;
        key_first = query_position > half_window ? query_position - half_window : 0;
        const unsigned int symmetric_limit = query_position + half_window + 1;
        key_limit = symmetric_limit < key_value_tokens ? symmetric_limit : key_value_tokens;
    }

    unsigned int n_head_log2 = 1;
    while (n_head_log2 * 2 <= query_heads) n_head_log2 *= 2;
    float alibi_slope = 0.0f;
    if (max_alibi_bias > 0.0f) {
        const float m0 = powf(2.0f, -max_alibi_bias / (float) n_head_log2);
        const float m1 = powf(2.0f, -(max_alibi_bias / 2.0f) / (float) n_head_log2);
        alibi_slope = query_head < n_head_log2
            ? powf(m0, (float) (query_head + 1))
            : powf(m1, (float) (2 * (query_head - n_head_log2) + 1));
    }

    __shared__ float shared_maximum;
    __shared__ float shared_sum;
    __shared__ float shared_alpha;
    __shared__ float shared_beta;
    if (threadIdx.x == 0) {
        shared_maximum = sinks != nullptr ? sinks[query_head] : -3.402823466e+38F;
        shared_sum = sinks != nullptr ? 1.0f : 0.0f;
    }
    __syncthreads();

    float weighted = 0.0f;
    const unsigned int query_offset =
        ((sequence * query_tokens + query_token) * query_heads + query_head) * key_width;
    for (unsigned int key_token = key_first; key_token < key_limit; ++key_token) {
        if (threadIdx.x == 0) {
            bool included = !(causal && key_token >= causal_limit && (block_ids == nullptr ||
                block_ids[query_position] < 0.0f ||
                block_ids[key_token] != block_ids[query_position]));
            if (!included) {
                shared_alpha = 1.0f;
                shared_beta = 0.0f;
            } else {
                const unsigned int key_offset =
                    ((sequence * key_value_tokens + key_token) * key_value_heads + key_value_head) * key_width;
                float dot = 0.0f;
                for (unsigned int channel = 0; channel < key_width; ++channel) {
                    dot += query[query_offset + channel] * cache_read(key, key_offset + channel);
                }
                float score = dot * scale;
                if (alibi_slope > 0.0f) {
                    const int distance = (int) query_position - (int) key_token;
                    score -= (float) (distance < 0 ? -distance : distance) * alibi_slope;
                }
                if (relative_bias != nullptr) {
                    score += relative_bias[relative_position_bucket(
                        (int) key_token - (int) query_position, relative_buckets, relative_bidirectional) * query_heads + query_head];
                }
                if (softcap > 0.0f) score = softcap * tanhf(score / softcap);
                if (key_bias != nullptr) score += key_bias[key_token];
                const float next_maximum = fmaxf(shared_maximum, score);
                shared_alpha = expf(shared_maximum - next_maximum);
                shared_beta = expf(score - next_maximum);
                shared_sum = shared_sum * shared_alpha + shared_beta;
                shared_maximum = next_maximum;
            }
        }
        __syncthreads();
        if (threadIdx.x < value_width) {
            const unsigned int value_offset =
                ((sequence * key_value_tokens + key_token) * key_value_heads + key_value_head) * value_width;
            weighted = weighted * shared_alpha + shared_beta * cache_read(value, value_offset + threadIdx.x);
        }
        __syncthreads();
    }
    if (threadIdx.x < value_width) {
        const unsigned int output_offset =
            ((sequence * query_tokens + query_token) * query_heads + query_head) * value_width;
        output[output_offset + threadIdx.x] = weighted / shared_sum;
    }
}

// attention_online_f32 reads an F32 cache, attention_online_bf16_cache_f32 a BF16 one, each key and value widened to f32
// as it is read.
extern "C" __global__ void attention_online_f32(
        const float * query,
        const float * key,
        const float * value,
        const float * relative_bias,
        const float * sinks,
        const float * block_ids,
        const float * key_bias,
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
        unsigned int relative_bidirectional) {
    attention_online<float>(query, key, value, relative_bias, sinks, block_ids, key_bias, output, key_width, value_width, query_heads, key_value_heads, query_tokens, key_value_tokens, sequences, scale, softcap, max_alibi_bias, causal, query_start, window, symmetric_window, relative_buckets, relative_bidirectional);
}

extern "C" __global__ void attention_online_bf16_cache_f32(
        const float * query,
        const __nv_bfloat16 * key,
        const __nv_bfloat16 * value,
        const float * relative_bias,
        const float * sinks,
        const float * block_ids,
        const float * key_bias,
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
        unsigned int relative_bidirectional) {
    attention_online<__nv_bfloat16>(query, key, value, relative_bias, sinks, block_ids, key_bias, output, key_width, value_width, query_heads, key_value_heads, query_tokens, key_value_tokens, sequences, scale, softcap, max_alibi_bias, causal, query_start, window, symmetric_window, relative_buckets, relative_bidirectional);
}

// attention_online_init_f32: zero the query-chunk output rows and reset
// the per-row online softmax stats (max=-inf, sum=0) ahead of the key-tile
// loop. Stats layout: [max rows][sum rows].
extern "C" __global__ void attention_online_init_f32(
        float * output,
        float * stats,
        unsigned int output_count,
        unsigned int stat_rows,
        unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    if (index < output_count) {
        output[index] = 0.0f;
    }
    if (index < stat_rows) {
        stats[index] = -3.402823466e+38F;
        stats[stat_rows + index] = 0.0f;
    }
}

// attention_online_softmax_f32: exact online softmax over one L2-resident
// score tile (SGEMM QK^T staging, layout [heads][chunk][key_chunk]). One
// warp per (head, row), ATTENTION_ONLINE_ROWS rows a block: update the running
// max, rescale the accumulated output row (skip the exact *1.0 identity), write
// probabilities in place for the PV accumulate, fold the tile sum into the
// running sum. The sum accumulates in FP64 in registers and reduces across the
// warp (a block a row, reducing through shared memory, took half a vision
// encoder's attention). Port of the proven external online softmax structure;
// output rows live in the interleaved [token][head][channel] layout.
// query_start is the absolute position of the chunk's first row; with
// causal set a row reads keys below its own position plus one, and a
// window keeps only the last `window` of those; a masked key writes a
// zero probability so the PV accumulate ignores it. softcap > 0 caps
// the scaled score with tanh before the bias, as the reference does.
#define ATTENTION_ONLINE_ROWS 8u
extern "C" __global__ void attention_online_softmax_f32(
        float * scores,
        float * output,
        float * stats,
        const float * key_bias,
        unsigned int key_start,
        unsigned int keys,
        unsigned int key_chunk,
        unsigned int rows,
        unsigned int chunk,
        unsigned int width,
        unsigned int heads,
        float scale,
        unsigned int query_start,
        unsigned int causal,
        unsigned int window,
        float softcap) {
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int pair = blockIdx.x * ATTENTION_ONLINE_ROWS + (threadIdx.x >> 5u);
    if (pair >= heads * rows) {
        return;
    }
    const unsigned int head = pair / rows;
    const unsigned int row = pair % rows;
    const unsigned int base = (head * chunk + row) * key_chunk;
    const unsigned int stat = head * chunk + row;
    const unsigned int stat_rows = heads * chunk;
    const unsigned int causal_limit = causal ? query_start + row + 1 : 0xFFFFFFFFu;
    const unsigned int key_first = window > 0 && causal_limit > window ? causal_limit - window : 0;
    float local_maximum = -3.402823466e+38F;
    for (unsigned int j = lane; j < keys; j += 32u) {
        const unsigned int key_token = key_start + j;
        if (key_token >= causal_limit || key_token < key_first) {
            continue;
        }
        float score = scores[base + j] * scale;
        if (softcap > 0.0f) {
            score = softcap * tanhf(score / softcap);
        }
        const float bias = key_bias != nullptr ? key_bias[key_token] : 0.0f;
        local_maximum = fmaxf(local_maximum, score + bias);
    }
    for (unsigned int offset = 16u; offset > 0u; offset >>= 1u) {
        local_maximum = fmaxf(local_maximum, __shfl_xor_sync(0xFFFFFFFFu, local_maximum, offset));
    }
    const float old = stats[stat];
    const float maximum = fmaxf(old, local_maximum);
    const float alpha = old <= -3.4028233e+38F ? 0.0f : expf(old - maximum);
    __syncwarp();
    if (lane == 0) {
        stats[stat] = maximum;
    }
    if (alpha != 1.0f) {
        float * output_row = output + (row * heads + head) * width;
        for (unsigned int channel = lane; channel < width; channel += 32u) {
            output_row[channel] *= alpha;
        }
    }
    double sum = 0.0;
    for (unsigned int j = lane; j < keys; j += 32u) {
        const unsigned int key_token = key_start + j;
        if (key_token >= causal_limit || key_token < key_first) {
            scores[base + j] = 0.0f;
            continue;
        }
        float score = scores[base + j] * scale;
        if (softcap > 0.0f) {
            score = softcap * tanhf(score / softcap);
        }
        const float bias = key_bias != nullptr ? key_bias[key_token] : 0.0f;
        const float probability = expf(score + bias - maximum);
        scores[base + j] = probability;
        sum += (double) probability;
    }
    for (unsigned int offset = 16u; offset > 0u; offset >>= 1u) {
        sum += __shfl_xor_sync(0xFFFFFFFFu, sum, offset);
    }
    if (lane == 0) {
        stats[stat_rows + stat] = stats[stat_rows + stat] * alpha + (float) sum;
    }
}

// attention_online_finalize_f32: divide the accumulated interleaved output
// rows by the final online sums.
extern "C" __global__ void attention_online_finalize_f32(
        float * output,
        const float * stats,
        unsigned int chunk,
        unsigned int width,
        unsigned int heads,
        unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const unsigned int z = index / width;
    const unsigned int head = z % heads;
    const unsigned int row = z / heads;
    const float sum = stats[heads * chunk + head * chunk + row];
    if (attention_sum_is_nan(sum)) {
        output[index] = sum;
    } else if (sum > 0.0f) {
        output[index] /= sum;
    }
}

// Fast F32 attention follows Torch 7661cd9c and CUTLASS da5e086d.
// TF32x3 products retain the source small-big, big-small, big-big order.
// Each 128-key panel has its own accumulator and source epilogue rescaling.
__device__ __forceinline__ void attention_tf32_split(float x, unsigned int &big, unsigned int &small) {
    big = __float_as_uint(x) & 0xffffe000u;
    const float residual = __fsub_rn(x, __uint_as_float(big));
    small = (__float_as_uint(residual) + 0x1000u) & 0xffffe000u;
}

__device__ __forceinline__ void attention_tf32_mma(float *c, const unsigned int *a, const unsigned int *b) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void attention_tf32_product(float *c, const float *a, const float *b) {
    unsigned int ah[4], al[4], bh[2], bl[2];
    for (unsigned int i = 0; i < 4; ++i)
        attention_tf32_split(a[i], ah[i], al[i]);
    for (unsigned int i = 0; i < 2; ++i)
        attention_tf32_split(b[i], bh[i], bl[i]);
    attention_tf32_mma(c, al, bh);
    attention_tf32_mma(c, ah, bl);
    attention_tf32_mma(c, ah, bh);
}

__device__ void attention_fast_f32(const float *query, const float *key, const float *value,
                                   const float *key_bias, float *output, unsigned int key_width,
                                   unsigned int value_width, unsigned int query_heads,
                                   unsigned int key_value_heads, unsigned int query_tokens,
                                   unsigned int key_value_tokens, unsigned int sequences, float scale) {
    const unsigned int BM = 32, BN = 128, LD = 132;
    const unsigned int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const unsigned int g = lane / 4, t = lane % 4;
    const unsigned int row0 = (warp % 2) * 16, col0 = (warp / 2) * 16;
    const unsigned int row_base = blockIdx.x * BM, head = blockIdx.y, seq = blockIdx.z;
    const unsigned int kvhead = head / (query_heads / key_value_heads);
    extern __shared__ float attention_fast_shared[];
    float *p = attention_fast_shared, *maximum = p + BM * LD, *sum = maximum + BM, *rescale = sum + BM;
    if (threadIdx.x < BM) {
        maximum[threadIdx.x] = -3.402823466e+38F;
        sum[threadIdx.x] = 0;
    }
    __syncthreads();
    for (unsigned int base = 0; base < key_value_tokens; base += BN) {
        float scores[2][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}};
        float score_stage[2][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}};
        for (unsigned int ch = 0; ch < key_width; ch += 8) {
            float a[4];
            for (unsigned int i = 0; i < 4; ++i) {
                unsigned int r = row_base + row0 + g + (i % 2) * 8, c = ch + t + (i / 2) * 4;
                a[i] = r < query_tokens && c < key_width
                           ? query[((seq * query_tokens + r) * query_heads + head) * key_width + c]
                           : 0;
            }
            for (unsigned int tile = 0; tile < 2; ++tile) {
                float b[2];
                for (unsigned int i = 0; i < 2; ++i) {
                    unsigned int r = base + col0 + tile * 8 + g, c = ch + t + i * 4;
                    b[i] =
                        r < key_value_tokens && c < key_width
                            ? key[((seq * key_value_tokens + r) * key_value_heads + kvhead) * key_width + c]
                            : 0;
                }
                attention_tf32_product(score_stage[tile], a, b);
                if (ch % 32 == 0)
                    for (unsigned int i = 0; i < 4; ++i) {
                        scores[tile][i] = __fadd_rn(scores[tile][i], score_stage[tile][i]);
                        score_stage[tile][i] = 0;
                    }
            }
        }
        for (unsigned int tile = 0; tile < 2; ++tile)
            for (unsigned int i = 0; i < 4; ++i)
                p[(row0 + g + (i / 2) * 8) * LD + col0 + tile * 8 + 2 * t + i % 2] =
                    __fadd_rn(scores[tile][i], score_stage[tile][i]);
        __syncthreads();
        {
            const unsigned int row = threadIdx.x / 16, part = threadIdx.x % 16;
            float scaled[8], mx = -3.402823466e+38F;
            for (unsigned int i = 0; i < 8; ++i) {
                unsigned int col = (part / 4) * 32 + (part % 4) * 2 + i % 2 + 8 * (i / 2);
                float s = __fmul_rn(p[row * LD + col], scale);
                if (key_bias && base + col < key_value_tokens)
                    s = __fadd_rn(s, key_bias[base + col]);
                scaled[i] = __fmul_rn(s, 1.4426950408889634074f);
                if (base + col < key_value_tokens)
                    mx = fmaxf(mx, scaled[i]);
            }
            for (unsigned int offset = 1; offset <= 8; offset *= 2)
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, offset));
            mx = fmaxf(mx, maximum[row]);
            const float alpha = __nv_exp2f(maximum[row] - mx);
            float total = 0;
            for (unsigned int i = 0; i < 8; ++i) {
                unsigned int col = (part / 4) * 32 + (part % 4) * 2 + i % 2 + 8 * (i / 2);
                float probability = base + col < key_value_tokens ? __nv_exp2f(scaled[i] - mx) : 0;
                p[row * LD + col] = probability;
                total = __fadd_rn(total, probability);
            }
            total = __fadd_rn(total, __shfl_xor_sync(0xffffffffu, total, 1));
            total = __fadd_rn(total, __shfl_xor_sync(0xffffffffu, total, 2));
            float s1 = __shfl_sync(0xffffffffu, total, (threadIdx.x & 16) + 4);
            float s2 = __shfl_sync(0xffffffffu, total, (threadIdx.x & 16) + 8);
            float s3 = __shfl_sync(0xffffffffu, total, (threadIdx.x & 16) + 12);
            if (part == 0) {
                sum[row] = __fadd_rn(__fmul_rn(sum[row], alpha), total);
                sum[row] = __fadd_rn(sum[row], s1);
                sum[row] = __fadd_rn(sum[row], s2);
                sum[row] = __fadd_rn(sum[row], s3);
                maximum[row] = mx;
                rescale[row] = alpha;
            }
        }
        __syncthreads();
        for (unsigned int out_base = 0; out_base < value_width; out_base += BN) {
            float accum[2][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}};
            float accum_stage[2][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}};
            for (unsigned int k = 0; k < BN; k += 8) {
                float a[4];
                for (unsigned int i = 0; i < 4; ++i)
                    a[i] = p[(row0 + g + (i % 2) * 8) * LD + k + t + (i / 2) * 4];
                for (unsigned int tile = 0; tile < 2; ++tile) {
                    float b[2];
                    for (unsigned int i = 0; i < 2; ++i) {
                        unsigned int r = base + k + t + i * 4, c = out_base + col0 + tile * 8 + g;
                        b[i] = r < key_value_tokens && c < value_width
                                   ? value[((seq * key_value_tokens + r) * key_value_heads + kvhead) *
                                               value_width +
                                           c]
                                   : 0;
                    }
                    attention_tf32_product(accum_stage[tile], a, b);
                    if (k % 32 == 0)
                        for (unsigned int i = 0; i < 4; ++i) {
                            accum[tile][i] = __fadd_rn(accum[tile][i], accum_stage[tile][i]);
                            accum_stage[tile][i] = 0;
                        }
                }
            }
            for (unsigned int tile = 0; tile < 2; ++tile)
                for (unsigned int i = 0; i < 4; ++i) {
                    accum[tile][i] = __fadd_rn(accum[tile][i], accum_stage[tile][i]);
                    unsigned int row = row0 + g + (i / 2) * 8, r = row_base + row,
                                 c = out_base + col0 + tile * 8 + 2 * t + i % 2;
                    if (r < query_tokens && c < value_width) {
                        unsigned int index =
                            ((seq * query_tokens + r) * query_heads + head) * value_width + c;
                        float alpha = base + BN >= key_value_tokens ? __frcp_rn(sum[row]) : 1;
                        output[index] =
                            base == 0
                                ? __fmul_rn(alpha, accum[tile][i])
                                : (base + BN >= key_value_tokens
                                       ? __fmaf_rn(alpha, accum[tile][i],
                                                   __fmul_rn(__fmul_rn(alpha, rescale[row]), output[index]))
                                       : __fmaf_rn(rescale[row], output[index], accum[tile][i]));
                    }
                }
        }
        __syncthreads();
    }
}

// Tiled exact flash-style attention for large non-causal workloads with
// optional per-key bias (no relative bias/sinks/blocks/softcap/alibi/window). One block owns a
// 32-query-row tile of one head; K/V stream through shared memory in
// 32-token tiles with an exact online softmax, so scores never touch
// global memory. Each of 8 warps owns 4 query rows; within a row, lane j
// owns tile token j for the score and broadcasts its probability for the
// value accumulation. Dynamic shared: Q tile (32*key_width) + K tile
// transposed (key_width*32) + V tile (32*value_width) floats.
extern "C" __global__ void attention_tiled_f32(
        const float * query,
        const float * key,
        const float * value,
        const float * key_bias,
        float * output,
        unsigned int key_width,
        unsigned int value_width,
        unsigned int query_heads,
        unsigned int key_value_heads,
        unsigned int query_tokens,
        unsigned int key_value_tokens,
        unsigned int sequences,
        float scale) {
    if (key_width > 128u) {
        attention_fast_f32(query, key, value, key_bias, output, key_width, value_width,
            query_heads, key_value_heads, query_tokens, key_value_tokens, sequences, scale);
        return;
    }
    const unsigned int TILE = 32u;
    const unsigned int CW = 4u; // output channels per lane (width <= 128)
    const unsigned int row_tile = blockIdx.x;
    const unsigned int query_head = blockIdx.y;
    const unsigned int sequence = blockIdx.z;
    if (sequence >= sequences || query_head >= query_heads ||
        key_width > TILE * CW || value_width > TILE * CW) {
        return;
    }
    const unsigned int group_size = query_heads / key_value_heads;
    const unsigned int key_value_head = query_head / group_size;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31u;

    extern __shared__ float shared[];
    float * q_tile = shared;                       // [TILE][key_width]
    float * k_tile = q_tile + TILE * key_width;    // [key_width][TILE] transposed
    float * v_tile = k_tile + TILE * key_width;    // [TILE][value_width]

    // Cooperative Q tile load (rows beyond the sequence zero-fill).
    const unsigned int row_base = row_tile * TILE;
    for (unsigned int index = threadIdx.x; index < TILE * key_width; index += blockDim.x) {
        const unsigned int row = index / key_width;
        const unsigned int channel = index % key_width;
        const unsigned int token = row_base + row;
        q_tile[index] = token < query_tokens
            ? query[((sequence * query_tokens + token) * query_heads + query_head) * key_width + channel]
            : 0.0f;
    }
    __syncthreads();

    // Per-warp row state: 4 rows, per-lane CW output channels each.
    float maximum[4] = {-3.402823466e+38F, -3.402823466e+38F, -3.402823466e+38F, -3.402823466e+38F};
    float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float accumulator[4][CW];
    for (unsigned int r = 0; r < 4; ++r) {
        for (unsigned int i = 0; i < CW; ++i) {
            accumulator[r][i] = 0.0f;
        }
    }

    for (unsigned int tile_base = 0; tile_base < key_value_tokens; tile_base += TILE) {
        // Cooperative K (transposed) and V tile loads.
        for (unsigned int index = threadIdx.x; index < TILE * key_width; index += blockDim.x) {
            const unsigned int token = index / key_width;
            const unsigned int channel = index % key_width;
            const unsigned int kv = tile_base + token;
            k_tile[channel * TILE + token] = kv < key_value_tokens
                ? key[((sequence * key_value_tokens + kv) * key_value_heads + key_value_head) * key_width + channel]
                : 0.0f;
        }
        for (unsigned int index = threadIdx.x; index < TILE * value_width; index += blockDim.x) {
            const unsigned int token = index / value_width;
            const unsigned int channel = index % value_width;
            const unsigned int kv = tile_base + token;
            v_tile[index] = kv < key_value_tokens
                ? value[((sequence * key_value_tokens + kv) * key_value_heads + key_value_head) * value_width + channel]
                : 0.0f;
        }
        __syncthreads();

        for (unsigned int r = 0; r < 4; ++r) {
            const unsigned int row = warp * 4u + r;
            // Lane j scores tile token j against the shared Q row.
            float score = -3.402823466e+38F;
            if (tile_base + lane < key_value_tokens) {
                float dot = 0.0f;
                const float * q_row = q_tile + row * key_width;
                const float * k_column = k_tile + lane;
                for (unsigned int channel = 0; channel < key_width; ++channel) {
                    dot += q_row[channel] * k_column[channel * TILE];
                }
                score = dot * scale + (key_bias != nullptr ? key_bias[tile_base + lane] : 0.0f);
            }
            float tile_maximum = score;
            for (unsigned int offset = 16; offset > 0; offset >>= 1) {
                tile_maximum = fmaxf(tile_maximum, __shfl_xor_sync(0xffffffffu, tile_maximum, offset));
            }
            const float next_maximum = fmaxf(maximum[r], tile_maximum);
            const float alpha = expf(maximum[r] - next_maximum);
            maximum[r] = next_maximum;
            const float probability = tile_base + lane < key_value_tokens
                ? expf(score - next_maximum)
                : 0.0f;
            float tile_sum = probability;
            for (unsigned int offset = 16; offset > 0; offset >>= 1) {
                tile_sum += __shfl_xor_sync(0xffffffffu, tile_sum, offset);
            }
            sum[r] = sum[r] * alpha + tile_sum;
            for (unsigned int i = 0; i < CW; ++i) {
                accumulator[r][i] *= alpha;
            }
            for (unsigned int j = 0; j < TILE; ++j) {
                const float p = __shfl_sync(0xffffffffu, probability, j);
                const float * v_row = v_tile + j * value_width;
                for (unsigned int i = 0; i < CW; ++i) {
                    const unsigned int channel = lane + 32u * i;
                    if (channel < value_width) {
                        accumulator[r][i] += p * v_row[channel];
                    }
                }
            }
        }
        __syncthreads();
    }

    for (unsigned int r = 0; r < 4; ++r) {
        const unsigned int row = warp * 4u + r;
        const unsigned int token = row_base + row;
        if (token >= query_tokens) {
            continue;
        }
        const unsigned int output_offset =
            ((sequence * query_tokens + token) * query_heads + query_head) * value_width;
        for (unsigned int i = 0; i < CW; ++i) {
            const unsigned int channel = lane + 32u * i;
            if (channel < value_width) {
                output[output_offset + channel] = attention_row_value(accumulator[r][i], sum[r]);
            }
        }
    }
}

// attention_bf16_stage_rows issues one 128-token panel's cp.async copies of a
// pre-packed BF16 tensor into the given shared stage (16 bytes = 8 bf16
// channels per copy; pointer pre-offset to the sequence and KV head); rows
// past the sequence zero-fill directly.
__device__ __forceinline__ void attention_bf16_stage_rows(
        const unsigned short * base,
        __nv_bfloat16 * stage,
        unsigned int tile_base,
        unsigned int key_value_tokens,
        unsigned int token_stride,
        unsigned int head_width,
        unsigned int BN) {
    const unsigned int W = 128u;
    const unsigned int LD = W + 8u;
    const unsigned int copies = BN * (W / 8u);
    for (unsigned int copy = threadIdx.x; copy < copies; copy += blockDim.x) {
        const unsigned int token = copy / (W / 8u);
        const unsigned int channel = (copy % (W / 8u)) * 8u;
        __nv_bfloat16 * destination = stage + token * LD + channel;
        const unsigned int kv = tile_base + token;
        if (kv < key_value_tokens && channel < head_width) {
            __pipeline_memcpy_async(
                destination, base + (size_t) kv * token_stride + channel, 16);
        } else {
            for (unsigned int i = 0; i < 8u; ++i) {
                destination[i] = __float2bfloat16(0.0f);
            }
        }
    }
}

// attention_ldx4 / attention_ldx4_trans / attention_mma: the ldmatrix +
// mma.sync primitives behind the BF16 flash kernel. nvcc's wmma path lowers
// generic-space fragment loads to scalar loads (no LDSM), which leaves the
// tensor pipe idle; explicit ldmatrix restores 16-bytes-per-lane fragment
// loads. mma.m16n8k16 drives the same HMMA.16816 hardware ops as wmma
// m16n16k16 (identical per-element FMA order), so numerics stay in the
// declared kernel class. Register mappings below are the architected
// sm_80+ fragment layouts, verified by a basis-product calibration probe.
__device__ __forceinline__ unsigned int attention_shared_address(const void * pointer) {
    return (unsigned int) __cvta_generic_to_shared(pointer);
}

__device__ __forceinline__ void attention_ldx4(
        unsigned int address,
        unsigned int & r0, unsigned int & r1, unsigned int & r2, unsigned int & r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(address));
}

__device__ __forceinline__ void attention_ldx4_trans(
        unsigned int address,
        unsigned int & r0, unsigned int & r1, unsigned int & r2, unsigned int & r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(address));
}

__device__ __forceinline__ void attention_mma(
        float * c,
        const unsigned int * a,
        unsigned int b0, unsigned int b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// BF16 attention retains F32 accumulators and unnormalized BF16 probabilities.
// The native memory-efficient attention contract uses forward 128-token panels.
// Scale and log2(e) multiplication retain separate F32 rounding boundaries.
// Each 32-column group uses a quad reduction before ordered row accumulation.
// One 512-thread block owns 32 query rows and 128 padded channels.
// K and V share their panel because the score product completes before V loads.
// O spills into dead Q/S storage and the shared footprint is 69504 bytes.
extern "C" __global__ void __launch_bounds__(512, 1) attention_native_bf16_f32(
        const float * query,
        const unsigned short * key,
        const unsigned short * value,
        const float * key_bias,
        float * output,
        unsigned int query_heads,
        unsigned int key_value_heads,
        unsigned int query_tokens,
        unsigned int key_value_tokens,
        unsigned int sequences,
        unsigned int head_width,
        float scale) {
    const unsigned int W = 128u;
    const unsigned int BM = 32u;
    const unsigned int BN = 128u;
    const unsigned int QLD = W + 8u;  // bf16 lds, multiple of 8
    const unsigned int KLD = W + 8u;
    const unsigned int VLD = W + 8u;
    const unsigned int SLD = BN + 4u; // f32 ld, multiple of 4
    const unsigned int PLD = BN + 8u; // bf16 ld, multiple of 8
    const unsigned int OLD = W + 4u;  // f32 ld, multiple of 4
    const unsigned int row_tile = blockIdx.x;
    const unsigned int query_head = blockIdx.y;
    const unsigned int sequence = blockIdx.z;
    if (sequence >= sequences || query_head >= query_heads) {
        return;
    }
    const unsigned int key_value_head = query_head / (query_heads / key_value_heads);
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31u;

    extern __shared__ unsigned char attention_shared[];
    __nv_bfloat16 * k_tile = (__nv_bfloat16 *) attention_shared;    // [BN][KLD]
    __nv_bfloat16 * v_tile = k_tile;                // [BN][VLD]
    unsigned char * qs_region = (unsigned char *) (v_tile + BN * VLD);
    __nv_bfloat16 * q_tile = (__nv_bfloat16 *) qs_region;           // [BM][QLD]
    float * s_tile = (float *) (q_tile + BM * QLD);                 // [BM][SLD]
    // O spill scratch (final store only) aliases Q+S: Q lives only until
    // the fragment preload and S is dead once P is written.
    float * o_tile = (float *) qs_region;                           // [BM][OLD]
    __nv_bfloat16 * p_tile = (__nv_bfloat16 *) (s_tile + BM * SLD); // [BM][PLD]
    float * alpha_row = (float *) (p_tile + BM * PLD);              // [BM]
    float * maximum_row = alpha_row + BM;                           // [BM]
    float * sum_row = maximum_row + BM;                             // [BM]

    const unsigned int row_base = row_tile * BM;
    for (unsigned int index = threadIdx.x; index < BM * W; index += blockDim.x) {
        const unsigned int row = index >> 7;
        const unsigned int channel = index & 127u;
        const unsigned int token = row_base + row;
        q_tile[row * QLD + channel] = __float2bfloat16(token < query_tokens && channel < head_width
            ? query[((sequence * query_tokens + token) * query_heads + query_head) * head_width + channel]
            : 0.0f);
    }
    for (unsigned int index = threadIdx.x; index < BM; index += blockDim.x) {
        maximum_row[index] = -3.402823466e+38F;
        sum_row[index] = 0.0f;
    }

    // Each warp owns a 16-by-16 score and output tile. Fragment g is the row
    // within the 8-row group, t = the lane's element pair.
    const unsigned int s_row0 = (warp & 1u) * 16u;
    const unsigned int s_column0 = (warp >> 1) * 16u;
    const unsigned int o_column_base = (warp >> 1) * 16u;
    const unsigned int g = lane >> 2;
    const unsigned int t = lane & 3u;
    // ldmatrix lane->address roles (16x16 operand = four 8x8 tiles).
    const unsigned int a_row = (lane & 7u) + ((lane >> 3u) & 1u) * 8u;  // tiles: row-halves first
    const unsigned int a_col = (lane >> 4u) * 8u;
    const unsigned int bk_row = (lane & 7u) + (lane >> 4u) * 8u;        // tiles: column-halves first
    const unsigned int bk_col = ((lane >> 3u) & 1u) * 8u;

    // O accumulators: two 16x8 column tiles, register-resident across the
    // whole stream; element rows are architected (g and g+8).
    float weighted[2][4];
    for (unsigned int tile = 0; tile < 2u; ++tile) {
        for (unsigned int element = 0; element < 4u; ++element) {
            weighted[tile][element] = 0.0f;
        }
    }
    __syncthreads();

    // Q fragments are panel-invariant: load the warp's score-row strip once
    // (q_tile is dead afterwards; the O spill scratch overlays it).
    unsigned int q_frag[8][4];
    for (unsigned int k = 0; k < 8u; ++k) {
        attention_ldx4(
            attention_shared_address(q_tile + (s_row0 + a_row) * QLD + k * 16u + a_col),
            q_frag[k][0], q_frag[k][1], q_frag[k][2], q_frag[k][3]);
    }
    __syncthreads();

    // K/V pointers start at the sequence and KV head.
    const unsigned int kv_token_stride = key_value_heads * head_width;
    const unsigned short * key_base = key +
        sequence * key_value_tokens * kv_token_stride + key_value_head * head_width;
    const unsigned short * value_base = value +
        sequence * key_value_tokens * kv_token_stride + key_value_head * head_width;
    for (unsigned int tile_base = 0; tile_base < key_value_tokens; tile_base += BN) {
        __syncthreads();
        attention_bf16_stage_rows(key_base, k_tile, tile_base, key_value_tokens, kv_token_stride, head_width, BN);
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();
        const __nv_bfloat16 * k_stage = k_tile;
        {
            // S tile: two n-halves of the warp's 16x16 score tile.
            float scores[2][4] = {{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}};
            const __nv_bfloat16 * k_rows = k_stage + (s_column0 + bk_row) * KLD + bk_col;
            for (unsigned int k = 0; k < 8u; ++k) {
                unsigned int b0, b1, b2, b3;
                attention_ldx4(
                    attention_shared_address(k_rows + k * 16u), b0, b1, b2, b3);
                attention_mma(scores[0], q_frag[k], b0, b1);
                attention_mma(scores[1], q_frag[k], b2, b3);
            }
            float * s_low = s_tile + (s_row0 + g) * SLD + s_column0 + 2u * t;
            float * s_high = s_tile + (s_row0 + 8u + g) * SLD + s_column0 + 2u * t;
            *(float2 *) s_low = make_float2(scores[0][0], scores[0][1]);
            *(float2 *) s_high = make_float2(scores[0][2], scores[0][3]);
            *(float2 *) (s_low + 8u) = make_float2(scores[1][0], scores[1][1]);
            *(float2 *) (s_high + 8u) = make_float2(scores[1][2], scores[1][3]);
        }
        // K is dead after scores and shares storage with this panel's V.
        __syncthreads();
        attention_bf16_stage_rows(value_base, v_tile, tile_base, key_value_tokens, kv_token_stride, head_width, BN);
        __pipeline_commit();
        // Sixteen lanes per row preserve the native four-quad reduction order.
        {
            const unsigned int row = threadIdx.x >> 4;
            const unsigned int part = threadIdx.x & 15u;
            const unsigned int remaining = key_value_tokens - tile_base;
            const unsigned int valid = remaining < BN ? remaining : BN;
            // One shared read per column: the scaled scores stay in
            // registers across the max and probability passes.
            const float log2e = 1.4426950408889634074f;
            float scaled[8];
            for (unsigned int i = 0; i < 8u; ++i) {
                const unsigned int key_token = tile_base + (part / 4u) * 32u + (part % 4u) * 2u + (i % 2u) + 8u * (i / 2u);
                const float score = __fmul_rn(s_tile[row * SLD + (part / 4u) * 32u + (part % 4u) * 2u + (i % 2u) + 8u * (i / 2u)], scale) +
                    (key_bias != nullptr && key_token < key_value_tokens ? key_bias[key_token] : 0.0f);
                scaled[i] = __fmul_rn(score, log2e);
            }
            float local_maximum = -3.402823466e+38F;
            for (unsigned int i = 0; i < 8u; ++i) {
                if ((part / 4u) * 32u + (part % 4u) * 2u + (i % 2u) + 8u * (i / 2u) < valid) {
                    local_maximum = fmaxf(local_maximum, scaled[i]);
                }
            }
            local_maximum = fmaxf(local_maximum, __shfl_xor_sync(0xffffffffu, local_maximum, 1));
            local_maximum = fmaxf(local_maximum, __shfl_xor_sync(0xffffffffu, local_maximum, 2));
            local_maximum = fmaxf(local_maximum, __shfl_xor_sync(0xffffffffu, local_maximum, 4));
            local_maximum = fmaxf(local_maximum, __shfl_xor_sync(0xffffffffu, local_maximum, 8));
            const float next_maximum = fmaxf(maximum_row[row], local_maximum);
            const float alpha = __nv_exp2f(maximum_row[row] - next_maximum);
            float local_sum = 0.0f;
            for (unsigned int i = 0; i < 8u; ++i) {
                const float probability = (part / 4u) * 32u + (part % 4u) * 2u + (i % 2u) + 8u * (i / 2u) < valid
                    ? __nv_exp2f(scaled[i] - next_maximum)
                    : 0.0f;
                p_tile[row * PLD + (part / 4u) * 32u + (part % 4u) * 2u + (i % 2u) + 8u * (i / 2u)] = __float2bfloat16(probability);
                local_sum += probability;
            }
            local_sum += __shfl_xor_sync(0xffffffffu, local_sum, 1);
            local_sum += __shfl_xor_sync(0xffffffffu, local_sum, 2);
            const float second_group = __shfl_sync(0xffffffffu, local_sum, (threadIdx.x & 16u) + 4u);
            const float third_group = __shfl_sync(0xffffffffu, local_sum, (threadIdx.x & 16u) + 8u);
            const float fourth_group = __shfl_sync(0xffffffffu, local_sum, (threadIdx.x & 16u) + 12u);
            __syncwarp();
            if (part == 0) {
                maximum_row[row] = next_maximum;
                sum_row[row] = __fmul_rn(sum_row[row], alpha) + local_sum;
                sum_row[row] += second_group;
                sum_row[row] += third_group;
                sum_row[row] += fourth_group;
                alpha_row[row] = alpha;
            }
        }
        // V, P and row statistics must be published before the value product.
        __pipeline_wait_prior(0);
        __syncthreads();

        {
            // Register-resident rescale: each accumulator element's row is
            // architected (g in the low half, g+8 in the high), so the
            // running-max correction is a plain multiply. Steady-state
            // panels carry alpha == 1 exactly and x*1.0f is exact.
            const float alpha_low = alpha_row[s_row0 + g];
            const float alpha_high = alpha_row[s_row0 + 8u + g];
            for (unsigned int tile = 0; tile < 2u; ++tile) {
                weighted[tile][0] *= alpha_low;
                weighted[tile][1] *= alpha_low;
                weighted[tile][2] *= alpha_high;
                weighted[tile][3] *= alpha_high;
            }
        }

        {
            // Value product: P fragments load once per token quarter and
            // feed both O column tiles.
            for (unsigned int k = 0; k < 8u; ++k) {
                unsigned int p_frag[4];
                attention_ldx4(
                    attention_shared_address(
                        p_tile + (s_row0 + a_row) * PLD + k * 16u + a_col),
                    p_frag[0], p_frag[1], p_frag[2], p_frag[3]);
                const __nv_bfloat16 * v_rows =
                    v_tile + (k * 16u + a_row) * VLD + o_column_base + a_col;
                unsigned int b0, b1, b2, b3;
                attention_ldx4_trans(
                    attention_shared_address(v_rows), b0, b1, b2, b3);
                attention_mma(weighted[0], p_frag, b0, b1);
                attention_mma(weighted[1], p_frag, b2, b3);

            }
        }
        // The next panel barrier protects shared K/V storage until this product completes.
    }

    for (unsigned int tile = 0; tile < 2u; ++tile) {
        const unsigned int column = o_column_base + tile * 8u + 2u * t;
        *(float2 *) (o_tile + (s_row0 + g) * OLD + column) =
            make_float2(weighted[tile][0], weighted[tile][1]);
        *(float2 *) (o_tile + (s_row0 + 8u + g) * OLD + column) =
            make_float2(weighted[tile][2], weighted[tile][3]);
    }
    __syncthreads();

    for (unsigned int index = threadIdx.x; index < BM * W; index += blockDim.x) {
        const unsigned int row = index >> 7;
        const unsigned int channel = index & 127u;
        const unsigned int token = row_base + row;
        if (token < query_tokens && channel < head_width) {
            output[((sequence * query_tokens + token) * query_heads + query_head) * head_width + channel] =
                o_tile[row * OLD + channel] * attention_row_frcp(sum_row[row]);
        }
    }
}


// Dense prompt attention on tensor cores (attention_flash_<width>_f32): a block takes 64 rows of one sequence's
// key/value head, a row being a query token and one of the group's query heads (token-major, so a group's heads
// share every key and value tile), four warps of 16 rows. Key and value tiles of ATTENTION_FLASH_KEYS(width) tokens
// copy from the f32 cache into shared memory by cp.async (the value tile during the scores, the next key tile during
// the value product); each warp keeps its queries as FP16 mma fragments, its scores and an online softmax in
// registers and its output in f32 accumulators, so no score leaves the multiprocessor. Products are m16n8k16 FP16
// with f32 accumulation, each operand split into its FP16 rounding and what that rounding left (half2_remainder): a
// score is the high parts' product plus each high part with the other's remainder, three products where one was, and
// an output likewise, so the kernel differs from the strided-batched f32 path at the remainders' rounding rather than
// the operands'. Rounded once, MiniCPM5-1B's IFEval answered 37 of 55 prompts against the f32 path's 40; split, 39,
// for 1.9 to 4.5 times the f32 path's speed at 16384 tokens. The scores scale, cap and take the key
// bias as attention_online_softmax_f32 does, and a row with no key it may read writes zeros.
#define ATTENTION_FLASH_ROWS 64u
#define ATTENTION_FLASH_THREADS 128u
#define ATTENTION_FLASH_KEY_PAD 8u
#define ATTENTION_FLASH_VALUE_PAD 4u

// attention_flash_stage copies a tile of KEYS cache rows from first into shared rows of STRIDE floats; rows past the
// keys zero-fill, so a masked key's value multiplies a zero probability by zeros. A BF16 cache's rows read eight
// channels a thread and widen as they land, so its tile is the F32 cache's of the same values; the copy waits on its
// read where the F32 copy runs beside the product.
template <unsigned int WIDTH, unsigned int KEYS, unsigned int STRIDE>
__device__ __forceinline__ void attention_flash_stage(
        const __nv_bfloat16 * cache, float * tile, unsigned int first, unsigned int keys, size_t token_stride) {
    const unsigned int copies = KEYS * (WIDTH / 8u);
    for (unsigned int copy = threadIdx.x; copy < copies; copy += ATTENTION_FLASH_THREADS) {
        const unsigned int row = copy / (WIDTH / 8u);
        const unsigned int channel = (copy % (WIDTH / 8u)) * 8u;
        float4 * destination = reinterpret_cast<float4 *>(tile + row * STRIDE + channel);
        if (first + row < keys) {
            const uint4 packed = *reinterpret_cast<const uint4 *>(cache + (size_t) (first + row) * token_stride + channel);
            const __nv_bfloat162 * pairs = reinterpret_cast<const __nv_bfloat162 *>(&packed);
            const float2 a = __bfloat1622float2(pairs[0]), b = __bfloat1622float2(pairs[1]);
            const float2 c = __bfloat1622float2(pairs[2]), d = __bfloat1622float2(pairs[3]);
            destination[0] = make_float4(a.x, a.y, b.x, b.y);
            destination[1] = make_float4(c.x, c.y, d.x, d.y);
        } else {
            destination[0] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            destination[1] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }
}

template <unsigned int WIDTH, unsigned int KEYS, unsigned int STRIDE>
__device__ __forceinline__ void attention_flash_stage(
        const float * cache, float * tile, unsigned int first, unsigned int keys, size_t token_stride) {
    const unsigned int copies = KEYS * (WIDTH / 4u);
    for (unsigned int copy = threadIdx.x; copy < copies; copy += ATTENTION_FLASH_THREADS) {
        const unsigned int row = copy / (WIDTH / 4u);
        const unsigned int channel = (copy % (WIDTH / 4u)) * 4u;
        float * destination = tile + row * STRIDE + channel;
        if (first + row < keys) {
            __pipeline_memcpy_async(destination, cache + (size_t) (first + row) * token_stride + channel, 16);
        } else {
            *reinterpret_cast<float4 *>(destination) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }
}

template <unsigned int WIDTH, unsigned int KEYS, typename KV>
__device__ __forceinline__ void attention_flash(
        const float * query,
        const KV * key,
        const KV * value,
        const float * key_bias,
        float * output,
        unsigned int query_heads,
        unsigned int key_value_heads,
        unsigned int query_tokens,
        unsigned int key_value_tokens,
        unsigned int key_capacity_tokens,
        float scale,
        unsigned int query_start,
        unsigned int causal,
        unsigned int window,
        float softcap) {
    constexpr unsigned int KEY_STRIDE = WIDTH + ATTENTION_FLASH_KEY_PAD;
    constexpr unsigned int VALUE_STRIDE = WIDTH + ATTENTION_FLASH_VALUE_PAD;
    extern __shared__ float attention_flash_shared[];
    float * key_tile = attention_flash_shared;
    float * value_tile = key_tile + KEYS * KEY_STRIDE;
    const unsigned int key_value_head = blockIdx.y;
    const unsigned int sequence = blockIdx.z;
    const unsigned int group = query_heads / key_value_heads;
    const unsigned int rows = query_tokens * group;
    const unsigned int block_row = blockIdx.x * ATTENTION_FLASH_ROWS;
    if (block_row >= rows) {
        return;
    }
    const unsigned int warp = threadIdx.x / CUDA_WARP_WIDTH;
    const unsigned int lane = threadIdx.x % CUDA_WARP_WIDTH;
    const unsigned int g = lane / 4u;
    const unsigned int t = lane % 4u;

    // A lane's two rows: g and g + 8 of its warp's 16.
    unsigned int row_token[2];
    unsigned int row_head[2];
    bool row_valid[2];
    unsigned int row_limit[2];
    unsigned int row_first[2];
    for (unsigned int half = 0; half < 2u; ++half) {
        const unsigned int row = block_row + warp * 16u + g + 8u * half;
        row_valid[half] = row < rows;
        row_token[half] = row_valid[half] ? row / group : 0u;
        row_head[half] = key_value_head * group + (row_valid[half] ? row % group : 0u);
        // A causal row reads keys below its position plus one, and a window keeps the last window of those.
        const unsigned int limit = causal ? query_start + row_token[half] + 1u : key_value_tokens;
        row_limit[half] = row_valid[half] ? min(limit, key_value_tokens) : 0u;
        row_first[half] = causal && window > 0u && limit > window ? limit - window : 0u;
    }

    // Queries as m16n8k16 A fragments, one per 16 channels.
    unsigned int queries[WIDTH / 16u][4];
    unsigned int queries_lo[WIDTH / 16u][4];
    const float * query_row[2];
    for (unsigned int half = 0; half < 2u; ++half) {
        query_row[half] = query + (((size_t) sequence * query_tokens + row_token[half]) * query_heads + row_head[half]) * WIDTH;
    }
#pragma unroll
    for (unsigned int step = 0; step < WIDTH / 16u; ++step) {
        const unsigned int channel = step * 16u + 2u * t;
        float2 low = make_float2(0.0f, 0.0f), high = low, low_far = low, high_far = low;
        if (row_valid[0]) {
            low = *reinterpret_cast<const float2 *>(query_row[0] + channel);
            low_far = *reinterpret_cast<const float2 *>(query_row[0] + channel + 8u);
        }
        if (row_valid[1]) {
            high = *reinterpret_cast<const float2 *>(query_row[1] + channel);
            high_far = *reinterpret_cast<const float2 *>(query_row[1] + channel + 8u);
        }
        queries[step][0] = half2_pack(low.x, low.y);
        queries[step][1] = half2_pack(high.x, high.y);
        queries[step][2] = half2_pack(low_far.x, low_far.y);
        queries[step][3] = half2_pack(high_far.x, high_far.y);
        queries_lo[step][0] = half2_remainder(low.x, low.y, queries[step][0]);
        queries_lo[step][1] = half2_remainder(high.x, high.y, queries[step][1]);
        queries_lo[step][2] = half2_remainder(low_far.x, low_far.y, queries[step][2]);
        queries_lo[step][3] = half2_remainder(high_far.x, high_far.y, queries[step][3]);
    }

    // The keys any row of the block reads.
    const unsigned int last_row = min(rows, block_row + ATTENTION_FLASH_ROWS) - 1u;
    const unsigned int first_limit = causal ? query_start + block_row / group + 1u : key_value_tokens;
    const unsigned int key_end = causal ? min(key_value_tokens, query_start + last_row / group + 1u) : key_value_tokens;
    const unsigned int key_begin = causal && window > 0u && first_limit > window ? first_limit - window : 0u;

    const size_t token_stride = (size_t) key_value_heads * WIDTH;
    const KV * key_cache = key + (size_t) sequence * key_capacity_tokens * token_stride + key_value_head * WIDTH;
    const KV * value_cache = value + (size_t) sequence * key_capacity_tokens * token_stride + key_value_head * WIDTH;

    float weighted[WIDTH / 8u][4];
#pragma unroll
    for (unsigned int tile = 0; tile < WIDTH / 8u; ++tile) {
        weighted[tile][0] = weighted[tile][1] = weighted[tile][2] = weighted[tile][3] = 0.0f;
    }
    float maximum[2] = {-INFINITY, -INFINITY};
    float sum[2] = {0.0f, 0.0f};

    if (key_begin < key_end) {
        attention_flash_stage<WIDTH, KEYS, KEY_STRIDE>(key_cache, key_tile, key_begin, key_end, token_stride);
        __pipeline_commit();
    }
    for (unsigned int base = key_begin; base < key_end; base += KEYS) {
        // The key tile is in, and every warp finished the last value product, so the value tile is free.
        __pipeline_wait_prior(0);
        __syncthreads();
        attention_flash_stage<WIDTH, KEYS, VALUE_STRIDE>(value_cache, value_tile, base, key_end, token_stride);
        __pipeline_commit();

        float scores[KEYS / 8u][4];
#pragma unroll
        for (unsigned int column = 0; column < KEYS / 8u; ++column) {
            scores[column][0] = scores[column][1] = scores[column][2] = scores[column][3] = 0.0f;
            const float * key_row = key_tile + (column * 8u + g) * KEY_STRIDE + 2u * t;
#pragma unroll
            for (unsigned int step = 0; step < WIDTH / 16u; ++step) {
                const float2 near = *reinterpret_cast<const float2 *>(key_row + step * 16u);
                const float2 far = *reinterpret_cast<const float2 *>(key_row + step * 16u + 8u);
                const unsigned int k0 = half2_pack(near.x, near.y), k1 = half2_pack(far.x, far.y);
                mma_f16_16x8x16(scores[column], queries[step], k0, k1);
                mma_f16_16x8x16(scores[column], queries_lo[step], k0, k1);
                mma_f16_16x8x16(scores[column], queries[step],
                    half2_remainder(near.x, near.y, k0), half2_remainder(far.x, far.y, k1));
            }
        }

        // Scale, cap, bias and mask the scores, then the running maximum of each of the lane's rows over its quad.
        float tile_maximum[2] = {-INFINITY, -INFINITY};
#pragma unroll
        for (unsigned int column = 0; column < KEYS / 8u; ++column) {
#pragma unroll
            for (unsigned int element = 0; element < 4u; ++element) {
                const unsigned int half = element / 2u;
                const unsigned int key_token = base + column * 8u + 2u * t + element % 2u;
                float score = scores[column][element] * scale;
                if (softcap > 0.0f) {
                    score = softcap * tanhf(score / softcap);
                }
                if (key_bias != nullptr && key_token < key_value_tokens) {
                    score += key_bias[key_token];
                }
                const bool attended = key_token < row_limit[half] && key_token >= row_first[half];
                scores[column][element] = attended ? score : -INFINITY;
                tile_maximum[half] = fmaxf(tile_maximum[half], scores[column][element]);
            }
        }
        float rescale[2];
#pragma unroll
        for (unsigned int half = 0; half < 2u; ++half) {
            tile_maximum[half] = fmaxf(tile_maximum[half], __shfl_xor_sync(0xffffffffu, tile_maximum[half], 1));
            tile_maximum[half] = fmaxf(tile_maximum[half], __shfl_xor_sync(0xffffffffu, tile_maximum[half], 2));
            const float next = fmaxf(maximum[half], tile_maximum[half]);
            rescale[half] = maximum[half] == -INFINITY ? 0.0f : expf(maximum[half] - next);
            maximum[half] = next;
            sum[half] *= rescale[half];
        }
#pragma unroll
        for (unsigned int column = 0; column < KEYS / 8u; ++column) {
#pragma unroll
            for (unsigned int element = 0; element < 4u; ++element) {
                const unsigned int half = element / 2u;
                const float score = scores[column][element];
                const float probability = score == -INFINITY ? 0.0f : expf(score - maximum[half]);
                scores[column][element] = probability;
                sum[half] += probability;
            }
        }
#pragma unroll
        for (unsigned int tile = 0; tile < WIDTH / 8u; ++tile) {
            weighted[tile][0] *= rescale[0];
            weighted[tile][1] *= rescale[0];
            weighted[tile][2] *= rescale[1];
            weighted[tile][3] *= rescale[1];
        }

        // The value tile is in, and every warp finished its scores, so the next key tile may load during the product.
        __pipeline_wait_prior(0);
        __syncthreads();
        if (base + KEYS < key_end) {
            attention_flash_stage<WIDTH, KEYS, KEY_STRIDE>(key_cache, key_tile, base + KEYS, key_end, token_stride);
            __pipeline_commit();
        }
#pragma unroll
        for (unsigned int step = 0; step < KEYS / 16u; ++step) {
            // Two score tiles' accumulators are the A fragment of 16 keys.
            const unsigned int probabilities[4] = {
                half2_pack(scores[2u * step][0], scores[2u * step][1]),
                half2_pack(scores[2u * step][2], scores[2u * step][3]),
                half2_pack(scores[2u * step + 1u][0], scores[2u * step + 1u][1]),
                half2_pack(scores[2u * step + 1u][2], scores[2u * step + 1u][3])};
            const float * value_rows = value_tile + (step * 16u + 2u * t) * VALUE_STRIDE + g;
            const unsigned int probabilities_lo[4] = {
                half2_remainder(scores[2u * step][0], scores[2u * step][1], probabilities[0]),
                half2_remainder(scores[2u * step][2], scores[2u * step][3], probabilities[1]),
                half2_remainder(scores[2u * step + 1u][0], scores[2u * step + 1u][1], probabilities[2]),
                half2_remainder(scores[2u * step + 1u][2], scores[2u * step + 1u][3], probabilities[3])};
#pragma unroll
            for (unsigned int tile = 0; tile < WIDTH / 8u; ++tile) {
                const float * column = value_rows + tile * 8u;
                const unsigned int v0 = half2_pack(column[0], column[VALUE_STRIDE]);
                const unsigned int v1 = half2_pack(column[8u * VALUE_STRIDE], column[9u * VALUE_STRIDE]);
                mma_f16_16x8x16(weighted[tile], probabilities, v0, v1);
                mma_f16_16x8x16(weighted[tile], probabilities_lo, v0, v1);
                mma_f16_16x8x16(weighted[tile], probabilities,
                    half2_remainder(column[0], column[VALUE_STRIDE], v0),
                    half2_remainder(column[8u * VALUE_STRIDE], column[9u * VALUE_STRIDE], v1));
            }
        }
    }

#pragma unroll
    for (unsigned int half = 0; half < 2u; ++half) {
        sum[half] += __shfl_xor_sync(0xffffffffu, sum[half], 1);
        sum[half] += __shfl_xor_sync(0xffffffffu, sum[half], 2);
    }
#pragma unroll
    for (unsigned int half = 0; half < 2u; ++half) {
        if (!row_valid[half]) {
            continue;
        }
        const float inverse = attention_row_inverse(sum[half]);
        float * out = output + (((size_t) sequence * query_tokens + row_token[half]) * query_heads + row_head[half]) * WIDTH + 2u * t;
#pragma unroll
        for (unsigned int tile = 0; tile < WIDTH / 8u; ++tile) {
            *reinterpret_cast<float2 *>(out + tile * 8u) =
                make_float2(weighted[tile][2u * half] * inverse, weighted[tile][2u * half + 1u] * inverse);
        }
    }
}

// attention_flash_64_f32, attention_flash_128_f32 and attention_flash_256_f32 take heads of 64, 128 and 256
// channels; the key tiles narrow as the heads widen, so a block's accumulators and tiles fit two blocks a
// multiprocessor.
// attention_flash_<width>_bf16_cache_f32 read a BF16 cache.
#define ATTENTION_FLASH_ARGUMENTS(KV) \
        const float * query, const KV * key, const KV * value, const float * key_bias, float * output, \
        unsigned int query_heads, unsigned int key_value_heads, unsigned int query_tokens, \
        unsigned int key_value_tokens, unsigned int key_capacity_tokens, float scale, unsigned int query_start, \
        unsigned int causal, unsigned int window, float softcap
#define ATTENTION_FLASH_FORWARD \
        query, key, value, key_bias, output, query_heads, key_value_heads, query_tokens, key_value_tokens, \
        key_capacity_tokens, scale, query_start, causal, window, softcap

extern "C" __global__ void __launch_bounds__(ATTENTION_FLASH_THREADS) attention_flash_64_f32(ATTENTION_FLASH_ARGUMENTS(float)) {
    attention_flash<64u, 64u>(ATTENTION_FLASH_FORWARD);
}

extern "C" __global__ void __launch_bounds__(ATTENTION_FLASH_THREADS) attention_flash_128_f32(ATTENTION_FLASH_ARGUMENTS(float)) {
    attention_flash<128u, 32u>(ATTENTION_FLASH_FORWARD);
}

extern "C" __global__ void __launch_bounds__(ATTENTION_FLASH_THREADS) attention_flash_256_f32(ATTENTION_FLASH_ARGUMENTS(float)) {
    attention_flash<256u, 16u>(ATTENTION_FLASH_FORWARD);
}

extern "C" __global__ void __launch_bounds__(ATTENTION_FLASH_THREADS) attention_flash_64_bf16_cache_f32(
        ATTENTION_FLASH_ARGUMENTS(__nv_bfloat16)) {
    attention_flash<64u, 64u>(ATTENTION_FLASH_FORWARD);
}

extern "C" __global__ void __launch_bounds__(ATTENTION_FLASH_THREADS) attention_flash_128_bf16_cache_f32(
        ATTENTION_FLASH_ARGUMENTS(__nv_bfloat16)) {
    attention_flash<128u, 32u>(ATTENTION_FLASH_FORWARD);
}

extern "C" __global__ void __launch_bounds__(ATTENTION_FLASH_THREADS) attention_flash_256_bf16_cache_f32(
        ATTENTION_FLASH_ARGUMENTS(__nv_bfloat16)) {
    attention_flash<256u, 16u>(ATTENTION_FLASH_FORWARD);
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

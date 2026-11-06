// ops_f32_selection.cuh: argmax and top-k selection, gathers, sparse attention and its indexer, and RWKV7, in ops_f32.cu's order. ops_f32.cu
// includes it once; it compiles only there.

// argmax_f32 writes each row's index of its largest value, the lower index on a tie, or -1 where the row holds a NaN.
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
	const auto consider = [&](float value, unsigned int candidate) {
		invalid |= isnan(value);
		if (best_index == 0xffffffffU || top_k_before(value, candidate, best_value, best_index)) {
			best_index = candidate;
			best_value = value;
		}
	};
	// The ordering is total, so the order the values are read in leaves the result as a serial scan's.
	unsigned int vectored = 0;
	if (reinterpret_cast<size_t>(input + base) % sizeof(float4) == 0) {
		vectored = width / ARGMAX_VECTOR * ARGMAX_VECTOR;
		const float4 * quads = reinterpret_cast<const float4 *>(input + base);
		for (unsigned int quad = lane; quad < vectored / ARGMAX_VECTOR; quad += blockDim.x) {
			const float4 value = quads[quad];
			const unsigned int candidate = quad * ARGMAX_VECTOR;
			consider(value.x, candidate);
			consider(value.y, candidate + 1);
			consider(value.z, candidate + 2);
			consider(value.w, candidate + 3);
		}
	}
	for (unsigned int candidate = vectored + lane; candidate < width; candidate += blockDim.x) {
		consider(input[base + candidate], candidate);
	}
	__shared__ float values[ARGMAX_THREADS];
	__shared__ unsigned int indices[ARGMAX_THREADS];
	__shared__ unsigned int invalids[ARGMAX_THREADS];
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
		unsigned int rows,
		unsigned int block_size,
		const unsigned int * first_query,
		unsigned int output_width) {
	const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row >= rows) {
		return;
	}
	const unsigned int query_start = block_size ? *first_query : 0;
	const size_t input_base = (size_t) row * width;
	const size_t output_base = (size_t) row * output_width;
	for (unsigned int slot = 0; slot < output_width; ++slot) {
		output[output_base + slot] = -1.0f;
	}
	const unsigned int visible = block_size ? min(width, (query_start + row + 1) / block_size) : width;
	for (unsigned int candidate = 0; candidate < visible; ++candidate) {
		const float candidate_value = input[input_base + candidate];
		const int worst = (int) output[output_base + k - 1];
		if (worst >= 0 && !top_k_before(candidate_value, candidate,
				input[input_base + (unsigned int) worst], (unsigned int) worst)) {
			continue;
		}
		unsigned int insertion = 0, end = k;
		while (insertion < end) {
			const unsigned int middle = insertion + (end - insertion) / 2;
			const int selected = (int) output[output_base + middle];
			if (selected < 0 || top_k_before(
					candidate_value, candidate,
					input[input_base + (unsigned int) selected], (unsigned int) selected)) {
				end = middle;
			} else {
				insertion = middle + 1;
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
	if (block_size) {
		const unsigned int blocks = min(k, visible);
		for (int block = (int) blocks - 1; block >= 0; --block) {
			const unsigned int first = (unsigned int) output[output_base + block] * block_size;
			for (unsigned int token = 0; token < block_size; ++token) {
				output[output_base + block * block_size + token] = (float) (first + token);
			}
		}
		for (unsigned int token = visible * block_size; token <= query_start + row; ++token) {
			output[output_base + blocks * block_size + token - visible * block_size] = (float) token;
		}
	}
}

#define TOP_K_BLOCK_RADIX_BITS 8
#define TOP_K_BLOCK_RADIX_BINS 256
#define TOP_K_BLOCK_KEY_BITS 64

// top_k_key orders a candidate as top_k_before does, larger first: a
// finite value before a NaN, a larger value before a smaller, and equal
// values (zero, its sign and a subnormal compare equal under the flushed
// comparison) the lower index first. The value takes the high word, the
// reversed index the low, so every candidate of a row has its own key.
__device__ __forceinline__ unsigned long long top_k_key(float value, unsigned int index) {
	unsigned int high = 0u;
	if (!isnan(value)) {
		unsigned int bits = __float_as_uint(value);
		if ((bits & 0x7fffffffu) < 0x00800000u) {
			bits = 0u;
		}
		high = (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
	}
	return ((unsigned long long) high << 32) | (unsigned long long) (0xffffffffu - index);
}

// top_k_block_f32 writes top_k_f32's output with a block a row: a radix
// select finds the row's k-th key, the keys at or above it are gathered and
// sorted larger first in shared memory, and the row's output is written
// from them, a selected block's tokens and the incomplete block's tokens
// after them as top_k_f32 writes them.
extern "C" __global__ void top_k_block_f32(
		const float * input,
		float * output,
		unsigned int width,
		unsigned int k,
		unsigned int rows,
		unsigned int block_size,
		const unsigned int * first_query,
		unsigned int output_width,
		unsigned int capacity) {
	extern __shared__ unsigned long long selected[];
	__shared__ unsigned int histogram[TOP_K_BLOCK_RADIX_BINS];
	__shared__ unsigned long long prefix;
	__shared__ unsigned int remaining;
	__shared__ unsigned int gathered;
	const unsigned int row = blockIdx.x;
	if (row >= rows) {
		return;
	}
	const unsigned int query_start = block_size ? *first_query : 0;
	const size_t input_base = (size_t) row * width;
	const size_t output_base = (size_t) row * output_width;
	const unsigned int visible = block_size ? min(width, (query_start + row + 1) / block_size) : width;
	const unsigned int chosen = min(k, visible);
	unsigned long long threshold = 0ull;
	if (visible > k) {
		if (threadIdx.x == 0) {
			prefix = 0ull;
			remaining = k;
		}
		for (int shift = TOP_K_BLOCK_KEY_BITS - TOP_K_BLOCK_RADIX_BITS; shift >= 0; shift -= TOP_K_BLOCK_RADIX_BITS) {
			for (unsigned int bin = threadIdx.x; bin < TOP_K_BLOCK_RADIX_BINS; bin += blockDim.x) {
				histogram[bin] = 0u;
			}
			__syncthreads();
			const unsigned long long resolved = shift == TOP_K_BLOCK_KEY_BITS - TOP_K_BLOCK_RADIX_BITS
				? 0ull : ~0ull << (shift + TOP_K_BLOCK_RADIX_BITS);
			const unsigned long long want = prefix;
			for (unsigned int candidate = threadIdx.x; candidate < visible; candidate += blockDim.x) {
				const unsigned long long key = top_k_key(input[input_base + candidate], candidate);
				if ((key & resolved) == want) {
					atomicAdd(&histogram[(key >> shift) & (TOP_K_BLOCK_RADIX_BINS - 1)], 1u);
				}
			}
			__syncthreads();
			if (threadIdx.x == 0) {
				unsigned int above = 0u;
				unsigned int bin = TOP_K_BLOCK_RADIX_BINS - 1;
				while (above + histogram[bin] < remaining) {
					above += histogram[bin];
					--bin;
				}
				prefix |= (unsigned long long) bin << shift;
				remaining -= above;
			}
			__syncthreads();
		}
		threshold = prefix;
	}
	if (threadIdx.x == 0) {
		gathered = 0u;
	}
	__syncthreads();
	for (unsigned int candidate = threadIdx.x; candidate < visible; candidate += blockDim.x) {
		const unsigned long long key = top_k_key(input[input_base + candidate], candidate);
		if (key >= threshold) {
			selected[atomicAdd(&gathered, 1u)] = key;
		}
	}
	unsigned int padded = 1u;
	while (padded < chosen) {
		padded <<= 1;
	}
	for (unsigned int slot = chosen + threadIdx.x; slot < padded; slot += blockDim.x) {
		selected[slot] = 0ull;
	}
	__syncthreads();
	for (unsigned int size = 2u; size <= padded; size <<= 1) {
		for (unsigned int stride = size >> 1; stride > 0u; stride >>= 1) {
			for (unsigned int slot = threadIdx.x; slot < padded; slot += blockDim.x) {
				const unsigned int partner = slot ^ stride;
				if (partner > slot) {
					const unsigned long long left = selected[slot];
					const unsigned long long right = selected[partner];
					if (((slot & size) == 0u) ? left < right : left > right) {
						selected[slot] = right;
						selected[partner] = left;
					}
				}
			}
			__syncthreads();
		}
	}
	for (unsigned int position = threadIdx.x; position < output_width; position += blockDim.x) {
		float value = -1.0f;
		if (block_size) {
			const unsigned int rank = position / block_size;
			if (rank < chosen) {
				const unsigned int block = 0xffffffffu - (unsigned int) selected[rank];
				value = (float) (block * block_size + position % block_size);
			} else if (position >= chosen * block_size) {
				const unsigned int token = visible * block_size + position - chosen * block_size;
				if (token <= query_start + row) {
					value = (float) token;
				}
			}
		} else if (position < chosen) {
			value = (float) (0xffffffffu - (unsigned int) selected[position]);
		}
		output[output_base + position] = value;
	}
}

extern "C" __global__ void top_k_pairs_f32(
		const float * partials,
		float * output,
		unsigned int candidates,
		unsigned int k,
		unsigned int rows) {
	constexpr unsigned int max_k = 64;
	constexpr unsigned int pair_values = 2;
	const unsigned int row = blockIdx.x;
	const unsigned int lane = threadIdx.x;
	if (row >= rows || k > max_k) {
		return;
	}
	const size_t input_base = (size_t) row * candidates * pair_values;
	const size_t output_base = (size_t) row * k * pair_values;
	unsigned int selected[max_k];
	__shared__ float values[256];
	__shared__ unsigned int indices[256];
	for (unsigned int slot = 0; slot < k; ++slot) {
		float best_value = 0.0f;
		unsigned int best = 0xffffffffU;
		for (unsigned int candidate = lane; candidate < candidates; candidate += blockDim.x) {
			const size_t pair = input_base + candidate * pair_values;
			const float raw = partials[pair];
			if (raw < 0.0f) {
				continue;
			}
			const unsigned int id = (unsigned int) raw;
			bool used = false;
			for (unsigned int previous = 0; previous < slot; ++previous) {
				used |= selected[previous] == id;
			}
			const float value = partials[pair + 1];
			if (!used && (best == 0xffffffffU || top_k_before(value, id, best_value, best))) {
				best = id;
				best_value = value;
			}
		}
		values[lane] = best_value;
		indices[lane] = best;
		__syncthreads();
		for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
			if (lane < stride) {
				const unsigned int right = indices[lane + stride];
				if (right != 0xffffffffU &&
					(indices[lane] == 0xffffffffU || top_k_before(
						values[lane + stride], right, values[lane], indices[lane]))) {
					values[lane] = values[lane + stride];
					indices[lane] = right;
				}
			}
			__syncthreads();
		}
		selected[slot] = indices[0];
		if (lane == 0) {
			output[output_base + slot * pair_values] = (float) indices[0];
			output[output_base + slot * pair_values + 1] = values[0];
		}
		__syncthreads();
	}
}

extern "C" __global__ void top_k_partials_f32(
		const float * input,
		float * output,
		unsigned int width,
		unsigned int k,
		unsigned int chunk,
		unsigned int chunks,
		unsigned int rows) {
	constexpr unsigned int max_k = 64;
	constexpr unsigned int pair_values = 2;
	const unsigned int index = blockIdx.x;
	const unsigned int lane = threadIdx.x;
	if (index >= chunks * rows || k > max_k) {
		return;
	}
	const unsigned int row = index / chunks;
	const unsigned int part = index % chunks;
	const unsigned int first = part * chunk;
	const unsigned int last = min(first + chunk, width);
	const size_t input_base = (size_t) row * width;
	const size_t output_base = (size_t) index * k * pair_values;
	unsigned int selected[max_k];
	for (unsigned int slot = 0; slot < k; ++slot) {
		float best_value = 0.0f;
		unsigned int best = 0xffffffffU;
		for (unsigned int candidate = first + lane; candidate < last; candidate += CUDA_WARP_WIDTH) {
			bool used = false;
			for (unsigned int previous = 0; previous < slot; ++previous) {
				used |= selected[previous] == candidate;
			}
			const float value = input[input_base + candidate];
			if (!used && (best == 0xffffffffU || top_k_before(value, candidate, best_value, best))) {
				best = candidate;
				best_value = value;
			}
		}
		for (unsigned int offset = CUDA_WARP_WIDTH / 2; offset > 0; offset >>= 1) {
			const float right_value = __shfl_down_sync(0xffffffff, best_value, offset);
			const unsigned int right = __shfl_down_sync(0xffffffff, best, offset);
			if (right != 0xffffffffU &&
				(best == 0xffffffffU || top_k_before(right_value, right, best_value, best))) {
				best = right;
				best_value = right_value;
			}
		}
		best = __shfl_sync(0xffffffff, best, 0);
		best_value = __shfl_sync(0xffffffff, best_value, 0);
		selected[slot] = best;
		if (lane == 0) {
			output[output_base + slot * pair_values] =
				best == 0xffffffffU ? -1.0f : (float) best;
			output[output_base + slot * pair_values + 1] = best_value;
		}
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
		const unsigned int * first_query,
		unsigned int count) {
	const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= count) {
		return;
	}
	const unsigned int query_start = *first_query;
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

#define SPARSE_ATTENTION_GROUPED_THREADS 256
#define SPARSE_ATTENTION_GROUPED_PAIRS 16

// sparse_attention_grouped_f32 computes sparse_attention_f32's outputs with
// the same operations in the same order, once per query head instead of once
// per value channel. A block takes one query token, one key/value head and a
// chunk of the query heads that share it; each tile of selected slots stages
// its keys and values in shared memory. A slot's dot product runs over the
// channels in order, each head's maximum over its valid slots, and each
// head's sum and each (head, channel) output accumulate over the slots in
// order.
extern "C" __global__ void sparse_attention_grouped_f32(
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
		const unsigned int * first_query,
		unsigned int chunk_heads,
		unsigned int tile) {
	extern __shared__ float shared[];
	__shared__ unsigned int valid_count;
	const unsigned int query_token = blockIdx.x;
	const unsigned int key_value_head = blockIdx.y;
	const unsigned int group = query_heads / key_value_heads;
	const unsigned int chunk_start = blockIdx.z * chunk_heads;
	if (query_token >= query_tokens || chunk_start >= group) {
		return;
	}
	const unsigned int first_head = key_value_head * group + chunk_start;
	const unsigned int heads = min(chunk_heads, group - chunk_start);
	const unsigned int query_start = *first_query;
	const unsigned int key_stride = key_width + 1;
	float * queries = shared;
	float * keys = queries + chunk_heads * key_width;
	float * values = keys + tile * key_stride;
	float * scores = values + tile * value_width;
	float * maxima = scores + chunk_heads * tile;
	float * sums = maxima + chunk_heads;
	int * rows = reinterpret_cast<int *>(sums + chunk_heads);
	const size_t index_base = (size_t) query_token * selected;
	for (unsigned int element = threadIdx.x; element < heads * key_width; element += blockDim.x) {
		const unsigned int head = element / key_width;
		queries[element] = query[((size_t) query_token * query_heads + first_head + head) * key_width + element % key_width];
	}
	for (unsigned int head = threadIdx.x; head < heads; head += blockDim.x) {
		maxima[head] = -3.402823466e+38F;
		sums[head] = 0.0f;
	}
	if (threadIdx.x == 0) {
		valid_count = 0;
	}
	// Each pass stages a tile's slots: its rows (-1 for a slot the scalar
	// kernel skips) and the rows' keys, and the second pass their values.
	for (unsigned int pass = 0; pass < 2; ++pass) {
		float weighted[SPARSE_ATTENTION_GROUPED_PAIRS];
#pragma unroll
		for (unsigned int pair = 0; pair < SPARSE_ATTENTION_GROUPED_PAIRS; ++pair) {
			weighted[pair] = 0.0f;
		}
		for (unsigned int base = 0; base < selected; base += tile) {
			const unsigned int count = min(tile, selected - base);
			__syncthreads();
			for (unsigned int slot = threadIdx.x; slot < count; slot += blockDim.x) {
				const float raw = indices[index_base + base + slot];
				int row = -1;
				if (isfinite(raw) && raw >= 0.0f && raw < (float) key_value_tokens) {
					const unsigned int key_token = (unsigned int) raw;
					if (!(raw != (float) key_token || causal && key_token > query_start + query_token)) {
						row = (int) key_token;
					}
				}
				rows[slot] = row;
			}
			__syncthreads();
			for (unsigned int element = threadIdx.x; element < count * key_width; element += blockDim.x) {
				const unsigned int slot = element / key_width;
				if (rows[slot] >= 0) {
					keys[slot * key_stride + element % key_width] =
						key[((size_t) rows[slot] * key_value_heads + key_value_head) * key_width + element % key_width];
				}
			}
			if (pass == 1) {
				for (unsigned int element = threadIdx.x; element < count * value_width; element += blockDim.x) {
					const unsigned int slot = element / value_width;
					if (rows[slot] >= 0) {
						values[element] =
							value[((size_t) rows[slot] * key_value_heads + key_value_head) * value_width + element % value_width];
					}
				}
			}
			__syncthreads();
			for (unsigned int item = threadIdx.x; item < heads * count; item += blockDim.x) {
				const unsigned int head = item / count;
				const unsigned int slot = item % count;
				if (rows[slot] < 0) {
					continue;
				}
				const float * head_query = queries + head * key_width;
				const float * slot_key = keys + slot * key_stride;
				float dot = 0.0f;
				for (unsigned int channel = 0; channel < key_width; ++channel) {
					dot += head_query[channel] * slot_key[channel];
				}
				if (pass == 0) {
					scores[head * tile + slot] = dot * scale;
				} else {
					// The per-channel kernel's code generator fuses this
					// product and difference; an explicit fused multiply-add
					// keeps the same rounding whatever this kernel's
					// generator would choose.
					scores[head * tile + slot] = expf(__fmaf_rn(dot, scale, -maxima[head]));
				}
			}
			__syncthreads();
			for (unsigned int head = threadIdx.x; head < heads; head += blockDim.x) {
				for (unsigned int slot = 0; slot < count; ++slot) {
					if (rows[slot] < 0) {
						continue;
					}
					if (pass == 0) {
						maxima[head] = fmaxf(maxima[head], scores[head * tile + slot]);
					} else {
						sums[head] += scores[head * tile + slot];
					}
				}
			}
			if (pass == 0 && threadIdx.x == 0) {
				for (unsigned int slot = 0; slot < count; ++slot) {
					valid_count += rows[slot] >= 0;
				}
			}
			if (pass == 1) {
#pragma unroll
				for (unsigned int pair = 0; pair < SPARSE_ATTENTION_GROUPED_PAIRS; ++pair) {
					const unsigned int item = threadIdx.x + pair * blockDim.x;
					if (item >= heads * value_width) {
						continue;
					}
					const unsigned int head = item / value_width;
					const unsigned int channel = item % value_width;
					for (unsigned int slot = 0; slot < count; ++slot) {
						if (rows[slot] < 0) {
							continue;
						}
						weighted[pair] += scores[head * tile + slot] * values[slot * value_width + channel];
					}
				}
			}
		}
		__syncthreads();
		if (pass == 0 && valid_count == 0) {
			for (unsigned int item = threadIdx.x; item < heads * value_width; item += blockDim.x) {
				output[((size_t) query_token * query_heads + first_head + item / value_width) * value_width + item % value_width] = 0.0f;
			}
			return;
		}
		if (pass == 1) {
#pragma unroll
			for (unsigned int pair = 0; pair < SPARSE_ATTENTION_GROUPED_PAIRS; ++pair) {
				const unsigned int item = threadIdx.x + pair * blockDim.x;
				if (item < heads * value_width) {
					const unsigned int head = item / value_width;
					output[((size_t) query_token * query_heads + first_head + head) * value_width + item % value_width] =
						weighted[pair] / sums[head];
				}
			}
		}
	}
}

#define SPARSE_ATTENTION_SPLIT_THREADS 256
#define SPARSE_ATTENTION_SPLIT_HEADS 16
#define SPARSE_ATTENTION_SPLIT_KEY_LANES 8

// sparse_attention_split_f32 computes sparse_attention_f32's outputs for a
// step of few query tokens, a decode step's: the selected slots split into
// spans of span slots across gridDim.z, and a block takes one query token, one
// key/value head and one span. A warp scores a slot for every query head that
// shares the key/value head, its lanes splitting the channels; each head's
// maximum and sum reduce over the span's valid slots; and a thread a value
// channel weighs the span's values for every head. With one span the block
// writes the outputs; with more, each head's maximum, sum and weighted values
// as attention_decode_f32's partials, which attention_decode_combine_f32
// folds. Sums run in another order than the per-channel kernel's, so outputs
// differ at rounding level. A head with no valid slot outputs zeros.
//   - at most SPARSE_ATTENTION_SPLIT_HEADS query heads a key/value head, key
//     and value widths at most SPARSE_ATTENTION_SPLIT_THREADS
//   - scores[h * span + s] = slot s's scaled dot product for head h, then its
//     exponential against the head's maximum; a slot the per-channel kernel
//     skips weighs zero
extern "C" __global__ void __launch_bounds__(SPARSE_ATTENTION_SPLIT_THREADS) sparse_attention_split_f32(
		const float * query,
		const float * key,
		const float * value,
		const float * indices,
		float * output,
		float * partials,
		unsigned int key_width,
		unsigned int value_width,
		unsigned int query_heads,
		unsigned int key_value_heads,
		unsigned int query_tokens,
		unsigned int key_value_tokens,
		unsigned int selected,
		float scale,
		unsigned int causal,
		const unsigned int * first_query,
		unsigned int span,
		unsigned int splits) {
	extern __shared__ float shared[];
	__shared__ float maxima[SPARSE_ATTENTION_SPLIT_HEADS];
	__shared__ float sums[SPARSE_ATTENTION_SPLIT_HEADS];
	const unsigned int query_token = blockIdx.x;
	const unsigned int key_value_head = blockIdx.y;
	const unsigned int split = blockIdx.z;
	if (query_token >= query_tokens) {
		return;
	}
	const unsigned int group = query_heads / key_value_heads;
	const unsigned int first_head = key_value_head * group;
	const unsigned int first_slot = split * span;
	const unsigned int slots = first_slot < selected ? min(span, selected - first_slot) : 0;
	const unsigned int query_start = *first_query;
	const unsigned int lane = threadIdx.x % CUDA_WARP_WIDTH;
	const unsigned int warp = threadIdx.x / CUDA_WARP_WIDTH;
	const unsigned int warps = blockDim.x / CUDA_WARP_WIDTH;
	float * queries = shared;
	float * scores = queries + group * key_width;
	int * rows = reinterpret_cast<int *>(scores + group * span);
	for (unsigned int element = threadIdx.x; element < group * key_width; element += blockDim.x) {
		queries[element] = query[((size_t) query_token * query_heads + first_head) * key_width + element];
	}
	for (unsigned int slot = threadIdx.x; slot < slots; slot += blockDim.x) {
		const float raw = indices[(size_t) query_token * selected + first_slot + slot];
		int row = -1;
		if (isfinite(raw) && raw >= 0.0f && raw < (float) key_value_tokens) {
			const unsigned int key_token = (unsigned int) raw;
			if (!(raw != (float) key_token || causal && key_token > query_start + query_token)) {
				row = (int) key_token;
			}
		}
		rows[slot] = row;
	}
	__syncthreads();
	for (unsigned int slot = warp; slot < slots; slot += warps) {
		const int row = rows[slot];
		if (row < 0) {
			continue;
		}
		const float * slot_key = key + ((size_t) row * key_value_heads + key_value_head) * key_width;
		float keys[SPARSE_ATTENTION_SPLIT_KEY_LANES];
#pragma unroll
		for (unsigned int part = 0; part < SPARSE_ATTENTION_SPLIT_KEY_LANES; ++part) {
			const unsigned int channel = part * CUDA_WARP_WIDTH + lane;
			keys[part] = channel < key_width ? slot_key[channel] : 0.0f;
		}
		for (unsigned int head = 0; head < group; ++head) {
			float dot = 0.0f;
#pragma unroll
			for (unsigned int part = 0; part < SPARSE_ATTENTION_SPLIT_KEY_LANES; ++part) {
				const unsigned int channel = part * CUDA_WARP_WIDTH + lane;
				if (channel < key_width) {
					dot += queries[head * key_width + channel] * keys[part];
				}
			}
			dot = warp_sum_down(dot);
			if (lane == 0) {
				scores[head * span + slot] = dot * scale;
			}
		}
	}
	__syncthreads();
	for (unsigned int head = warp; head < group; head += warps) {
		float maximum = -3.402823466e+38F;
		for (unsigned int slot = lane; slot < slots; slot += CUDA_WARP_WIDTH) {
			if (rows[slot] >= 0) {
				maximum = fmaxf(maximum, scores[head * span + slot]);
			}
		}
		for (unsigned int offset = CUDA_WARP_WIDTH / 2; offset > 0; offset /= 2) {
			maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, offset));
		}
		float sum = 0.0f;
		for (unsigned int slot = lane; slot < slots; slot += CUDA_WARP_WIDTH) {
			const float probability = rows[slot] >= 0 ? expf(scores[head * span + slot] - maximum) : 0.0f;
			scores[head * span + slot] = probability;
			sum += probability;
		}
		sum = warp_sum_down(sum);
		if (lane == 0) {
			maxima[head] = maximum;
			sums[head] = sum;
		}
	}
	__syncthreads();
	for (unsigned int channel = threadIdx.x; channel < value_width; channel += blockDim.x) {
		float weighted[SPARSE_ATTENTION_SPLIT_HEADS] = {};
		for (unsigned int slot = 0; slot < slots; ++slot) {
			const int row = rows[slot];
			if (row < 0) {
				continue;
			}
			const float slot_value = value[((size_t) row * key_value_heads + key_value_head) * value_width + channel];
#pragma unroll
			for (unsigned int head = 0; head < SPARSE_ATTENTION_SPLIT_HEADS; ++head) {
				if (head < group) {
					weighted[head] += scores[head * span + slot] * slot_value;
				}
			}
		}
#pragma unroll
		for (unsigned int head = 0; head < SPARSE_ATTENTION_SPLIT_HEADS; ++head) {
			if (head >= group) {
				continue;
			}
			const size_t row = (size_t) query_token * query_heads + first_head + head;
			if (splits == 1) {
				output[row * value_width + channel] = attention_row_value(weighted[head], sums[head]);
				continue;
			}
			float * partial = partials + (row * splits + split) * (value_width + 2);
			partial[2 + channel] = weighted[head];
			if (channel == 0) {
				partial[0] = maxima[head];
				partial[1] = sums[head];
			}
		}
	}
}

#define SPARSE_ATTENTION_MMA_WIDTH 256
#define SPARSE_ATTENTION_MMA_ROWS 16
#define SPARSE_ATTENTION_MMA_CELLS 32
#define SPARSE_ATTENTION_MMA_THREADS 128
#define SPARSE_ATTENTION_MMA_Q_STRIDE (SPARSE_ATTENTION_MMA_WIDTH + 4)
#define SPARSE_ATTENTION_MMA_S_STRIDE (SPARSE_ATTENTION_MMA_CELLS + 4)

// half2_pack rounds two f32 values to an FP16 pair, the first in the low half, as an mma fragment register holds them.
__device__ __forceinline__ unsigned int half2_pack(float low, float high) {
	const __half2 packed = __floats2half2_rn(low, high);
	return *reinterpret_cast<const unsigned int *>(&packed);
}

// half2_remainder packs what FP16 rounding left of two values, given their packed rounding.
__device__ __forceinline__ unsigned int half2_remainder(float low, float high, unsigned int packed) {
	const __half2 rounded = *reinterpret_cast<const __half2 *>(&packed);
	return half2_pack(low - __low2float(rounded), high - __high2float(rounded));
}

// mma_f16_16x8x16 multiplies FP16 fragments on tensor cores and accumulates in f32.
__device__ __forceinline__ void mma_f16_16x8x16(float * c, const unsigned int * a, unsigned int b0, unsigned int b1) {
	asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
		"{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
		: "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
		: "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// sparse_attention_mma_f32 computes a prompt's selected attention on tensor
// cores for 256-channel heads, at most 16 query heads to a key/value head: a
// block takes a query token and a key/value head, its query heads as the 16
// rows of m16n8k16 products (absent rows zero), and walks the selected slots
// 32 at a time with an online softmax. q.k and p.v are FP16 products with f32
// accumulation over the keys and values the launcher converts to FP16 (half the
// bytes each query token gathers); queries and probabilities round to FP16 as
// their fragments load, as attention_flash_<width>_f32's do, so the result
// differs from sparse_attention_grouped_f32 at FP16 rounding of the operands. A
// query with no valid slot writes zeros.
//   - each lane reads the keys and values of its mma fragments from the FP16
//     rows itself, so the shared memory holds the queries and the scores alone
//     and several blocks share a multiprocessor
extern "C" __global__ void __launch_bounds__(SPARSE_ATTENTION_MMA_THREADS) sparse_attention_mma_f32(
		const float * query,
		const __half * key,
		const __half * value,
		const float * indices,
		float * output,
		unsigned int query_heads,
		unsigned int key_value_heads,
		unsigned int query_tokens,
		unsigned int key_value_tokens,
		unsigned int selected,
		float scale,
		unsigned int causal,
		const unsigned int * first_query) {
	extern __shared__ float shared[];
	float * queries = shared;
	float * scores = queries + SPARSE_ATTENTION_MMA_ROWS * SPARSE_ATTENTION_MMA_Q_STRIDE;
	float * maxima = scores + SPARSE_ATTENTION_MMA_ROWS * SPARSE_ATTENTION_MMA_S_STRIDE;
	float * sums = maxima + SPARSE_ATTENTION_MMA_ROWS;
	float * rescale = sums + SPARSE_ATTENTION_MMA_ROWS;
	int * rows = reinterpret_cast<int *>(rescale + SPARSE_ATTENTION_MMA_ROWS);
	const unsigned int query_token = blockIdx.x;
	const unsigned int key_value_head = blockIdx.y;
	if (query_token >= query_tokens) {
		return;
	}
	const unsigned int group = query_heads / key_value_heads;
	const unsigned int first_head = key_value_head * group;
	const unsigned int query_start = *first_query;
	const unsigned int lane = threadIdx.x % CUDA_WARP_WIDTH;
	const unsigned int warp = threadIdx.x / CUDA_WARP_WIDTH;
	const unsigned int row = lane / 4;
	const unsigned int column = lane % 4;
	for (unsigned int element = threadIdx.x; element < SPARSE_ATTENTION_MMA_ROWS * SPARSE_ATTENTION_MMA_WIDTH; element += blockDim.x) {
		const unsigned int head = element / SPARSE_ATTENTION_MMA_WIDTH;
		const unsigned int channel = element % SPARSE_ATTENTION_MMA_WIDTH;
		queries[head * SPARSE_ATTENTION_MMA_Q_STRIDE + channel] = head < group
			? query[((size_t) query_token * query_heads + first_head + head) * SPARSE_ATTENTION_MMA_WIDTH + channel]
			: 0.0f;
	}
	if (threadIdx.x < SPARSE_ATTENTION_MMA_ROWS) {
		maxima[threadIdx.x] = -INFINITY;
		sums[threadIdx.x] = 0.0f;
	}
	float accumulated[8][4];
#pragma unroll
	for (int tile = 0; tile < 8; ++tile) {
		accumulated[tile][0] = accumulated[tile][1] = accumulated[tile][2] = accumulated[tile][3] = 0.0f;
	}
	const size_t index_base = (size_t) query_token * selected;
	for (unsigned int base = 0; base < selected; base += SPARSE_ATTENTION_MMA_CELLS) {
		const unsigned int count = min((unsigned int) SPARSE_ATTENTION_MMA_CELLS, selected - base);
		__syncthreads();
		if (threadIdx.x < SPARSE_ATTENTION_MMA_CELLS) {
			int slot_row = -1;
			if (threadIdx.x < count) {
				const float raw = indices[index_base + base + threadIdx.x];
				if (isfinite(raw) && raw >= 0.0f && raw < (float) key_value_tokens) {
					const unsigned int key_token = (unsigned int) raw;
					if (!(raw != (float) key_token || causal && key_token > query_start + query_token)) {
						slot_row = (int) key_token;
					}
				}
			}
			rows[threadIdx.x] = slot_row;
		}
		__syncthreads();
		// A cell's key and value rows in the cache, or none for a cell that is not attended, whose weights read as zeros.
		const auto cell_row = [&](const __half * cache, unsigned int cell) -> const __half * {
			return rows[cell] >= 0
				? cache + ((size_t) rows[cell] * key_value_heads + key_value_head) * SPARSE_ATTENTION_MMA_WIDTH
				: nullptr;
		};
		// A warp scores its 8 cells for all 16 rows.
		float score[4] = {0.0f, 0.0f, 0.0f, 0.0f};
		const __half * key_row = cell_row(key, 8 * warp + row);
#pragma unroll 4
		for (unsigned int step = 0; step < SPARSE_ATTENTION_MMA_WIDTH / 16; ++step) {
			const unsigned int k = step * 16 + 2 * column;
			const float2 low = *reinterpret_cast<const float2 *>(queries + row * SPARSE_ATTENTION_MMA_Q_STRIDE + k);
			const float2 high = *reinterpret_cast<const float2 *>(queries + (row + 8) * SPARSE_ATTENTION_MMA_Q_STRIDE + k);
			const float2 low_far = *reinterpret_cast<const float2 *>(queries + row * SPARSE_ATTENTION_MMA_Q_STRIDE + k + 8);
			const float2 high_far = *reinterpret_cast<const float2 *>(queries + (row + 8) * SPARSE_ATTENTION_MMA_Q_STRIDE + k + 8);
			const unsigned int a[4] = {
				half2_pack(low.x, low.y), half2_pack(high.x, high.y), half2_pack(low_far.x, low_far.y), half2_pack(high_far.x, high_far.y)};
			const unsigned int near_key = key_row ? __ldg(reinterpret_cast<const unsigned int *>(key_row + k)) : 0u;
			const unsigned int far_key = key_row ? __ldg(reinterpret_cast<const unsigned int *>(key_row + k + 8)) : 0u;
			mma_f16_16x8x16(score, a, near_key, far_key);
		}
		const unsigned int cell_column = 8 * warp + 2 * column;
		scores[row * SPARSE_ATTENTION_MMA_S_STRIDE + cell_column] = score[0] * scale;
		scores[row * SPARSE_ATTENTION_MMA_S_STRIDE + cell_column + 1] = score[1] * scale;
		scores[(row + 8) * SPARSE_ATTENTION_MMA_S_STRIDE + cell_column] = score[2] * scale;
		scores[(row + 8) * SPARSE_ATTENTION_MMA_S_STRIDE + cell_column + 1] = score[3] * scale;
		__syncthreads();
		// A thread a row: the running maximum, the rescale of what came before, the chunk's weights.
		if (threadIdx.x < SPARSE_ATTENTION_MMA_ROWS) {
			float * row_scores = scores + threadIdx.x * SPARSE_ATTENTION_MMA_S_STRIDE;
			const float previous = maxima[threadIdx.x];
			float maximum = previous;
			for (unsigned int cell = 0; cell < count; ++cell) {
				if (rows[cell] >= 0) {
					maximum = fmaxf(maximum, row_scores[cell]);
				}
			}
			float sum = 0.0f;
			for (unsigned int cell = 0; cell < SPARSE_ATTENTION_MMA_CELLS; ++cell) {
				const float weight = cell < count && rows[cell] >= 0 ? expf(row_scores[cell] - maximum) : 0.0f;
				row_scores[cell] = weight;
				sum += weight;
			}
			const float factor = maximum == -INFINITY ? 0.0f : expf(previous - maximum);
			sums[threadIdx.x] = sums[threadIdx.x] * factor + sum;
			maxima[threadIdx.x] = maximum;
			rescale[threadIdx.x] = factor;
		}
		__syncthreads();
		// A warp accumulates 64 output channels for all 16 rows.
		const float low_factor = rescale[row];
		const float high_factor = rescale[row + 8];
#pragma unroll
		for (int tile = 0; tile < 8; ++tile) {
			accumulated[tile][0] *= low_factor;
			accumulated[tile][1] *= low_factor;
			accumulated[tile][2] *= high_factor;
			accumulated[tile][3] *= high_factor;
		}
#pragma unroll
		for (unsigned int step = 0; step < SPARSE_ATTENTION_MMA_CELLS / 16; ++step) {
			const unsigned int k = step * 16 + 2 * column;
			const float2 low = *reinterpret_cast<const float2 *>(scores + row * SPARSE_ATTENTION_MMA_S_STRIDE + k);
			const float2 high = *reinterpret_cast<const float2 *>(scores + (row + 8) * SPARSE_ATTENTION_MMA_S_STRIDE + k);
			const float2 low_far = *reinterpret_cast<const float2 *>(scores + row * SPARSE_ATTENTION_MMA_S_STRIDE + k + 8);
			const float2 high_far = *reinterpret_cast<const float2 *>(scores + (row + 8) * SPARSE_ATTENTION_MMA_S_STRIDE + k + 8);
			const unsigned int a[4] = {
				half2_pack(low.x, low.y), half2_pack(high.x, high.y), half2_pack(low_far.x, low_far.y), half2_pack(high_far.x, high_far.y)};
			// The four cells a lane's B fragment reads: k and k + 1, and the two eight further on.
			const __half * cells[4] = {cell_row(value, k), cell_row(value, k + 1), cell_row(value, k + 8), cell_row(value, k + 9)};
#pragma unroll
			for (int tile = 0; tile < 8; ++tile) {
				const unsigned int channel = 64 * warp + 8 * tile + row;
				unsigned short v[4];
#pragma unroll
				for (int cell = 0; cell < 4; ++cell) {
					v[cell] = cells[cell] ? __ldg(reinterpret_cast<const unsigned short *>(cells[cell] + channel)) : 0u;
				}
				mma_f16_16x8x16(accumulated[tile], a, v[0] | (unsigned int) v[1] << 16, v[2] | (unsigned int) v[3] << 16);
			}
		}
	}
	__syncthreads();
#pragma unroll
	for (int tile = 0; tile < 8; ++tile) {
		const unsigned int channel = 64 * warp + 8 * tile + 2 * column;
#pragma unroll
		for (int half = 0; half < 2; ++half) {
			const unsigned int head = row + 8 * half;
			if (head < group) {
				const float total = sums[head];
				const float inverse = attention_row_inverse(total);
				float * out = output + ((size_t) query_token * query_heads + first_head + head) * SPARSE_ATTENTION_MMA_WIDTH + channel;
				out[0] = accumulated[tile][2 * half] * inverse;
				out[1] = accumulated[tile][2 * half + 1] * inverse;
			}
		}
	}
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

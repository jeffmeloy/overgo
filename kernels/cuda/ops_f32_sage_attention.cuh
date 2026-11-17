// 8-bit attention for head width 128 on sm_89 (eight-bit-attention/kernel), after SageAttention2's algorithm (arXiv
// 2411.10958, Apache-2.0): Q and K quantised to INT8 per 64-token block, K first smoothed by its mean over the tokens,
// which shifts each query's scores alike and leaves its softmax as it was; Q.K on INT8 tensor cores
// (mma m16n8k32 s8.s8.s32); the softmax online in F32; P scaled by 448 into FP8 E4M3 and V into FP8 E4M3 per channel,
// P.V on FP8 tensor cores (mma m16n8k32 e4m3.e4m3.f32), each key tile's product added to an F32 output. Tensors are
// interleaved [sequence][token][head][channel], query heads sharing a key/value head in groups (query_heads /
// key_value_heads), and a key's bias is added to its scaled score, as attention_native_bf16_f32 adds it.
//
// Six launches: sage_block_stats_f32 and sage_channel_stats_f32 for K's means and V's scales, sage_quantize_i8_f32 for
// Q and for K, sage_quantize_v_fp8_f32, then attention_sage_128_f32. The quantised operands, their scales and the
// statistics are the launches' staging.

#define SAGE_WIDTH 128u
#define SAGE_BLOCK 64u
// Shared row strides in bytes, padded so a quad's fragment loads fall in distinct banks.
#define SAGE_KLD 144u
#define SAGE_VLD 80u

__device__ __forceinline__ void sage_mma_s8(int * c, const uint32_t * a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void sage_mma_e4m3(float * c, const uint32_t * a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// sage_fp8x4 packs four floats into FP8 E4M3, the first in the lowest byte.
__device__ __forceinline__ uint32_t sage_fp8x4(float x0, float x1, float x2, float x3) {
    const uint32_t lo = __nv_cvt_float2_to_fp8x2(make_float2(x0, x1), __NV_SATFINITE, __NV_E4M3);
    const uint32_t hi = __nv_cvt_float2_to_fp8x2(make_float2(x2, x3), __NV_SATFINITE, __NV_E4M3);
    return lo | (hi << 16);
}

// sage_block_stats_f32: over each 64-token block of each key/value head, each channel's sum of K and absolute maximum
// of V. Grid (blocks, heads, sequences), 128 threads, a channel each; both [sequence][head][block][channel].
extern "C" __global__ void sage_block_stats_f32(
        const float * key, const float * value, float * key_sums, float * value_maxima,
        unsigned int heads, unsigned int tokens, unsigned int sequences) {
    const unsigned int block = blockIdx.x, head = blockIdx.y, sequence = blockIdx.z, channel = threadIdx.x;
    const unsigned int blocks = (tokens + SAGE_BLOCK - 1u) / SAGE_BLOCK;
    if (block >= blocks || head >= heads || sequence >= sequences || channel >= SAGE_WIDTH) {
        return;
    }
    const unsigned int first = block * SAGE_BLOCK, last = min(first + SAGE_BLOCK, tokens);
    float sum = 0.0f, maximum = 0.0f;
    for (unsigned int token = first; token < last; ++token) {
        const unsigned int at = ((sequence * tokens + token) * heads + head) * SAGE_WIDTH + channel;
        sum += key[at];
        maximum = fmaxf(maximum, fabsf(value[at]));
    }
    const unsigned int out = ((sequence * heads + head) * blocks + block) * SAGE_WIDTH + channel;
    key_sums[out] = sum;
    value_maxima[out] = maximum;
}

// sage_channel_stats_f32: each channel's mean of K over a sequence's tokens and V's FP8 scale, its absolute maximum
// over 448, from the blocks' statistics in block order. Grid (heads, sequences), 128 threads, a channel each; both
// [sequence][head][channel].
extern "C" __global__ void sage_channel_stats_f32(
        const float * key_sums, const float * value_maxima, float * key_mean, float * value_scales,
        unsigned int heads, unsigned int tokens, unsigned int sequences) {
    const unsigned int head = blockIdx.x, sequence = blockIdx.y, channel = threadIdx.x;
    if (head >= heads || sequence >= sequences || channel >= SAGE_WIDTH) {
        return;
    }
    const unsigned int blocks = (tokens + SAGE_BLOCK - 1u) / SAGE_BLOCK;
    float sum = 0.0f, maximum = 0.0f;
    for (unsigned int block = 0; block < blocks; ++block) {
        const unsigned int at = ((sequence * heads + head) * blocks + block) * SAGE_WIDTH + channel;
        sum += key_sums[at];
        maximum = fmaxf(maximum, value_maxima[at]);
    }
    const unsigned int out = (sequence * heads + head) * SAGE_WIDTH + channel;
    key_mean[out] = tokens == 0 ? 0.0f : sum / (float) tokens;
    value_scales[out] = fmaxf(maximum, 1e-20f) / 448.0f;
}

// sage_quantize_i8_f32: each 64-token block of each head as INT8 against its own scale, its absolute maximum over 127,
// the head's channel means subtracted first where mean is given (K's smoothing). Grid (blocks, heads, sequences), 128
// threads, a channel each; the codes keep the input's layout, the scales are [sequence][head][block].
extern "C" __global__ void sage_quantize_i8_f32(
        const float * x, const float * mean, signed char * codes, float * scales,
        unsigned int heads, unsigned int tokens, unsigned int sequences) {
    __shared__ float maxima[SAGE_WIDTH];
    const unsigned int block = blockIdx.x, head = blockIdx.y, sequence = blockIdx.z, channel = threadIdx.x;
    const unsigned int blocks = (tokens + SAGE_BLOCK - 1u) / SAGE_BLOCK;
    if (block >= blocks || head >= heads || sequence >= sequences) {
        return;
    }
    const float shift = mean == nullptr ? 0.0f : mean[(sequence * heads + head) * SAGE_WIDTH + channel];
    const unsigned int first = block * SAGE_BLOCK, last = min(first + SAGE_BLOCK, tokens);
    float maximum = 0.0f;
    for (unsigned int token = first; token < last; ++token) {
        maximum = fmaxf(maximum, fabsf(x[((sequence * tokens + token) * heads + head) * SAGE_WIDTH + channel] - shift));
    }
    maxima[channel] = maximum;
    __syncthreads();
    for (unsigned int stride = SAGE_WIDTH / 2u; stride > 0u; stride >>= 1) {
        if (channel < stride) {
            maxima[channel] = fmaxf(maxima[channel], maxima[channel + stride]);
        }
        __syncthreads();
    }
    const float scale = fmaxf(maxima[0], 1e-20f) / 127.0f;
    if (channel == 0u) {
        scales[(sequence * heads + head) * blocks + block] = scale;
    }
    for (unsigned int token = first; token < last; ++token) {
        const unsigned int at = ((sequence * tokens + token) * heads + head) * SAGE_WIDTH + channel;
        codes[at] = (signed char) fmaxf(-127.0f, fminf(127.0f, rintf((x[at] - shift) / scale)));
    }
}

// sage_quantize_v_fp8_f32: each 64-token block of each key/value head's values as FP8 E4M3 against its channels' scales
// (sage_channel_stats_f32), written transposed, [sequence][head][channel][token], each channel's tokens padded with
// zeros to padded (a multiple of 64). Grid (blocks, heads, sequences), 128 threads, a channel each, each writing its
// channel's 64 codes of the block.
extern "C" __global__ void sage_quantize_v_fp8_f32(
        const float * value, const float * scales, unsigned char * codes,
        unsigned int heads, unsigned int tokens, unsigned int padded, unsigned int sequences) {
    const unsigned int block = blockIdx.x, head = blockIdx.y, sequence = blockIdx.z, channel = threadIdx.x;
    if (block * SAGE_BLOCK >= padded || head >= heads || sequence >= sequences || channel >= SAGE_WIDTH) {
        return;
    }
    const float scale = scales[(sequence * heads + head) * SAGE_WIDTH + channel];
    uint32_t words[SAGE_BLOCK / 4u];
    for (unsigned int word = 0; word < SAGE_BLOCK / 4u; ++word) {
        float v[4];
        for (unsigned int i = 0; i < 4u; ++i) {
            const unsigned int token = block * SAGE_BLOCK + word * 4u + i;
            v[i] = token < tokens ? value[((sequence * tokens + token) * heads + head) * SAGE_WIDTH + channel] / scale : 0.0f;
        }
        words[word] = sage_fp8x4(v[0], v[1], v[2], v[3]);
    }
    uint4 * row = reinterpret_cast<uint4 *>(codes + ((size_t) (sequence * heads + head) * SAGE_WIDTH + channel) * padded + block * SAGE_BLOCK);
    for (unsigned int part = 0; part < SAGE_BLOCK / 16u; ++part) {
        row[part] = make_uint4(words[part * 4u], words[part * 4u + 1u], words[part * 4u + 2u], words[part * 4u + 3u]);
    }
}

// attention_sage_128_f32: non-causal attention of 64 query rows a block, four warps of 16 rows, over 64-key tiles. A
// warp's scores of a tile are eight 16x8 INT32 tiles; their softmax probabilities, times 448, are its FP8 A operand
// directly: in each 32-key chunk the thread's k positions 4t..4t+3 hold the keys 2t, 2t+1, 8+2t and 9+2t of the chunk
// and 16+4t..16+4t+3 the keys 16 on, which is how the score tiles hold them, and V's B operand is read in the same
// order, so the product sums over the same keys. Grid (row blocks, query heads, sequences), 128 threads.
extern "C" __global__ void __launch_bounds__(128) attention_sage_128_f32(
        const signed char * query, const float * query_scales,
        const signed char * key, const float * key_scales,
        const unsigned char * value, const float * value_scales,
        const float * key_bias, float * output,
        unsigned int query_heads, unsigned int key_value_heads,
        unsigned int query_tokens, unsigned int key_value_tokens, unsigned int sequences, float scale) {
    __shared__ __align__(16) unsigned char keys[SAGE_BLOCK * SAGE_KLD];
    __shared__ __align__(16) unsigned char values[SAGE_WIDTH * SAGE_VLD];
    const unsigned int row_block = blockIdx.x, query_head = blockIdx.y, sequence = blockIdx.z;
    if (query_head >= query_heads || sequence >= sequences) {
        return;
    }
    const unsigned int key_value_head = query_head / (query_heads / key_value_heads);
    const unsigned int warp = threadIdx.x >> 5, lane = threadIdx.x & 31u, g = lane >> 2, t = lane & 3u;
    const unsigned int query_blocks = (query_tokens + SAGE_BLOCK - 1u) / SAGE_BLOCK;
    const unsigned int key_blocks = (key_value_tokens + SAGE_BLOCK - 1u) / SAGE_BLOCK;
    const unsigned int padded = key_blocks * SAGE_BLOCK;
    const unsigned int row0 = row_block * SAGE_BLOCK + warp * 16u + g, row1 = row0 + 8u;
    const float log2e = 1.4426950408889634074f;
    const float query_scale = query_scales[(sequence * query_heads + query_head) * query_blocks + row_block];

    // The warp's Q rows as four 32-channel A operands.
    uint32_t q[4][4];
    for (unsigned int chunk = 0; chunk < 4u; ++chunk) {
        for (unsigned int half = 0; half < 2u; ++half) {
            const unsigned int channel = chunk * 32u + half * 16u + t * 4u;
            q[chunk][half * 2u] = row0 < query_tokens
                ? *reinterpret_cast<const uint32_t *>(query + ((sequence * query_tokens + row0) * query_heads + query_head) * SAGE_WIDTH + channel) : 0u;
            q[chunk][half * 2u + 1u] = row1 < query_tokens
                ? *reinterpret_cast<const uint32_t *>(query + ((sequence * query_tokens + row1) * query_heads + query_head) * SAGE_WIDTH + channel) : 0u;
        }
    }

    float o[16][4];
    for (unsigned int tile = 0; tile < 16u; ++tile) {
        o[tile][0] = o[tile][1] = o[tile][2] = o[tile][3] = 0.0f;
    }
    float maximum0 = -INFINITY, maximum1 = -INFINITY, sum0 = 0.0f, sum1 = 0.0f;

    for (unsigned int block = 0; block < key_blocks; ++block) {
        __syncthreads();
        // The tile's 64 keys, 128 INT8 channels each, and its V, 128 channels of 64 FP8 keys.
        for (unsigned int part = threadIdx.x; part < SAGE_BLOCK * 8u; part += blockDim.x) {
            const unsigned int row = part >> 3, column = (part & 7u) * 16u, token = block * SAGE_BLOCK + row;
            uint4 bytes = make_uint4(0u, 0u, 0u, 0u);
            if (token < key_value_tokens) {
                bytes = *reinterpret_cast<const uint4 *>(key + ((sequence * key_value_tokens + token) * key_value_heads + key_value_head) * SAGE_WIDTH + column);
            }
            *reinterpret_cast<uint4 *>(keys + row * SAGE_KLD + column) = bytes;
        }
        for (unsigned int part = threadIdx.x; part < SAGE_WIDTH * 4u; part += blockDim.x) {
            const unsigned int channel = part >> 2, column = (part & 3u) * 16u;
            *reinterpret_cast<uint4 *>(values + channel * SAGE_VLD + column) = *reinterpret_cast<const uint4 *>(
                value + ((size_t) (sequence * key_value_heads + key_value_head) * SAGE_WIDTH + channel) * padded + block * SAGE_BLOCK + column);
        }
        __syncthreads();

        // Scores: eight 16x8 tiles of keys, each over four 32-channel chunks.
        const float key_scale = key_scales[(sequence * key_value_heads + key_value_head) * key_blocks + block];
        const float score_scale = query_scale * key_scale * scale;
        float s[8][4];
        float tile_maximum0 = -INFINITY, tile_maximum1 = -INFINITY;
        for (unsigned int tile = 0; tile < 8u; ++tile) {
            int c[4] = {0, 0, 0, 0};
            const unsigned char * row = keys + (tile * 8u + g) * SAGE_KLD;
            for (unsigned int chunk = 0; chunk < 4u; ++chunk) {
                const uint32_t b0 = *reinterpret_cast<const uint32_t *>(row + chunk * 32u + t * 4u);
                const uint32_t b1 = *reinterpret_cast<const uint32_t *>(row + chunk * 32u + 16u + t * 4u);
                sage_mma_s8(c, q[chunk], b0, b1);
            }
            for (unsigned int i = 0; i < 4u; ++i) {
                const unsigned int token = block * SAGE_BLOCK + tile * 8u + t * 2u + (i & 1u);
                float score = -INFINITY;
                if (token < key_value_tokens) {
                    score = ((float) c[i] * score_scale + (key_bias != nullptr ? key_bias[token] : 0.0f)) * log2e;
                }
                s[tile][i] = score;
            }
            tile_maximum0 = fmaxf(tile_maximum0, fmaxf(s[tile][0], s[tile][1]));
            tile_maximum1 = fmaxf(tile_maximum1, fmaxf(s[tile][2], s[tile][3]));
        }
        for (unsigned int offset = 1u; offset < 4u; offset <<= 1) {
            tile_maximum0 = fmaxf(tile_maximum0, __shfl_xor_sync(0xffffffffu, tile_maximum0, offset));
            tile_maximum1 = fmaxf(tile_maximum1, __shfl_xor_sync(0xffffffffu, tile_maximum1, offset));
        }
        const float next0 = fmaxf(maximum0, tile_maximum0), next1 = fmaxf(maximum1, tile_maximum1);
        // A row with every key masked so far keeps nothing, and is rescaled from nothing.
        const float base0 = next0 == -INFINITY ? 0.0f : next0, base1 = next1 == -INFINITY ? 0.0f : next1;
        const float correction0 = __nv_exp2f(maximum0 - base0), correction1 = __nv_exp2f(maximum1 - base1);
        maximum0 = next0;
        maximum1 = next1;
        float tile_sum0 = 0.0f, tile_sum1 = 0.0f;
        for (unsigned int tile = 0; tile < 8u; ++tile) {
            s[tile][0] = __nv_exp2f(s[tile][0] - base0);
            s[tile][1] = __nv_exp2f(s[tile][1] - base0);
            s[tile][2] = __nv_exp2f(s[tile][2] - base1);
            s[tile][3] = __nv_exp2f(s[tile][3] - base1);
            tile_sum0 += s[tile][0] + s[tile][1];
            tile_sum1 += s[tile][2] + s[tile][3];
        }
        for (unsigned int offset = 1u; offset < 4u; offset <<= 1) {
            tile_sum0 += __shfl_xor_sync(0xffffffffu, tile_sum0, offset);
            tile_sum1 += __shfl_xor_sync(0xffffffffu, tile_sum1, offset);
        }
        sum0 = sum0 * correction0 + tile_sum0;
        sum1 = sum1 * correction1 + tile_sum1;

        // P.V over the tile's two 32-key chunks, P as FP8 times 448. The FP8 tensor cores accumulate in FP22, keeping
        // 13 mantissa bits, so each output tile sums one key tile from zero and adds it to the F32 output, which
        // confines the loss to a tile (SageAttention2's two-level accumulation).
        uint32_t p[2][4];
        for (unsigned int chunk = 0; chunk < 2u; ++chunk) {
            const unsigned int first = chunk * 4u;
            p[chunk][0] = sage_fp8x4(s[first][0] * 448.0f, s[first][1] * 448.0f, s[first + 1u][0] * 448.0f, s[first + 1u][1] * 448.0f);
            p[chunk][1] = sage_fp8x4(s[first][2] * 448.0f, s[first][3] * 448.0f, s[first + 1u][2] * 448.0f, s[first + 1u][3] * 448.0f);
            p[chunk][2] = sage_fp8x4(s[first + 2u][0] * 448.0f, s[first + 2u][1] * 448.0f, s[first + 3u][0] * 448.0f, s[first + 3u][1] * 448.0f);
            p[chunk][3] = sage_fp8x4(s[first + 2u][2] * 448.0f, s[first + 2u][3] * 448.0f, s[first + 3u][2] * 448.0f, s[first + 3u][3] * 448.0f);
        }
        for (unsigned int tile = 0; tile < 16u; ++tile) {
            float r[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (unsigned int chunk = 0; chunk < 2u; ++chunk) {
                const unsigned char * row = values + (tile * 8u + g) * SAGE_VLD + chunk * 32u;
                const uint32_t b0 = (uint32_t) *reinterpret_cast<const unsigned short *>(row + t * 2u)
                    | ((uint32_t) *reinterpret_cast<const unsigned short *>(row + 8u + t * 2u) << 16);
                const uint32_t b1 = (uint32_t) *reinterpret_cast<const unsigned short *>(row + 16u + t * 2u)
                    | ((uint32_t) *reinterpret_cast<const unsigned short *>(row + 24u + t * 2u) << 16);
                sage_mma_e4m3(r, p[chunk], b0, b1);
            }
            o[tile][0] = o[tile][0] * correction0 + r[0];
            o[tile][1] = o[tile][1] * correction0 + r[1];
            o[tile][2] = o[tile][2] * correction1 + r[2];
            o[tile][3] = o[tile][3] * correction1 + r[3];
        }
    }

    const float inverse0 = sum0 > 0.0f ? 1.0f / (sum0 * 448.0f) : 0.0f;
    const float inverse1 = sum1 > 0.0f ? 1.0f / (sum1 * 448.0f) : 0.0f;
    for (unsigned int tile = 0; tile < 16u; ++tile) {
        const unsigned int channel = tile * 8u + t * 2u;
        const float v0 = value_scales[(sequence * key_value_heads + key_value_head) * SAGE_WIDTH + channel];
        const float v1 = value_scales[(sequence * key_value_heads + key_value_head) * SAGE_WIDTH + channel + 1u];
        if (row0 < query_tokens) {
            float * out = output + ((sequence * query_tokens + row0) * query_heads + query_head) * SAGE_WIDTH + channel;
            out[0] = o[tile][0] * v0 * inverse0;
            out[1] = o[tile][1] * v1 * inverse0;
        }
        if (row1 < query_tokens) {
            float * out = output + ((sequence * query_tokens + row1) * query_heads + query_head) * SAGE_WIDTH + channel;
            out[0] = o[tile][2] * v0 * inverse1;
            out[1] = o[tile][3] * v1 * inverse1;
        }
    }
}

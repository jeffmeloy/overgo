// ops_f32_tail.cuh: MAD norms, the 64-key BF16 tiled attention, native SiLU and RMS means, in ops_f32.cu's order. ops_f32.cu
// includes it once; it compiles only there.

extern "C" __global__ void mad_norm_f32(
        const float * input,
        float * output,
        unsigned int width,
        unsigned int rows,
        float epsilon) {
    const unsigned int row_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_index >= rows) {
        return;
    }
    const float * row = input + (size_t)row_index * width;
    float * out = output + (size_t)row_index * width;
    double location, inverse, c;
    mad_row_facts(row, width, epsilon, &location, &inverse, &c);
    for (unsigned int index = 0; index < width; ++index) {
        out[index] = (float)(((double)row[index] - location) * inverse);
    }
}

extern "C" __global__ void mad_norm_backward_f32(
        const float * incoming,
        const float * input,
        const float * output,
        float * gradient,
        unsigned int width,
        unsigned int rows,
        float epsilon) {
    const unsigned int row_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_index >= rows) {
        return;
    }
    const float * row = input + (size_t)row_index * width;
    const float * dy = incoming + (size_t)row_index * width;
    const float * normalized = output + (size_t)row_index * width;
    float * dx = gradient + (size_t)row_index * width;
    double location, inverse, c;
    mad_row_facts(row, width, epsilon, &location, &inverse, &c);
    double sum = 0.0;
    double weighted = 0.0;
    for (unsigned int index = 0; index < width; ++index) {
        sum += (double)dy[index];
        weighted += (double)dy[index] * (double)normalized[index];
    }
    const double t1 = inverse * (sum + c * weighted);
    const double t2 = inverse * weighted;
    const double two_over_width = 2.0 / (double)width;
    for (unsigned int index = 0; index < width; ++index) {
        const float a = mad_raw_weight(mad_value_rank(row, width, index), width) / (float)width;
        const float raw = mad_raw_weight(mad_distance_rank(row, width, location, index), width);
        const float b = (float)((double)raw * two_over_width * ((double)row[index] >= location ? 1.0 : -1.0));
        dx[index] = (float)(inverse * (double)dy[index] - (double)a * t1 - (double)b * t2);
    }
}

// The 64-key path retains the established BF16 attention accumulation contract.
extern "C" __global__ void __launch_bounds__(512, 1) attention_tiled_bf16_f32(
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
    const unsigned int BM = 64u;
    const unsigned int BN = 64u;
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
    __nv_bfloat16 * k_tile = (__nv_bfloat16 *) attention_shared;    // [2][BN][KLD]
    __nv_bfloat16 * v_tile = k_tile + 2u * BN * KLD;                // [BN][VLD]
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
        q_tile[row * QLD + channel] = __float2bfloat16(token < query_tokens
            ? query[((sequence * query_tokens + token) * query_heads + query_head) * W + channel]
            : 0.0f);
    }
    for (unsigned int index = threadIdx.x; index < BM; index += blockDim.x) {
        maximum_row[index] = -3.402823466e+38F;
        sum_row[index] = 0.0f;
    }

    // Warp tiling: warp w owns S rows (w%4)*16 x columns (w/4)*16 and O rows
    // (w%4)*16 x columns (w/4)*32. Architected fragment lane roles: g = row
    // within the 8-row group, t = the lane's element pair.
    const unsigned int s_row0 = (warp & 3u) * 16u;
    const unsigned int s_column0 = (warp >> 2) * 16u;
    const unsigned int o_column_base = (warp >> 2) * 32u;
    const unsigned int g = lane >> 2;
    const unsigned int t = lane & 3u;
    // ldmatrix lane->address roles (16x16 operand = four 8x8 tiles).
    const unsigned int a_row = (lane & 7u) + ((lane >> 3u) & 1u) * 8u;  // tiles: row-halves first
    const unsigned int a_col = (lane >> 4u) * 8u;
    const unsigned int bk_row = (lane & 7u) + (lane >> 4u) * 8u;        // tiles: column-halves first
    const unsigned int bk_col = ((lane >> 3u) & 1u) * 8u;

    // O accumulators: four 16x8 column tiles, register-resident across the
    // whole stream; element rows are architected (g and g+8).
    float weighted[4][4];
    for (unsigned int tile = 0; tile < 4u; ++tile) {
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

    // Pre-offset K/V to the sequence and KV head; preload K panel 0.
    const unsigned int kv_token_stride = key_value_heads * W;
    const unsigned short * key_base = key +
        sequence * key_value_tokens * kv_token_stride + key_value_head * W;
    const unsigned short * value_base = value +
        sequence * key_value_tokens * kv_token_stride + key_value_head * W;
    attention_bf16_stage_rows(key_base, k_tile, 0u, key_value_tokens, kv_token_stride, W, BN);
    __pipeline_commit();

    unsigned int stage = 0u;
    for (unsigned int tile_base = 0; tile_base < key_value_tokens; tile_base += BN, stage ^= 1u) {
        const bool has_next = tile_base + BN < key_value_tokens;
        // K for this tile is the only outstanding group; the barrier both
        // publishes it and closes out the previous value product, so v_tile
        // is free for this panel's V copy (which then overlaps scores).
        __pipeline_wait_prior(0);
        __syncthreads();
        attention_bf16_stage_rows(
            value_base, v_tile, tile_base, key_value_tokens, kv_token_stride, W, BN);
        __pipeline_commit();
        const __nv_bfloat16 * k_stage = k_tile + stage * BN * KLD;

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
        // The next K panel loads during softmax and the value product.
        if (has_next) {
            attention_bf16_stage_rows(
                key_base, k_tile + (stage ^ 1u) * BN * KLD,
                tile_base + BN, key_value_tokens, kv_token_stride, W, BN);
            __pipeline_commit();
        }
        __syncthreads();

        // Exact online softmax on the staged F32 scores: eight threads per
        // row, reduced with warp shuffles.
        {
            const unsigned int row = threadIdx.x >> 3;
            const unsigned int part = threadIdx.x & 7u;
            const unsigned int remaining = key_value_tokens - tile_base;
            const unsigned int valid = remaining < BN ? remaining : BN;
            // One shared read per column: the scaled scores stay in
            // registers across the max and probability passes.
            float scaled[8];
            for (unsigned int i = 0; i < 8u; ++i) {
                const unsigned int key_token = tile_base + part + 8u * i;
                scaled[i] = s_tile[row * SLD + part + 8u * i] * scale +
                    (key_bias != nullptr && key_token < key_value_tokens ? key_bias[key_token] : 0.0f);
            }
            float local_maximum = -3.402823466e+38F;
            for (unsigned int i = 0; i < 8u; ++i) {
                if (part + 8u * i < valid) {
                    local_maximum = fmaxf(local_maximum, scaled[i]);
                }
            }
            local_maximum = fmaxf(local_maximum, __shfl_xor_sync(0xffffffffu, local_maximum, 1));
            local_maximum = fmaxf(local_maximum, __shfl_xor_sync(0xffffffffu, local_maximum, 2));
            local_maximum = fmaxf(local_maximum, __shfl_xor_sync(0xffffffffu, local_maximum, 4));
            const float next_maximum = fmaxf(maximum_row[row], local_maximum);
            const float alpha = expf(maximum_row[row] - next_maximum);
            float local_sum = 0.0f;
            for (unsigned int i = 0; i < 8u; ++i) {
                const float probability = part + 8u * i < valid
                    ? expf(scaled[i] - next_maximum)
                    : 0.0f;
                p_tile[row * PLD + part + 8u * i] = __float2bfloat16(probability);
                local_sum += probability;
            }
            local_sum += __shfl_xor_sync(0xffffffffu, local_sum, 1);
            local_sum += __shfl_xor_sync(0xffffffffu, local_sum, 2);
            local_sum += __shfl_xor_sync(0xffffffffu, local_sum, 4);
            __syncwarp();
            if (part == 0) {
                maximum_row[row] = next_maximum;
                sum_row[row] = sum_row[row] * alpha + local_sum;
                alpha_row[row] = alpha;
            }
        }
        // V panel must be resident before the value product: all-but-one
        // leaves only the next K (or nothing on the last panel)
        // outstanding. One barrier publishes P, the row stats, and V.
        __pipeline_wait_prior(has_next ? 1 : 0);
        __syncthreads();

        {
            // Register-resident rescale: each accumulator element's row is
            // architected (g in the low half, g+8 in the high), so the
            // running-max correction is a plain multiply. Steady-state
            // panels carry alpha == 1 exactly and x*1.0f is exact.
            const float alpha_low = alpha_row[s_row0 + g];
            const float alpha_high = alpha_row[s_row0 + 8u + g];
            for (unsigned int tile = 0; tile < 4u; ++tile) {
                weighted[tile][0] *= alpha_low;
                weighted[tile][1] *= alpha_low;
                weighted[tile][2] *= alpha_high;
                weighted[tile][3] *= alpha_high;
            }
        }

        {
            // Value product: P fragments load once per token quarter and
            // feed all four O column tiles.
            for (unsigned int k = 0; k < 4u; ++k) {
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
                attention_ldx4_trans(
                    attention_shared_address(v_rows + 16u), b0, b1, b2, b3);
                attention_mma(weighted[2], p_frag, b0, b1);
                attention_mma(weighted[3], p_frag, b2, b3);
            }
        }
        // No trailing barrier: the next panel's top barrier (after the K
        // wait) closes this value product before v_tile is rewritten.
    }

    for (unsigned int tile = 0; tile < 4u; ++tile) {
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
        if (token < query_tokens) {
            output[((sequence * query_tokens + token) * query_heads + query_head) * W + channel] =
                attention_row_value(o_tile[row * OLD + channel], sum_row[row]);
        }
    }
}

extern "C" __global__ void silu_native_f32(
        const float * input,
        float * output,
        unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        const float value = input[index];
        output[index] = __fdiv_rn(value, __fadd_rn(1.0f, __nv_expf(-value)));
    }
}

// rms_norm_mean_inverse is the factor a native-mean RMS norm multiplies a row of width values by, the block's threads
// together: the row's sum of squares, four adjacent values a lane where the row is wide enough, the vectorized
// reduction's order, times the separately rounded ratio of rows to elements, plus epsilon, to the reciprocal root.
__device__ __forceinline__ float rms_norm_mean_inverse(
        const float * row_values,
        unsigned int width,
        unsigned int rows,
        float epsilon) {
    float sum_squares = 0.0f;
    if (width >= 128u && width % 4u == 0u) {
        // Four adjacent values per lane preserve the vectorized reduction order.
        float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        if (threadIdx.x < 32u) {
            for (unsigned int column = threadIdx.x * 4u; column < width; column += 128u) {
                for (unsigned int part = 0; part < 4u; ++part) {
                    const float value = row_values[column + part];
                    sums[part] += __fmul_rn(value, value);
                }
            }
            sum_squares = ((sums[0] + sums[1]) + sums[2]) + sums[3];
        }
    } else {
        for (unsigned int column = threadIdx.x; column < width; column += blockDim.x) {
            const float value = row_values[column];
            sum_squares += value * value;
        }
    }
    __shared__ float partial[256];
    sum_squares = block_sum_f32(sum_squares, partial);
    // Native mean projects the sum with a separately rounded element-count ratio.
    const float factor = __fdiv_rn((float) rows, (float) ((unsigned long long) rows * width));
    return rsqrtf(__fadd_rn(__fmul_rn(sum_squares, factor), epsilon));
}

template <bool ROUND>
__device__ __forceinline__ void rms_norm_mean_elementwise(
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
    const float inverse = rms_norm_mean_inverse(input + offset, width, rows, epsilon);
    for (unsigned int column = threadIdx.x; column < width; column += blockDim.x) {
        output[offset + column] = elementwise_store<ROUND>(input[offset + column] * inverse);
    }
}

// RMS_NORM_STEPS is how many per-channel steps rms_norm_mean_affine_bf16_round_f32 applies at most, and each step's kind
// is RMS_NORM_STEP_BITS bits of its kinds, the first step's lowest: none, a multiply or an add by a vector of width.
constexpr unsigned int RMS_NORM_STEPS = 3;
constexpr unsigned int RMS_NORM_STEP_BITS = 2;
constexpr unsigned int RMS_NORM_STEP_MASK = 3;
constexpr unsigned int RMS_NORM_STEP_MULTIPLY = 1;
constexpr unsigned int RMS_NORM_STEP_ADD = 2;

// rms_norm_mean_affine_value is a row's value at column normalized by inverse and rounded to BF16, then taken through
// the steps kinds packs, each multiplying or adding the step's vector at column and rounding to BF16, in order.
__device__ __forceinline__ float rms_norm_mean_affine_value(
        float value,
        float inverse,
        const float * step0,
        const float * step1,
        const float * step2,
        unsigned int kinds,
        unsigned int column) {
    const float * steps[RMS_NORM_STEPS] = {step0, step1, step2};
    float result = bf16_rounded(value * inverse);
    for (unsigned int step = 0; step < RMS_NORM_STEPS; ++step) {
        const unsigned int kind = (kinds >> (step * RMS_NORM_STEP_BITS)) & RMS_NORM_STEP_MASK;
        if (kind == RMS_NORM_STEP_MULTIPLY) {
            result = bf16_rounded(result * steps[step][column]);
        } else if (kind == RMS_NORM_STEP_ADD) {
            result = bf16_rounded(result + steps[step][column]);
        }
    }
    return result;
}

// rms_norm_mean_affine_bf16_round_f32: a native-mean RMS norm rounded to BF16 (rms_norm_mean_bf16_round_f32), then up to
// RMS_NORM_STEPS steps each multiplying or adding a per-channel vector of width and rounding to BF16, in order
// (rms_norm_mean_affine_value), one row of width values a block, as MOSS's denoiser normalizes, weights and modulates
// (latentimage rotaryGraph.norm and modulate): each value reads its row once and is written once.
extern "C" __global__ void rms_norm_mean_affine_bf16_round_f32(
        const float * input,
        float * output,
        const float * step0,
        const float * step1,
        const float * step2,
        unsigned int kinds,
        unsigned int width,
        unsigned int rows,
        float epsilon) {
    const unsigned int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    const unsigned int offset = row * width;
    const float inverse = rms_norm_mean_inverse(input + offset, width, rows, epsilon);
    for (unsigned int column = threadIdx.x; column < width; column += blockDim.x) {
        output[offset + column] = rms_norm_mean_affine_value(input[offset + column], inverse, step0, step1, step2, kinds, column);
    }
}

// MOSS_ROTATE_MAX_WIDTH is the widest head moss_norm_rotate_bf16_round_f32 holds in shared memory, twice MOSS's 128.
constexpr unsigned int MOSS_ROTATE_MAX_WIDTH = 256;

// moss_norm_rotate_bf16_round_f32: MOSS's q or k heads normalized and stepped (rms_norm_mean_affine_value over each
// head's width channels, the norm's rows heads * tokens) and rotated (moss_rotate_value) in one kernel, as the denoiser
// normalizes, weights and rotates them (latentimage rotaryGraph.norm and rotate): one block a head's row, its stepped
// values held in shared memory for the rotation's half-turn and never stored. The heads are [width, heads, tokens] and
// the tables [width, 1, tokens], read for every head.
extern "C" __global__ void moss_norm_rotate_bf16_round_f32(
        const float * input,
        const float * step0,
        const float * step1,
        const float * step2,
        unsigned int kinds,
        float epsilon,
        const float * cosine,
        const float * sine,
        float * output,
        unsigned int width,
        unsigned int heads,
        unsigned int tokens) {
    __shared__ float values[MOSS_ROTATE_MAX_WIDTH];
    const unsigned int row = blockIdx.x;
    const unsigned int rows = heads * tokens;
    if (row >= rows || width > MOSS_ROTATE_MAX_WIDTH || width % 2u != 0u) {
        return;
    }
    const unsigned int offset = row * width;
    const float inverse = rms_norm_mean_inverse(input + offset, width, rows, epsilon);
    for (unsigned int channel = threadIdx.x; channel < width; channel += blockDim.x) {
        values[channel] = rms_norm_mean_affine_value(input[offset + channel], inverse, step0, step1, step2, kinds, channel);
    }
    __syncthreads();
    const unsigned int token = row / heads;
    const unsigned int half = width / 2u;
    for (unsigned int channel = threadIdx.x; channel < width; channel += blockDim.x) {
        const float turned = channel < half ? -values[channel + half] : values[channel - half];
        const unsigned int table = token * width + channel;
        output[offset + channel] = moss_rotate_value(values[channel], turned, cosine[table], sine[table]);
    }
}

// quantize_int8_convrot_norm_columns_f32: the activation side of an int8 ConvRot product whose activation is a native-mean
// RMS norm of input and the steps after it (rms_norm_mean_affine_bf16_round_f32), computed here and never stored: one
// block per token column of inner values, the norm's rows the columns, normalized and stepped as the affine norm does,
// then rotated and quantized as quantize_int8_convrot_columns_f32 does (int8_convrot_rotate, int8_convrot_quantize).
extern "C" __global__ void quantize_int8_convrot_norm_columns_f32(
        const float * input,
        const float * step0,
        const float * step1,
        const float * step2,
        unsigned int kinds,
        float epsilon,
        signed char * output,
        float * scale,
        unsigned int inner,
        unsigned int columns) {
    __shared__ float values[INT8_CONVROT_MAX_INNER];
    __shared__ float warp_amax[CUDA_WARP_WIDTH];
    const unsigned int column = blockIdx.x;
    if (column >= columns || inner > INT8_CONVROT_MAX_INNER || inner % INT8_CONVROT_GROUP != 0) {
        return;
    }
    const float * source = input + (size_t) column * inner;
    const float inverse = rms_norm_mean_inverse(source, inner, columns, epsilon);
    for (unsigned int index = threadIdx.x; index < inner; index += blockDim.x) {
        values[index] = rms_norm_mean_affine_value(source[index], inverse, step0, step1, step2, kinds, index);
    }
    __syncthreads();
    int8_convrot_rotate(values, inner);
    int8_convrot_quantize<false>(values, warp_amax, output + (size_t) column * inner, scale + column, inner, 0, 0);
}

extern "C" __global__ void rms_norm_mean_f32(
        const float * input,
        float * output,
        unsigned int width,
        unsigned int rows,
        float epsilon) {
    rms_norm_mean_elementwise<false>(input, output, width, rows, epsilon);
}

extern "C" __global__ void rms_norm_mean_bf16_round_f32(
        const float * input,
        float * output,
        unsigned int width,
        unsigned int rows,
        float epsilon) {
    rms_norm_mean_elementwise<true>(input, output, width, rows, epsilon);
}

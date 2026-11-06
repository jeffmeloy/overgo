// ops_f32_experts.cuh: block dots, routed experts on q8 activations, the expert doorbell and grouped and prompt expert products, in ops_f32.cu's order. ops_f32.cu
// includes it once; it compiles only there.

// ---- Block dots ----
// A block dot gives one lane BLOCK_DOT_WEIGHTS weights of a quantized block.
// The lane reads the block's scales once for them and dots them with the matching activations.
// A row of a block storage holds whole blocks.
// BLOCK_DOT_WEIGHTS = the weights of one IQ3_S sign byte = the nibbles of one 32-bit IQ4 word.
constexpr unsigned int BLOCK_DOT_WEIGHTS = 8;

// block_dot_codebooks holds the codebooks of the block dots in shared memory.
//   - lanes of a warp read different entries at once
//   - constant memory serves one address at a time, and shared memory one address a bank
struct block_dot_codebooks {
    unsigned int iq3_s[sizeof(iq3SGrid) / sizeof(iq3SGrid[0])];
    float iq4_nl[sizeof(iq4nl_values) / sizeof(iq4nl_values[0])];
};

// block_dot_codebooks_load copies the codebooks the block dots of two storages read.
//   - every thread of the block calls it before any returns, because it waits for the whole block
__device__ __forceinline__ void block_dot_codebooks_load(
        block_dot_codebooks & codebooks, unsigned int first, unsigned int second) {
    if (first == GGML_IQ3_S || second == GGML_IQ3_S) {
        for (unsigned int index = threadIdx.x; index < sizeof(codebooks.iq3_s) / sizeof(codebooks.iq3_s[0]); index += blockDim.x) {
            codebooks.iq3_s[index] = iq3SGrid[index];
        }
    }
    if (first == GGML_IQ4_NL || first == GGML_IQ4_XS || second == GGML_IQ4_NL || second == GGML_IQ4_XS) {
        for (unsigned int index = threadIdx.x; index < sizeof(codebooks.iq4_nl) / sizeof(codebooks.iq4_nl[0]); index += blockDim.x) {
            codebooks.iq4_nl[index] = iq4nl_values[index];
        }
    }
    __syncthreads();
}

// The block loads read weight bytes through the read-only cache.
__device__ __forceinline__ unsigned int load_u8(const unsigned char * source) {
    return __ldg(source);
}

__device__ __forceinline__ unsigned int load_u16(const unsigned char * source) {
    return __ldg(reinterpret_cast<const unsigned short *>(source));
}

// load_u32_halves reads four bytes that start 2-byte aligned.
__device__ __forceinline__ unsigned int load_u32_halves(const unsigned char * source) {
    return load_u16(source) | load_u16(source + 2) << 16;
}

__device__ __forceinline__ float load_half(const unsigned char * source) {
    return __half2float(__ushort_as_half((unsigned short) load_u16(source)));
}

// byte_float returns byte BYTE of bytes as a float.
//   - the byte becomes the low mantissa bits of 2^23, and 2^23 then subtracts exactly
//   - a byte permute and an add replace a conversion
//   - conversion rate = 1/8 of the add rate on compute capability 8.9 (CUDA C++ Programming Guide, arithmetic instruction throughput)
template <unsigned int BYTE>
__device__ __forceinline__ float byte_float(unsigned int bytes) {
    return __int_as_float((int) __byte_perm(bytes, 0x4b000000u, 0x7540u | BYTE)) - 8388608.0f;
}

// signed_byte_float returns signed byte BYTE of bytes as a float.
//   - the flipped top bit offsets the byte by 128, which subtracts with 2^23
template <unsigned int BYTE>
__device__ __forceinline__ float signed_byte_float(unsigned int bytes) {
    return __int_as_float((int) __byte_perm(bytes ^ 0x80808080u, 0x4b000000u, 0x7540u | BYTE)) - 8388736.0f;
}

// iq3_s_quad_dot dots the four magnitudes of an IQ3_S grid entry with x.
//   - bit i of signs negates magnitude i, for i < 4
__device__ __forceinline__ float iq3_s_quad_dot(unsigned int grid, unsigned int signs, float4 x) {
    const float w0 = byte_float<0>(grid);
    const float w1 = byte_float<1>(grid);
    const float w2 = byte_float<2>(grid);
    const float w3 = byte_float<3>(grid);
    return (signs & 1 ? -w0 : w0) * x.x + (signs & 2 ? -w1 : w1) * x.y +
        (signs & 4 ? -w2 : w2) * x.z + (signs & 8 ? -w3 : w3) * x.w;
}

// iq4_nibble_dot dots the eight IQ4_NL codebook values packed in four bytes with x.
//   - the low nibbles weigh low and the high nibbles weigh high, byte by byte
__device__ __forceinline__ float iq4_nibble_dot(unsigned int packed, float4 low, float4 high, const float * values) {
    return values[packed & 0x0f] * low.x + values[(packed >> 8) & 0x0f] * low.y +
        values[(packed >> 16) & 0x0f] * low.z + values[(packed >> 24) & 0x0f] * low.w +
        values[(packed >> 4) & 0x0f] * high.x + values[(packed >> 12) & 0x0f] * high.y +
        values[(packed >> 20) & 0x0f] * high.z + values[packed >> 28] * high.w;
}

// q8_quad_dot dots four signed bytes with x.
__device__ __forceinline__ float q8_quad_dot(unsigned int bytes, float4 x) {
    return signed_byte_float<0>(bytes) * x.x + signed_byte_float<1>(bytes) * x.y +
        signed_byte_float<2>(bytes) * x.z + signed_byte_float<3>(bytes) * x.w;
}

// iq3_s_quad_weights writes the four magnitudes of an IQ3_S grid entry, bit i of signs negating magnitude i.
__device__ __forceinline__ void iq3_s_quad_weights(unsigned int grid, unsigned int signs, float * w) {
    const float w0 = byte_float<0>(grid);
    const float w1 = byte_float<1>(grid);
    const float w2 = byte_float<2>(grid);
    const float w3 = byte_float<3>(grid);
    w[0] = signs & 1 ? -w0 : w0;
    w[1] = signs & 2 ? -w1 : w1;
    w[2] = signs & 4 ? -w2 : w2;
    w[3] = signs & 8 ? -w3 : w3;
}

// iq4_nibble_weights writes the eight IQ4_NL codebook values packed in four bytes: the low nibbles, then the high.
__device__ __forceinline__ void iq4_nibble_weights(unsigned int packed, const float * values, float * w) {
    #pragma unroll
    for (unsigned int i = 0; i < BLOCK_DOT_WEIGHTS / 2; ++i) {
        w[i] = values[(packed >> (8 * i)) & 0x0f];
        w[BLOCK_DOT_WEIGHTS / 2 + i] = values[(packed >> (8 * i + 4)) & 0x0f];
    }
}

// q8_quad_weights writes four signed bytes.
__device__ __forceinline__ void q8_quad_weights(unsigned int bytes, float * w) {
    w[0] = signed_byte_float<0>(bytes);
    w[1] = signed_byte_float<1>(bytes);
    w[2] = signed_byte_float<2>(bytes);
    w[3] = signed_byte_float<3>(bytes);
}

// A block format names a storage's block width in weights, its block bytes, its block dot and its weights.
//   - width and bytes = ggml's layout, which the storage's dequant_*_value function reads
//   - dot(block, group, x, codebooks) dots BLOCK_DOT_WEIGHTS weights of the block with x
//   - x starts at the block's first weight, 16-byte aligned
//   - weights(block, group, codebooks, w) writes the same weights as whole numbers and returns their scale:
//     w[0..3] from weight offset(group), w[4..7] from offset(group) + split, all in one BLOCK_SCALE_WEIGHTS-weight span of one scale
//   - the weights times their scale decode as the storage's dequant_*_value function decodes them

// BLOCK_SCALE_WEIGHTS: the weights one scale of every block format covers, a sub-block or a block of 32.
constexpr unsigned int BLOCK_SCALE_WEIGHTS = 32;

// iq3_s_blocks dots weights 8 * group to 8 * group + 7.
struct iq3_s_blocks {
    static constexpr unsigned int width = 256;
    static constexpr unsigned int bytes = 110;
    static constexpr unsigned int split = BLOCK_DOT_WEIGHTS / 2;
    __device__ static __forceinline__ unsigned int offset(unsigned int group) { return BLOCK_DOT_WEIGHTS * group; }
    // entries reads a group's two grid entries and its signs, and returns its scale.
    __device__ static __forceinline__ float entries(const unsigned char * block, unsigned int group,
            const block_dot_codebooks & codebooks, unsigned int & first, unsigned int & second, unsigned int & signs) {
        const unsigned int pair = group / 8;
        const unsigned int half = group / 4 % 2;
        const unsigned int sub_group = group % 4;
        const unsigned int indices = load_u16(block + 2 + pair * 16 + half * 8 + sub_group * 2);
        const unsigned int high = load_u8(block + 66 + pair * 2 + half);
        signs = load_u8(block + 74 + pair * 8 + half * 4 + sub_group);
        const unsigned int group_scale = (load_u8(block + 106 + pair) >> (4 * half)) & 0x0f;
        first = codebooks.iq3_s[(indices & 0xff) | ((high << (8 - 2 * sub_group)) & 0x0100)];
        second = codebooks.iq3_s[(indices >> 8) | ((high << (7 - 2 * sub_group)) & 0x0100)];
        return load_half(block) * (float) (1 + 2 * group_scale);
    }
    __device__ static __forceinline__ float dot(
            const unsigned char * block, unsigned int group, const float * x, const block_dot_codebooks & codebooks) {
        unsigned int first, second, signs;
        const float scale = entries(block, group, codebooks, first, second, signs);
        const float * group_x = x + BLOCK_DOT_WEIGHTS * group;
        return scale * (iq3_s_quad_dot(first, signs, load_f32x4(group_x)) +
            iq3_s_quad_dot(second, signs >> 4, load_f32x4(group_x + 4)));
    }
    __device__ static __forceinline__ float weights(
            const unsigned char * block, unsigned int group, const block_dot_codebooks & codebooks, float * w) {
        unsigned int first, second, signs;
        const float scale = entries(block, group, codebooks, first, second, signs);
        iq3_s_quad_weights(first, signs, w);
        iq3_s_quad_weights(second, signs >> 4, w + split);
        return scale;
    }
};

// iq4_nl_blocks dots weights 4 * group to 4 * group + 3 and 16 + 4 * group to 19 + 4 * group.
struct iq4_nl_blocks {
    static constexpr unsigned int width = 32;
    static constexpr unsigned int bytes = 18;
    static constexpr unsigned int split = width / 2;
    __device__ static __forceinline__ unsigned int offset(unsigned int group) { return BLOCK_DOT_WEIGHTS / 2 * group; }
    __device__ static __forceinline__ float dot(
            const unsigned char * block, unsigned int group, const float * x, const block_dot_codebooks & codebooks) {
        const unsigned int packed = load_u32_halves(block + 2 + 4 * group);
        return load_half(block) *
            iq4_nibble_dot(packed, load_f32x4(x + 4 * group), load_f32x4(x + 16 + 4 * group), codebooks.iq4_nl);
    }
    __device__ static __forceinline__ float weights(
            const unsigned char * block, unsigned int group, const block_dot_codebooks & codebooks, float * w) {
        iq4_nibble_weights(load_u32_halves(block + 2 + 4 * group), codebooks.iq4_nl, w);
        return load_half(block);
    }
};

// iq4_xs_blocks dots one quarter of a 32-weight sub-block, laid out as an IQ4_NL block.
//   - group / 4 = the sub-block, group % 4 = the quarter
struct iq4_xs_blocks {
    static constexpr unsigned int width = 256;
    static constexpr unsigned int bytes = 136;
    static constexpr unsigned int split = iq4_nl_blocks::split;
    __device__ static __forceinline__ unsigned int offset(unsigned int group) {
        return iq4_nl_blocks::width * (group / 4) + iq4_nl_blocks::offset(group % 4);
    }
    // sub_block_scale returns the scale of the group's sub-block.
    __device__ static __forceinline__ float sub_block_scale(const unsigned char * block, unsigned int sub_block) {
        const unsigned int high_scales = load_u16(block + 2);
        const unsigned int sub_scale = ((load_u8(block + 4 + sub_block / 2) >> (4 * (sub_block % 2))) & 0x0f) |
            (((high_scales >> (2 * sub_block)) & 0x03) << 4);
        return load_half(block) * (float) ((int) sub_scale - 32);
    }
    __device__ static __forceinline__ const unsigned char * nibbles(const unsigned char * block, unsigned int group) {
        return block + 8 + 16 * (group / 4) + 4 * (group % 4);
    }
    __device__ static __forceinline__ float dot(
            const unsigned char * block, unsigned int group, const float * x, const block_dot_codebooks & codebooks) {
        const unsigned int sub_block = group / 4;
        const unsigned int quarter = group % 4;
        const unsigned int packed = __ldg(reinterpret_cast<const unsigned int *>(nibbles(block, group)));
        const float * sub_x = x + iq4_nl_blocks::width * sub_block;
        return sub_block_scale(block, sub_block) *
            iq4_nibble_dot(packed, load_f32x4(sub_x + 4 * quarter), load_f32x4(sub_x + 16 + 4 * quarter),
                codebooks.iq4_nl);
    }
    __device__ static __forceinline__ float weights(
            const unsigned char * block, unsigned int group, const block_dot_codebooks & codebooks, float * w) {
        iq4_nibble_weights(__ldg(reinterpret_cast<const unsigned int *>(nibbles(block, group))), codebooks.iq4_nl, w);
        return sub_block_scale(block, group / 4);
    }
};

// q8_0_blocks dots weights 8 * group to 8 * group + 7.
struct q8_0_blocks {
    static constexpr unsigned int width = Q8_0_BLOCK_WIDTH;
    static constexpr unsigned int bytes = Q8_0_BLOCK_BYTES;
    static constexpr unsigned int split = BLOCK_DOT_WEIGHTS / 2;
    __device__ static __forceinline__ unsigned int offset(unsigned int group) { return BLOCK_DOT_WEIGHTS * group; }
    __device__ static __forceinline__ float dot(
            const unsigned char * block, unsigned int group, const float * x, const block_dot_codebooks &) {
        const unsigned char * quantized = block + Q8_0_SCALE_BYTES + BLOCK_DOT_WEIGHTS * group;
        const float * group_x = x + BLOCK_DOT_WEIGHTS * group;
        return load_half(block) * (q8_quad_dot(load_u32_halves(quantized), load_f32x4(group_x)) +
            q8_quad_dot(load_u32_halves(quantized + 4), load_f32x4(group_x + 4)));
    }
    __device__ static __forceinline__ float weights(
            const unsigned char * block, unsigned int group, const block_dot_codebooks &, float * w) {
        const unsigned char * quantized = block + Q8_0_SCALE_BYTES + BLOCK_DOT_WEIGHTS * group;
        q8_quad_weights(load_u32_halves(quantized), w);
        q8_quad_weights(load_u32_halves(quantized + split), w + split);
        return load_half(block);
    }
};

// visit_block_storage calls visit with the block format of a storage, and refuses a storage without one.
template <typename Visit>
__device__ __forceinline__ bool visit_block_storage(unsigned int storage, Visit visit) {
    switch (storage) {
    case GGML_IQ3_S: visit(iq3_s_blocks{}); return true;
    case GGML_IQ4_XS: visit(iq4_xs_blocks{}); return true;
    case GGML_IQ4_NL: visit(iq4_nl_blocks{}); return true;
    case GGML_Q8_0: visit(q8_0_blocks{}); return true;
    default: return false;
    }
}

// visit_block_format calls visit with the block format of a storage.
//   - it refuses a storage without a block dot, or activations x that do not start 16-byte aligned, and then does not call visit
//   - a row of a block storage spans whole blocks of at least 32 weights, so each of its activation rows keeps x's alignment
template <typename Visit>
__device__ __forceinline__ bool visit_block_format(unsigned int storage, const float * x, Visit visit) {
    if ((reinterpret_cast<size_t>(x) & (sizeof(float4) - 1)) != 0) return false;
    return visit_block_storage(storage, visit);
}

// block_row returns the first block of the row of a bank that starts at weight first.
template <typename Format>
__device__ __forceinline__ const unsigned char * block_row(const void * bank, size_t first) {
    return reinterpret_cast<const unsigned char *>(bank) + first / Format::width * Format::bytes;
}

// A warp takes the groups of BLOCK_DOT_WEIGHTS weights of block_dot_step<Format>() blocks at once.
//   - a lane keeps one group of each block it takes, so the group's offsets stay fixed across blocks
template <typename Format>
__device__ __forceinline__ constexpr unsigned int block_dot_step() {
    static_assert(CUDA_WARP_WIDTH % (Format::width / BLOCK_DOT_WEIGHTS) == 0, "a warp covers whole blocks");
    return CUDA_WARP_WIDTH / (Format::width / BLOCK_DOT_WEIGHTS);
}

// block_rows_dot adds one lane's share of each row's dot with x to sums.
//   - rows of one storage share the activation reads
template <typename Format, unsigned int ROWS>
__device__ __forceinline__ void block_rows_dot(
        const unsigned char * const * rows, const float * x, unsigned int width, unsigned int lane,
        const block_dot_codebooks & codebooks, float * sums) {
    constexpr unsigned int groups = Format::width / BLOCK_DOT_WEIGHTS;
    const unsigned int group = lane % groups;
    #pragma unroll 1
    for (unsigned int block = lane / groups; block < width / Format::width; block += block_dot_step<Format>()) {
        #pragma unroll
        for (unsigned int row = 0; row < ROWS; ++row) {
            sums[row] += Format::dot(
                rows[row] + (size_t) block * Format::bytes, group, x + (size_t) block * Format::width, codebooks);
        }
    }
}

// moe_row_weights_dot gives one lane its share of the dot of x with a routed row, one weight a lane in turn.
//   - it stays out of line, so the block dots' loops keep their own registers
__device__ __noinline__ float moe_row_weights_dot(
        const void * bank, size_t first, unsigned int storage, const float * x, unsigned int width, unsigned int lane) {
    float sum = 0.0f;
    for (unsigned int channel = lane; channel < width; channel += CUDA_WARP_WIDTH) {
        sum += x[channel] * moe_expert_value(bank, first + channel, storage);
    }
    return sum;
}

// moe_row_dot gives one lane its share of the dot of x with the routed row that starts at weight first.
//   - a block storage takes block dots
//   - another storage, or x off 16-byte alignment, takes one weight a lane in turn
__device__ __forceinline__ float moe_row_dot(
        const void * bank, size_t first, unsigned int storage, const float * x, unsigned int width,
        unsigned int lane, const block_dot_codebooks & codebooks) {
    float sum = 0.0f;
    if (visit_block_format(storage, x, [&](auto format) {
            using Format = decltype(format);
            const unsigned char * row = block_row<Format>(bank, first);
            block_rows_dot<Format, 1>(&row, x, width, lane, codebooks, &sum);
        })) {
        return sum;
    }
    return moe_row_weights_dot(bank, first, storage, x, width, lane);
}

// block_slots_dot gives one lane its share of a token's routed rows, each dotted with its slot's activations and weighted.
//   - lane s < slots holds slot s's row and weight
//   - the warp takes the slots' blocks in turn across slot boundaries
//   - activations = the first slot's, and each slot's follow the previous slot's
template <typename Format>
__device__ __forceinline__ float block_slots_dot(
        const unsigned char * row, float weight, const float * activations, unsigned int slots,
        unsigned int width, unsigned int lane, const block_dot_codebooks & codebooks) {
    constexpr unsigned int groups = Format::width / BLOCK_DOT_WEIGHTS;
    const unsigned int row_blocks = width / Format::width;
    const unsigned int group = lane % groups;
    unsigned int slot = lane / groups / row_blocks;
    unsigned int block = lane / groups % row_blocks;
    float sum = 0.0f;
    #pragma unroll 2
    for (unsigned int base = 0; base < slots * row_blocks; base += block_dot_step<Format>()) {
        // Every lane shuffles, because a shuffle waits for the whole warp.
        const unsigned char * slot_row = reinterpret_cast<const unsigned char *>(
            __shfl_sync(0xffffffffu, reinterpret_cast<unsigned long long>(row), slot));
        const float slot_weight = __shfl_sync(0xffffffffu, weight, slot);
        if (slot < slots) {
            sum += slot_weight * Format::dot(slot_row + (size_t) block * Format::bytes, group,
                activations + (size_t) slot * width + (size_t) block * Format::width, codebooks);
        }
        block += block_dot_step<Format>();
        while (block >= row_blocks) {
            block -= row_blocks;
            ++slot;
        }
    }
    return sum;
}

// moe_down_channel sums one output channel of one token over the token's routed slots.
//   - a slot adds its weight times its down row's dot with its activations
//   - width = the lanes that share the channel, and lane 0 holds the sum
//   - codebooks != nullptr => a block storage takes block dots across the slots, and a warp shares the channel
//   - otherwise each lane takes every width-th weight and partial sums restart every MOE_DOWN_TILE rows
__device__ __forceinline__ float moe_down_channel(
        const unsigned int * route_state, const float * activations, const void * down,
        const float * expert_scale, const float * down_bias, unsigned int hidden, unsigned int top_k,
        unsigned int intermediate, unsigned int normalize_top_k, float routed_scale, unsigned int expert_storage,
        unsigned int expert_index_divisor, unsigned int indexed_experts, unsigned int index,
        unsigned int lane, unsigned int width, const block_dot_codebooks * codebooks) {
    const unsigned int output_channel = index % hidden;
    const unsigned int token = index / hidden;
    const unsigned int * selected = route_state + (size_t) token * (2 * top_k + 1);
    const float * routes = reinterpret_cast<const float *>(selected + top_k);
    const float * selected_sum = routes + top_k;
    const float * token_activations = activations + (size_t) token * top_k * intermediate;
    // A slot's weight = its route, scaled, normalized when asked, times its expert's scale.
    const auto slot_weight = [&](unsigned int slot, unsigned int expert) {
        float route = routes[slot] * routed_scale;
        if (normalize_top_k) route /= fmaxf(*selected_sum, 6.103515625e-5f);
        return route * (expert_scale ? expert_scale[expert] : 1.0f);
    };
    const auto slot_bank = [&](unsigned int expert) {
        return indexed_experts ? reinterpret_cast<const void * const *>(down)[expert] : down;
    };
    const auto slot_row = [&](unsigned int expert) {
        return ((size_t) (indexed_experts ? 0 : expert) * hidden + output_channel) * intermediate;
    };
    float result = 0.0f;
    if (codebooks) {
        // Lane s < top_k holds slot s, and top_k <= MaxMoETopK < CUDA_WARP_WIDTH.
        const unsigned int expert = selected[lane < top_k ? lane : 0] / expert_index_divisor;
        const float weight = slot_weight(lane < top_k ? lane : 0, expert);
        if (visit_block_format(expert_storage, token_activations, [&](auto format) {
                using Format = decltype(format);
                result = block_slots_dot<Format>(block_row<Format>(slot_bank(expert), slot_row(expert)), weight,
                    token_activations, top_k, intermediate, lane, *codebooks);
            })) {
            if (down_bias && lane < top_k) result += weight * down_bias[(size_t) expert * hidden + output_channel];
            return warp_sum_down(result);
        }
    }
    for (unsigned int slot = 0; slot < top_k; ++slot) {
        const unsigned int expert = selected[slot] / expert_index_divisor;
        const void * bank = slot_bank(expert);
        const float weight = slot_weight(slot, expert);
        const float * slot_activations = token_activations + (size_t) slot * intermediate;
        const size_t row = slot_row(expert);
        for (unsigned int tile = 0; tile < intermediate; tile += MOE_DOWN_TILE) {
            const unsigned int tile_count = intermediate - tile < MOE_DOWN_TILE ? intermediate - tile : MOE_DOWN_TILE;
            const size_t down_offset = row + tile;
            const float * tile_activations = slot_activations + tile;
            float partial = 0.0f;
            if (expert_storage == GGML_F32) {
                const float * down_row = reinterpret_cast<const float *>(bank) + down_offset;
                for (unsigned int inner = lane; inner < tile_count; inner += width) {
                    partial += tile_activations[inner] * down_row[inner];
                }
            } else if (expert_storage == GGML_BF16) {
                const unsigned short * down_row = reinterpret_cast<const unsigned short *>(bank) + down_offset;
                for (unsigned int inner = lane; inner < tile_count; inner += width) {
                    partial += tile_activations[inner] * __uint_as_float((unsigned int) down_row[inner] << 16);
                }
            } else {
                for (unsigned int inner = lane; inner < tile_count; inner += width) {
                    partial += tile_activations[inner] * moe_expert_value(bank, down_offset + inner, expert_storage);
                }
            }
            if (width > 1) partial = warp_sum_down(partial);
            if (tile + tile_count == intermediate && down_bias) {
                partial += down_bias[(size_t) expert * hidden + output_channel];
            }
            result += weight * partial;
        }
    }
    return result;
}

// moe_hidden_dot_f32 gives each routed row of a launch group a warp.
//   - the warp dots the row's gate and up weights with its token's input and stores the row's activation
//   - a block storage takes block dots
//   - another storage, or an input off 16-byte alignment, takes one weight a lane in turn
extern "C" __global__ void moe_hidden_dot_f32(
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
    __shared__ block_dot_codebooks codebooks;
    block_dot_codebooks_load(codebooks, gate_storage, up_storage);
    const unsigned int index = (blockIdx.x * blockDim.x + threadIdx.x) / CUDA_WARP_WIDTH;
    if (index >= count) return;
    const unsigned int lane = threadIdx.x % CUDA_WARP_WIDTH;
    const unsigned int inner_index = index % intermediate;
    const unsigned int selected = index / intermediate;
    const unsigned int slot = selected % top_k;
    const unsigned int token = selected / top_k;
    const float * x = input + (size_t) token * hidden;
    const unsigned int expert =
        route_state[(size_t) token * (2 * top_k + 1) + slot] / expert_index_divisor;
    if (indexed_experts) {
        if (gated) gate = reinterpret_cast<const void * const *>(gate)[expert];
        up = reinterpret_cast<const void * const *>(up)[expert];
    }
    const size_t gate_row =
        (size_t) (indexed_experts ? 0 : expert) * (fused_gate_up ? 2 : 1) * intermediate * hidden +
        (size_t) inner_index * hidden;
    const size_t up_row = gate_row + (fused_gate_up ? (size_t) intermediate * hidden : 0);
    // Gate and up rows of one block storage take one loop, so their reads are in flight together.
    float dots[2] = {0.0f, 0.0f};
    const bool paired = gated && gate_storage == up_storage && visit_block_format(up_storage, x, [&](auto format) {
        using Format = decltype(format);
        const unsigned char * rows[2] = {block_row<Format>(gate, gate_row), block_row<Format>(up, up_row)};
        block_rows_dot<Format, 2>(rows, x, hidden, lane, codebooks, dots);
    });
    float gate_dot = dots[0];
    float up_dot = dots[1];
    if (!paired) {
        if (gated) gate_dot = moe_row_dot(gate, gate_row, gate_storage, x, hidden, lane, codebooks);
        up_dot = moe_row_dot(up, up_row, up_storage, x, hidden, lane, codebooks);
    }
    gate_dot = warp_sum_down(gate_dot);
    up_dot = warp_sum_down(up_dot);
    if (lane != 0) return;
    if (gate_bias) {
        gate_dot += gate_bias[(size_t) expert * intermediate + inner_index];
        up_dot += up_bias[(size_t) expert * intermediate + inner_index];
    }
    activations[index] = moe_activation(gate_dot, up_dot, gated, activation, routing, swiglu_clamp);
}

// moe_down_dot_f32 gives each output channel of a launch group's tokens a warp, which sums the channel over its routed slots.
//   - a channel's tokens take adjacent warps, so the rows of the experts they share are read together
extern "C" __global__ void moe_down_dot_f32(
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
    __shared__ block_dot_codebooks codebooks;
    block_dot_codebooks_load(codebooks, expert_storage, expert_storage);
    const unsigned int warp = (blockIdx.x * blockDim.x + threadIdx.x) / CUDA_WARP_WIDTH;
    if (warp >= count) return;
    const unsigned int lane = threadIdx.x % CUDA_WARP_WIDTH;
    const unsigned int tokens = count / hidden;
    const unsigned int index = warp % tokens * hidden + warp / tokens;
    const float result = moe_down_channel(route_state, activations, down, expert_scale, down_bias, hidden, top_k,
        intermediate, normalize_top_k, routed_scale, expert_storage, expert_index_divisor, indexed_experts,
        index, lane, CUDA_WARP_WIDTH, &codebooks);
    if (lane == 0) output[index] = result;
}

// ---- Routed experts on q8 activations ----
// A decode group's routed products take its input and its slots' activations as q8 blocks (Q8_INPUT_BLOCK_BYTES: a float
// scale, then MOE_Q8_WIDTH int8) and dot a routed row's MOE_Q8_WIDTH-weight sub-blocks with them in integers, as llama.cpp's
// vec_dot_*_q8_1 dots and Strata's native expert kernels do (Strata src/kernels/cuda/iq_kernels.cu).
//   - a sub-block's dot takes dp4a over eight words of whole numbers: a block format's codebook values fit int8
//   - its weight scale and the q8 scale then weigh the integer sum in F32
constexpr unsigned int MOE_Q8_WIDTH = 32;
constexpr unsigned int MOE_Q8_WORDS = MOE_Q8_WIDTH / 4;
// MOE_Q8_ROW_LANES: the lanes that share a routed row, half a warp; Flash-Next's 2,560-wide rows give each lane five
// sub-blocks, as Strata's 16-lane rows do.
constexpr unsigned int MOE_Q8_ROW_LANES = CUDA_WARP_WIDTH / 2;
// MOE_SPAN_TOKENS: the most tokens of a group that read their shared experts' rows once, a verify span's.
//   - a routed slot, token * top_k + slot, takes MOE_SPAN_SLOT_BITS bits of a packed word
constexpr unsigned int MOE_SPAN_TOKENS = 4;
constexpr unsigned int MOE_SPAN_SLOT_BITS = 8;

// q8_scale returns a q8 block's scale.
__device__ __forceinline__ float q8_scale(const unsigned char * block) {
    return *reinterpret_cast<const float *>(block);
}

// q8_values loads a q8 block's values as eight words.
__device__ __forceinline__ void q8_values(const unsigned char * block, int * x) {
    #pragma unroll
    for (unsigned int word = 0; word < MOE_Q8_WORDS; ++word) {
        x[word] = reinterpret_cast<const int *>(block + sizeof(float))[word];
    }
}

// iq4_q8_values returns eight packed IQ4 indices' int8 codebook values: the low nibbles' word, then the high nibbles'.
//   - the sixteen values sit in four words, which byte permutes select from (llama.cpp's get_int_from_table_16)
__device__ __forceinline__ int2 iq4_q8_values(unsigned int packed) {
    // -127 -104 -83 -65 | -49 -35 -22 -10 | 1 13 25 38 | 53 69 89 113, iq4nl_values as int8
    constexpr unsigned int table0 = 0xbfad9881u, table1 = 0xf6eaddcfu, table2 = 0x26190d01u, table3 = 0x71594535u;
    const unsigned int selection = 0x32103210u | ((packed & 0x88888888u) >> 1);
    unsigned int parts[2];
    #pragma unroll
    for (unsigned int half = 0; half < 2; ++half) {
        const unsigned int shift = 16 * half;
        const unsigned int low = __byte_perm(table0, table1, packed >> shift);
        const unsigned int high = __byte_perm(table2, table3, packed >> shift);
        parts[half] = __byte_perm(low, high, selection >> shift);
    }
    return make_int2((int) __byte_perm(parts[0], parts[1], 0x6420), (int) __byte_perm(parts[0], parts[1], 0x7531));
}

// iq4_q8_dot dots four words of packed IQ4 indices with x: each word's low nibbles x[word], its high nibbles x[word + 4].
template <typename Packed>
__device__ __forceinline__ int iq4_q8_dot(Packed packed, const int * x) {
    int sum = 0;
    #pragma unroll
    for (unsigned int word = 0; word < 4; ++word) {
        const int2 values = iq4_q8_values(packed(word));
        sum = __dp4a(values.x, x[word], sum);
        sum = __dp4a(values.y, x[word + 4], sum);
    }
    return sum;
}

// A q8 format names a block storage's MOE_Q8_WIDTH-weight sub-blocks of one scale.
//   - width and bytes = the block's, as its block format's
//   - dot(block, sub, x, codebooks, scale) returns sub-block sub's integer dot with x and sets scale to its weight scale

// iq3_s_q8: a sub-block's eight grid entries and their signs, as block_iq3_s holds them.
struct iq3_s_q8 {
    static constexpr unsigned int width = iq3_s_blocks::width;
    static constexpr unsigned int bytes = iq3_s_blocks::bytes;
    __device__ static __forceinline__ int dot(
            const unsigned char * block, unsigned int sub, const int * x, const block_dot_codebooks & codebooks,
            float & scale) {
        const unsigned int indices[2] = {load_u32_halves(block + 2 + sub * 8), load_u32_halves(block + 6 + sub * 8)};
        const unsigned int high = load_u8(block + 66 + sub);
        const unsigned int signs = load_u32_halves(block + 74 + sub * 4);
        int sum = 0;
        #pragma unroll
        for (unsigned int entry = 0; entry < 8; entry += 2) {
            const unsigned int pair = indices[entry / 4] >> (8 * (entry % 4));
            const unsigned int first = codebooks.iq3_s[(pair & 0xff) | ((high << (8 - entry)) & 0x100)];
            const unsigned int second = codebooks.iq3_s[((pair >> 8) & 0xff) | ((high << (7 - entry)) & 0x100)];
            const unsigned int sign = (signs >> (4 * entry)) & 0xff;
            // A byte whose sign bit is set becomes 0xff, and x ^ m - m negates the bytes m marks.
            const unsigned int first_signs = __vcmpne4(((sign & 0x03) << 7) | ((sign & 0x0c) << 21), 0);
            const unsigned int second_signs = __vcmpne4(((sign & 0x30) << 3) | ((sign & 0xc0) << 17), 0);
            sum = __dp4a((int) __vsub4(first ^ first_signs, first_signs), x[entry], sum);
            sum = __dp4a((int) __vsub4(second ^ second_signs, second_signs), x[entry + 1], sum);
        }
        scale = load_half(block) * (float) (1 + 2 * ((load_u8(block + 106 + sub / 2) >> (4 * (sub % 2))) & 0x0f));
        return sum;
    }
};

// iq4_xs_q8: a sub-block's sixteen bytes of nibbles and its six-bit scale.
struct iq4_xs_q8 {
    static constexpr unsigned int width = iq4_xs_blocks::width;
    static constexpr unsigned int bytes = iq4_xs_blocks::bytes;
    __device__ static __forceinline__ int dot(
            const unsigned char * block, unsigned int sub, const int * x, const block_dot_codebooks &, float & scale) {
        const unsigned int * packed = reinterpret_cast<const unsigned int *>(block + 8 + 16 * sub);
        scale = iq4_xs_blocks::sub_block_scale(block, sub);
        return iq4_q8_dot([&](unsigned int word) { return __ldg(packed + word); }, x);
    }
};

// iq4_nl_q8: the block's sixteen bytes of nibbles.
struct iq4_nl_q8 {
    static constexpr unsigned int width = iq4_nl_blocks::width;
    static constexpr unsigned int bytes = iq4_nl_blocks::bytes;
    __device__ static __forceinline__ int dot(
            const unsigned char * block, unsigned int, const int * x, const block_dot_codebooks &, float & scale) {
        scale = load_half(block);
        return iq4_q8_dot([&](unsigned int word) { return load_u32_halves(block + 2 + 4 * word); }, x);
    }
};

// q8_0_q8: the block's 32 signed bytes.
struct q8_0_q8 {
    static constexpr unsigned int width = q8_0_blocks::width;
    static constexpr unsigned int bytes = q8_0_blocks::bytes;
    __device__ static __forceinline__ int dot(
            const unsigned char * block, unsigned int, const int * x, const block_dot_codebooks &, float & scale) {
        int sum = 0;
        #pragma unroll
        for (unsigned int word = 0; word < MOE_Q8_WORDS; ++word) {
            sum = __dp4a((int) load_u32_halves(block + Q8_0_SCALE_BYTES + 4 * word), x[word], sum);
        }
        scale = load_half(block);
        return sum;
    }
};

// visit_q8_format calls visit with the q8 format of a storage, and refuses a storage without one.
//   - the routed-expert launch names the same storages (moeQ8Storage)
template <typename Visit>
__device__ __forceinline__ bool visit_q8_format(unsigned int storage, Visit visit) {
    switch (storage) {
    case GGML_IQ3_S: visit(iq3_s_q8{}); return true;
    case GGML_IQ4_XS: visit(iq4_xs_q8{}); return true;
    case GGML_IQ4_NL: visit(iq4_nl_q8{}); return true;
    case GGML_Q8_0: visit(q8_0_q8{}); return true;
    default: return false;
    }
}

// moe_q8_rows_dot adds a lane's share of two rows' dots with each of COLUMNS q8 columns to sums[row][column].
//   - the lane takes sub-blocks first, first + step, ... of the rows
//   - a column's sums do not depend on the other columns; the rows' reads and decode, alike for every column, are the
//     compiler's to share
template <typename Format, unsigned int COLUMNS>
__device__ __forceinline__ void moe_q8_rows_dot(
        const unsigned char * const * rows, const unsigned char * const * columns, unsigned int sub_blocks,
        unsigned int first, unsigned int step, const block_dot_codebooks & codebooks, float (*sums)[COLUMNS]) {
    constexpr unsigned int subs = Format::width / MOE_Q8_WIDTH;
    #pragma unroll 1
    for (unsigned int sub = first; sub < sub_blocks; sub += step) {
        int x[COLUMNS][MOE_Q8_WORDS];
        float x_scale[COLUMNS];
        #pragma unroll
        for (unsigned int column = 0; column < COLUMNS; ++column) {
            const unsigned char * values = columns[column] + (size_t) sub * Q8_INPUT_BLOCK_BYTES;
            q8_values(values, x[column]);
            x_scale[column] = q8_scale(values);
        }
        #pragma unroll
        for (unsigned int row = 0; row < 2; ++row) {
            const unsigned char * block = rows[row] + (size_t) (sub / subs) * Format::bytes;
            #pragma unroll
            for (unsigned int column = 0; column < COLUMNS; ++column) {
                float scale;
                const int dot = Format::dot(block, sub % subs, x[column], codebooks, scale);
                sums[row][column] += scale * x_scale[column] * (float) dot;
            }
        }
    }
}

// moe_q8_row_sum adds a value across its half warp, whose first lane holds the sum.
__device__ __forceinline__ float moe_q8_row_sum(float value) {
    const unsigned int mask = 0xffffu << (threadIdx.x % CUDA_WARP_WIDTH / MOE_Q8_ROW_LANES * MOE_Q8_ROW_LANES);
    for (unsigned int offset = MOE_Q8_ROW_LANES / 2; offset > 0; offset /= 2) {
        value += __shfl_down_sync(mask, value, offset, MOE_Q8_ROW_LANES);
    }
    return value;
}

// moe_span_slot is column's routed slot among slots: the whole slot where one column is stored, else the column's
// MOE_SPAN_SLOT_BITS bits of a span's packed word (a group of 26 Flash-Next tokens routes 260 slots, past those bits).
__device__ __forceinline__ unsigned int moe_span_slot(unsigned int slots, unsigned int columns, unsigned int column) {
    return columns == 1 ? slots : slots >> (MOE_SPAN_SLOT_BITS * column) & ((1u << MOE_SPAN_SLOT_BITS) - 1);
}

// moe_q8_hidden stores the activations of one expert's gate and up row for each of COLUMNS routed slots.
//   - slots is one column's slot, or packs a span's, MOE_SPAN_SLOT_BITS bits each, the first in the low bits
//   - input holds each token's q8 blocks, hidden / MOE_Q8_WIDTH a token
template <typename Format, unsigned int COLUMNS>
__device__ __forceinline__ void moe_q8_hidden(
        const unsigned char * gate_row, const unsigned char * up_row, const unsigned char * input, unsigned int slots,
        const float * gate_bias, const float * up_bias, float * activations, unsigned int hidden, unsigned int top_k,
        unsigned int intermediate, unsigned int inner_index, unsigned int expert, unsigned int routing,
        unsigned int activation, float swiglu_clamp, unsigned int lane, const block_dot_codebooks & codebooks) {
    const unsigned char * rows[2] = {gate_row, up_row};
    const unsigned char * columns[COLUMNS];
    #pragma unroll
    for (unsigned int column = 0; column < COLUMNS; ++column) {
        const unsigned int slot = moe_span_slot(slots, COLUMNS, column);
        columns[column] = input + (size_t) (slot / top_k) * (hidden / MOE_Q8_WIDTH) * Q8_INPUT_BLOCK_BYTES;
    }
    float dots[2][COLUMNS] = {};
    moe_q8_rows_dot<Format, COLUMNS>(rows, columns, hidden / MOE_Q8_WIDTH, lane, MOE_Q8_ROW_LANES, codebooks, dots);
    #pragma unroll
    for (unsigned int column = 0; column < COLUMNS; ++column) {
        float gate_dot = moe_q8_row_sum(dots[0][column]);
        float up_dot = moe_q8_row_sum(dots[1][column]);
        if (lane == 0) {
            if (gate_bias) {
                gate_dot += gate_bias[(size_t) expert * intermediate + inner_index];
                up_dot += up_bias[(size_t) expert * intermediate + inner_index];
            }
            const unsigned int slot = moe_span_slot(slots, COLUMNS, column);
            activations[(size_t) slot * intermediate + inner_index] =
                moe_activation(gate_dot, up_dot, 1, activation, routing, swiglu_clamp);
        }
    }
}

// moe_q8_banks points a routed row's gate and up at its expert's blocks.
template <typename Format>
__device__ __forceinline__ void moe_q8_banks(
        const void * gate, const void * up, unsigned int expert, unsigned int hidden, unsigned int intermediate,
        unsigned int inner_index, unsigned int fused_gate_up, unsigned int indexed_experts,
        const unsigned char *& gate_row, const unsigned char *& up_row) {
    if (indexed_experts) {
        gate = reinterpret_cast<const void * const *>(gate)[expert];
        up = reinterpret_cast<const void * const *>(up)[expert];
    }
    const size_t first =
        (size_t) (indexed_experts ? 0 : expert) * (fused_gate_up ? 2 : 1) * intermediate * hidden +
        (size_t) inner_index * hidden;
    gate_row = block_row<Format>(gate, first);
    up_row = block_row<Format>(up, first + (fused_gate_up ? (size_t) intermediate * hidden : 0));
}

// moe_q8_columns stores routed row index's activation for each of the columns slots packs, its half warp's.
//   - a storage without a q8 format stores NaN, which no caller's launch names
__device__ __forceinline__ void moe_q8_columns(
        const unsigned char * input, const void * gate, const void * up, const float * gate_bias,
        const float * up_bias, float * activations, unsigned int hidden, unsigned int top_k, unsigned int intermediate,
        unsigned int routing, unsigned int storage, unsigned int fused_gate_up, unsigned int activation,
        float swiglu_clamp, unsigned int indexed_experts, unsigned int expert, unsigned int inner_index,
        unsigned int slots, unsigned int columns, unsigned int lane, const block_dot_codebooks & codebooks) {
    if (!visit_q8_format(storage, [&](auto format) {
            using Format = decltype(format);
            const unsigned char * gate_row;
            const unsigned char * up_row;
            moe_q8_banks<Format>(gate, up, expert, hidden, intermediate, inner_index, fused_gate_up, indexed_experts,
                gate_row, up_row);
            switch (columns) {
            case 1:
                moe_q8_hidden<Format, 1>(gate_row, up_row, input, slots, gate_bias, up_bias, activations, hidden,
                    top_k, intermediate, inner_index, expert, routing, activation, swiglu_clamp, lane, codebooks);
                break;
            case 2:
                moe_q8_hidden<Format, 2>(gate_row, up_row, input, slots, gate_bias, up_bias, activations, hidden,
                    top_k, intermediate, inner_index, expert, routing, activation, swiglu_clamp, lane, codebooks);
                break;
            case 3:
                moe_q8_hidden<Format, 3>(gate_row, up_row, input, slots, gate_bias, up_bias, activations, hidden,
                    top_k, intermediate, inner_index, expert, routing, activation, swiglu_clamp, lane, codebooks);
                break;
            default:
                moe_q8_hidden<Format, MOE_SPAN_TOKENS>(gate_row, up_row, input, slots, gate_bias, up_bias,
                    activations, hidden, top_k, intermediate, inner_index, expert, routing, activation,
                    swiglu_clamp, lane, codebooks);
            }
        }) && lane == 0) {
        activations[(size_t) moe_span_slot(slots, columns, 0) * intermediate + inner_index] = __int_as_float(0x7fc00000);
    }
}

// ---- The expert doorbell ----
// A decode group's routed experts in host memory are computed by the CPU beside the card, as Strata's doorbell does
// (src/core/layer.cpp, THE DOORBELL): the first block of the group's hidden products hands the CPU the group's routes
// and inputs in host memory the card maps and rings, the card's products skip the experts on_host marks, and the last
// block of its down products to finish waits for the CPU's answer and adds the CPU's slots, each weighed as
// moe_down_q8_f32 weighs a slot. Neither takes a launch of its own: a decode step's 48 routed nodes would add 96.
//   - words, host memory the card maps: the ring, the CPU's answer, the last ring that handed the CPU slots, the ring
//     a wait gave up on, and the host's abandonment of the run
//   - the CPU answers a ring with its number once its slots' products are written; rings and answers only grow, and a
//     ring that hands slots over holds the card until its answer, so the CPU reads a later ring's words only after it
constexpr unsigned int MOE_DOORBELL_RING = 0;
constexpr unsigned int MOE_DOORBELL_ANSWER = 1;
constexpr unsigned int MOE_DOORBELL_WORK = 2;
constexpr unsigned int MOE_DOORBELL_FAILED = 3;
constexpr unsigned int MOE_DOORBELL_ABANDONED = 4;
// MOE_DOORBELL_HOST_SLOT marks a handed-over slot's expert the CPU computes.
constexpr unsigned int MOE_DOORBELL_HOST_SLOT = 0x80000000u;
// MOE_DOORBELL_WAIT_NS: the longest a wait for an answer lasts, well inside the display watchdog's two seconds.
constexpr unsigned long long MOE_DOORBELL_WAIT_NS = 250000000ull;
// MOE_DOORBELL_SLOTS: the most routed slots a group hands over; MOE_DOORBELL_VECTORS: the answer's float4s a thread
// of the last block reads at once.
constexpr unsigned int MOE_DOORBELL_SLOTS = 128;
constexpr unsigned int MOE_DOORBELL_VECTORS = 4;

// moe_doorbell is a doorbell's card memory: the rings so far, the ring the group's down products wait for (0: none),
// the down products' blocks finished, then the addresses of the words, the slot ids, the input rows and the answer
// rows in host memory the card maps.
//   - ids[t * top_k + s] = token t's slot s's routed expert, with MOE_DOORBELL_HOST_SLOT where on_host marks it,
//     rows[t * hidden + c] = token t's input, answer[(t * top_k + s) * hidden + c] = the CPU's down product of the slot
struct moe_doorbell {
    unsigned int rings;
    unsigned int waiting;
    unsigned int finished;
    unsigned int padding;
    unsigned int * words;
    unsigned int * ids;
    float * rows;
    const float * answer;
};

// global_nanoseconds reads the card's global timer.
__device__ __forceinline__ unsigned long long global_nanoseconds() {
    unsigned long long now;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
    return now;
}

// moe_doorbell_publish: a block rings for a decode group of tokens, handing the CPU the group's routes and inputs when
// one of its routed experts is on host. Every thread of the block calls it.
//   - every writer fences its writes to the system before the block's barrier, so the ring follows them on the link
__device__ __noinline__ void moe_doorbell_publish(
        const unsigned int * route_state, const unsigned char * on_host, const float * input, moe_doorbell * doorbell,
        unsigned int tokens, unsigned int top_k, unsigned int hidden, unsigned int expert_index_divisor) {
    const unsigned int token_words = 2 * top_k + 1;
    bool hosted = false;
    for (unsigned int index = threadIdx.x; index < tokens * top_k; index += blockDim.x) {
        hosted = hosted ||
            on_host[route_state[(size_t) (index / top_k) * token_words + index % top_k] / expert_index_divisor] != 0;
    }
    const bool work = __syncthreads_or(hosted) != 0;
    if (work) {
        for (unsigned int index = threadIdx.x; index < tokens * top_k; index += blockDim.x) {
            const unsigned int id = route_state[(size_t) (index / top_k) * token_words + index % top_k];
            doorbell->ids[index] = on_host[id / expert_index_divisor] ? id | MOE_DOORBELL_HOST_SLOT : id;
        }
        for (unsigned int index = threadIdx.x; index < tokens * hidden; index += blockDim.x) {
            doorbell->rows[index] = input[index];
        }
        __threadfence_system();
    }
    __syncthreads();
    if (threadIdx.x != 0) return;
    volatile unsigned int * words = doorbell->words;
    const unsigned int ring = doorbell->rings + 1;
    doorbell->rings = ring;
    doorbell->waiting = work ? ring : 0u;
    if (work) words[MOE_DOORBELL_WORK] = ring;
    __threadfence_system();
    words[MOE_DOORBELL_RING] = ring;
}

// moe_doorbell_add: the last block of a group's down products to finish waits for the CPU's answer to the group's
// ring and adds the CPU's slots to every output. Every thread of every block calls it once its outputs are written.
//   - a block's outputs are fenced before it counts itself finished, so the last block reads every block's
//   - the answer is read past the card's caches, float4s at once, the rows' latency over the link overlapped
//   - a wait past MOE_DOORBELL_WAIT_NS, or the host's abandonment, records the ring and writes NaN
__device__ __noinline__ void moe_doorbell_add(
        const unsigned int * route_state, const unsigned char * on_host, const float * expert_scale,
        moe_doorbell * doorbell, float * output, unsigned int tokens, unsigned int hidden, unsigned int top_k,
        unsigned int normalize_top_k, float routed_scale, unsigned int expert_index_divisor) {
    __shared__ bool last;
    __shared__ unsigned int host_slots;
    __shared__ unsigned int slot_index[MOE_DOORBELL_SLOTS];
    __shared__ float slot_weight[MOE_DOORBELL_SLOTS];
    const unsigned int ring = doorbell->waiting;
    if (ring == 0u) return;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) last = atomicAdd(&doorbell->finished, 1u) == gridDim.x - 1;
    __syncthreads();
    if (!last) return;
    if (threadIdx.x == 0) {
        doorbell->finished = 0;
        volatile unsigned int * words = doorbell->words;
        const unsigned long long start = global_nanoseconds();
        bool failed = false;
        while ((int) (words[MOE_DOORBELL_ANSWER] - ring) < 0) {
            if (words[MOE_DOORBELL_ABANDONED] != 0u || global_nanoseconds() - start > MOE_DOORBELL_WAIT_NS) {
                failed = true;
                break;
            }
        }
        if (failed) words[MOE_DOORBELL_FAILED] = ring;
        __threadfence();
        // Each host slot's weight = its route, scaled, normalized when asked, times its expert's scale.
        unsigned int count = 0;
        for (unsigned int slot = 0; slot < tokens * top_k && !failed; ++slot) {
            const unsigned int * selected = route_state + (size_t) (slot / top_k) * (2 * top_k + 1);
            const float * routes = reinterpret_cast<const float *>(selected + top_k);
            const unsigned int expert = selected[slot % top_k] / expert_index_divisor;
            if (!on_host[expert]) continue;
            float weight = routes[slot % top_k] * routed_scale;
            if (normalize_top_k) weight /= fmaxf(routes[top_k], 6.103515625e-5f);
            weight *= expert_scale ? expert_scale[expert] : 1.0f;
            slot_index[count] = slot;
            slot_weight[count++] = weight;
        }
        host_slots = failed ? MOE_DOORBELL_SLOTS + 1 : count;
    }
    __syncthreads();
    const unsigned int vectors = tokens * hidden / 4;
    float4 * output4 = reinterpret_cast<float4 *>(output);
    const float4 * answer4 = reinterpret_cast<const float4 *>(doorbell->answer);
    for (unsigned int first = threadIdx.x; first < vectors; first += blockDim.x * MOE_DOORBELL_VECTORS) {
        float4 sums[MOE_DOORBELL_VECTORS] = {};
        for (unsigned int host = 0; host < host_slots && host_slots <= MOE_DOORBELL_SLOTS; ++host) {
            const unsigned int slot = slot_index[host];
            const float weight = slot_weight[host];
#pragma unroll
            for (unsigned int part = 0; part < MOE_DOORBELL_VECTORS; ++part) {
                const unsigned int vector = first + part * blockDim.x;
                if (vector >= vectors || vector * 4 / hidden != slot / top_k) continue;
                const float4 value = __ldcv(answer4 + (size_t) slot * (hidden / 4) + vector % (hidden / 4));
                sums[part].x += weight * value.x;
                sums[part].y += weight * value.y;
                sums[part].z += weight * value.z;
                sums[part].w += weight * value.w;
            }
        }
#pragma unroll
        for (unsigned int part = 0; part < MOE_DOORBELL_VECTORS; ++part) {
            const unsigned int vector = first + part * blockDim.x;
            if (vector >= vectors) continue;
            float4 value = __ldcg(output4 + vector);
            if (host_slots > MOE_DOORBELL_SLOTS) {
                value = make_float4(__int_as_float(0x7fc00000), __int_as_float(0x7fc00000), __int_as_float(0x7fc00000),
                    __int_as_float(0x7fc00000));
            } else {
                value.x += sums[part].x;
                value.y += sums[part].y;
                value.z += sums[part].z;
                value.w += sums[part].w;
            }
            output4[vector] = value;
        }
    }
}

// moe_hidden_q8_f32 gives each routed row of a decode group half a warp, which stores the row's activation.
//   - input = the group's q8 blocks, and gate and up share a q8 format
//   - on_host != nullptr => a row of an expert it marks is the CPU's, and its half warp stores nothing
//   - doorbell != nullptr => the first block rings it for the group, handing over doorbell_input (moe_doorbell_publish)
extern "C" __global__ void moe_hidden_q8_f32(
        const unsigned char * input,
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
        unsigned int count,
        const unsigned char * on_host,
        moe_doorbell * doorbell,
        const float * doorbell_input) {
    if (doorbell && blockIdx.x == 0) {
        moe_doorbell_publish(route_state, on_host, doorbell_input, doorbell, count / (top_k * intermediate), top_k, hidden,
            expert_index_divisor);
    }
    __shared__ block_dot_codebooks codebooks;
    block_dot_codebooks_load(codebooks, gate_storage, up_storage);
    const unsigned int index = (blockIdx.x * blockDim.x + threadIdx.x) / MOE_Q8_ROW_LANES;
    if (index >= count) return;
    const unsigned int selected = index / intermediate;
    const unsigned int expert =
        route_state[(size_t) (selected / top_k) * (2 * top_k + 1) + selected % top_k] / expert_index_divisor;
    if (on_host && on_host[expert]) return;
    moe_q8_columns(input, gate, up, gate_bias, up_bias, activations, hidden, top_k, intermediate, routing, up_storage,
        fused_gate_up, activation, swiglu_clamp, indexed_experts, expert, index % intermediate, selected, 1,
        threadIdx.x % MOE_Q8_ROW_LANES, codebooks);
}

// moe_hidden_span_q8_f32 is moe_hidden_q8_f32 for a group of at most MOE_SPAN_TOKENS tokens, which reads each routed
// expert's gate and up rows once for all the group's tokens that route to it.
//   - the half warp of a row's first token that routes to its expert dots it with that token's input and every later
//     one's, and a later token's half warp leaves the row to it
//   - a token that routes twice to the expert, through the index divisor or repeated selections, leaves each of the
//     expert's half warps its own row, as every one of them finds
//   - a token's activations are bit for bit those it computes alone
//   - a row of an expert on_host marks is the CPU's for every token, as in moe_hidden_q8_f32
extern "C" __global__ void moe_hidden_span_q8_f32(
        const unsigned char * input,
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
        unsigned int count,
        const unsigned char * on_host,
        moe_doorbell * doorbell,
        const float * doorbell_input) {
    if (doorbell && blockIdx.x == 0) {
        moe_doorbell_publish(route_state, on_host, doorbell_input, doorbell, count / (top_k * intermediate), top_k, hidden,
            expert_index_divisor);
    }
    __shared__ block_dot_codebooks codebooks;
    block_dot_codebooks_load(codebooks, gate_storage, up_storage);
    const unsigned int index = (blockIdx.x * blockDim.x + threadIdx.x) / MOE_Q8_ROW_LANES;
    if (index >= count) return;
    const unsigned int lane = threadIdx.x % MOE_Q8_ROW_LANES;
    const unsigned int half = threadIdx.x % CUDA_WARP_WIDTH / MOE_Q8_ROW_LANES * MOE_Q8_ROW_LANES;
    const unsigned int tokens = count / (top_k * intermediate);
    const unsigned int selected = index / intermediate;
    const unsigned int token = selected / top_k;
    const unsigned int token_words = 2 * top_k + 1;
    const unsigned int expert = route_state[(size_t) token * token_words + selected % top_k] / expert_index_divisor;
    // The half warp's lanes share the row, so they leave together, before the ballots.
    if (on_host && on_host[expert]) return;
    // Lane s of the half reads slots s, s + MOE_Q8_ROW_LANES, ... of each token.
    unsigned int slots = selected;
    unsigned int columns = 1;
    bool earlier = false;
    bool repeated = false;
    for (unsigned int other = 0; other < tokens && other < MOE_SPAN_TOKENS; ++other) {
        unsigned int routes = 0;
        for (unsigned int first = 0; first < top_k; first += MOE_Q8_ROW_LANES) {
            const unsigned int slot = first + lane;
            const bool routed = slot < top_k &&
                route_state[(size_t) other * token_words + slot] / expert_index_divisor == expert;
            routes |= (__ballot_sync(0xffffu << half, routed) >> half & 0xffffu) << first;
        }
        repeated |= __popc(routes) > 1;
        if (other == token || routes == 0) continue;
        earlier |= other < token;
        if (other > token) slots |= (other * top_k + __ffs(routes) - 1) << (MOE_SPAN_SLOT_BITS * columns++);
    }
    if (repeated) {
        slots = selected;
        columns = 1;
    } else if (earlier) {
        return;
    }
    moe_q8_columns(input, gate, up, gate_bias, up_bias, activations, hidden, top_k, intermediate, routing, up_storage,
        fused_gate_up, activation, swiglu_clamp, indexed_experts, expert, index % intermediate, slots, columns, lane,
        codebooks);
}

// moe_down_q8_f32 gives each output channel of a decode group's tokens a warp, which sums the channel over its routed
// slots' sub-blocks against their q8 activations.
//   - activations = the group's q8 blocks, intermediate / MOE_Q8_WIDTH a routed slot
//   - a channel's tokens take adjacent warps, so the rows of the experts they share are read together
//   - a slot of an expert on_host marks is the CPU's, which the last block to finish adds once the CPU answers (moe_doorbell_add)
//   - its launch's blocks are MOE_DOWN_THREADS wide, and an SM keeps MOE_DOWN_RESIDENT_BLOCKS of them, as many as before
//     the doorbell's epilogue, whose registers would otherwise cost one
constexpr unsigned int MOE_DOWN_THREADS = 256;
constexpr unsigned int MOE_DOWN_RESIDENT_BLOCKS = 6;
extern "C" __global__ void __launch_bounds__(MOE_DOWN_THREADS, MOE_DOWN_RESIDENT_BLOCKS) moe_down_q8_f32(
        const unsigned int * route_state,
        const unsigned char * activations,
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
        unsigned int count,
        const unsigned char * on_host,
        moe_doorbell * doorbell) {
    __shared__ block_dot_codebooks codebooks;
    block_dot_codebooks_load(codebooks, expert_storage, expert_storage);
    const unsigned int warp = (blockIdx.x * blockDim.x + threadIdx.x) / CUDA_WARP_WIDTH;
    if (warp < count) {
        const unsigned int lane = threadIdx.x % CUDA_WARP_WIDTH;
        const unsigned int tokens = count / hidden;
        const unsigned int index = warp % tokens * hidden + warp / tokens;
        const unsigned int output_channel = index % hidden;
        const unsigned int token = index / hidden;
        const unsigned int * selected = route_state + (size_t) token * (2 * top_k + 1);
        const float * routes = reinterpret_cast<const float *>(selected + top_k);
        const float selected_sum = routes[top_k];
        const unsigned int row_subs = intermediate / MOE_Q8_WIDTH;
        const unsigned char * token_values = activations + (size_t) token * top_k * row_subs * Q8_INPUT_BLOCK_BYTES;
        float sum = 0.0f;
        if (!visit_q8_format(expert_storage, [&](auto format) {
                using Format = decltype(format);
                constexpr unsigned int subs = Format::width / MOE_Q8_WIDTH;
                for (unsigned int item = lane; item < top_k * row_subs; item += CUDA_WARP_WIDTH) {
                    const unsigned int slot = item / row_subs;
                    const unsigned int sub = item % row_subs;
                    const unsigned int expert = selected[slot] / expert_index_divisor;
                    if (on_host && on_host[expert]) continue;
                    // A slot's weight = its route, scaled, normalized when asked, times its expert's scale.
                    float weight = routes[slot] * routed_scale;
                    if (normalize_top_k) weight /= fmaxf(selected_sum, 6.103515625e-5f);
                    weight *= expert_scale ? expert_scale[expert] : 1.0f;
                    const void * bank = indexed_experts ? reinterpret_cast<const void * const *>(down)[expert] : down;
                    const unsigned char * row = block_row<Format>(
                        bank, ((size_t) (indexed_experts ? 0 : expert) * hidden + output_channel) * intermediate);
                    const unsigned char * values = token_values + (size_t) (slot * row_subs + sub) * Q8_INPUT_BLOCK_BYTES;
                    int x[MOE_Q8_WORDS];
                    q8_values(values, x);
                    float scale;
                    const int dot = Format::dot(row + (size_t) (sub / subs) * Format::bytes, sub % subs, x, codebooks, scale);
                    float product = scale * q8_scale(values) * (float) dot;
                    if (down_bias && sub == 0) product += down_bias[(size_t) expert * hidden + output_channel];
                    sum += weight * product;
                }
            })) {
            sum = __int_as_float(0x7fc00000);
        }
        sum = warp_sum_down(sum);
        if (lane == 0) output[index] = sum;
    }
    if (doorbell) {
        moe_doorbell_add(route_state, on_host, expert_scale, doorbell, output, count / hidden, hidden, top_k,
            normalize_top_k, routed_scale, expert_index_divisor);
    }
}

// ---- Grouped expert products ----
// A prompt group's routed rows, ordered by expert, multiply as tiles on the tensor cores.
//   - tiles[MOE_TILE_WORDS * t] = the tile's expert, then its first position and its rows, at most MOE_TILE_TOKENS
//   - positions[p] = the routed slot, token * top_k + slot, of the p-th routed row in expert order
//   - a block computes MOE_TILE_ROWS weight rows of one tile, and a warp MOE_MMA_ROWS of them
//   - a step decodes MOE_TILE_DEPTH weights of each row into shared memory: a block format's whole numbers, which f16 holds
//     exactly, and their scale a BLOCK_SCALE_WEIGHTS-weight span; and it gathers the tile's inputs as f16
//   - mma.m16n8k16 multiplies a span's whole numbers by the inputs into F32, and the span's scale weighs that sum: an
//     input keeps about 2^-12 of its value, finer than the 8-bit activations the reference engines multiply a prompt's
//     experts by (Strata's MMQ) and the round's decode multiplies them by (q8_1); a low f16 part of what an input rounds
//     away, multiplied too, had doubled the products (a 16K Flash-Next read 0.45 s slower) for answers no closer to the
//     reference's (Flash-Next's 24: 20 token for token either way, scored within 0.0987 a token with it, 0.0729 without)
//   - a tile reads each weight of its expert once for all its rows
constexpr unsigned int MOE_TILE_WORDS = 3;
constexpr unsigned int MOE_TILE_TOKENS = 32;
constexpr unsigned int MOE_TILE_ROWS = 64;
constexpr unsigned int MOE_TILE_DEPTH = 64;
constexpr unsigned int MOE_TILE_SPANS = MOE_TILE_DEPTH / BLOCK_SCALE_WEIGHTS;
// MOE_MMA_ROWS, MOE_MMA_TOKENS, MOE_MMA_DEPTH: the rows, columns and inner width of one mma tile; a warp takes one row tile.
constexpr unsigned int MOE_MMA_ROWS = 16;
constexpr unsigned int MOE_MMA_TOKENS = 8;
constexpr unsigned int MOE_MMA_DEPTH = 16;
constexpr unsigned int MOE_TILE_THREADS = MOE_TILE_ROWS / MOE_MMA_ROWS * CUDA_WARP_WIDTH;
constexpr unsigned int MOE_TILE_COLUMNS = MOE_TILE_TOKENS / MOE_MMA_TOKENS;
// MOE_TILE_STRIDE: a tile row's f16 stride; eight halves of padding keep rows 16-byte aligned and put ldmatrix's eight rows in distinct banks.
constexpr unsigned int MOE_TILE_STRIDE = MOE_TILE_DEPTH + 8;
// MOE_MMA_ELEMENTS: the accumulators a lane holds of one mma tile, rows g and g + 8 by columns 2t and 2t + 1.
constexpr unsigned int MOE_MMA_ELEMENTS = 4;

// moe_tile_store writes four weights as f16.
__device__ __forceinline__ void moe_tile_store(__half * at, const float * w) {
    const __half2 low = __floats2half2_rn(w[0], w[1]);
    const __half2 high = __floats2half2_rn(w[2], w[3]);
    *reinterpret_cast<uint2 *>(at) = make_uint2(
        *reinterpret_cast<const unsigned int *>(&low), *reinterpret_cast<const unsigned int *>(&high));
}

// moe_tile_source names a tile's MOE_TILE_ROWS weight rows of a bank: the weight that starts the first, the
// weights a row holds, which are also the inner width, the rows that lie in the bank, and the bank's storage.
struct moe_tile_source {
    const void * bank;
    size_t first;
    unsigned int depth;
    unsigned int rows;
    unsigned int storage;
};

// moe_tile_elements decodes weights k0 to k0 + MOE_TILE_DEPTH of a tile's rows one weight a thread in turn, as f16 under a scale of 1.
//   - it serves a storage without a block format, and stays out of line so the block formats' steps keep their code compact
__device__ __noinline__ void moe_tile_elements(__half * tile, float * scales, moe_tile_source source, unsigned int k0, unsigned int thread) {
    for (unsigned int item = thread; item < MOE_TILE_ROWS * MOE_TILE_DEPTH; item += MOE_TILE_THREADS) {
        const unsigned int row = item / MOE_TILE_DEPTH;
        const unsigned int column = item % MOE_TILE_DEPTH;
        tile[row * MOE_TILE_STRIDE + column] = __float2half_rn(row < source.rows && k0 + column < source.depth
            ? moe_expert_value(source.bank, source.first + (size_t) row * source.depth + k0 + column, source.storage)
            : 0.0f);
        if (column % BLOCK_SCALE_WEIGHTS == 0) scales[row * MOE_TILE_SPANS + column / BLOCK_SCALE_WEIGHTS] = 1.0f;
    }
}

// moe_tile_blocks decodes weights k0 to k0 + MOE_TILE_DEPTH of a tile's rows of a block storage and their scales.
//   - a thread takes a group of BLOCK_DOT_WEIGHTS weights of a row in each pass, and the passes' reads are in flight together
//   - the groups of a span write its scale alike; rows past the bank, and blocks past the row, take zeros
template <typename Format>
__device__ __forceinline__ void moe_tile_blocks(__half * tile, float * scales, const moe_tile_source & source, unsigned int k0,
        const block_dot_codebooks & codebooks, unsigned int thread) {
    constexpr unsigned int groups = MOE_TILE_DEPTH / BLOCK_DOT_WEIGHTS;
    constexpr unsigned int block_groups = Format::width / BLOCK_DOT_WEIGHTS;
    #pragma unroll 2
    for (unsigned int pass = 0; pass < MOE_TILE_ROWS * groups / MOE_TILE_THREADS; ++pass) {
        const unsigned int item = pass * MOE_TILE_THREADS + thread;
        const unsigned int row = item / groups;
        const unsigned int global = k0 / BLOCK_DOT_WEIGHTS + item % groups;
        const unsigned int block = global / block_groups;
        const unsigned int group = global % block_groups;
        float w[BLOCK_DOT_WEIGHTS] = {};
        float scale = 0.0f;
        if (row < source.rows && block * Format::width < source.depth) {
            scale = Format::weights(block_row<Format>(source.bank, source.first + (size_t) row * source.depth) +
                (size_t) block * Format::bytes, group, codebooks, w);
        }
        const unsigned int column = block * Format::width + Format::offset(group) - k0;
        __half * at = tile + row * MOE_TILE_STRIDE + column;
        moe_tile_store(at, w);
        moe_tile_store(at + Format::split, w + BLOCK_DOT_WEIGHTS / 2);
        scales[row * MOE_TILE_SPANS + column / BLOCK_SCALE_WEIGHTS] = scale;
    }
}

// moe_tile_weights decodes weights k0 to k0 + MOE_TILE_DEPTH of a tile's rows and their spans' scales into shared memory.
__device__ __forceinline__ void moe_tile_weights(__half * tile, float * scales, const moe_tile_source & source, unsigned int k0,
        const block_dot_codebooks & codebooks, unsigned int thread) {
    if (!visit_block_storage(source.storage, [&](auto format) {
            moe_tile_blocks<decltype(format)>(tile, scales, source, k0, codebooks, thread);
        })) {
        moe_tile_elements(tile, scales, source, k0, thread);
    }
}

// moe_tile_inputs gathers columns k0 to k0 + MOE_TILE_DEPTH of a tile's input rows as f16.
//   - THREADS threads share the gather, thread being this one's place among them
//   - a thread reads four columns at once where they lie 16-byte aligned, and stores them once all its reads are in flight
//   - row_input(row) = the input row of tile row row; rows from count and columns from depth take zeros
template <unsigned int THREADS, typename RowInput>
__device__ __forceinline__ void moe_tile_inputs(__half * tile, unsigned int count, unsigned int k0, unsigned int depth, RowInput row_input, unsigned int thread) {
    constexpr unsigned int quad = sizeof(float4) / sizeof(float);
    constexpr unsigned int quads = MOE_TILE_TOKENS * MOE_TILE_DEPTH / quad / THREADS;
    float4 values[quads];
    #pragma unroll
    for (unsigned int pass = 0; pass < quads; ++pass) {
        const unsigned int item = pass * THREADS + thread;
        const unsigned int row = item / (MOE_TILE_DEPTH / quad);
        const unsigned int column = item % (MOE_TILE_DEPTH / quad) * quad;
        values[pass] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (row >= count) continue;
        const float * source = row_input(row) + k0 + column;
        if (k0 + column + quad - 1 < depth && reinterpret_cast<size_t>(source) % sizeof(float4) == 0) {
            values[pass] = *reinterpret_cast<const float4 *>(source);
        } else {
            if (k0 + column < depth) values[pass].x = source[0];
            if (k0 + column + 1 < depth) values[pass].y = source[1];
            if (k0 + column + 2 < depth) values[pass].z = source[2];
            if (k0 + column + 3 < depth) values[pass].w = source[3];
        }
    }
    #pragma unroll
    for (unsigned int pass = 0; pass < quads; ++pass) {
        const unsigned int item = pass * THREADS + thread;
        const unsigned int row = item / (MOE_TILE_DEPTH / quad);
        const unsigned int column = item % (MOE_TILE_DEPTH / quad) * quad;
        const float w[quad] = {values[pass].x, values[pass].y, values[pass].z, values[pass].w};
        moe_tile_store(tile + row * MOE_TILE_STRIDE + column, w);
    }
}

// moe_tile_product adds a step's products of a warp's MOE_MMA_ROWS decoded rows with the tile's inputs to sums.
//   - each span's whole numbers multiply the inputs, and the span's scale of each row weighs the span's sums
//   - the lane addresses of ldmatrix are attention_tiled_bf16_f32's: A takes row halves first, B column halves first
__device__ __forceinline__ void moe_tile_product(const __half * weights, const float * scales, const __half * inputs,
        float (&sums)[MOE_TILE_COLUMNS][MOE_MMA_ELEMENTS]) {
    const unsigned int lane = threadIdx.x % CUDA_WARP_WIDTH;
    const unsigned int a_row = (lane & 7) + ((lane >> 3) & 1) * 8;
    const unsigned int a_column = (lane >> 4) * 8;
    const unsigned int b_row = (lane & 7) + (lane >> 4) * 8;
    const unsigned int b_column = ((lane >> 3) & 1) * 8;
    #pragma unroll
    for (unsigned int span = 0; span < MOE_TILE_SPANS; ++span) {
        float parts[MOE_TILE_COLUMNS][MOE_MMA_ELEMENTS] = {};
        #pragma unroll
        for (unsigned int k = span * BLOCK_SCALE_WEIGHTS; k < (span + 1) * BLOCK_SCALE_WEIGHTS; k += MOE_MMA_DEPTH) {
            unsigned int a[4];
            attention_ldx4(attention_shared_address(weights + a_row * MOE_TILE_STRIDE + k + a_column), a[0], a[1], a[2], a[3]);
            #pragma unroll
            for (unsigned int column = 0; column < MOE_TILE_COLUMNS; column += 2) {
                unsigned int b0, b1, b2, b3;
                attention_ldx4(attention_shared_address(inputs +
                        (column * MOE_MMA_TOKENS + b_row) * MOE_TILE_STRIDE + k + b_column),
                    b0, b1, b2, b3);
                mma_f16_16x8x16(parts[column], a, b0, b1);
                mma_f16_16x8x16(parts[column + 1], a, b2, b3);
            }
        }
        // A lane's elements 0 and 1 lie in row g and 2 and 3 in row g + 8.
        const float low = scales[(lane / 4) * MOE_TILE_SPANS + span];
        const float high = scales[(lane / 4 + 8) * MOE_TILE_SPANS + span];
        #pragma unroll
        for (unsigned int column = 0; column < MOE_TILE_COLUMNS; ++column) {
            sums[column][0] += low * parts[column][0];
            sums[column][1] += low * parts[column][1];
            sums[column][2] += high * parts[column][2];
            sums[column][3] += high * parts[column][3];
        }
    }
}

// moe_tile_element names the row and the tile token of a lane's accumulator element of one mma tile column of the warp
// whose rows start at warp_row.
__device__ __forceinline__ void moe_tile_element(unsigned int warp_row, unsigned int column, unsigned int element,
        unsigned int & row, unsigned int & token) {
    const unsigned int lane = threadIdx.x % CUDA_WARP_WIDTH;
    row = warp_row + lane / 4 + element / 2 * 8;
    token = column * MOE_MMA_TOKENS + lane % 4 * 2 + element % 2;
}

// MOE_HIDDEN_THREADS: the hidden products' block, a MOE_TILE_THREADS half for the gate rows and one for the up rows.
constexpr unsigned int MOE_HIDDEN_THREADS = 2 * MOE_TILE_THREADS;

// moe_grouped_hidden_f32 gives a tile's routed rows their activations: a block takes MOE_TILE_ROWS gate rows and the same up rows.
//   - the first half of the block decodes and multiplies the gate rows, the second the up rows, and the up half takes
//     the gate half's sums through shared memory
//   - activations[p * intermediate + r] = the activation of position p's row r
extern "C" __global__ void __launch_bounds__(MOE_HIDDEN_THREADS) moe_grouped_hidden_f32(
        const float * input,
        const unsigned int * tiles,
        const unsigned int * positions,
        const void * gate,
        const void * up,
        const float * gate_bias,
        const float * up_bias,
        float * activations,
        unsigned int hidden,
        unsigned int intermediate,
        unsigned int top_k,
        unsigned int gate_storage,
        unsigned int up_storage,
        unsigned int fused,
        unsigned int activation,
        unsigned int routing,
        float swiglu_clamp,
        unsigned int indexed) {
    __shared__ block_dot_codebooks codebooks;
    __shared__ __align__(16) __half weights[2 * MOE_TILE_ROWS * MOE_TILE_STRIDE];
    __shared__ float scales[2 * MOE_TILE_ROWS * MOE_TILE_SPANS];
    __shared__ __align__(16) __half inputs[MOE_TILE_TOKENS * MOE_TILE_STRIDE];
    __shared__ const float * input_rows[MOE_TILE_TOKENS];
    block_dot_codebooks_load(codebooks, gate_storage, up_storage);
    const unsigned int * tile = tiles + (size_t) MOE_TILE_WORDS * blockIdx.x;
    const unsigned int expert = tile[0], first = tile[1], count = tile[2];
    const unsigned int row0 = blockIdx.y * MOE_TILE_ROWS;
    if (indexed) {
        gate = reinterpret_cast<const void * const *>(gate)[expert];
        up = reinterpret_cast<const void * const *>(up)[expert];
    }
    const size_t gate_first = (indexed ? 0 : (size_t) expert * (fused ? 2 : 1) * intermediate * hidden) + (size_t) row0 * hidden;
    const unsigned int rows = min(MOE_TILE_ROWS, intermediate - row0);
    // Gate rows fill the tile's first half and the same up rows its second; a fused bank holds its up rows after its gate rows.
    const unsigned int half = threadIdx.x / MOE_TILE_THREADS;
    const unsigned int thread = threadIdx.x % MOE_TILE_THREADS;
    const moe_tile_source source = half == 0
        ? moe_tile_source{gate, gate_first, hidden, rows, gate_storage}
        : moe_tile_source{up, gate_first + (fused ? (size_t) intermediate * hidden : 0), hidden, rows, up_storage};
    // The tile's input rows, read once; the first step's barrier publishes them.
    if (threadIdx.x < count) input_rows[threadIdx.x] = input + (size_t) (positions[first + threadIdx.x] / top_k) * hidden;
    const auto input_row = [&](unsigned int row) { return input_rows[row]; };
    const unsigned int warp_row = thread / CUDA_WARP_WIDTH * MOE_MMA_ROWS;
    const unsigned int half_row = half * MOE_TILE_ROWS + warp_row;
    float sums[MOE_TILE_COLUMNS][MOE_MMA_ELEMENTS] = {};
    for (unsigned int k0 = 0; k0 < hidden; k0 += MOE_TILE_DEPTH) {
        __syncthreads();
        moe_tile_weights(weights + half * MOE_TILE_ROWS * MOE_TILE_STRIDE, scales + half * MOE_TILE_ROWS * MOE_TILE_SPANS,
            source, k0, codebooks, thread);
        moe_tile_inputs<MOE_HIDDEN_THREADS>(inputs, count, k0, hidden, input_row, threadIdx.x);
        __syncthreads();
        moe_tile_product(weights + half_row * MOE_TILE_STRIDE, scales + half_row * MOE_TILE_SPANS, inputs, sums);
    }
    // The gate half's sums cross to the up half through the weights' shared memory.
    float * gate_dots = reinterpret_cast<float *>(weights);
    __syncthreads();
    if (half == 0) {
        #pragma unroll
        for (unsigned int column = 0; column < MOE_TILE_COLUMNS; ++column) {
            #pragma unroll
            for (unsigned int element = 0; element < MOE_MMA_ELEMENTS; ++element) {
                unsigned int row, token;
                moe_tile_element(warp_row, column, element, row, token);
                gate_dots[row * MOE_TILE_TOKENS + token] = sums[column][element];
            }
        }
    }
    __syncthreads();
    if (half == 0) return;
    #pragma unroll
    for (unsigned int column = 0; column < MOE_TILE_COLUMNS; ++column) {
        #pragma unroll
        for (unsigned int element = 0; element < MOE_MMA_ELEMENTS; ++element) {
            unsigned int row, token;
            moe_tile_element(warp_row, column, element, row, token);
            float gate_dot = gate_dots[row * MOE_TILE_TOKENS + token];
            row += row0;
            if (token >= count || row >= intermediate) continue;
            float up_dot = sums[column][element];
            if (gate_bias) {
                gate_dot += gate_bias[(size_t) expert * intermediate + row];
                up_dot += up_bias[(size_t) expert * intermediate + row];
            }
            activations[(size_t) (first + token) * intermediate + row] =
                moe_activation(gate_dot, up_dot, 1, activation, routing, swiglu_clamp);
        }
    }
}

// moe_grouped_down_f32 gives a tile's routed slots their down products: a block takes MOE_TILE_ROWS output channels.
//   - slots[s * hidden + h] = routed slot s's channel h, its down bias added
extern "C" __global__ void __launch_bounds__(MOE_TILE_THREADS) moe_grouped_down_f32(
        const float * activations,
        const unsigned int * tiles,
        const unsigned int * positions,
        const void * down,
        const float * down_bias,
        float * slots,
        unsigned int hidden,
        unsigned int intermediate,
        unsigned int storage,
        unsigned int indexed) {
    __shared__ block_dot_codebooks codebooks;
    __shared__ __align__(16) __half weights[MOE_TILE_ROWS * MOE_TILE_STRIDE];
    __shared__ float scales[MOE_TILE_ROWS * MOE_TILE_SPANS];
    __shared__ __align__(16) __half inputs[MOE_TILE_TOKENS * MOE_TILE_STRIDE];
    block_dot_codebooks_load(codebooks, storage, storage);
    const unsigned int * tile = tiles + (size_t) MOE_TILE_WORDS * blockIdx.x;
    const unsigned int expert = tile[0], first = tile[1], count = tile[2];
    const unsigned int row0 = blockIdx.y * MOE_TILE_ROWS;
    if (indexed) down = reinterpret_cast<const void * const *>(down)[expert];
    const moe_tile_source source = {down, (indexed ? 0 : (size_t) expert * hidden * intermediate) + (size_t) row0 * intermediate,
        intermediate, min(MOE_TILE_ROWS, hidden - row0), storage};
    const auto input_row = [&](unsigned int row) { return activations + (size_t) (first + row) * intermediate; };
    const unsigned int warp_row = threadIdx.x / CUDA_WARP_WIDTH * MOE_MMA_ROWS;
    float sums[MOE_TILE_COLUMNS][MOE_MMA_ELEMENTS] = {};
    for (unsigned int k0 = 0; k0 < intermediate; k0 += MOE_TILE_DEPTH) {
        __syncthreads();
        moe_tile_weights(weights, scales, source, k0, codebooks, threadIdx.x);
        moe_tile_inputs<MOE_TILE_THREADS>(inputs, count, k0, intermediate, input_row, threadIdx.x);
        __syncthreads();
        moe_tile_product(weights + warp_row * MOE_TILE_STRIDE, scales + warp_row * MOE_TILE_SPANS, inputs, sums);
    }
    #pragma unroll
    for (unsigned int column = 0; column < MOE_TILE_COLUMNS; ++column) {
        #pragma unroll
        for (unsigned int element = 0; element < MOE_MMA_ELEMENTS; ++element) {
            unsigned int channel, token;
            moe_tile_element(warp_row, column, element, channel, token);
            channel += row0;
            if (token >= count || channel >= hidden) continue;
            slots[(size_t) positions[first + token] * hidden + channel] =
                sums[column][element] + (down_bias ? down_bias[(size_t) expert * hidden + channel] : 0.0f);
        }
    }
}

// ---- A prompt's expert products on tensor cores (Strata's prefill, src/prefill/prefill.cpp) ----
// An expert with enough routed rows is dequantized to f16 by its storage's formulas, its rows gathered as f16, and its
// gate and up products, activation and down product taken by cuBLAS between these kernels, each row's output then
// written to its routed slot as moe_grouped_down_f32 writes it.

// moe_expert_dequant_f16 expands count weights of one expert's bank, from its first, to f16.
extern "C" __global__ void moe_expert_dequant_f16(
        const void * weights,
        __half * output,
        unsigned int storage,
        unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    output[index] = __float2half(moe_expert_value(weights, index, storage));
}

// moe_gather_rows_f16 gathers an expert's routed rows, each its token's input row, in its positions' order, as f16.
extern "C" __global__ void moe_gather_rows_f16(
        const float * input,
        const unsigned int * positions,
        __half * output,
        unsigned int hidden,
        unsigned int top_k,
        unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const unsigned int token = positions[index / hidden] / top_k;
    output[index] = __float2half(input[(size_t) token * hidden + index % hidden]);
}

// moe_activate_f16 activates each row's gate and up products, its first intermediate and next intermediate columns,
// as moe_grouped_hidden_f32 does, as f16 for the down product.
extern "C" __global__ void moe_activate_f16(
        const float * gate_up,
        __half * output,
        unsigned int intermediate,
        unsigned int activation,
        unsigned int routing,
        float swiglu_clamp,
        unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const float * row = gate_up + (size_t) (index / intermediate) * 2u * intermediate;
    const unsigned int column = index % intermediate;
    output[index] = __float2half(
        moe_activation(row[column], row[intermediate + column], 1u, activation, routing, swiglu_clamp));
}

// moe_scatter_rows_f32 writes an expert's down products, a row each of its routed rows, to their routed slots.
extern "C" __global__ void moe_scatter_rows_f32(
        const float * rows,
        const unsigned int * positions,
        float * slots,
        unsigned int hidden,
        unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    slots[(size_t) positions[index / hidden] * hidden + index % hidden] = rows[index];
}

// ops_f32_attention_tensor.cuh: attention on tensor cores, the BF16 native and in-registers kernels and the flash
// kernels, with concatenation, in ops_f32.cu's order after ops_f32_attention.cuh. ops_f32.cu includes it once; it
// compiles only there.

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

// BF16 attention with its scores, softmax and output in registers (attention_registers_bf16_f32): the native kernel's
// arithmetic, one BF16 m16n8k16 product a term with F32 accumulation, the scores scaled, biased and multiplied by
// log2(e) at their own F32 roundings and exponentiated in base 2, the probabilities rounded to BF16 for the value
// product and summed unrounded, and the output multiplied by the rounded inverse of its row's sum, without the native
// kernel's shared score and probability tiles. A block takes 64 query rows of one head, four warps of 16, and streams
// the native kernel's 128-key panels as two 64-key tiles of K and V through shared memory by cp.async, each load
// beside a product; Q is staged through the value tile's storage before the first. A row's running maximum is taken
// over a whole panel before its probabilities are rounded, as the native kernel takes it: taken over each 64-key tile,
// a probability was rounded against a smaller maximum wherever a panel's larger score lay in its second half, and on
// random heads at MOSS's 4118 tokens the kernels agreed to 82 dB where they agree to F32's rounding, which over a
// pass's 42 layers put 129,881 of 524,288 values of MOSS's first prediction outside the checkpoint's frozen comparison
// (moss-checkpoint-regression). A row's sum takes a panel in the native kernel's order too, its four 32-key groups
// each summed lane by lane and over the quad, then added in turn (attention_registers_group), so the two kernels'
// outputs are the same arithmetic (moss-attention: the native kernel took 28.9 ms a layer of MOSS at 12432 tokens,
// 44% of a pass).
#define ATTENTION_REGISTERS_ROWS 64u
#define ATTENTION_REGISTERS_KEYS 64u
#define ATTENTION_REGISTERS_THREADS 128u

// attention_registers_scores: a warp's 16 query rows against one 64-key tile held in tile, scaled, biased and
// multiplied by log2(e) at their own F32 roundings, the keys past the last masked, each of the lane's two rows' largest
// score taken into maximum.
__device__ __forceinline__ void attention_registers_scores(
        float (&scores)[8][4],
        const unsigned int (&q_frag)[8][4],
        const __nv_bfloat16 * tile,
        unsigned int tile_base,
        unsigned int key_value_tokens,
        const float * key_bias,
        float scale,
        unsigned int bk_row,
        unsigned int bk_col,
        unsigned int t,
        float (&maximum)[2]) {
    const unsigned int LD = 128u + 8u;
    const float log2e = 1.4426950408889634074f;
#pragma unroll
    for (unsigned int column = 0; column < 8u; ++column) {
        scores[column][0] = scores[column][1] = scores[column][2] = scores[column][3] = 0.0f;
    }
#pragma unroll
    for (unsigned int pair = 0; pair < 4u; ++pair) {
        const __nv_bfloat16 * k_rows = tile + (pair * 16u + bk_row) * LD + bk_col;
#pragma unroll
        for (unsigned int k = 0; k < 8u; ++k) {
            unsigned int b0, b1, b2, b3;
            attention_ldx4(attention_shared_address(k_rows + k * 16u), b0, b1, b2, b3);
            attention_mma(scores[2u * pair], q_frag[k], b0, b1);
            attention_mma(scores[2u * pair + 1u], q_frag[k], b2, b3);
        }
    }
#pragma unroll
    for (unsigned int column = 0; column < 8u; ++column) {
#pragma unroll
        for (unsigned int element = 0; element < 4u; ++element) {
            const unsigned int half = element / 2u;
            const unsigned int key_token = tile_base + column * 8u + 2u * t + element % 2u;
            const bool held = key_token < key_value_tokens;
            const float score = __fmul_rn(scores[column][element], scale) +
                (key_bias != nullptr && held ? key_bias[key_token] : 0.0f);
            const float scaled = __fmul_rn(score, log2e);
            scores[column][element] = held ? scaled : -INFINITY;
            if (held) {
                maximum[half] = fmaxf(maximum[half], scaled);
            }
        }
    }
}

// attention_registers_probabilities: a tile's scores exponentiated in base 2 against their rows' running maximum.
__device__ __forceinline__ void attention_registers_probabilities(float (&scores)[8][4], const float (&maximum)[2]) {
#pragma unroll
    for (unsigned int column = 0; column < 8u; ++column) {
#pragma unroll
        for (unsigned int element = 0; element < 4u; ++element) {
            const float score = scores[column][element];
            scores[column][element] = score == -INFINITY ? 0.0f : __nv_exp2f(score - maximum[element / 2u]);
        }
    }
}

// attention_registers_group: one of a row's 32-key groups of a panel summed as the native kernel sums it, unrounded:
// the lane's eight probabilities of the group in its order (a column's two keys, then the next column's), then the
// quad's four lanes, each pair first. The lane holds keys column * 8 + 2t and the next of the tile, so columns first to
// first + 3 are the native kernel's group of 32 keys, part t of it.
__device__ __forceinline__ float attention_registers_group(
        const float (&probabilities)[8][4], unsigned int first, unsigned int half) {
    float group = 0.0f;
#pragma unroll
    for (unsigned int column = 0; column < 4u; ++column) {
        group += probabilities[first + column][2u * half];
        group += probabilities[first + column][2u * half + 1u];
    }
    group += __shfl_xor_sync(0xffffffffu, group, 1);
    group += __shfl_xor_sync(0xffffffffu, group, 2);
    return group;
}

// attention_registers_values: a tile's probabilities, rounded to BF16 as the native kernel's P, times the 64 values
// held in tile, accumulated into the warp's output.
__device__ __forceinline__ void attention_registers_values(
        float (&weighted)[16][4], const float (&probabilities)[8][4], const __nv_bfloat16 * tile,
        unsigned int a_row, unsigned int a_col) {
    const unsigned int LD = 128u + 8u;
#pragma unroll
    for (unsigned int step = 0; step < 4u; ++step) {
        // Two score accumulators are the A fragment of 16 keys.
        unsigned int p_frag[4];
        const __nv_bfloat162 p0 = __floats2bfloat162_rn(probabilities[2u * step][0], probabilities[2u * step][1]);
        const __nv_bfloat162 p1 = __floats2bfloat162_rn(probabilities[2u * step][2], probabilities[2u * step][3]);
        const __nv_bfloat162 p2 = __floats2bfloat162_rn(probabilities[2u * step + 1u][0], probabilities[2u * step + 1u][1]);
        const __nv_bfloat162 p3 = __floats2bfloat162_rn(probabilities[2u * step + 1u][2], probabilities[2u * step + 1u][3]);
        p_frag[0] = *reinterpret_cast<const unsigned int *>(&p0);
        p_frag[1] = *reinterpret_cast<const unsigned int *>(&p1);
        p_frag[2] = *reinterpret_cast<const unsigned int *>(&p2);
        p_frag[3] = *reinterpret_cast<const unsigned int *>(&p3);
        const __nv_bfloat16 * v_rows = tile + (step * 16u + a_row) * LD + a_col;
#pragma unroll
        for (unsigned int column = 0; column < 8u; ++column) {
            unsigned int b0, b1, b2, b3;
            attention_ldx4_trans(attention_shared_address(v_rows + column * 16u), b0, b1, b2, b3);
            attention_mma(weighted[2u * column], p_frag, b0, b1);
            attention_mma(weighted[2u * column + 1u], p_frag, b2, b3);
        }
    }
}
extern "C" __global__ void __launch_bounds__(ATTENTION_REGISTERS_THREADS) attention_registers_bf16_f32(
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
    const unsigned int BM = ATTENTION_REGISTERS_ROWS;
    const unsigned int BN = ATTENTION_REGISTERS_KEYS;
    const unsigned int LD = W + 8u;  // bf16 lds, multiple of 8
    const unsigned int row_tile = blockIdx.x;
    const unsigned int query_head = blockIdx.y;
    const unsigned int sequence = blockIdx.z;
    if (sequence >= sequences || query_head >= query_heads) {
        return;
    }
    const unsigned int key_value_head = query_head / (query_heads / key_value_heads);
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int g = lane >> 2;
    const unsigned int t = lane & 3u;

    extern __shared__ unsigned char attention_registers_shared[];
    __nv_bfloat16 * k_tile = (__nv_bfloat16 *) attention_registers_shared;  // [BN][LD]
    __nv_bfloat16 * v_tile = k_tile + BN * LD;                               // [BN][LD]
    __nv_bfloat16 * q_tile = v_tile;                                         // [BM][LD], dead before the first V

    const unsigned int row_base = row_tile * BM;
    for (unsigned int index = threadIdx.x; index < BM * W; index += blockDim.x) {
        const unsigned int row = index >> 7;
        const unsigned int channel = index & 127u;
        const unsigned int token = row_base + row;
        q_tile[row * LD + channel] = __float2bfloat16(token < query_tokens && channel < head_width
            ? query[((sequence * query_tokens + token) * query_heads + query_head) * head_width + channel]
            : 0.0f);
    }
    __syncthreads();

    // ldmatrix lane->address roles, as attention_native_bf16_f32 takes them.
    const unsigned int a_row = (lane & 7u) + ((lane >> 3u) & 1u) * 8u;
    const unsigned int a_col = (lane >> 4u) * 8u;
    const unsigned int bk_row = (lane & 7u) + (lane >> 4u) * 8u;
    const unsigned int bk_col = ((lane >> 3u) & 1u) * 8u;
    unsigned int q_frag[8][4];
#pragma unroll
    for (unsigned int k = 0; k < 8u; ++k) {
        attention_ldx4(
            attention_shared_address(q_tile + (warp * 16u + a_row) * LD + k * 16u + a_col),
            q_frag[k][0], q_frag[k][1], q_frag[k][2], q_frag[k][3]);
    }
    __syncthreads();

    const unsigned int kv_token_stride = key_value_heads * head_width;
    const unsigned short * key_base = key +
        sequence * key_value_tokens * kv_token_stride + key_value_head * head_width;
    const unsigned short * value_base = value +
        sequence * key_value_tokens * kv_token_stride + key_value_head * head_width;

    float weighted[16][4];
#pragma unroll
    for (unsigned int tile = 0; tile < 16u; ++tile) {
        weighted[tile][0] = weighted[tile][1] = weighted[tile][2] = weighted[tile][3] = 0.0f;
    }
    float maximum[2] = {-3.402823466e+38F, -3.402823466e+38F};
    float sum[2] = {0.0f, 0.0f};

    // A panel is the native kernel's 128 keys, two 64-key tiles: its rows' maximum is taken over both before any
    // probability is rounded, as the native kernel and PyTorch's memory-efficient attention take theirs. The two shared
    // tiles serve in turn: the panel's first keys in k_tile, its second in v_tile, its first values in k_tile once its
    // scores are taken, its second in v_tile, and the next panel's first keys in k_tile during the second product.
    attention_bf16_stage_rows(key_base, k_tile, 0u, key_value_tokens, kv_token_stride, head_width, BN);
    __pipeline_commit();
    for (unsigned int panel = 0; panel < key_value_tokens; panel += 2u * BN) {
        const bool second = panel + BN < key_value_tokens;
        // The first key tile is in, and every warp finished the last panel's products, so v_tile is free.
        __pipeline_wait_prior(0);
        __syncthreads();
        if (second) {
            attention_bf16_stage_rows(key_base, v_tile, panel + BN, key_value_tokens, kv_token_stride, head_width, BN);
            __pipeline_commit();
        }
        float first[8][4], following[8][4];
        float panel_maximum[2] = {-3.402823466e+38F, -3.402823466e+38F};
        attention_registers_scores(first, q_frag, k_tile, panel, key_value_tokens, key_bias, scale, bk_row, bk_col, t, panel_maximum);

        // The second key tile is in, and every warp finished its scores against k_tile, so the first values may load.
        __pipeline_wait_prior(0);
        __syncthreads();
        attention_bf16_stage_rows(value_base, k_tile, panel, key_value_tokens, kv_token_stride, head_width, BN);
        __pipeline_commit();
        if (second) {
            attention_registers_scores(following, q_frag, v_tile, panel + BN, key_value_tokens, key_bias, scale, bk_row, bk_col, t, panel_maximum);
        }

        // Each of the lane's two rows' running maximum over its quad and the panel.
        float alpha[2];
#pragma unroll
        for (unsigned int half = 0; half < 2u; ++half) {
            panel_maximum[half] = fmaxf(panel_maximum[half], __shfl_xor_sync(0xffffffffu, panel_maximum[half], 1));
            panel_maximum[half] = fmaxf(panel_maximum[half], __shfl_xor_sync(0xffffffffu, panel_maximum[half], 2));
            const float next = fmaxf(maximum[half], panel_maximum[half]);
            alpha[half] = __nv_exp2f(maximum[half] - next);
            maximum[half] = next;
        }
        attention_registers_probabilities(first, maximum);
        if (second) {
            attention_registers_probabilities(following, maximum);
        } else {
#pragma unroll
            for (unsigned int column = 0; column < 8u; ++column) {
                following[column][0] = following[column][1] = following[column][2] = following[column][3] = 0.0f;
            }
        }
        // Each row's sum, rescaled and then the panel's four 32-key groups added in order, as the native kernel adds them.
#pragma unroll
        for (unsigned int half = 0; half < 2u; ++half) {
            const float group0 = attention_registers_group(first, 0u, half);
            const float group1 = attention_registers_group(first, 4u, half);
            const float group2 = attention_registers_group(following, 0u, half);
            const float group3 = attention_registers_group(following, 4u, half);
            sum[half] = __fmul_rn(sum[half], alpha[half]) + group0;
            sum[half] += group1;
            sum[half] += group2;
            sum[half] += group3;
        }
#pragma unroll
        for (unsigned int tile = 0; tile < 16u; ++tile) {
            weighted[tile][0] *= alpha[0];
            weighted[tile][1] *= alpha[0];
            weighted[tile][2] *= alpha[1];
            weighted[tile][3] *= alpha[1];
        }

        // The first values are in, and every warp finished its scores against v_tile, so the second values may load.
        __pipeline_wait_prior(0);
        __syncthreads();
        if (second) {
            attention_bf16_stage_rows(value_base, v_tile, panel + BN, key_value_tokens, kv_token_stride, head_width, BN);
            __pipeline_commit();
        }
        attention_registers_values(weighted, first, k_tile, a_row, a_col);
        if (second) {
            // The second values are in, and every warp finished its product with k_tile, so the next keys may load.
            __pipeline_wait_prior(0);
            __syncthreads();
            if (panel + 2u * BN < key_value_tokens) {
                attention_bf16_stage_rows(key_base, k_tile, panel + 2u * BN, key_value_tokens, kv_token_stride, head_width, BN);
                __pipeline_commit();
            }
            attention_registers_values(weighted, following, v_tile, a_row, a_col);
        }
    }

#pragma unroll
    for (unsigned int half = 0; half < 2u; ++half) {
        const unsigned int token = row_base + warp * 16u + g + 8u * half;
        if (token >= query_tokens) {
            continue;
        }
        const float inverse = attention_row_frcp(sum[half]);
        float * out = output + ((sequence * query_tokens + token) * query_heads + query_head) * head_width + 2u * t;
#pragma unroll
        for (unsigned int tile = 0; tile < 16u; ++tile) {
            if (tile * 8u + 2u * t < head_width) {
                *reinterpret_cast<float2 *>(out + tile * 8u) =
                    make_float2(weighted[tile][2u * half] * inverse, weighted[tile][2u * half + 1u] * inverse);
            }
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

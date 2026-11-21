// ops_f32_int8_product.cuh: the int8 ConvRot product on tensor cores with its scales applied in its epilogue, as
// comfy_kitchen's CUTLASS int8 product applies them (cutlass_gemm_int8.cu at v0.2.40: a 128 x 256 x 64 tile, 64 x 64
// a warp, three stages, Sm80's m16n8k32 instruction, and an epilogue visitor multiplying each exact int32 entry by its
// token's scale and then its weight row's and rounding it to BF16), in ops_f32.cu's order after ops_f32_tail.cuh,
// whose norm it reads. ops_f32.cu includes it once; it compiles only there.

// A block multiplies INT8_PRODUCT_TOKENS token rows of the quantized activation by INT8_PRODUCT_FEATURES weight rows,
// INT8_PRODUCT_DEPTH inner values a stage, INT8_PRODUCT_STAGES stages in flight, by INT8_PRODUCT_THREADS threads: eight
// warps, two along the tokens and four along the features, each INT8_PRODUCT_WARP_TOKENS by INT8_PRODUCT_WARP_FEATURES.
// A stage holds both tiles' rows of INT8_PRODUCT_DEPTH bytes, copied INT8_PRODUCT_CHUNK bytes at a time and swizzled
// (int8_product_offset). The epilogue stages the scaled entries of INT8_PRODUCT_STAGED_TOKENS tokens at a time in the
// stages' memory, a row INT8_PRODUCT_STAGED_STRIDE floats, so a warp writes each token's entries together.
constexpr unsigned int INT8_PRODUCT_TOKENS = 128;
constexpr unsigned int INT8_PRODUCT_FEATURES = 256;
constexpr unsigned int INT8_PRODUCT_DEPTH = 64;
constexpr unsigned int INT8_PRODUCT_STAGES = 3;
constexpr unsigned int INT8_PRODUCT_THREADS = 256;
constexpr unsigned int INT8_PRODUCT_WARP_TOKENS = 64;
constexpr unsigned int INT8_PRODUCT_WARP_FEATURES = 64;
constexpr unsigned int INT8_PRODUCT_CHUNK = 16;
constexpr unsigned int INT8_PRODUCT_CHUNKS = INT8_PRODUCT_DEPTH / INT8_PRODUCT_CHUNK;
constexpr unsigned int INT8_PRODUCT_LINE = 128;
constexpr unsigned int INT8_PRODUCT_TOKEN_BYTES = INT8_PRODUCT_TOKENS * INT8_PRODUCT_DEPTH;
constexpr unsigned int INT8_PRODUCT_STAGE_BYTES = (INT8_PRODUCT_TOKENS + INT8_PRODUCT_FEATURES) * INT8_PRODUCT_DEPTH;
constexpr unsigned int INT8_PRODUCT_SHARED_BYTES = INT8_PRODUCT_STAGES * INT8_PRODUCT_STAGE_BYTES;
constexpr unsigned int INT8_PRODUCT_STAGED_TOKENS = INT8_PRODUCT_WARP_TOKENS;
constexpr unsigned int INT8_PRODUCT_STAGED_STRIDE = INT8_PRODUCT_FEATURES + 8;
constexpr unsigned int INT8_PRODUCT_WARPS = INT8_PRODUCT_THREADS / CUDA_WARP_WIDTH;
constexpr unsigned int INT8_PRODUCT_FEATURE_WARPS = INT8_PRODUCT_FEATURES / INT8_PRODUCT_WARP_FEATURES;
constexpr unsigned int INT8_PRODUCT_M_TILES = INT8_PRODUCT_WARP_TOKENS / 16;
constexpr unsigned int INT8_PRODUCT_N_TILES = INT8_PRODUCT_WARP_FEATURES / 8;
static_assert(INT8_PRODUCT_STAGED_TOKENS * INT8_PRODUCT_STAGED_STRIDE * sizeof(float) <= INT8_PRODUCT_SHARED_BYTES,
              "the epilogue's staged tokens fit in the stages' memory");

// The epilogues: the scaled entries written (comfy_kitchen's own), added with a gate to a residual
// (gated_add_bf16_round_f32), or, a head of MOSS_HEAD_WIDTH entries at a time, normalized, stepped and rotated
// (moss_norm_rotate_bf16_round_f32).
constexpr unsigned int INT8_PRODUCT_PLAIN = 0;
constexpr unsigned int INT8_PRODUCT_GATED_ADD = 1;
constexpr unsigned int INT8_PRODUCT_NORM_ROTATE = 2;

// MOSS_HEAD_WIDTH is the head the norm and rotation epilogue holds, MOSS's, a warp's lanes MOSS_HEAD_LANE adjacent
// channels each.
constexpr unsigned int MOSS_HEAD_LANE = 4;
constexpr unsigned int MOSS_HEAD_WIDTH = CUDA_WARP_WIDTH * MOSS_HEAD_LANE;
static_assert(INT8_PRODUCT_FEATURES % MOSS_HEAD_WIDTH == 0, "a feature tile holds whole heads");

// int8_product_offset is where a tile's row of INT8_PRODUCT_DEPTH bytes keeps its chunk in a stage: two rows a line of
// INT8_PRODUCT_LINE bytes, the line's eight chunks exclusive-ored with the line's place among eight, so the eight rows
// an ldmatrix reads at one chunk and the eight chunks a quarter warp copies each fall in their own banks.
__device__ __forceinline__ unsigned int int8_product_offset(unsigned int row, unsigned int chunk) {
    const unsigned int line = row >> 1;
    return line * INT8_PRODUCT_LINE + (((((row & 1u) << 2) | chunk) ^ (line & 7u)) * INT8_PRODUCT_CHUNK);
}

// int8_product_copy copies a chunk from global memory to shared memory without waiting, or zeros it where the row lies
// past the operand.
__device__ __forceinline__ void int8_product_copy(unsigned int destination, const signed char * source, bool valid) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(destination), "l"(source),
                 "r"(valid ? INT8_PRODUCT_CHUNK : 0u)
                 : "memory");
}

// int8_product_stage copies one stage: the block's token rows of the activation and feature rows of the weight, from
// inner value depth on.
__device__ __forceinline__ void int8_product_stage(
        unsigned int stage,
        const signed char * activation,
        const signed char * weight,
        unsigned int inner,
        unsigned int features,
        unsigned int tokens,
        unsigned int first_token,
        unsigned int first_feature,
        unsigned int depth) {
#pragma unroll
    for (unsigned int pass = 0; pass < INT8_PRODUCT_TOKENS * INT8_PRODUCT_CHUNKS / INT8_PRODUCT_THREADS; ++pass) {
        const unsigned int copy = pass * INT8_PRODUCT_THREADS + threadIdx.x;
        const unsigned int row = copy / INT8_PRODUCT_CHUNKS;
        const unsigned int chunk = copy % INT8_PRODUCT_CHUNKS;
        const bool valid = first_token + row < tokens;
        const size_t source = (size_t) (valid ? first_token + row : 0u) * inner + depth + chunk * INT8_PRODUCT_CHUNK;
        int8_product_copy(stage + int8_product_offset(row, chunk), activation + source, valid);
    }
#pragma unroll
    for (unsigned int pass = 0; pass < INT8_PRODUCT_FEATURES * INT8_PRODUCT_CHUNKS / INT8_PRODUCT_THREADS; ++pass) {
        const unsigned int copy = pass * INT8_PRODUCT_THREADS + threadIdx.x;
        const unsigned int row = copy / INT8_PRODUCT_CHUNKS;
        const unsigned int chunk = copy % INT8_PRODUCT_CHUNKS;
        const bool valid = first_feature + row < features;
        const size_t source = (size_t) (valid ? first_feature + row : 0u) * inner + depth + chunk * INT8_PRODUCT_CHUNK;
        int8_product_copy(stage + INT8_PRODUCT_TOKEN_BYTES + int8_product_offset(row, chunk), weight + source, valid);
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

__device__ __forceinline__ void int8_product_ldx4(unsigned int address, unsigned int * fragment) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(fragment[0]), "=r"(fragment[1]), "=r"(fragment[2]), "=r"(fragment[3])
                 : "r"(address));
}

__device__ __forceinline__ void int8_product_mma(int * accumulator, const unsigned int * a, unsigned int b0, unsigned int b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(accumulator[0]), "+r"(accumulator[1]), "+r"(accumulator[2]), "+r"(accumulator[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// int8_product_fragments loads a warp's fragments of one 32-byte half of a stage's rows: its four 16-row tiles of the
// activation, and its eight 8-row tiles of the weight, two to an ldmatrix.
__device__ __forceinline__ void int8_product_fragments(
        unsigned int stage,
        unsigned int activation_lane,
        unsigned int weight_lane,
        unsigned int half,
        unsigned int (&a)[INT8_PRODUCT_M_TILES][4],
        unsigned int (&b)[INT8_PRODUCT_N_TILES / 2][4]) {
#pragma unroll
    for (unsigned int m = 0; m < INT8_PRODUCT_M_TILES; ++m) {
        int8_product_ldx4(stage + ((activation_lane + m * 16u * INT8_PRODUCT_DEPTH) ^ (half * 32u)), a[m]);
    }
#pragma unroll
    for (unsigned int pair = 0; pair < INT8_PRODUCT_N_TILES / 2u; ++pair) {
        int8_product_ldx4(stage + ((weight_lane + pair * 16u * INT8_PRODUCT_DEPTH) ^ (half * 32u)), b[pair]);
    }
}

// int8_product_multiply adds a warp's products of one half's fragments to its accumulators.
__device__ __forceinline__ void int8_product_multiply(
        int (&accumulator)[INT8_PRODUCT_M_TILES][INT8_PRODUCT_N_TILES][4],
        const unsigned int (&a)[INT8_PRODUCT_M_TILES][4],
        const unsigned int (&b)[INT8_PRODUCT_N_TILES / 2][4]) {
#pragma unroll
    for (unsigned int m = 0; m < INT8_PRODUCT_M_TILES; ++m) {
#pragma unroll
        for (unsigned int n = 0; n < INT8_PRODUCT_N_TILES; ++n) {
            int8_product_mma(accumulator[m][n], a[m], b[n / 2u][(n % 2u) * 2u], b[n / 2u][(n % 2u) * 2u + 1u]);
        }
    }
}

// int8_product_scaled is an exact int32 entry as comfy_kitchen's epilogue scales it: times its token's scale, then its
// weight row's, each product rounded to F32, rounded to BF16.
__device__ __forceinline__ float int8_product_scaled(int entry, float token_scale, float weight_scale) {
    return bf16_rounded(__fmul_rn(__fmul_rn(__int2float_rn(entry), token_scale), weight_scale));
}

// int8_product_norm_rotate normalizes, steps and rotates a head's row a warp holds, MOSS_HEAD_LANE adjacent channels
// a lane, as moss_norm_rotate_bf16_round_f32 does: the norm's sum is rms_norm_mean_inverse's over a block whose first
// warp holds the row (each lane's squares in order, then the lanes' tree from the halves 16 apart down), its scale
// shared (rms_norm_mean_scale), the steps' values at the lane's channels read once (rms_norm_mean_step_values), and the
// half-turn read from the lane 16 away, rotated by the lane's cosines and sines.
__device__ __forceinline__ float4 int8_product_norm_rotate(
        float * values,
        const float (&stepped)[MOSS_HEAD_LANE][RMS_NORM_STEPS],
        unsigned int kinds,
        float epsilon,
        float4 cosines,
        float4 sines,
        unsigned int lane,
        unsigned int rows) {
    const float sum = ((__fmul_rn(values[0], values[0]) + __fmul_rn(values[1], values[1])) + __fmul_rn(values[2], values[2])) +
        __fmul_rn(values[3], values[3]);
    const float inverse = rms_norm_mean_scale(__shfl_sync(0xffffffffu, warp_sum_down(sum), 0), MOSS_HEAD_WIDTH, rows, epsilon);
    for (unsigned int part = 0; part < MOSS_HEAD_LANE; ++part) {
        values[part] = rms_norm_mean_stepped(values[part], inverse, stepped[part], kinds);
    }
    const float turn = lane < CUDA_WARP_WIDTH / 2u ? -1.0f : 1.0f;
    float turned[MOSS_HEAD_LANE];
    for (unsigned int part = 0; part < MOSS_HEAD_LANE; ++part) {
        turned[part] = turn * __shfl_xor_sync(0xffffffffu, values[part], CUDA_WARP_WIDTH / 2u);
    }
    return make_float4(
        moss_rotate_value(values[0], turned[0], cosines.x, sines.x),
        moss_rotate_value(values[1], turned[1], cosines.y, sines.y),
        moss_rotate_value(values[2], turned[2], cosines.z, sines.z),
        moss_rotate_value(values[3], turned[3], cosines.w, sines.w));
}

// int8_convrot_product multiplies tokens rows of the quantized activation (inner int8 values a row, its F32 scale per
// row) by features rows of an int8 ConvRot weight (inner int8 values a row, its F32 scale per row) into output, a
// token's features together, inner a multiple of INT8_PRODUCT_DEPTH and features of four. Each int32 entry is exact,
// so the tensor cores' order changes no bit; the epilogue scales it (int8_product_scaled) and writes it as EPILOGUE
// says. A grid's x runs over the token tiles and its y over the feature tiles, so the blocks that read one tile of the
// weight run together.
template <unsigned int EPILOGUE>
__device__ __forceinline__ void int8_convrot_product(
        const signed char * activation,
        const float * activation_scale,
        const signed char * weight,
        const float * weight_scale,
        const float * residual,
        const float * gate,
        const float * step0,
        const float * step1,
        const float * step2,
        unsigned int kinds,
        float epsilon,
        const float * cosine,
        const float * sine,
        float * output,
        unsigned int inner,
        unsigned int features,
        unsigned int tokens) {
    extern __shared__ __align__(INT8_PRODUCT_LINE) unsigned char int8_product_shared[];
    const unsigned int shared = (unsigned int) __cvta_generic_to_shared(int8_product_shared);
    const unsigned int warp = threadIdx.x / CUDA_WARP_WIDTH;
    const unsigned int lane = threadIdx.x % CUDA_WARP_WIDTH;
    const unsigned int warp_token = (warp / INT8_PRODUCT_FEATURE_WARPS) * INT8_PRODUCT_WARP_TOKENS;
    const unsigned int warp_feature = (warp % INT8_PRODUCT_FEATURE_WARPS) * INT8_PRODUCT_WARP_FEATURES;
    const unsigned int first_token = blockIdx.x * INT8_PRODUCT_TOKENS;
    const unsigned int first_feature = blockIdx.y * INT8_PRODUCT_FEATURES;
    const unsigned int depths = inner / INT8_PRODUCT_DEPTH;
    for (unsigned int stage = 0; stage + 1 < INT8_PRODUCT_STAGES; ++stage) {
        if (stage < depths) {
            int8_product_stage(shared + stage * INT8_PRODUCT_STAGE_BYTES, activation, weight, inner, features, tokens,
                               first_token, first_feature, stage * INT8_PRODUCT_DEPTH);
        } else {
            asm volatile("cp.async.commit_group;\n" ::: "memory");
        }
    }
    // ldmatrix's lane roles: the activation's 16 x 32 fragment as rows 0-7 and 8-15 of each half of the 32 bytes, the
    // weight's two 8-row fragments' halves in turn. A fragment 16 rows on keeps its swizzle 1,024 bytes on, and the
    // second 32 bytes of a stage's row lie 32 bytes away by exclusive or.
    const unsigned int activation_lane = int8_product_offset(warp_token + (lane & 7u) + ((lane >> 3) & 1u) * 8u, lane >> 4);
    const unsigned int weight_lane =
        INT8_PRODUCT_TOKEN_BYTES + int8_product_offset(warp_feature + (lane & 7u) + (lane >> 4) * 8u, (lane >> 3) & 1u);
    int accumulator[INT8_PRODUCT_M_TILES][INT8_PRODUCT_N_TILES][4];
#pragma unroll
    for (unsigned int m = 0; m < INT8_PRODUCT_M_TILES; ++m) {
#pragma unroll
        for (unsigned int n = 0; n < INT8_PRODUCT_N_TILES; ++n) {
            accumulator[m][n][0] = accumulator[m][n][1] = accumulator[m][n][2] = accumulator[m][n][3] = 0;
        }
    }
    // The fragments are loaded a half ahead of the products that read them, as CUTLASS's multistage loop loads them:
    // a stage's second half while its first multiplies, the next stage's first half while its second does.
    unsigned int a[2][INT8_PRODUCT_M_TILES][4];
    unsigned int b[2][INT8_PRODUCT_N_TILES / 2][4];
    asm volatile("cp.async.wait_group %0;\n" ::"n"(INT8_PRODUCT_STAGES - 2) : "memory");
    __syncthreads();
    int8_product_fragments(shared, activation_lane, weight_lane, 0u, a[0], b[0]);
    for (unsigned int depth = 0; depth < depths; ++depth) {
        int8_product_fragments(shared + (depth % INT8_PRODUCT_STAGES) * INT8_PRODUCT_STAGE_BYTES, activation_lane, weight_lane,
                               1u, a[1], b[1]);
        const unsigned int next = depth + INT8_PRODUCT_STAGES - 1;
        if (next < depths) {
            int8_product_stage(shared + (next % INT8_PRODUCT_STAGES) * INT8_PRODUCT_STAGE_BYTES, activation, weight, inner,
                               features, tokens, first_token, first_feature, next * INT8_PRODUCT_DEPTH);
        } else {
            asm volatile("cp.async.commit_group;\n" ::: "memory");
        }
        int8_product_multiply(accumulator, a[0], b[0]);
        asm volatile("cp.async.wait_group %0;\n" ::"n"(INT8_PRODUCT_STAGES - 2) : "memory");
        __syncthreads();
        if (depth + 1 < depths) {
            int8_product_fragments(shared + ((depth + 1) % INT8_PRODUCT_STAGES) * INT8_PRODUCT_STAGE_BYTES, activation_lane,
                                   weight_lane, 0u, a[0], b[0]);
        }
        int8_product_multiply(accumulator, a[1], b[1]);
    }
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");

    // The epilogue: a warp's entries, scaled, staged a half of the tile's tokens at a time, then each token's row read
    // back by one warp and written whole.
    float * staged = reinterpret_cast<float *>(int8_product_shared);
    const unsigned int group = lane >> 2;
    const unsigned int quad = lane & 3u;
    float row_scale[INT8_PRODUCT_N_TILES][2];
#pragma unroll
    for (unsigned int n = 0; n < INT8_PRODUCT_N_TILES; ++n) {
        const unsigned int feature = first_feature + warp_feature + n * 8u + quad * 2u;
        row_scale[n][0] = feature < features ? weight_scale[feature] : 0.0f;
        row_scale[n][1] = feature + 1u < features ? weight_scale[feature + 1u] : 0.0f;
    }
    const unsigned int heads = features / MOSS_HEAD_WIDTH;
    // What every token's row reads at a lane's channels, read once: the norm's steps, or each head's gates.
    const unsigned int channel = lane * MOSS_HEAD_LANE;
    float stepped[MOSS_HEAD_LANE][RMS_NORM_STEPS];
    float4 gates[INT8_PRODUCT_FEATURES / MOSS_HEAD_WIDTH];
#pragma unroll
    for (unsigned int part = 0; part < MOSS_HEAD_LANE; ++part) {
        if (EPILOGUE == INT8_PRODUCT_NORM_ROTATE) {
            rms_norm_mean_step_values(step0, step1, step2, kinds, channel + part, stepped[part]);
        }
    }
#pragma unroll
    for (unsigned int head = 0; head < INT8_PRODUCT_FEATURES / MOSS_HEAD_WIDTH; ++head) {
        const unsigned int feature = first_feature + head * MOSS_HEAD_WIDTH + channel;
        gates[head] = EPILOGUE == INT8_PRODUCT_GATED_ADD && feature < features ? load_f32x4_any(gate + feature) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    for (unsigned int half = 0; half < INT8_PRODUCT_TOKENS / INT8_PRODUCT_STAGED_TOKENS; ++half) {
        __syncthreads();
        if (warp_token == half * INT8_PRODUCT_STAGED_TOKENS) {
#pragma unroll
            for (unsigned int m = 0; m < INT8_PRODUCT_M_TILES; ++m) {
#pragma unroll
                for (unsigned int part = 0; part < 2u; ++part) {
                    const unsigned int row = m * 16u + part * 8u + group;
                    const unsigned int token = first_token + half * INT8_PRODUCT_STAGED_TOKENS + row;
                    const float token_scale = token < tokens ? activation_scale[token] : 0.0f;
#pragma unroll
                    for (unsigned int n = 0; n < INT8_PRODUCT_N_TILES; ++n) {
                        *reinterpret_cast<float2 *>(staged + row * INT8_PRODUCT_STAGED_STRIDE + warp_feature + n * 8u + quad * 2u) =
                            make_float2(int8_product_scaled(accumulator[m][n][2 * part], token_scale, row_scale[n][0]),
                                        int8_product_scaled(accumulator[m][n][2 * part + 1], token_scale, row_scale[n][1]));
                    }
                }
            }
        }
        __syncthreads();
        for (unsigned int row = warp; row < INT8_PRODUCT_STAGED_TOKENS; row += INT8_PRODUCT_WARPS) {
            const unsigned int token = first_token + half * INT8_PRODUCT_STAGED_TOKENS + row;
            if (token >= tokens) {
                break;
            }
            // A token's table, its heads' both, as the rotation reads it for every head.
            const size_t table = (size_t) token * MOSS_HEAD_WIDTH + channel;
            const float4 cosines = EPILOGUE == INT8_PRODUCT_NORM_ROTATE ? load_f32x4_any(cosine + table) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            const float4 sines = EPILOGUE == INT8_PRODUCT_NORM_ROTATE ? load_f32x4_any(sine + table) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
            for (unsigned int head = 0; head < INT8_PRODUCT_FEATURES / MOSS_HEAD_WIDTH; ++head) {
                const unsigned int column = head * MOSS_HEAD_WIDTH + channel;
                const unsigned int feature = first_feature + column;
                if (first_feature + head * MOSS_HEAD_WIDTH >= features) {
                    break;
                }
                const float4 entry = *reinterpret_cast<const float4 *>(staged + row * INT8_PRODUCT_STAGED_STRIDE + column);
                const size_t place = (size_t) token * features + feature;
                if (EPILOGUE == INT8_PRODUCT_NORM_ROTATE) {
                    float values[MOSS_HEAD_LANE] = {entry.x, entry.y, entry.z, entry.w};
                    *reinterpret_cast<float4 *>(output + place) =
                        int8_product_norm_rotate(values, stepped, kinds, epsilon, cosines, sines, lane, heads * tokens);
                } else if (feature < features) {
                    if (EPILOGUE == INT8_PRODUCT_GATED_ADD) {
                        const float4 stream = load_f32x4_any(residual + place);
                        *reinterpret_cast<float4 *>(output + place) = make_float4(
                            bf16_rounded(stream.x + bf16_rounded(entry.x * gates[head].x)),
                            bf16_rounded(stream.y + bf16_rounded(entry.y * gates[head].y)),
                            bf16_rounded(stream.z + bf16_rounded(entry.z * gates[head].z)),
                            bf16_rounded(stream.w + bf16_rounded(entry.w * gates[head].w)));
                    } else {
                        *reinterpret_cast<float4 *>(output + place) = entry;
                    }
                }
            }
        }
    }
}

// int8_convrot_product_bf16_f32: comfy_kitchen's int8 ConvRot product (int8_convrot_product), each entry scaled and
// written.
extern "C" __global__ void __launch_bounds__(INT8_PRODUCT_THREADS, 1) int8_convrot_product_bf16_f32(
        const signed char * activation,
        const float * activation_scale,
        const signed char * weight,
        const float * weight_scale,
        float * output,
        unsigned int inner,
        unsigned int features,
        unsigned int tokens) {
    int8_convrot_product<INT8_PRODUCT_PLAIN>(activation, activation_scale, weight, weight_scale, nullptr, nullptr, nullptr,
                                             nullptr, nullptr, 0u, 0.0f, nullptr, nullptr, output, inner, features, tokens);
}

// int8_convrot_product_gated_add_bf16_f32: an int8 ConvRot product (int8_convrot_product) whose entries a gated
// residual sum reads alone (gated_add_bf16_round_f32): each scaled entry times its feature's gate rounded to BF16, plus
// the residual, rounded to BF16, written in the residual's place's stead.
extern "C" __global__ void __launch_bounds__(INT8_PRODUCT_THREADS, 1) int8_convrot_product_gated_add_bf16_f32(
        const signed char * activation,
        const float * activation_scale,
        const signed char * weight,
        const float * weight_scale,
        const float * residual,
        const float * gate,
        float * output,
        unsigned int inner,
        unsigned int features,
        unsigned int tokens) {
    int8_convrot_product<INT8_PRODUCT_GATED_ADD>(activation, activation_scale, weight, weight_scale, residual, gate, nullptr,
                                                 nullptr, nullptr, 0u, 0.0f, nullptr, nullptr, output, inner, features, tokens);
}

// int8_convrot_product_norm_rotate_bf16_f32: an int8 ConvRot product (int8_convrot_product) whose heads of
// MOSS_HEAD_WIDTH entries MOSS normalizes, steps and rotates alone (moss_norm_rotate_bf16_round_f32), as the denoiser's q
// and k projections are: each token's heads scaled and normalized and rotated (int8_product_norm_rotate), the norm's
// rows heads * tokens, the tables [MOSS_HEAD_WIDTH, 1, tokens], read for every head.
extern "C" __global__ void __launch_bounds__(INT8_PRODUCT_THREADS, 1) int8_convrot_product_norm_rotate_bf16_f32(
        const signed char * activation,
        const float * activation_scale,
        const signed char * weight,
        const float * weight_scale,
        const float * step0,
        const float * step1,
        const float * step2,
        unsigned int kinds,
        float epsilon,
        const float * cosine,
        const float * sine,
        float * output,
        unsigned int inner,
        unsigned int features,
        unsigned int tokens) {
    int8_convrot_product<INT8_PRODUCT_NORM_ROTATE>(activation, activation_scale, weight, weight_scale, nullptr, nullptr, step0,
                                                   step1, step2, kinds, epsilon, cosine, sine, output, inner, features, tokens);
}

// The activation quantizer of the int8 ConvRot product, as comfy_kitchen's quantize_int8_rowwise_convrot64_kernel forms
// it with its input activation: one block a token column, a warp a rotation group of INT8_CONVROT_GROUP values at a
// time, each lane holding INT8_QUANTIZE_LANE of them, two runs of INT8_QUANTIZE_RUN adjacent values INT8_QUANTIZE_HALF
// apart, so a group's first radix-4 stage is a lane's own, its second and third a quad's of lanes and its fourth a
// lane's and the lane half a warp away's. A warp holds at most INT8_QUANTIZE_GROUPS groups, INT8_QUANTIZE_WARPS warps a
// row of INT8_CONVROT_MAX_INNER values, in registers until they are quantized.
constexpr unsigned int INT8_QUANTIZE_LANE = 8;
constexpr unsigned int INT8_QUANTIZE_RUN = 4;
constexpr unsigned int INT8_QUANTIZE_GROUPS = 4;
constexpr unsigned int INT8_QUANTIZE_WARPS = INT8_CONVROT_MAX_INNER / INT8_CONVROT_GROUP / INT8_QUANTIZE_GROUPS;
constexpr unsigned int INT8_QUANTIZE_HALF = INT8_CONVROT_GROUP / 2;
static_assert(INT8_QUANTIZE_LANE * CUDA_WARP_WIDTH == INT8_CONVROT_GROUP, "a warp holds a group");

// The activations the quantizer computes as it reads a column: none, a native-mean RMS norm and its steps
// (rms_norm_mean_affine_bf16_round_f32), or MOSS's SiTU of a gate-and-up product's halves (moss_situ_halves_value).
constexpr unsigned int INT8_QUANTIZE_PLAIN = 0;
constexpr unsigned int INT8_QUANTIZE_NORM = 1;
constexpr unsigned int INT8_QUANTIZE_SITU = 2;

// int8_quantize_butterfly is the output at digit of one radix-4 stage of int8_convrot_rotate over the four inputs at
// digits 0 to 3: half the left-to-right sum of the inputs, the one at digit 3 subtracted for digit 0, at 2 for 1, at 1
// for 2 and at 0 for 3, each sum correctly rounded as int8_convrot_rotate's.
__device__ __forceinline__ float int8_quantize_butterfly(float x0, float x1, float x2, float x3, unsigned int digit) {
    const float first = digit == 3u ? __fsub_rn(x1, x0) : (digit == 2u ? __fsub_rn(x0, x1) : __fadd_rn(x0, x1));
    const float second = digit == 1u ? __fsub_rn(first, x2) : __fadd_rn(first, x2);
    return __fmul_rn(INT8_CONVROT_HALF, digit == 0u ? __fsub_rn(second, x3) : __fadd_rn(second, x3));
}

// int8_quantize_across turns the radix-4 stage whose digit is two bits of a lane's index from shift over each value a
// lane holds: its four inputs are the values at the same place in the lanes whose index differs from its own there.
__device__ __forceinline__ void int8_quantize_across(float * values, unsigned int lane, unsigned int shift) {
    const unsigned int digit = (lane >> shift) & 3u;
    const unsigned int base = lane & ~(3u << shift);
#pragma unroll
    for (unsigned int slot = 0; slot < INT8_QUANTIZE_LANE; ++slot) {
        const float x0 = __shfl_sync(0xffffffffu, values[slot], base);
        const float x1 = __shfl_sync(0xffffffffu, values[slot], base | (1u << shift));
        const float x2 = __shfl_sync(0xffffffffu, values[slot], base | (2u << shift));
        const float x3 = __shfl_sync(0xffffffffu, values[slot], base | (3u << shift));
        values[slot] = int8_quantize_butterfly(x0, x1, x2, x3, digit);
    }
}

// int8_quantize_rotate rotates the group a warp holds, as int8_convrot_rotate does: a lane's values at places
// lane * INT8_QUANTIZE_RUN + run and INT8_QUANTIZE_HALF later, the stages a place's base-4 digits from the least
// significant: within a run, across a quad of lanes, across lanes four apart, and across the halves and the lanes half
// a warp apart.
__device__ __forceinline__ void int8_quantize_rotate(float * values, unsigned int lane) {
#pragma unroll
    for (unsigned int half = 0; half < 2u; ++half) {
        float * run = values + half * INT8_QUANTIZE_RUN;
        const float x0 = run[0], x1 = run[1], x2 = run[2], x3 = run[3];
        run[0] = int8_quantize_butterfly(x0, x1, x2, x3, 0u);
        run[1] = int8_quantize_butterfly(x0, x1, x2, x3, 1u);
        run[2] = int8_quantize_butterfly(x0, x1, x2, x3, 2u);
        run[3] = int8_quantize_butterfly(x0, x1, x2, x3, 3u);
    }
    int8_quantize_across(values, lane, 0u);
    int8_quantize_across(values, lane, 2u);
    const unsigned int high = lane >> 4;
#pragma unroll
    for (unsigned int place = 0; place < INT8_QUANTIZE_RUN; ++place) {
        const float own0 = values[place], own1 = values[INT8_QUANTIZE_RUN + place];
        const float other0 = __shfl_xor_sync(0xffffffffu, own0, CUDA_WARP_WIDTH / 2u);
        const float other1 = __shfl_xor_sync(0xffffffffu, own1, CUDA_WARP_WIDTH / 2u);
        const float x0 = high ? other0 : own0, x1 = high ? own0 : other0;
        const float x2 = high ? other1 : own1, x3 = high ? own1 : other1;
        values[place] = int8_quantize_butterfly(x0, x1, x2, x3, high);
        values[INT8_QUANTIZE_RUN + place] = int8_quantize_butterfly(x0, x1, x2, x3, 2u + high);
    }
}

// int8_quantize_nearest is a rotated value quantized as int8_convrot_quantize quantizes it to the nearest, as a byte:
// the value and the scale each rounded to BF16, divided, rounded to BF16, to the nearest integer, ties to even,
// saturated.
__device__ __forceinline__ unsigned int int8_quantize_nearest(float value, float divisor) {
    const float quotient = bf16_rounded(__fdiv_rn(bf16_rounded(value), divisor));
    return (unsigned int) (unsigned char) (signed char) __float2int_rn(fminf(fmaxf(rintf(quotient), INT8_LOWEST), INT8_HIGHEST));
}

// int8_quantize_columns quantizes each token column of inner values as comfy_kitchen's convrot64_quantize_row does
// with its input activation (ACTIVATION): the column's values read and activated, each group rotated
// (int8_quantize_rotate), the scale the column's absolute maximum times 1/127, at least 1e-30, and each value quantized
// to the nearest (int8_quantize_nearest), the values int8_convrot_rotate and int8_convrot_quantize give. A norm's
// inverse is rms_norm_mean_inverse's, the norm's rows the columns; a SiTU reads a column of 2 * inner values, its hidden
// half and then its gates. blockDim is a power of two of warps, enough that none holds more than INT8_QUANTIZE_GROUPS
// groups.
template <unsigned int ACTIVATION>
__device__ __forceinline__ void int8_quantize_columns(
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
    __shared__ float warp_amax[INT8_QUANTIZE_WARPS];
    const unsigned int column = blockIdx.x;
    const unsigned int warps = blockDim.x / CUDA_WARP_WIDTH;
    const unsigned int groups = inner / INT8_CONVROT_GROUP;
    if (column >= columns || inner > INT8_CONVROT_MAX_INNER || inner % INT8_CONVROT_GROUP != 0 || warps > INT8_QUANTIZE_WARPS ||
        groups > warps * INT8_QUANTIZE_GROUPS) {
        return;
    }
    const unsigned int warp = threadIdx.x / CUDA_WARP_WIDTH;
    const unsigned int lane = threadIdx.x % CUDA_WARP_WIDTH;
    const float * source = input + (size_t) column * inner * (ACTIVATION == INT8_QUANTIZE_SITU ? 2u : 1u);
    const float inverse = ACTIVATION == INT8_QUANTIZE_NORM ? rms_norm_mean_inverse(source, inner, columns, epsilon) : 0.0f;
    float values[INT8_QUANTIZE_GROUPS][INT8_QUANTIZE_LANE];
    float amax = 0.0f;
#pragma unroll
    for (unsigned int held = 0; held < INT8_QUANTIZE_GROUPS; ++held) {
        const unsigned int group = warp + held * warps;
        if (group >= groups) {
            break;
        }
#pragma unroll
        for (unsigned int half = 0; half < 2u; ++half) {
            const unsigned int index = group * INT8_CONVROT_GROUP + half * INT8_QUANTIZE_HALF + lane * INT8_QUANTIZE_RUN;
            const float4 read = load_f32x4_any(source + index);
            float raw[INT8_QUANTIZE_RUN] = {read.x, read.y, read.z, read.w};
            if (ACTIVATION == INT8_QUANTIZE_SITU) {
                const float4 gates = load_f32x4_any(source + inner + index);
                raw[0] = moss_situ_halves_value(raw[0], gates.x);
                raw[1] = moss_situ_halves_value(raw[1], gates.y);
                raw[2] = moss_situ_halves_value(raw[2], gates.z);
                raw[3] = moss_situ_halves_value(raw[3], gates.w);
            }
#pragma unroll
            for (unsigned int run = 0; run < INT8_QUANTIZE_RUN; ++run) {
                values[held][half * INT8_QUANTIZE_RUN + run] = ACTIVATION == INT8_QUANTIZE_NORM
                    ? rms_norm_mean_affine_value(raw[run], inverse, step0, step1, step2, kinds, index + run)
                    : raw[run];
            }
        }
        int8_quantize_rotate(values[held], lane);
#pragma unroll
        for (unsigned int slot = 0; slot < INT8_QUANTIZE_LANE; ++slot) {
            amax = fmaxf(amax, fabsf(values[held][slot]));
        }
    }
    for (unsigned int offset = CUDA_WARP_WIDTH / 2; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, offset));
    }
    if (lane == 0) {
        warp_amax[warp] = amax;
    }
    __syncthreads();
    amax = 0.0f;
    for (unsigned int other = 0; other < warps; ++other) {
        amax = fmaxf(amax, warp_amax[other]);
    }
    const float row_scale = fmaxf(__fmul_rn(amax, INT8_INVERSE_127), INT8_SCALE_FLOOR);
    if (threadIdx.x == 0) {
        scale[column] = row_scale;
    }
    const float divisor = bf16_rounded(row_scale);
    signed char * destination = output + (size_t) column * inner;
#pragma unroll
    for (unsigned int held = 0; held < INT8_QUANTIZE_GROUPS; ++held) {
        const unsigned int group = warp + held * warps;
        if (group >= groups) {
            break;
        }
#pragma unroll
        for (unsigned int half = 0; half < 2u; ++half) {
            const float * run = values[held] + half * INT8_QUANTIZE_RUN;
            const unsigned int packed = int8_quantize_nearest(run[0], divisor) | int8_quantize_nearest(run[1], divisor) << 8 |
                int8_quantize_nearest(run[2], divisor) << 16 | int8_quantize_nearest(run[3], divisor) << 24;
            *reinterpret_cast<unsigned int *>(destination + group * INT8_CONVROT_GROUP + half * INT8_QUANTIZE_HALF +
                                              lane * INT8_QUANTIZE_RUN) = packed;
        }
    }
}

// quantize_int8_convrot_columns_f32: the activation side of the int8 ConvRot product, as comfy_kitchen's CUDA
// int8_linear forms it (moss-int8, its product matched bit for bit): each token column of inner F32 values, inner a
// multiple of INT8_CONVROT_GROUP and at most INT8_CONVROT_MAX_INNER, rotated and quantized to the nearest with its scale
// stored per column (int8_quantize_columns).
extern "C" __global__ void quantize_int8_convrot_columns_f32(
        const float * input,
        signed char * output,
        float * scale,
        unsigned int inner,
        unsigned int columns) {
    int8_quantize_columns<INT8_QUANTIZE_PLAIN>(input, nullptr, nullptr, nullptr, 0u, 0.0f, output, scale, inner, columns);
}

// quantize_int8_convrot_norm_columns_f32: the activation side of an int8 ConvRot product whose activation is a native-mean
// RMS norm of input and the steps after it (rms_norm_mean_affine_bf16_round_f32), computed here and never stored: each
// token column normalized and stepped as the affine norm does, the norm's rows the columns, then rotated and quantized
// (int8_quantize_columns).
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
    int8_quantize_columns<INT8_QUANTIZE_NORM>(input, step0, step1, step2, kinds, epsilon, output, scale, inner, columns);
}

// quantize_int8_convrot_situ_columns_f32: the activation side of an int8 ConvRot product whose activation is MOSS's SiTU
// of one gate-and-up product's halves (moss_situ_halves_value), computed here and never stored, as comfy_kitchen's
// quantize_int8_rowwise_convrot64 computes it with its input activation: each token column's inner values the
// activation of the value at its place in the column's first half of the product and the second's, then rotated and
// quantized (int8_quantize_columns).
extern "C" __global__ void quantize_int8_convrot_situ_columns_f32(
        const float * product,
        signed char * output,
        float * scale,
        unsigned int inner,
        unsigned int columns) {
    int8_quantize_columns<INT8_QUANTIZE_SITU>(product, nullptr, nullptr, nullptr, 0u, 0.0f, output, scale, inner, columns);
}

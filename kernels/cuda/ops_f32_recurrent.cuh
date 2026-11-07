// ops_f32_recurrent.cuh: gated delta nets, gated linear attention, RWKV6, row sums and the Walsh-Hadamard transform, in ops_f32.cu's order. ops_f32.cu
// includes it once; it compiles only there.

// Native chunk phases share the executor's bounded BLAS workspace.
// A chunk is 64 tokens, the boundary transformers 5.14.1's chunked gated delta rule takes, so chunks split where its do.
// The scan and the four-accumulator reductions each sum in one fixed order, which the parity tests pin.
extern "C" __global__ void gated_delta_chunk_f32(
        const float * query, const float * key, const float * value,
        const float * gate, const float * beta, const float * initial,
        float * output, float * workspace,
        unsigned int width, unsigned int q_heads, unsigned int k_heads,
        unsigned int heads, unsigned int tokens, unsigned int sequences,
        unsigned int start, unsigned int scan_block, unsigned int repeat_interleave,
        unsigned int phase, float scale) {
    constexpr unsigned int chunk = 64;
    constexpr unsigned int vector_slots = 9;
    const unsigned int group = blockIdx.x;
    const unsigned int head = group % heads, sequence = group / heads;
    const size_t groups = (size_t) heads * sequences;
    const size_t tile = (size_t) chunk * width;
    const size_t matrix = (size_t) chunk * chunk;
    const size_t state_size = (size_t) width * width;
    float * q = workspace + group * tile;
    float * k = q + groups * tile;
    float * kb = k + groups * tile;
    float * vb = kb + groups * tile;
    float * mixed = vb + groups * tile;
    float * cumulative_key = mixed + groups * tile;
    float * v_new = cumulative_key + groups * tile;
    float * q_exp = v_new + groups * tile;
    float * temporary = q_exp + groups * tile;
    float * triangle = workspace + vector_slots * groups * tile + group * matrix;
    float * scores = triangle + groups * matrix;
    float * g = workspace + vector_slots * groups * tile + 2 * groups * matrix + group * chunk;
    float * state = workspace + vector_slots * groups * tile + 2 * groups * matrix + groups * chunk + group * state_size;
    float * update = state + groups * state_size;
    const unsigned int lane = threadIdx.x;
    __shared__ float scan[chunk];
    __shared__ float row_before[chunk];
    if (phase == 0) {
        if (lane < chunk) {
            scan[lane] = start + lane < tokens ? gate[((size_t) sequence * tokens + start + lane) * heads + head] : 0.0f;
        }
        __syncthreads();
        for (unsigned int base = 0; base < chunk; base += scan_block) {
            if (lane == 0 && base != 0) scan[base] = __fadd_rn(scan[base], scan[base - 1]);
            __syncthreads();
            for (unsigned int stride = 1; stride < scan_block; stride <<= 1) {
                if (lane < scan_block / 2) {
                    const unsigned int split = (lane / stride) * (2 * stride) + stride;
                    const unsigned int target = base + split + lane % stride;
                    scan[target] = __fadd_rn(scan[target], scan[base + split - 1]);
                }
                __syncthreads();
            }
        }
        if (lane < chunk) g[lane] = scan[lane];
        const unsigned int qh = repeat_interleave ? head / (heads / q_heads) : head % q_heads;
        const unsigned int kh = repeat_interleave ? head / (heads / k_heads) : head % k_heads;
        for (size_t index = lane; index < tile; index += blockDim.x) {
            const unsigned int row = (unsigned int) (index / width), col = (unsigned int) (index % width);
            float qi = 0.0f, ki = 0.0f, vi = 0.0f, bi = 0.0f;
            if (start + row < tokens) {
                const size_t token = (size_t) sequence * tokens + start + row;
                qi = query[(token * q_heads + qh) * width + col];
                ki = key[(token * k_heads + kh) * width + col];
                vi = value[(token * heads + head) * width + col];
                bi = beta[token * heads + head];
            }
            q[index] = __fmul_rn(qi, scale);
            k[index] = ki;
            kb[index] = __fmul_rn(ki, bi);
            vb[index] = __fmul_rn(vi, bi);
        }
        if (start == 0) {
            for (size_t index = lane; index < state_size; index += blockDim.x) {
                state[index] = initial[group * state_size + (index % width) * width + index / width];
            }
        }
    } else if (phase == 1) {
        for (size_t index = lane; index < matrix; index += blockDim.x) {
            const unsigned int row = (unsigned int) (index / chunk), col = (unsigned int) (index % chunk);
            triangle[index] = col < row ? -__fmul_rn(triangle[index], __nv_expf(__fsub_rn(g[row], g[col]))) : -0.0f;
        }
        __syncthreads();
        for (unsigned int row = 1; row < chunk; ++row) {
            if (lane < row) row_before[lane] = triangle[row * chunk + lane];
            __syncthreads();
            if (lane < row) {
                float partial[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                for (unsigned int inner = 0; inner < row; ++inner) {
                    partial[inner % 4] = __fadd_rn(partial[inner % 4], __fmul_rn(row_before[inner], triangle[inner * chunk + lane]));
                }
                const float sum = __fadd_rn(__fadd_rn(__fadd_rn(partial[0], partial[1]), partial[2]), partial[3]);
                triangle[row * chunk + lane] = __fadd_rn(row_before[lane], sum);
            }
            __syncthreads();
        }
        for (size_t index = lane; index < matrix; index += blockDim.x) {
            triangle[index] = __fadd_rn(triangle[index], index / chunk == index % chunk ? 1.0f : 0.0f);
        }
    } else if (phase == 2) {
        for (size_t index = lane; index < tile; index += blockDim.x) {
            const float decay = __nv_expf(g[index / width]);
            kb[index] = __fmul_rn(kb[index], decay);
            q_exp[index] = __fmul_rn(q[index], decay);
        }
    } else if (phase == 3) {
        for (size_t index = lane; index < tile; index += blockDim.x) v_new[index] = __fsub_rn(mixed[index], temporary[index]);
        for (size_t index = lane; index < matrix; index += blockDim.x) {
            const unsigned int row = (unsigned int) (index / chunk), col = (unsigned int) (index % chunk);
            const float decay = col <= row ? __nv_expf(__fsub_rn(g[row], g[col])) : 0.0f;
            scores[index] = __fmul_rn(scores[index], decay);
        }
    } else if (phase == 4) {
        for (size_t index = lane; index < tile; index += blockDim.x) {
            const unsigned int row = (unsigned int) (index / width), col = (unsigned int) (index % width);
            if (start + row < tokens) {
                output[(((size_t) sequence * tokens + start + row) * heads + head) * width + col] = __fadd_rn(temporary[index], vb[index]);
            }
            kb[index] = __fmul_rn(k[index], __nv_expf(__fsub_rn(g[chunk - 1], g[row])));
        }
        const float decay = __nv_expf(g[chunk - 1]);
        for (size_t index = lane; index < state_size; index += blockDim.x) state[index] = __fmul_rn(state[index], decay);
    } else if (phase == 5) {
        for (size_t index = lane; index < state_size; index += blockDim.x) {
            state[index] = __fadd_rn(state[index], update[index]);
            if ((size_t) start + chunk >= tokens) {
                const size_t output_base = (size_t) tokens * groups * width + group * state_size;
                output[output_base + (index % width) * width + index / width] = state[index];
            }
        }
    }
}

extern "C" __global__ void gated_delta_net_f32(
        const float * query,
        const float * key,
        const float * value,
        const float * gate,
        const float * beta,
        const float * input_state,
        float * output,
        unsigned int size,
        unsigned int query_heads,
        unsigned int key_heads,
        unsigned int heads,
        unsigned int tokens,
        unsigned int sequences,
        unsigned int gate_width,
        unsigned int repeat_interleave) {
    const unsigned int index = blockIdx.x;
    const unsigned int count = heads * sequences;
    if (index >= count) {
        return;
    }
    const unsigned int head = index % heads;
    const unsigned int sequence = index / heads;
    const unsigned int attention_elements =
        size * heads * tokens * sequences;
    const unsigned int state_elements = size * size;
    float * state = output + attention_elements + index * state_elements;
    const float * source_state = input_state + index * state_elements;
    const float scale = rsqrtf((float) size);
    const unsigned int lane = threadIdx.x % CUDA_WARP_WIDTH;
    const unsigned int warp = threadIdx.x / CUDA_WARP_WIDTH;
    const unsigned int warps = blockDim.x / CUDA_WARP_WIDTH;
    for (unsigned int row = warp; row < size; row += warps) {
        float * state_row = state + row * size;
        const float * source_row = source_state + row * size;
        for (unsigned int column = lane; column < size; column += CUDA_WARP_WIDTH) {
            state_row[column] = source_row[column];
        }
        for (unsigned int token = 0; token < tokens; ++token) {
            const unsigned int value_base =
                ((sequence * tokens + token) * heads + head) * size;
            const unsigned int query_head = repeat_interleave
                ? head / (heads / query_heads)
                : head % query_heads;
            const unsigned int key_head = repeat_interleave
                ? head / (heads / key_heads)
                : head % key_heads;
            const unsigned int query_base =
                ((sequence * tokens + token) * query_heads + query_head) * size;
            const unsigned int key_base =
                ((sequence * tokens + token) * key_heads + key_head) * size;
            const unsigned int gate_base =
                ((sequence * tokens + token) * heads + head) * gate_width;
            const float beta_value =
                beta[(sequence * tokens + token) * heads + head];
            if (gate_width == 1) {
                const float gate_scale = expf(gate[gate_base]);
                for (unsigned int column = lane; column < size; column += CUDA_WARP_WIDTH) {
                    state_row[column] *= gate_scale;
                }
            } else {
                for (unsigned int column = lane; column < size; column += CUDA_WARP_WIDTH) {
                    state_row[column] *= expf(gate[gate_base + column]);
                }
            }
            float dot = 0.0f;
            for (unsigned int column = lane; column < size; column += CUDA_WARP_WIDTH) {
                dot += state_row[column] * key[key_base + column];
            }
            for (unsigned int offset = CUDA_WARP_WIDTH / 2; offset > 0; offset /= 2) {
                dot += __shfl_down_sync(0xffffffff, dot, offset);
            }
            dot = __shfl_sync(0xffffffff, dot, 0);
            const float delta =
                (value[value_base + row] - dot) * beta_value;
            for (unsigned int column = lane; column < size; column += CUDA_WARP_WIDTH) {
                state_row[column] += delta * key[key_base + column];
            }
            dot = 0.0f;
            for (unsigned int column = lane; column < size; column += CUDA_WARP_WIDTH) {
                dot += state_row[column] * query[query_base + column];
            }
            for (unsigned int offset = CUDA_WARP_WIDTH / 2; offset > 0; offset /= 2) {
                dot += __shfl_down_sync(0xffffffff, dot, offset);
            }
            if (lane == 0) {
                output[value_base + row] = dot * scale;
            }
        }
    }
}

#define GATED_DELTA_NET_ROW_COLUMNS 8

// gated_delta_net_rows_f32 computes gated_delta_net_f32's outputs and final
// state with the same operations in the same order, a warp a state row with
// the row in registers: a lane holds the columns lane, lane + 32, ..., as the
// per-head kernel's lane reads them, so each dot product and update keeps its
// order, and the blocks of a head's rows spread across the card.
extern "C" __global__ void gated_delta_net_rows_f32(
        const float * query,
        const float * key,
        const float * value,
        const float * gate,
        const float * beta,
        const float * input_state,
        float * output,
        unsigned int size,
        unsigned int query_heads,
        unsigned int key_heads,
        unsigned int heads,
        unsigned int tokens,
        unsigned int sequences,
        unsigned int gate_width,
        unsigned int repeat_interleave) {
    const unsigned int index = blockIdx.x;
    if (index >= heads * sequences) {
        return;
    }
    const unsigned int lane = threadIdx.x % CUDA_WARP_WIDTH;
    const unsigned int row = blockIdx.y * (blockDim.x / CUDA_WARP_WIDTH) + threadIdx.x / CUDA_WARP_WIDTH;
    if (row >= size) {
        return;
    }
    const unsigned int head = index % heads;
    const unsigned int sequence = index / heads;
    const unsigned int attention_elements = size * heads * tokens * sequences;
    const unsigned int state_elements = size * size;
    float * state_row = output + attention_elements + index * state_elements + row * size;
    const float * source_row = input_state + index * state_elements + row * size;
    const float scale = rsqrtf((float) size);
    const unsigned int query_head = repeat_interleave ? head / (heads / query_heads) : head % query_heads;
    const unsigned int key_head = repeat_interleave ? head / (heads / key_heads) : head % key_heads;
    float state[GATED_DELTA_NET_ROW_COLUMNS];
#pragma unroll
    for (unsigned int part = 0; part < GATED_DELTA_NET_ROW_COLUMNS; ++part) {
        const unsigned int column = lane + part * CUDA_WARP_WIDTH;
        state[part] = column < size ? source_row[column] : 0.0f;
    }
    for (unsigned int token = 0; token < tokens; ++token) {
        const unsigned int value_base = ((sequence * tokens + token) * heads + head) * size;
        const unsigned int query_base = ((sequence * tokens + token) * query_heads + query_head) * size;
        const unsigned int key_base = ((sequence * tokens + token) * key_heads + key_head) * size;
        const unsigned int gate_base = ((sequence * tokens + token) * heads + head) * gate_width;
        const float beta_value = beta[(sequence * tokens + token) * heads + head];
        if (gate_width == 1) {
            const float gate_scale = expf(gate[gate_base]);
#pragma unroll
            for (unsigned int part = 0; part < GATED_DELTA_NET_ROW_COLUMNS; ++part) {
                if (lane + part * CUDA_WARP_WIDTH < size) {
                    state[part] *= gate_scale;
                }
            }
        } else {
#pragma unroll
            for (unsigned int part = 0; part < GATED_DELTA_NET_ROW_COLUMNS; ++part) {
                const unsigned int column = lane + part * CUDA_WARP_WIDTH;
                if (column < size) {
                    state[part] *= expf(gate[gate_base + column]);
                }
            }
        }
        float dot = 0.0f;
#pragma unroll
        for (unsigned int part = 0; part < GATED_DELTA_NET_ROW_COLUMNS; ++part) {
            const unsigned int column = lane + part * CUDA_WARP_WIDTH;
            if (column < size) {
                dot += state[part] * key[key_base + column];
            }
        }
        for (unsigned int offset = CUDA_WARP_WIDTH / 2; offset > 0; offset /= 2) {
            dot += __shfl_down_sync(0xffffffff, dot, offset);
        }
        dot = __shfl_sync(0xffffffff, dot, 0);
        const float delta = (value[value_base + row] - dot) * beta_value;
#pragma unroll
        for (unsigned int part = 0; part < GATED_DELTA_NET_ROW_COLUMNS; ++part) {
            const unsigned int column = lane + part * CUDA_WARP_WIDTH;
            if (column < size) {
                state[part] += delta * key[key_base + column];
            }
        }
        dot = 0.0f;
#pragma unroll
        for (unsigned int part = 0; part < GATED_DELTA_NET_ROW_COLUMNS; ++part) {
            const unsigned int column = lane + part * CUDA_WARP_WIDTH;
            if (column < size) {
                dot += state[part] * query[query_base + column];
            }
        }
        for (unsigned int offset = CUDA_WARP_WIDTH / 2; offset > 0; offset /= 2) {
            dot += __shfl_down_sync(0xffffffff, dot, offset);
        }
        if (lane == 0) {
            output[value_base + row] = dot * scale;
        }
    }
#pragma unroll
    for (unsigned int part = 0; part < GATED_DELTA_NET_ROW_COLUMNS; ++part) {
        const unsigned int column = lane + part * CUDA_WARP_WIDTH;
        if (column < size) {
            state_row[column] = state[part];
        }
    }
}

// The key-head kernel's shape (Strata's gdn_rec_kh_kernel, src/prefill/kernels.cu): a block takes a key head's
// GDN_KH_COLUMNS value columns of every value head sharing the key head, GDN_KH_GROUPS row groups of
// GDN_KH_ROWS key rows a thread, and stages GDN_KH_TOKENS tokens' inputs a buffer while the block computes the
// tokens before.
constexpr unsigned int GDN_KH_SIZE = 128;
constexpr unsigned int GDN_KH_COLUMNS = 32;
constexpr unsigned int GDN_KH_GROUPS = 4;
constexpr unsigned int GDN_KH_ROWS = GDN_KH_SIZE / GDN_KH_GROUPS;
constexpr unsigned int GDN_KH_TOKENS = 8;
constexpr unsigned int GDN_KH_THREADS = GDN_KH_COLUMNS * GDN_KH_GROUPS;

__device__ __forceinline__ void gdn_kh_copy4(float * shared, const float * global) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"((unsigned int) __cvta_generic_to_shared(shared)), "l"(global));
}

__device__ __forceinline__ void gdn_kh_copy16(float * shared, const float * global) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"((unsigned int) __cvta_generic_to_shared(shared)), "l"(global));
}

// gated_delta_net_keyhead computes gated_delta_net_rows_f32's outputs and final state for VPK value heads a
// key head, a 128-wide state and one gate a head: each thread holds GDN_KH_ROWS key rows of one value column
// of each of the key head's value heads, so a token's key and query are read once for the VPK heads, and the
// inputs of the next GDN_KH_TOKENS tokens arrive by cp.async while these are computed. A token takes its key
// product of the state before its decay, decays that product, and decays and updates the state in one fused
// multiply-add a row before its query product, in Strata's order: decaying the state first took a multiply a
// row and head more, 408 ms of Flash-Next's 16K read against Strata's 269. The products sum in another order
// than the rows kernel's.
template <unsigned int VPK>
__device__ __forceinline__ void gated_delta_net_keyhead(
        const float * query,
        const float * key,
        const float * value,
        const float * gate,
        const float * beta,
        const float * input_state,
        float * output,
        unsigned int key_heads,
        unsigned int heads,
        unsigned int tokens,
        unsigned int sequences,
        unsigned int repeat_interleave) {
    __shared__ __align__(16) float sq[2][GDN_KH_TOKENS][GDN_KH_SIZE];
    __shared__ __align__(16) float sk[2][GDN_KH_TOKENS][GDN_KH_SIZE];
    __shared__ __align__(16) float sv[2][GDN_KH_TOKENS][VPK][GDN_KH_COLUMNS];
    __shared__ float sg[2][GDN_KH_TOKENS][VPK];
    __shared__ float sb[2][GDN_KH_TOKENS][VPK];
    __shared__ float key_sums[VPK][GDN_KH_GROUPS][GDN_KH_COLUMNS];
    __shared__ float query_sums[VPK][GDN_KH_GROUPS][GDN_KH_COLUMNS];
    const unsigned int column_blocks = GDN_KH_SIZE / GDN_KH_COLUMNS;
    const unsigned int block = blockIdx.x % column_blocks;
    const unsigned int key_head = (blockIdx.x / column_blocks) % key_heads;
    const unsigned int sequence = blockIdx.x / (column_blocks * key_heads);
    if (sequence >= sequences) {
        return;
    }
    const unsigned int lane = threadIdx.x, group = threadIdx.y, thread = group * GDN_KH_COLUMNS + lane;
    const unsigned int column = block * GDN_KH_COLUMNS + lane;
    const size_t state_elements = (size_t) GDN_KH_SIZE * GDN_KH_SIZE;
    const size_t attention_elements = (size_t) GDN_KH_SIZE * heads * tokens * sequences;
    unsigned int head_of[VPK];
    float state[VPK][GDN_KH_ROWS];
#pragma unroll
    for (unsigned int j = 0; j < VPK; ++j) {
        head_of[j] = repeat_interleave ? key_head * VPK + j : key_head + j * key_heads;
        const float * source = input_state + (sequence * heads + head_of[j]) * state_elements + (size_t) column * GDN_KH_SIZE + group * GDN_KH_ROWS;
#pragma unroll
        for (unsigned int r = 0; r < GDN_KH_ROWS; ++r) {
            state[j][r] = source[r];
        }
    }
    const unsigned int staged_blocks = (tokens + GDN_KH_TOKENS - 1) / GDN_KH_TOKENS;
    auto stage = [&](unsigned int staged) {
        const unsigned int buffer = staged & 1, first = staged * GDN_KH_TOKENS;
        const unsigned int pieces = GDN_KH_SIZE / 4;
        for (unsigned int p = thread; p < GDN_KH_TOKENS * 2 * pieces; p += GDN_KH_THREADS) {
            const unsigned int i = p / (2 * pieces), which = (p % (2 * pieces)) / pieces, at = (p % pieces) * 4;
            if (first + i < tokens) {
                const size_t row = ((size_t) (sequence * tokens + first + i) * key_heads + key_head) * GDN_KH_SIZE + at;
                gdn_kh_copy16(which ? &sk[buffer][i][at] : &sq[buffer][i][at], (which ? key : query) + row);
            }
        }
        const unsigned int value_pieces = GDN_KH_COLUMNS / 4;
        for (unsigned int p = thread; p < GDN_KH_TOKENS * VPK * value_pieces; p += GDN_KH_THREADS) {
            const unsigned int i = p / (VPK * value_pieces), j = (p % (VPK * value_pieces)) / value_pieces, at = (p % value_pieces) * 4;
            if (first + i < tokens) {
                gdn_kh_copy16(&sv[buffer][i][j][at], value + ((size_t) (sequence * tokens + first + i) * heads + head_of[j]) * GDN_KH_SIZE + block * GDN_KH_COLUMNS + at);
            }
        }
        for (unsigned int p = thread; p < 2 * GDN_KH_TOKENS * VPK; p += GDN_KH_THREADS) {
            const unsigned int which = p / (GDN_KH_TOKENS * VPK), i = (p % (GDN_KH_TOKENS * VPK)) / VPK, j = p % VPK;
            if (first + i < tokens) {
                const size_t at = (size_t) (sequence * tokens + first + i) * heads + head_of[j];
                gdn_kh_copy4(which ? &sb[buffer][i][j] : &sg[buffer][i][j], (which ? beta : gate) + at);
            }
        }
    };
    const float scale = rsqrtf((float) GDN_KH_SIZE);
    if (staged_blocks > 0) {
        stage(0);
    }
    asm volatile("cp.async.commit_group;\n" ::);
    for (unsigned int staged = 0; staged < staged_blocks; ++staged) {
        // The buffer staged next was last read by the block before, whose last token's second barrier every thread has passed.
        if (staged + 1 < staged_blocks) {
            stage(staged + 1);
        }
        asm volatile("cp.async.commit_group;\n" ::);
        asm volatile("cp.async.wait_group 1;\n" ::);
        __syncthreads();
        const unsigned int buffer = staged & 1, first = staged * GDN_KH_TOKENS;
        const unsigned int count = min(GDN_KH_TOKENS, tokens - first);
        for (unsigned int i = 0; i < count; ++i) {
            float keys[GDN_KH_ROWS];
#pragma unroll
            for (unsigned int r = 0; r < GDN_KH_ROWS; ++r) {
                keys[r] = sk[buffer][i][group * GDN_KH_ROWS + r];
            }
            float decay[VPK], products[VPK], delta[VPK], queried[VPK];
#pragma unroll
            for (unsigned int j = 0; j < VPK; ++j) {
                decay[j] = expf(sg[buffer][i][j]);
                products[j] = 0.0f;
                queried[j] = 0.0f;
            }
#pragma unroll
            for (unsigned int r = 0; r < GDN_KH_ROWS; ++r) {
#pragma unroll
                for (unsigned int j = 0; j < VPK; ++j) {
                    products[j] = fmaf(state[j][r], keys[r], products[j]);
                }
            }
#pragma unroll
            for (unsigned int j = 0; j < VPK; ++j) {
                key_sums[j][group][lane] = products[j];
            }
            __syncthreads();
#pragma unroll
            for (unsigned int j = 0; j < VPK; ++j) {
                float dot = 0.0f;
#pragma unroll
                for (unsigned int g = 0; g < GDN_KH_GROUPS; ++g) {
                    dot += key_sums[j][g][lane];
                }
                delta[j] = (sv[buffer][i][j][lane] - decay[j] * dot) * sb[buffer][i][j];
            }
#pragma unroll
            for (unsigned int r = 0; r < GDN_KH_ROWS; ++r) {
                const float query = sq[buffer][i][group * GDN_KH_ROWS + r];
#pragma unroll
                for (unsigned int j = 0; j < VPK; ++j) {
                    state[j][r] = fmaf(decay[j], state[j][r], keys[r] * delta[j]);
                    queried[j] = fmaf(state[j][r], query, queried[j]);
                }
            }
#pragma unroll
            for (unsigned int j = 0; j < VPK; ++j) {
                query_sums[j][group][lane] = queried[j];
            }
            __syncthreads();
            if (group < VPK) {
                float dot = 0.0f;
#pragma unroll
                for (unsigned int g = 0; g < GDN_KH_GROUPS; ++g) {
                    dot += query_sums[group][g][lane];
                }
                output[((size_t) (sequence * tokens + first + i) * heads + head_of[group]) * GDN_KH_SIZE + column] = dot * scale;
            }
        }
    }
#pragma unroll
    for (unsigned int j = 0; j < VPK; ++j) {
        float * target = output + attention_elements + (sequence * heads + head_of[j]) * state_elements + (size_t) column * GDN_KH_SIZE + group * GDN_KH_ROWS;
#pragma unroll
        for (unsigned int r = 0; r < GDN_KH_ROWS; ++r) {
            target[r] = state[j][r];
        }
    }
}

#define GATED_DELTA_NET_KEYHEAD(VPK)                                                                                     \
    extern "C" __global__ void __launch_bounds__(GDN_KH_THREADS) gated_delta_net_keyhead##VPK##_f32(                     \
            const float * query, const float * key, const float * value, const float * gate, const float * beta,      \
            const float * input_state, float * output, unsigned int key_heads, unsigned int heads, unsigned int tokens, \
            unsigned int sequences, unsigned int repeat_interleave) {                                                 \
        gated_delta_net_keyhead<VPK>(query, key, value, gate, beta, input_state, output, key_heads, heads, tokens,  \
            sequences, repeat_interleave);                                                                            \
    }
GATED_DELTA_NET_KEYHEAD(1)
GATED_DELTA_NET_KEYHEAD(2)
GATED_DELTA_NET_KEYHEAD(3)
GATED_DELTA_NET_KEYHEAD(4)

// gated_delta_net_backward_f32: VJP of gated_delta_net_f32, the BPTT through the
// per-(sequence,head,row) gated delta recurrence. One thread per (sequence,head,
// row): the token loop is inherently sequential per row, rows are independent.
// A forward recompute pass writes each token's post-update state s_t into the
// safter scratch; sprime (post-gate, pre-delta) is recomputed from safter[t-1].
// The reverse pass carries ds through ds_carry scratch. Internal math is double
// to mirror the host f64 reference; grouped q/k, dgate, and dbeta accumulate
// across rows/heads via atomicAdd, so the caller must zero-initialize
// d_query/d_key/d_value/d_gate/d_beta before launch (d_input_state is written
// once per element). scratch buffers safter (count*size*tokens*size) and
// ds_carry (count*size*size) are doubles; they need no pre-init.
extern "C" __global__ void gated_delta_net_backward_f32(
        const float * query,
        const float * key,
        const float * value,
        const float * gate,
        const float * beta,
        const float * input_state,
        const float * d_output,
        float * d_query,
        float * d_key,
        float * d_value,
        float * d_gate,
        float * d_beta,
        float * d_input_state,
        double * safter,
        double * ds_carry,
        unsigned int size,
        unsigned int query_heads,
        unsigned int key_heads,
        unsigned int heads,
        unsigned int tokens,
        unsigned int sequences,
        unsigned int gate_width,
        unsigned int repeat_interleave) {
    const unsigned int index = blockIdx.x;
    const unsigned int count = heads * sequences;
    if (index >= count) {
        return;
    }
    const unsigned int head = index % heads;
    const unsigned int sequence = index / heads;
    const unsigned int query_head = repeat_interleave
        ? head / (heads / query_heads)
        : head % query_heads;
    const unsigned int key_head = repeat_interleave
        ? head / (heads / key_heads)
        : head % key_heads;
    const double scale = 1.0 / sqrt((double) size);
    const unsigned int state_elements = size * size;
    for (unsigned int row = threadIdx.x; row < size; row += blockDim.x) {
        const float * state_in = input_state + index * state_elements + row * size;
        double * row_safter = safter + ((size_t)(index * size + row)) * tokens * size;
        double * ds = ds_carry + ((size_t)(index * size + row)) * size;
        // Forward recompute: write s_t (post-update state) into row_safter[t].
        for (unsigned int token = 0; token < tokens; ++token) {
            const unsigned int value_base =
                ((sequence * tokens + token) * heads + head) * size;
            const unsigned int key_base =
                ((sequence * tokens + token) * key_heads + key_head) * size;
            const unsigned int gate_base =
                ((sequence * tokens + token) * heads + head) * gate_width;
            const double beta_value =
                (double) beta[(sequence * tokens + token) * heads + head];
            const double * prev = token == 0
                ? (const double *) 0
                : row_safter + (size_t)(token - 1) * size;
            double dot = 0.0;
            for (unsigned int column = 0; column < size; ++column) {
                const double g = gate_width == 1
                    ? (double) gate[gate_base]
                    : (double) gate[gate_base + column];
                const double base = token == 0
                    ? (double) state_in[column]
                    : prev[column];
                dot += base * exp(g) * (double) key[key_base + column];
            }
            const double delta =
                ((double) value[value_base + row] - dot) * beta_value;
            double * cur = row_safter + (size_t)token * size;
            for (unsigned int column = 0; column < size; ++column) {
                const double g = gate_width == 1
                    ? (double) gate[gate_base]
                    : (double) gate[gate_base + column];
                const double base = token == 0
                    ? (double) state_in[column]
                    : prev[column];
                cur[column] = base * exp(g) + delta * (double) key[key_base + column];
            }
        }
        // Reverse pass over tokens.
        for (unsigned int column = 0; column < size; ++column) {
            ds[column] = 0.0;
        }
        for (int token = (int) tokens - 1; token >= 0; --token) {
            const unsigned int value_base =
                ((sequence * tokens + token) * heads + head) * size;
            const unsigned int query_base =
                ((sequence * tokens + token) * query_heads + query_head) * size;
            const unsigned int key_base =
                ((sequence * tokens + token) * key_heads + key_head) * size;
            const unsigned int gate_base =
                ((sequence * tokens + token) * heads + head) * gate_width;
            const unsigned int beta_index =
                (sequence * tokens + token) * heads + head;
            const double beta_value = (double) beta[beta_index];
            const double go_value = (double) d_output[value_base + row] * scale;
            const double * saf_cur = row_safter + (size_t)token * size;
            const double * prev = token == 0
                ? (const double *) 0
                : row_safter + (size_t)(token - 1) * size;
            // dsnew = ds + go*q ; dQuery += go*s_t ; dDelta = dsnew . k
            double d_delta = 0.0;
            for (unsigned int column = 0; column < size; ++column) {
                const double dsnew = ds[column] + go_value * (double) query[query_base + column];
                atomicAdd(&d_query[query_base + column], (float)(go_value * saf_cur[column]));
                d_delta += dsnew * (double) key[key_base + column];
            }
            // recompute dot from s'_t (= prev . exp(gate))
            double dot = 0.0;
            for (unsigned int column = 0; column < size; ++column) {
                const double g = gate_width == 1
                    ? (double) gate[gate_base]
                    : (double) gate[gate_base + column];
                const double base = token == 0
                    ? (double) state_in[column]
                    : prev[column];
                dot += base * exp(g) * (double) key[key_base + column];
            }
            const double v_minus = (double) value[value_base + row] - dot;
            const double delta = v_minus * beta_value;
            atomicAdd(&d_value[value_base + row], (float)(d_delta * beta_value));
            atomicAdd(&d_beta[beta_index], (float)(d_delta * v_minus));
            const double d_dot = -d_delta * beta_value;
            // dSp = dsnew + d_dot*k ; dKey += delta*dsnew + d_dot*s' ;
            // dGate += dSp*s' ; ds = dSp*exp(gate)
            double gate_accum = 0.0;
            for (unsigned int column = 0; column < size; ++column) {
                const double g = gate_width == 1
                    ? (double) gate[gate_base]
                    : (double) gate[gate_base + column];
                const double gexp = exp(g);
                const double sprime = (token == 0
                    ? (double) state_in[column]
                    : prev[column]) * gexp;
                const double kc = (double) key[key_base + column];
                const double dsnew = ds[column] + go_value * (double) query[query_base + column];
                const double dsp = dsnew + d_dot * kc;
                atomicAdd(&d_key[key_base + column], (float)(delta * dsnew + d_dot * sprime));
                if (gate_width == 1) {
                    gate_accum += dsp * sprime;
                } else {
                    atomicAdd(&d_gate[gate_base + column], (float)(dsp * sprime));
                }
                ds[column] = dsp * gexp;
            }
            if (gate_width == 1) {
                atomicAdd(&d_gate[gate_base], (float) gate_accum);
            }
        }
        for (unsigned int column = 0; column < size; ++column) {
            d_input_state[index * state_elements + row * size + column] = (float) ds[column];
        }
    }
}

// GDN_BACKWARD_WARPS, GDN_BACKWARD_WARP_ROWS and GDN_BACKWARD_COLUMNS shape gated_delta_net_rows_backward_f32: a block of
// the default 256 threads holds GDN_BACKWARD_BLOCK_ROWS rows of a head's state, a warp GDN_BACKWARD_WARP_ROWS of them and
// a lane GDN_BACKWARD_COLUMNS columns of each (lane, lane + 32, ...), so a state up to GDN_BACKWARD_SIZE_MAX wide, and a
// thread GDN_BACKWARD_THREAD_VALUES values of it. Measured on the RTX 4090 D over Qwen3.5-9B's 32 value heads of 128 and
// 360 tokens: two rows a warp took 1.1 ms, four 1.7 ms (half the blocks to hide the scan's latency), one 1.4 ms (twice
// the atomic sums across blocks).
constexpr unsigned int GDN_BACKWARD_WARPS = 8;
constexpr unsigned int GDN_BACKWARD_WARP_ROWS = 2;
constexpr unsigned int GDN_BACKWARD_COLUMNS = 4;
constexpr unsigned int GDN_BACKWARD_BLOCK_ROWS = GDN_BACKWARD_WARPS * GDN_BACKWARD_WARP_ROWS;
constexpr unsigned int GDN_BACKWARD_SIZE_MAX = CUDA_WARP_WIDTH * GDN_BACKWARD_COLUMNS;
constexpr unsigned int GDN_BACKWARD_THREAD_VALUES = GDN_BACKWARD_WARP_ROWS * GDN_BACKWARD_COLUMNS;

// gdn_backward_warp_sum is value summed over a warp's lanes, which every lane holds.
__device__ __forceinline__ float gdn_backward_warp_sum(float value) {
    return __shfl_sync(0xffffffffu, warp_sum_down(value), 0);
}

// gdn_backward_columns reads a lane's columns of a row of size values, zero past the row.
__device__ __forceinline__ void gdn_backward_columns(float (&columns)[GDN_BACKWARD_COLUMNS], const float * row, unsigned int lane, unsigned int size) {
#pragma unroll
    for (unsigned int part = 0; part < GDN_BACKWARD_COLUMNS; ++part) {
        const unsigned int column = lane + part * CUDA_WARP_WIDTH;
        columns[part] = column < size ? row[column] : 0.0f;
    }
}

// gdn_backward_step carries a warp's rows of a head's state over one token as gated_delta_net_f32 does: each row decays
// by the token's gate, then moves along the key by the delta rule's error against the token's value, scaled by beta.
// Rows past the state's width stay zero.
__device__ __forceinline__ void gdn_backward_step(
        float (&state)[GDN_BACKWARD_WARP_ROWS][GDN_BACKWARD_COLUMNS],
        const float * key,
        const float * value,
        float decay,
        float beta,
        unsigned int lane,
        unsigned int first_row,
        unsigned int size) {
    float keys[GDN_BACKWARD_COLUMNS];
    gdn_backward_columns(keys, key, lane, size);
#pragma unroll
    for (unsigned int index = 0; index < GDN_BACKWARD_WARP_ROWS; ++index) {
        float dot = 0.0f;
#pragma unroll
        for (unsigned int part = 0; part < GDN_BACKWARD_COLUMNS; ++part) {
            state[index][part] *= decay;
            dot += state[index][part] * keys[part];
        }
        const unsigned int row = first_row + index;
        const float delta = ((row < size ? value[row] : 0.0f) - gdn_backward_warp_sum(dot)) * beta;
#pragma unroll
        for (unsigned int part = 0; part < GDN_BACKWARD_COLUMNS; ++part) {
            state[index][part] += delta * keys[part];
        }
    }
}

// gdn_backward_slot moves a thread's values of the state to (keep) or from its slot of scratch, own the thread's first.
__device__ __forceinline__ void gdn_backward_slot(
        float (&state)[GDN_BACKWARD_WARP_ROWS][GDN_BACKWARD_COLUMNS], float * own, unsigned int slot, bool keep) {
#pragma unroll
    for (unsigned int index = 0; index < GDN_BACKWARD_THREAD_VALUES; ++index) {
        float * at = own + ((size_t) slot * GDN_BACKWARD_THREAD_VALUES + index) * blockDim.x;
        float & held = state[index / GDN_BACKWARD_COLUMNS][index % GDN_BACKWARD_COLUMNS];
        if (keep) {
            *at = held;
        } else {
            held = *at;
        }
    }
}

// gated_delta_net_rows_backward_f32 is gated_delta_net_f32's VJP for one sequence from a zero state, one gate a head and
// query and key heads alike (head % key_heads), in f32: from d_output it gives d_query, d_key, d_value, d_gate and
// d_beta as hostmath.GatedDeltaNetBackward does. A block of GDN_BACKWARD_WARPS warps takes GDN_BACKWARD_BLOCK_ROWS rows
// of one head's state, whose recurrences are their own (blockIdx.x = head * groups + group, groups = ceil(size /
// GDN_BACKWARD_BLOCK_ROWS)): a first pass keeps the rows before every chunk-th token, then each chunk, the last first,
// is carried again from its checkpoint keeping the rows before each of its tokens and walked back token by token, the
// rows' gradient carried in registers. A thread keeps only its own values in scratch, ceil(tokens / chunk) + chunk
// slots of GDN_BACKWARD_THREAD_VALUES laid out thread-fastest, gridDim.x * blockDim.x times that many floats. A token's
// query and key gradients sum over the block's rows in shared memory and then over groups and the value heads sharing
// the key head by atomicAdd, as its gate and beta gradients sum over groups: the caller zeroes d_query, d_key, d_gate
// and d_beta. d_value is written once.
extern "C" __global__ void __launch_bounds__(GDN_BACKWARD_WARPS * CUDA_WARP_WIDTH) gated_delta_net_rows_backward_f32(
        const float * query,
        const float * key,
        const float * value,
        const float * gate,
        const float * beta,
        const float * d_output,
        float * d_query,
        float * d_key,
        float * d_value,
        float * d_gate,
        float * d_beta,
        float * scratch,
        unsigned int size,
        unsigned int key_heads,
        unsigned int heads,
        unsigned int tokens,
        unsigned int chunk) {
    // Two of each, so a token's sums are written while the token before it is still read: one barrier a token.
    __shared__ float query_sums[2][GDN_BACKWARD_WARPS][GDN_BACKWARD_SIZE_MAX];
    __shared__ float key_sums[2][GDN_BACKWARD_WARPS][GDN_BACKWARD_SIZE_MAX];
    __shared__ float gate_sums[2][GDN_BACKWARD_WARPS];
    __shared__ float beta_sums[2][GDN_BACKWARD_WARPS];
    const unsigned int groups = (size + GDN_BACKWARD_BLOCK_ROWS - 1) / GDN_BACKWARD_BLOCK_ROWS;
    const unsigned int head = blockIdx.x / groups;
    if (head >= heads) {
        return;
    }
    const unsigned int key_head = head % key_heads;
    const unsigned int lane = threadIdx.x % CUDA_WARP_WIDTH, warp = threadIdx.x / CUDA_WARP_WIDTH;
    const unsigned int first_row = (blockIdx.x % groups) * GDN_BACKWARD_BLOCK_ROWS + warp * GDN_BACKWARD_WARP_ROWS;
    const unsigned int checkpoints = (tokens + chunk - 1) / chunk;
    float * own = scratch + (size_t) blockIdx.x * (checkpoints + chunk) * GDN_BACKWARD_THREAD_VALUES * blockDim.x + threadIdx.x;
    const float scale = rsqrtf((float) size);
    float state[GDN_BACKWARD_WARP_ROWS][GDN_BACKWARD_COLUMNS] = {};
    float carry[GDN_BACKWARD_WARP_ROWS][GDN_BACKWARD_COLUMNS] = {};
    for (unsigned int token = 0; token < (checkpoints - 1) * chunk; ++token) {
        if (token % chunk == 0) {
            gdn_backward_slot(state, own, token / chunk, true);
        }
        const size_t at = (size_t) token * heads + head;
        gdn_backward_step(state, key + ((size_t) token * key_heads + key_head) * size, value + at * size, expf(gate[at]), beta[at], lane, first_row, size);
    }
    gdn_backward_slot(state, own, checkpoints - 1, true);
    for (unsigned int checkpoint = checkpoints; checkpoint-- > 0;) {
        const unsigned int first = checkpoint * chunk, last = min(tokens, first + chunk);
        gdn_backward_slot(state, own, checkpoint, false);
        for (unsigned int token = first; token < last; ++token) {
            gdn_backward_slot(state, own, checkpoints + token - first, true);
            const size_t at = (size_t) token * heads + head;
            if (token + 1 < last) {
                gdn_backward_step(state, key + ((size_t) token * key_heads + key_head) * size, value + at * size, expf(gate[at]), beta[at], lane, first_row, size);
            }
        }
        for (unsigned int token = last; token-- > first;) {
            // state holds the rows before the token.
            gdn_backward_slot(state, own, checkpoints + token - first, false);
            const size_t at = (size_t) token * heads + head;
            const size_t key_at = ((size_t) token * key_heads + key_head) * size;
            const float decay = expf(gate[at]), beta_value = beta[at];
            float keys[GDN_BACKWARD_COLUMNS], queries[GDN_BACKWARD_COLUMNS];
            float query_part[GDN_BACKWARD_COLUMNS] = {}, key_part[GDN_BACKWARD_COLUMNS] = {};
            gdn_backward_columns(keys, key + key_at, lane, size);
            gdn_backward_columns(queries, query + key_at, lane, size);
            float gate_part = 0.0f, beta_part = 0.0f;
#pragma unroll
            for (unsigned int index = 0; index < GDN_BACKWARD_WARP_ROWS; ++index) {
                const unsigned int row = first_row + index;
                const bool held = row < size;
                const float target = held ? value[at * size + row] : 0.0f;
                const float output_gradient = held ? d_output[at * size + row] * scale : 0.0f;
                float decayed[GDN_BACKWARD_COLUMNS], total[GDN_BACKWARD_COLUMNS];
                float dot = 0.0f;
#pragma unroll
                for (unsigned int part = 0; part < GDN_BACKWARD_COLUMNS; ++part) {
                    decayed[part] = state[index][part] * decay;
                    dot += decayed[part] * keys[part];
                }
                const float error = target - gdn_backward_warp_sum(dot);
                const float delta = error * beta_value;
                // total is the gradient of the rows after the token: the carry from the tokens after it and the readout's.
                float delta_gradient = 0.0f;
#pragma unroll
                for (unsigned int part = 0; part < GDN_BACKWARD_COLUMNS; ++part) {
                    total[part] = carry[index][part] + output_gradient * queries[part];
                    query_part[part] += output_gradient * (decayed[part] + delta * keys[part]);
                    delta_gradient += total[part] * keys[part];
                }
                delta_gradient = gdn_backward_warp_sum(delta_gradient);
                if (held && lane == 0) {
                    d_value[at * size + row] = delta_gradient * beta_value;
                }
                beta_part += delta_gradient * error;
                const float dot_gradient = -delta_gradient * beta_value;
#pragma unroll
                for (unsigned int part = 0; part < GDN_BACKWARD_COLUMNS; ++part) {
                    const float decayed_gradient = total[part] + dot_gradient * keys[part];
                    key_part[part] += delta * total[part] + dot_gradient * decayed[part];
                    gate_part += decayed_gradient * decayed[part];
                    carry[index][part] = decayed_gradient * decay;
                }
            }
            const unsigned int parity = token & 1;
#pragma unroll
            for (unsigned int part = 0; part < GDN_BACKWARD_COLUMNS; ++part) {
                const unsigned int column = lane + part * CUDA_WARP_WIDTH;
                if (column < size) {
                    query_sums[parity][warp][column] = query_part[part];
                    key_sums[parity][warp][column] = key_part[part];
                }
            }
            gate_part = warp_sum_down(gate_part);
            if (lane == 0) {
                gate_sums[parity][warp] = gate_part;
                beta_sums[parity][warp] = beta_part;
            }
            __syncthreads();
            for (unsigned int index = threadIdx.x; index < 2 * size; index += blockDim.x) {
                const unsigned int column = index % size;
                const bool keyed = index >= size;
                float sum = 0.0f;
                for (unsigned int part = 0; part < GDN_BACKWARD_WARPS; ++part) {
                    sum += keyed ? key_sums[parity][part][column] : query_sums[parity][part][column];
                }
                atomicAdd((keyed ? d_key : d_query) + key_at + column, sum);
            }
            if (threadIdx.x == 0) {
                float gate_sum = 0.0f, beta_sum = 0.0f;
                for (unsigned int part = 0; part < GDN_BACKWARD_WARPS; ++part) {
                    gate_sum += gate_sums[parity][part];
                    beta_sum += beta_sums[parity][part];
                }
                atomicAdd(d_gate + at, gate_sum);
                atomicAdd(d_beta + at, beta_sum);
            }
        }
    }
}

extern "C" __global__ void gated_linear_attention_f32(
		const float * key,
		const float * value,
		const float * receptance,
		const float * decay,
		const float * input_state,
		float * output,
		unsigned int width,
		unsigned int key_heads,
		unsigned int heads,
		unsigned int tokens,
		unsigned int sequences,
		float scale) {
	const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int count = heads * sequences;
	if (index >= count) {
		return;
	}
	const unsigned int head = index % heads;
	const unsigned int key_head = head / (heads / key_heads);
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
		const unsigned int key_base =
			((sequence * tokens + token) * key_heads + key_head) * width;
		for (unsigned int row = 0; row < width; ++row) {
			float sum = 0.0f;
			float * state_row = state + row * width;
			for (unsigned int column = 0; column < width; ++column) {
				const unsigned int vector_index = vector_base + column;
				const float effective_key = key[key_base + column] *
					(1.0f - decay[vector_index]);
				const float next = state_row[column] * decay[vector_index] +
					effective_key * value[key_base + row];
				state_row[column] = next;
				sum += receptance[vector_index] * next;
			}
			output[vector_base + row] = sum * scale;
		}
	}
}

extern "C" __global__ void rwkv6_f32(
		const float * key,
		const float * value,
		const float * receptance,
		const float * first,
		const float * decay,
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
		for (unsigned int column = 0; column < width; ++column) {
			output[vector_base + column] = 0.0f;
		}
		for (unsigned int row = 0; row < width; ++row) {
			float * state_row = state + row * width;
			const float key_value = key[vector_base + row];
			const float receptance_value = receptance[vector_base + row];
			const float first_value = first[head * width + row];
			const float decay_value = decay[vector_base + row];
			for (unsigned int column = 0; column < width; ++column) {
				const float kv = key_value * value[vector_base + column];
				const float previous = state_row[column];
				output[vector_base + column] +=
					(previous + kv * first_value) * receptance_value;
				state_row[column] = previous * decay_value + kv;
			}
		}
	}
}

extern "C" __global__ void sum_rows_f32(
		const float * input,
		float * output,
		unsigned int width,
		unsigned int rows) {
	const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row >= rows) {
		return;
	}
	float sum = 0.0f;
	for (unsigned int column = 0; column < width; ++column) {
		sum += input[row * width + column];
	}
	output[row] = sum;
}

extern "C" __global__ void fwht_f32(
		const float * input,
		float * output,
		unsigned int width,
		unsigned int rows) {
	const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row >= rows) {
		return;
	}
	const size_t base = (size_t) row * width;
	for (unsigned int column = 0; column < width; ++column) {
		output[base + column] = input[base + column];
	}
	for (unsigned int stride = 1; stride < width; stride *= 2) {
		for (unsigned int block = 0; block < width; block += 2 * stride) {
			for (unsigned int offset = 0; offset < stride; ++offset) {
				const size_t first = base + block + offset;
				const size_t second = first + stride;
				const float a = output[first];
				const float b = output[second];
				output[first] = a + b;
				output[second] = a - b;
			}
		}
	}
	const float scale = rsqrtf((float) width);
	for (unsigned int column = 0; column < width; ++column) {
		output[base + column] *= scale;
	}
}

__device__ bool top_k_before(
		float left,
		unsigned int left_index,
		float right,
		unsigned int right_index) {
	const bool left_nan = isnan(left);
	const bool right_nan = isnan(right);
	if (left_nan != right_nan) {
		return !left_nan;
	}
	if (left == right || left_nan) {
		return left_index < right_index;
	}
	return left > right;
}

// ARGMAX_THREADS: a row's block; a wide row (a vocabulary's logits) is read four values a load across them, so the
// row's reads stay in flight together.
constexpr unsigned int ARGMAX_THREADS = 1024;
constexpr unsigned int ARGMAX_VECTOR = 4;

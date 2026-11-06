// ops_f32_vae.cuh: the Wan causal 3-D VAE decode, in ops_f32.cu's order. ops_f32.cu
// includes it once; it compiles only there.

// ---- Wan causal 3-D VAE decode (streamed chunk-major, latentvideo) ----

extern "C" __global__ void vae_temporal_downsample_f32(
		float * out,
		const float * input,
		const float * prior,
		const float * weight,
		const float * bias,
		unsigned int channels,
		unsigned int frames,
		unsigned int prior_frames,
		unsigned int spatial,
		unsigned int output_frames,
		unsigned int count) {
	const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= count) return;
	const unsigned int output_channel = index / (output_frames * spatial);
	const unsigned int remainder = index - output_channel * output_frames * spatial;
	const unsigned int output_frame = remainder / spatial;
	const unsigned int position = remainder - output_frame * spatial;
	double sum = (double) bias[output_channel];
	for (unsigned int input_channel = 0; input_channel < channels; ++input_channel) {
		for (unsigned int kernel_time = 0; kernel_time < 3; ++kernel_time) {
			const unsigned int joined_frame = 2 * output_frame + kernel_time;
			if (joined_frame > frames) continue;
			float value = 0.0f;
			if (joined_frame == 0) {
				if (prior_frames > 0) value = prior[(input_channel * prior_frames + prior_frames - 1) * spatial + position];
			} else {
				value = input[(input_channel * frames + joined_frame - 1) * spatial + position];
			}
			const unsigned int weight_index = (output_channel * channels + input_channel) * 3 + kernel_time;
			sum += (double) value * (double) weight[weight_index];
		}
	}
	out[index] = (float) sum;
}

// vae_conv_tile: one tile of a stride-1 same-padded causal 3-D convolution,
// fp32 FMA accumulation in the fixed ascending kernel-value order from a fused
// mandatory bias. The kernel's extents are literals, and so is whether the
// input stands at half the outputs' height and width, each of its pixels
// under four of them (the resample stack's nearest-neighbour upsample, fused).
//
// A block of 256 threads computes a tile of 256 positions by 96 output
// channels, each thread 8 positions by 12 channels of it in registers. The
// device runs four warps' arithmetic in a cycle and one warp's read of shared
// memory, so the reads set the pace: a thread's 8 inputs and 12 weights lie
// side by side in the tile and are read four at a time, 5 reads for 96 FMAs,
// where a 4 by 4 block read 8 values singly for 16. The codec's widths are
// multiples of 96, so a tile that wide leaves no channel idle and loads a
// position's inputs once for 96 channels.
//
// A tile is loaded whole input channels at a time, as many as make at most 32
// kernel values, so a kernel value's place in a load fixes its offsets in
// time, height and width when the kernel is compiled, and loading an input is
// an add, a test and a read. A thread loads the inputs of its own position; a
// warp's 32 positions of one kernel value lie side by side in the input and
// are one read of the device's memory. A warp loads 32 kernel values of one
// output channel, which lie side by side in the weight. A thread's reads of
// the device's memory are all issued before any is stored, so they are in
// flight together: read one at a time, the weights alone cost as much as the
// arithmetic.
#define VAE_CONV_TILE 96
#define VAE_CONV_POSITIONS 256
#define VAE_CONV_K 32
#define VAE_CONV_POSITION_BLOCK 8
#define VAE_CONV_CHANNEL_BLOCK 12
#define VAE_CONV_WEIGHT_ROW (VAE_CONV_TILE + 4)
#define VAE_CONV_WEIGHT_ROUNDS (VAE_CONV_K * VAE_CONV_TILE / VAE_CONV_POSITIONS)
template <int KT, int KH, int KW, bool UPSAMPLED>
static __device__ __forceinline__ void vae_conv_tile(
        float * out,
        const float * x,
        const float * cache,
        const float * weight,
        const float * bias,
        const int c_in,
        const int c_out,
        const int in_t,
        const int in_h,
        const int in_w,
        const int cache_t,
        float (* activations)[VAE_CONV_POSITIONS],
        float (* weights)[VAE_CONV_WEIGHT_ROW]) {
    // The kernel values of one input channel, the input channels a load takes,
    // and the kernel values that is.
    const int taps = KT * KH * KW;
    const int load_channels = VAE_CONV_K / taps;
    const int load_values = load_channels * taps;
    const int lane = threadIdx.x;
    const int tx = lane % (VAE_CONV_TILE / VAE_CONV_CHANNEL_BLOCK);
    const int ty = lane / (VAE_CONV_TILE / VAE_CONV_CHANNEL_BLOCK);
    const int plane = in_h * in_w;
    const int positions = in_t * plane;
    const int source_w = UPSAMPLED ? in_w / 2 : in_w;
    const int source_plane = (UPSAMPLED ? in_h / 2 : in_h) * source_w;
    const int channel_tiles = (c_out + VAE_CONV_TILE - 1) / VAE_CONV_TILE;
    const int position_tile = blockIdx.x / channel_tiles;
    const int channel_tile = blockIdx.x - position_tile * channel_tiles;
    const int kernel_values = c_in * taps;
    // The position this thread loads the inputs of. For each offset in time,
    // the buffer that frame is in, where the frame begins in a channel of it
    // and how long a channel of it is; for each offset in height and in width,
    // where that row and that column begin; and for each kernel value of a
    // channel, whether its input lies inside the volume.
    const int own = position_tile * VAE_CONV_POSITIONS + lane;
    const int clamped = own < positions ? own : positions - 1;
    const int own_t = clamped / plane;
    const int own_rem = clamped - own_t * plane;
    const int own_h = own_rem / in_w;
    const int own_w = own_rem - own_h * in_w;
    const float * buffer[KT];
    int frame[KT];
    int channel[KT];
    int row[KH];
    int column[KW];
    unsigned int inside = 0;
    #pragma unroll
    for (int kh = 0; kh < KH; ++kh) {
        const int ih = own_h + kh - KH / 2;
        row[kh] = (UPSAMPLED ? ih >> 1 : ih) * source_w;
    }
    #pragma unroll
    for (int kw = 0; kw < KW; ++kw) {
        const int iw = own_w + kw - KW / 2;
        column[kw] = UPSAMPLED ? iw >> 1 : iw;
    }
    #pragma unroll
    for (int kt = 0; kt < KT; ++kt) {
        const int padded_t = own_t + kt - (2 * (KT / 2) - cache_t);
        const bool in_cache = padded_t < cache_t;
        buffer[kt] = in_cache ? cache : x;
        frame[kt] = (in_cache ? padded_t : padded_t - cache_t) * source_plane;
        channel[kt] = (in_cache ? cache_t : in_t) * source_plane;
        #pragma unroll
        for (int kh = 0; kh < KH; ++kh) {
            const int ih = own_h + kh - KH / 2;
            #pragma unroll
            for (int kw = 0; kw < KW; ++kw) {
                const int iw = own_w + kw - KW / 2;
                if (own < positions && padded_t >= 0 && padded_t < cache_t + in_t &&
                        ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                    inside |= 1u << ((kt * KH + kh) * KW + kw);
                }
            }
        }
    }
    float acc[VAE_CONV_POSITION_BLOCK][VAE_CONV_CHANNEL_BLOCK];
    for (int i = 0; i < VAE_CONV_POSITION_BLOCK; ++i) {
        const int position = position_tile * VAE_CONV_POSITIONS + ty * VAE_CONV_POSITION_BLOCK + i;
        for (int j = 0; j < VAE_CONV_CHANNEL_BLOCK; ++j) {
            const int co = channel_tile * VAE_CONV_TILE + tx * VAE_CONV_CHANNEL_BLOCK + j;
            acc[i][j] = (position < positions && co < c_out) ? bias[co] : 0.0f;
        }
    }
    for (int first = 0; first < c_in; first += load_channels) {
        float input[VAE_CONV_K];
        float fetched[VAE_CONV_WEIGHT_ROUNDS];
        #pragma unroll
        for (int k = 0; k < load_values; ++k) {
            const int ci = first + k / taps;
            const int tap = k % taps;
            const int kt = tap / (KH * KW);
            float value = 0.0f;
            if (ci < c_in && ((inside >> tap) & 1u)) {
                value = buffer[kt][ci * channel[kt] + frame[kt] + row[tap / KW % KH] + column[tap % KW]];
            }
            input[k] = value;
        }
        #pragma unroll
        for (int round = 0; round < VAE_CONV_WEIGHT_ROUNDS; ++round) {
            const int flat = lane + round * VAE_CONV_POSITIONS;
            const int k = flat % VAE_CONV_K;
            const int input_kernel = first * taps + k;
            const int load_co = channel_tile * VAE_CONV_TILE + flat / VAE_CONV_K;
            fetched[round] = (k < load_values && load_co < c_out && input_kernel < kernel_values)
                ? weight[load_co * kernel_values + input_kernel]
                : 0.0f;
        }
        #pragma unroll
        for (int k = 0; k < load_values; ++k) {
            activations[k][lane] = input[k];
        }
        #pragma unroll
        for (int round = 0; round < VAE_CONV_WEIGHT_ROUNDS; ++round) {
            const int flat = lane + round * VAE_CONV_POSITIONS;
            if (flat % VAE_CONV_K < load_values) {
                weights[flat % VAE_CONV_K][flat / VAE_CONV_K] = fetched[round];
            }
        }
        __syncthreads();
        for (int k = 0; k < load_values; ++k) {
            float a[VAE_CONV_POSITION_BLOCK];
            float b[VAE_CONV_CHANNEL_BLOCK];
            const float4 * a4 = reinterpret_cast<const float4 *>(&activations[k][ty * VAE_CONV_POSITION_BLOCK]);
            const float4 * b4 = reinterpret_cast<const float4 *>(&weights[k][tx * VAE_CONV_CHANNEL_BLOCK]);
            for (int i = 0; i < VAE_CONV_POSITION_BLOCK / 4; ++i) {
                const float4 v = a4[i];
                a[4 * i] = v.x; a[4 * i + 1] = v.y; a[4 * i + 2] = v.z; a[4 * i + 3] = v.w;
            }
            for (int j = 0; j < VAE_CONV_CHANNEL_BLOCK / 4; ++j) {
                const float4 v = b4[j];
                b[4 * j] = v.x; b[4 * j + 1] = v.y; b[4 * j + 2] = v.z; b[4 * j + 3] = v.w;
            }
            for (int i = 0; i < VAE_CONV_POSITION_BLOCK; ++i) {
                for (int j = 0; j < VAE_CONV_CHANNEL_BLOCK; ++j) {
                    acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
                }
            }
        }
        __syncthreads();
    }
    for (int i = 0; i < VAE_CONV_POSITION_BLOCK; ++i) {
        const int position = position_tile * VAE_CONV_POSITIONS + ty * VAE_CONV_POSITION_BLOCK + i;
        if (position >= positions) continue;
        for (int j = 0; j < VAE_CONV_CHANNEL_BLOCK; ++j) {
            const int co = channel_tile * VAE_CONV_TILE + tx * VAE_CONV_CHANNEL_BLOCK + j;
            if (co < c_out) {
                out[co * positions + position] = acc[i][j];
            }
        }
    }
}

// vae_causal_conv3d_f32: stride-1 same-padded causal 3-D convolution over one
// streamed chunk with a temporal prefix cache of cache_t frames (vae_conv_tile).
// Output dims equal input dims (stride 1 everywhere in the decoder). It has
// the three kernels the codec has, 3x3x3, temporal 3x1x1 and 1x1x1, and writes
// nothing for another: the launch refuses one. Launched with 256 threads a
// block, a block for each 256 positions of each 96 output channels.
extern "C" __global__ void vae_causal_conv3d_f32(
        float * out,
        const float * x,
        const float * cache,
        const float * weight,
        const float * bias,
        int c_in,
        int c_out,
        int in_t,
        int in_h,
        int in_w,
        int kt_n,
        int kh_n,
        int kw_n,
        int cache_t) {
    __shared__ __align__(16) float activations[VAE_CONV_K][VAE_CONV_POSITIONS];
    __shared__ __align__(16) float weights[VAE_CONV_K][VAE_CONV_WEIGHT_ROW];
    if (kt_n == 3 && kh_n == 3 && kw_n == 3) {
        vae_conv_tile<3, 3, 3, false>(out, x, cache, weight, bias, c_in, c_out, in_t, in_h, in_w, cache_t, activations, weights);
    } else if (kt_n == 1 && kh_n == 1 && kw_n == 1) {
        vae_conv_tile<1, 1, 1, false>(out, x, cache, weight, bias, c_in, c_out, in_t, in_h, in_w, cache_t, activations, weights);
    } else if (kt_n == 3 && kh_n == 1 && kw_n == 1) {
        vae_conv_tile<3, 1, 1, false>(out, x, cache, weight, bias, c_in, c_out, in_t, in_h, in_w, cache_t, activations, weights);
    }
}

// vae_upsample2d_f32: nearest-neighbor 2x spatial upsample fused with a
// same-padded 3x3 Conv2d per frame (Wan resample stack), fp32 FMA, fused
// mandatory bias (vae_conv_tile). The outputs are frames of out_h by out_w
// and the input frames of half that. Launched as vae_causal_conv3d_f32 is.
extern "C" __global__ void vae_upsample2d_f32(
        float * out,
        const float * x,
        const float * weight,
        const float * bias,
        int c_in,
        int c_out,
        int frames,
        int out_h,
        int out_w) {
    __shared__ __align__(16) float activations[VAE_CONV_K][VAE_CONV_POSITIONS];
    __shared__ __align__(16) float weights[VAE_CONV_K][VAE_CONV_WEIGHT_ROW];
    vae_conv_tile<1, 3, 3, true>(out, x, x, weight, bias, c_in, c_out, frames, out_h, out_w, 0, activations, weights);
}

// vae_downsample2d_f32: the resample stack's stride-2 3x3 Conv2d per frame,
// its input padded by a zero row below and a zero column to the right, f64
// accumulation in the fixed ascending kernel-value order from the bias
// (reference engine convention). The outputs are frames of out_h by out_w.
// Launched as vae_causal_conv3d_f32 is, a block for each 256 positions of
// each 96 output channels.
//
// The device has one double-precision unit for every 64 of single, and this
// kernel is bound by them. A thread that sums one output spends three of
// their operations on a product, two operands widened and the product added;
// a thread here sums 8 positions by 6 channels and widens 14 operands for 48
// products. A block computes its tile in two halves of 128 positions, so that
// a thread holds no more than 48 sums, and loads each half as vae_conv_tile
// loads a tile: three input channels a load, a thread's reads of the device's
// memory issued together.
#define VAE_DOWN_POSITIONS (VAE_CONV_POSITIONS / 2)
#define VAE_DOWN_POSITION_BLOCK 8
#define VAE_DOWN_CHANNEL_BLOCK 6
#define VAE_DOWN_TAPS 9
#define VAE_DOWN_VALUES (VAE_CONV_K / VAE_DOWN_TAPS * VAE_DOWN_TAPS)
#define VAE_DOWN_INPUT_ROUNDS ((VAE_DOWN_VALUES + 1) / 2)
extern "C" __global__ void vae_downsample2d_f32(
        float * out,
        const float * x,
        const float * weight,
        const float * bias,
        int channels,
        int frames,
        int height,
        int width,
        int out_h,
        int out_w) {
    __shared__ __align__(16) float activations[VAE_CONV_K][VAE_DOWN_POSITIONS];
    __shared__ __align__(16) float weights[VAE_CONV_K][VAE_CONV_WEIGHT_ROW];
    const int lane = threadIdx.x;
    const int tx = lane % (VAE_CONV_TILE / VAE_DOWN_CHANNEL_BLOCK);
    const int ty = lane / (VAE_CONV_TILE / VAE_DOWN_CHANNEL_BLOCK);
    const int plane = out_h * out_w;
    const int positions = frames * plane;
    const int channel_tiles = (channels + VAE_CONV_TILE - 1) / VAE_CONV_TILE;
    const int position_tile = blockIdx.x / channel_tiles;
    const int channel_tile = blockIdx.x - position_tile * channel_tiles;
    const int kernel_values = channels * VAE_DOWN_TAPS;
    // A thread loads, for one position of the half, the kernel values of one
    // parity: two threads share a position.
    const int parity = lane / VAE_DOWN_POSITIONS;
    const int column = lane % VAE_DOWN_POSITIONS;
    for (int half = 0; half < VAE_CONV_POSITIONS / VAE_DOWN_POSITIONS; ++half) {
        const int origin = position_tile * VAE_CONV_POSITIONS + half * VAE_DOWN_POSITIONS;
        const int own = origin + column;
        const int clamped = own < positions ? own : positions - 1;
        const int own_t = clamped / plane;
        const int own_rem = clamped - own_t * plane;
        const int own_h = own_rem / out_w;
        const int own_w = own_rem - own_h * out_w;
        const int source = (own_t * height + 2 * own_h) * width + 2 * own_w;
        unsigned int inside = 0;
        #pragma unroll
        for (int tap = 0; tap < VAE_DOWN_TAPS; ++tap) {
            if (own < positions && 2 * own_h + tap / 3 < height && 2 * own_w + tap % 3 < width) {
                inside |= 1u << tap;
            }
        }
        double acc[VAE_DOWN_POSITION_BLOCK][VAE_DOWN_CHANNEL_BLOCK];
        for (int i = 0; i < VAE_DOWN_POSITION_BLOCK; ++i) {
            const int position = origin + ty * VAE_DOWN_POSITION_BLOCK + i;
            for (int j = 0; j < VAE_DOWN_CHANNEL_BLOCK; ++j) {
                const int co = channel_tile * VAE_CONV_TILE + tx * VAE_DOWN_CHANNEL_BLOCK + j;
                acc[i][j] = (position < positions && co < channels) ? (double) bias[co] : 0.0;
            }
        }
        for (int first = 0; first < channels; first += VAE_CONV_K / VAE_DOWN_TAPS) {
            float input[VAE_DOWN_INPUT_ROUNDS];
            float fetched[VAE_CONV_WEIGHT_ROUNDS];
            #pragma unroll
            for (int round = 0; round < VAE_DOWN_INPUT_ROUNDS; ++round) {
                const int k = 2 * round + parity;
                const int ci = first + k / VAE_DOWN_TAPS;
                const int tap = k % VAE_DOWN_TAPS;
                float value = 0.0f;
                if (k < VAE_DOWN_VALUES && ci < channels && ((inside >> tap) & 1u)) {
                    value = x[ci * frames * height * width + source + tap / 3 * width + tap % 3];
                }
                input[round] = value;
            }
            #pragma unroll
            for (int round = 0; round < VAE_CONV_WEIGHT_ROUNDS; ++round) {
                const int flat = lane + round * VAE_CONV_POSITIONS;
                const int k = flat % VAE_CONV_K;
                const int input_kernel = first * VAE_DOWN_TAPS + k;
                const int load_co = channel_tile * VAE_CONV_TILE + flat / VAE_CONV_K;
                fetched[round] = (k < VAE_DOWN_VALUES && load_co < channels && input_kernel < kernel_values)
                    ? weight[load_co * kernel_values + input_kernel]
                    : 0.0f;
            }
            #pragma unroll
            for (int round = 0; round < VAE_DOWN_INPUT_ROUNDS; ++round) {
                if (2 * round + parity < VAE_DOWN_VALUES) {
                    activations[2 * round + parity][column] = input[round];
                }
            }
            #pragma unroll
            for (int round = 0; round < VAE_CONV_WEIGHT_ROUNDS; ++round) {
                const int flat = lane + round * VAE_CONV_POSITIONS;
                if (flat % VAE_CONV_K < VAE_DOWN_VALUES) {
                    weights[flat % VAE_CONV_K][flat / VAE_CONV_K] = fetched[round];
                }
            }
            __syncthreads();
            for (int k = 0; k < VAE_DOWN_VALUES; ++k) {
                double a[VAE_DOWN_POSITION_BLOCK];
                double b[VAE_DOWN_CHANNEL_BLOCK];
                const float4 * a4 = reinterpret_cast<const float4 *>(&activations[k][ty * VAE_DOWN_POSITION_BLOCK]);
                for (int i = 0; i < VAE_DOWN_POSITION_BLOCK / 4; ++i) {
                    const float4 v = a4[i];
                    a[4 * i] = (double) v.x; a[4 * i + 1] = (double) v.y; a[4 * i + 2] = (double) v.z; a[4 * i + 3] = (double) v.w;
                }
                for (int j = 0; j < VAE_DOWN_CHANNEL_BLOCK; ++j) {
                    b[j] = (double) weights[k][tx * VAE_DOWN_CHANNEL_BLOCK + j];
                }
                for (int i = 0; i < VAE_DOWN_POSITION_BLOCK; ++i) {
                    for (int j = 0; j < VAE_DOWN_CHANNEL_BLOCK; ++j) {
                        acc[i][j] += a[i] * b[j];
                    }
                }
            }
            __syncthreads();
        }
        for (int i = 0; i < VAE_DOWN_POSITION_BLOCK; ++i) {
            const int position = origin + ty * VAE_DOWN_POSITION_BLOCK + i;
            if (position >= positions) continue;
            for (int j = 0; j < VAE_DOWN_CHANNEL_BLOCK; ++j) {
                const int co = channel_tile * VAE_CONV_TILE + tx * VAE_DOWN_CHANNEL_BLOCK + j;
                if (co < channels) {
                    out[co * positions + position] = (float) acc[i][j];
                }
            }
        }
    }
}

// vae_channel_rms_norm_f32: RMS over channels at each position, sqrt(C)
// scale, 1e-12 zero guard, f64 accumulation (reference engine convention),
// optional fused SiLU applied after the f32 rounding (host rounding order).
// One thread per position.
//
// The squares are summed in the order a block of 256 threads summed them
// when every position had a block to itself: 256 partial sums, each of the
// channels one thread strode over, folded by halves from the widest stride
// down. A thread takes those partial sums in bit-reversed order and folds
// them on a stack, which is the same tree with the same operands at every
// node, so each value is the value that kernel gave. Partial sums past the
// channels are zero and fold to nothing, so the tree is walked only as wide
// as the channels reach.
#define VAE_RMS_FOLD 256
#define VAE_RMS_FOLD_BITS 8
extern "C" __global__ void vae_channel_rms_norm_f32(
        float * out,
        const float * x,
        const float * gamma,
        int channels,
        int plane,
        int apply_silu) {
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= plane) {
        return;
    }
    int bits = 0;
    while (bits < VAE_RMS_FOLD_BITS && (1 << bits) < channels) {
        ++bits;
    }
    double folded[VAE_RMS_FOLD_BITS + 1];
    int depth = 0;
    for (unsigned int leaf = 0; leaf < (1u << bits); ++leaf) {
        const int first = bits ? (int) (__brev(leaf) >> (32 - bits)) : 0;
        double sum = 0.0;
        for (int ch = first; ch < channels; ch += VAE_RMS_FOLD) {
            const double v = (double) x[ch * plane + pos];
            sum += v * v;
        }
        for (unsigned int carry = leaf; carry & 1u; carry >>= 1) {
            sum = folded[--depth] + sum;
        }
        folded[depth++] = sum;
    }
    double norm = sqrt(folded[0]);
    if (norm < 1.0e-12) {
        norm = 1.0e-12;
    }
    const double scale = sqrt((double) channels) / norm;
    for (int ch = 0; ch < channels; ++ch) {
        const int idx = ch * plane + pos;
        float value = (float) ((double) x[idx] * scale * (double) gamma[ch]);
        if (apply_silu) {
            const double s = (double) value;
            value = (float) (s / (1.0 + exp(-s)));
        }
        out[idx] = value;
    }
}

// vae_spatial_attention_f32: QKV-packed per-frame spatial self-attention.
// Layout qkv[section*channels*groups*frame + (ch*groups+g)*frame + pos],
// sections q=0,k=1,v=2; groups is the frame count, frame the spatial extent.
// One block per (group, query); shared = frame floats (8-aligned) + one
// double per thread. Residual add stays with the caller.
extern "C" __global__ void vae_spatial_attention_f32(
        const float * qkv,
        float * out,
        int channels,
        int groups,
        int frame,
        float scale) {
    extern __shared__ unsigned char vae_attention_scratch[];
    float * scores = (float *) vae_attention_scratch;
    const int score_bytes = ((frame * (int) sizeof(float) + 7) / 8) * 8;
    double * reduce = (double *) (vae_attention_scratch + score_bytes);
    const int qi = blockIdx.x % frame;
    const int g = blockIdx.x / frame;
    const int tid = threadIdx.x;
    const int koff = channels * groups * frame;
    const int voff = 2 * channels * groups * frame;
    for (int kj = tid; kj < frame; kj += blockDim.x) {
        float dot = 0.0f;
        for (int ch = 0; ch < channels; ++ch) {
            dot = fmaf(qkv[(ch * groups + g) * frame + qi],
                       qkv[koff + (ch * groups + g) * frame + kj], dot);
        }
        scores[kj] = dot * scale;
    }
    __syncthreads();
    float local_max = -3.402823466e38f;
    for (int j = tid; j < frame; j += blockDim.x) {
        local_max = fmaxf(local_max, scores[j]);
    }
    reduce[tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride; stride >>= 1) {
        if (tid < stride) {
            reduce[tid] = fmax(reduce[tid], reduce[tid + stride]);
        }
        __syncthreads();
    }
    const float mx = (float) reduce[0];
    __syncthreads();
    double sum = 0.0;
    for (int j = tid; j < frame; j += blockDim.x) {
        const float e = expf(scores[j] - mx);
        scores[j] = e;
        sum += e;
    }
    reduce[tid] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride; stride >>= 1) {
        if (tid < stride) {
            reduce[tid] += reduce[tid + stride];
        }
        __syncthreads();
    }
    const float inv = (float) (1.0 / reduce[0]);
    for (int ch = tid; ch < channels; ch += blockDim.x) {
        float acc = 0.0f;
        for (int j = 0; j < frame; ++j) {
            acc = fmaf(scores[j] * inv, qkv[voff + (ch * groups + g) * frame + j], acc);
        }
        out[(ch * groups + g) * frame + qi] = acc;
    }
}

// vae_time_interleave_f32: temporal upsample reshape - the time conv's
// doubled channel output [2c][t][s] interleaves to [c][2t][s]; even output
// frames from the first channel half.
extern "C" __global__ void vae_time_interleave_f32(
        float * out,
        const float * convolved,
        int channels,
        int frames,
        int spatial,
        unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const int out_frames = 2 * frames;
    const int ch = (int) index / (out_frames * spatial);
    int rem = (int) index - ch * out_frames * spatial;
    const int ti = rem / spatial;
    const int pos = rem - ti * spatial;
    out[index] = convolved[(((ti & 1) * channels + ch) * frames + ti / 2) * spatial + pos];
}

// vae_temporal_cache_update_f32: reference temporal_cache_update semantics.
// mode 0: keep the last two input frames, joining the prior cache's final
// frame when the chunk is a single frame; mode 1: zero prefix then the chunk
// (temporal upsample first-cache convention after the skipped chunk).
extern "C" __global__ void vae_temporal_cache_update_f32(
        float * out,
        const float * current,
        const float * prior,
        int channels,
        int current_frames,
        int prior_frames,
        int out_frames,
        int spatial,
        int mode,
        unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const int ch = (int) index / (out_frames * spatial);
    int rem = (int) index - ch * out_frames * spatial;
    const int ti = rem / spatial;
    const int pos = rem - ti * spatial;
    if (mode) {
        out[index] = ti == 0 ? 0.0f : current[(ch * current_frames + ti - 1) * spatial + pos];
        return;
    }
    if (current_frames >= 2) {
        const int src = current_frames - out_frames + ti;
        out[index] = current[(ch * current_frames + src) * spatial + pos];
        return;
    }
    if (prior_frames > 0 && ti == 0) {
        out[index] = prior[(ch * prior_frames + prior_frames - 1) * spatial + pos];
        return;
    }
    const int src = ti - (prior_frames > 0 ? 1 : 0);
    out[index] = current[(ch * current_frames + src) * spatial + pos];
}

// vae_causal_stage_f16: the input of a causal convolution as a convolution
// library takes it. A convolution over time reads the frames before its
// chunk, which vae_conv_tile reads from the cache beside the chunk and a
// library takes only as part of its input. Here the chunk stands behind them
// in one tensor of out_frames frames, in half precision: the frames before
// the chunk are the cache's, and zero where the cache holds fewer than
// out_frames - frames. With out_frames equal to frames it is the chunk in
// half precision. The cache is single precision, as vae_temporal_cache_update_f32
// keeps one, or half precision when cache_half is set, as vae_tail_frames_f16
// keeps one: a frame kept in half precision is the half this would write of it.
extern "C" __global__ void vae_causal_stage_f16(
        unsigned short * out,
        const float * x,
        const void * cache,
        int frames,
        int cache_frames,
        int cache_half,
        int out_frames,
        int spatial,
        unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const int ch = (int) index / (out_frames * spatial);
    const int rem = (int) index - ch * out_frames * spatial;
    const int ti = rem / spatial;
    const int pos = rem - ti * spatial;
    const int lead = out_frames - frames;
    if (ti >= lead) {
        out[index] = __half_as_ushort(__float2half_rn(x[(ch * frames + ti - lead) * spatial + pos]));
        return;
    }
    if (ti < lead - cache_frames) {
        out[index] = 0;
        return;
    }
    const int kept = (ch * cache_frames + ti - (lead - cache_frames)) * spatial + pos;
    out[index] = cache_half
        ? ((const unsigned short *) cache)[kept]
        : __half_as_ushort(__float2half_rn(((const float *) cache)[kept]));
}

// vae_tail_frames_f16: the last out_frames frames of each channel of a staged
// tensor of frames frames, which are the frames the next chunk's convolution
// reads before its own: the cache vae_temporal_cache_update_f32 keeps of a
// convolution's input, read from what vae_causal_stage_f16 staged of it.
extern "C" __global__ void vae_tail_frames_f16(
        unsigned short * out,
        const unsigned short * staged,
        int frames,
        int out_frames,
        int spatial,
        unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const int ch = (int) index / (out_frames * spatial);
    const int rem = (int) index - ch * out_frames * spatial;
    out[index] = staged[(ch * frames + frames - out_frames) * spatial + rem];
}

// vae_spatial_stage_f16: the input of a convolution over a frame's rows and
// columns as a convolution library takes it, where a kernel reads it in
// place. Each frame of in_h by in_w is written in half precision as a frame
// of out_h by out_w: a value to each scale by scale of its positions, and
// zero beyond the frame. At scale two, out twice in, it is the frame enlarged
// to its nearest value, which vae_upsample2d_f32 reads as it convolves. At
// scale one, out one more than in, it is the frame with the row of zeros
// below it and the column to its right that vae_downsample2d_f32 reads past
// its edge.
extern "C" __global__ void vae_spatial_stage_f16(
        unsigned short * out,
        const float * x,
        int in_h,
        int in_w,
        int out_h,
        int out_w,
        int scale,
        unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const int plane = out_h * out_w;
    const int frame = (int) index / plane;
    const int rem = (int) index - frame * plane;
    const int row = rem / out_w / scale;
    const int column = rem % out_w / scale;
    out[index] = row < in_h && column < in_w
        ? __half_as_ushort(__float2half_rn(x[(frame * in_h + row) * in_w + column]))
        : 0;
}

// vae_widen_bias_f32: what a convolution library wrote in half precision, in
// single precision with its channel's bias added: each of a channel's plane
// values takes the channel's bias, which stays single precision.
extern "C" __global__ void vae_widen_bias_f32(
        float * out,
        const unsigned short * convolved,
        const float * bias,
        int plane,
        unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    out[index] = __half2float(__ushort_as_half(convolved[index])) + bias[(int) index / plane];
}

__device__ __forceinline__ unsigned int mad_value_rank(
        const float * row,
        unsigned int width,
        unsigned int index) {
    unsigned int rank = 0;
    const float value = row[index];
    for (unsigned int other = 0; other < width; ++other) {
        rank += row[other] < value || (row[other] == value && other < index);
    }
    return rank;
}

__device__ __forceinline__ float mad_raw_weight(unsigned int rank, unsigned int width) {
    return 2.0f * ((float)rank + 0.5f) / (float)width - 1.0f;
}

__device__ __forceinline__ unsigned int mad_distance_rank(
        const float * row,
        unsigned int width,
        double location,
        unsigned int index) {
    const double distance = fabs((double)row[index] - location);
    const unsigned int value_rank = mad_value_rank(row, width, index);
    const bool left = (double)row[index] < location;
    unsigned int rank = 0;
    for (unsigned int other = 0; other < width; ++other) {
        if (other == index) {
            continue;
        }
        const double other_distance = fabs((double)row[other] - location);
        if (other_distance < distance) {
            ++rank;
            continue;
        }
        if (other_distance != distance) {
            continue;
        }
        const bool other_left = (double)row[other] < location;
        const unsigned int other_value_rank = mad_value_rank(row, width, other);
        if ((other_left && !left) ||
            (other_left == left && left && other_value_rank > value_rank) ||
            (other_left == left && !left && other_value_rank < value_rank)) {
            ++rank;
        }
    }
    return rank;
}

__device__ __forceinline__ void mad_row_facts(
        const float * row,
        unsigned int width,
        float epsilon,
        double * location_out,
        double * inverse_out,
        double * c_out) {
    double location = 0.0;
    for (unsigned int index = 0; index < width; ++index) {
        const float normalized = mad_raw_weight(mad_value_rank(row, width, index), width) / (float)width;
        location += (double)row[index] * (double)normalized;
    }
    const double two_over_width = 2.0 / (double)width;
    double scale = 0.0;
    double c = 0.0;
    for (unsigned int index = 0; index < width; ++index) {
        const double distance = fabs((double)row[index] - location);
        const float raw = mad_raw_weight(mad_distance_rank(row, width, location, index), width);
        scale += distance * (double)raw;
        c += (double)raw * two_over_width * ((double)row[index] >= location ? 1.0 : -1.0);
    }
    *location_out = location;
    *inverse_out = 1.0 / (scale * two_over_width + (double)epsilon);
    *c_out = c;
}

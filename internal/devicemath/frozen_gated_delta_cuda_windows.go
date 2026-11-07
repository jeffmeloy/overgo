//go:build windows

package devicemath

import (
	"errors"
	"math"

	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/kernel"
	"overgo/internal/hostmath"
)

// gatedDeltaBackwardRows and gatedDeltaBackwardWidth are the state rows a block of gated_delta_net_rows_backward_f32
// holds and the widest state it takes (GDN_BACKWARD_BLOCK_ROWS and GDN_BACKWARD_SIZE_MAX): each slot of a block's
// scratch holds their product of values.
const (
	gatedDeltaBackwardRows  = 16
	gatedDeltaBackwardWidth = 128
)

// siluGate is weighted_rms_gate_f32's activation code for a gate taken through SiLU.
const siluGate = 1

// keyBlocks are the blocks of the convolution's rows the key heads take, q and k, before v's.
const keyBlocks = 2

// FrozenGatedDeltaLayer is one Qwen3.5 gated-delta decoder layer on the card with every weight frozen, as
// hostmath.HybridDecoderLayerForward computes a linear layer from a zero recurrent state: out = h + MLP(RMSNorm(h,
// PostNorm)), h = x + GatedDeltaMix(RMSNorm(x, InputNorm)). Its weights are put on the card once in F32; Forward keeps
// on the card what Backward reads, and Backward gives the gradient with respect to the layer's input alone.
type FrozenGatedDeltaLayer struct {
	frozenLayer
	weights frozenGatedDeltaWeights
	cache   frozenGatedDeltaCache
	work    frozenGatedDeltaWork
	chunk   int
}

// frozenGatedDeltaWeights are the mix's weights on the card, the projections row-major [out, in].
type frozenGatedDeltaWeights struct {
	q, k, v, z, beta, alpha, convQ, convK, convV, timeStep, decay, norm, out driver.DevicePtr
}

// frozenGatedDeltaCache is what the forward keeps for the backward: the convolution before its SiLU, the q, k and v
// blocks token-major in turn; z; the gate products, beta and the gates; the recurrence's output with its final state
// after it; and the zero state.
type frozenGatedDeltaCache struct {
	convolved, z, alpha, beta, gate, recurrent, zero driver.DevicePtr
}

// frozenGatedDeltaWork is the passes' scratch by width: q, k and v blocks (two), the normalized q and k, value rows
// (three), a value a token and head (two), and the recurrence backward's checkpoints.
type frozenGatedDeltaWork struct {
	channels, channels2, keys, values, values2, values3, heads, heads2, recurrence driver.DevicePtr
}

// OpenFrozenGatedDeltaLayer puts the gated-delta decoder layer w on the card for sequences of d.Tokens tokens, every
// weight frozen, with the cache its forward keeps and the scratch its backward takes, in one allocation the layer
// releases on Close; it runs on this session's module and cuBLAS handle. Its convolution has no bias, as Qwen3.5's has
// none, and its heads are at most 128 wide. Its memory grows with the tokens: the cache and scratch by rows of the
// layer's widths, the recurrence backward's checkpoints by a state a value head for about twice the root of the tokens.
func (s *ResidentOpsSession) OpenFrozenGatedDeltaLayer(w hostmath.HybridLayerWeights, d hostmath.HybridLayerDims) (*FrozenGatedDeltaLayer, error) {
	g := d.GDN
	tokens, hidden, inter, heads, width := d.Tokens, d.Hidden, d.Inter, g.ValueHeads, g.HeadDim
	keyDim, valDim := g.KeyHeads*width, heads*width
	channels := keyBlocks*keyDim + valDim
	if s == nil || s.closed || !w.IsLinear || min(tokens, hidden, inter, g.KeyHeads, heads, width, g.ConvK) <= 0 || width > gatedDeltaBackwardWidth ||
		g.Tokens != tokens || g.Hidden != hidden || g.OutDim != hidden || w.GDN.ConvBiasQ != nil || w.GDN.ConvBiasK != nil || w.GDN.ConvBiasV != nil {
		return nil, errors.New("frozen gated-delta layer: needs an open session and a linear layer of positive geometry, heads at most 128 wide and no convolution bias")
	}
	chunk := int(math.Ceil(math.Sqrt(float64(tokens))))
	slots := (tokens+chunk-1)/chunk + chunk
	blocks := heads * ((width + gatedDeltaBackwardRows - 1) / gatedDeltaBackwardRows)
	l := &FrozenGatedDeltaLayer{chunk: chunk}
	lw, c, k := &l.weights, &l.cache, &l.work
	weights := []frozenWeight{
		{frozenBuffer{&lw.q, keyDim * hidden}, w.GDN.Wq}, {frozenBuffer{&lw.k, keyDim * hidden}, w.GDN.Wk}, {frozenBuffer{&lw.v, valDim * hidden}, w.GDN.Wv},
		{frozenBuffer{&lw.z, valDim * hidden}, w.GDN.Wz}, {frozenBuffer{&lw.beta, heads * hidden}, w.GDN.Wbeta}, {frozenBuffer{&lw.alpha, heads * hidden}, w.GDN.Walpha},
		{frozenBuffer{&lw.convQ, keyDim * g.ConvK}, w.GDN.ConvQ}, {frozenBuffer{&lw.convK, keyDim * g.ConvK}, w.GDN.ConvK}, {frozenBuffer{&lw.convV, valDim * g.ConvK}, w.GDN.ConvV},
		{frozenBuffer{&lw.timeStep, heads}, w.GDN.TimeStep}, {frozenBuffer{&lw.decay, heads}, w.GDN.A}, {frozenBuffer{&lw.norm, width}, w.GDN.Norm},
		{frozenBuffer{&lw.out, hidden * valDim}, w.GDN.Wout},
	}
	buffers := []frozenBuffer{
		{&c.convolved, tokens * channels}, {&c.z, tokens * valDim},
		{&c.alpha, tokens * heads}, {&c.beta, tokens * heads}, {&c.gate, tokens * heads},
		{&c.recurrent, tokens*valDim + heads*width*width}, {&c.zero, heads * width * width},
		{&k.channels, tokens * channels}, {&k.channels2, tokens * channels}, {&k.keys, keyBlocks * tokens * keyDim},
		{&k.values, tokens * valDim}, {&k.values2, tokens * valDim}, {&k.values3, tokens * valDim},
		{&k.heads, tokens * heads}, {&k.heads2, tokens * heads},
		{&k.recurrence, blocks * slots * gatedDeltaBackwardRows * gatedDeltaBackwardWidth},
	}
	if err := l.open(s, "frozen gated-delta layer", l, w, d, weights, buffers); err != nil {
		return nil, err
	}
	return l, nil
}

// mixBlocks are the q, k and v blocks of a buffer laid out as the convolution's rows, token-major each.
type mixBlocks struct{ q, k, v driver.DevicePtr }

// blocks splits base into the convolution's q, k and v blocks.
func (l *FrozenGatedDeltaLayer) blocks(base driver.DevicePtr) mixBlocks {
	keyRows := l.d.Tokens * l.d.GDN.KeyHeads * l.d.GDN.HeadDim
	return mixBlocks{base, ptrAt(base, keyRows), ptrAt(base, keyBlocks*keyRows)}
}

// convolve runs the short convolution over input's q, k and v blocks into output's, each block with its kernels;
// transpose 1 runs its VJP with respect to its input.
func (l *FrozenGatedDeltaLayer) convolve(r *frozenRun, input, output driver.DevicePtr, transpose int) {
	g := l.d.GDN
	in, out := l.blocks(input), l.blocks(output)
	keyDim, valDim := g.KeyHeads*g.HeadDim, g.ValueHeads*g.HeadDim
	for _, block := range []struct {
		input, weights, output driver.DevicePtr
		width                  int
	}{{in.q, l.weights.convQ, out.q, keyDim}, {in.k, l.weights.convK, out.k, keyDim}, {in.v, l.weights.convV, out.v, valDim}} {
		r.each("depthwise_causal_conv_f32", l.d.Tokens*block.width, block.input, block.weights, block.output, block.width, l.d.Tokens, g.ConvK, transpose)
	}
}

// forwardMix queues hostmath.GatedDeltaMixForward from a zero state over the normed rows.
func (l *FrozenGatedDeltaLayer) forwardMix(r *frozenRun) {
	d, g, w, c, k, a := l.d, l.d.GDN, &l.weights, &l.cache, &l.work, &l.around
	tokens, hidden, keyHeads, heads, width := d.Tokens, d.Hidden, g.KeyHeads, g.ValueHeads, g.HeadDim
	keyDim, valDim, gates := keyHeads*width, heads*width, tokens*heads
	projected := l.blocks(k.channels)
	r.project(a.normed, w.q, projected.q, tokens, hidden, keyDim)
	r.project(a.normed, w.k, projected.k, tokens, hidden, keyDim)
	r.project(a.normed, w.v, projected.v, tokens, hidden, valDim)
	r.project(a.normed, w.z, c.z, tokens, hidden, valDim)
	r.project(a.normed, w.beta, c.beta, tokens, hidden, heads)
	r.project(a.normed, w.alpha, c.alpha, tokens, hidden, heads)
	l.convolve(r, k.channels, c.convolved, 0)
	l.recurrenceInputs(r)
	// beta = sigmoid(Wbeta·x); gate = softplus(Walpha·x + time step) * A, the head's values broadcast over tokens.
	r.each("sigmoid_f32", gates, c.beta, c.beta, gates)
	r.each("broadcast_add_f32", gates, c.alpha, w.timeStep, c.gate, gates, heads, tokens, 1, 1, heads, 1, 1, 1, heads, tokens, 1)
	r.each("softplus_f32", gates, c.gate, c.gate, gates)
	r.each("broadcast_multiply_f32", gates, c.gate, w.decay, c.gate, gates, heads, tokens, 1, 1, heads, 1, 1, 1, heads, tokens, 1)
	normalized, mixed := l.blocks(k.keys), l.blocks(k.channels)
	warps := int(deviceBlockThreads) / kernel.WarpThreads()
	r.kernel("gated_delta_net_rows_f32", driver.Dim3{X: uint32(heads), Y: uint32((width + warps - 1) / warps), Z: 1},
		normalized.q, normalized.k, mixed.v, c.gate, c.beta, c.zero, c.recurrent, width, keyHeads, keyHeads, heads, tokens, 1, 1, 0)
	r.rows("weighted_rms_gate_f32", gates, c.recurrent, c.recurrent, c.z, w.norm, k.values, siluGate, 0, width, gates, g.Eps)
	r.project(k.values, w.out, a.mixed, tokens, valDim, hidden)
}

// recurrenceInputs puts the recurrence's inputs from the cached convolution in work: its SiLU in channels, v the third
// block, and q and k divided by their L2 norms a head in keys.
func (l *FrozenGatedDeltaLayer) recurrenceInputs(r *frozenRun) {
	g := l.d.GDN
	count, keyRows := l.d.Tokens*(keyBlocks*g.KeyHeads+g.ValueHeads)*g.HeadDim, keyBlocks*l.d.Tokens*g.KeyHeads
	r.each("silu_f32", count, l.cache.convolved, l.work.channels, count)
	r.rows("l2_norm_f32", keyRows, l.work.channels, l.work.keys, g.HeadDim, keyRows, g.Eps)
}

// backwardMix queues the VJP of forwardMix with respect to the normed rows alone, as hostmath.GatedDeltaMixBackward
// gives DX, from the cache the last forward kept.
func (l *FrozenGatedDeltaLayer) backwardMix(r *frozenRun) {
	d, g, w, c, k, a := l.d, l.d.GDN, &l.weights, &l.cache, &l.work, &l.around
	tokens, hidden, keyHeads, heads, width := d.Tokens, d.Hidden, g.KeyHeads, g.ValueHeads, g.HeadDim
	keyDim, valDim, gates := keyHeads*width, heads*width, tokens*heads
	channels, keyRows := tokens*(keyBlocks*keyDim+valDim), keyBlocks*tokens*keyHeads
	// mix = Wout·(RMSNorm(o)·Norm ⊙ SiLU(z)): dgated, then dnormed = dgated⊙SiLU(z) back to do, and dz.
	r.back(a.mixed, w.out, k.values, tokens, hidden, valDim, false)
	r.each("silu_f32", tokens*valDim, c.z, k.values2, tokens*valDim)
	r.rows("weighted_rms_norm_f32", gates, c.recurrent, w.norm, k.values3, width, gates, 1, g.Eps)
	r.each("multiply_f32", tokens*valDim, k.values, k.values2, k.values2, tokens*valDim)
	r.each("multiply_f32", tokens*valDim, k.values, k.values3, k.values3, tokens*valDim)
	r.each("silu_backward_f32", tokens*valDim, k.values3, c.z, k.values3, tokens*valDim)
	r.rows("rms_norm_input_backward_f32", gates, k.values2, c.recurrent, w.norm, k.values, width, gates, g.Eps)
	// The recurrence's VJP into channels2, the q and k gradients summed over the value heads that share them.
	l.recurrenceInputs(r)
	normalized, mixed, gradients := l.blocks(k.keys), l.blocks(k.channels), l.blocks(k.channels2)
	r.zero(k.channels2, keyRows*width)
	r.zero(k.heads, gates)
	r.zero(k.heads2, gates)
	r.rows("gated_delta_net_rows_backward_f32", heads*((width+gatedDeltaBackwardRows-1)/gatedDeltaBackwardRows),
		normalized.q, normalized.k, mixed.v, c.gate, c.beta, k.values, gradients.q, gradients.k, gradients.v,
		k.heads, k.heads2, k.recurrence, width, keyHeads, heads, tokens, l.chunk)
	r.each("gated_delta_gates_backward_f32", gates, c.alpha, w.timeStep, w.decay, c.beta, k.heads, k.heads2, k.heads, k.heads2, heads, gates)
	// Through the L2 norms of q and k, the SiLU, then the convolution back to the projections.
	r.each("l2_norm_backward_f32", keyRows, k.channels, k.channels2, k.channels2, width, keyRows, g.Eps)
	r.each("silu_backward_f32", channels, k.channels2, c.convolved, k.channels2, channels)
	l.convolve(r, k.channels2, k.channels, 1)
	projections := l.blocks(k.channels)
	r.back(projections.q, w.q, a.normed, tokens, keyDim, hidden, false)
	r.back(projections.k, w.k, a.normed, tokens, keyDim, hidden, true)
	r.back(projections.v, w.v, a.normed, tokens, valDim, hidden, true)
	r.back(k.values3, w.z, a.normed, tokens, valDim, hidden, true)
	r.back(k.heads2, w.beta, a.normed, tokens, heads, hidden, true)
	r.back(k.heads, w.alpha, a.normed, tokens, heads, hidden, true)
}
